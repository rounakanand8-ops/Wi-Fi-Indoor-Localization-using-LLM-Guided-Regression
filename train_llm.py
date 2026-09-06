"""
Fine-tune an NVIDIA LLM backbone as an (X, Y) location regressor.

Freezing policy: attention projections + normalization layers are trainable,
MLP/feed-forward blocks stay frozen (see llm_regressor.py for the exact
matching rule and a printed report of what got unfrozen).

Requires a real GPU and the actual model weights — this cannot run in a
network-sandboxed environment. Tested here only via a dummy backbone
(see the conversation) to validate the freeze/pool/backward mechanics.

Usage
-----
python train_llm.py --env office --backbone google/gemma-3-1b-it
python train_llm.py --env all --epochs 5 --batch-size 8

Lighter-weight backbone for CPU-only or low-VRAM machines:
python train_llm.py --env office --backbone google/gemma-3-270m-it
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_utils import ENVIRONMENTS, SOTA_MEAN_ERROR_M, build_features
from evaluate import compute_metrics, print_report
from llm_regressor import LLMRegressor, apply_freeze_policy
from text_serialize import serialize_batch


class LocationTextDataset(Dataset):
    def __init__(self, texts: list[str], targets_norm: np.ndarray, tokenizer, max_length: int = 128):
        self.texts = texts
        self.targets = torch.tensor(targets_norm, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "target": self.targets[idx],
        }


def train_one_environment(
    env_name: str,
    data_dir: Path,
    backbone_name: str,
    epochs: int,
    batch_size: int,
    backbone_lr: float,
    head_lr: float,
    device: str,
) -> dict:
    from transformers import AutoModel, AutoTokenizer

    data = build_features(env_name, data_dir)

    # Normalize targets to roughly unit scale — makes MSE loss well-behaved
    # regardless of an environment's raw coordinate range (e.g. corridor's
    # X goes up to 56m while Y only spans 0-1m).
    y_mean = data.y_train.mean(axis=0)
    y_std = data.y_train.std(axis=0) + 1e-6
    y_train_norm = (data.y_train - y_mean) / y_std

    train_texts = serialize_batch(data.X_train, data.feature_names)
    test_texts = serialize_batch(data.X_test, data.feature_names)

    print(f"Loading backbone '{backbone_name}' ...")
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Force float32: many current backbones (e.g. Gemma 3) default to
    # bfloat16 weights, which would otherwise mismatch the float32
    # regression head's dtype during the matmul in LLMRegressor.forward.
    backbone = AutoModel.from_pretrained(backbone_name, torch_dtype=torch.float32)

    print("\nApplying freeze policy (attention + norm trainable, MLP frozen):")
    apply_freeze_policy(backbone)

    hidden_size = backbone.config.hidden_size
    model = LLMRegressor(backbone, hidden_size=hidden_size).to(device)

    train_ds = LocationTextDataset(train_texts, y_train_norm, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        model.trainable_parameter_groups(backbone_lr=backbone_lr, head_lr=head_lr)
    )
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            loss = loss_fn(pred, batch["target"].to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch["target"])
        epoch_loss /= len(train_ds)
        print(f"  epoch {epoch + 1}/{epochs}  train MSE (normalized) = {epoch_loss:.4f}")

    # --- Evaluate ---
    model.eval()
    test_ds = LocationTextDataset(test_texts, np.zeros_like(data.y_test), tokenizer)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    preds_norm = []
    with torch.no_grad():
        for batch in test_loader:
            pred = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            preds_norm.append(pred.cpu().numpy())
    preds_norm = np.concatenate(preds_norm, axis=0)
    preds = preds_norm * y_std + y_mean  # de-normalize back to meters

    metrics = compute_metrics(data.y_test, preds)
    print_report(env_name, metrics, SOTA_MEAN_ERROR_M[env_name])
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="all", choices=["all", *ENVIRONMENTS.keys()])
    parser.add_argument("--backbone", default="google/gemma-3-1b-it")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", default=str(Path(__file__).parent / "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    envs = list(ENVIRONMENTS.keys()) if args.env == "all" else [args.env]

    all_metrics = {}
    for env_name in envs:
        print(f"\n{'=' * 60}\n{env_name}\n{'=' * 60}")
        all_metrics[env_name] = train_one_environment(
            env_name, Path(args.data_dir), args.backbone,
            args.epochs, args.batch_size, args.backbone_lr, args.head_lr, args.device,
        )

    if len(envs) > 1:
        print("\n=== Summary ===")
        for env_name, m in all_metrics.items():
            sota = SOTA_MEAN_ERROR_M[env_name]
            beat = m["mean_euclidean_error_m"] < sota
            print(f"  {env_name:16s}: {m['mean_euclidean_error_m']:.3f} m "
                  f"(SOTA {sota:.2f} m) {'✅' if beat else '❌'}")


if __name__ == "__main__":
    main()
