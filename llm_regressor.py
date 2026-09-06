"""
NVIDIA LLM backbone as a (partially) trainable feature extractor for
(X, Y) regression.

Freezing policy
----------------
- Every parameter starts frozen.
- A parameter is UNFROZEN if its fully-qualified name contains an
  attention-related keyword (e.g. "attn", "attention", "q_proj", "k_proj",
  "v_proj", "o_proj") OR a normalization-related keyword (e.g. "norm",
  "ln_", "layernorm").
- Everything else — including MLP / feed-forward blocks ("mlp",
  "feed_forward", "fc1", "fc2", "gate_proj", "up_proj", "down_proj"),
  token embeddings, and the LM head — stays frozen.

Matching is done by substring on parameter names rather than hardcoding one
architecture's exact module layout, so the same function works whether the
backbone turns out to use "self_attn.q_proj" (Llama-style) or "attention.query"
(BERT-style) naming — you can inspect exactly what got unfrozen via
`report_trainable_parameters()` before committing to a training run.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn

ATTENTION_PATTERNS = [
    r"attn", r"attention", r"q_proj", r"k_proj", r"v_proj", r"o_proj",
    r"query", r"key", r"value",
]
NORM_PATTERNS = [r"norm", r"ln_", r"layernorm"]
# Explicitly listed for clarity in the report even though they're frozen by
# default (i.e. "everything not matched above"):
MLP_PATTERNS = [
    r"mlp", r"feed_forward", r"fc1", r"fc2", r"gate_proj", r"up_proj", r"down_proj",
]


def _matches_any(name: str, patterns: list[str]) -> bool:
    name = name.lower()
    return any(re.search(p, name) for p in patterns)


def apply_freeze_policy(backbone: nn.Module, verbose: bool = True) -> dict:
    """
    Freeze all backbone parameters, then unfreeze attention + norm params.
    Returns a summary dict; prints a per-module trainable report if verbose.
    """
    summary = {"trainable": [], "frozen_mlp": [], "frozen_other": []}

    for name, param in backbone.named_parameters():
        if _matches_any(name, ATTENTION_PATTERNS) or _matches_any(name, NORM_PATTERNS):
            param.requires_grad = True
            summary["trainable"].append(name)
        else:
            param.requires_grad = False
            if _matches_any(name, MLP_PATTERNS):
                summary["frozen_mlp"].append(name)
            else:
                summary["frozen_other"].append(name)

    n_trainable = sum(backbone.get_parameter(n).numel() for n in summary["trainable"])
    n_frozen = sum(
        backbone.get_parameter(n).numel()
        for n in summary["frozen_mlp"] + summary["frozen_other"]
    )

    if verbose:
        print(f"Trainable (attention + norm): {len(summary['trainable'])} tensors, "
              f"{n_trainable:,} params")
        print(f"Frozen MLP:                   {len(summary['frozen_mlp'])} tensors")
        print(f"Frozen other (embeddings, lm_head, etc.): "
              f"{len(summary['frozen_other'])} tensors")
        print(f"Total trainable fraction: {n_trainable / (n_trainable + n_frozen):.1%}")
        print("\nFirst few trainable parameter names (verify these look right "
              "for your backbone before training):")
        for n in summary["trainable"][:10]:
            print(f"  [TRAIN] {n}")
        print("First few frozen MLP parameter names:")
        for n in summary["frozen_mlp"][:5]:
            print(f"  [FROZEN-MLP] {n}")

    return summary


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool a (batch, seq, hidden) tensor over real (non-padding) tokens."""
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


class LLMRegressor(nn.Module):
    """
    Wraps a Hugging Face causal LM as an encoder: backbone -> mean pool ->
    small MLP head -> (X, Y). The backbone follows the freeze policy above;
    the head is always fully trainable.
    """

    def __init__(self, backbone: nn.Module, hidden_size: int, head_hidden: int = 256):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.GELU(),
            nn.Linear(head_hidden // 2, 2),  # (X, Y), predicted in normalized units
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]
        pooled = mean_pool(last_hidden, attention_mask)
        return self.head(pooled)

    def trainable_parameter_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": self.head.parameters(), "lr": head_lr},
        ]
