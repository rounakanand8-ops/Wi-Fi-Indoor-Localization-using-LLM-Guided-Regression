# WiFi RTT/RSS Indoor Localization — TabPFN

Predicts device location (X, Y) in meters from 5-AP WiFi RTT + RSS readings,
across three environments: corridor, lecture theatre, office.

## Why TabPFN and not a text LLM

RTT/RSS-to-(X,Y) is a numeric tabular regression problem, not a language
problem. A text-generating LLM (Llama, GPT, etc. via a chat prompt) has no
mechanism for precise numeric interpolation and will not approach the
published benchmarks. TabPFN is a transformer too, but a different kind:
it's pretrained via in-context learning over millions of synthetic tabular
datasets, so at inference time it predicts directly from your training rows
with no fine-tuning step — this is what lets it perform well from a
comparatively small number of labeled examples, which is the property you
were originally after.

## Setup

```bash
pip install -r requirements.txt
```

TabPFN's pretrained weights are gated on Hugging Face. One-time setup:

1. Visit https://huggingface.co/Prior-Labs/tabpfn_3 and accept the license
   (requires a free Hugging Face account).
2. Authenticate locally:
   ```bash
   hf auth login
   # or: export HF_TOKEN=your_token_here
   ```

## Run

```bash
# Full pipeline, all three environments, primary model:
python train.py --model tabpfn --env all

# Single environment:
python train.py --model tabpfn --env corridor

# GPU strongly recommended if available (falls back to CPU automatically):
python train.py --model tabpfn --env all --device cuda
```

`--model rf` and `--model mlp` are included only as local smoke-test
baselines — useful if you want to sanity-check the data pipeline without
waiting on TabPFN's model download, but they carry no state-of-the-art
claim.

## What's reported

For each environment: per-axis MAE and MSE (as requested), plus mean/median/
p90 Euclidean positioning error in meters — the last one is the metric the
published SOTA numbers (corridor 1.85 m, theatre 1.23 m, office 1.98 m) are
expressed in, so that's the number to compare directly against. `train.py`
prints a ✅/❌ per environment and an overall summary.

## Files

- `data_utils.py` — loading + preprocessing (sentinel capping, detected
  flags, dropping AP columns that are never detected in a given
  environment — e.g. AP1 in the corridor set).
- `models.py` — model factory (`tabpfn`, `rf`, `mlp`).
- `evaluate.py` — metrics + report printing.
- `train.py` — CLI entry point.

## Alternative: Google Gemma backbone, partial fine-tuning

`train_llm.py` fine-tunes a Google Gemma model (default `google/gemma-3-1b-it`;
a lighter `google/gemma-3-270m-it` option is available for CPU-only or
low-VRAM machines) as the feature extractor instead of TabPFN, with the
MLP/feed-forward blocks frozen and only attention projections + normalization
layers trainable, feeding into a small trained regression head.

```bash
pip install transformers accelerate
huggingface-cli login   # accept Gemma's license on its HF page first
python train_llm.py --env office --backbone google/gemma-3-1b-it
```

Notes:
- **Needs a real GPU for the 1B variant**; the 270M variant is small enough
  to fine-tune on CPU, just slowly. Even with the MLP frozen, forward/
  backward passes still run through the whole network to propagate
  gradients to the attention layers, so activation memory scales with
  model size regardless of what's frozen.
- The freeze policy (`llm_regressor.py`) matches parameters by name
  (`attn`/`q_proj`/etc. and `norm` → trainable; everything else, including
  `mlp`/`gate_proj`/`up_proj`/`down_proj` → frozen) rather than hardcoding
  one architecture's module layout — Gemma uses the same naming convention
  as the Llama-style architecture this was designed against, so it applies
  unchanged. Run it once and read the printed report before a long training
  run to confirm it unfroze what you expect.
- I validated the dataset/dataloader/forward/backward/optimizer mechanics
  in this sandbox using a local dummy transformer with Gemma-style naming
  (this sandbox can't reach huggingface.co at all — confirmed with a direct
  connection test — regardless of which vendor's model you point it at, so
  I can't download or run real weights here for any backbone). The loss
  decreases correctly step over step; I have not validated actual accuracy
  against the real Gemma weights, which you'll need to run yourself.
- Expect this to train much more slowly than TabPFN (minutes-to-hours vs.
  seconds) and to need real hyperparameter tuning (learning rates, epochs,
  batch size) to get competitive. Treat the TabPFN pipeline as your
  accuracy baseline to beat, not the other way around.

## Notes / next steps if TabPFN falls short on any one environment

- Try `MultiOutputRegressor` replaced with two independently-tuned TabPFN
  instances (X and Y sometimes benefit from different feature subsets).
- Add k-fold ensembling (TabPFN is fast enough at inference to average
  several fits with different `random_state`).
- If you want a from-scratch-trained alternative for comparison, an
  FT-Transformer (Feature Tokenizer + Transformer) is a reasonable next
  architecture to add — happy to build that too if TabPFN doesn't clear
  the bar on all three environments.
