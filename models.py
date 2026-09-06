"""
Model factory.

Primary model: TabPFN (https://huggingface.co/Prior-Labs) — a transformer
pretrained via in-context learning over millions of synthetic tabular
datasets. It predicts directly from the training rows at inference time
with no gradient-descent fine-tuning, which is what makes it work well with
comparatively few labeled examples (a few thousand rows here, vs. the
millions a text LLM would need to learn a numeric regression task from
scratch). This is the "transformer that predicts well from limited data"
tool your reasoning was pointing at.

TabPFNRegressor predicts a single scalar target, so we wrap it in
sklearn's MultiOutputRegressor to predict [X, Y] jointly (fits one internal
TabPFN instance per target column).

`build_model("rf", ...)` / `build_model("mlp", ...)` are baselines with NO
special claim to being state-of-the-art — they exist only so the rest of
this pipeline (data loading, metrics, reporting) can be smoke-tested in
environments where TabPFN's pretrained weights can't be downloaded
(gated on Hugging Face + needs network access).
"""

from __future__ import annotations

from sklearn.multioutput import MultiOutputRegressor


def build_model(name: str, device: str = "auto", random_state: int = 0):
    name = name.lower()

    if name == "tabpfn":
        # Requires: pip install tabpfn, and a Hugging Face account that has
        # accepted the license at https://huggingface.co/Prior-Labs/tabpfn_3,
        # then `hf auth login` (or set HF_TOKEN) once, locally.
        from tabpfn import TabPFNRegressor

        base = TabPFNRegressor(device=device, random_state=random_state)
        return MultiOutputRegressor(base)

    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor

        base = RandomForestRegressor(
            n_estimators=400, max_depth=None, random_state=random_state, n_jobs=-1
        )
        return MultiOutputRegressor(base)

    if name == "mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        base = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=2000,
            random_state=random_state,
        )
        # MLP is directly multi-output, no wrapper needed.
        return make_pipeline(StandardScaler(), base)

    raise ValueError(f"Unknown model '{name}'. Choose from: tabpfn, rf, mlp")
