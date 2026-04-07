"""
Chronos-T5-Small on ETTh1
==========================
Three experiments on top of a *real* pretrained foundation model:

  Exp-1  zero_shot  : pipeline.predict(), no gradient updates
  Exp-2  rank_sweep : LoRA r=4/8/16 and SFT — % training data vs MSE
  Exp-3  b_init     : B=0 (standard LoRA) vs B~N (random) — step-level NLL

Each experiment saves its own JSON so they can run in parallel on separate GPUs.
After all three finish, run --plot to merge results and generate figures.

Usage (3 terminals / tmux panes):
    CUDA_VISIBLE_DEVICES=0 python experiment.py --exp zero_shot
    CUDA_VISIBLE_DEVICES=1 python experiment.py --exp rank_sweep
    CUDA_VISIBLE_DEVICES=2 python experiment.py --exp b_init

Then generate plots once all JSON files exist:
    python experiment.py --plot
"""

import argparse, json, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# ── 0. Config ──────────────────────────────────────────────────────────────────
CONTEXT_LEN     = 512   # matches chronos-t5-small context_length
FORECAST_LEN    = 64    # matches chronos-t5-small prediction_length (native horizon)
TARGET_COL      = "OT"
TRAIN_FRACS     = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
FINETUNE_EPOCHS = 3
B_INIT_EPOCHS   = 5     # longer run for the B-init loss-curve experiment
BATCH_SIZE      = 16
LR              = 5e-5
# DEVICE is resolved at runtime from CUDA_VISIBLE_DEVICES (see parse_args)
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
SEED            = 42
MODEL_ID        = "amazon/chronos-t5-small"
LORA_RANKS      = (4, 8, 16)

torch.manual_seed(SEED)
np.random.seed(SEED)

# resolve ETTh1.csv one directory above this file
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ETTh1.csv")
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))


# ── 1. Data ────────────────────────────────────────────────────────────────────
def load_etth1() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"ETTh1: {df.shape}  cols={list(df.columns)}")
    return df


def make_sequences(series: np.ndarray, ctx: int, fcst: int):
    """Sliding-window split into (context, forecast) pairs."""
    X, y = [], []
    total = ctx + fcst
    for i in range(len(series) - total + 1):
        X.append(series[i : i + ctx])
        y.append(series[i + ctx : i + total])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def split_dataset(df: pd.DataFrame):
    """
    Standard ETTh1 60/20/20 split.

    Returns RAW sequences for Chronos (it does its own MeanScaleUniformBins
    normalisation internally).  The scaler is kept only for computing
    normalised MSE so results stay comparable with other benchmarks.
    """
    n         = len(df)
    train_end = int(n * 0.6)
    val_end   = int(n * 0.8)
    series    = df[TARGET_COL].values.astype(np.float32)

    # fit scaler for MSE normalisation only — do NOT apply to the sequences
    scaler = StandardScaler()
    scaler.fit(series[:train_end].reshape(-1, 1))

    # sequences use raw values so Chronos tokeniser gets the original scale
    X_tr, y_tr = make_sequences(series[:train_end],        CONTEXT_LEN, FORECAST_LEN)
    X_va, y_va = make_sequences(series[train_end:val_end], CONTEXT_LEN, FORECAST_LEN)
    X_te, y_te = make_sequences(series[val_end:],          CONTEXT_LEN, FORECAST_LEN)
    return X_tr, y_tr, X_va, y_va, X_te, y_te, scaler


def get_loader(X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


# ── 2. LoRA ────────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with a low-rank adapter (Hu et al., 2022).

    Forward:  out = W x + (A B x) * (alpha/r)

    Initialisation:
      A ~ N(0, 0.01)   — small random, so A is non-trivial from step 0
      B = 0 or ~N      — zero keeps the initial delta = 0 (standard LoRA);
                          random B breaks this guarantee (the ablation case)
    """
    def __init__(
        self,
        original: nn.Linear,
        r: int          = 8,
        alpha: float    = 16.0,
        zero_init_B: bool = True,
    ):
        super().__init__()
        self.original = original
        # freeze base weights — only LoRA params will be updated
        for p in self.original.parameters():
            p.requires_grad = False

        self.r     = r
        self.scale = alpha / r
        d_in, d_out = original.in_features, original.out_features

        self.lora_A = nn.Parameter(torch.randn(d_in, r) * 0.01)
        if zero_init_B:
            self.lora_B = nn.Parameter(torch.zeros(r, d_out))
        else:
            # Kaiming uniform — the default PyTorch init a naive user would get
            # if they initialised B like a regular Linear layer.
            # This produces a non-trivial initial delta and lets the plot clearly
            # show why B=0 is the right choice.
            b = torch.empty(r, d_out)
            nn.init.kaiming_uniform_(b)
            self.lora_B = nn.Parameter(b)

    # proxy attributes expected by PyTorch internals / other layers
    @property
    def weight(self):       return self.original.weight
    @property
    def bias(self):         return self.original.bias
    @property
    def in_features(self):  return self.original.in_features
    @property
    def out_features(self): return self.original.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scale


def inject_lora(
    t5_model: nn.Module,
    r: int           = 8,
    zero_init_B: bool = True,
    target_modules   = ("q", "v"),   # attention projections to adapt
) -> nn.Module:
    """
    Walk all named modules in the T5 model and replace every Linear whose
    leaf name is in `target_modules` with a LoRALinear.

    Targeting q and v (standard choice from the LoRA paper) covers both
    encoder self-attention and decoder self/cross-attention.
    """
    replaced = 0
    for name, module in list(t5_model.named_modules()):
        leaf = name.split(".")[-1]
        if leaf not in target_modules or not isinstance(module, nn.Linear):
            continue
        # navigate to the parent module
        parts  = name.split(".")
        parent = t5_model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], LoRALinear(module, r=r, zero_init_B=zero_init_B))
        replaced += 1
    print(f"  LoRA injected into {replaced} layers  (r={r}, zero_B={zero_init_B})")
    return t5_model


def freeze_all(model: nn.Module) -> None:
    """Freeze every parameter (call before inject_lora for pure adapter training)."""
    for p in model.parameters():
        p.requires_grad = False


def count_trainable(model: nn.Module) -> int:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")
    return trainable


# ── 3. Chronos tokenisation helpers ───────────────────────────────────────────
def encode_batch(pipeline, context_np: np.ndarray, target_np: np.ndarray):
    """
    Tokenise a numpy batch with Chronos's MeanScaleUniformBins tokenizer.

    Returns input_ids, attention_mask, target_ids — all on CPU.
    `target_ids` is passed as `labels` to the T5 model, which computes the
    cross-entropy (NLL) loss internally.
    """
    context = torch.tensor(context_np, dtype=torch.float32)
    target  = torch.tensor(target_np,  dtype=torch.float32)

    # context → token ids + scale (used to normalise the target consistently)
    input_ids, attention_mask, scale = pipeline.tokenizer.context_input_transform(context)

    # target → token ids   (bypasses the prediction_length assertion so we can
    #                        freely choose FORECAST_LEN independently of the
    #                        model config, should it differ)
    target_ids, _, _ = pipeline.tokenizer._input_transform(context=target, scale=scale)
    if pipeline.tokenizer.config.use_eos_token:
        target_ids, _ = pipeline.tokenizer._append_eos_token(
            target_ids, torch.ones_like(target_ids, dtype=torch.bool)
        )

    return input_ids, attention_mask, target_ids


# ── 4. Zero-shot evaluation ────────────────────────────────────────────────────
def _predict_median(pipeline, X: np.ndarray) -> np.ndarray:
    """Run Chronos predict and return median forecasts (B, forecast_len)."""
    all_preds = []
    for i in range(0, len(X), BATCH_SIZE):
        ctx = torch.tensor(X[i : i + BATCH_SIZE], dtype=torch.float32)
        with torch.no_grad():
            samples = pipeline.predict(
                ctx,
                prediction_length       = FORECAST_LEN,
                num_samples             = 20,
                limit_prediction_length = False,
            )
        all_preds.append(samples.median(dim=1).values.cpu().numpy())
    return np.concatenate(all_preds)


def _normalised_mse(preds: np.ndarray, true: np.ndarray, scaler: StandardScaler) -> float:
    """Scale both arrays to z-score space before computing MSE so the metric
    is dataset-agnostic and comparable across benchmarks."""
    p = scaler.transform(preds.reshape(-1, 1)).flatten()
    t = scaler.transform(true.reshape(-1, 1)).flatten()
    return mean_squared_error(t, p)


def run_zero_shot(pipeline, X_test: np.ndarray, y_test: np.ndarray,
                  scaler: StandardScaler) -> float:
    """
    Use the pretrained Chronos pipeline directly — no fine-tuning at all.
    Median of 20 sample paths vs. ground truth → normalised MSE.
    """
    print("\n── Zero-Shot ──")
    pipeline.model.eval()
    preds = _predict_median(pipeline, X_test)
    mse   = _normalised_mse(preds, y_test, scaler)
    print(f"  Zero-Shot MSE: {mse:.4f}")
    return mse


# ── 5. Training helpers ────────────────────────────────────────────────────────
def train_one_epoch(pipeline, loader: DataLoader, optimizer, record_steps: bool = False):
    """
    One pass over `loader`.  Returns (avg_loss, step_losses).
    `step_losses` is populated only when `record_steps=True`.
    """
    t5_model = pipeline.model.model.to(DEVICE)
    t5_model.train()
    step_losses, total = [], 0.0

    for xb, yb in loader:
        input_ids, attn_mask, target_ids = encode_batch(pipeline, xb.numpy(), yb.numpy())
        input_ids   = input_ids.to(DEVICE)
        attn_mask   = attn_mask.to(DEVICE)
        target_ids  = target_ids.to(DEVICE)

        optimizer.zero_grad()
        # T5 computes cross-entropy loss when labels are provided
        out  = t5_model(input_ids=input_ids, attention_mask=attn_mask, labels=target_ids)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in t5_model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        total += loss.item()
        if record_steps:
            step_losses.append(loss.item())

    return total / len(loader), step_losses


def eval_mse(pipeline, X_test: np.ndarray, y_test: np.ndarray,
             scaler: StandardScaler) -> float:
    """Evaluate normalised MSE using the pipeline's sampling-based predict."""
    pipeline.model.eval()
    preds = _predict_median(pipeline, X_test)
    return _normalised_mse(preds, y_test, scaler)


def _fresh_pipeline():
    """Load a clean copy of Chronos-T5-Small from the hub (or local cache)."""
    from chronos import ChronosPipeline
    return ChronosPipeline.from_pretrained(
        MODEL_ID, device_map=DEVICE, torch_dtype=torch.float32
    )


# ── 6. Exp-2: LoRA rank sweep + SFT ──────────────────────────────────────────
def run_lora_ranks(X_tr, y_tr, X_te, y_te, scaler):
    """
    For each (rank, train_frac) combination:
      - Load a fresh pretrained Chronos
      - Freeze all weights, inject LoRA adapters
      - Fine-tune for FINETUNE_EPOCHS
      - Evaluate normalised MSE on test set

    Returns dict: { rank → { frac → mse } }
    """
    results = {r: {} for r in LORA_RANKS}

    for r in LORA_RANKS:
        print(f"\n── LoRA r={r} ──")
        for frac in TRAIN_FRACS:
            n = max(1, int(len(X_tr) * frac))
            print(f"  frac={frac:.0%}  n_samples={n}")

            pipeline = _fresh_pipeline()
            freeze_all(pipeline.model.model)
            inject_lora(pipeline.model.model, r=r, zero_init_B=True)
            count_trainable(pipeline.model.model)

            opt    = torch.optim.AdamW(
                [p for p in pipeline.model.model.parameters() if p.requires_grad], lr=LR
            )
            loader = get_loader(X_tr[:n], y_tr[:n])

            for ep in range(FINETUNE_EPOCHS):
                avg_loss, _ = train_one_epoch(pipeline, loader, opt)
                print(f"    ep {ep+1}/{FINETUNE_EPOCHS}  loss={avg_loss:.4f}")

            mse = eval_mse(pipeline, X_te, y_te, scaler)
            results[r][frac] = mse
            print(f"    MSE={mse:.4f}")
            del pipeline  # free GPU memory between runs

    return results


def run_sft(X_tr, y_tr, X_te, y_te, scaler):
    """
    Standard full fine-tuning — all model parameters are updated.
    Returns dict: { frac → mse }
    """
    results = {}
    print("\n── Standard Fine-Tuning (SFT) ──")
    for frac in TRAIN_FRACS:
        n = max(1, int(len(X_tr) * frac))
        print(f"  frac={frac:.0%}  n_samples={n}")

        pipeline = _fresh_pipeline()
        count_trainable(pipeline.model.model)   # all params trainable in SFT

        opt    = torch.optim.AdamW(pipeline.model.model.parameters(), lr=LR)
        loader = get_loader(X_tr[:n], y_tr[:n])

        for ep in range(FINETUNE_EPOCHS):
            avg_loss, _ = train_one_epoch(pipeline, loader, opt)
            print(f"    ep {ep+1}/{FINETUNE_EPOCHS}  loss={avg_loss:.4f}")

        mse = eval_mse(pipeline, X_te, y_te, scaler)
        results[frac] = mse
        print(f"    MSE={mse:.4f}")
        del pipeline

    return results


# ── 7. Exp-3: B-matrix initialisation comparison ─────────────────────────────
def run_b_init_experiment(X_tr, y_tr, r: int = 8, n_epochs: int = B_INIT_EPOCHS):
    """
    Train two LoRA adapters that are identical except for lora_B init:
      'B=0 (standard)' : lora_B = zeros  → initial delta = 0
      'B~N (random)'   : lora_B ~ N(0, 0.01) → initial delta ≠ 0

    Records per-step NLL loss for both, so the difference is visible
    from the very first step.

    Uses 20 % of training data to keep the experiment fast while
    still having enough steps to show convergence behaviour.
    """
    print(f"\n── B-Init Experiment  r={r}  epochs={n_epochs} ──")
    n_sub  = max(1, int(len(X_tr) * 0.2))
    loader = get_loader(X_tr[:n_sub], y_tr[:n_sub])

    curves = {}
    for label, zero_B in [("B=0 (standard LoRA)", True), ("B~Kaiming (naive init)", False)]:
        print(f"\n  {label}")
        pipeline = _fresh_pipeline()
        freeze_all(pipeline.model.model)
        inject_lora(pipeline.model.model, r=r, zero_init_B=zero_B)

        opt        = torch.optim.AdamW(
            [p for p in pipeline.model.model.parameters() if p.requires_grad], lr=LR
        )
        step_losses = []
        for ep in range(n_epochs):
            _, ep_steps = train_one_epoch(pipeline, loader, opt, record_steps=True)
            step_losses.extend(ep_steps)
            print(f"    ep {ep+1}/{n_epochs}  "
                  f"avg={sum(ep_steps)/len(ep_steps):.4f}  "
                  f"total_steps={len(step_losses)}")

        curves[label] = step_losses
        del pipeline

    return curves


# ── 8. Plotting ────────────────────────────────────────────────────────────────
_BG    = "#0d1117"
_GRID  = "#2a2d36"
_TEXT  = "#c9d1d9"
_SPINE = "#30363d"
_WHITE = "#ffffff"

def _style(ax, fig):
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.grid(color=_GRID, linestyle="--", linewidth=0.7, zorder=0)
    ax.tick_params(colors="#8b949e", labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(_SPINE)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)


def plot_rank_comparison(zero_mse, lora_results, sft_results, path):
    fracs_pct = [f * 100 for f in TRAIN_FRACS]
    fig, ax   = plt.subplots(figsize=(11, 6))
    _style(ax, fig)

    ax.axhline(zero_mse, color="#f0c040", linewidth=1.8, linestyle=":",
               label=f"Zero-Shot (MSE={zero_mse:.3f})", zorder=3)

    rank_style = {4: ("#43ff56", "o"), 8: ("#ab47bc", "^"), 16: ("#26c6da", "D")}
    for r, (color, marker) in rank_style.items():
        mse_vals = [lora_results[r][f] for f in TRAIN_FRACS]
        ax.plot(fracs_pct, mse_vals, marker=marker, markersize=7,
                color=color, linewidth=2.2, label=f"LoRA r={r}", zorder=4)

    sft_vals = [sft_results[f] for f in TRAIN_FRACS]
    ax.plot(fracs_pct, sft_vals, marker="s", markersize=7,
            color="#ef5350", linewidth=2.2, label="Standard Fine-Tuning", zorder=4)

    ax.set_xlabel("Training Data Used (%)", fontsize=12, labelpad=10)
    ax.set_ylabel("MSE (normalised)", fontsize=12, labelpad=10)
    ax.set_title(
        "Chronos-T5-Small on ETTh1 — Zero-Shot vs LoRA (r=4/8/16) vs SFT",
        color=_WHITE, fontsize=14, fontweight="bold", pad=16,
    )
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(facecolor="#161b22", edgecolor=_SPINE, labelcolor=_TEXT,
              fontsize=10, loc="upper right")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {path}")
    plt.close()


def _smooth(xs, w=10):
    """Simple moving average for readability."""
    return np.convolve(xs, np.ones(w) / w, mode="valid")


def plot_b_init_curves(curves, path):
    """
    Per-step NLL training loss for B=0 vs B~N.
    Raw trace shown faintly; smoothed trace shown boldly.
    """
    palette = {"B=0 (standard LoRA)": "#43ff56", "B~Kaiming (naive init)": "#ef5350"}

    fig, ax = plt.subplots(figsize=(10, 5))
    _style(ax, fig)

    for label, losses in curves.items():
        color = palette[label]
        steps = list(range(len(losses)))
        # faint raw trace
        ax.plot(steps, losses, color=color, linewidth=0.8, alpha=0.3)
        # bold smoothed trace
        sm = _smooth(losses, w=max(1, len(losses) // 50))
        ax.plot(range(len(sm)), sm, color=color, linewidth=2.2, label=label)

    # mark epoch boundaries
    steps_per_epoch = len(curves[list(curves.keys())[0]]) // B_INIT_EPOCHS
    for ep in range(1, B_INIT_EPOCHS):
        ax.axvline(ep * steps_per_epoch, color=_GRID, linewidth=0.9, linestyle="--")
        ax.text(ep * steps_per_epoch + 2, ax.get_ylim()[1] * 0.97,
                f"ep{ep+1}", color="#8b949e", fontsize=8, va="top")

    ax.set_xlabel("Training Step", fontsize=12, labelpad=10)
    ax.set_ylabel("Training Loss (NLL)", fontsize=12, labelpad=10)
    ax.set_title(
        "LoRA B-Matrix Init: B=0 (standard) vs B~N (random) — Step-Level NLL",
        color=_WHITE, fontsize=13, fontweight="bold", pad=14,
    )
    ax.legend(facecolor="#161b22", edgecolor=_SPINE, labelcolor=_TEXT, fontsize=11)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {path}")
    plt.close()


# ── 9. Result persistence ─────────────────────────────────────────────────────
def _save(name: str, data) -> None:
    path = os.path.join(OUT_DIR, f"results_{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")


def _load(name: str):
    path = os.path.join(OUT_DIR, f"results_{name}.json")
    with open(path) as f:
        return json.load(f)


# ── 10. Per-experiment entry points ───────────────────────────────────────────
def exp_zero_shot():
    print(f"[zero_shot]  device={DEVICE}")
    df = load_etth1()
    _, _, _, _, X_te, y_te, scaler = split_dataset(df)
    pipeline = _fresh_pipeline()
    mse = run_zero_shot(pipeline, X_te, y_te, scaler)
    del pipeline
    _save("zero_shot", {"zero_shot_mse": mse})


def exp_rank_sweep():
    print(f"[rank_sweep]  device={DEVICE}")
    df = load_etth1()
    X_tr, y_tr, _, _, X_te, y_te, scaler = split_dataset(df)

    lora_results = run_lora_ranks(X_tr, y_tr, X_te, y_te, scaler)
    sft_results  = run_sft(X_tr, y_tr, X_te, y_te, scaler)

    # convert int keys to strings for JSON serialisation
    lora_serial = {str(r): {str(f): v for f, v in d.items()}
                   for r, d in lora_results.items()}
    sft_serial  = {str(f): v for f, v in sft_results.items()}
    _save("rank_sweep", {"lora": lora_serial, "sft": sft_serial})


def exp_b_init():
    print(f"[b_init]  device={DEVICE}")
    df = load_etth1()
    X_tr, y_tr, *_ = split_dataset(df)
    curves = run_b_init_experiment(X_tr, y_tr, r=8, n_epochs=B_INIT_EPOCHS)
    _save("b_init", curves)


# ── 11. Plot (merges all JSON results) ────────────────────────────────────────
def make_plots():
    # ── load results ──────────────────────────────────────────────────────────
    zs   = _load("zero_shot")
    rs   = _load("rank_sweep")
    bi   = _load("b_init")

    zero_mse = zs["zero_shot_mse"]

    # restore numeric keys (JSON serialises them as strings)
    lora_results = {int(r): {float(f): v for f, v in d.items()}
                    for r, d in rs["lora"].items()}
    sft_results  = {float(f): v for f, v in rs["sft"].items()}

    # ── CSV summary ───────────────────────────────────────────────────────────
    rows = [
        {
            "train_pct":     f * 100,
            "lora_r4_mse":   lora_results[4][f],
            "lora_r8_mse":   lora_results[8][f],
            "lora_r16_mse":  lora_results[16][f],
            "sft_mse":       sft_results[f],
            "zero_shot_mse": zero_mse,
        }
        for f in TRAIN_FRACS
    ]
    csv_path = os.path.join(OUT_DIR, "chronos_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"CSV: {csv_path}")

    # ── figures ───────────────────────────────────────────────────────────────
    plot_rank_comparison(
        zero_mse, lora_results, sft_results,
        os.path.join(OUT_DIR, "chronos_rank_comparison.png"),
    )
    plot_b_init_curves(
        bi,
        os.path.join(OUT_DIR, "chronos_b_init_curves.png"),
    )
    print("Done.")


# ── 12. CLI ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--exp",
        choices=["zero_shot", "rank_sweep", "b_init"],
        help="Which experiment to run on this GPU",
    )
    group.add_argument(
        "--plot",
        action="store_true",
        help="Merge saved JSON results and produce figures (no GPU needed)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.plot:
        make_plots()
    elif args.exp == "zero_shot":
        exp_zero_shot()
    elif args.exp == "rank_sweep":
        exp_rank_sweep()
    elif args.exp == "b_init":
        exp_b_init()
