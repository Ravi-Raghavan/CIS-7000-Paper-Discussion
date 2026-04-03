"""
ETTh1 Benchmark: Zero-Shot vs LoRA vs Standard Fine-Tuning
=================================================================
Evaluates TimesFM-Based Surrogate Model across three training regimes on the ETTh1 dataset.
Plots % Training Data vs MSE for each method.
"""

import os
os.environ["USE_TF"] = "0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

# Helper Function 
def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")
    return trainable, total

# ════════════════════════════════════════════════════════════════════════════
# 0. CONFIG
# ════════════════════════════════════════════════════════════════════════════
CONTEXT_LEN   = 512          # TimesFM default context window
FORECAST_LEN  = 96           # prediction horizon (ETTh1 standard)
TARGET_COL    = "OT"         # target variable in ETTh1
TRAIN_FRACS   = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]  # x-axis
FINETUNE_EPOCHS = 5
BATCH_SIZE    = 32
LR            = 1e-4
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SEED          = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════
def load_etth1():
    """Load ETTh1 from HuggingFace datasets."""
    print("Loading ETTh1 dataset...")
    try:
        ds = load_dataset("ETDataset/ett-small", "ETTh1", trust_remote_code=True)
        df = ds["train"].to_pandas()
    except Exception:
        # Fallback: download CSV directly
        url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
        print(f"  Downloading from: {url}")
        df = pd.read_csv(url, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  Shape: {df.shape}  |  Columns: {list(df.columns)}")
    return df


def make_sequences(series: np.ndarray, context_len: int, forecast_len: int):
    """Slide a window over the series to create (X, y) pairs."""
    X, y = [], []
    total = context_len + forecast_len
    for i in range(len(series) - total + 1):
        X.append(series[i : i + context_len])
        y.append(series[i + context_len : i + total])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def split_dataset(df, target_col=TARGET_COL):
    """Standard ETTh1 train/val/test split (60/20/20)."""
    n = len(df)
    train_end = int(n * 0.6)
    val_end   = int(n * 0.8)
    series = df[target_col].values

    scaler = StandardScaler()
    train_raw = series[:train_end]
    scaler.fit(train_raw.reshape(-1, 1))
    scaled = scaler.transform(series.reshape(-1, 1)).flatten()

    X_train, y_train = make_sequences(scaled[:train_end],   CONTEXT_LEN, FORECAST_LEN)
    X_val,   y_val   = make_sequences(scaled[train_end:val_end], CONTEXT_LEN, FORECAST_LEN)
    X_test,  y_test  = make_sequences(scaled[val_end:],     CONTEXT_LEN, FORECAST_LEN)
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


# ════════════════════════════════════════════════════════════════════════════
# 2. SURROGATE MODEL to Replicate TimesFM
# ════════════════════════════════════════════════════════════════════════════
class TimeSeriesTransformer(nn.Module):
    """
    Decoder-only Transformer (GPT-style) for time-series forecasting.
    """
    def __init__(self, context_len=CONTEXT_LEN, forecast_len=FORECAST_LEN,
                 d_model=128, nhead=4, num_layers=3, patch_size=32):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = context_len // patch_size
        self.forecast_len = forecast_len
        self.d_model = d_model

        # ---- Patch embedding ----
        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_emb     = nn.Embedding(self.num_patches, d_model)

        # ---- Decoder-only stack ----
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # ---- Output head ----
        self.head = nn.Linear(d_model * self.num_patches, forecast_len)

    def _causal_mask(self, size, device):
        return torch.triu(
            torch.full((size, size), float('-inf'), device=device),
            diagonal=1
        )

    def forward(self, x):
        """
        x: (B, context_len)
        """
        B = x.size(0)
        device = x.device

        # ---- Patchify ----
        patches = x.view(B, self.num_patches, self.patch_size)  # (B, P, patch_size)
        h = self.patch_embed(patches)                           # (B, P, d_model)

        # ---- Add positional encoding ----
        pos = torch.arange(self.num_patches, device=device)
        h = h + self.pos_emb(pos)

        # ---- Causal mask ----
        tgt_mask = self._causal_mask(self.num_patches, device)

        # ---- Decoder (no encoder memory) ----
        # memory=None → acts like GPT-style self-attention
        h = self.decoder(tgt=h, memory=h, tgt_mask=tgt_mask)

        # ---- Flatten + predict ----
        h = h.reshape(B, -1)
        return self.head(h)


# ════════════════════════════════════════════════════════════════════════════
# 3. LoRA Implementation
# ════════════════════════════════════════════════════════════════════════════
class LoRALinear(nn.Module):
    """Wraps an existing Linear layer with a low-rank adapter (r=8)."""
    def __init__(self, original: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original
        for p in self.original.parameters():
            p.requires_grad = False             # freeze base weights
        self.r     = r
        self.alpha = alpha
        d_in  = original.in_features
        d_out = original.out_features
        self.lora_A = nn.Parameter(torch.randn(d_in,  r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, d_out))
        self.scale  = alpha / r
    
    # ── Proxy attributes PyTorch internals expect on nn.Linear ──────────────
    @property
    def weight(self):
        return self.original.weight
 
    @property
    def bias(self):
        return self.original.bias
 
    @property
    def in_features(self):
        return self.original.in_features
 
    @property
    def out_features(self):
        return self.original.out_features

    def forward(self, x):
        base_out = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scale
        return base_out + lora_out


def inject_lora(model: nn.Module, r: int = 8) -> nn.Module:
    """Replace every Linear in the Decoder with a LoRA-wrapped version."""
    for name, module in list(model.decoder.named_modules()):
        if isinstance(module, nn.Linear):
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.decoder
            for part in parent_name.split("."):
                if part:
                    parent = getattr(parent, part)
            setattr(parent, child_name, LoRALinear(module, r=r))
    return model


# ════════════════════════════════════════════════════════════════════════════
# 4. TRAINING HELPERS
# ════════════════════════════════════════════════════════════════════════════
def get_dataloader(X, y, batch_size=BATCH_SIZE, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(model, loader, epochs=FINETUNE_EPOCHS, lr=LR):
    model = model.to(DEVICE)
    opt   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        print(f"    Epoch {ep+1}/{epochs}  loss={total_loss/len(loader):.4f}")
    return model


def evaluate_model(model, X_test, y_test):
    model.eval()
    all_preds, all_true = [], []
    loader = get_dataloader(X_test, y_test, shuffle=False)
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            preds = model(xb).cpu().numpy()
            all_preds.append(preds)
            all_true.append(yb.numpy())
    preds = np.concatenate(all_preds)
    true  = np.concatenate(all_true)
    return mean_squared_error(true.flatten(), preds.flatten())


# ════════════════════════════════════════════════════════════════════════════
# 5. ZERO-SHOT INFERENCE
# ════════════════════════════════════════════════════════════════════════════
def run_zero_shot(X_test, y_test):
    """
    Zero-shot: use TimesFM (or our surrogate with random weights) directly
    on the test set — no gradient updates whatsoever.
    """
    print("\n── Zero-Shot Inference ──")
    model = TimeSeriesTransformer().to(DEVICE)
    mse   = evaluate_model(model, X_test, y_test)
    print(f"  Zero-Shot MSE: {mse:.4f}")
    return mse


# ════════════════════════════════════════════════════════════════════════════
# 6. LORA FINE-TUNING
# ════════════════════════════════════════════════════════════════════════════
def run_lora(X_train_full, y_train_full, X_test, y_test, train_fracs, lora_rank = 8):
    """Fine-tune only LoRA adapters; backbone is frozen."""
    print(f"\n── LoRA Fine-Tuning: Rank = {lora_rank} ──")
    results = {}
    for frac in train_fracs:
        n = max(1, int(len(X_train_full) * frac))
        X_sub, y_sub = X_train_full[:n], y_train_full[:n]
        print(f"  frac={frac:.0%}  n_samples={n}")

        # Fresh backbone each run (simulates loading pretrained checkpoint)
        base_model = TimeSeriesTransformer().to(DEVICE)
        model = inject_lora(base_model, r = lora_rank)
        print("    Manual LoRA injected.")

        # Count trainable params
        count_trainable_params(model)

        loader = get_dataloader(X_sub, y_sub)
        model  = train_model(model, loader)
        mse    = evaluate_model(model, X_test, y_test)
        results[frac] = mse
        print(f"    MSE: {mse:.4f}")
    return results


# ════════════════════════════════════════════════════════════════════════════
# 7. STANDARD FINE-TUNING
# ════════════════════════════════════════════════════════════════════════════
def run_sft(X_train_full, y_train_full, X_test, y_test, train_fracs):
    """Fine-tune ALL parameters of the model (standard full fine-tuning)."""
    print("\n── Standard Fine-Tuning (SFT) ──")
    results = {}
    for frac in train_fracs:
        n = max(1, int(len(X_train_full) * frac))
        X_sub, y_sub = X_train_full[:n], y_train_full[:n]
        print(f"  frac={frac:.0%}  n_samples={n}")

        model  = TimeSeriesTransformer().to(DEVICE)          # fresh each run
        count_trainable_params(model)           # all params are trainable in SFT   
        loader = get_dataloader(X_sub, y_sub)
        model  = train_model(model, loader)
        mse    = evaluate_model(model, X_test, y_test)
        results[frac] = mse
        print(f"    MSE: {mse:.4f}")
    return results


# ════════════════════════════════════════════════════════════════════════════
# 8. PLOTTING
# ════════════════════════════════════════════════════════════════════════════
def plot_results(zero_shot_mse, lora_results_rank_4, lora_results_rank_8, lora_results_rank_16, sft_results, save_path="results.png"):
    fracs_pct = [f * 100 for f in TRAIN_FRACS]

    lora_mse_4  = [lora_results_rank_4[f]  for f in TRAIN_FRACS]
    lora_mse_8  = [lora_results_rank_8[f]  for f in TRAIN_FRACS]
    lora_mse_16 = [lora_results_rank_16[f] for f in TRAIN_FRACS]
    sft_mse     = [sft_results[f]          for f in TRAIN_FRACS]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Grid
    ax.grid(color="#2a2d36", linestyle="--", linewidth=0.7, zorder=0)

    # Zero-shot horizontal line
    ax.axhline(zero_shot_mse, color="#f0c040", linewidth=1.8,
               linestyle=":", label=f"Zero-Shot (MSE={zero_shot_mse:.3f})", zorder=3)

    # LoRA lines (one per rank)
    lora_series = [
        (lora_mse_4,  "#43ff56", "o", "LoRA r=4",  9),
        (lora_mse_8,  "#ab47bc", "^", "LoRA r=8",  9),
        (lora_mse_16, "#26c6da", "D", "LoRA r=16", 9),
    ]
    for mse_vals, color, marker, label, y_offset in lora_series:
        ax.plot(fracs_pct, mse_vals, marker=marker, markersize=7,
                color=color, linewidth=2.2, label=label, zorder=4)
        # for x, y in zip(fracs_pct, mse_vals):
        #     ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
        #                 xytext=(0, y_offset), ha="center", fontsize=7.5, color=color)

    # SFT line
    ax.plot(fracs_pct, sft_mse, marker="s", markersize=7,
            color="#ef5350", linewidth=2.2,
            label="Standard Fine-Tuning", zorder=4)
    # for x, y in zip(fracs_pct, sft_mse):
    #     ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
    #                 xytext=(0, -16), ha="center", fontsize=7.5, color="#ef5350")

    # Axes styling
    ax.set_xlabel("Training Data Used (%)", color="#c9d1d9", fontsize=12, labelpad=10)
    ax.set_ylabel("MSE (normalised)", color="#c9d1d9", fontsize=12, labelpad=10)
    ax.set_title("TimesFM on ETTh1 — Zero-Shot vs LoRA (r=4/8/16) vs Standard Fine-Tuning",
                 color="#ffffff", fontsize=14, fontweight="bold", pad=16)

    ax.tick_params(colors="#8b949e", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d36")

    ax.xaxis.set_major_formatter(mticker.PercentFormatter())

    ax.legend(facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#c9d1d9", fontsize=10, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n✓ Plot saved to: {save_path}")
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print(" TimesFM ETTh1 Benchmark")
    print(f" Device: {DEVICE}  |  Context: {CONTEXT_LEN}  |  Horizon: {FORECAST_LEN}")
    print("=" * 60)

    # ── Data ────────────────────────────────────────────────────────────────
    df = load_etth1()
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = split_dataset(df)
    print(f"\nTrain: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # ── Experiments ─────────────────────────────────────────────────────────
    zero_shot_mse = run_zero_shot(X_test, y_test)
    lora_results_rank_4 = run_lora(X_train, y_train, X_test, y_test, TRAIN_FRACS, lora_rank = 4)
    lora_results_rank_8  = run_lora(X_train, y_train, X_test, y_test, TRAIN_FRACS, lora_rank = 8)
    lora_results_rank_16 = run_lora(X_train, y_train, X_test, y_test, TRAIN_FRACS, lora_rank = 16)
    sft_results   = run_sft(X_train, y_train, X_test, y_test, TRAIN_FRACS)

    # ── Results Table ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Train %':>10} {'LoRA MSE':>12} {'SFT MSE':>12}")
    print("-" * 36)
    for f in TRAIN_FRACS:
        print(f"{f*100:>9.0f}% {lora_results_rank_4[f]:>12.4f} {lora_results_rank_8[f]:>12.4f} {lora_results_rank_16[f]:>12.4f} {sft_results[f]:>12.4f}")
    print(f"\n{'Zero-Shot MSE':>23} {zero_shot_mse:>12.4f}")
    print("=" * 60)

    # ── Save Results to CSV ──────────────────────────────────────────────────
    rows = []
    for f in TRAIN_FRACS:
        rows.append({
            "train_pct":      f * 100,
            "lora_r4_mse":    lora_results_rank_4[f],
            "lora_r8_mse":    lora_results_rank_8[f],
            "lora_r16_mse":   lora_results_rank_16[f],
            "sft_mse":        sft_results[f],
            "zero_shot_mse":  zero_shot_mse,
        })
    results_df = pd.DataFrame(rows)
    csv_path = "timesfm_etth1_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to: {csv_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_results(zero_shot_mse, lora_results_rank_4, lora_results_rank_8, lora_results_rank_16, sft_results,
                 save_path="timesfm_etth1_results.png")


if __name__ == "__main__":
    main()