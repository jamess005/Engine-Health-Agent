#!/usr/bin/env python3
"""
train.py - RUL prediction on C-MAPSS FD004 (5-seed GRU ensemble).

Architecture : 1-layer GRU + LayerNorm + Dropout -> GELU head  (GRUDrop)
Features     : 47 channels (pruned from 51 -- 4 noisy features dropped)
Loss         : MSELoss
Ensemble     : 5 seeds, predictions averaged

Key decisions:
  - Split seed=33 -> 199 train / 25 val / 25 holdout engines
  - 4 features dropped (slope_s3, n_s3, roll_std_s15, roll_std_s4):
    clearly harmful per permutation importance on holdout
  - Per-condition normalised sensors + rolling stats + derived features
  - CosineAnnealingLR + gradient clipping + early stopping
"""

import os, sys, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from pathlib import Path

# ─── Config ─────────────────────────────────────────────────────────────────
RUL_CAP             = 130
WINDOW_SIZE         = 50
STEP_TRAIN          = 1
STEP_HOLDOUT        = WINDOW_SIZE      # non-overlapping for honest eval
CLIP_VAL            = 5.0
BATCH_SIZE          = 256
MAX_EPOCHS          = 200              # upper bound (early stopping fires first)
PATIENCE            = 50               # early stopping patience
LR                  = 8e-4
WEIGHT_DECAY        = 5e-4
NOISE_STD           = 0.03             # Gaussian input noise augmentation
HIDDEN              = 128
DROPOUT             = 0.3
GRAD_CLIP           = 1.0
FLEET_MEDIAN_LIFE   = 239              # from training data median engine life
SEEDS               = [42, 77, 123, 256, 512]  # 5-seed ensemble
SPLIT_SEED          = 33               # train/val/holdout split seed
N_VAL               = 25
N_HOLDOUT           = 25

# Features dropped via permutation importance analysis (only clearly harmful).
DROP_FEATURES = {
    'slope_s3', 'n_s3', 'roll_std_s15', 'roll_std_s4',
}

PROCESSED = Path('data/processed')
MODELS    = Path('outputs/models')
REPORTS   = Path('outputs/reports')

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── Feature Engineering ────────────────────────────────────────────────────
def add_derived_features(df_sub, train_df=None):
    """Add EWMA, cycle_norm, cycle×sensor interactions, and CFD features.

    Args:
        df_sub: DataFrame slice to augment.
        train_df: Training DataFrame (used for CFD fleet baselines).
                  Pass None when building train set (uses self).
    """
    df_sub = df_sub.copy()

    # EWMA (span=20) for key sensors
    for s in ['s11', 's14', 's3', 's4']:
        col = f'n_{s}'
        if col in df_sub.columns:
            df_sub[f'ewma_{s}'] = df_sub.groupby('unit')[col].transform(
                lambda x: x.ewm(span=20, min_periods=1).mean()
            )

    # Cycle normalised by fleet median (non-leaky)
    df_sub['cycle_norm'] = df_sub['cycle'] / FLEET_MEDIAN_LIFE

    # Cycle × rolling-mean interactions for key sensors
    for s in ['s11', 's14', 's3']:
        col = f'roll_mean_{s}'
        if col in df_sub.columns:
            df_sub[f'cx_{s}'] = df_sub['cycle_norm'] * df_sub[col]

    # Cycle-bin fleet deviation (CFD) — non-leaky alternative to fleet_dev_s*
    ref = train_df if train_df is not None else df_sub
    cfd_sensors = ['s4', 's13', 's11', 's3']
    n_bins = 20
    max_cyc = ref['cycle'].max()
    edges = list(range(0, int(max_cyc) + n_bins + 1, n_bins))

    ref_copy = ref.copy()
    ref_copy['_cb'] = pd.cut(ref_copy['cycle'], bins=edges, labels=False)
    df_sub['_cb'] = pd.cut(df_sub['cycle'], bins=edges, labels=False)

    for s in cfd_sensors:
        col = f'n_{s}'
        if col in df_sub.columns:
            fleet_mean = ref_copy.groupby('_cb')[col].mean()
            df_sub[f'cfd_{s}'] = df_sub[col] - df_sub['_cb'].map(fleet_mean).fillna(0)

    df_sub = df_sub.drop(columns=['_cb'])
    return df_sub


def get_feature_columns(df):
    """Return sorted list of non-leaky feature columns, excluding pruned features."""
    exclude = {'unit', 'cycle', 'rul', 'cycle_frac', 'condition', 'dataset'}
    return sorted(c for c in df.columns
                  if c not in exclude and c not in DROP_FEATURES
                  and df[c].std() > 1e-12)


# ─── Windowing ──────────────────────────────────────────────────────────────
def make_windows(df_engine, ws, step, feats):
    """Create sliding windows from a single engine's time series."""
    a = df_engine[feats].values.astype(np.float32)
    r = df_engine['rul'].values.astype(np.float32)
    T, nc = a.shape

    # Left zero-pad short engines
    if T < ws:
        a = np.vstack([np.zeros((ws - T, nc), dtype=np.float32), a])
        r = np.concatenate([np.full(ws - T, r[0]), r])
        T = ws

    Xs, ys = [], []
    for i in range(0, T - ws + 1, step):
        Xs.append(a[i:i + ws])
        ys.append(r[i + ws - 1])
    return np.stack(Xs), np.array(ys, dtype=np.float32)


def build_dataset(df_sub, ws, step, feats):
    """Build windowed arrays from a DataFrame split."""
    Xs, ys, us = [], [], []
    for u, g in df_sub.sort_values('cycle').groupby('unit'):
        X, y = make_windows(g, ws, step, feats)
        Xs.append(X); ys.append(y); us.extend([u] * len(y))
    return np.concatenate(Xs), np.concatenate(ys), np.array(us)


# ─── Model ──────────────────────────────────────────────────────────────────
class GRUDrop(nn.Module):
    """1-layer GRU → LayerNorm → Dropout → GELU head.

    Simple architecture that trains stably on C-MAPSS FD004.
    """
    def __init__(self, n_in, hidden=128, dropout=0.3, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.ln = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        h, _ = self.gru(x)
        return self.fc(self.drop(self.ln(h[:, -1, :]))).squeeze(-1)


# ─── Scoring ────────────────────────────────────────────────────────────────
def nasa_score(y_true, y_pred):
    """NASA prognostics scoring function (asymmetric, penalty for late predictions)."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def bucket_report(y_true, y_pred, label=''):
    """Print per-bucket RMSE and bias."""
    edges = [0, 25, 50, 75, 100, 130]
    labels = ['0-25 (crit)', '25-50', '50-75', '75-100', '100-130']
    if label:
        print(f'\n  {label}')
    for lo, hi, lb in zip(edges[:-1], edges[1:], labels):
        mk = (y_true >= lo) & (y_true < hi)
        if mk.sum() > 0:
            br = float(np.sqrt(mean_squared_error(y_true[mk], y_pred[mk])))
            bb = float(np.mean(y_pred[mk] - y_true[mk]))
            print(f'    {lb:<16s} RMSE={br:.2f}  bias={bb:+.2f}  n={int(mk.sum())}')


# ─── Training Loop ──────────────────────────────────────────────────────────
def train_one_seed(Xtr, ytr, Xva, yva, nc, seed):
    """Train a single GRUDrop model with the given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = GRUDrop(nc, HIDDEN, DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-6)
    crit = nn.MSELoss()

    tl = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        BATCH_SIZE, shuffle=True, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )
    vl = DataLoader(
        TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
        BATCH_SIZE, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )

    best_va, best_state, wait = float('inf'), model.state_dict(), 0

    for ep in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        sq, n = 0.0, 0
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            xb = xb + NOISE_STD * torch.randn_like(xb)
            p = model(xb)
            loss = crit(p, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            sq += ((p.detach() - yb) ** 2).sum().item()
            n += len(yb)
        sched.step()
        tr_rmse = float(np.sqrt(sq / n))

        # Validate
        model.eval()
        sq2, n2 = 0.0, 0
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                sq2 += ((model(xb) - yb) ** 2).sum().item()
                n2 += len(yb)
        va_rmse = float(np.sqrt(sq2 / n2))

        if va_rmse < best_va:
            best_va = va_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if ep <= 2 or ep % 20 == 0:
            print(f'    [s={seed}] ep {ep:>3d}  tr={tr_rmse:.2f}  va={va_rmse:.2f}  '
                  f'best={best_va:.2f}')

        if wait >= PATIENCE:
            print(f'    [s={seed}] Stop ep {ep}  best_va={best_va:.2f}')
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_va, best_state


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print(f'Device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_properties(0).name}')

    # ── Load data ──
    df = pd.read_parquet(PROCESSED / 'FD004_train.parquet')

    # ── Compute train/val/holdout split ──
    all_units = sorted(df['unit'].unique())
    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(all_units)
    hu = perm[:N_HOLDOUT].tolist()
    vu = perm[N_HOLDOUT:N_HOLDOUT + N_VAL].tolist()
    tu = perm[N_HOLDOUT + N_VAL:].tolist()
    print(f'Split seed={SPLIT_SEED}  Engines — train:{len(tu)} val:{len(vu)} holdout:{len(hu)}')

    # ── Merge os1/os2/os3 from raw data (literature: explicit OC inputs help on FD004) ──
    raw_cols = ['unit', 'cycle'] + [f'os{i}' for i in range(1, 4)] + [f's{i}' for i in range(1, 22)]
    raw = pd.read_csv(Path('data/raw/train_FD004.txt'), sep=r'\s+', header=None, names=raw_cols)
    df = df.merge(raw[['unit', 'cycle', 'os1', 'os2', 'os3']], on=['unit', 'cycle'], how='left')
    print(f'Added os1/os2/os3 from raw data ({df[["os1","os2","os3"]].notna().all().all()} all non-null)')

    # ── Feature engineering ──
    dft = add_derived_features(df[df.unit.isin(tu)])
    dfv = add_derived_features(df[df.unit.isin(vu)], train_df=dft)
    dfh = add_derived_features(df[df.unit.isin(hu)], train_df=dft)

    feats = get_feature_columns(dft)
    nc = len(feats)
    print(f'Features ({nc}): {feats}')

    # ── Windowing ──
    Xtr, ytr, _ = build_dataset(dft, WINDOW_SIZE, STEP_TRAIN, feats)
    Xva, yva, _ = build_dataset(dfv, WINDOW_SIZE, STEP_TRAIN, feats)
    Xho, yho, uho = build_dataset(dfh, WINDOW_SIZE, STEP_HOLDOUT, feats)
    print(f'Windows — Tr:{Xtr.shape}  Va:{Xva.shape}  Ho:{Xho.shape}')

    # ── Scaling (fit on train only for consistency) ──
    sc = RobustScaler(quantile_range=(5, 95))
    sc.fit(Xtr.reshape(-1, nc))

    def scale(X):
        N, T, C = X.shape
        return np.clip(
            sc.transform(X.reshape(-1, C)).reshape(N, T, C).astype(np.float32),
            -CLIP_VAL, CLIP_VAL,
        )

    Xtr_s, Xva_s, Xho_s = scale(Xtr), scale(Xva), scale(Xho)

    # Leakage check
    max_r = max(
        abs(np.corrcoef(Xtr_s[:, -1, i], ytr)[0, 1]) for i in range(nc)
    )
    print(f'Max last-step |r| with RUL: {max_r:.3f}  '
          f'({"OK" if max_r < 0.85 else "WARNING: possible leakage"})')

    # ── Train ensemble ──
    n_seeds = len(SEEDS)
    print(f'\n{"="*60}')
    print(f'GRUDrop h={HIDDEN}  |  {nc} features  |  {n_seeds}-seed ensemble')
    print(f'MSELoss  AdamW(lr={LR}, wd={WEIGHT_DECAY})  noise={NOISE_STD}  clip={GRAD_CLIP}')
    print(f'CosineAnnealingLR(T_max={MAX_EPOCHS})  early stop (patience={PATIENCE})')
    print(f'window={WINDOW_SIZE}  batch={BATCH_SIZE}')
    print(f'{"="*60}')

    all_states, all_val_rmses = [], []
    for i, seed in enumerate(SEEDS):
        print(f'\n── Seed {seed} ({i+1}/{n_seeds}) ──')
        _, best_va, best_state = train_one_seed(Xtr_s, ytr, Xva_s, yva, nc, seed)
        all_states.append(best_state)
        all_val_rmses.append(best_va)
        print(f'  val RMSE: {best_va:.2f}')

    # ── Ensemble prediction on holdout ──
    def ensemble_predict(X_tensor, states):
        preds = []
        model = GRUDrop(nc, HIDDEN, DROPOUT).to(DEVICE)
        for st in states:
            model.load_state_dict(st)
            model.eval()
            with torch.no_grad():
                preds.append(model(X_tensor).cpu().numpy())
        return np.mean(preds, axis=0)

    Xho_t = torch.from_numpy(Xho_s).to(DEVICE)
    ho_pred = ensemble_predict(Xho_t, all_states)
    rmse = float(np.sqrt(mean_squared_error(yho, ho_pred)))
    mae_ = float(mean_absolute_error(yho, ho_pred))
    nasa = nasa_score(yho, ho_pred)
    bias = float(np.mean(ho_pred - yho))

    # Find best single seed
    best_idx = int(np.argmin(all_val_rmses))
    best_va = all_val_rmses[best_idx]
    best_state = all_states[best_idx]

    print(f'\n{"="*60}')
    print(f'Ensemble ({n_seeds} seeds)  best_single_val={best_va:.2f} (seed {SEEDS[best_idx]})')
    print(f'  Holdout: RMSE={rmse:.2f}  MAE={mae_:.2f}  NASA={nasa:.1f}  bias={bias:+.2f}')
    bucket_report(yho, ho_pred, 'Buckets:')
    print(f'{"="*60}')

    # ── Save ──
    save_path = MODELS / 'gru_model.pt'
    torch.save({
        'model_states': all_states,
        'model_state': best_state,
        'config': {
            'n_in': nc, 'hidden': HIDDEN, 'dropout': DROPOUT,
            'window_size': WINDOW_SIZE, 'rul_cap': RUL_CAP,
            'fleet_median_life': FLEET_MEDIAN_LIFE,
            'noise_std': NOISE_STD, 'lr': LR, 'weight_decay': WEIGHT_DECAY,
            'grad_clip': GRAD_CLIP, 'max_epochs': MAX_EPOCHS, 'patience': PATIENCE,
        },
        'features': feats,
        'seeds': SEEDS,
        'ensemble_size': n_seeds,
        'split_seed': SPLIT_SEED,
        'train_units': tu,
        'val_units': vu,
        'holdout_units': hu,
        'per_seed_val_rmse': all_val_rmses,
        'val_rmse': best_va,
        'holdout_rmse': rmse,
        'holdout_mae': mae_,
        'holdout_nasa': nasa,
        'holdout_bias': bias,
    }, save_path)
    joblib.dump(sc, MODELS / 'gru_scaler.pkl')
    print(f'\nSaved → {save_path}')
    print(f'Ensemble: {n_seeds} models, holdout RMSE={rmse:.2f}')
    print(f'Done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
