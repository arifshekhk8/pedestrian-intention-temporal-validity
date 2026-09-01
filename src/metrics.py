"""00_common.py — shared utilities for the F1-first optimization program (PLAN.md).

Everything here is metric/selection plumbing used by scripts 01-06: data loading with
the frozen split asserts, model builders (classes imported from the frozen sources —
pipeline/03_bilstm_model.py and transformer/phase1_setup/00_transformer_model.py, never
copied), checkpoint->probability loaders (CPU, full-batch, matching
transformer/phase5_analysis/05_compare_vs_lstm.py exactness), the pre-registered
threshold-selection rule (PLAN.md section 3.1), and metric/bootstrap helpers with F1
first-class (PLAN.md section 7).
"""
import importlib.util
import pickle
import random
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEQ_DIR = Path(os.environ.get("PCIP_SEQ_DIR", ROOT / "data" / "pie_clean"))
# Checkpoint trees are NOT in git (see docs/REPRODUCE.md); fetch the release, then set PCIP_CKPT_ROOT.
CKPT_ROOT = Path(os.environ.get("PCIP_CKPT_ROOT", ROOT / "checkpoints"))

# frozen checkpoints (never overwritten)
LSTM_MULTISEED = CKPT_ROOT / "bilstm_multiseed"
TF_RUNS_FINAL = CKPT_ROOT / "transformer_final"
# cached searches (val-only JSONs)
LSTM_GRID = CKPT_ROOT / "bilstm_grid"
TF_SEARCH = CKPT_ROOT / "transformer_search"

TRAIN_SETS = {"set01", "set02", "set04"}
VAL_SETS = {"set05", "set06"}
TEST_SETS = {"set03"}
POS_WEIGHT = 1.682          # frozen anchor (clean train split 1366 neg / 812 pos)
SEEDS = [42, 0, 1, 2, 3]
TAU_LO, TAU_HI = 0.05, 0.95  # pre-registered threshold bounds


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BiLSTM = _load_module("m03_bilstm", ROOT / "pipeline" / "03_bilstm_model.py").BiLSTMIntentPredictor
_tfm = _load_module("m00_tfmodel", ROOT / "transformer" / "phase1_setup" / "00_transformer_model.py")
TransformerIntentPredictor = _tfm.TransformerIntentPredictor
count_params = _tfm.count_params

# the three frozen configs (values verified against the stored search summaries)
BASELINE_LSTM_CFG = dict(lr=1e-3, dropout=0.3, hidden=128, num_layers=2)
SEARCHED_TF_CFG = dict(d_model=128, nhead=4, num_layers=4, dim_ff=512, dropout=0.1,
                       pool="last", pos="sin",
                       lr=1e-3, schedule="plateau", weight_decay=1e-5, optimizer="adam")
DEFAULT_TF_CFG = dict(d_model=128, nhead=4, num_layers=2, dim_ff=256, dropout=0.1,
                      pool="cls", pos="learned",
                      lr=1e-3, schedule="plateau", weight_decay=1e-5, optimizer="adam")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def load_splits():
    """(Xtr, ytr, Xva, yva, Xte, yte) raw arrays, with the frozen-protocol asserts."""
    X = np.load(SEQ_DIR / "X.npy").astype(np.float32)
    y = np.load(SEQ_DIR / "y.npy").astype(np.float32)
    with open(SEQ_DIR / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    sid = np.array([m["set_id"] for m in meta])
    assert X.shape == (4906, 16, 5), f"unexpected X shape {X.shape}"
    tr, va, te = (np.isin(sid, sorted(s)) for s in (TRAIN_SETS, VAL_SETS, TEST_SETS))
    out = (X[tr], y[tr], X[va], y[va], X[te], y[te])
    assert len(out[1]) == 2178 and len(out[3]) == 634 and len(out[5]) == 2094
    assert int(out[5].sum()) == 681, "test positive count drifted"
    return out


def build_lstm(cfg):
    return BiLSTM(input_dim=5, hidden_dim=cfg["hidden"],
                  num_layers=cfg["num_layers"], dropout=cfg["dropout"])


def build_transformer(cfg):
    return TransformerIntentPredictor(
        input_dim=5, d_model=cfg["d_model"], nhead=cfg.get("nhead", 4),
        num_layers=cfg["num_layers"], dim_ff=cfg["dim_ff"], dropout=cfg["dropout"],
        pool=cfg["pool"], pos=cfg["pos"])


def lstm_cfg_id(c):
    do = "NA" if c["num_layers"] == 1 else f"{c['dropout']:.1f}"
    return f"lr{c['lr']:.0e}_do{do}_h{c['hidden']}_nl{c['num_layers']}"


@torch.no_grad()
def prob_fn_from_run_dir(run_dir, model):
    """Load a run dir's checkpoint + norm stats into `model` (CPU) and return a closure
    that maps a RAW feature array X -> per-window sigmoid probabilities. Matches the
    phase5 comparison's exactness path (full batch, CPU, weights_only=False)."""
    run_dir = Path(run_dir)
    mean = np.load(run_dir / "norm_mean.npy")
    std = np.load(run_dir / "norm_std.npy")
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    @torch.no_grad()
    def probs(X):
        Xn = ((X - mean) / std).astype(np.float32)
        return torch.sigmoid(model(torch.from_numpy(Xn)).squeeze(-1)).numpy()

    return probs


# ---------------------------------------------------------------- metrics
def metrics_at(y, probs, thr):
    """Full metric dict at a fixed threshold. AUC/PR-AUC are threshold-free."""
    pred = (probs >= thr).astype(int)
    return dict(
        f1=f1_score(y, pred, zero_division=0),
        acc=accuracy_score(y, pred),
        auc=roc_auc_score(y, probs),
        pr_auc=average_precision_score(y, probs),
        prec=precision_score(y, pred, zero_division=0),
        rec=recall_score(y, pred, zero_division=0),
    )


def auc_fast(y, s):
    """Tie-corrected Mann-Whitney ROC-AUC (Issue-4 form) — cheap inside bootstraps."""
    n_pos = y.sum()
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def f1_from_preds(y_bool, pred_bool):
    tp = np.count_nonzero(pred_bool & y_bool)
    fp = np.count_nonzero(pred_bool & ~y_bool)
    fn = np.count_nonzero(~pred_bool & y_bool)
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def acc_from_preds(y_bool, pred_bool):
    return np.count_nonzero(pred_bool == y_bool) / y_bool.size


# ---------------------------------------------------------------- threshold rule
def best_threshold(y, probs, lo=TAU_LO, hi=TAU_HI):
    """PLAN.md section 3.1: tau* = argmax F1 over all achievable cutoffs on VAL probs;
    tie-break higher acc, then smaller |tau-0.5|; bounded [lo, hi]; fallback 0.5.
    NEVER call this on test data — callers pass val probabilities only."""
    y_bool = y.astype(bool)
    cands = np.unique(probs)
    cands = cands[(cands >= lo) & (cands <= hi)]
    cands = np.unique(np.append(cands, 0.5))
    best_key, best_t = None, 0.5
    for t in cands:
        pred = probs >= t
        f1 = f1_from_preds(y_bool, pred)
        acc = acc_from_preds(y_bool, pred)
        key = (round(f1, 10), round(acc, 10), -abs(t - 0.5))
        if best_key is None or key > best_key:
            best_key, best_t = key, float(t)
    return best_t


# ---------------------------------------------------------------- bootstraps
def paired_bootstrap(y, va, vb, stat_a, stat_b, B=10000, seed=42):
    """delta_b = stat_a(y[idx], va[idx]) - stat_b(y[idx], vb[idx]) with the SAME
    resample indices for both sides each iteration (paired; rng reset per comparison).
    va/vb may be probability vectors (with an AUC stat) or boolean prediction vectors
    (with f1/acc stats) — the stat functions define the metric."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        deltas[b] = stat_a(yb, va[idx]) - stat_b(yb, vb[idx])
    return deltas


def bootstrap_ci(y, v, stat, B=10000, seed=42):
    """Independent percentile 95% CI on one model's own statistic."""
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        vals[b] = stat(y[idx], v[idx])
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))
