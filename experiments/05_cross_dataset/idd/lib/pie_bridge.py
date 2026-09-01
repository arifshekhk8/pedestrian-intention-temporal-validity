"""pie_bridge.py — read-only bridge to the frozen PIE implementation.

Loads the existing project's engine, model classes and eval helpers via importlib WITHOUT
importing them as packages and WITHOUT modifying a single byte on disk. Same technique the
JAAD track used (`journal_prep/cross_dataset_validation/03_jaad_fourfamily_engine.py`).

Nothing here writes outside `idd_ped_crossdataset/`.
"""
from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import numpy as np

FOLDER = Path(__file__).resolve().parent.parent
ROOT = FOLDER.parent

ENGINE_PATH = ROOT / "journal_prep" / "issue12_unified_pipeline" / "12_unified_engine.py"
COMMON_PATH = ROOT / "f1_optimization" / "00_common.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_engine():
    return _load("iddped_engine_ro", ENGINE_PATH)


def load_common():
    return _load("iddped_common_ro", COMMON_PATH)


# ---------------------------------------------------------------------------
# The four headline PIE models ("-F1", one per family) — the models the paper reports.
# cfg values verified against the stored final.json of each run dir (PHASE0 audit §2.7).
# run_dir_tmpl is formatted with the seed.
PIE_ARMS = {
    "BiLSTM-F1": dict(
        family="bilstm",
        cfg=dict(lr=1e-3, dropout=0.3, hidden=256, num_layers=2),
        run_dir_tmpl="f1_optimization/runs_f1/lstm_lr1e-03_do0.3_h256_nl2/pw1.682/seed{seed}",
        n_params=2_237_313,
    ),
    "Transformer-F1": dict(
        family="transformer",
        cfg=dict(d_model=128, nhead=4, num_layers=4, dim_ff=512, dropout=0.1,
                 pool="last", pos="sin", lr=1e-3, schedule="plateau",
                 weight_decay=1e-5, optimizer="adam"),
        run_dir_tmpl="f1_optimization/runs_f1/transformer_searched/pw1.682/seed{seed}",
        n_params=794_241,
    ),
    "GRU-F1": dict(
        family="gru",
        cfg=dict(lr=5e-4, dropout=0.3, hidden=256, num_layers=2),
        run_dir_tmpl="gru/phase4_final/runs_final/gru_f1_winner/seed{seed}",
        n_params=1_678_209,
    ),
    "Vanilla_RNN-F1": dict(
        family="birnn",
        cfg=dict(lr=1e-4, dropout=0.2, hidden=256, num_layers=2),
        run_dir_tmpl="rnn/phase4_final/runs_final/rnn_f1_winner/seed{seed}",
        n_params=560_001,
    ),
}

SEEDS = [42, 0, 1, 2, 3]

# The frozen BiLSTM baseline, used only for the parity gate.
LSTM_PARITY_DIR = ROOT / "journal_prep" / "issue2_clean_protocol" / "runs_clean" / "multiseed"
BASELINE_LSTM_CFG = dict(lr=1e-3, dropout=0.3, hidden=128, num_layers=2)


def arm_run_dir(arm: str, seed: int) -> Path:
    return ROOT / PIE_ARMS[arm]["run_dir_tmpl"].format(seed=seed)


# ---------------------------------------------------------------------------
def load_iddped(seq_dir: Path):
    """Return (X, y, meta) for a built IDD-PeD sequence directory."""
    X = np.load(seq_dir / "X.npy").astype(np.float32)
    y = np.load(seq_dir / "y.npy").astype(np.float32)
    with open(seq_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    assert len(X) == len(y) == len(meta)
    return X, y, meta


def split_masks(meta):
    s = np.array([m["split"] for m in meta])
    return {k: (s == k) for k in ("train", "val", "test")}


def ped_clusters(meta, mask=None):
    """Index groups for the pedestrian-cluster bootstrap: one group per pedestrian track."""
    idx = np.arange(len(meta)) if mask is None else np.where(mask)[0]
    keys = np.array([f"{meta[i]['video_id']}/{meta[i]['ped_id']}" for i in idx])
    return [idx[keys == k] for k in np.unique(keys)]


# ---------------------------------------------------------------------------
PIE_W, PIE_H = 1920, 1080


def to_pie_frame(X, meta, mode="rescale"):
    """Map IDD-PeD raw-pixel boxes into PIE's 1920x1080 coordinate frame.

    Required for zero-shot transfer only: the PIE models were trained on raw pixel
    coordinates from a 1920x1080 camera and standardized with PIE's own train statistics,
    while 29 of IDD-PeD's 33 videos are 1920x1440 (GoPro 4:3).

    mode='rescale' : scale x by 1920/W and y by 1080/H per video, so every box lives in a
                     1920x1080 frame. This matches the *coordinate convention* the models
                     expect. It is an approximation, NOT a true geometric rectification —
                     a 4:3 GoPro has a different vertical field of view from PIE's rig, so
                     the same pixel row does not correspond to the same world elevation.
                     Disclosed as such.
    mode='raw'     : feed IDD-PeD pixels unchanged (the sensitivity check).

    Speed (channel 4) is never touched: both datasets record km/h on the same scale.
    """
    if mode == "raw":
        return X.copy()
    if mode != "rescale":
        raise ValueError(mode)
    out = X.copy()
    w = np.array([m["width"] for m in meta], dtype=np.float32)[:, None]
    h = np.array([m["height"] for m in meta], dtype=np.float32)[:, None]
    sx, sy = PIE_W / w, PIE_H / h
    out[:, :, 0] *= sx
    out[:, :, 2] *= sx
    out[:, :, 1] *= sy
    out[:, :, 3] *= sy
    return out
