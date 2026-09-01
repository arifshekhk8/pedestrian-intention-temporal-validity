"""
03_jaad_fourfamily_engine.py  —  Cross-dataset validation, Track A1: four-family JAAD runs.

Train-on-JAAD / test-on-JAAD, bbox-only (4-D), all four families (bilstm / transformer /
gru / birnn), 5 seeds, CPU, F1-first selection -- the identical frozen protocol used for
PIE, reusing the SAME training loop (`train_run`) from
`journal_prep/issue12_unified_pipeline/12_unified_engine.py` byte-for-byte.

Constraint: this session may only write inside `journal_prep/cross_dataset_validation/`.
The unified engine hardcodes input_dim=5 in its four model-builder functions (PIE always
has bbox+ego-speed). Rather than edit that file, we `importlib`-load it read-only and
monkey-patch its in-memory MODEL_REGISTRY dict with input_dim=4 builder variants -- the
underlying model classes (BiLSTM, TransformerIntentPredictor, RecurrentIntentPredictor)
already accept `input_dim` as a constructor argument; only the engine's own wrapper
functions hardcode 5. Zero bytes on disk outside this folder are touched. train_run()
itself is called unmodified (same frozen protocol: batch 32, <=100 epochs, patience 15,
ReduceLROnPlateau, train-only z-score, pos_weight applied only in the loss).

Test set is touched exactly once per run, by this script's own `evaluate_test()` (this is
the designated single script for JAAD test evaluation -- mirrors the PIE convention that
`train_run` has no test code path at all).

Command:
  python journal_prep/cross_dataset_validation/03_jaad_fourfamily_engine.py \
      --seq-dir journal_prep/cross_dataset_validation/sequences_jaad_clean \
      --out-dir journal_prep/cross_dataset_validation/runs_jaad
"""
import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ENGINE_PATH = ROOT / "journal_prep" / "issue12_unified_pipeline" / "12_unified_engine.py"

FAMILIES = ["bilstm", "transformer", "gru", "birnn"]
SEEDS = [42, 0, 1, 2, 3]


def load_engine():
    spec = importlib.util.spec_from_file_location("unified_engine_ro", ENGINE_PATH)
    ue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ue)
    return ue


def patch_registry_4d(ue):
    """Monkey-patch the imported module's MODEL_REGISTRY in-memory (this process only)
    with input_dim=4 builders. Does not modify 12_unified_engine.py on disk."""

    def _build_bilstm_4d(cfg):
        return ue.BiLSTM(input_dim=4, hidden_dim=cfg["hidden"],
                          num_layers=cfg["num_layers"], dropout=cfg["dropout"])

    def _build_transformer_4d(cfg):
        return ue.TransformerIntentPredictor(
            input_dim=4, d_model=cfg["d_model"], nhead=cfg.get("nhead", 4),
            num_layers=cfg["num_layers"], dim_ff=cfg["dim_ff"], dropout=cfg["dropout"],
            pool=cfg["pool"], pos=cfg["pos"])

    def _build_gru_4d(cfg):
        return ue.RecurrentIntentPredictor("gru", input_dim=4, hidden_dim=cfg["hidden"],
                                           num_layers=cfg["num_layers"], dropout=cfg["dropout"])

    def _build_birnn_4d(cfg):
        return ue.RecurrentIntentPredictor("rnn", input_dim=4, hidden_dim=cfg["hidden"],
                                           num_layers=cfg["num_layers"], dropout=cfg["dropout"])

    ue.MODEL_REGISTRY["bilstm"] = _build_bilstm_4d
    ue.MODEL_REGISTRY["transformer"] = _build_transformer_4d
    ue.MODEL_REGISTRY["gru"] = _build_gru_4d
    ue.MODEL_REGISTRY["birnn"] = _build_birnn_4d


def load_jaad_splits(seq_dir: Path):
    X = np.load(seq_dir / "X.npy").astype(np.float32)
    y = np.load(seq_dir / "y.npy").astype(np.float32)
    with open(seq_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    split = np.array([m["split"] for m in meta])
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def evaluate_test(ue, model, Xte_n, yte, device):
    """The one place JAAD test windows are touched, exactly once per run."""
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(Xte_n).to(device)).squeeze(-1)).cpu().numpy()
    return ue.metrics_at(yte, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", default=str(HERE / "sequences_jaad_clean"))
    ap.add_argument("--out-dir", default=str(HERE / "runs_jaad"))
    ap.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    ue = load_engine()
    patch_registry_4d(ue)

    seq_dir = Path(args.seq_dir)
    out_root = Path(args.out_dir)
    Xtr, ytr, Xva, yva, Xte, yte = load_jaad_splits(seq_dir)
    n_pos, n_neg = int(ytr.sum()), int((ytr == 0).sum())
    pos_weight = n_neg / max(n_pos, 1)
    print(f"JAAD clean splits: train {len(ytr)} (pos {n_pos}, neg {n_neg}) | "
          f"val {len(yva)} | test {len(yte)} | pos_weight={pos_weight:.4f}")

    device = torch.device("cpu")
    all_results = []

    for family in args.families:
        cfg = ue.PRESETS[family]
        for seed in args.seeds:
            run_dir = out_root / family / f"seed{seed}"
            data = (Xtr, ytr, Xva, yva, Xte, yte)
            result = ue.train_run(family, cfg, seed, device, data,
                                  pos_weight=pos_weight, select="f1", out_dir=str(run_dir))

            # re-load the just-saved checkpoint + norm stats and touch test ONCE
            ckpt = torch.load(run_dir / "best.pt", weights_only=False)
            mean = np.load(run_dir / "norm_mean.npy")
            std = np.load(run_dir / "norm_std.npy")
            model = ue.MODEL_REGISTRY[family](cfg).to(device)
            model.load_state_dict(ckpt["model"])
            Xte_n = (Xte - mean) / std
            test_metrics = evaluate_test(ue, model, Xte_n, yte, device)
            (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

            row = {"family": family, "seed": seed, "n_params": result["n_params"],
                  "val": result["val"], "test": test_metrics}
            all_results.append(row)
            print(f"[{family} seed={seed}] val_f1={result['val']['f1']:.4f} "
                  f"val_auc={result['val']['auc']:.4f} | "
                  f"test_f1={test_metrics['f1']:.4f} test_auc={test_metrics['auc']:.4f} "
                  f"test_acc={test_metrics['acc']:.4f}")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "all_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {out_root / 'all_results.json'}")


if __name__ == "__main__":
    main()
