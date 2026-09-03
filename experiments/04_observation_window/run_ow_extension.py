"""01_run_ow_extension.py — Observation-window extension (OW 32 & 64, four F1 families).

Supervisor directive (2026-07-19): run OW 32 and 64 on all four families, evaluating only the
F1-optimised (headline) model of each. OW 16 stays as the already-published reference; we do NOT
retrain it. See PLAN.md.

Protocol is the frozen one — reused verbatim:
  train  = 12_unified_engine.train_run (train-only z-score, BCEWithLogitsLoss(pos_weight),
           <=100 epochs, early-stop patience 15, plateau LR, select="f1" = F1->acc->AUC checkpoint)
  eval   = f1_optimization/00_common: tau* = argmax val-F1 (best_threshold), then test set03 once
           at tau* and at 0.5; per-seed mean +/- std and the 5-seed probability ensemble (+ CM).
The ONLY differences vs OW 16: the observation window (32/64) and thus the sequence tensors.

Each family runs its published F1 recipe with pos_weight held fixed at its OW-16 value (single-
variable isolation; class balance is ~constant across windows so this is immaterial):
  BiLSTM-F1       lr1e-3  do0.3 h256 nl2                                   pw 1.682
  Transformer-F1  d128 nhead4 L4 ff512 do0.1 pool=last pos=sin, plateau   pw 2.5   (searched)
  GRU-F1          lr5e-4  do0.3 h256 nl2                                   pw 1.682
  Vanilla RNN-F1  lr1e-4  do0.2 h256 nl2                                   pw 1.682

CPU only (recurrent training is bit-reproducible on CPU; transformer is context-free).
Writes: runs_ow/<label>/ow<W>/seed<s>/{best.pt,norm_*,final.json}, 01_ow_results.json/.csv.
"""
import csv
import importlib.util
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _load("engine12", ROOT / "src" / "engine.py")
C = _load("f1_common", ROOT / "src" / "metrics.py")

CPU = torch.device("cpu")
SEEDS = [42, 0, 1, 2, 3]
WINDOWS = [32, 64]
TRAIN_SETS, VAL_SETS, TEST_SETS = {"set01", "set02", "set04"}, {"set05", "set06"}, {"set03"}

# Each family's published F1 recipe (config, fixed pos_weight, display label, source arm).
SEARCHED_TF_CFG = dict(d_model=128, nhead=4, num_layers=4, dim_ff=512, dropout=0.1,
                       pool="last", pos="sin", lr=1e-3, schedule="plateau",
                       weight_decay=1e-5, optimizer="adam")
FAMILIES = [
    dict(key="bilstm",      label="BiLSTM-F1",      source="LSTM",
         cfg=dict(lr=1e-3, dropout=0.3, hidden=256, num_layers=2), pw=1.682),
    dict(key="transformer", label="Transformer-F1", source="pre-LN Transformer encoder",
         cfg=dict(SEARCHED_TF_CFG), pw=2.5),
    dict(key="gru",         label="GRU-F1",         source="GRU",
         cfg=dict(lr=5e-4, dropout=0.3, hidden=256, num_layers=2), pw=1.682),
    dict(key="birnn",       label="Vanilla RNN-F1", source="vanilla (Elman) RNN, tanh",
         cfg=dict(lr=1e-4, dropout=0.2, hidden=256, num_layers=2), pw=1.682),
]


def load_window(W):
    """Load OW-W clean sequences and split by recording set (same rule as OW 16)."""
    d = HERE / f"seq_ow{W}"
    X = np.load(d / "X.npy").astype(np.float32)
    y = np.load(d / "y.npy").astype(np.float32)
    meta = pickle.load(open(d / "meta.pkl", "rb"))
    sid = np.array([m["set_id"] for m in meta])
    assert X.shape[1] == W and X.shape[2] == 5, f"bad shape {X.shape} for OW{W}"
    tr, va, te = (np.isin(sid, sorted(s)) for s in (TRAIN_SETS, VAL_SETS, TEST_SETS))
    return (X[tr], y[tr], X[va], y[va], X[te], y[te])


def transformer_builder_for(W):
    """The engine's transformer builder hardcodes seq_len=16; rebuild it so the sinusoidal
    positional-encoding buffer spans W tokens (sin-PE extrapolates to any length)."""
    def _build(cfg):
        return E.TransformerIntentPredictor(
            input_dim=5, d_model=cfg["d_model"], nhead=cfg.get("nhead", 4),
            num_layers=cfg["num_layers"], dim_ff=cfg["dim_ff"], dropout=cfg["dropout"],
            pool=cfg["pool"], pos=cfg["pos"], seq_len=W)
    return _build


def build_model(fam, W):
    if fam["key"] == "transformer":
        return transformer_builder_for(W)(fam["cfg"])
    return E.MODEL_REGISTRY[fam["key"]](fam["cfg"])


def main():
    out_root = HERE / "runs_ow"
    results = {}
    t_start = time.time()

    for W in WINDOWS:
        data = load_window(W)
        Xtr, ytr, Xva, yva, Xte, yte = data
        n_tr, n_va, n_te = len(ytr), len(yva), len(yte)
        # point the engine's transformer builder at this window (recurrent builders are len-agnostic)
        E.MODEL_REGISTRY["transformer"] = transformer_builder_for(W)
        print(f"\n{'='*78}\nOW {W}: train {n_tr} / val {n_va} / test {n_te} "
              f"(test pos {int(yte.sum())}, {yte.mean()*100:.1f}%)\n{'='*78}")

        for fam in FAMILIES:
            per_seed = []
            pv_list, pt_list = [], []
            for s in SEEDS:
                run_dir = out_root / fam["label"].replace(" ", "_") / f"ow{W}" / f"seed{s}"
                r = E.train_run(fam["key"], fam["cfg"], s, CPU, data,
                                pos_weight=fam["pw"], select="f1", out_dir=run_dir)
                # probabilities from the saved F1 checkpoint (CPU, full batch — matches OW-16 eval)
                pf = C.prob_fn_from_run_dir(run_dir, build_model(fam, W))
                pv, pt = pf(Xva), pf(Xte)
                pv_list.append(pv); pt_list.append(pt)
                tau = C.best_threshold(yva, pv)
                m_tau = C.metrics_at(yte, pt, tau)
                m_05 = C.metrics_at(yte, pt, 0.5)
                per_seed.append(dict(seed=s, tau=float(tau), n_params=r["n_params"],
                                     best_epoch=r["best_epoch"], seconds=r["seconds"],
                                     test=m_tau, test_at_05=m_05))
                print(f"  {fam['label']:16s} OW{W} seed{s:<3d} tau={tau:.3f} "
                      f"F1={m_tau['f1']:.4f} acc={m_tau['acc']:.4f} auc={m_tau['auc']:.4f} "
                      f"({r['seconds']}s, ep{r['best_epoch']})")

            # 5-seed probability ensemble (deployable predictor + confusion matrix)
            pv_e = np.mean(pv_list, axis=0)
            pt_e = np.mean(pt_list, axis=0)
            tau_e = C.best_threshold(yva, pv_e)
            me = C.metrics_at(yte, pt_e, tau_e)
            pred_e = (pt_e >= tau_e).astype(int)
            yb = yte.astype(int)
            cm = [[int(((yb == 0) & (pred_e == 0)).sum()), int(((yb == 0) & (pred_e == 1)).sum())],
                  [int(((yb == 1) & (pred_e == 0)).sum()), int(((yb == 1) & (pred_e == 1)).sum())]]
            f1s = np.array([p["test"]["f1"] for p in per_seed])
            accs = np.array([p["test"]["acc"] for p in per_seed])
            aucs = np.array([p["test"]["auc"] for p in per_seed])

            results.setdefault(f"ow{W}", {})[fam["label"]] = dict(
                source=fam["source"], cfg=fam["cfg"], pos_weight=fam["pw"],
                n_params=per_seed[0]["n_params"], n_train=n_tr, n_val=n_va, n_test=n_te,
                per_seed=per_seed,
                test_f1_mean=float(f1s.mean()), test_f1_std=float(f1s.std(ddof=1)),
                test_acc_mean=float(accs.mean()), test_acc_std=float(accs.std(ddof=1)),
                test_auc_mean=float(aucs.mean()), test_auc_std=float(aucs.std(ddof=1)),
                ens=dict(tau=float(tau_e), test=me, test_confusion_matrix=cm))
            print(f"  -> {fam['label']:16s} OW{W}: F1 {f1s.mean():.4f}±{f1s.std(ddof=1):.4f} "
                  f"acc {accs.mean():.4f} auc {aucs.mean():.4f} | ens F1 {me['f1']:.4f}@{tau_e:.3f}\n")

    (HERE / "01_ow_results.json").write_text(json.dumps(results, indent=2))
    # flat CSV: per-seed-mean rows
    with open(HERE / "01_ow_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window", "family", "params", "n_train", "n_test", "pos_weight",
                    "f1_mean", "f1_std", "acc_mean", "auc_mean", "ens_f1", "ens_tau"])
        for W in WINDOWS:
            for fam in FAMILIES:
                r = results[f"ow{W}"][fam["label"]]
                w.writerow([W, fam["label"], r["n_params"], r["n_train"], r["n_test"],
                            fam["pw"], round(r["test_f1_mean"], 4), round(r["test_f1_std"], 4),
                            round(r["test_acc_mean"], 4), round(r["test_auc_mean"], 4),
                            round(r["ens"]["test"]["f1"], 4), round(r["ens"]["tau"], 3)])

    print(f"\nTOTAL {time.time()-t_start:.0f}s. wrote 01_ow_results.json, 01_ow_results.csv, runs_ow/")


if __name__ == "__main__":
    main()
