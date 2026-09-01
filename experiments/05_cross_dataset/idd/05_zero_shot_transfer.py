"""05_zero_shot_transfer.py — EXPERIMENT A: true out-of-domain generalization.

The four frozen PIE-trained headline models (BiLSTM-F1 / Transformer-F1 / GRU-F1 /
Vanilla_RNN-F1, 5 seeds each = 20 checkpoints) are evaluated **directly** on IDD-PeD.

Strictly enforced:
  * NO fine-tuning on IDD-PeD.
  * NO hyperparameter tuning on IDD-PeD.
  * NO threshold tuning on IDD-PeD — the operating point tau* is fitted on **PIE's**
    validation split and carried over unchanged. tau=0.5 is also reported.
  * NO IDD-PeD normalization — every window is standardized with the **PIE training**
    mean/std saved beside each checkpoint (`norm_mean.npy` / `norm_std.npy`).
  * NO use of IDD-PeD labels except to score the predictions.

Leads with AUC / PR-AUC because they are threshold-free and therefore robust to the very
different base rate (IDD-PeD test 7.1 % positive vs PIE 32.5 %); F1/Acc at a fixed,
non-re-tuned threshold follow.

A PARITY GATE runs first: the frozen BiLSTM's per-seed PIE test AUC is regenerated from its
checkpoints and asserted equal to the stored final.json values, exactly as
`rnn/phase4_final/05_rnn_test_eval.py` does. If the frozen baseline has drifted, abort
before emitting any IDD-PeD number.

Writes  results/expA_zero_shot.json / .csv
        results/expA_probs/*.npy       (cached probability vectors for the bootstrap)

Run from the repo root:
    python idd_ped_crossdataset/scripts/05_zero_shot_transfer.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
FOLDER = HERE.parent
sys.path.insert(0, str(FOLDER / "src"))
from pie_bridge import (PIE_ARMS, SEEDS, BASELINE_LSTM_CFG, LSTM_PARITY_DIR,  # noqa: E402
                        arm_run_dir, load_common, load_engine, load_iddped,
                        ped_clusters, split_masks, to_pie_frame)

SEQ_DIR = FOLDER / "data" / "sequences_iddped_clean"
SEQ_DIR_CP = FOLDER / "data" / "sequences_iddped_clean_cp_anchor"
OUT_JSON = FOLDER / "results" / "expA_zero_shot.json"
OUT_CSV = FOLDER / "results" / "expA_zero_shot.csv"
PCACHE = FOLDER / "results" / "expA_probs"

B_BOOT = 10_000
RNG_SEED = 42


def cluster_ci(y, v, stat, groups, B=B_BOOT, seed=RNG_SEED):
    """Pedestrian-cluster percentile CI — resample TRACKS, each contributing all its windows.
    Same discipline as f1_optimization/07_cluster_bootstrap.py."""
    rng = np.random.default_rng(seed)
    k = len(groups)
    vals = np.empty(B)
    for b in range(B):
        idx = np.concatenate([groups[i] for i in rng.integers(0, k, k)])
        vals[b] = stat(y[idx], v[idx])
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def parity_gate(common, Xte_pie, yte_pie):
    print("=== PARITY GATE: frozen BiLSTM PIE test AUC (checkpoints vs stored final.json) ===")
    worst = 0.0
    for seed in SEEDS:
        rd = LSTM_PARITY_DIR / f"seed{seed}"
        model = common.build_lstm(BASELINE_LSTM_CFG)
        pf = common.prob_fn_from_run_dir(rd, model)
        recomputed = roc_auc_score(yte_pie, pf(Xte_pie))
        stored = json.loads((rd / "final.json").read_text())["test"]["auc"]
        d = abs(recomputed - stored)
        worst = max(worst, d)
        print(f"  seed {seed:2d}: recomputed {recomputed:.6f}  stored {stored:.6f}  |Δ| {d:.2e}")
    assert worst < 1e-4, f"PARITY GATE FAILED: frozen BiLSTM drifted (max |Δ| = {worst:.2e})"
    print(f"  PARITY GATE PASS (max |Δ| = {worst:.2e})\n")
    return worst


def build_model(engine, arm):
    spec = PIE_ARMS[arm]
    return engine.MODEL_REGISTRY[spec["family"]](spec["cfg"])


def main():
    common = load_common()
    engine = load_engine()

    # ---- PIE side (val for tau*, test for the parity gate) -----------------
    Xtr_p, ytr_p, Xva_p, yva_p, Xte_p, yte_p = common.load_splits()
    parity_delta = parity_gate(common, Xte_p, yte_p)

    # ---- IDD-PeD side ------------------------------------------------------
    results, csv_rows = [], []
    PCACHE.mkdir(parents=True, exist_ok=True)

    for protocol, seq_dir in (("strict", SEQ_DIR), ("cp_anchor", SEQ_DIR_CP)):
        X_raw, y_all, meta = load_iddped(seq_dir)
        masks = split_masks(meta)
        te = masks["test"]
        y = y_all[te]
        meta_te = [m for m, k in zip(meta, te) if k]
        groups = ped_clusters(meta_te)
        print(f"\n### protocol={protocol}: IDD-PeD test {te.sum():,} windows, "
              f"{int(y.sum()):,} positive ({100*y.mean():.1f} %), "
              f"{len(groups):,} pedestrian clusters")

        for coord in ("rescale", "raw"):
            X_te = to_pie_frame(X_raw[te], meta_te, mode=coord)

            for arm in PIE_ARMS:
                per_seed, prob_stack = [], []
                for seed in SEEDS:
                    rd = arm_run_dir(arm, seed)
                    model = build_model(engine, arm)
                    pf = common.prob_fn_from_run_dir(rd, model)

                    # tau* fitted on PIE VALIDATION only — never on IDD-PeD
                    tau = common.best_threshold(yva_p, pf(Xva_p))

                    p = pf(X_te)
                    prob_stack.append(p)
                    m05 = common.metrics_at(y, p, 0.5)
                    mts = common.metrics_at(y, p, tau)
                    per_seed.append(dict(seed=seed, tau_pie_val=tau,
                                         at_0p5=m05, at_tau=mts))

                P = np.mean(np.stack(prob_stack), axis=0)      # 5-seed probability ensemble
                taus = [d["tau_pie_val"] for d in per_seed]
                tau_ens = float(np.mean(taus))
                ens05 = common.metrics_at(y, P, 0.5)
                ensts = common.metrics_at(y, P, tau_ens)

                def agg(key, sub):
                    v = [d[sub][key] for d in per_seed]
                    return float(np.mean(v)), float(np.std(v, ddof=1))

                row = dict(
                    experiment="A_zero_shot", protocol=protocol, coord=coord, model=arm,
                    n_params=PIE_ARMS[arm]["n_params"],
                    n_test=int(te.sum()), n_pos=int(y.sum()), n_clusters=len(groups),
                    tau_pie_val_mean=tau_ens,
                    per_seed={k: agg(k, "at_0p5") for k in ("auc", "pr_auc")},
                    per_seed_at_0p5={k: agg(k, "at_0p5")
                                     for k in ("f1", "acc", "prec", "rec")},
                    per_seed_at_tau={k: agg(k, "at_tau")
                                     for k in ("f1", "acc", "prec", "rec")},
                    ensemble_at_0p5=ens05, ensemble_at_tau=ensts,
                    seeds=per_seed,
                )

                # cluster CIs on the ensemble (main protocol + rescale only — the headline)
                if protocol == "strict" and coord == "rescale":
                    auc_ci = cluster_ci(y, P, common.auc_fast, groups)
                    yb = y.astype(bool)
                    f1_ci = cluster_ci(y, (P >= tau_ens),
                                       lambda a, b: common.f1_from_preds(a.astype(bool), b),
                                       groups)
                    row["ensemble_auc_cluster_ci95"] = [float(auc_ci[0]), float(auc_ci[1])]
                    row["ensemble_f1_cluster_ci95"] = [float(f1_ci[0]), float(f1_ci[1])]
                    np.save(PCACHE / f"expA_{arm}_ens.npy", P)
                    np.save(PCACHE / "expA_y_test.npy", y)

                results.append(row)
                csv_rows.append(dict(
                    experiment="A_zero_shot", protocol=protocol, coord=coord, model=arm,
                    n_test=row["n_test"], n_pos=row["n_pos"],
                    auc_mean=row["per_seed"]["auc"][0], auc_std=row["per_seed"]["auc"][1],
                    pr_auc_mean=row["per_seed"]["pr_auc"][0],
                    pr_auc_std=row["per_seed"]["pr_auc"][1],
                    f1_at_0p5_mean=row["per_seed_at_0p5"]["f1"][0],
                    f1_at_0p5_std=row["per_seed_at_0p5"]["f1"][1],
                    acc_at_0p5_mean=row["per_seed_at_0p5"]["acc"][0],
                    f1_at_tau_mean=row["per_seed_at_tau"]["f1"][0],
                    f1_at_tau_std=row["per_seed_at_tau"]["f1"][1],
                    acc_at_tau_mean=row["per_seed_at_tau"]["acc"][0],
                    prec_at_tau_mean=row["per_seed_at_tau"]["prec"][0],
                    rec_at_tau_mean=row["per_seed_at_tau"]["rec"][0],
                    tau_pie_val=row["tau_pie_val_mean"],
                    ens_auc=ens05["auc"], ens_pr_auc=ens05["pr_auc"],
                    ens_f1_at_tau=ensts["f1"], ens_acc_at_tau=ensts["acc"],
                    auc_ci_lo=row.get("ensemble_auc_cluster_ci95", ["", ""])[0],
                    auc_ci_hi=row.get("ensemble_auc_cluster_ci95", ["", ""])[1],
                ))
                ci = row.get("ensemble_auc_cluster_ci95")
                ci_s = f"  clusterCI[{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""
                print(f"  [{coord:7s}] {arm:16s} AUC {row['per_seed']['auc'][0]:.3f}"
                      f"±{row['per_seed']['auc'][1]:.3f}  PR-AUC "
                      f"{row['per_seed']['pr_auc'][0]:.3f}  F1@0.5 "
                      f"{row['per_seed_at_0p5']['f1'][0]:.3f}  F1@τ* "
                      f"{row['per_seed_at_tau']['f1'][0]:.3f} (τ*={tau_ens:.3f}){ci_s}")

    OUT_JSON.write_text(json.dumps(dict(
        parity_gate_max_delta=parity_delta,
        pie_reference=dict(test_windows=len(yte_p), pos_rate=float(yte_p.mean())),
        results=results), indent=2))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
