"""06_independent_replication.py — EXPERIMENT B: independent replication on IDD-PeD.

The same four model families are trained from scratch on IDD-PeD alone, under the identical
frozen protocol, and evaluated on IDD-PeD's official test sets.

What is held IDENTICAL to the PIE study (Phase 8 of the brief):
  * architecture — the very same classes, loaded read-only from the existing project;
  * the exact PIE headline ("-F1") hyperparameter configs, NOT re-tuned on IDD-PeD;
  * the training loop — `train_run()` from `12_unified_engine.py`, called unmodified;
  * feature semantics [x1, y1, x2, y2, ego_speed], obs_len 16, TTE in [30, 60];
  * loss BCEWithLogits, optimizer/schedule from the cfg, batch 32, <=100 epochs,
    patience 15, ReduceLROnPlateau, train-only z-score, F1-first selection;
  * seeds [42, 0, 1, 2, 3]; test touched exactly once, by this script.

What is ADAPTED, and only because the dataset differs:
  * the data source and its splits;
  * pos_weight = IDD-PeD train neg/pos (a dataset property, not a tuned knob);
  * normalization statistics are fitted on IDD-PeD TRAIN only (this is a legitimate
    independent replication — contrast Experiment A, which must use PIE's statistics).

Also runs the **bbox-only (4-D) ablation**, dropping the ego-speed channel. This is not a
new modality — it is the direct replication of the PIE study's published bbox-only ablation,
and it is the test of the paper's ego-speed-dominance claim on a second dataset.

The unified engine hardcodes `input_dim=5` inside its four builder wrappers and asserts the
PIE array shape in `load_splits()`. Neither file is edited: the engine is importlib-loaded
read-only and its in-memory MODEL_REGISTRY is monkey-patched for the 4-D arm, exactly as
`journal_prep/cross_dataset_validation/03_jaad_fourfamily_engine.py` does. `train_run()`
itself is called verbatim, and `load_splits()` is never called.

Writes  checkpoints/<protocol>/<variant>/<family>/seed<N>/   (best.pt, norm stats, final.json)
        results/expB_independent.json / .csv
        results/expB_probs/*.npy

Run from the repo root (~40-80 min on the M4 CPU):
    python idd_ped_crossdataset/scripts/06_independent_replication.py
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FOLDER = HERE.parent
sys.path.insert(0, str(FOLDER / "src"))
from pie_bridge import (PIE_ARMS, SEEDS, load_common, load_engine,  # noqa: E402
                        load_iddped, ped_clusters, split_masks)

SEQ = {"strict": FOLDER / "data" / "sequences_iddped_clean",
       "cp_anchor": FOLDER / "data" / "sequences_iddped_clean_cp_anchor"}
CKPT = FOLDER / "checkpoints"
OUT_JSON = FOLDER / "results" / "expB_independent.json"
OUT_CSV = FOLDER / "results" / "expB_independent.csv"
PCACHE = FOLDER / "results" / "expB_probs"

B_BOOT, RNG_SEED = 10_000, 42


def patch_registry(engine, input_dim):
    """Rebuild MODEL_REGISTRY in memory for a given input_dim. Does not touch disk."""
    def bilstm(cfg):
        return engine.BiLSTM(input_dim=input_dim, hidden_dim=cfg["hidden"],
                             num_layers=cfg["num_layers"], dropout=cfg["dropout"])

    def transformer(cfg):
        return engine.TransformerIntentPredictor(
            input_dim=input_dim, d_model=cfg["d_model"], nhead=cfg.get("nhead", 4),
            num_layers=cfg["num_layers"], dim_ff=cfg["dim_ff"], dropout=cfg["dropout"],
            pool=cfg["pool"], pos=cfg["pos"])

    def gru(cfg):
        return engine.RecurrentIntentPredictor("gru", input_dim=input_dim,
                                               hidden_dim=cfg["hidden"],
                                               num_layers=cfg["num_layers"],
                                               dropout=cfg["dropout"])

    def birnn(cfg):
        return engine.RecurrentIntentPredictor("rnn", input_dim=input_dim,
                                               hidden_dim=cfg["hidden"],
                                               num_layers=cfg["num_layers"],
                                               dropout=cfg["dropout"])

    engine.MODEL_REGISTRY.update(bilstm=bilstm, transformer=transformer,
                                 gru=gru, birnn=birnn)


def cluster_ci(y, v, stat, groups, B=B_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    k = len(groups)
    vals = np.empty(B)
    for b in range(B):
        idx = np.concatenate([groups[i] for i in rng.integers(0, k, k)])
        vals[b] = stat(y[idx], v[idx])
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


@torch.no_grad()
def test_probs(model, Xte_n, device):
    """The one place IDD-PeD test windows are touched, once per run."""
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(Xte_n).to(device)).squeeze(-1)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=["strict", "cp_anchor"])
    ap.add_argument("--variants", nargs="+", default=["5d", "4d_bbox_only"])
    ap.add_argument("--families", nargs="+", default=list(PIE_ARMS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                    help="cpu = bit-reproducible for recurrent families (Issue-12 finding); "
                         "mps = faster but nn.LSTM training on MPS is process-history-dependent, "
                         "so runs are NOT exactly reproducible. Recorded in every final.json.")
    args = ap.parse_args()

    common = load_common()
    engine = load_engine()
    device = torch.device(args.device)
    print(f"device = {device}")

    results, csv_rows = [], []
    PCACHE.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    for protocol in args.protocols:
        X_all, y_all, meta = load_iddped(SEQ[protocol])
        masks = split_masks(meta)

        for variant in args.variants:
            dim = 5 if variant == "5d" else 4
            X = X_all if dim == 5 else X_all[:, :, :4]
            patch_registry(engine, dim)

            Xtr, ytr = X[masks["train"]], y_all[masks["train"]]
            Xva, yva = X[masks["val"]], y_all[masks["val"]]
            Xte, yte = X[masks["test"]], y_all[masks["test"]]
            meta_te = [m for m, k in zip(meta, masks["test"]) if k]
            groups = ped_clusters(meta_te)

            pos_weight = float((ytr == 0).sum()) / max(float(ytr.sum()), 1.0)
            print(f"\n### protocol={protocol} variant={variant} (input_dim={dim})")
            print(f"    train {len(ytr):,} (pos {int(ytr.sum())}) | val {len(yva):,} "
                  f"(pos {int(yva.sum())}) | test {len(yte):,} (pos {int(yte.sum())}) | "
                  f"pos_weight {pos_weight:.3f}")

            for arm in args.families:
                spec = PIE_ARMS[arm]
                family, cfg = spec["family"], spec["cfg"]
                per_seed, prob_stack, taus = [], [], []

                for seed in args.seeds:
                    run_dir = CKPT / protocol / variant / arm / f"seed{seed}"
                    t0 = time.time()
                    res = engine.train_run(family, cfg, seed, device,
                                           (Xtr, ytr, Xva, yva, Xte, yte),
                                           pos_weight=pos_weight, select="f1",
                                           out_dir=str(run_dir))

                    # reload the saved checkpoint + IDD-PeD train norm stats, touch test ONCE
                    ck = torch.load(run_dir / "best.pt", map_location="cpu",
                                    weights_only=False)
                    mean = np.load(run_dir / "norm_mean.npy")
                    std = np.load(run_dir / "norm_std.npy")
                    model = engine.MODEL_REGISTRY[family](cfg).to(device)
                    model.load_state_dict(ck["model"])

                    # tau* on IDD-PeD VALIDATION only (legitimate for a replication)
                    pv = test_probs(model, ((Xva - mean) / std).astype(np.float32), device)
                    tau = common.best_threshold(yva, pv)
                    taus.append(tau)

                    p = test_probs(model, ((Xte - mean) / std).astype(np.float32), device)
                    prob_stack.append(p)
                    m05 = common.metrics_at(yte, p, 0.5)
                    mts = common.metrics_at(yte, p, tau)
                    (run_dir / "test_metrics.json").write_text(json.dumps(
                        dict(tau=tau, at_0p5=m05, at_tau=mts), indent=2))
                    per_seed.append(dict(seed=seed, tau_val=tau, at_0p5=m05, at_tau=mts,
                                         val=res["val"], n_params=res["n_params"],
                                         best_epoch=res["best_epoch"],
                                         seconds=round(time.time() - t0, 1)))

                P = np.mean(np.stack(prob_stack), axis=0)
                tau_ens = float(np.mean(taus))
                ens05 = common.metrics_at(yte, P, 0.5)
                ensts = common.metrics_at(yte, P, tau_ens)

                def agg(key, sub):
                    v = [d[sub][key] for d in per_seed]
                    return float(np.mean(v)), float(np.std(v, ddof=1))

                row = dict(experiment="B_independent", protocol=protocol, variant=variant,
                           model=arm, n_params=per_seed[0]["n_params"],
                           pos_weight=pos_weight,
                           n_train=len(ytr), n_val=len(yva), n_test=len(yte),
                           n_test_pos=int(yte.sum()), n_clusters=len(groups),
                           tau_val_mean=tau_ens,
                           per_seed={k: agg(k, "at_0p5") for k in ("auc", "pr_auc")},
                           per_seed_at_0p5={k: agg(k, "at_0p5")
                                            for k in ("f1", "acc", "prec", "rec")},
                           per_seed_at_tau={k: agg(k, "at_tau")
                                            for k in ("f1", "acc", "prec", "rec")},
                           ensemble_at_0p5=ens05, ensemble_at_tau=ensts,
                           seeds=per_seed)

                if protocol == "strict":
                    auc_ci = cluster_ci(yte, P, common.auc_fast, groups)
                    f1_ci = cluster_ci(yte, (P >= tau_ens),
                                       lambda a, b: common.f1_from_preds(a.astype(bool), b),
                                       groups)
                    row["ensemble_auc_cluster_ci95"] = [float(auc_ci[0]), float(auc_ci[1])]
                    row["ensemble_f1_cluster_ci95"] = [float(f1_ci[0]), float(f1_ci[1])]
                    np.save(PCACHE / f"expB_{variant}_{arm}_ens.npy", P)
                    np.save(PCACHE / f"expB_{variant}_y_test.npy", yte)

                results.append(row)
                csv_rows.append(dict(
                    experiment="B_independent", protocol=protocol, variant=variant,
                    model=arm, n_params=row["n_params"], pos_weight=round(pos_weight, 3),
                    n_test=row["n_test"], n_test_pos=row["n_test_pos"],
                    auc_mean=row["per_seed"]["auc"][0], auc_std=row["per_seed"]["auc"][1],
                    pr_auc_mean=row["per_seed"]["pr_auc"][0],
                    pr_auc_std=row["per_seed"]["pr_auc"][1],
                    f1_at_0p5_mean=row["per_seed_at_0p5"]["f1"][0],
                    f1_at_tau_mean=row["per_seed_at_tau"]["f1"][0],
                    f1_at_tau_std=row["per_seed_at_tau"]["f1"][1],
                    acc_at_tau_mean=row["per_seed_at_tau"]["acc"][0],
                    prec_at_tau_mean=row["per_seed_at_tau"]["prec"][0],
                    rec_at_tau_mean=row["per_seed_at_tau"]["rec"][0],
                    tau_val=tau_ens, ens_auc=ens05["auc"], ens_pr_auc=ens05["pr_auc"],
                    ens_f1_at_tau=ensts["f1"], ens_acc_at_tau=ensts["acc"],
                    auc_ci_lo=row.get("ensemble_auc_cluster_ci95", ["", ""])[0],
                    auc_ci_hi=row.get("ensemble_auc_cluster_ci95", ["", ""])[1],
                ))
                ci = row.get("ensemble_auc_cluster_ci95")
                ci_s = f"  clusterCI[{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""
                print(f"  {arm:16s} AUC {row['per_seed']['auc'][0]:.3f}"
                      f"±{row['per_seed']['auc'][1]:.3f}  PR-AUC "
                      f"{row['per_seed']['pr_auc'][0]:.3f}  F1@τ* "
                      f"{row['per_seed_at_tau']['f1'][0]:.3f}"
                      f"±{row['per_seed_at_tau']['f1'][1]:.3f}  "
                      f"({sum(d['seconds'] for d in per_seed):.0f}s){ci_s}")

                OUT_JSON.write_text(json.dumps(results, indent=2))

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    print(f"\nTotal wall clock: {(time.time()-t_start)/60:.1f} min")
    print(f"Wrote {OUT_JSON}\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
