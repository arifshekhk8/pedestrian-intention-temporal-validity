"""
phase_matched_stats.py — inference for the phase-matched control.

phase_matched_control.py trains the models and reports per-seed means. It stops
there, so its conclusions ("the ranking reorders", "the input streams swap") were
point estimates only. This script adds the same statistical machinery the main
comparison uses, so the controlled arm is tested to the same standard:

  - pedestrian-clustered bootstrap (B = 10,000), resampling pedestrians not windows
  - every contrast paired on the SAME bootstrap replicates
  - Holm-Bonferroni across the whole family of tests

It re-uses the checkpoints already in runs/phase_matched/ (20 of them: 4 families
x 5 seeds), so nothing is retrained.

Two families of tests are reported separately, each Holm-corrected within itself:
  A. between models, on the phase-matched data  (does the ranking survive?)
  B. standard vs phase-matched, per model       (is the drop real?)

Usage:  python experiments/02_model_comparison/phase_matched_stats.py
"""
import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E = _load("engine", ROOT / "src" / "engine.py")
M = _load("matched", HERE / "matched_comparison.py")

FAMILIES = {k: v for k, v in M.FAMILIES.items() if k != "BiLSTM-h128"}
PM_DIR = ROOT / "data" / "pie_phase_matched"


def splits(meta, X, y):
    sid = np.array([m["set_id"] for m in meta])
    tr = np.isin(sid, sorted(E.TRAIN_SETS))
    va = np.isin(sid, sorted(E.VAL_SETS))
    tt = np.isin(sid, sorted(E.TEST_SETS))
    return tr, va, tt


def lr_probs(Xtr, ytr, Xva, Xte, sel, pw):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    A, V, B = (sel(Z).reshape(len(Z), -1) for Z in (Xtr, Xva, Xte))
    sc = StandardScaler().fit(A)
    lr = LogisticRegression(max_iter=5000, class_weight={0: 1.0, 1: pw})
    lr.fit(sc.transform(A), ytr)
    return (lr.predict_proba(sc.transform(V))[:, 1],
            lr.predict_proba(sc.transform(B))[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()

    # ---------------------------------------------------------------- data
    Xp = np.load(PM_DIR / "X.npy")
    yp = np.load(PM_DIR / "y.npy")
    mp = pickle.load(open(PM_DIR / "meta.pkl", "rb"))
    tr, va, tt = splits(mp, Xp, yp)
    ytr, yva, yte = (yp[tr].astype(np.float32), yp[va].astype(np.float32),
                     yp[tt].astype(np.float32))
    pw = float((yp[tr] == 0).sum() / max((yp[tr] == 1).sum(), 1))
    groups = np.array([f"{m['set_id']}/{m['video_id']}/{m['ped_id']}"
                       for m, k in zip(mp, tt) if k])
    print(f"phase-matched test: {int(tt.sum())} windows / {len(set(groups))} pedestrians"
          f"   pos_weight {pw:.4f}")

    # ------------------------------------------------- ensemble probabilities
    P, TAU = {}, {}
    for label, (family, cfg) in FAMILIES.items():
        pv, pt = [], []
        for s in M.SEEDS:
            d = ROOT / "runs" / "phase_matched" / label.replace(" ", "_") / f"seed{s}"
            if not (d / "best.pt").exists():
                raise SystemExit(f"missing checkpoint: {d}")
            pv.append(M.probs_for(d, family, cfg, Xp[va]))
            pt.append(M.probs_for(d, family, cfg, Xp[tt]))
        pv, pt = np.mean(pv, axis=0), np.mean(pt, axis=0)
        TAU[label] = M.best_threshold(yva, pv)
        P[label] = pt
    for nm, sel in (("LR bbox + ego-speed (80)", lambda Z: Z),
                    ("LR ego-speed only (16)", lambda Z: Z[:, :, 4:5]),
                    ("LR bbox only (64)", lambda Z: Z[:, :, :4])):
        pv, pt = lr_probs(Xp[tr], ytr, Xp[va], Xp[tt], sel, pw)
        TAU[nm] = M.best_threshold(yva, pv)
        P[nm] = pt

    print(f"\n{'model':28s} {'AUC':>8s} {'PR-AUC':>8s} {'F1':>8s}  tau")
    for k in P:
        print(f"{k:28s} {M.auc(yte,P[k]):8.4f} {M.pr_auc(yte,P[k]):8.4f} "
              f"{M.f1_at(yte,P[k],TAU[k]):8.4f}  {TAU[k]:.3f}")

    idx_by_g, picks = M.make_resamples(groups, args.bootstrap)

    # -------------------------------------------- A. between models, matched data
    names = list(P)
    tests_a = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            for mname, sa, sb in (
                    ("AUC", M.auc, M.auc),
                    ("PR-AUC", M.pr_auc, M.pr_auc),
                    ("F1@tau", (lambda y_, p_, t=TAU[a]: M.f1_at(y_, p_, t)),
                               (lambda y_, p_, t=TAU[b]: M.f1_at(y_, p_, t)))):
                o, lo, hi, p = M.cluster_delta(yte, P[a], P[b], idx_by_g, picks, sa, sb)
                tests_a[f"{a} - {b} [{mname}]"] = dict(delta=o, ci=[lo, hi], p=p)
    adj, rej = M.holm([v["p"] for v in tests_a.values()])
    for (k, v), pa, r in zip(tests_a.items(), adj, rej):
        v["p_holm"], v["significant_holm"] = pa, bool(r)
        v["significant_uncorrected"] = v["p"] <= 0.05

    print(f"\n=== A. between models on phase-matched data "
          f"({sum(rej)}/{len(tests_a)} survive Holm) ===")
    for k, v in sorted(tests_a.items(), key=lambda t: t[1]["p"]):
        if v["significant_holm"]:
            print(f"  {k:52s} {v['delta']:+.4f}  p_holm={v['p_holm']:.4f}")

    # ------------------------------- B. standard vs phase-matched, same model
    # The two protocols have DIFFERENT test sets (phase-matching drops 115 negative
    # pedestrians), so a paired bootstrap is not available. Both arms are instead
    # resampled independently over their own pedestrians and the difference of the
    # two bootstrap distributions is taken -- valid for independent samples, just
    # less powerful than pairing.
    Xc = np.load(ROOT / "data" / "pie_clean" / "X.npy")
    yc = np.load(ROOT / "data" / "pie_clean" / "y.npy")
    mc = pickle.load(open(ROOT / "data" / "pie_clean" / "meta.pkl", "rb"))
    ctr, cva, ctt = splits(mc, Xc, yc)
    yc_va, yc_te = yc[cva].astype(np.float32), yc[ctt].astype(np.float32)
    gc = np.array([f"{m['set_id']}/{m['video_id']}/{m['ped_id']}"
                   for m, k in zip(mc, ctt) if k])
    print(f"\nstandard test: {int(ctt.sum())} windows / {len(set(gc))} pedestrians")

    Pc, TAUc = {}, {}
    for label, (family, cfg) in FAMILIES.items():
        pv, pt = [], []
        for s in M.SEEDS:
            d = ROOT / "runs" / "matched" / label.replace(" ", "_") / f"seed{s}"
            if not (d / "best.pt").exists():
                d = ROOT / "runs" / "matched" / family / f"seed{s}"
            pv.append(M.probs_for(d, family, cfg, Xc[cva]))
            pt.append(M.probs_for(d, family, cfg, Xc[ctt]))
        pv, pt = np.mean(pv, axis=0), np.mean(pt, axis=0)
        TAUc[label], Pc[label] = M.best_threshold(yc_va, pv), pt
    for nm, sel in (("LR bbox + ego-speed (80)", lambda Z: Z),
                    ("LR ego-speed only (16)", lambda Z: Z[:, :, 4:5]),
                    ("LR bbox only (64)", lambda Z: Z[:, :, :4])):
        pv, pt = lr_probs(Xc[ctr], yc[ctr].astype(np.float32), Xc[cva], Xc[ctt],
                          sel, M.POS_WEIGHT)
        TAUc[nm], Pc[nm] = M.best_threshold(yc_va, pv), pt

    ig_c, pk_c = M.make_resamples(gc, args.bootstrap, seed=7)

    def boot(y, p, idx_by_g, picks, stat):
        d = np.empty(len(picks))
        for b in range(len(picks)):
            idx = np.concatenate([idx_by_g[i] for i in picks[b]])
            yy = y[idx]
            s = yy.sum()
            d[b] = np.nan if (s == 0 or s == len(yy)) else stat(yy, p[idx])
        return d[~np.isnan(d)]

    tests_b = {}
    for k in P:
        for mname, st_m, st_c in (
                ("AUC", M.auc, M.auc),
                ("PR-AUC", M.pr_auc, M.pr_auc),
                ("F1@tau", (lambda y_, p_, t=TAU[k]: M.f1_at(y_, p_, t)),
                           (lambda y_, p_, t=TAUc[k]: M.f1_at(y_, p_, t)))):
            bm = boot(yte, P[k], idx_by_g, picks, st_m)
            bc = boot(yc_te, Pc[k], ig_c, pk_c, st_c)
            n = min(len(bm), len(bc))
            d = bm[:n] - bc[:n]
            p_lo = (1 + np.sum(d <= 0)) / (n + 1)
            p_hi = (1 + np.sum(d >= 0)) / (n + 1)
            tests_b[f"{k} [{mname}]"] = dict(
                standard=float(st_c(yc_te, Pc[k])), matched=float(st_m(yte, P[k])),
                delta=float(st_m(yte, P[k]) - st_c(yc_te, Pc[k])),
                ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                p=float(min(1.0, 2 * min(p_lo, p_hi))))
    adjb, rejb = M.holm([v["p"] for v in tests_b.values()])
    for (k, v), pa, r in zip(tests_b.items(), adjb, rejb):
        v["p_holm"], v["significant_holm"] = pa, bool(r)

    print(f"\n=== B. standard vs phase-matched, unpaired bootstrap "
          f"({sum(rejb)}/{len(tests_b)} survive Holm) ===")
    for k, v in tests_b.items():
        flag = "*" if v["significant_holm"] else " "
        print(f" {flag}{k:34s} {v['standard']:.4f} -> {v['matched']:.4f}  "
              f"{v['delta']:+.4f}  CI[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]  "
              f"p_holm={v['p_holm']:.4f}")

    note = ("Arm B is unpaired: phase-matching drops 115 negative pedestrians, so the two "
            "protocols have different test sets (1,873 windows / 476 pedestrians vs 2,094 "
            "/ 541). Each arm is bootstrapped over its own pedestrians and the difference "
            "of the two independent distributions is reported. This is valid but less "
            "powerful than the paired procedure used in arm A.")

    out = dict(
        protocol=dict(bootstrap=args.bootstrap,
                      correction="Holm-Bonferroni within each arm separately",
                      seeds=M.SEEDS, device=M.DEVICE, pos_weight=pw,
                      n_test=int(tt.sum()), n_test_pedestrians=len(set(groups)),
                      n_test_standard=int(ctt.sum()),
                      n_test_pedestrians_standard=len(set(gc)),
                      threshold="one tau per model per protocol, fitted on that "
                                "protocol's validation split"),
        point=({k: dict(tau=float(TAU[k]), auc=M.auc(yte, P[k]),
                        pr_auc=M.pr_auc(yte, P[k]), f1=M.f1_at(yte, P[k], TAU[k]))
                for k in P}),
        between_models=tests_a,
        standard_vs_matched=tests_b,
        note_on_cross_protocol=note)
    (HERE / "phase_matched_stats.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE/'phase_matched_stats.json'}")


if __name__ == "__main__":
    main()
