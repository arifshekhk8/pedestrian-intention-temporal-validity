"""full_comparison_table.py — every model, both protocols, one table.

The numbers for the four neural families, the linear baselines and the tree
ensembles currently live in three separate result files, computed at three
different times. This script recomputes all of them in a single pass on both
protocols so the comparison is guaranteed to be like-for-like, and writes one
table.

Held constant across every row and both protocols:

  splits          train set01/02/04 | val set05/06 | test set03
  class weight    each protocol's own train-split neg/pos ratio
                  (event-anchored 1.682, phase-matched 1.5665)
  threshold       one tau per model, argmax F1 on POOLED VALIDATION
                  probabilities, frozen before the test split is touched
  seeds           42, 0, 1, 2, 3 for everything stochastic
  reporting       per-seed mean +- sd; deterministic models have no spread
  inference       pedestrian-clustered bootstrap, B = 10,000, Holm-corrected

The two protocols have different eligible pedestrians, so they are NOT paired.
Read the two columns as two separate experiments, not as a paired delta.

Usage:  python experiments/02_model_comparison/full_comparison_table.py
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

SEEDS = M.SEEDS
NEURAL = {k: v for k, v in M.FAMILIES.items() if k != "BiLSTM-h128"}

# pos_weight: use each protocol's DECLARED constant, not a recomputation.
# 1366/812 is 1.68226..., and the project (and the manuscript) declares 1.682.
# That rounding is not cosmetic here: the LR(80) validation-F1 surface has two
# near-equal optima at tau ~ 0.43 and tau ~ 0.67, and 1.682 vs 1.6823 flips
# which one wins, moving test F1 by 0.030. Reproduce the declared protocol.
PROTOCOLS = {
    "event-anchored": dict(data=ROOT / "data" / "pie_clean", runs="matched",
                           pos_weight=M.POS_WEIGHT),
    "phase-matched": dict(data=ROOT / "data" / "pie_phase_matched_trainonly",
                          runs="phase_matched_trainonly", pos_weight=None),
}

LINEAR = {
    "LR, box + speed (80)": lambda X: X,
    "LR, box only (64)": lambda X: X[:, :, :4],
    "LR, speed only (16)": lambda X: X[:, :, 4:5],
    "LR, last frame only (5)": lambda X: X[:, -1:, :],
}


def splits(seq_dir):
    X = np.load(seq_dir / "X.npy").astype(np.float32)
    y = np.load(seq_dir / "y.npy").astype(np.float32)
    meta = pickle.load(open(seq_dir / "meta.pkl", "rb"))
    sid = np.array([m["set_id"] for m in meta])
    tr, va, tt = (np.isin(sid, sorted(s)) for s in
                  (E.TRAIN_SETS, E.VAL_SETS, E.TEST_SETS))
    groups = np.array([f"{m['set_id']}/{m['video_id']}/{m['ped_id']}"
                       for m, k in zip(meta, tt) if k])
    return X[tr], y[tr], X[va], y[va], X[tt], y[tt], groups


def agg(per):
    return {k: (float(np.mean([q[k] for q in per])),
                float(np.std([q[k] for q in per], ddof=1)) if len(per) > 1 else 0.0)
            for k in per[0]}


def score(yte, probs, tau):
    return dict(auc=M.auc(yte, probs), pr_auc=M.pr_auc(yte, probs),
                f1=M.f1_at(yte, probs, tau))


def run(name, cfgd):
    Xtr, ytr, Xva, yva, Xte, yte, groups = splits(cfgd["data"])
    pw = cfgd["pos_weight"] or float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    print(f"\n### {name}: train {len(ytr)}  val {len(yva)}  test {len(yte)} "
          f"({len(set(groups))} pedestrians)   pos_weight {pw:.4f}")

    rows, probs = {}, {}

    # ---------------------------------------------------- neural families
    for label, (fam, cfg) in NEURAL.items():
        pv, pt = [], []
        for s in SEEDS:
            d = ROOT / "runs" / cfgd["runs"] / label.replace(" ", "_") / f"seed{s}"
            pv.append(M.probs_for(d, fam, cfg, Xva))
            pt.append(M.probs_for(d, fam, cfg, Xte))
        tau = M.best_threshold(yva, np.mean(pv, axis=0))
        rows[label] = dict(group="neural", tau=float(tau),
                           mean_sd=agg([score(yte, p, tau) for p in pt]))
        probs[label] = np.mean(pt, axis=0)

    # ------------------------------------------------------------- linear
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    for label, view in LINEAR.items():
        A, V, B = (view(Z).reshape(len(Z), -1) for Z in (Xtr, Xva, Xte))
        sc = StandardScaler().fit(A)
        lr = LogisticRegression(max_iter=5000,
                                class_weight={0: 1.0, 1: pw}).fit(sc.transform(A), ytr)
        pv = lr.predict_proba(sc.transform(V))[:, 1]
        pt = lr.predict_proba(sc.transform(B))[:, 1]
        tau = M.best_threshold(yva, pv)
        rows[label] = dict(group="linear", tau=float(tau), n_features=A.shape[1],
                           mean_sd=agg([score(yte, pt, tau)]))
        probs[label] = pt

    # -------------------------------------------------------------- trees
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    A, V, B = (Z.reshape(len(Z), -1) for Z in (Xtr, Xva, Xte))
    ctors = {"Decision tree": DecisionTreeClassifier,
             "Random forest": RandomForestClassifier,
             "Extra trees": ExtraTreesClassifier}
    for label, ctor in ctors.items():
        pv, pt = [], []
        for s in SEEDS:
            kw = dict(class_weight={0: 1.0, 1: pw}, random_state=s)
            if label != "Decision tree":
                kw["n_jobs"] = -1
            clf = ctor(**kw).fit(A, ytr)
            pv.append(clf.predict_proba(V)[:, 1])
            pt.append(clf.predict_proba(B)[:, 1])
        tau = M.best_threshold(yva, np.mean(pv, axis=0))
        rows[label] = dict(group="tree", tau=float(tau),
                           mean_sd=agg([score(yte, p, tau) for p in pt]))
        probs[label] = np.mean(pt, axis=0)

    # ------------------------------------------------------------ trivial
    prev = float(yte.mean())
    rows["Always positive"] = dict(group="trivial", tau=0.0, mean_sd=dict(
        auc=(0.5, 0.0), pr_auc=(prev, 0.0), f1=(2 * prev / (1 + prev), 0.0)))
    probs["Always positive"] = np.ones_like(yte, dtype=float)

    # --------------------------- everything against the linear reference
    ref = "LR, box + speed (80)"
    idx_by_g, picks = M.make_resamples(groups, 10000)
    tau_ref = rows[ref]["tau"]
    tests = {}
    for label in rows:
        if label == ref or rows[label]["group"] == "trivial":
            continue
        ta = rows[label]["tau"]
        for mn, sa, sb in (("AUC", M.auc, M.auc),
                           ("PR-AUC", M.pr_auc, M.pr_auc),
                           ("F1", (lambda y_, p_, t=ta: M.f1_at(y_, p_, t)),
                                  (lambda y_, p_, t=tau_ref: M.f1_at(y_, p_, t)))):
            o, lo, hi, p = M.cluster_delta(np.asarray(yte), probs[label], probs[ref],
                                           idx_by_g, picks, sa, sb)
            tests[f"{label} - {ref} [{mn}]"] = dict(delta=o, ci=[lo, hi], p=p)
    adjs, rejs = M.holm([v["p"] for v in tests.values()])
    for (k, v), a, r in zip(tests.items(), adjs, rejs):
        v["p_holm"], v["significant_holm"] = a, bool(r)

    return dict(pos_weight=pw, n_test=int(len(yte)),
                n_test_pedestrians=int(len(set(groups))),
                rows=rows, vs_linear=tests, prevalence=prev)


def cell(ms, k):
    m, s = ms[k]
    return f"{m:.4f} ± {s:.4f}" if s else f"{m:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="FULL_COMPARISON.md")
    ap.parse_args()

    res = {name: run(name, c) for name, c in PROTOCOLS.items()}
    A, B = res["event-anchored"], res["phase-matched"]
    order = (list(NEURAL) + list(LINEAR) +
             ["Decision tree", "Random forest", "Extra trees", "Always positive"])
    GROUP = {"neural": "Neural sequence models", "linear": "Linear baselines",
             "tree": "Tree ensembles", "trivial": "Trivial reference"}

    L = ["# Every model, both protocols", "",
         "Produced by `full_comparison_table.py`, which recomputes every row in one pass so the",
         "comparison is like-for-like. Per-seed mean ± sd over seeds 42, 0, 1, 2, 3; deterministic",
         "models have no spread. One τ per model from pooled validation probabilities, frozen before",
         "the test split is scored.", "",
         f"| | event-anchored | phase-matched |", "|---|---|---|",
         f"| test windows | {A['n_test']:,} | {B['n_test']:,} |",
         f"| test pedestrians | {A['n_test_pedestrians']} | {B['n_test_pedestrians']} |",
         f"| positive prevalence | {A['prevalence']:.1%} | {B['prevalence']:.1%} |",
         f"| class weight | {A['pos_weight']:.4f} | {B['pos_weight']:.4f} |", "",
         "The two protocols drop different pedestrians, so the columns are **not paired**. Read them",
         "as two experiments, not as a paired difference.", ""]

    for metric, title in (("auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("f1", "F1")):
        L += [f"## {title}", "",
              "| model | event-anchored | phase-matched | Δ |", "|---|---|---|---|"]
        last = None
        for label in order:
            g = A["rows"][label]["group"]
            if g != last:
                L.append(f"| **{GROUP[g]}** | | | |")
                last = g
            a = A["rows"][label]["mean_sd"]
            b = B["rows"][label]["mean_sd"]
            L.append(f"| {label} | {cell(a, metric)} | {cell(b, metric)} | "
                     f"{b[metric][0] - a[metric][0]:+.4f} |")
        L.append("")

    for name, r in (("event-anchored", A), ("phase-matched", B)):
        n = sum(v["significant_holm"] for v in r["vs_linear"].values())
        L += [f"## Against the linear reference — {name}", "",
              "Every model contrasted with LR box + speed (80). Pedestrian-clustered bootstrap,",
              f"B = 10,000, Holm across {len(r['vs_linear'])} tests. **{n} survive.**", "",
              "| contrast | Δ | 95% CI | p_Holm | |", "|---|---|---|---|---|"]
        for k, v in sorted(r["vs_linear"].items(), key=lambda t: t[1]["p_holm"]):
            L.append(f"| {k.replace(' - LR, box + speed (80)', '')} | {v['delta']:+.4f} | "
                     f"[{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] | {v['p_holm']:.4f} | "
                     f"{'**worse**' if v['significant_holm'] and v['delta'] < 0 else ('**better**' if v['significant_holm'] else 'n.s.')} |")
        L.append("")

    (HERE / "FULL_COMPARISON.md").write_text("\n".join(L))
    (HERE / "full_comparison_results.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {HERE / 'FULL_COMPARISON.md'}")
    print(f"wrote {HERE / 'full_comparison_results.json'}")


if __name__ == "__main__":
    main()
