"""tree_baselines.py — do tree ensembles beat the linear baseline on the clean data?

trivial_baselines.py established that logistic regression on the 80 raw window
values is statistically indistinguishable from the best of four tuned neural
families. That leaves an obvious gap: a linear model is only one kind of simple
model, and axis-aligned tree ensembles are the other obvious family to try on
tabular input of this shape.

This script adds decision tree, random forest and extra trees under EXACTLY the
contract the linear baselines use, so the comparison means something:

  data            data/pie_clean/  (4,906 leak-free, event-anchored windows)
  splits          train set01/02/04 | val set05/06 | test set03
  features        the same four flattened subsets LR uses
  class weight    1.682, the train split's own neg/pos ratio
  threshold       tau fitted on VALIDATION only, argmax F1, never on test
  test set        touched once, after tau is frozen
  search          NONE. Library defaults, exactly as the linear baselines get
                  no hyperparameter search. The point of a reference model is
                  that it is not tuned.

The tree models are stochastic, so each is fitted with the same five seeds the
neural families use and reported as mean +- sd. LR is deterministic and has no
spread.

Inference uses the pedestrian-clustered bootstrap (B = 10,000) with Holm
correction, resampling pedestrians rather than windows, because 2,094 test
windows come from only 541 pedestrians.

Usage:  python experiments/02_model_comparison/tree_baselines.py
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

SEEDS = M.SEEDS                  # [42, 0, 1, 2, 3]
POS_WEIGHT = M.POS_WEIGHT        # set from the chosen dataset's own train split

# the same four flattened views the linear baselines use
VIEWS = {
    "box + speed (80)": lambda X: X,
    "box only (64)": lambda X: X[:, :, :4],
    "speed only (16)": lambda X: X[:, :, 4:5],
    "last frame only (5)": lambda X: X[:, -1:, :],
}


def make_models(seed):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    cw = {0: 1.0, 1: POS_WEIGHT}
    return {
        "Decision tree": DecisionTreeClassifier(class_weight=cw, random_state=seed),
        "Random forest": RandomForestClassifier(class_weight=cw, random_state=seed,
                                                n_jobs=-1),
        "Extra trees": ExtraTreesClassifier(class_weight=cw, random_state=seed,
                                            n_jobs=-1),
    }


def fit_linear(Xtr, ytr, Xva, Xte, view):
    """The reference linear baseline, reproduced here so both arms are scored by
    the same code path on the same call."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    A, V, B = (view(Z).reshape(len(Z), -1) for Z in (Xtr, Xva, Xte))
    sc = StandardScaler().fit(A)
    lr = LogisticRegression(max_iter=5000, class_weight={0: 1.0, 1: POS_WEIGHT})
    lr.fit(sc.transform(A), ytr)
    return (lr.predict_proba(sc.transform(V))[:, 1],
            lr.predict_proba(sc.transform(B))[:, 1])


def main():
    global POS_WEIGHT
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--data", default=str(ROOT / "data" / "pie_clean"),
                    help="window set to run on; the phase-matched control is "
                         "data/pie_phase_matched_trainonly")
    ap.add_argument("--runs-subdir", default="matched",
                    help="where the neural reference checkpoints live")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    SEQ = Path(args.data)
    X = np.load(SEQ / "X.npy").astype(np.float32)
    y = np.load(SEQ / "y.npy").astype(np.float32)
    with open(SEQ / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    sid = np.array([m["set_id"] for m in meta])
    tr, va, tt = (np.isin(sid, sorted(s)) for s in
                  (E.TRAIN_SETS, E.VAL_SETS, E.TEST_SETS))
    Xtr, ytr, Xva, yva, Xte, yte = X[tr], y[tr], X[va], y[va], X[tt], y[tt]
    # the class weight is a property of the split, not a constant
    POS_WEIGHT = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    print(f"data {SEQ.name}   pos_weight {POS_WEIGHT:.4f}")
    test_meta = [m for m in meta if m["set_id"] in E.TEST_SETS]
    groups = np.array([f"{m['set_id']}/{m['video_id']}/{m['ped_id']}" for m in test_meta])
    print(f"train {len(ytr)}  val {len(yva)}  test {len(yte)}  "
          f"test pedestrians {len(set(groups))}\n")

    table, probs = {}, {}

    # ------------------------------------------------------ tree ensembles
    for vname, view in VIEWS.items():
        A, V, B = (view(Z).reshape(len(Z), -1) for Z in (Xtr, Xva, Xte))
        for mname in ("Decision tree", "Random forest", "Extra trees"):
            pv_l, pt_l = [], []
            for s in SEEDS:
                clf = make_models(s)[mname]
                clf.fit(A, ytr)
                pv_l.append(clf.predict_proba(V)[:, 1])
                pt_l.append(clf.predict_proba(B)[:, 1])
            # one tau per model, fitted on the POOLED validation probabilities,
            # exactly as the neural families get one tau from their pooled val
            tau = M.best_threshold(yva, np.mean(pv_l, axis=0))
            per = [dict(auc=M.auc(yte, p), pr_auc=M.pr_auc(yte, p),
                        f1=M.f1_at(yte, p, tau)) for p in pt_l]
            agg = {k: (float(np.mean([q[k] for q in per])),
                       float(np.std([q[k] for q in per], ddof=1))) for k in per[0]}
            key = f"{mname} [{vname}]"
            table[key] = dict(model=mname, view=vname, tau=float(tau),
                              n_features=A.shape[1], mean_sd=agg,
                              per_seed=per, stochastic=True)
            probs[key] = np.mean(pt_l, axis=0)     # 5-seed ensemble, as elsewhere

    # ------------------------------------------------- the linear reference
    for vname, view in VIEWS.items():
        pv, pt = fit_linear(Xtr, ytr, Xva, Xte, view)
        tau = M.best_threshold(yva, pv)
        key = f"Logistic regression [{vname}]"
        table[key] = dict(model="Logistic regression", view=vname, tau=float(tau),
                          n_features=view(Xtr).reshape(len(Xtr), -1).shape[1],
                          mean_sd=dict(auc=(M.auc(yte, pt), 0.0),
                                       pr_auc=(M.pr_auc(yte, pt), 0.0),
                                       f1=(M.f1_at(yte, pt, tau), 0.0)),
                          per_seed=None, stochastic=False)
        probs[key] = pt

    # ---------------------------------------- the best neural family, for scale
    fam, cfg = M.FAMILIES["Vanilla RNN"]
    pv_l, pt_l = [], []
    for s in SEEDS:
        d = ROOT / "runs" / args.runs_subdir / "Vanilla_RNN" / f"seed{s}"
        pv_l.append(M.probs_for(d, fam, cfg, Xva))
        pt_l.append(M.probs_for(d, fam, cfg, Xte))
    tau_nn = M.best_threshold(yva, np.mean(pv_l, axis=0))
    per = [dict(auc=M.auc(yte, p), pr_auc=M.pr_auc(yte, p),
                f1=M.f1_at(yte, p, tau_nn)) for p in pt_l]
    table["Vanilla RNN [box + speed (80)]"] = dict(
        model="Vanilla RNN", view="box + speed (80)", tau=float(tau_nn),
        n_features=80, stochastic=True, per_seed=per,
        mean_sd={k: (float(np.mean([q[k] for q in per])),
                     float(np.std([q[k] for q in per], ddof=1))) for k in per[0]})
    probs["Vanilla RNN [box + speed (80)]"] = np.mean(pt_l, axis=0)

    # ------------------------------------------------------------- reporting
    print(f"{'model':34s} {'feat':>5s} {'AUC':>16s} {'PR-AUC':>16s} {'F1':>16s}")
    for vname in VIEWS:
        print(f"-- {vname}")
        for mname in ("Decision tree", "Random forest", "Extra trees",
                      "Logistic regression", "Vanilla RNN"):
            key = f"{mname} [{vname}]"
            if key not in table:
                continue
            r = table[key]["mean_sd"]
            f = table[key]["n_features"]
            def cell(k):
                m, s = r[k]
                return f"{m:.4f}+-{s:.4f}" if s else f"{m:.4f}        "
            print(f"   {mname:31s} {f:5d} {cell('auc'):>16s} "
                  f"{cell('pr_auc'):>16s} {cell('f1'):>16s}")

    # ------------------------------------------- clustered bootstrap vs LR(80)
    # One test family: every model on the full 80-feature view, contrasted with
    # the linear reference. Holm across all of it.
    idx_by_g, picks = M.make_resamples(groups, args.bootstrap)
    ref = "Logistic regression [box + speed (80)]"
    tau_ref = table[ref]["tau"]
    contrasts = [f"{m} [box + speed (80)]" for m in
                 ("Decision tree", "Random forest", "Extra trees", "Vanilla RNN")]

    tests = {}
    for key in contrasts:
        ta = table[key]["tau"]
        for mn, sa, sb in (("AUC", M.auc, M.auc),
                           ("PR-AUC", M.pr_auc, M.pr_auc),
                           ("F1@tau", (lambda y_, p_, t=ta: M.f1_at(y_, p_, t)),
                                      (lambda y_, p_, t=tau_ref: M.f1_at(y_, p_, t)))):
            o, lo, hi, p = M.cluster_delta(np.asarray(yte), probs[key], probs[ref],
                                           idx_by_g, picks, sa, sb)
            tests[f"{table[key]['model']} - LR(80) [{mn}]"] = dict(
                delta=o, ci=[lo, hi], p=p)
    adj, rej = M.holm([v["p"] for v in tests.values()])
    for (k, v), a, r in zip(tests.items(), adj, rej):
        v["p_holm"], v["significant_holm"] = a, bool(r)

    print(f"\n=== vs the linear reference, pedestrian-clustered bootstrap "
          f"(B={args.bootstrap}), Holm across {len(tests)} tests ===")
    for k, v in tests.items():
        verdict = "DIFFERENT" if v["significant_holm"] else (
            "drops after Holm" if v["ci"][0] > 0 or v["ci"][1] < 0
            else "not distinguishable")
        print(f"  {k:38s} {v['delta']:+.4f} "
              f"[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]  p_holm={v['p_holm']:.4f}  {verdict}")

    # ------------------------------------------------------------------
    # A validation search for the trees only. The linear baseline never gets
    # one, so this is deliberately asymmetric IN THE TREES' FAVOUR: if they
    # still lose after a search their opponent never had, the gap is not a
    # tuning artefact. Selection is by validation AUC; test is untouched.
    # ------------------------------------------------------------------
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    A, V, B = (Z.reshape(len(Z), -1) for Z in (Xtr, Xva, Xte))
    CW = {0: 1.0, 1: POS_WEIGHT}
    grids = {
        "Decision tree": [dict(max_depth=d, min_samples_leaf=l)
                          for d in (3, 5, 8, 12, None) for l in (1, 5, 20)],
        "Random forest": [dict(n_estimators=n, max_depth=d, min_samples_leaf=l)
                          for n in (100, 300) for d in (None, 8, 16) for l in (1, 5)],
        "Extra trees": [dict(n_estimators=n, max_depth=d, min_samples_leaf=l)
                        for n in (100, 300) for d in (None, 8, 16) for l in (1, 5)],
    }
    ctors = {"Decision tree": DecisionTreeClassifier,
             "Random forest": RandomForestClassifier,
             "Extra trees": ExtraTreesClassifier}
    searched = {}
    print("\n=== validation search, trees only (selection by val AUC) ===")
    for mname, grid in grids.items():
        best, best_auc = None, -1.0
        for cfg in grid:
            kw = dict(cfg, class_weight=CW, random_state=42)
            if mname != "Decision tree":
                kw["n_jobs"] = -1
            a = M.auc(yva, ctors[mname](**kw).fit(A, ytr).predict_proba(V)[:, 1])
            if a > best_auc:
                best, best_auc = cfg, a
        pv_l, pt_l = [], []
        for s in SEEDS:
            kw = dict(best, class_weight=CW, random_state=s)
            if mname != "Decision tree":
                kw["n_jobs"] = -1
            clf = ctors[mname](**kw).fit(A, ytr)
            pv_l.append(clf.predict_proba(V)[:, 1])
            pt_l.append(clf.predict_proba(B)[:, 1])
        tau = M.best_threshold(yva, np.mean(pv_l, axis=0))
        per = [dict(auc=M.auc(yte, p), pr_auc=M.pr_auc(yte, p),
                    f1=M.f1_at(yte, p, tau)) for p in pt_l]
        agg = {k: (float(np.mean([q[k] for q in per])),
                   float(np.std([q[k] for q in per], ddof=1))) for k in per[0]}
        key = f"{mname} (searched) [box + speed (80)]"
        searched[key] = dict(model=mname, cfg=best, n_configs=len(grid),
                             val_auc=float(best_auc), tau=float(tau), mean_sd=agg)
        probs[key] = np.mean(pt_l, axis=0)
        table[key] = dict(searched[key], view="box + speed (80)", n_features=80,
                          stochastic=True, per_seed=per)
        print(f"  {mname:16s} {len(grid):3d} configs -> {best}")
        print(f"      test AUC {agg['auc'][0]:.4f}+-{agg['auc'][1]:.4f}   "
              f"PR-AUC {agg['pr_auc'][0]:.4f}   F1 {agg['f1'][0]:.4f}")

    tests_s = {}
    for key in searched:
        ta = table[key]["tau"]
        for mn, sa, sb in (("AUC", M.auc, M.auc),
                           ("PR-AUC", M.pr_auc, M.pr_auc),
                           ("F1@tau", (lambda y_, p_, t=ta: M.f1_at(y_, p_, t)),
                                      (lambda y_, p_, t=tau_ref: M.f1_at(y_, p_, t)))):
            o, lo, hi, p = M.cluster_delta(np.asarray(yte), probs[key], probs[ref],
                                           idx_by_g, picks, sa, sb)
            tests_s[f"{table[key]['model']} (searched) - LR(80) [{mn}]"] = dict(
                delta=o, ci=[lo, hi], p=p)
    adj_s, rej_s = M.holm([v["p"] for v in tests_s.values()])
    for (k, v), a, r in zip(tests_s.items(), adj_s, rej_s):
        v["p_holm"], v["significant_holm"] = a, bool(r)
    print(f"\n=== searched trees vs the UNsearched linear reference "
          f"(Holm across {len(tests_s)}) ===")
    for k, v in tests_s.items():
        verdict = "DIFFERENT" if v["significant_holm"] else "not distinguishable"
        print(f"  {k:46s} {v['delta']:+.4f} "
              f"[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]  p_holm={v['p_holm']:.4f}  {verdict}")

    out = dict(
        protocol=dict(data=str(SEQ), seeds=SEEDS, pos_weight=POS_WEIGHT,
                      threshold="one tau per model, pooled validation, argmax F1",
                      search="none - library defaults, matching the linear baselines",
                      bootstrap=f"pedestrian-clustered, shared replicates, B={args.bootstrap}",
                      correction=f"Holm-Bonferroni across {len(tests)} tests, alpha=0.05",
                      n_test=int(len(yte)), n_test_pedestrians=int(len(set(groups)))),
        results=table, vs_linear=tests, searched_trees=searched,
        vs_linear_searched=tests_s)
    name = args.out or ("tree_baselines_results.json" if SEQ.name == "pie_clean"
                        else f"tree_baselines_{SEQ.name}_results.json")
    (HERE / name).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / name}")


if __name__ == "__main__":
    main()
