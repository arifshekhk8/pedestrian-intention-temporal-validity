"""
trivial_baselines.py — how much of this task needs a neural network at all?

The literature on PIE/JAAD reports no genuinely trivial baseline. The weakest
comparison in PCPA (WACV 2021) is a deep CNN on a single frame; nobody reports
majority-class, speed-only, or a logistic regression on the raw window.

This script fills that gap on the leak-free protocol. Every baseline is fitted on
the train split only, standardised with train-split statistics only, thresholded
on validation only, and evaluated once on set03 — the same contract the neural
models obey. class_weight matches the neural pos_weight (1.682). No baseline
hyperparameter is tuned, which if anything handicaps them.

Usage:  python experiments/02_model_comparison/trivial_baselines.py
"""
import importlib.util, json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E = _load("engine", ROOT / "src" / "engine.py")
M = _load("matched", HERE / "matched_comparison.py")

BASELINES = {
    "LR ego-speed only (16 feats)":   lambda X: X[:, :, 4:5],
    "LR bbox only (64 feats)":        lambda X: X[:, :, :4],
    "LR bbox + ego-speed (80 feats)": lambda X: X,
    "LR last frame only (5 feats)":   lambda X: X[:, -1:, :],
}


def main():
    Xtr, ytr, Xva, yva, Xte, yte = E.load_splits()
    import pickle
    with open(E.SEQ_DIR / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    tm = [m for m in meta if m["set_id"] in E.TEST_SETS]
    groups = np.array([f"{m['set_id']}/{m['video_id']}/{m['ped_id']}" for m in tm])

    out, probs = {}, {}
    # majority class: always predict the positive class
    out["majority class (always positive)"] = dict(
        auc=0.5, pr_auc=float(yte.mean()),
        f1=float(2 * yte.sum() / (2 * yte.sum() + (1 - yte).sum())), tau=None)

    for name, sel in BASELINES.items():
        A, V, B = (sel(Z).reshape(len(Z), -1) for Z in (Xtr, Xva, Xte))
        sc = StandardScaler().fit(A)
        lr = LogisticRegression(max_iter=5000, class_weight={0: 1.0, 1: 1.682})
        lr.fit(sc.transform(A), ytr)
        pv = lr.predict_proba(sc.transform(V))[:, 1]
        pt = lr.predict_proba(sc.transform(B))[:, 1]
        tau = M.best_threshold(yva, pv)
        probs[name] = pt
        out[name] = dict(auc=M.auc(yte, pt), pr_auc=M.pr_auc(yte, pt),
                         f1=M.f1_at(yte, pt, tau), tau=float(tau))

    print(f"{'baseline':40s} {'AUC':>8s} {'PR-AUC':>8s} {'F1':>8s}")
    for k, v in out.items():
        print(f"{k:40s} {v['auc']:8.4f} {v['pr_auc']:8.4f} {v['f1']:8.4f}")

    # is the best linear baseline distinguishable from the best neural model?
    best_nn = ROOT / "runs" / "matched" / "Vanilla_RNN"
    cfg = M.FAMILIES["Vanilla RNN"][1]
    pt_nn = np.mean([M.probs_for(best_nn / f"seed{s}", "birnn", cfg, Xte)
                     for s in M.SEEDS], axis=0)
    idx_by_g, picks = M.make_resamples(groups, 10000)
    lin = probs["LR bbox + ego-speed (80 feats)"]
    tau_lin, tau_nn = out["LR bbox + ego-speed (80 feats)"]["tau"], 0.4970
    cmp = {}
    for nm, sa, sb in (("AUC", M.auc, M.auc), ("PR-AUC", M.pr_auc, M.pr_auc),
                       ("F1", (lambda y_, p_, t=tau_lin: M.f1_at(y_, p_, t)),
                              (lambda y_, p_, t=tau_nn: M.f1_at(y_, p_, t)))):
        o, lo, hi, p = M.cluster_delta(np.asarray(yte), lin, pt_nn, idx_by_g, picks, sa, sb)
        cmp[f"LR(80) - Vanilla RNN [{nm}]"] = dict(delta=o, ci=[lo, hi], p=p)
        print(f"  LR(80) - Vanilla RNN [{nm}]: {o:+.4f} CI [{lo:+.4f},{hi:+.4f}] p={p:.4f}")

    (HERE / "trivial_baselines_results.json").write_text(
        json.dumps(dict(baselines=out, vs_best_neural=cmp), indent=2))
    print(f"\nwrote {HERE/'trivial_baselines_results.json'}")


if __name__ == "__main__":
    main()
