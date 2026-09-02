"""
matched_comparison.py — the four-family comparison with everything held constant.

WHY THIS SCRIPT EXISTS
----------------------
The original per-family studies each answered their own question, so their published
arms differ along four axes at once: hyperparameter-search budget, model capacity,
checkpoint-selection rule (best-val-F1 vs best-val-AUC), class weight, and training
device. A cross-family difference measured across those arms cannot be attributed to
the architecture. This script rebuilds the comparison with a single provenance.

HELD CONSTANT (every row)
  data            data/pie_clean/  (4,906 leak-free windows)
  splits          train set01/02/04 | val set05/06 | test set03
  engine          src/engine.py, one training loop
  device          cpu   (bit-reproducible; MPS recurrent training is not)
  seeds           42, 0, 1, 2, 3
  pos_weight      1.682  (the train split's own neg/pos ratio)
  checkpoint rule best validation AUC          <- same for all four families
  threshold       ONE tau per family, fitted on POOLED validation probabilities
  test set        touched once, after all selection is frozen

VARIES BY DESIGN (this is correct, not a confound)
  the architecture, and each family's own val-selected hyperparameters.
  Forcing identical hyperparameters across families would handicap three of them;
  the controlled quantity is the search budget and the protocol, not the config.
  Capacity is NOT matched -- it is reported per row (560k..2.24M) and must be
  disclosed. See Greff et al. (2017) on why fixing parameter count biases such
  comparisons.

CONFIGS -- each family's own validation-AUC-selected winner, from its own search:
  bilstm       lr1e-04 do0.2 h256 nl2   (journal_prep issue-8 grid winner)
  transformer  d128 nhead4 L4 ff512     (78-config staged search winner)
  gru          lr1e-03 do0.2 h256 nl2   (gru phase-3 AUC winner)
  birnn        lr1e-04 do0.2 h256 nl2   (rnn phase-3 winner; F1 and AUC agree)

Usage:  python experiments/02_model_comparison/matched_comparison.py [--runs_dir DIR]
"""
import argparse, json, pickle, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E = _load("engine", ROOT / "src" / "engine.py")

SEEDS = [42, 0, 1, 2, 3]
POS_WEIGHT = 1.682
SELECT = "auc"
DEVICE = "cpu"

FAMILIES = {
    "BiLSTM":       ("bilstm",      dict(lr=1e-4, dropout=0.2, hidden=256, num_layers=2)),
    "Transformer":  ("transformer", dict(E.TRANSFORMER_SEARCHED_CFG)),
    "GRU":          ("gru",         dict(lr=1e-3, dropout=0.2, hidden=256, num_layers=2)),
    "Vanilla RNN":  ("birnn",       dict(lr=1e-4, dropout=0.2, hidden=256, num_layers=2)),
    # Guard row: the BiLSTM's own grid search found its h256 winner statistically
    # indistinguishable from this hand-set h128 baseline (issue-8: d +0.0006, p=0.914).
    # Included so "BiLSTM is worst" cannot be an artefact of one config choice.
    "BiLSTM-h128":  ("bilstm",      dict(lr=1e-3, dropout=0.3, hidden=128, num_layers=2)),
}


# ---------------------------------------------------------------- metrics
def auc(y, s):
    y = np.asarray(y).astype(bool)
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    sv = np.asarray(s)[order]
    r[order] = np.arange(1, len(s) + 1, dtype=float)
    # average ranks over ties
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return (r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def pr_auc(y, s):
    y = np.asarray(y).astype(bool)
    order = np.argsort(-np.asarray(s), kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(int(y.sum()), 1)
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def f1_at(y, s, thr):
    y = np.asarray(y).astype(bool)
    p = np.asarray(s) >= thr
    tp = int((y & p).sum()); fp = int((~y & p).sum()); fn = int((y & ~p).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def acc_at(y, s, thr):
    y = np.asarray(y).astype(bool)
    return float(((np.asarray(s) >= thr) == y).mean())


def best_threshold(y, s, lo=0.05, hi=0.95):
    """argmax F1 over unique probabilities, bounded. VALIDATION ONLY."""
    cand = np.unique(np.asarray(s))
    cand = cand[(cand >= lo) & (cand <= hi)]
    if len(cand) == 0:
        return 0.5
    scores = [(f1_at(y, s, t), -abs(t - 0.5), t) for t in cand]
    return float(max(scores)[2])


# ---------------------------------------------------------------- inference
def probs_for(run_dir, family, cfg, X):
    ck = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=False)
    model = E.MODEL_REGISTRY[family](cfg)
    model.load_state_dict(ck["model"] if "model" in ck else ck["state_dict"])
    model.eval()
    mu = np.load(Path(run_dir) / "norm_mean.npy")
    sd = np.load(Path(run_dir) / "norm_std.npy")
    Z = (X - mu) / sd
    with torch.no_grad():
        out = model(torch.tensor(Z, dtype=torch.float32)).squeeze(-1)
        return torch.sigmoid(out).numpy()


# ---------------------------------------------------------------- clustered bootstrap
def cluster_bootstrap_delta(y, pa, pb, groups, stat, B=10000, seed=42):
    """Paired bootstrap resampling PEDESTRIANS, not windows."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by_g = [np.where(inv == i)[0] for i in range(len(uniq))]
    obs = stat(y, pa) - stat(y, pb)
    deltas = np.empty(B)
    G = len(uniq)
    for b in range(B):
        pick = rng.integers(0, G, G)
        idx = np.concatenate([idx_by_g[i] for i in pick])
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            deltas[b] = np.nan; continue
        deltas[b] = stat(yy, pa[idx]) - stat(yy, pb[idx])
    d = deltas[~np.isnan(deltas)]
    return float(obs), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def cluster_bootstrap_delta_2(y, pa, pb, groups, stat_a, stat_b, B=10000, seed=42):
    """Paired pedestrian-clustered bootstrap where each arm uses its OWN threshold."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by_g = [np.where(inv == i)[0] for i in range(len(uniq))]
    obs = stat_a(y, pa) - stat_b(y, pb)
    deltas = np.empty(B); G = len(uniq)
    for b in range(B):
        pick = rng.integers(0, G, G)
        idx = np.concatenate([idx_by_g[i] for i in pick])
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            deltas[b] = np.nan; continue
        deltas[b] = stat_a(yy, pa[idx]) - stat_b(yy, pb[idx])
    d = deltas[~np.isnan(deltas)]
    return float(obs), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default=str(ROOT / "runs" / "matched"))
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()
    runs = Path(args.runs_dir); runs.mkdir(parents=True, exist_ok=True)

    data = E.load_splits()
    Xtr, ytr, Xva, yva, Xte, yte = data
    with open(E.SEQ_DIR / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    test_meta = [m for m in meta if m["set_id"] in E.TEST_SETS]
    assert len(test_meta) == len(yte), (len(test_meta), len(yte))
    groups = np.array([f"{m['set_id']}/{m['video_id']}/{m['ped_id']}" for m in test_meta])
    print(f"val n={len(yva)}  test n={len(yte)}  test pedestrians={len(set(groups))}")

    table, probs_store = {}, {}
    for label, (family, cfg) in FAMILIES.items():
        t0 = time.time()
        pv_list, pt_list, nparams = [], [], None
        for s in SEEDS:
            d = runs / label.replace(" ", "_") / f"seed{s}"
            if not (d / "best.pt").exists():
                d.mkdir(parents=True, exist_ok=True)
                r = E.train_run(family, cfg, s, DEVICE, data,
                                pos_weight=POS_WEIGHT, select=SELECT, out_dir=d)
                nparams = r["n_params"]
            else:
                nparams = json.loads((d / "final.json").read_text())["n_params"]
            pv_list.append(probs_for(d, family, cfg, Xva))
            pt_list.append(probs_for(d, family, cfg, Xte))
        pv = np.mean(pv_list, axis=0)          # pooled val probs -> ONE tau per family
        tau = best_threshold(yva, pv)
        per_seed = [dict(f1_tau=f1_at(yte, p, tau), f1_05=f1_at(yte, p, 0.5),
                         acc=acc_at(yte, p, tau), auc=auc(yte, p), pr_auc=pr_auc(yte, p))
                    for p in pt_list]
        pt_ens = np.mean(pt_list, axis=0)
        agg = {k: (float(np.mean([d[k] for d in per_seed])),
                   float(np.std([d[k] for d in per_seed], ddof=1))) for k in per_seed[0]}
        table[label] = dict(family=family, cfg=cfg, n_params=nparams, tau=float(tau),
                            per_seed=per_seed, mean_sd=agg,
                            ens=dict(f1_tau=f1_at(yte, pt_ens, tau), auc=auc(yte, pt_ens),
                                     pr_auc=pr_auc(yte, pt_ens)),
                            seconds=round(time.time() - t0, 1))
        probs_store[label] = pt_ens
        print(f"{label:13s} params={nparams:>9,}  tau={tau:.4f}  "
              f"F1@tau={agg['f1_tau'][0]:.4f}+-{agg['f1_tau'][1]:.4f}  "
              f"AUC={agg['auc'][0]:.4f}+-{agg['auc'][1]:.4f}  "
              f"PR-AUC={agg['pr_auc'][0]:.4f}  ({table[label]['seconds']}s)")

    # pairwise pedestrian-clustered bootstrap on the seed-averaged probabilities
    labels = list(FAMILIES)
    pairs = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            f1a = (lambda yy, pp, t=table[a]["tau"]: f1_at(yy, pp, t))
            f1b = (lambda yy, pp, t=table[b]["tau"]: f1_at(yy, pp, t))
            for name, stat in (("AUC", auc), ("PR-AUC", pr_auc), ("F1@tau", None)):
                if name == "F1@tau":
                    o, lo, hi = cluster_bootstrap_delta_2(np.asarray(yte), probs_store[a],
                                                          probs_store[b], groups, f1a, f1b,
                                                          B=args.bootstrap)
                else:
                    o, lo, hi = cluster_bootstrap_delta(np.asarray(yte), probs_store[a],
                                                        probs_store[b], groups, stat,
                                                        B=args.bootstrap)
                pairs[f"{a} - {b} [{name}]"] = dict(delta=o, ci=[lo, hi],
                                                    excludes_zero=bool(lo > 0 or hi < 0))
                print(f"  {a} - {b} [{name}]: {o:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
                      f"  {'DIFFERENT' if (lo > 0 or hi < 0) else 'not distinguishable'}")

    out = dict(protocol=dict(seeds=SEEDS, pos_weight=POS_WEIGHT, select=SELECT,
                             device=DEVICE, threshold="one tau per family, pooled val",
                             bootstrap="pedestrian-clustered, B=%d" % args.bootstrap,
                             n_test=len(yte), n_test_pedestrians=len(set(groups))),
               families=table, pairwise=pairs)
    (HERE / "matched_comparison_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE/'matched_comparison_results.json'}")


if __name__ == "__main__":
    main()
