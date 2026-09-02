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
def make_resamples(groups, B, seed=42):
    """Precompute B pedestrian-level resamples ONCE so every comparison is paired
    on the same bootstrap replicates (and so we only pay for this once)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by_g = [np.where(inv == i)[0] for i in range(len(uniq))]
    G = len(uniq)
    picks = rng.integers(0, G, size=(B, G), dtype=np.int32)
    return idx_by_g, picks


def cluster_delta(y, pa, pb, idx_by_g, picks, stat_a, stat_b):
    """Paired pedestrian-clustered bootstrap. Each arm may use its own statistic
    (needed for F1, where each family keeps its own threshold).
    Returns observed delta, 95% percentile CI, and a two-sided bootstrap p-value."""
    obs = stat_a(y, pa) - stat_b(y, pb)
    B = len(picks)
    deltas = np.empty(B)
    for b in range(B):
        idx = np.concatenate([idx_by_g[i] for i in picks[b]])
        yy = y[idx]
        sm = yy.sum()
        if sm == 0 or sm == len(yy):
            deltas[b] = np.nan
            continue
        deltas[b] = stat_a(yy, pa[idx]) - stat_b(yy, pb[idx])
    d = deltas[~np.isnan(deltas)]
    n = len(d)
    # two-sided bootstrap p: how often the replicate distribution sits on the
    # other side of zero (add-one smoothing so p is never exactly 0)
    p_lo = (1 + np.sum(d <= 0)) / (n + 1)
    p_hi = (1 + np.sum(d >= 0)) / (n + 1)
    pval = float(min(1.0, 2 * min(p_lo, p_hi)))
    return float(obs), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), pval


def holm(pvals, alpha=0.05):
    """Holm-Bonferroni step-down. Returns (adjusted p, reject) in input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])   # enforce monotonicity
        adj[i] = min(1.0, running)
    return adj, [adj[i] <= alpha for i in range(m)]


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

    # pairwise pedestrian-clustered bootstrap, shared replicates, Holm-corrected
    labels = list(FAMILIES)
    idx_by_g, picks = make_resamples(groups, args.bootstrap)
    print(f"\nbootstrap: B={args.bootstrap}, pedestrian-clustered, shared replicates")
    tests = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            ta, tb = table[a]["tau"], table[b]["tau"]
            for name, sa, sb in (
                ("AUC", auc, auc),
                ("PR-AUC", pr_auc, pr_auc),
                ("F1@tau", (lambda y_, p_, t=ta: f1_at(y_, p_, t)),
                           (lambda y_, p_, t=tb: f1_at(y_, p_, t))),
            ):
                o, lo, hi, pv = cluster_delta(np.asarray(yte), probs_store[a],
                                              probs_store[b], idx_by_g, picks, sa, sb)
                tests.append(dict(key=f"{a} - {b} [{name}]", delta=o, ci=[lo, hi], p=pv))
            print(f"  done {a} vs {b}")

    adj, rej = holm([t["p"] for t in tests])
    pairs = {}
    for t, pa_, rj in zip(tests, adj, rej):
        t["p_holm"] = pa_
        t["significant_holm"] = bool(rj)
        t["significant_uncorrected"] = bool(t["ci"][0] > 0 or t["ci"][1] < 0)
        pairs[t["key"]] = {k: v for k, v in t.items() if k != "key"}

    print(f"\n{'comparison':38s} {'delta':>9s} {'95% CI':>22s} {'p':>8s} {'p_holm':>8s}  verdict")
    for t in tests:
        v = "DIFFERENT" if t["significant_holm"] else (
            "drops after Holm" if t["significant_uncorrected"] else "not distinguishable")
        print(f"  {t['key']:36s} {t['delta']:+9.4f} "
              f"[{t['ci'][0]:+.4f},{t['ci'][1]:+.4f}] {t['p']:8.4f} {t['p_holm']:8.4f}  {v}")

    out = dict(protocol=dict(seeds=SEEDS, pos_weight=POS_WEIGHT, select=SELECT,
                             device=DEVICE, threshold="one tau per family, pooled val",
                             bootstrap="pedestrian-clustered, shared replicates, B=%d" % args.bootstrap,
                             correction="Holm-Bonferroni across all %d tests, alpha=0.05" % (len(pairs)),
                             n_test=len(yte), n_test_pedestrians=len(set(groups))),
               families=table, pairwise=pairs)
    (HERE / "matched_comparison_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE/'matched_comparison_results.json'}")


if __name__ == "__main__":
    main()
