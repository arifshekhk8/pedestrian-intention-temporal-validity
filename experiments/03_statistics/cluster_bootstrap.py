"""07_cluster_bootstrap.py — pedestrian-cluster bootstrap check (audit finding F3).

The window-level bootstrap (Issue 4; 05/06 here) resamples the 2094 test windows as
i.i.d., but they belong to only ~541 pedestrians with 50%-overlap windows — so the
effective sample size is smaller and window-level CIs may be too narrow. This script
recomputes the key intervals with a CLUSTER bootstrap: resample PEDESTRIANS with
replacement (each drawn pedestrian contributes all of its windows), 10k resamples,
same rng discipline (default_rng(42), same resampled clusters applied to both sides
of every paired delta).

Recomputed from the cached probability vectors (probs_cache/, written by 05 — no model
is loaded, no test probability recomputed, nothing selected here):
1. The three pre-registered endpoint deltas (ens vectors, fixed val-fitted taus):
   (i) A3-A0, (ii) B3-B0, (iii) B3-A3 — do the verdicts survive clustering?
2. Absolute cluster CIs (F1@tau, AUC) for the four headline arms A0/A3/B0/B3.
3. The Issue-4-style absolute AUC CI for the frozen BiLSTM ensemble (the quoted
   "[0.92, 0.95]") under clustering.

Outputs: 07_cluster_bootstrap.json, 07_cluster_bootstrap.md
"""
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("f1_common", HERE / "00_common.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

PCACHE = HERE / "probs_cache"
B, RNG_SEED = 10000, 42
ARMS = {  # arm -> (group prefix in probs_cache, operating rule)
    "A0": ("lstm_frozen", 0.5), "A3": ("lstm_a3", "tau"),
    "B0": ("tf_frozen", 0.5), "B3": ("tf_b3", "tau"),
}


def load_ped_clusters():
    with open(C.SEQ_DIR / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    te = [m for m in meta if m["set_id"] == "set03"]
    peds = np.array([m["ped_id"] for m in te])
    assert len(peds) == 2094
    uniq = np.unique(peds)
    groups = [np.where(peds == p)[0] for p in uniq]
    return uniq, groups


def cluster_indices(groups, rng):
    k = len(groups)
    chosen = rng.integers(0, k, k)
    return np.concatenate([groups[i] for i in chosen])


def cluster_ci(y, v, stat, groups, B=B, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for b in range(B):
        idx = cluster_indices(groups, rng)
        vals[b] = stat(y[idx], v[idx])
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def cluster_paired_delta(y, va, vb, stat, groups, B=B, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    deltas = np.empty(B)
    for b in range(B):
        idx = cluster_indices(groups, rng)
        yb = y[idx]
        deltas[b] = stat(yb, va[idx]) - stat(yb, vb[idx])
    return [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]


def main():
    yte = np.load(PCACHE / "y_test.npy")
    fa = json.loads((HERE / "05_final_arms.json").read_text())
    uniq, groups = load_ped_clusters()
    print(f"test: {len(yte)} windows in {len(uniq)} pedestrian clusters "
          f"(median {int(np.median([len(g) for g in groups]))} windows/ped)")

    ens = {}
    for arm, (g, rule) in ARMS.items():
        p = np.load(PCACHE / f"{g}_ens_test.npy")
        tau = 0.5 if rule == 0.5 else fa["arms"][arm]["ens"]["tau"]
        ens[arm] = dict(probs=p, tau=tau, preds=(p >= tau))
    yb = yte.astype(bool)

    results = dict(n_clusters=int(len(uniq)), B=B, endpoints={}, absolute={})
    pairs = {"i": ("A3", "A0"), "ii": ("B3", "B0"), "iii": ("B3", "A3")}
    orig = {k: e for k, e in
            zip(["i", "ii", "iii"],
                json.loads((HERE / "06_comparison_results.json").read_text())["endpoints"])}
    print()
    for name, (a, b) in pairs.items():
        d = C.f1_from_preds(yb, ens[a]["preds"]) - C.f1_from_preds(yb, ens[b]["preds"])
        ci = cluster_paired_delta(yb, ens[a]["preds"], ens[b]["preds"],
                                  C.f1_from_preds, groups)
        excl = not (ci[0] <= 0 <= ci[1])
        results["endpoints"][name] = dict(pair=f"{a} vs {b}", delta_f1=float(d),
                                          cluster_ci=ci, ci_excludes_0=excl,
                                          window_ci=orig[name]["f1_ci"])
        print(f"({name}) {a}-{b}: dF1={d:+.4f}  window CI [{orig[name]['f1_ci'][0]:+.4f},"
              f"{orig[name]['f1_ci'][1]:+.4f}]  CLUSTER CI [{ci[0]:+.4f},{ci[1]:+.4f}]"
              f"  excludes0={excl}")
    print()
    for arm in ARMS:
        f1ci = cluster_ci(yb, ens[arm]["preds"], C.f1_from_preds, groups)
        aucci = cluster_ci(yte, ens[arm]["probs"], C.auc_fast, groups)
        results["absolute"][arm] = dict(tau=ens[arm]["tau"], f1_cluster_ci=f1ci,
                                        auc_cluster_ci=aucci)
        print(f"{arm}: ens F1 cluster CI [{f1ci[0]:.4f},{f1ci[1]:.4f}]  "
              f"AUC cluster CI [{aucci[0]:.4f},{aucci[1]:.4f}]")

    (HERE / "07_cluster_bootstrap.json").write_text(json.dumps(results, indent=2))

    e = results["endpoints"]
    verdict_i = "IMPROVED (cluster CI still excludes 0)" if e["i"]["ci_excludes_0"] else \
        "downgraded to NO SIGNIFICANT CHANGE under clustering — report both intervals"
    L = ["# 07 — Pedestrian-cluster bootstrap (audit robustness check)", "",
         f"Windows are pedestrian-correlated ({len(uniq)} clusters for 2094 windows, "
         "50% overlap), so i.i.d.-window CIs understate uncertainty. This recomputes "
         "the pre-registered endpoints and headline absolute CIs by resampling "
         "PEDESTRIANS (all-windows-per-drawn-ped, 10k resamples, paired clusters "
         "across models).", "",
         "| endpoint | dF1 | window CI (05/06) | cluster CI | verdict under clustering |",
         "|---|---|---|---|---|"]
    for name in ["i", "ii", "iii"]:
        r = e[name]
        v = ("effect holds" if r["ci_excludes_0"] else
             ("TIE (unchanged)" if name in ("ii", "iii") else "no longer significant"))
        L.append(f"| ({name}) {r['pair']} | {r['delta_f1']:+.4f} | "
                 f"[{r['window_ci'][0]:+.4f}, {r['window_ci'][1]:+.4f}] | "
                 f"[{r['cluster_ci'][0]:+.4f}, {r['cluster_ci'][1]:+.4f}] | {v} |")
    L += ["", "| arm | ens F1 95% cluster CI | ens AUC 95% cluster CI |", "|---|---|---|"]
    for arm in ARMS:
        r = results["absolute"][arm]
        L.append(f"| {arm} | [{r['f1_cluster_ci'][0]:.4f}, {r['f1_cluster_ci'][1]:.4f}] "
                 f"| [{r['auc_cluster_ci'][0]:.4f}, {r['auc_cluster_ci'][1]:.4f}] |")
    L += ["", f"**Endpoint (i) under clustering: {verdict_i}.** Cluster intervals are "
          "the ones to quote in the manuscript wherever a CI appears (they are wider "
          "and honest to the dependence structure); window-level intervals remain in "
          "the original reports as the pre-registered primary analysis, now explicitly "
          "labeled as window-level.", ""]
    (HERE / "07_cluster_bootstrap.md").write_text("\n".join(L))
    print("\nwrote 07_cluster_bootstrap.json, 07_cluster_bootstrap.md")


if __name__ == "__main__":
    main()
