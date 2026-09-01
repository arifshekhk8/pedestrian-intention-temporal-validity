"""08_family_equivalence.py — does the PIE "all four families tie" finding reproduce?

The PIE study's second headline claim is that BiLSTM ~ Transformer ~ GRU ~ vanilla RNN on
F1 (0.844 / 0.847 / 0.849 / 0.852, all CIs overlapping) — i.e. the input signal carries the
task, not the architecture or its gating.

This script tests the same claim on IDD-PeD with the same machinery: **paired** pedestrian-
cluster bootstrap on the 5-seed probability ensembles, resampling pedestrian TRACKS with
replacement and applying the *same* resample to both sides of each comparison
(mirrors `f1_optimization/07_cluster_bootstrap.py`).

Also reports rank agreement between PIE and IDD-PeD orderings (Spearman + Kendall).

Consumes only cached probability vectors written by 05/06 — no model is loaded, no test
probability is recomputed, nothing is selected here.

Writes  results/family_equivalence.json / .md
        figures/fig4_family_equivalence.png

Run from the repo root (after 05 and 06):
    python idd_ped_crossdataset/scripts/08_family_equivalence.py
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FOLDER = HERE.parent
sys.path.insert(0, str(FOLDER / "src"))
from pie_bridge import load_common, load_iddped, ped_clusters, split_masks  # noqa: E402

RES = FOLDER / "results"
SEQ = FOLDER / "data" / "sequences_iddped_clean"
MODELS = ["BiLSTM-F1", "Transformer-F1", "GRU-F1", "Vanilla_RNN-F1"]
NICE = {"BiLSTM-F1": "BiLSTM", "Transformer-F1": "Transformer",
        "GRU-F1": "GRU", "Vanilla_RNN-F1": "Vanilla RNN"}

# PIE in-domain per-seed-mean (journal_prep/Analysis/model_comparison.md)
PIE_F1 = {"BiLSTM-F1": 0.844, "Transformer-F1": 0.847, "GRU-F1": 0.849,
          "Vanilla_RNN-F1": 0.852}
PIE_AUC = {"BiLSTM-F1": 0.940, "Transformer-F1": 0.947, "GRU-F1": 0.941,
           "Vanilla_RNN-F1": 0.948}

B, RNG_SEED = 10_000, 42


def paired_cluster_delta(y, va, vb, stat, groups, B=B, seed=RNG_SEED):
    """delta_b = stat(y[idx], va[idx]) - stat(y[idx], vb[idx]) with the SAME clustered
    resample on both sides each iteration; rng reset per comparison."""
    rng = np.random.default_rng(seed)
    k = len(groups)
    out = np.empty(B)
    for b in range(B):
        idx = np.concatenate([groups[i] for i in rng.integers(0, k, k)])
        yb = y[idx]
        out[b] = stat(yb, va[idx]) - stat(yb, vb[idx])
    return out


def main():
    common = load_common()
    _, y_all, meta = load_iddped(SEQ)
    te = split_masks(meta)["test"]
    meta_te = [m for m, k in zip(meta, te) if k]
    groups = ped_clusters(meta_te)
    y = np.load(RES / "expB_probs" / "expB_5d_y_test.npy")
    assert len(y) == te.sum()

    out = {"n_test": int(te.sum()), "n_pos": int(y.sum()), "n_clusters": len(groups),
           "bootstrap_B": B, "cluster_unit": "pedestrian track", "comparisons": {}}

    for expname, loader in (("A_zero_shot", lambda m: RES / "expA_probs" / f"expA_{m}_ens.npy"),
                            ("B_independent", lambda m: RES / "expB_probs" / f"expB_5d_{m}_ens.npy")):
        probs = {}
        for m in MODELS:
            p = loader(m)
            if p.exists():
                probs[m] = np.load(p)
        if len(probs) < 2:
            print(f"skipping {expname}: cached probabilities missing")
            continue

        yA = (np.load(RES / "expA_probs" / "expA_y_test.npy")
              if expname == "A_zero_shot" else y)
        block = {}
        for a, b in itertools.combinations([m for m in MODELS if m in probs], 2):
            d = paired_cluster_delta(yA, probs[a], probs[b], common.auc_fast, groups)
            lo, hi = np.percentile(d, [2.5, 97.5])
            block[f"{NICE[a]} - {NICE[b]}"] = dict(
                delta_auc=float(d.mean()), ci95=[float(lo), float(hi)],
                excludes_zero=bool(lo > 0 or hi < 0))
        out["comparisons"][expname] = block

        # rank agreement with PIE
        aucs = {m: float(common.auc_fast(yA, probs[m])) for m in probs}
        order_idd = [m for m in sorted(aucs, key=aucs.get, reverse=True)]
        pie_vals = [PIE_AUC[m] for m in probs]
        idd_vals = [aucs[m] for m in probs]
        sp = spearmanr(pie_vals, idd_vals)
        kt = kendalltau(pie_vals, idd_vals)
        out.setdefault("rank_agreement", {})[expname] = dict(
            iddped_auc=aucs, iddped_order=order_idd,
            pie_order=[m for m in sorted(PIE_AUC, key=PIE_AUC.get, reverse=True)],
            spearman_rho=float(sp.statistic), spearman_p=float(sp.pvalue),
            kendall_tau=float(kt.statistic), kendall_p=float(kt.pvalue))

    (RES / "family_equivalence.json").write_text(json.dumps(out, indent=2))

    # ---------------------------------------------------------------- report
    md = ["# Does the \"all four families tie\" finding reproduce on IDD-PeD?\n",
          "PIE's second headline claim is that BiLSTM ≈ Transformer ≈ GRU ≈ vanilla RNN — the "
          "*input signal* carries the task, not the architecture or its gating. Tested here "
          "with the same machinery: **paired pedestrian-cluster bootstrap** (B = 10,000, "
          "resampling tracks, same resample on both sides), on the 5-seed probability "
          "ensembles.\n",
          f"IDD-PeD test: **{out['n_test']:,} windows, {out['n_pos']:,} positive, "
          f"{out['n_clusters']:,} pedestrian clusters**.\n"]

    for expname, block in out["comparisons"].items():
        md.append(f"## {expname.replace('_', ' ')}\n")
        md.append("| comparison | Δ AUC | 95 % cluster CI | CI excludes 0? | verdict |")
        md.append("|---|---|---|---|---|")
        for k, v in block.items():
            verdict = "**difference**" if v["excludes_zero"] else "tie"
            md.append(f"| {k} | {v['delta_auc']:+.4f} | "
                      f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | "
                      f"{'yes' if v['excludes_zero'] else 'no'} | {verdict} |")
        md.append("")
        n_diff = sum(v["excludes_zero"] for v in block.values())
        md.append(f"**{n_diff} of {len(block)}** pairwise comparisons show a difference whose "
                  f"95 % pedestrian-cluster CI excludes zero.\n")

    md.append("## Rank agreement with PIE\n")
    md.append("| experiment | PIE AUC order | IDD-PeD AUC order | Spearman ρ | p | Kendall τ | p |")
    md.append("|---|---|---|---|---|---|---|")
    for expname, r in out.get("rank_agreement", {}).items():
        md.append(f"| {expname.replace('_',' ')} | "
                  f"{' > '.join(NICE[m] for m in r['pie_order'])} | "
                  f"{' > '.join(NICE[m] for m in r['iddped_order'])} | "
                  f"{r['spearman_rho']:+.3f} | {r['spearman_p']:.3f} | "
                  f"{r['kendall_tau']:+.3f} | {r['kendall_p']:.3f} |")
    md.append("")
    md.append("With only four models, rank-correlation p-values cannot reach significance "
              "(the minimum attainable two-sided p for n = 4 is 0.083 for Spearman). They are "
              "reported for completeness; the pairwise CIs above are the substantive test.\n")

    (RES / "family_equivalence.md").write_text("\n".join(md))

    # ---------------------------------------------------------------- figure
    if "B_independent" in out["comparisons"]:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
        for i, expname in enumerate([e for e in ("A_zero_shot", "B_independent")
                                     if e in out["comparisons"]]):
            block = out["comparisons"][expname]
            names = list(block)
            yy = np.arange(len(names))
            for j, k in enumerate(names):
                v = block[k]
                c = "#c0392b" if v["excludes_zero"] else "#7f8c8d"
                ax[i].plot(v["ci95"], [j, j], c=c, lw=3)
                ax[i].plot(v["delta_auc"], j, "o", c="#2c3e50", ms=6)
            ax[i].axvline(0, ls="--", c="k", lw=1)
            ax[i].set_yticks(yy); ax[i].set_yticklabels(names, fontsize=8)
            ax[i].set_xlabel("Δ AUC (paired pedestrian-cluster bootstrap)")
            ax[i].set_title(f"({'ab'[i]}) {expname.replace('_', ' ')}")
            ax[i].grid(axis="x", alpha=.3)
        fig.suptitle("Pairwise family differences on IDD-PeD — grey CI crosses zero (tie)",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(FOLDER / "figures" / "fig4_family_equivalence.png", dpi=200)
        plt.close(fig)

    print(f"Wrote {RES/'family_equivalence.json'}\nWrote {RES/'family_equivalence.md'}")
    for expname, block in out["comparisons"].items():
        n_diff = sum(v["excludes_zero"] for v in block.values())
        print(f"  {expname}: {n_diff}/{len(block)} pairwise CIs exclude zero")


if __name__ == "__main__":
    main()
