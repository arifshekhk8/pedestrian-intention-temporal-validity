"""06_f1_first_comparison.py — Phase F6: the pre-registered endpoints + verdicts
(PLAN.md §7). Pure analysis: reads 05's outputs (probs_cache/ + 05_final_arms.json);
no model is loaded, no test probability is recomputed, nothing is selected here.

Endpoints (fixed before any result was known):
  (i)   A3 vs A0 — Delta-F1: what the F1-first optimization bought the LSTM.
  (ii)  B3 vs B0 — Delta-F1: same for the transformer.
  (iii) B3 vs A3 — the family verdict under F1-first (Delta-F1 primary, Delta-acc
        secondary, Delta-AUC tertiary; WIN/TIE/LOSS template).

Method: 10k paired percentile bootstrap over the 2094 shared test windows
(rng(42) reset per comparison, same resample indices both sides), metrics recomputed
per resample at each arm's FIXED val-fitted tau (tau is a fitted parameter). Headline
on seed-averaged probability vectors (ensemble) + per-seed-pair deltas + paired t over
the 5 seeds (secondary, n=5). Everything else in the arms table is descriptive.

Outputs: 06_comparison_results.json, 06_comparison_report.md, 06_figure.png
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("f1_common", HERE.parent.parent / "src" / "metrics.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

PCACHE = HERE / "probs_cache"
B = 10000
RNG_SEED = 42
ARM_ORDER = ["A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3", "B4"]


def load():
    fa = json.loads((HERE / "05_final_arms.json").read_text())
    yte = np.load(PCACHE / "y_test.npy")
    ens = {}
    for arm, a in fa["arms"].items():
        ens[arm] = dict(probs=np.load(PCACHE / f"{a['group']}_ens_test.npy"),
                        tau=a["ens"]["tau"])
    return fa, yte, ens


def endpoint(name, yte, ea, eb, fa, arm_a, arm_b):
    """Paired bootstrap of Delta-F1 / Delta-acc / Delta-AUC: arm_a minus arm_b,
    on ensemble vectors at each arm's own fixed tau."""
    yb = yte.astype(bool)
    pa = (ea["probs"] >= ea["tau"])
    pb = (eb["probs"] >= eb["tau"])
    d_f1 = C.paired_bootstrap(yb, pa, pb, C.f1_from_preds, C.f1_from_preds, B, RNG_SEED)
    d_acc = C.paired_bootstrap(yb, pa, pb, C.acc_from_preds, C.acc_from_preds, B, RNG_SEED)
    d_auc = C.paired_bootstrap(yte, ea["probs"], eb["probs"], C.auc_fast, C.auc_fast, B, RNG_SEED)

    f1_a = C.f1_from_preds(yb, pa)
    f1_b = C.f1_from_preds(yb, pb)
    seeds_a = [p["test"]["f1"] for p in fa["arms"][arm_a]["per_seed"]]
    seeds_b = [p["test"]["f1"] for p in fa["arms"][arm_b]["per_seed"]]
    t = stats.ttest_rel(seeds_a, seeds_b)
    out = dict(name=name, a=arm_a, b=arm_b,
               ens_f1_a=f1_a, ens_f1_b=f1_b,
               delta_f1=f1_a - f1_b,
               f1_ci=[float(np.percentile(d_f1, 2.5)), float(np.percentile(d_f1, 97.5))],
               delta_acc=float(C.acc_from_preds(yb, pa) - C.acc_from_preds(yb, pb)),
               acc_ci=[float(np.percentile(d_acc, 2.5)), float(np.percentile(d_acc, 97.5))],
               delta_auc=float(C.auc_fast(yte, ea["probs"]) - C.auc_fast(yte, eb["probs"])),
               auc_ci=[float(np.nanpercentile(d_auc, 2.5)), float(np.nanpercentile(d_auc, 97.5))],
               t_stat=float(t.statistic), t_p=float(t.pvalue),
               per_seed_delta=[float(x - y) for x, y in zip(seeds_a, seeds_b)])
    ci = out["f1_ci"]
    excl = not (ci[0] <= 0 <= ci[1])
    if name == "iii":
        if excl and out["t_p"] < 0.05 and out["delta_f1"] > 0:
            out["verdict"] = "WIN"
        elif excl and out["t_p"] < 0.05 and out["delta_f1"] < 0:
            out["verdict"] = "LOSS"
        else:
            out["verdict"] = "TIE"
    else:
        out["verdict"] = "IMPROVED" if (excl and out["delta_f1"] > 0) else \
                         ("DEGRADED" if (excl and out["delta_f1"] < 0) else "NO SIGNIFICANT CHANGE")
    return out


def main():
    fa, yte, ens = load()
    eps = [endpoint("i", yte, ens["A3"], ens["A0"], fa, "A3", "A0"),
           endpoint("ii", yte, ens["B3"], ens["B0"], fa, "B3", "B0"),
           endpoint("iii", yte, ens["B3"], ens["A3"], fa, "B3", "A3")]
    for e in eps:
        print(f"({e['name']}) {e['a']} vs {e['b']}: dF1={e['delta_f1']:+.4f} "
              f"CI [{e['f1_ci'][0]:+.4f}, {e['f1_ci'][1]:+.4f}] p={e['t_p']:.4f} "
              f"-> {e['verdict']}")

    (HERE / "06_comparison_results.json").write_text(json.dumps(dict(
        endpoints=eps, B=B, rng_seed=RNG_SEED), indent=2))

    # ---------------- report ----------------
    sel = fa["selection"]
    arms = fa["arms"]
    L = ["# 06 — F1-first comparison: endpoints and verdicts", "",
         f"Test = set03, N=2094 (touched once per arm, in 05 only). {B:,} paired "
         "percentile bootstrap resamples (`np.random.default_rng(42)`, same indices "
         "both sides). Metric hierarchy per the supervisor: **F1 -> acc -> AUC**. "
         "All thresholds are val-fitted (per-seed and ensemble tau*), fixed before 05 "
         "touched test.", "",
         "## Arms (test)", "",
         "The two right-hand columns are DIFFERENT statistics of the same 5 runs "
         "(the repo's 0.932-vs-0.942 reconciliation discipline): the per-seed column "
         "is the citable model-quality number; the ensemble column combines the 5 "
         "seeds' probabilities into one deployable predictor before scoring.", "",
         "| arm | description | test F1 @tau* (5-seed mean ± std) | ens F1 @tau* | "
         "ens F1 @0.5 | acc (5-seed) | ens AUC |", "|---|---|---|---|---|---|---|"]
    for arm in ARM_ORDER:
        a = arms[arm]
        L.append(f"| {arm} | {a['desc']} | {a['test_f1_mean']:.4f} ± {a['test_f1_std']:.4f} "
                 f"| {a['ens']['test']['f1']:.4f} | {a['ens']['test_at_05']['f1']:.4f} "
                 f"| {a['test_acc_mean']:.4f} | {a['ens']['test']['auc']:.4f} |")

    names = {"i": "A3 vs A0 — what F1-first optimization bought the LSTM",
             "ii": "B3 vs B0 — what it bought the transformer",
             "iii": "B3 vs A3 — family verdict under F1-first"}
    L += ["", "## Pre-registered endpoints (ensemble vectors, paired bootstrap)", ""]
    for e in eps:
        L += [f"### ({e['name']}) {names[e['name']]}", "",
              f"**Delta-F1 = {e['delta_f1']:+.4f}**, 95% CI "
              f"[{e['f1_ci'][0]:+.4f}, {e['f1_ci'][1]:+.4f}] "
              f"({'excludes' if not (e['f1_ci'][0] <= 0 <= e['f1_ci'][1]) else 'includes'} 0). "
              f"Delta-acc = {e['delta_acc']:+.4f} CI [{e['acc_ci'][0]:+.4f}, {e['acc_ci'][1]:+.4f}]; "
              f"Delta-AUC = {e['delta_auc']:+.4f} CI [{e['auc_ci'][0]:+.4f}, {e['auc_ci'][1]:+.4f}]. "
              f"Paired t over 5 seeds: t={e['t_stat']:.3f}, p={e['t_p']:.4f} (secondary, n=5).", "",
              f"**Verdict: {e['verdict']}.**", "",
              "Per-seed-pair Delta-F1: " + ", ".join(f"{d:+.4f}" for d in e["per_seed_delta"]), ""]

    L += ["## Gates (04_selection.json)", "", "```json",
          json.dumps({k: v.get("gates") for k, v in sel.items() if isinstance(v, dict)}, indent=2),
          "```", "",
          "## Honesty notes", "",
          "- Every arm's F1@0.5 is reported alongside F1@tau* (arms table) — literature "
          "numbers are typically at each paper's own operating point.",
          "- tau* is fitted on val (N=634, 155 pos) and applied unchanged to test; the "
          "val->test transfer is visible per arm in `05_final_arms.json` "
          "(`val_at_tau` vs `test`).",
          "- A0/B0 restate the frozen models exactly (parity-gated in 05).", ""]
    (HERE / "06_comparison_report.md").write_text("\n".join(L))

    make_figure(fa, eps)
    print("wrote 06_comparison_results.json, 06_comparison_report.md, 06_figure.png")


def make_figure(fa, eps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = fa["arms"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white",
                             gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    xs = np.arange(len(ARM_ORDER))
    means = [arms[a]["test_f1_mean"] for a in ARM_ORDER]
    stds = [arms[a]["test_f1_std"] for a in ARM_ORDER]
    enss = [arms[a]["ens"]["test"]["f1"] for a in ARM_ORDER]
    colors = ["#2563eb"] * 4 + ["#dc2626"] * 4 + ["#9ca3af"]
    ax.bar(xs, means, yerr=stds, color=colors, capsize=4, zorder=2, width=0.62)
    ax.scatter(xs, enss, marker="D", color="black", s=32, zorder=3, label="ensemble @tau*")
    for x, a in zip(xs, ARM_ORDER):
        for p in arms[a]["per_seed"]:
            ax.scatter(x, p["test"]["f1"], color="black", alpha=0.3, s=12, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(ARM_ORDER)
    ax.set_ylabel("test F1 (set03)")
    ax.set_ylim(min(means) - 0.03, max(enss) + 0.02)
    ax.axhline(arms["A0"]["test_f1_mean"], color="#2563eb", ls=":", alpha=0.6, zorder=1)
    ax.axhline(0.85, color="green", ls="--", alpha=0.5, zorder=1)
    ax.annotate("literature ceiling ~0.85", (0.1, 0.851), fontsize=8, color="green")
    ax.set_title("F1-first arms ladder (bars = 5-seed mean ± std; diamonds = ensemble)")
    ax.grid(axis="y", alpha=0.3, zorder=1)
    ax.legend(loc="lower right", fontsize=8)

    ax2 = axes[1]
    labels = [f"({e['name']}) {e['a']}-{e['b']}" for e in eps]
    d = [e["delta_f1"] for e in eps]
    lo = [e["delta_f1"] - e["f1_ci"][0] for e in eps]
    hi = [e["f1_ci"][1] - e["delta_f1"] for e in eps]
    cols = ["#2563eb", "#dc2626", "#7c3aed"]
    ax2.barh(range(3), d, xerr=[lo, hi], color=cols, capsize=5, height=0.5, zorder=2)
    ax2.axvline(0, color="black", lw=1, zorder=3)
    for i, e in enumerate(eps):
        ax2.annotate(f"{e['delta_f1']:+.4f} ({e['verdict']})", (e["f1_ci"][1] + 0.001, i),
                     va="center", fontsize=9, fontweight="bold")
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Delta test F1 (paired bootstrap, 95% CI)")
    ax2.set_title("Pre-registered endpoints")
    ax2.grid(axis="x", alpha=0.3, zorder=1)
    fig.suptitle("F1-first optimization — supervisor metric hierarchy (F1 > acc > AUC)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(HERE / "06_figure.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
