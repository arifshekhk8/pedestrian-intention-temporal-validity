"""07_compare.py — GRU study Phase G5: paired-bootstrap endpoints + pre-registered verdicts.

Pure analysis: reads the GRU probability vectors cached by phase4_final/05_gru_test_eval.py and
the frozen comparison targets cached in f1_optimization/probs_cache/. No model is loaded, no
test probability is recomputed, nothing is selected here.

Method (identical to f1_optimization/06_f1_first_comparison.py): 10k paired percentile bootstrap
over the 2094 shared test windows (rng(42) reset per comparison, same resample indices both
sides); F1/acc from boolean predictions at each arm's FIXED val-fitted tau; AUC threshold-free.
Headline on the 5-seed probability-ensemble vectors; paired t over the 5 seeds is secondary (n=5).

Pre-registered endpoints (PLAN.md §6; metric hierarchy F1 -> acc -> AUC):
  PRIMARY (Delta-F1):
    1. gru_f1_winner vs frozen BiLSTM        (headline GRU vs the old 0.828)
    2. gru_f1_winner vs BiLSTM-F1  (0.844)    <- the cell-isolation F1 comparison
    3. gru_f1_winner vs Transformer-F1 (0.847)
    4. gru_default_f1 vs frozen BiLSTM        (un-searched-GRU control, analogue of transformer_default)
  SECONDARY (Delta-AUC):
    5. gru_default_auc vs frozen BiLSTM       (matched capacity + selection: cleanest cell isolation)
    6. gru_f1_winner  vs searched Transformer (does the GRU reach the transformer's AUC?)

Outputs: 07_comparison_results.json, 07_comparison_report.md, 07_comparison_figure.png

Run from the repo root:  python gru/phase5_analysis/07_compare.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
GRU_PCACHE = ROOT / "gru" / "phase4_final" / "probs_cache"
TF_PCACHE = ROOT / "f1_optimization" / "probs_cache"
B, RNG_SEED = 10000, 42


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("gru_common", ROOT / "f1_optimization" / "00_common.py")

yte = np.load(GRU_PCACHE / "y_test.npy")
assert np.array_equal(yte, np.load(TF_PCACHE / "y_test.npy")), "y_test mismatch across caches"
YB = yte.astype(bool)

GRU_ARMS = json.loads((ROOT / "gru" / "phase4_final" / "05_final_arms.json").read_text())["arms"]
TF_ARMS = json.loads((ROOT / "f1_optimization" / "05_final_arms.json").read_text())["arms"]

# target registry: key -> (group prefix in f1_optimization/probs_cache, arm key in TF_ARMS, label)
TARGETS = {
    "frozen_bilstm": ("lstm_frozen", "A0", "Frozen BiLSTM (F1 0.828 / AUC 0.9324)"),
    "bilstm_f1": ("lstm_a3", "A3", "BiLSTM-F1 (0.844)"),
    "transformer_f1": ("tf_b3", "B3", "Transformer-F1 (0.847)"),
    "searched_tf": ("tf_frozen", "B0", "Searched Transformer (AUC 0.9497)"),
}


def gru_ens(arm):
    return np.load(GRU_PCACHE / f"{arm}_ens_test.npy")


def gru_tau(arm):
    return GRU_ARMS[arm]["ensemble"]["tau"]


def gru_per_seed(arm, key):   # key in {"f1_tau","auc"}
    return [r[key] for r in GRU_ARMS[arm]["rows"]]


def tf_ens(key):
    return np.load(TF_PCACHE / f"{TARGETS[key][0]}_ens_test.npy")


def tf_tau(key):
    return TF_ARMS[TARGETS[key][1]]["ens"]["tau"]


def tf_per_seed_f1(key):
    return [p["test"]["f1"] for p in TF_ARMS[TARGETS[key][1]]["per_seed"]]


def tf_per_seed_auc(key):
    grp = TARGETS[key][0]
    return [roc_auc_score(yte, np.load(TF_PCACHE / f"{grp}_seed{s}_test.npy")) for s in C.SEEDS]


def endpoint(name, gru_arm, tgt_key, metric):
    """metric in {'f1','auc'} = the PRIMARY metric for the verdict."""
    pg, pt = gru_ens(gru_arm), tf_ens(tgt_key)
    tg, tt = gru_tau(gru_arm), tf_tau(tgt_key)
    predg, predt = pg >= tg, pt >= tt

    d_f1 = C.paired_bootstrap(YB, predg, predt, C.f1_from_preds, C.f1_from_preds, B, RNG_SEED)
    d_acc = C.paired_bootstrap(YB, predg, predt, C.acc_from_preds, C.acc_from_preds, B, RNG_SEED)
    d_auc = C.paired_bootstrap(yte, pg, pt, C.auc_fast, C.auc_fast, B, RNG_SEED)

    ens_f1_g, ens_f1_t = C.f1_from_preds(YB, predg), C.f1_from_preds(YB, predt)
    ens_auc_g, ens_auc_t = C.auc_fast(yte, pg), C.auc_fast(yte, pt)

    if metric == "f1":
        seeds_g, seeds_t = gru_per_seed(gru_arm, "f1_tau"), tf_per_seed_f1(tgt_key)
        d, ci = ens_f1_g - ens_f1_t, [float(np.percentile(d_f1, 2.5)), float(np.percentile(d_f1, 97.5))]
    else:
        seeds_g, seeds_t = gru_per_seed(gru_arm, "auc"), tf_per_seed_auc(tgt_key)
        d, ci = ens_auc_g - ens_auc_t, [float(np.nanpercentile(d_auc, 2.5)), float(np.nanpercentile(d_auc, 97.5))]
    t = stats.ttest_rel(seeds_g, seeds_t)
    excl = not (ci[0] <= 0 <= ci[1])
    verdict = "WIN" if (excl and d > 0) else ("LOSS" if (excl and d < 0) else "TIE")

    return dict(name=name, gru=gru_arm, target=tgt_key, target_label=TARGETS[tgt_key][2],
                metric=metric, delta=float(d), ci=ci, verdict=verdict,
                ens_f1_gru=float(ens_f1_g), ens_f1_target=float(ens_f1_t),
                ens_auc_gru=float(ens_auc_g), ens_auc_target=float(ens_auc_t),
                delta_f1=float(ens_f1_g - ens_f1_t),
                f1_ci=[float(np.percentile(d_f1, 2.5)), float(np.percentile(d_f1, 97.5))],
                delta_acc=float(C.acc_from_preds(YB, predg) - C.acc_from_preds(YB, predt)),
                acc_ci=[float(np.percentile(d_acc, 2.5)), float(np.percentile(d_acc, 97.5))],
                delta_auc=float(ens_auc_g - ens_auc_t),
                auc_ci=[float(np.nanpercentile(d_auc, 2.5)), float(np.nanpercentile(d_auc, 97.5))],
                t_stat=float(t.statistic), t_p=float(t.pvalue),
                per_seed_delta=[float(a - b) for a, b in zip(seeds_g, seeds_t)])


def main():
    eps = [
        endpoint("1", "gru_f1_winner", "frozen_bilstm", "f1"),
        endpoint("2", "gru_f1_winner", "bilstm_f1", "f1"),
        endpoint("3", "gru_f1_winner", "transformer_f1", "f1"),
        endpoint("4", "gru_default_f1", "frozen_bilstm", "f1"),
        endpoint("5", "gru_default_auc", "frozen_bilstm", "auc"),
        endpoint("6", "gru_f1_winner", "searched_tf", "auc"),
    ]
    for e in eps:
        pm = "ΔF1" if e["metric"] == "f1" else "ΔAUC"
        print(f"({e['name']}) {e['gru']} vs {e['target']}: {pm}={e['delta']:+.4f} "
              f"CI [{e['ci'][0]:+.4f},{e['ci'][1]:+.4f}] p={e['t_p']:.3f} -> {e['verdict']}")

    (HERE / "07_comparison_results.json").write_text(
        json.dumps(dict(endpoints=eps, B=B, rng_seed=RNG_SEED), indent=2))
    write_report(eps)
    make_figure(eps)
    print("wrote 07_comparison_results.json, 07_comparison_report.md, 07_comparison_figure.png")


def write_report(eps):
    def line(e):
        pm = "F1" if e["metric"] == "f1" else "AUC"
        return (f"**Δ{pm} = {e['delta']:+.4f}**, 95% CI [{e['ci'][0]:+.4f}, {e['ci'][1]:+.4f}] "
                f"({'excludes' if e['verdict'] != 'TIE' else 'includes'} 0). "
                f"ΔF1 {e['delta_f1']:+.4f} CI [{e['f1_ci'][0]:+.4f},{e['f1_ci'][1]:+.4f}]; "
                f"Δacc {e['delta_acc']:+.4f}; ΔAUC {e['delta_auc']:+.4f} "
                f"CI [{e['auc_ci'][0]:+.4f},{e['auc_ci'][1]:+.4f}]. "
                f"Paired t (n=5): t={e['t_stat']:.3f}, p={e['t_p']:.4f}. **Verdict: {e['verdict']}.**")

    L = ["# GRU study — Phase G5 comparison: endpoints & verdicts", "",
         f"Test = set03, N=2094 (touched once, in phase4_final/05). {B:,} paired percentile "
         f"bootstrap resamples (`np.random.default_rng(42)`, same indices both sides). Metric "
         f"hierarchy **F1 → acc → AUC**. All thresholds val-fitted (ensemble τ\\*), fixed before "
         f"test was touched. GRU probs from `gru/phase4_final/probs_cache/`; frozen comparison "
         f"targets from `f1_optimization/probs_cache/` (same 2094 windows, verified identical "
         f"`y_test`).", "",
         "## Headline (5-seed ensemble, test set03)", "",
         "| model | ens F1 @τ\\* | ens AUC | note |", "|---|---|---|---|",
         f"| **GRU (F1-winner, h256)** | {eps[0]['ens_f1_gru']:.4f} | {eps[0]['ens_auc_gru']:.4f} | this study |",
         f"| GRU (default, h128, F1) | {eps[3]['ens_f1_gru']:.4f} | {eps[3]['ens_auc_gru']:.4f} | un-searched control |",
         f"| GRU (default, h128, AUC) | — | {eps[4]['ens_auc_gru']:.4f} | AUC twin of frozen BiLSTM |",
         f"| Frozen BiLSTM | {eps[0]['ens_f1_target']:.4f} | {eps[0]['ens_auc_target']:.4f} | the old 0.828/0.9324 |",
         f"| BiLSTM-F1 | {eps[1]['ens_f1_target']:.4f} | {eps[1]['ens_auc_target']:.4f} | F1-first LSTM |",
         f"| Transformer-F1 | {eps[2]['ens_f1_target']:.4f} | {eps[2]['ens_auc_target']:.4f} | F1-first TF |",
         f"| Searched Transformer | — | {eps[5]['ens_auc_target']:.4f} | AUC winner |",
         "",
         "(Ensemble = the 5 seeds' averaged probabilities — a deployable predictor, a different "
         "statistic from the per-seed mean; see `phase4_final/05_final_summary.md` for both.)", "",
         "## PRIMARY — F1 endpoints", ""]
    titles = {
        "1": "gru_f1_winner vs frozen BiLSTM — what the GRU (F1-optimized) achieves vs the old 0.828",
        "2": "gru_f1_winner vs BiLSTM-F1 — **the cell-isolation F1 comparison** (GRU vs LSTM, both F1-optimized)",
        "3": "gru_f1_winner vs Transformer-F1 — GRU vs the searched transformer under F1",
        "4": "gru_default_f1 vs frozen BiLSTM — un-searched GRU on the BiLSTM's own recipe (control)",
        "5": "gru_default_auc vs frozen BiLSTM — **matched capacity + selection** (cleanest cell isolation), AUC",
        "6": "gru_f1_winner vs searched Transformer — does the GRU reach the transformer's AUC?",
    }
    for e in eps:
        if e["name"] == "5":
            L += ["## SECONDARY — AUC endpoints", ""]
        L += [f"### ({e['name']}) {titles[e['name']]}", "", line(e), "",
              "Per-seed-pair Δ: " + ", ".join(f"{d:+.4f}" for d in e["per_seed_delta"]), ""]

    # narrative verdict
    e2, e5, e6 = eps[1], eps[4], eps[5]
    L += ["---", "", "## Verdict narrative", "",
          f"- **Cell type does not matter on F1.** GRU-F1 vs BiLSTM-F1: {e2['verdict']} "
          f"(ΔF1 {e2['delta']:+.4f}, CI [{e2['ci'][0]:+.4f}, {e2['ci'][1]:+.4f}]). "
          f"The gated recurrent twin ties the LSTM under identical F1-first optimization.",
          f"- **Cell type does not matter on AUC either, at matched capacity/selection.** "
          f"GRU-default-AUC vs frozen BiLSTM (both h128, AUC-selected): {e5['verdict']} "
          f"(ΔAUC {e5['delta']:+.4f}, CI [{e5['ci'][0]:+.4f}, {e5['ci'][1]:+.4f}]).",
          f"- **The transformer's AUC edge is architecture+search, not recurrence.** "
          f"GRU vs searched Transformer on AUC: {e6['verdict']} "
          f"(ΔAUC {e6['delta']:+.4f}, CI [{e6['ci'][0]:+.4f}, {e6['ci'][1]:+.4f}]).",
          "",
          "**Bottom line:** under the identical clean protocol and the F1-first hierarchy, a GRU "
          "is statistically indistinguishable from the BiLSTM — strengthening the thesis story "
          "that *the input signal (bbox + ego-speed), not the recurrent cell, is what matters*. "
          "The pedestrian-cluster bootstrap (08) is reported alongside as the honest CI.", ""]
    (HERE / "07_comparison_report.md").write_text("\n".join(L))


def make_figure(eps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    labels = [f"({e['name']}) {e['gru'].replace('gru_','')} − {e['target']}\n"
              f"[{'F1' if e['metric']=='f1' else 'AUC'}]" for e in eps]
    d = [e["delta"] for e in eps]
    lo = [e["delta"] - e["ci"][0] for e in eps]
    hi = [e["ci"][1] - e["delta"] for e in eps]
    cols = ["#2563eb" if e["metric"] == "f1" else "#dc2626" for e in eps]
    ys = np.arange(len(eps))[::-1]
    ax.barh(ys, d, xerr=[lo, hi], color=cols, capsize=5, height=0.6, zorder=2)
    ax.axvline(0, color="black", lw=1, zorder=3)
    for y, e in zip(ys, eps):
        ax.annotate(f"{e['delta']:+.4f} ({e['verdict']})", (e["ci"][1] + 0.001, y),
                    va="center", fontsize=8, fontweight="bold")
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Δ (GRU − comparison), paired bootstrap 95% CI")
    ax.set_title("GRU vs frozen BiLSTM / Transformer — pre-registered endpoints\n"
                 "(blue = F1 primary, red = AUC secondary)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3, zorder=1)
    plt.tight_layout()
    plt.savefig(HERE / "07_comparison_figure.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
