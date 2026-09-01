"""07_compare.py — RNN study Phase R5: paired-bootstrap endpoints + pre-registered verdicts.

Pure analysis: reads the RNN probability vectors cached by phase4_final/05_rnn_test_eval.py, the
frozen comparison targets cached in f1_optimization/probs_cache/, and the GRU arm cached in
gru/phase4_final/probs_cache/. No model is loaded, no test probability is recomputed, nothing is
selected here.

Method (identical to f1_optimization/06 and gru/07): 10k paired percentile bootstrap over the
2094 shared test windows (rng(42) reset per comparison, same resample indices both sides);
F1/acc from boolean predictions at each arm's FIXED val-fitted tau; AUC threshold-free. Headline
on the 5-seed probability-ensemble vectors; paired t over the 5 seeds is secondary (n=5).

Pre-registered endpoints (PLAN.md §6; metric hierarchy F1 -> acc -> AUC; Δ = RNN − comparison).
Arm set confirmed at the R3 checkpoint = 4 arms (added rnn_winner_auc, the AUC-selected h256,
since the F1- and AUC-winners are the same config):
  PRIMARY (Delta-F1):
    1. rnn_f1_winner  vs frozen BiLSTM         (headline RNN vs the old 0.828)
    2. rnn_f1_winner  vs BiLSTM-F1  (0.844)     <- the gating-isolation F1 comparison (vs LSTM)
    3. rnn_f1_winner  vs Transformer-F1 (0.847)
    4. rnn_f1_winner  vs GRU-F1     (0.849)     <- un-gated vs gated recurrent (cell landscape)
    5. rnn_default_f1 vs frozen BiLSTM          (un-searched-RNN control)
  SECONDARY (Delta-AUC):
    6. rnn_default_auc vs frozen BiLSTM         (matched capacity + selection: cleanest isolation)
    7. rnn_winner_auc  vs frozen BiLSTM         (AUC-optimized large RNN vs the baseline)
    8. rnn_winner_auc  vs searched Transformer  (does the AUC-selected RNN reach the TF's AUC?)

Outputs: 07_comparison_results.json, 07_comparison_report.md, 07_comparison_figure.png

Run from the repo root:  python rnn/phase5_analysis/07_compare.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RNN_PCACHE = ROOT / "rnn" / "phase4_final" / "probs_cache"
RNN_ARMS_JSON = ROOT / "rnn" / "phase4_final" / "05_final_arms.json"
GRU_PCACHE = ROOT / "gru" / "phase4_final" / "probs_cache"
GRU_ARMS_JSON = ROOT / "gru" / "phase4_final" / "05_final_arms.json"
F1OPT = ROOT / "f1_optimization"
TF_PCACHE = F1OPT / "probs_cache"
B, RNG_SEED = 10000, 42


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("rnn_common", F1OPT / "00_common.py")

yte = np.load(RNN_PCACHE / "y_test.npy")
assert np.array_equal(yte, np.load(TF_PCACHE / "y_test.npy")), "y_test mismatch RNN vs f1_opt"
assert np.array_equal(yte, np.load(GRU_PCACHE / "y_test.npy")), "y_test mismatch RNN vs GRU"
YB = yte.astype(bool)


def engine_source(pcache, arms_json, arm, label):
    """Source from an issue12-engine-style 05_final_arms.json (RNN arms, GRU arm)."""
    arms = json.loads(arms_json.read_text())["arms"]
    return dict(
        label=label,
        probs=np.load(pcache / f"{arm}_ens_test.npy"),
        tau=arms[arm]["ensemble"]["tau"],
        f1_seeds=[r["f1_tau"] for r in arms[arm]["rows"]],
        auc_seeds=[r["auc"] for r in arms[arm]["rows"]],
    )


def f1opt_source(grp, armkey, label):
    """Source from f1_optimization/05_final_arms.json (frozen LSTM/Transformer targets)."""
    arms = json.loads((F1OPT / "05_final_arms.json").read_text())["arms"]
    return dict(
        label=label,
        probs=np.load(TF_PCACHE / f"{grp}_ens_test.npy"),
        tau=arms[armkey]["ens"]["tau"],
        f1_seeds=[p["test"]["f1"] for p in arms[armkey]["per_seed"]],
        auc_seeds=[roc_auc_score(yte, np.load(TF_PCACHE / f"{grp}_seed{s}_test.npy"))
                   for s in C.SEEDS],
    )


# named sources (built once; imported by 08_cluster_bootstrap.py)
SRC = {
    "rnn_f1_winner": engine_source(RNN_PCACHE, RNN_ARMS_JSON, "rnn_f1_winner",
                                   "RNN (F1-winner) — this study"),
    "rnn_winner_auc": engine_source(RNN_PCACHE, RNN_ARMS_JSON, "rnn_winner_auc",
                                    "RNN (winner h256, AUC-selected)"),
    "rnn_default_f1": engine_source(RNN_PCACHE, RNN_ARMS_JSON, "rnn_default_f1",
                                    "RNN (default h128, F1)"),
    "rnn_default_auc": engine_source(RNN_PCACHE, RNN_ARMS_JSON, "rnn_default_auc",
                                     "RNN (default h128, AUC)"),
    "frozen_bilstm": f1opt_source("lstm_frozen", "A0", "Frozen BiLSTM (F1 0.828 / AUC 0.9324)"),
    "bilstm_f1": f1opt_source("lstm_a3", "A3", "BiLSTM-F1 (0.844)"),
    "transformer_f1": f1opt_source("tf_b3", "B3", "Transformer-F1 (0.847)"),
    "searched_tf": f1opt_source("tf_frozen", "B0", "Searched Transformer (AUC 0.9497)"),
    "gru_f1": engine_source(GRU_PCACHE, GRU_ARMS_JSON, "gru_f1_winner", "GRU-F1 (0.849)"),
}

# (name, left source key = RNN arm, right source key = comparison target, primary metric)
ENDPOINTS = [
    ("1", "rnn_f1_winner", "frozen_bilstm", "f1"),
    ("2", "rnn_f1_winner", "bilstm_f1", "f1"),
    ("3", "rnn_f1_winner", "transformer_f1", "f1"),
    ("4", "rnn_f1_winner", "gru_f1", "f1"),
    ("5", "rnn_default_f1", "frozen_bilstm", "f1"),
    ("6", "rnn_default_auc", "frozen_bilstm", "auc"),
    ("7", "rnn_winner_auc", "frozen_bilstm", "auc"),
    ("8", "rnn_winner_auc", "searched_tf", "auc"),
]

TITLES = {
    "1": "rnn_f1_winner vs frozen BiLSTM — what the RNN (F1-optimized) achieves vs the old 0.828",
    "2": "rnn_f1_winner vs BiLSTM-F1 — **gating-isolation F1** (un-gated RNN vs the gated LSTM)",
    "3": "rnn_f1_winner vs Transformer-F1 — RNN vs the searched transformer under F1",
    "4": "rnn_f1_winner vs GRU-F1 — **cell landscape** (un-gated RNN vs gated GRU, both recurrent)",
    "5": "rnn_default_f1 vs frozen BiLSTM — un-searched RNN on the BiLSTM's own recipe (control)",
    "6": "rnn_default_auc vs frozen BiLSTM — **matched capacity + selection** (cleanest isolation), AUC",
    "7": "rnn_winner_auc vs frozen BiLSTM — AUC-optimized large RNN vs the baseline, AUC",
    "8": "rnn_winner_auc vs searched Transformer — does the AUC-selected RNN reach the TF's AUC?",
}


def endpoint(name, left_key, right_key, metric):
    L, R = SRC[left_key], SRC[right_key]
    pg, pt, tg, tt = L["probs"], R["probs"], L["tau"], R["tau"]
    predg, predt = pg >= tg, pt >= tt

    d_f1 = C.paired_bootstrap(YB, predg, predt, C.f1_from_preds, C.f1_from_preds, B, RNG_SEED)
    d_acc = C.paired_bootstrap(YB, predg, predt, C.acc_from_preds, C.acc_from_preds, B, RNG_SEED)
    d_auc = C.paired_bootstrap(yte, pg, pt, C.auc_fast, C.auc_fast, B, RNG_SEED)

    ens_f1_g, ens_f1_t = C.f1_from_preds(YB, predg), C.f1_from_preds(YB, predt)
    ens_auc_g, ens_auc_t = C.auc_fast(yte, pg), C.auc_fast(yte, pt)

    if metric == "f1":
        seeds_g, seeds_t = L["f1_seeds"], R["f1_seeds"]
        d = ens_f1_g - ens_f1_t
        ci = [float(np.percentile(d_f1, 2.5)), float(np.percentile(d_f1, 97.5))]
    else:
        seeds_g, seeds_t = L["auc_seeds"], R["auc_seeds"]
        d = ens_auc_g - ens_auc_t
        ci = [float(np.nanpercentile(d_auc, 2.5)), float(np.nanpercentile(d_auc, 97.5))]
    t = stats.ttest_rel(seeds_g, seeds_t)
    excl = not (ci[0] <= 0 <= ci[1])
    verdict = "WIN" if (excl and d > 0) else ("LOSS" if (excl and d < 0) else "TIE")

    return dict(name=name, rnn=left_key, target=right_key, target_label=R["label"],
                metric=metric, delta=float(d), ci=ci, verdict=verdict,
                ens_f1_rnn=float(ens_f1_g), ens_f1_target=float(ens_f1_t),
                ens_auc_rnn=float(ens_auc_g), ens_auc_target=float(ens_auc_t),
                delta_f1=float(ens_f1_g - ens_f1_t),
                f1_ci=[float(np.percentile(d_f1, 2.5)), float(np.percentile(d_f1, 97.5))],
                delta_acc=float(C.acc_from_preds(YB, predg) - C.acc_from_preds(YB, predt)),
                acc_ci=[float(np.percentile(d_acc, 2.5)), float(np.percentile(d_acc, 97.5))],
                delta_auc=float(ens_auc_g - ens_auc_t),
                auc_ci=[float(np.nanpercentile(d_auc, 2.5)), float(np.nanpercentile(d_auc, 97.5))],
                t_stat=float(t.statistic), t_p=float(t.pvalue),
                per_seed_delta=[float(a - b) for a, b in zip(seeds_g, seeds_t)])


def main():
    eps = [endpoint(*e) for e in ENDPOINTS]
    for e in eps:
        pm = "ΔF1" if e["metric"] == "f1" else "ΔAUC"
        print(f"({e['name']}) {e['rnn']} vs {e['target']}: {pm}={e['delta']:+.4f} "
              f"CI [{e['ci'][0]:+.4f},{e['ci'][1]:+.4f}] p={e['t_p']:.3f} -> {e['verdict']}")

    (HERE / "07_comparison_results.json").write_text(
        json.dumps(dict(endpoints=eps, B=B, rng_seed=RNG_SEED), indent=2))
    write_report(eps)
    make_figure(eps)
    print("wrote 07_comparison_results.json, 07_comparison_report.md, 07_comparison_figure.png")


def write_report(eps):
    by = {e["name"]: e for e in eps}

    def line(e):
        pm = "F1" if e["metric"] == "f1" else "AUC"
        return (f"**Δ{pm} = {e['delta']:+.4f}**, 95% CI [{e['ci'][0]:+.4f}, {e['ci'][1]:+.4f}] "
                f"({'excludes' if e['verdict'] != 'TIE' else 'includes'} 0). "
                f"ΔF1 {e['delta_f1']:+.4f} CI [{e['f1_ci'][0]:+.4f},{e['f1_ci'][1]:+.4f}]; "
                f"Δacc {e['delta_acc']:+.4f}; ΔAUC {e['delta_auc']:+.4f} "
                f"CI [{e['auc_ci'][0]:+.4f},{e['auc_ci'][1]:+.4f}]. "
                f"Paired t (n=5): t={e['t_stat']:.3f}, p={e['t_p']:.4f}. **Verdict: {e['verdict']}.**")

    L = ["# RNN study — Phase R5 comparison: endpoints & verdicts", "",
         f"Test = set03, N=2094 (touched once, in phase4_final/05). {B:,} paired percentile "
         f"bootstrap resamples (`np.random.default_rng(42)`, same indices both sides). Metric "
         f"hierarchy **F1 → acc → AUC**. All thresholds val-fitted (ensemble τ\\*), fixed before "
         f"test was touched. Δ = RNN − comparison. RNN probs from `rnn/phase4_final/probs_cache/`; "
         f"frozen targets from `f1_optimization/probs_cache/`; GRU from "
         f"`gru/phase4_final/probs_cache/` (same 2094 windows, verified identical `y_test`).", "",
         "## Headline (5-seed ensemble, test set03)", "",
         "| model | ens F1 @τ\\* | ens AUC | note |", "|---|---|---|---|",
         f"| **RNN (F1-winner, h256)** | {by['1']['ens_f1_rnn']:.4f} | {by['1']['ens_auc_rnn']:.4f} | this study (F1-selected) |",
         f"| RNN (winner h256, AUC-selected) | — | {by['7']['ens_auc_rnn']:.4f} | dedicated AUC-optimized RNN |",
         f"| RNN (default h128, F1) | {by['5']['ens_f1_rnn']:.4f} | {by['5']['ens_auc_rnn']:.4f} | un-searched control |",
         f"| RNN (default h128, AUC) | — | {by['6']['ens_auc_rnn']:.4f} | matched-size AUC twin of frozen BiLSTM |",
         f"| Frozen BiLSTM | {by['1']['ens_f1_target']:.4f} | {by['1']['ens_auc_target']:.4f} | the old 0.828/0.9324 |",
         f"| BiLSTM-F1 | {by['2']['ens_f1_target']:.4f} | {by['2']['ens_auc_target']:.4f} | F1-first LSTM |",
         f"| Transformer-F1 | {by['3']['ens_f1_target']:.4f} | {by['3']['ens_auc_target']:.4f} | F1-first TF |",
         f"| GRU-F1 | {by['4']['ens_f1_target']:.4f} | {by['4']['ens_auc_target']:.4f} | gated recurrent twin |",
         f"| Searched Transformer | — | {by['8']['ens_auc_target']:.4f} | AUC winner |",
         "",
         "(Ensemble = the 5 seeds' averaged probabilities — a deployable predictor, a different "
         "statistic from the per-seed mean; see `phase4_final/05_final_summary.md` for both.)", "",
         "## PRIMARY — F1 endpoints", ""]
    for e in eps:
        if e["name"] == "6":
            L += ["## SECONDARY — AUC endpoints", ""]
        L += [f"### ({e['name']}) {TITLES[e['name']]}", "", line(e), "",
              "Per-seed-pair Δ: " + ", ".join(f"{d:+.4f}" for d in e["per_seed_delta"]), ""]

    # outcome-adaptive verdict narrative
    e2, e4, e6, e7, e8 = by["2"], by["4"], by["6"], by["7"], by["8"]
    cell_iso = (e2, e4, e6)                       # the gating/cell-isolation endpoints
    any_loss = any(x["verdict"] == "LOSS" for x in cell_iso)
    all_tie = all(x["verdict"] == "TIE" for x in cell_iso)
    f1_ties = e2["verdict"] != "LOSS" and e4["verdict"] != "LOSS"   # ties-or-better vs LSTM & GRU
    reaches_tf = e8["verdict"] == "TIE"           # AUC-selected RNN ties the searched transformer
    tf_note = (" And — unlike the GRU, which lost to the searched transformer on AUC — the "
               "AUC-optimized vanilla RNN **ties the searched transformer** "
               f"(ΔAUC {e8['delta']:+.4f}, CI [{e8['ci'][0]:+.4f}, {e8['ci'][1]:+.4f}]): once "
               "an un-gated recurrent net gets the same search, it reaches the same AUC — direct "
               "confirmation that the transformer's edge was its *search*, not attention over "
               "recurrence.") if reaches_tf else ""
    if any_loss:
        bottom = ("the **un-gated** vanilla RNN falls measurably below the gated recurrent models "
                  "on at least one cell-isolation endpoint — **gating does buy something over "
                  "this window.** This bounds the \"cell-doesn't-matter\" claim: it holds among "
                  "*gated* cells (LSTM ≈ GRU) but not all the way down to an un-gated RNN. "
                  "Reported plainly as a LOSS, per the pre-registered templates." + tf_note)
    elif all_tie:
        bottom = ("the **un-gated** vanilla RNN is statistically indistinguishable from both the "
                  "gated LSTM (BiLSTM-F1) and the gated GRU on F1, and from the frozen BiLSTM on "
                  "AUC at matched capacity/selection. **Gating is not what matters over this "
                  "16-step window — the input signal (bbox + ego-speed) is.** Three cell types "
                  "(LSTM, GRU, vanilla RNN) tie." + tf_note)
    elif f1_ties:
        bottom = ("the **un-gated** vanilla RNN **matches or exceeds** the gated recurrent models "
                  "on every cell-isolation endpoint — it ties the gated LSTM (BiLSTM-F1) and the "
                  "gated GRU on F1, and ties-or-edges the frozen BiLSTM on AUC at matched "
                  "capacity/selection (no endpoint is a loss). **Removing the LSTM's gating costs "
                  "nothing measurable over this 16-step window** — the strongest form of the "
                  "thesis's central claim: the input signal (bbox + ego-speed), not the recurrent "
                  "cell or its gating, is what matters." + tf_note)
    else:
        bottom = ("mixed across endpoints — see the per-endpoint verdicts above; report each as "
                  "it landed, no aggregation." + tf_note)

    L += ["---", "", "## Verdict narrative", "",
          f"- **Gating isolation (vs the LSTM), F1.** RNN-F1 vs BiLSTM-F1: {e2['verdict']} "
          f"(ΔF1 {e2['delta']:+.4f}, CI [{e2['ci'][0]:+.4f}, {e2['ci'][1]:+.4f}]).",
          f"- **Cell landscape (vs the GRU), F1.** RNN-F1 vs GRU-F1: {e4['verdict']} "
          f"(ΔF1 {e4['delta']:+.4f}, CI [{e4['ci'][0]:+.4f}, {e4['ci'][1]:+.4f}]) — un-gated vs "
          f"gated recurrent.",
          f"- **Matched capacity + selection, AUC.** RNN-default-AUC vs frozen BiLSTM: "
          f"{e6['verdict']} (ΔAUC {e6['delta']:+.4f}, CI [{e6['ci'][0]:+.4f}, {e6['ci'][1]:+.4f}]).",
          f"- **vs the searched transformer, AUC.** RNN (AUC-selected h256) vs searched "
          f"Transformer: {e8['verdict']} (ΔAUC {e8['delta']:+.4f}, "
          f"CI [{e8['ci'][0]:+.4f}, {e8['ci'][1]:+.4f}]).",
          "",
          f"**Bottom line:** under the identical clean protocol and the F1-first hierarchy, "
          f"{bottom} The pedestrian-cluster bootstrap (08) is reported alongside as the honest "
          "CI.", ""]
    (HERE / "07_comparison_report.md").write_text("\n".join(L))


def make_figure(eps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    labels = [f"({e['name']}) {e['rnn'].replace('rnn_','')} − {e['target']}\n"
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
    ax.set_xlabel("Δ (RNN − comparison), paired bootstrap 95% CI")
    ax.set_title("Vanilla RNN vs BiLSTM / GRU / Transformer — pre-registered endpoints\n"
                 "(blue = F1 primary, red = AUC secondary)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3, zorder=1)
    plt.tight_layout()
    plt.savefig(HERE / "07_comparison_figure.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
