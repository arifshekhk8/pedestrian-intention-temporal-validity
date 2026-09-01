# Model comparison — every model in the four families (clean PIE protocol)

Test = PIE **set03**, 2,094 windows (32.5% positive), obs_len 16, TTE∈[30,60]. Two-stream input (bounding box + ego-speed) unless noted. Every model trained under the identical frozen protocol (train set01/02/04, val set05/06, pos_weight 1.682, 5 seeds); selection on validation only, test touched once. **All models are custom architectures trained from scratch — none are pretrained** (see `README.md` for the per-model academic source).

**Per-seed-mean** = mean±std over the 5 seeds (the paper numbers). **Ensemble** = the 5 seeds' averaged probabilities (one deployable predictor; source of the confusion matrices) — a different, slightly higher statistic.

| family | model | params | source (cite) | selection | Acc | AUC | **F1** |
|---|---|---|---|---|---|---|---|
| BiLSTM | **BiLSTM (baseline)** ⭐ | 594,561 | LSTM | val AUC | 0.883 ± 0.009 | 0.932 ± 0.011 | **0.828 ± 0.012** |
| BiLSTM | **BiLSTM bbox-only (4-D)** | 594,497 | LSTM | val AUC | 0.744 ± 0.007 | 0.753 ± 0.020 | **0.551 ± 0.028** |
| BiLSTM | **BiLSTM + attention** | 611,265 | LSTM + additive attention | val AUC | 0.879 ± 0.010 | 0.925 ± 0.010 | **0.821 ± 0.009** |
| BiLSTM | **BiLSTM-F1 (h256)** | 2,237,313 | LSTM | val F1 (hybrid) | 0.897 ± 0.006 | 0.940 ± 0.004 | **0.844 ± 0.008** |
| Transformer | **Transformer (searched)** ⭐ | 794,241 | pre-LN Transformer encoder | val AUC | 0.894 ± 0.009 | 0.950 ± 0.003 | **0.845 ± 0.013** |
| Transformer | **Transformer (default, un-searched)** | 268,417 | pre-LN Transformer encoder | val F1 (hybrid) | 0.878 ± 0.006 | 0.942 ± 0.004 | **0.821 ± 0.006** |
| Transformer | **Transformer-F1** | 794,241 | pre-LN Transformer encoder | val F1 (hybrid) | 0.896 ± 0.011 | 0.947 ± 0.003 | **0.847 ± 0.017** |
| GRU | **GRU-F1 (h256)** ⭐ | 1,678,209 | GRU | val F1 (hybrid) | 0.901 ± 0.010 | 0.941 ± 0.007 | **0.849 ± 0.011** |
| GRU | **GRU (default h128, F1)** | 446,081 | GRU | val F1 (hybrid) | 0.898 ± 0.010 | 0.939 ± 0.007 | **0.844 ± 0.020** |
| GRU | **GRU (default h128, AUC)** | 446,081 | GRU | val AUC | 0.898 ± 0.007 | 0.933 ± 0.010 | **0.840 ± 0.012** |
| RNN | **Vanilla RNN-F1 (h256)** ⭐ | 560,001 | vanilla (Elman) RNN, tanh | val F1 (hybrid) | 0.902 ± 0.008 | 0.948 ± 0.002 | **0.852 ± 0.012** |
| RNN | **Vanilla RNN (winner h256, AUC)** | 560,001 | vanilla (Elman) RNN, tanh | val AUC | 0.910 ± 0.006 | 0.948 ± 0.006 | **0.845 ± 0.022** |
| RNN | **Vanilla RNN (default h128, F1)** | 149,121 | vanilla (Elman) RNN, tanh | val F1 (hybrid) | 0.897 ± 0.007 | 0.942 ± 0.007 | **0.844 ± 0.013** |
| RNN | **Vanilla RNN (default h128, AUC)** | 149,121 | vanilla (Elman) RNN, tanh | val AUC | 0.889 ± 0.010 | 0.942 ± 0.008 | **0.836 ± 0.021** |

⭐ = the four headline models (one per family). Ablation sweeps (window/TTE/hidden-size/depth/grid — same architecture, swept settings) are catalogued in `README.md`, not repeated here.

## Ensemble @ operating threshold τ (source of the confusion matrices)

| model | τ | Acc | AUC | F1 | Prec | Rec | PR-AUC | TN / FP / FN / TP |
|---|---|---|---|---|---|---|---|---|
| BiLSTM (baseline) | 0.500 | 0.891 | 0.942 | 0.837 | 0.812 | 0.863 | 0.889 | 1277 / 136 / 93 / 588 |
| BiLSTM bbox-only (4-D) | 0.500 | 0.760 | 0.802 | 0.551 | 0.705 | 0.452 | 0.644 | 1284 / 129 / 373 / 308 |
| BiLSTM + attention | 0.500 | 0.882 | 0.933 | 0.825 | 0.798 | 0.853 | 0.872 | 1266 / 147 / 100 / 581 |
| BiLSTM-F1 (h256) | 0.516 | 0.905 | 0.947 | 0.856 | 0.849 | 0.862 | 0.888 | 1309 / 104 / 94 / 587 |
| Transformer (searched) | 0.500 | 0.898 | 0.956 | 0.849 | 0.821 | 0.880 | 0.910 | 1282 / 131 / 82 / 599 |
| Transformer (default, un-searched) | 0.500 | 0.891 | 0.942 | 0.838 | 0.812 | 0.865 | 0.892 | 1277 / 136 / 92 / 589 |
| Transformer-F1 | 0.654 | 0.907 | 0.955 | 0.857 | 0.858 | 0.855 | 0.911 | 1317 / 96 / 99 / 582 |
| GRU-F1 (h256) | 0.526 | 0.911 | 0.949 | 0.863 | 0.862 | 0.863 | 0.896 | 1319 / 94 / 93 / 588 |
| GRU (default h128, F1) | 0.500 | 0.902 | 0.946 | 0.852 | 0.838 | 0.866 | 0.892 | 1299 / 114 / 91 / 590 |
| GRU (default h128, AUC) | 0.486 | 0.899 | 0.942 | 0.847 | 0.835 | 0.859 | 0.887 | 1297 / 116 / 96 / 585 |
| Vanilla RNN-F1 (h256) | 0.530 | 0.908 | 0.955 | 0.859 | 0.855 | 0.863 | 0.912 | 1313 / 100 / 93 / 588 |
| Vanilla RNN (winner h256, AUC) | 0.497 | 0.910 | 0.954 | 0.863 | 0.855 | 0.872 | 0.906 | 1312 / 101 / 87 / 594 |
| Vanilla RNN (default h128, F1) | 0.608 | 0.904 | 0.947 | 0.851 | 0.864 | 0.838 | 0.890 | 1323 / 90 / 110 / 571 |
| Vanilla RNN (default h128, AUC) | 0.555 | 0.904 | 0.948 | 0.852 | 0.859 | 0.844 | 0.893 | 1319 / 94 / 106 / 575 |

---

## Observation-window extension (OW 32 & 64 · F1-optimised model per family)

*(Supervisor directive 2026-07-19; full method in `journal_prep/obs_window_extension/PLAN.md`.)*
The headline comparison above is at **OW = 16 frames (0.5 s @ 30 fps)**. This section extends the
window to **32 (1.07 s)** and **64 (2.13 s)** for the **F1-optimised model of each family** — the same
architecture and F1 recipe, only the observation window changes. OW-16 is the already-published
reference (not retrained). All numbers are **per-seed-mean ± std over 5 seeds** on test set03, τ\*
tuned on validation, computed with the same engine and eval helpers as the table above.

**Read within a window, and as a per-family trend — not across windows as absolutes.** The test set
is *not* a matched cohort across windows: a longer window needs a longer pre-crossing track
(`L ≥ obs_len + 30`), so the usable data shrinks and its cohort shifts — **test 2094 → 1009 → 458**,
**val 634 → 302 → 138** for OW 16 → 32 → 64. Positive rate stays ~32.5 % throughout, so `pos_weight`
is held at each family's OW-16 value (single-variable isolation).

### Per-seed-mean (the paper numbers)

| family | window | params | Acc | AUC | **F1** |
|---|---|---|---|---|---|
| BiLSTM-F1 | 16 (ref) | 2,237,313 | 0.897 | 0.940 | **0.844 ± 0.008** |
| BiLSTM-F1 | 32 | 2,237,313 | 0.889 ± 0.011 | 0.936 ± 0.005 | **0.837 ± 0.014** |
| BiLSTM-F1 | 64 | 2,237,313 | 0.872 ± 0.012 | 0.932 ± 0.006 | **0.818 ± 0.018** |
| Transformer-F1 | 16 (ref) | 794,241 | 0.896 | 0.947 | **0.847 ± 0.017** |
| Transformer-F1 | 32 | 794,241 | 0.885 ± 0.020 | 0.948 ± 0.004 | **0.838 ± 0.022** |
| Transformer-F1 | 64 | 794,241 | 0.869 ± 0.030 | 0.935 ± 0.010 | **0.819 ± 0.030** |
| GRU-F1 | 16 (ref) | 1,678,209 | 0.901 | 0.941 | **0.849 ± 0.011** |
| GRU-F1 | 32 | 1,678,209 | 0.891 ± 0.011 | 0.930 ± 0.009 | **0.834 ± 0.019** |
| GRU-F1 | 64 | 1,678,209 | 0.880 ± 0.019 | 0.935 ± 0.007 | **0.822 ± 0.029** |
| Vanilla RNN-F1 | 16 (ref) | 560,001 | 0.902 | 0.948 | **0.852 ± 0.012** |
| Vanilla RNN-F1 | 32 | 560,001 | 0.887 ± 0.011 | 0.942 ± 0.008 | **0.834 ± 0.012** |
| Vanilla RNN-F1 | 64 | 560,001 | 0.857 ± 0.016 | 0.929 ± 0.005 | **0.802 ± 0.020** |

### Ensemble @ τ (5-seed probability ensemble; source of the confusion cells)

| family | window | τ | Acc | AUC | F1 | TN / FP / FN / TP |
|---|---|---|---|---|---|---|
| BiLSTM-F1 | 32 | 0.53 | 0.897 | — | 0.846 | 620 / 60 / 44 / 285 |
| BiLSTM-F1 | 64 | 0.34 | 0.847 | — | 0.800 | 248 / 61 / 9 / 140 |
| Transformer-F1 | 32 | 0.43 | 0.894 | — | 0.850 | 598 / 82 / 25 / 304 |
| Transformer-F1 | 64 | 0.50 | 0.873 | — | 0.824 | 264 / 45 / 13 / 136 |
| GRU-F1 | 32 | 0.34 | 0.893 | — | 0.843 | 611 / 69 / 39 / 290 |
| GRU-F1 | 64 | 0.35 | 0.878 | — | 0.826 | 269 / 40 / 16 / 133 |
| Vanilla RNN-F1 | 32 | 0.38 | 0.894 | — | 0.846 | 607 / 73 / 34 / 295 |
| Vanilla RNN-F1 | 64 | 0.28 | 0.834 | — | 0.784 | 244 / 65 / 11 / 138 |

### Findings

1. **Longer windows do not help — F1 declines monotonically for every family** (≈0.85 @ 16 →
   ≈0.836 @ 32 → ≈0.815 @ 64). A 0.5 s window already carries the predictive dynamics; adding
   history dilutes the signal and (mechanically) shrinks the usable set. This *justifies the OW-16
   design choice* and extends the BiLSTM-only Issue-6 window sweep to all four families.
2. **At OW 16 and OW 32 the four families still tie** — F1 spread ≤ 0.004 at OW 32, well inside the
   ±0.012–0.022 per-seed noise. The "cell type / gating doesn't matter" conclusion holds at these
   horizons.
3. **At OW 64 the un-gated vanilla RNN alone falls behind.** It posts the lowest F1 (0.802), AUC
   (0.929), Acc (0.857) *and* ensemble F1 (0.784) of the four, and its F1 drop from OW-16 (−0.050) is
   roughly double the gated cells' (BiLSTM −0.026, GRU −0.027, Transformer −0.028). The signal is
   consistent across all four metrics and matches theory (a gate-less tanh RNN degrades over longer
   sequences), though per-seed CIs still overlap, so it is **directional, not a bootstrap-confirmed
   loss.** This is the experiment the RNN study pre-registered ("an un-gated RNN would likely fall
   behind over long sequences") — **confirmed in direction: the family equivalence is
   horizon-bounded. Gating is redundant over 0.5 s but begins to re-earn its keep by ~2 s.**

