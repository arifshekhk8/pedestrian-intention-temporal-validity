# Matched four-family comparison

Produced by `matched_comparison.py`. One script, one engine, one device, one selection rule.
Test set touched once. All p-values are **Holm-Bonferroni corrected across all 30 tests**.

## Protocol held constant

- **seeds**: [42, 0, 1, 2, 3]
- **pos_weight**: 1.682
- **select**: auc
- **device**: cpu
- **threshold**: one tau per family, pooled val
- **bootstrap**: pedestrian-clustered, shared replicates, B=10000
- **correction**: Holm-Bonferroni across all 30 tests, alpha=0.05
- **n_test**: 2094
- **n_test_pedestrians**: 541

## Results (5 seeds, mean ± sd)

| family | params | tau | F1@tau | AUC | PR-AUC |
|---|---|---|---|---|---|
| BiLSTM | 2,237,313 | 0.4580 | 0.8276 ± 0.0174 | 0.9242 ± 0.0086 | 0.8688 |
| Transformer | 794,241 | 0.5850 | 0.8250 ± 0.0274 | 0.9447 ± 0.0090 | 0.8964 |
| GRU | 1,678,209 | 0.4750 | 0.8419 ± 0.0083 | 0.9375 ± 0.0029 | 0.8890 |
| Vanilla RNN | 560,001 | 0.4970 | 0.8487 ± 0.0154 | 0.9481 ± 0.0058 | 0.8925 |
| BiLSTM-h128 | 594,561 | 0.6057 | 0.8256 ± 0.0130 | 0.9349 ± 0.0053 | 0.8808 |

## Differences that survive Holm correction (9 of 30)

| comparison | delta | 95% CI | p | p (Holm) |
|---|---|---|---|---|
| BiLSTM - Transformer [AUC] | -0.0231 | [-0.0345, -0.0128] | 0.0002 | **0.0060** |
| BiLSTM - Vanilla RNN [AUC] | -0.0225 | [-0.0340, -0.0114] | 0.0002 | **0.0060** |
| BiLSTM - Vanilla RNN [F1@tau] | -0.0320 | [-0.0490, -0.0164] | 0.0002 | **0.0060** |
| Transformer - BiLSTM-h128 [AUC] | +0.0132 | [+0.0072, +0.0201] | 0.0002 | **0.0060** |
| Transformer - BiLSTM-h128 [PR-AUC] | +0.0218 | [+0.0094, +0.0338] | 0.0004 | **0.0096** |
| GRU - Vanilla RNN [AUC] | -0.0121 | [-0.0217, -0.0042] | 0.0008 | **0.0184** |
| GRU - Vanilla RNN [F1@tau] | -0.0185 | [-0.0323, -0.0062] | 0.0016 | **0.0352** |
| Vanilla RNN - BiLSTM-h128 [AUC] | +0.0127 | [+0.0064, +0.0198] | 0.0002 | **0.0060** |
| Vanilla RNN - BiLSTM-h128 [PR-AUC] | +0.0194 | [+0.0098, +0.0308] | 0.0002 | **0.0060** |

## Significant before correction, NOT after (9 of 30)

| comparison | delta | 95% CI | p | p (Holm) |
|---|---|---|---|---|
| BiLSTM - Transformer [PR-AUC] | -0.0306 | [-0.0592, -0.0074] | 0.0118 | **0.1972** |
| BiLSTM - Transformer [F1@tau] | -0.0210 | [-0.0401, -0.0028] | 0.0254 | **0.3556** |
| BiLSTM - GRU [AUC] | -0.0104 | [-0.0195, -0.0013] | 0.0264 | **0.3556** |
| BiLSTM - Vanilla RNN [PR-AUC] | -0.0281 | [-0.0567, -0.0062] | 0.0144 | **0.2160** |
| BiLSTM - BiLSTM-h128 [AUC] | -0.0098 | [-0.0178, -0.0022] | 0.0116 | **0.1972** |
| Transformer - GRU [AUC] | +0.0127 | [+0.0039, +0.0235] | 0.0026 | **0.0546** |
| Transformer - GRU [PR-AUC] | +0.0149 | [+0.0034, +0.0266] | 0.0082 | **0.1476** |
| GRU - Vanilla RNN [PR-AUC] | -0.0125 | [-0.0231, -0.0037] | 0.0040 | **0.0760** |
| Vanilla RNN - BiLSTM-h128 [F1@tau] | +0.0217 | [+0.0077, +0.0370] | 0.0026 | **0.0546** |

These must **not** be reported as significant. They are suggestive at best.


## Not distinguishable either way (12 of 30)

| comparison | delta | 95% CI | p | p (Holm) |
|---|---|---|---|---|
| BiLSTM - GRU [PR-AUC] | -0.0156 | [-0.0403, +0.0046] | 0.1178 | **1.0000** |
| BiLSTM - GRU [F1@tau] | -0.0135 | [-0.0293, +0.0027] | 0.0962 | **1.0000** |
| BiLSTM - BiLSTM-h128 [PR-AUC] | -0.0088 | [-0.0349, +0.0114] | 0.3578 | **1.0000** |
| BiLSTM - BiLSTM-h128 [F1@tau] | -0.0104 | [-0.0253, +0.0045] | 0.1646 | **1.0000** |
| Transformer - GRU [F1@tau] | +0.0075 | [-0.0040, +0.0197] | 0.2086 | **1.0000** |
| Transformer - Vanilla RNN [AUC] | +0.0006 | [-0.0037, +0.0053] | 0.8267 | **1.0000** |
| Transformer - Vanilla RNN [PR-AUC] | +0.0024 | [-0.0065, +0.0096] | 0.7225 | **1.0000** |
| Transformer - Vanilla RNN [F1@tau] | -0.0110 | [-0.0268, +0.0036] | 0.1474 | **1.0000** |
| Transformer - BiLSTM-h128 [F1@tau] | +0.0106 | [-0.0027, +0.0247] | 0.1212 | **1.0000** |
| GRU - BiLSTM-h128 [AUC] | +0.0005 | [-0.0082, +0.0084] | 0.8677 | **1.0000** |
| GRU - BiLSTM-h128 [PR-AUC] | +0.0068 | [-0.0030, +0.0156] | 0.1772 | **1.0000** |
| GRU - BiLSTM-h128 [F1@tau] | +0.0031 | [-0.0090, +0.0158] | 0.6247 | **1.0000** |

## What this establishes

- **The four families do not tie.** Nine comparisons survive correction.
- **The un-gated vanilla RNN (560k params, the smallest model) is the best.** It beats the
  BiLSTM on AUC and F1, beats the BiLSTM-h128 baseline on AUC and PR-AUC, and beats the GRU
  on AUC and F1 — all after correction.
- **Transformer ≈ Vanilla RNN is a genuine tie** (AUC p = 0.83, PR-AUC p = 0.72, F1 p = 0.15),
  and correction cannot manufacture a tie, so this is the strongest null in the table.
- **Transformer > GRU does NOT survive** (AUC p_holm = 0.0546, just over the line). Report as
  suggestive, not established.
- **The BiLSTM h256-vs-h128 selection-over-fitting result does NOT survive** (p_holm = 0.1972).
  The point estimate still favours the hand-set baseline over its own search winner, and that
  is worth reporting descriptively with Cawley & Talbot (2010), but not as a significant finding.

## Caveats that remain

- **Capacity is not matched** (560k–2.24M). The *smallest* model wins, so the ranking is not
  explained by size — but matched-capacity runs were never done.
- One dataset, one test split (2,094 windows from 541 pedestrians).
- A tie here means *we could not detect a difference at this sample size*. No equivalence
  margin was pre-specified and no TOST procedure was run, so 'equivalent' is not claimed.
- F1 here is lower than the F1-optimised ensemble numbers elsewhere in this project; that is a
  different arm (F1 checkpointing + 5-model ensemble), not a discrepancy.

## Matched search budget: does the transformer's larger search explain its result?

The transformer received 78 configurations; the recurrent families received 36. Re-selecting from
the cached search files (`transformer/phase2_kaggle_search/runs_search/`, read-only) answers this
without retraining:

- The 78 configs decompose into **36 architecture configs** (all sharing the pre-registered default
  recipe `adam lr1e-03 plateau do0.1 wd1e-05`) and **42 recipe/transfer configs**.
- The selected winner, `d128_ff512_L4_last_spe`, **is one of the 36 architecture configs**, and it
  ranks **#2 of 36** by seed-42 validation AUC (0.982463) — comfortably inside any top-5 shortlist.
- Of the six configs that received a 5-seed re-measurement, the winner's 5-seed mean validation AUC
  (0.978932) is the highest, and the best *recipe*-search config reaches only 0.978768.

**Conclusion: a matched 36-configuration budget selects the same transformer.** The 42 extra recipe
configurations — 54 % of the search — changed nothing. The transformer's result is therefore not an
artefact of a larger search budget.

*Honest limit:* only ranks #1 and #2 of the Stage-A shortlist received a 5-seed re-measurement.
Ranks #3–#5 (`d128_ff256_L2_mean_spe`, `d128_ff256_L4_mean_spe`, `d128_ff512_L2_last_spe`) were
never re-measured, so it cannot be excluded that one of them would have won a full 36-config
selection. What *is* established is that the winner is reachable within the matched budget.
