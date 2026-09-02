# Matched four-family comparison

Produced by `matched_comparison.py`. Everything below comes from one script, one engine,
one device, one selection rule. Test set touched once.

## Protocol held constant

- **seeds**: [42, 0, 1, 2, 3]
- **pos_weight**: 1.682
- **select**: auc
- **device**: cpu
- **threshold**: one tau per family, pooled val
- **bootstrap**: pedestrian-clustered, B=2000
- **n_test**: 2094
- **n_test_pedestrians**: 541

## Results (5 seeds, mean +- sd)

| family | params | tau | F1@tau | AUC | PR-AUC |
|---|---|---|---|---|---|
| BiLSTM | 2,237,313 | 0.4580 | 0.8276 ± 0.0174 | 0.9242 ± 0.0086 | 0.8688 |
| Transformer | 794,241 | 0.5850 | 0.8250 ± 0.0274 | 0.9447 ± 0.0090 | 0.8964 |
| GRU | 1,678,209 | 0.4750 | 0.8419 ± 0.0083 | 0.9375 ± 0.0029 | 0.8890 |
| Vanilla RNN | 560,001 | 0.4970 | 0.8487 ± 0.0154 | 0.9481 ± 0.0058 | 0.8925 |
| BiLSTM-h128 | 594,561 | 0.6057 | 0.8256 ± 0.0130 | 0.9349 ± 0.0053 | 0.8808 |

## Pairwise pedestrian-clustered bootstrap (541 clusters)

| comparison | delta | 95% CI | verdict |
|---|---|---|---|
| BiLSTM - Transformer [AUC] | -0.0231 | [-0.0345, -0.0129] | **DIFFERENT** |
| BiLSTM - Transformer [PR-AUC] | -0.0306 | [-0.0594, -0.0072] | **DIFFERENT** |
| BiLSTM - Transformer [F1@tau] | -0.0210 | [-0.0390, -0.0026] | **DIFFERENT** |
| BiLSTM - GRU [AUC] | -0.0104 | [-0.0197, -0.0012] | **DIFFERENT** |
| BiLSTM - GRU [PR-AUC] | -0.0156 | [-0.0399, +0.0054] | not distinguishable |
| BiLSTM - GRU [F1@tau] | -0.0135 | [-0.0287, +0.0021] | not distinguishable |
| BiLSTM - Vanilla RNN [AUC] | -0.0225 | [-0.0342, -0.0115] | **DIFFERENT** |
| BiLSTM - Vanilla RNN [PR-AUC] | -0.0281 | [-0.0565, -0.0059] | **DIFFERENT** |
| BiLSTM - Vanilla RNN [F1@tau] | -0.0320 | [-0.0481, -0.0158] | **DIFFERENT** |
| BiLSTM - BiLSTM-h128 [AUC] | -0.0098 | [-0.0179, -0.0022] | **DIFFERENT** |
| BiLSTM - BiLSTM-h128 [PR-AUC] | -0.0088 | [-0.0339, +0.0116] | not distinguishable |
| BiLSTM - BiLSTM-h128 [F1@tau] | -0.0104 | [-0.0255, +0.0042] | not distinguishable |
| Transformer - GRU [AUC] | +0.0127 | [+0.0041, +0.0230] | **DIFFERENT** |
| Transformer - GRU [PR-AUC] | +0.0149 | [+0.0038, +0.0267] | **DIFFERENT** |
| Transformer - GRU [F1@tau] | +0.0075 | [-0.0041, +0.0197] | not distinguishable |
| Transformer - Vanilla RNN [AUC] | +0.0006 | [-0.0038, +0.0050] | not distinguishable |
| Transformer - Vanilla RNN [PR-AUC] | +0.0024 | [-0.0063, +0.0094] | not distinguishable |
| Transformer - Vanilla RNN [F1@tau] | -0.0110 | [-0.0269, +0.0034] | not distinguishable |
| Transformer - BiLSTM-h128 [AUC] | +0.0132 | [+0.0070, +0.0203] | **DIFFERENT** |
| Transformer - BiLSTM-h128 [PR-AUC] | +0.0218 | [+0.0089, +0.0330] | **DIFFERENT** |
| Transformer - BiLSTM-h128 [F1@tau] | +0.0106 | [-0.0027, +0.0240] | not distinguishable |
| GRU - Vanilla RNN [AUC] | -0.0121 | [-0.0215, -0.0045] | **DIFFERENT** |
| GRU - Vanilla RNN [PR-AUC] | -0.0125 | [-0.0231, -0.0040] | **DIFFERENT** |
| GRU - Vanilla RNN [F1@tau] | -0.0185 | [-0.0321, -0.0063] | **DIFFERENT** |
| GRU - BiLSTM-h128 [AUC] | +0.0005 | [-0.0082, +0.0084] | not distinguishable |
| GRU - BiLSTM-h128 [PR-AUC] | +0.0068 | [-0.0030, +0.0154] | not distinguishable |
| GRU - BiLSTM-h128 [F1@tau] | +0.0031 | [-0.0096, +0.0157] | not distinguishable |
| Vanilla RNN - BiLSTM-h128 [AUC] | +0.0127 | [+0.0066, +0.0199] | **DIFFERENT** |
| Vanilla RNN - BiLSTM-h128 [PR-AUC] | +0.0194 | [+0.0097, +0.0306] | **DIFFERENT** |
| Vanilla RNN - BiLSTM-h128 [F1@tau] | +0.0217 | [+0.0074, +0.0371] | **DIFFERENT** |

## Caveats

- **Multiple comparisons are NOT corrected.** 10 pairs x 3 metrics = 30 tests. Under a
  Bonferroni correction the marginal results (e.g. BiLSTM-GRU on AUC, CI upper bound
  -0.0012) would not survive. Apply a correction before quoting these as significant.
- Bootstrap B=2000 here for speed; re-run at B=10000 for publication.
- **Capacity is not matched** (560k to 2.24M). Notably the *smallest* model wins, so the
  ranking is not explained by size.
- One dataset, one test split (2,094 windows from 541 pedestrians).
- F1 here is lower than the F1-optimised ensemble numbers reported elsewhere in this
  project; that is a different arm (F1 checkpointing + 5-model ensemble), not a discrepancy.
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
