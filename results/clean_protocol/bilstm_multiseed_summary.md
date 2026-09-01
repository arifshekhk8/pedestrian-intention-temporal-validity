# Issue 2 hardening — Multi-seed clean baseline (5-D)

`04_train_bilstm.py` on `sequences_clean/`, pos_weight 1.682, seeds [42, 0, 1, 2, 3]. Test split = set03 (2,094 windows), touched once per seed on the best-val checkpoint.

## Per-seed test metrics

| seed | best epoch | AUC | F1 | Acc | P | R |
|---|---|---|---|---|---|---|
| 42 | 17 | 0.9131 | 0.8228 | 0.8840 | 0.8174 | 0.8282 |
| 0 | 5 | 0.9334 | 0.8376 | 0.8902 | 0.8068 | 0.8708 |
| 1 | 6 | 0.9432 | 0.8434 | 0.8926 | 0.8016 | 0.8899 |
| 2 | 8 | 0.9363 | 0.8174 | 0.8720 | 0.7624 | 0.8811 |
| 3 | 16 | 0.9358 | 0.8163 | 0.8749 | 0.7812 | 0.8546 |

## Mean ± std

| metric | mean | std |
|---|---|---|
| auc | 0.9324 | 0.0114 |
| f1 | 0.8275 | 0.0123 |
| acc | 0.8827 | 0.0091 |
| prec | 0.7939 | 0.0220 |
| rec | 0.8649 | 0.0244 |

**Headline: test AUC = 0.932 ± 0.011** across 5 seeds on the clean, leak-free protocol.

## Two observations for the paper

- **The canonical seed (42) is the *lowest* of the five (0.913).** The single-seed
  number we first reported is conservative, not cherry-picked-high; the mean is
  0.932. Report mean ± std as the headline and note seed 42 = 0.913 for
  reproducibility against earlier runs. (Seed 42 reproduced bit-for-bit:
  0.9131143… identical to the standalone `bilstm_baseline_clean/` run → determinism
  confirmed.)
- **best-epoch scatters widely (5, 6, 8, 16, 17).** Selection is on val AUC over a
  small, skewed val set (set05/06: set05 has only 13 pedestrians, val pos rate
  0.244 vs train 0.373). Noisy model selection is the likely cause of the 0.011
  spread and is direct motivation for the planned bootstrap CIs (Issue 4) and
  leave-one-set-out CV (Issue 5).