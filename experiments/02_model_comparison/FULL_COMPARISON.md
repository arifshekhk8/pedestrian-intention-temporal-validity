# Every model, both protocols

Produced by `full_comparison_table.py`, which recomputes every row in one pass so the
comparison is like-for-like. Per-seed mean ± sd over seeds 42, 0, 1, 2, 3; deterministic
models have no spread. One τ per model from pooled validation probabilities, frozen before
the test split is scored.

| | event-anchored | phase-matched |
|---|---|---|
| test windows | 2,094 | 1,873 |
| test pedestrians | 541 | 476 |
| positive prevalence | 32.5% | 36.4% |
| class weight | 1.6820 | 1.5665 |

The two protocols drop different pedestrians, so the columns are **not paired**. Read them
as two experiments, not as a paired difference.

## ROC-AUC

| model | event-anchored | phase-matched | Δ |
|---|---|---|---|
| **Neural sequence models** | | | |
| BiLSTM | 0.9242 ± 0.0086 | 0.8923 ± 0.0062 | -0.0319 |
| Transformer | 0.9447 ± 0.0090 | 0.8971 ± 0.0059 | -0.0476 |
| GRU | 0.9375 ± 0.0029 | 0.8817 ± 0.0147 | -0.0558 |
| Vanilla RNN | 0.9481 ± 0.0058 | 0.8872 ± 0.0113 | -0.0609 |
| **Linear baselines** | | | |
| LR, box + speed (80) | 0.9488 | 0.9053 | -0.0436 |
| LR, box only (64) | 0.9129 | 0.8979 | -0.0150 |
| LR, speed only (16) | 0.9335 | 0.8309 | -0.1026 |
| LR, last frame only (5) | 0.9251 | 0.8979 | -0.0271 |
| **Tree ensembles** | | | |
| Decision tree | 0.8170 ± 0.0202 | 0.7050 ± 0.0059 | -0.1121 |
| Random forest | 0.9154 ± 0.0018 | 0.8565 ± 0.0036 | -0.0589 |
| Extra trees | 0.9252 ± 0.0015 | 0.8337 ± 0.0031 | -0.0915 |
| **Trivial reference** | | | |
| Always positive | 0.5000 | 0.5000 | +0.0000 |

## PR-AUC

| model | event-anchored | phase-matched | Δ |
|---|---|---|---|
| **Neural sequence models** | | | |
| BiLSTM | 0.8688 ± 0.0089 | 0.7897 ± 0.0198 | -0.0791 |
| Transformer | 0.8964 ± 0.0188 | 0.7947 ± 0.0228 | -0.1017 |
| GRU | 0.8890 ± 0.0050 | 0.7684 ± 0.0318 | -0.1207 |
| Vanilla RNN | 0.8925 ± 0.0062 | 0.7749 ± 0.0103 | -0.1175 |
| **Linear baselines** | | | |
| LR, box + speed (80) | 0.9121 | 0.8283 | -0.0838 |
| LR, box only (64) | 0.8035 | 0.8216 | +0.0181 |
| LR, speed only (16) | 0.8538 | 0.6782 | -0.1756 |
| LR, last frame only (5) | 0.8757 | 0.8174 | -0.0583 |
| **Tree ensembles** | | | |
| Decision tree | 0.6489 ± 0.0218 | 0.5542 ± 0.0073 | -0.0948 |
| Random forest | 0.8664 ± 0.0034 | 0.7682 ± 0.0046 | -0.0982 |
| Extra trees | 0.8721 ± 0.0035 | 0.7326 ± 0.0034 | -0.1395 |
| **Trivial reference** | | | |
| Always positive | 0.3252 | 0.3636 | +0.0384 |

## F1

| model | event-anchored | phase-matched | Δ |
|---|---|---|---|
| **Neural sequence models** | | | |
| BiLSTM | 0.8276 ± 0.0174 | 0.7698 ± 0.0044 | -0.0577 |
| Transformer | 0.8250 ± 0.0274 | 0.7544 ± 0.0073 | -0.0707 |
| GRU | 0.8419 ± 0.0083 | 0.7401 ± 0.0353 | -0.1018 |
| Vanilla RNN | 0.8487 ± 0.0154 | 0.7553 ± 0.0358 | -0.0935 |
| **Linear baselines** | | | |
| LR, box + speed (80) | 0.8546 | 0.7621 | -0.0925 |
| LR, box only (64) | 0.7812 | 0.7407 | -0.0405 |
| LR, speed only (16) | 0.8199 | 0.7161 | -0.1038 |
| LR, last frame only (5) | 0.7903 | 0.7728 | -0.0176 |
| **Tree ensembles** | | | |
| Decision tree | 0.7516 ± 0.0257 | 0.6217 ± 0.0081 | -0.1299 |
| Random forest | 0.7588 ± 0.0032 | 0.7041 ± 0.0042 | -0.0546 |
| Extra trees | 0.7786 ± 0.0026 | 0.6890 ± 0.0063 | -0.0896 |
| **Trivial reference** | | | |
| Always positive | 0.4908 | 0.5333 | +0.0425 |

## Against the linear reference — event-anchored

Every model contrasted with LR box + speed (80). Pedestrian-clustered bootstrap,
B = 10,000, Holm across 30 tests. **13 survive.**

| contrast | Δ | 95% CI | p_Holm | |
|---|---|---|---|---|
| LR, last frame only (5) [AUC] | -0.0238 | [-0.0353, -0.0128] | 0.0060 | **worse** |
| LR, last frame only (5) [PR-AUC] | -0.0364 | [-0.0532, -0.0218] | 0.0060 | **worse** |
| LR, last frame only (5) [F1] | -0.0642 | [-0.0913, -0.0389] | 0.0060 | **worse** |
| Decision tree [AUC] | -0.0944 | [-0.1196, -0.0701] | 0.0060 | **worse** |
| Decision tree [PR-AUC] | -0.2100 | [-0.2559, -0.1320] | 0.0060 | **worse** |
| Decision tree [F1] | -0.1025 | [-0.1401, -0.0663] | 0.0060 | **worse** |
| Random forest [F1] | -0.0989 | [-0.1339, -0.0660] | 0.0060 | **worse** |
| Extra trees [F1] | -0.0739 | [-0.1073, -0.0413] | 0.0060 | **worse** |
| LR, box only (64) [F1] | -0.0733 | [-0.1094, -0.0383] | 0.0088 | **worse** |
| Random forest [AUC] | -0.0321 | [-0.0501, -0.0149] | 0.0088 | **worse** |
| Random forest [PR-AUC] | -0.0447 | [-0.0719, -0.0220] | 0.0088 | **worse** |
| LR, box only (64) [PR-AUC] | -0.1086 | [-0.1774, -0.0437] | 0.0190 | **worse** |
| Extra trees [PR-AUC] | -0.0396 | [-0.0673, -0.0139] | 0.0396 | **worse** |
| LR, box only (64) [AUC] | -0.0360 | [-0.0607, -0.0126] | 0.0510 | n.s. |
| Extra trees [AUC] | -0.0221 | [-0.0371, -0.0070] | 0.0864 | n.s. |
| LR, speed only (16) [AUC] | -0.0154 | [-0.0276, -0.0037] | 0.1470 | n.s. |
| BiLSTM [PR-AUC] | -0.0353 | [-0.0681, -0.0063] | 0.1960 | n.s. |
| LR, speed only (16) [PR-AUC] | -0.0583 | [-0.1066, -0.0080] | 0.3042 | n.s. |
| LR, speed only (16) [F1] | -0.0347 | [-0.0664, -0.0026] | 0.3984 | n.s. |
| BiLSTM [AUC] | -0.0169 | [-0.0336, -0.0008] | 0.4664 | n.s. |
| BiLSTM [F1] | -0.0232 | [-0.0510, +0.0041] | 0.9319 | n.s. |
| Transformer [AUC] | +0.0062 | [-0.0047, +0.0181] | 1.0000 | n.s. |
| Transformer [PR-AUC] | -0.0047 | [-0.0282, +0.0215] | 1.0000 | n.s. |
| Transformer [F1] | -0.0022 | [-0.0288, +0.0244] | 1.0000 | n.s. |
| GRU [AUC] | -0.0065 | [-0.0211, +0.0080] | 1.0000 | n.s. |
| GRU [PR-AUC] | -0.0197 | [-0.0452, +0.0086] | 1.0000 | n.s. |
| GRU [F1] | -0.0097 | [-0.0353, +0.0158] | 1.0000 | n.s. |
| Vanilla RNN [AUC] | +0.0056 | [-0.0041, +0.0164] | 1.0000 | n.s. |
| Vanilla RNN [PR-AUC] | -0.0072 | [-0.0299, +0.0202] | 1.0000 | n.s. |
| Vanilla RNN [F1] | +0.0088 | [-0.0139, +0.0319] | 1.0000 | n.s. |

## Against the linear reference — phase-matched

Every model contrasted with LR box + speed (80). Pedestrian-clustered bootstrap,
B = 10,000, Holm across 30 tests. **9 survive.**

| contrast | Δ | 95% CI | p_Holm | |
|---|---|---|---|---|
| LR, speed only (16) [AUC] | -0.0744 | [-0.0976, -0.0516] | 0.0060 | **worse** |
| LR, speed only (16) [PR-AUC] | -0.1500 | [-0.2001, -0.0804] | 0.0060 | **worse** |
| Decision tree [AUC] | -0.1839 | [-0.2228, -0.1457] | 0.0060 | **worse** |
| Decision tree [PR-AUC] | -0.2570 | [-0.3073, -0.1689] | 0.0060 | **worse** |
| Decision tree [F1] | -0.1280 | [-0.1787, -0.0783] | 0.0060 | **worse** |
| Extra trees [AUC] | -0.0694 | [-0.0985, -0.0411] | 0.0060 | **worse** |
| Extra trees [PR-AUC] | -0.0938 | [-0.1382, -0.0523] | 0.0060 | **worse** |
| Random forest [AUC] | -0.0455 | [-0.0706, -0.0209] | 0.0138 | **worse** |
| Extra trees [F1] | -0.0707 | [-0.1153, -0.0270] | 0.0220 | **worse** |
| Random forest [PR-AUC] | -0.0562 | [-0.0969, -0.0160] | 0.1176 | n.s. |
| Random forest [F1] | -0.0544 | [-0.0971, -0.0133] | 0.1840 | n.s. |
| GRU [PR-AUC] | -0.0561 | [-0.1002, -0.0131] | 0.2204 | n.s. |
| GRU [AUC] | -0.0176 | [-0.0323, -0.0033] | 0.3240 | n.s. |
| Vanilla RNN [PR-AUC] | -0.0448 | [-0.0826, -0.0083] | 0.3240 | n.s. |
| LR, speed only (16) [F1] | -0.0460 | [-0.0866, -0.0049] | 0.4544 | n.s. |
| BiLSTM [PR-AUC] | -0.0467 | [-0.0889, -0.0008] | 0.6569 | n.s. |
| Vanilla RNN [AUC] | -0.0140 | [-0.0280, -0.0005] | 0.6569 | n.s. |
| LR, last frame only (5) [AUC] | -0.0073 | [-0.0155, +0.0003] | 0.8371 | n.s. |
| BiLSTM [AUC] | -0.0109 | [-0.0251, +0.0030] | 1.0000 | n.s. |
| BiLSTM [F1] | +0.0117 | [-0.0144, +0.0382] | 1.0000 | n.s. |
| Transformer [AUC] | -0.0061 | [-0.0177, +0.0054] | 1.0000 | n.s. |
| Transformer [PR-AUC] | -0.0189 | [-0.0506, +0.0160] | 1.0000 | n.s. |
| Transformer [F1] | -0.0141 | [-0.0480, +0.0198] | 1.0000 | n.s. |
| GRU [F1] | +0.0008 | [-0.0260, +0.0278] | 1.0000 | n.s. |
| Vanilla RNN [F1] | +0.0102 | [-0.0157, +0.0366] | 1.0000 | n.s. |
| LR, box only (64) [AUC] | -0.0074 | [-0.0204, +0.0058] | 1.0000 | n.s. |
| LR, box only (64) [PR-AUC] | -0.0067 | [-0.0259, +0.0110] | 1.0000 | n.s. |
| LR, box only (64) [F1] | -0.0214 | [-0.0491, +0.0057] | 1.0000 | n.s. |
| LR, last frame only (5) [PR-AUC] | -0.0108 | [-0.0380, +0.0167] | 1.0000 | n.s. |
| LR, last frame only (5) [F1] | +0.0106 | [-0.0160, +0.0381] | 1.0000 | n.s. |
