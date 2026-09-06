# Ego-speed ablation under the phase-matched control

Produced by `ego_speed_ablation_phase_matched.py`. The single variable is feature 4
(`vehicle_speed`): 5-D versus 4-D bbox-only, same family, same config, same protocol.

- data `pie_phase_matched_trainonly`, 1,873 test windows from 476 pedestrians
- class weight 1.5665 (this split's own train ratio), seeds [42, 0, 1, 2, 3], select auc, cpu
- pedestrian-clustered, B=10000, Holm-Bonferroni across 12 tests
- the 5-D arms are the cached phase-matched checkpoints, not retrained

## Per-seed mean ± sd

| family | arm | AUC | PR-AUC | F1 |
|---|---|---|---|---|
| BiLSTM | 5D | 0.8923 ± 0.0062 | 0.7897 ± 0.0198 | 0.7698 ± 0.0044 |
| BiLSTM | 4D | 0.8970 ± 0.0036 | 0.8103 ± 0.0073 | 0.7506 ± 0.0072 |
| Transformer | 5D | 0.8971 ± 0.0059 | 0.7947 ± 0.0228 | 0.7544 ± 0.0073 |
| Transformer | 4D | 0.8959 ± 0.0027 | 0.7953 ± 0.0160 | 0.7591 ± 0.0067 |
| GRU | 5D | 0.8817 ± 0.0147 | 0.7684 ± 0.0318 | 0.7401 ± 0.0353 |
| GRU | 4D | 0.8871 ± 0.0056 | 0.7634 ± 0.0348 | 0.7626 ± 0.0097 |
| Vanilla RNN | 5D | 0.8872 ± 0.0113 | 0.7749 ± 0.0103 | 0.7553 ± 0.0358 |
| Vanilla RNN | 4D | 0.8936 ± 0.0055 | 0.7938 ± 0.0198 | 0.7420 ± 0.0207 |

## 5-D − 4-D, paired on the pedestrian-clustered bootstrap

| contrast | Δ | 95% CI | p_Holm | |
|---|---|---|---|---|
| BiLSTM 5D-4D [AUC] | -0.0051 | [-0.0120, +0.0014] | 0.8077 | n.s. |
| BiLSTM 5D-4D [PR-AUC] | -0.0347 | [-0.0651, -0.0009] | 0.4686 | n.s. |
| BiLSTM 5D-4D [F1@tau] | +0.0273 | [-0.0014, +0.0555] | 0.5619 | n.s. |
| Transformer 5D-4D [AUC] | +0.0016 | [-0.0034, +0.0062] | 1.0000 | n.s. |
| Transformer 5D-4D [PR-AUC] | +0.0041 | [-0.0042, +0.0118] | 1.0000 | n.s. |
| Transformer 5D-4D [F1@tau] | -0.0140 | [-0.0280, +0.0003] | 0.5619 | n.s. |
| GRU 5D-4D [AUC] | -0.0032 | [-0.0095, +0.0032] | 1.0000 | n.s. |
| GRU 5D-4D [PR-AUC] | -0.0021 | [-0.0276, +0.0230] | 1.0000 | n.s. |
| GRU 5D-4D [F1@tau] | -0.0068 | [-0.0208, +0.0067] | 1.0000 | n.s. |
| Vanilla RNN 5D-4D [AUC] | -0.0060 | [-0.0137, +0.0015] | 0.8077 | n.s. |
| Vanilla RNN 5D-4D [PR-AUC] | -0.0284 | [-0.0572, -0.0016] | 0.4392 | n.s. |
| Vanilla RNN 5D-4D [F1@tau] | +0.0313 | [-0.0010, +0.0631] | 0.5619 | n.s. |

**0 of 12 contrasts survive Holm.**

## Against the event-anchored answer

The same contrast on `data/pie_clean/`, from `ego_speed_ablation_results.json`.
The two protocols keep different pedestrians, so these columns are **not paired** —
read them as two experiments.

| family | ΔAUC event-anchored | ΔAUC phase-matched | change |
|---|---|---|---|
| BiLSTM | +0.1477 | -0.0046 | -0.1524 |
| Transformer | +0.0156 | +0.0012 | -0.0144 |
| GRU | +0.0538 | -0.0054 | -0.0593 |
| Vanilla RNN | +0.0715 | -0.0064 | -0.0779 |
