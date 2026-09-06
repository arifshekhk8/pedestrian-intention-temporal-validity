# Ego-speed ablation under the phase-matched control

Produced by `ego_speed_ablation_phase_matched.py`. The single variable is feature 4
(`vehicle_speed`): 5-D versus 4-D bbox-only, same family, same config, same protocol.

## Protocol

| | |
|---|---|
| data | `data/pie_phase_matched_trainonly` |
| split | train 2,084 / val 563 / test 1,873 windows |
| test pedestrians | 476 |
| test prevalence | 36.4% |
| class weight | 1.5665 (this split's own train ratio, not 1.682) |
| seeds | [42, 0, 1, 2, 3] |
| checkpoint rule | best validation AUC |
| threshold | one tau per arm, argmax F1 on pooled validation probabilities |
| device | cpu, torch 2.12.0 |
| inference | pedestrian-clustered, B=10000, seed 42 |
| correction | Holm-Bonferroni across 12 tests |

The 5-D arms are the cached phase-matched checkpoints and were **not** retrained;
only the 4-D arms are new (`runs/phase_matched_trainonly_bbox_only`). Accuracy is recorded per seed but not
bootstrapped — testing it would make 16 tests and shift every Holm-adjusted p below.

## Per-seed mean ± sd

| family | arm | params | AUC | PR-AUC | F1 | accuracy |
|---|---|---|---|---|---|---|
| BiLSTM | 5D | 2,237,313 | 0.8923 ± 0.0062 | 0.7897 ± 0.0198 | 0.7698 ± 0.0044 | 0.8184 ± 0.0037 |
| BiLSTM | 4D | 2,237,249 | 0.8970 ± 0.0036 | 0.8103 ± 0.0073 | 0.7506 ± 0.0072 | 0.7767 ± 0.0132 |
| Transformer | 5D | 794,241 | 0.8971 ± 0.0059 | 0.7947 ± 0.0228 | 0.7544 ± 0.0073 | 0.7828 ± 0.0155 |
| Transformer | 4D | 794,113 | 0.8959 ± 0.0027 | 0.7953 ± 0.0160 | 0.7591 ± 0.0067 | 0.7941 ± 0.0107 |
| GRU | 5D | 1,678,209 | 0.8817 ± 0.0147 | 0.7684 ± 0.0318 | 0.7401 ± 0.0353 | 0.7964 ± 0.0133 |
| GRU | 4D | 1,678,145 | 0.8871 ± 0.0056 | 0.7634 ± 0.0348 | 0.7626 ± 0.0097 | 0.8083 ± 0.0110 |
| Vanilla RNN | 5D | 560,001 | 0.8872 ± 0.0113 | 0.7749 ± 0.0103 | 0.7553 ± 0.0358 | 0.8104 ± 0.0211 |
| Vanilla RNN | 4D | 559,937 | 0.8936 ± 0.0055 | 0.7938 ± 0.0198 | 0.7420 ± 0.0207 | 0.7657 ± 0.0360 |

Dropping the channel removes 64 parameters from the input projection, so the two
arms are the same size to four significant figures — the contrast is the input, not
capacity.

### The raw per-seed AUC behind those means

| family | arm | seed 42 | seed 0 | seed 1 | seed 2 | seed 3 | selected epochs |
|---|---|---|---|---|---|---|---|
| BiLSTM | 5D | 0.8825 | 0.8921 | 0.8950 | 0.8927 | 0.8994 | 12, 7, 9, 8, 4 |
| BiLSTM | 4D | 0.8912 | 0.8997 | 0.8962 | 0.8976 | 0.9003 | 1, 2, 4, 5, 2 |
| Transformer | 5D | 0.8982 | 0.9026 | 0.8988 | 0.8991 | 0.8869 | 3, 6, 8, 7, 3 |
| Transformer | 4D | 0.8985 | 0.8969 | 0.8955 | 0.8914 | 0.8973 | 3, 2, 4, 5, 2 |
| GRU | 5D | 0.8737 | 0.8744 | 0.8998 | 0.8947 | 0.8658 | 8, 11, 1, 2, 9 |
| GRU | 4D | 0.8858 | 0.8921 | 0.8814 | 0.8937 | 0.8826 | 1, 4, 4, 1, 4 |
| Vanilla RNN | 5D | 0.8971 | 0.8917 | 0.8894 | 0.8678 | 0.8902 | 10, 13, 9, 26, 5 |
| Vanilla RNN | 4D | 0.8992 | 0.8871 | 0.8970 | 0.8964 | 0.8884 | 1, 4, 1, 1, 6 |

Full per-seed values including PR-AUC, F1, accuracy and training time are in
`ego_speed_ablation_phase_matched_results_per_seed.csv`.


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

**0 of 12 contrasts survive Holm.** A positive delta means
the 5-D arm scored higher, i.e. ego speed helped. Machine-readable copy in
`ego_speed_ablation_phase_matched_results_contrasts.csv`.

### Reading the metric directions

The sign is not consistent across metrics: for the BiLSTM and the vanilla RNN the
4-D arm is ahead on AUC and PR-AUC but behind on F1 and accuracy. That is a threshold
effect, not a contradiction. Each arm gets its own tau from its own pooled validation
probabilities, and the 4-D arms land much lower (BiLSTM 0.217 vs 0.486), which trades
precision for recall and costs accuracy at a 36.4 % prevalence. AUC and PR-AUC are
threshold-free and are the cleaner read on what the input channel contributes.

None of these directions is established: every interval crosses zero after correction.
The result is an absence of an ego-speed effect, not evidence that dropping the
channel helps.

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
