# What does ego-vehicle speed actually contribute?

Single-variable ablation: same family, same config, same protocol, ego-speed column removed
(5-D -> 4-D). Produced by `ego_speed_ablation.py`.

## Protocol

- **seeds**: [42, 0, 1, 2, 3]
- **pos_weight**: 1.682
- **select**: auc
- **device**: cpu
- **bootstrap**: pedestrian-clustered, B=10000
- **correction**: Holm-Bonferroni across 12 tests

## Per-family result

| family | 5-D AUC | 4-D AUC | drop | 5-D F1 | 4-D F1 |
|---|---|---|---|---|---|
| BiLSTM | 0.9242 | 0.7765 | **+0.1477** | 0.8276 | 0.5976 |
| Transformer | 0.9447 | 0.9291 | **+0.0156** | 0.8250 | 0.8099 |
| GRU | 0.9375 | 0.8836 | **+0.0538** | 0.8419 | 0.7535 |
| Vanilla RNN | 0.9481 | 0.8766 | **+0.0715** | 0.8487 | 0.7470 |

## Pedestrian-clustered bootstrap, Holm-corrected across 12 tests

| comparison | delta | 95% CI | p (Holm) | verdict |
|---|---|---|---|---|
| BiLSTM 5D-4D [AUC] | +0.1420 | [+0.1045, +0.1801] | 0.0024 | **DIFFERENT** |
| BiLSTM 5D-4D [PR-AUC] | +0.2100 | [+0.1490, +0.2700] | 0.0024 | **DIFFERENT** |
| BiLSTM 5D-4D [F1@tau] | +0.2209 | [+0.1717, +0.2731] | 0.0024 | **DIFFERENT** |
| Transformer 5D-4D [AUC] | +0.0126 | [-0.0041, +0.0282] | 0.2128 | not distinguishable |
| Transformer 5D-4D [PR-AUC] | +0.0373 | [-0.0079, +0.0879] | 0.2128 | not distinguishable |
| Transformer 5D-4D [F1@tau] | +0.0349 | [+0.0076, +0.0627] | 0.0378 | **DIFFERENT** |
| GRU 5D-4D [AUC] | +0.0418 | [+0.0197, +0.0636] | 0.0024 | **DIFFERENT** |
| GRU 5D-4D [PR-AUC] | +0.0653 | [+0.0278, +0.1112] | 0.0024 | **DIFFERENT** |
| GRU 5D-4D [F1@tau] | +0.0685 | [+0.0345, +0.1038] | 0.0024 | **DIFFERENT** |
| Vanilla RNN 5D-4D [AUC] | +0.0643 | [+0.0373, +0.0926] | 0.0024 | **DIFFERENT** |
| Vanilla RNN 5D-4D [PR-AUC] | +0.1287 | [+0.0657, +0.1987] | 0.0024 | **DIFFERENT** |
| Vanilla RNN 5D-4D [F1@tau] | +0.0828 | [+0.0508, +0.1169] | 0.0024 | **DIFFERENT** |

## What this shows

**Ego-speed dependence is strongly architecture-dependent — it is not one number.**
The drop ranges from +0.0126 AUC (Transformer, not significant) to +0.1477 (BiLSTM).
Quoting a single 'ego speed is worth +0.18 AUC' figure is therefore wrong: that figure
describes the BiLSTM specifically.

**Ego-speed masks architectural differences.** With the channel present the four families
span 0.9242-0.9481 AUC (a 0.024 spread). With it removed they span 0.7765-0.9291 — a spread
of **0.153, six times larger**. When an easy predictive channel is available every
architecture rides it and they look alike; remove it and modelling capacity separates them.

**A bounding-box-only Transformer (0.9291) is on par with a bbox+speed BiLSTM (0.9242).**
This resolves a standing disagreement in the literature: Achaji et al. (2022) report that
bounding boxes alone are near-ceiling, while IntFormer (Lorenzo et al., 2021) reports a
bbox-only F1 of 0.287. Both can be right — the answer depends entirely on the architecture
consuming the boxes.

## Caveat

This supersedes the project's earlier +0.179 AUC figure, which compared a locally-trained
CPU 5-D baseline against Kaggle-GPU 4-D runs whose shipped summary records pos_weight 1.44
while the notebook sets 1.682. The matched BiLSTM drop measured here (+0.1477 seed-mean,
+0.1420 on the ensemble) is close to it, so the original conclusion held for the BiLSTM —
but it was never a cross-architecture result.
