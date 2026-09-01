# Does the "all four families tie" finding reproduce on IDD-PeD?

PIE's second headline claim is that BiLSTM ≈ Transformer ≈ GRU ≈ vanilla RNN — the *input signal* carries the task, not the architecture or its gating. Tested here with the same machinery: **paired pedestrian-cluster bootstrap** (B = 10,000, resampling tracks, same resample on both sides), on the 5-seed probability ensembles.

IDD-PeD test: **2,357 windows, 168 positive, 757 pedestrian clusters**.

## A zero shot

| comparison | Δ AUC | 95 % cluster CI | CI excludes 0? | verdict |
|---|---|---|---|---|
| BiLSTM - Transformer | -0.0288 | [-0.0619, +0.0022] | no | tie |
| BiLSTM - GRU | +0.0013 | [-0.0208, +0.0248] | no | tie |
| BiLSTM - Vanilla RNN | +0.0103 | [-0.0234, +0.0464] | no | tie |
| Transformer - GRU | +0.0301 | [-0.0005, +0.0610] | no | tie |
| Transformer - Vanilla RNN | +0.0391 | [+0.0050, +0.0736] | yes | **difference** |
| GRU - Vanilla RNN | +0.0090 | [-0.0235, +0.0445] | no | tie |

**1 of 6** pairwise comparisons show a difference whose 95 % pedestrian-cluster CI excludes zero.

## B independent

| comparison | Δ AUC | 95 % cluster CI | CI excludes 0? | verdict |
|---|---|---|---|---|
| BiLSTM - Transformer | -0.0211 | [-0.0442, +0.0013] | no | tie |
| BiLSTM - GRU | +0.0207 | [+0.0030, +0.0367] | yes | **difference** |
| BiLSTM - Vanilla RNN | +0.0505 | [+0.0231, +0.0773] | yes | **difference** |
| Transformer - GRU | +0.0418 | [+0.0187, +0.0629] | yes | **difference** |
| Transformer - Vanilla RNN | +0.0717 | [+0.0414, +0.1019] | yes | **difference** |
| GRU - Vanilla RNN | +0.0299 | [+0.0069, +0.0531] | yes | **difference** |

**5 of 6** pairwise comparisons show a difference whose 95 % pedestrian-cluster CI excludes zero.

## Rank agreement with PIE

| experiment | PIE AUC order | IDD-PeD AUC order | Spearman ρ | p | Kendall τ | p |
|---|---|---|---|---|---|---|
| A zero shot | Vanilla RNN > Transformer > GRU > BiLSTM | Transformer > BiLSTM > GRU > Vanilla RNN | -0.400 | 0.600 | -0.333 | 0.750 |
| B independent | Vanilla RNN > Transformer > GRU > BiLSTM | Transformer > BiLSTM > GRU > Vanilla RNN | -0.400 | 0.600 | -0.333 | 0.750 |

With only four models, rank-correlation p-values cannot reach significance (the minimum attainable two-sided p for n = 4 is 0.083 for Spearman). They are reported for completeness; the pairwise CIs above are the substantive test.
