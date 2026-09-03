# Phase-matched negative sampling — a control for class-dependent timing bias

Produced by `phase_matched_control.py`.

## The problem this tests

In PIE's windowing, `crossing_point` for a **non**-crosser is defined as the last annotated
frame minus 2, so negative windows land in the final ~1-2 s of the track. For a crosser it is a
real behavioural event, usually mid-track. Measured on `data/pie_clean/`:

| class | min | median | max | (frames from anchor to end of track) |
|---|---|---|---|---|
| non-crossers | 32 | 46 | 62 | |
| crossers | 88 | 287 | 6606 | |

**Zero overlap.** That nuisance variable alone separates the classes with **AUC = 1.0000**.
This is not temporal leakage — the windows genuinely precede the crossing — it is a separate
class-dependent sampling bias, inherited from PIE's `extract_tracks_tte` and shared by the
wider benchmark family (MFT 2025 states the same convention explicitly).

## The control

Positives cannot move (their anchor is pinned by the crossing). Negatives are re-sampled
earlier, with frames-to-track-end drawn from the positive empirical distribution, floored at
88 (the positive minimum) and clipped to what each track allows. Features, splits, engine,
seeds, selection rule are unchanged; `pos_weight` is recomputed for the new train split.

- windows 4520 (pos 1648, neg 2872) vs 4,906 originally
- 115 negative pedestrians dropped. The script's `dropped` counter reports 114 — it only counts the
  `hi < MIN_TO_END` branch (track too short to place an early window). One further pedestrian lost
  every window to the 16-consecutive-frames check, which `continue`s without incrementing the
  counter. Counting pedestrians present in `data/pie_clean/` but absent from `data/pie_phase_matched/`
  gives 833 − 718 = **115**.
- splits 2084/563/1873, pos_weight 1.5665
- **AUC of `to_end` alone: 1.0000 → 0.7779** (reduced, not eliminated —
  negative tracks are shorter, so the distributions cannot be fully matched)

## Results

| model | AUC (orig → matched) | PR-AUC | F1 |
|---|---|---|---|
| BiLSTM | 0.9242 → **0.8842** (-0.0400) | 0.8688 → 0.7700 | 0.8276 → 0.7644 |
| Transformer | 0.9447 → **0.8928** (-0.0519) | 0.8964 → 0.7821 | 0.8250 → 0.7609 |
| GRU | 0.9375 → **0.8828** (-0.0547) | 0.8890 → 0.7563 | 0.8419 → 0.7642 |
| Vanilla RNN | 0.9481 → **0.8852** (-0.0629) | 0.8925 → 0.7723 | 0.8487 → 0.7531 |
| LR bbox + ego-speed (80) | 0.9488 → **0.9031** (-0.0457) | 0.9121 → 0.8184 | 0.8546 → 0.7561 |
| LR ego-speed only (16) | 0.9335 → **0.8257** (-0.1078) | 0.8538 → 0.6594 | 0.8199 → 0.7158 |
| LR bbox only (64) | 0.9129 → **0.8987** (-0.0142) | 0.8035 → 0.8118 | 0.7812 → 0.7578 |

## What it shows

**1. A real part of the original performance was the sampling artefact.** Every model loses
0.04–0.06 AUC, and the linear speed-only model loses 0.108.

**2. The ego-speed effect is the most affected — and its advantage reverses.**
Originally speed-only (0.9335) beat bbox-only (0.9129). Under phase-matched sampling
speed-only falls to 0.8257 while bbox-only barely moves (0.8987). **Bounding boxes are now the
more informative stream.** This directly supports the hypothesis that ego-speed was partly
encoding *"the car is about to pass this pedestrian"* rather than crossing behaviour.

**3. The linear baseline still matches every neural family** (LR 0.9031 vs best network
0.8928). That finding survives the control and is if anything stronger.

**4. Architecture differences shrink.** The four families span 0.8828–0.8928 (0.010) versus
0.024 on the original sampling — so part of the measured architecture gap was differential
exploitation of the artefact, not modelling capacity.

## Significance testing (`phase_matched_stats.py`)

The table above is per-seed means. The tests below use the 5-seed **ensemble** on the same
checkpoints, with pedestrian-clustered bootstrap (B = 10,000) and Holm correction applied
separately within each arm.

### Arm A — between models, on phase-matched data. **8 of 63 survive Holm.**

**None of them is a family-vs-family contrast.** All 18 tests among BiLSTM / Transformer /
GRU / Vanilla RNN fail correction on every metric. Every survivor is some model beating the
speed-only linear baseline:

| contrast | Δ | p_Holm |
|---|---|---|
| LR(80) − LR speed-only [PR-AUC] | +0.1590 | 0.0126 |
| LR(80) − LR speed-only [AUC] | +0.0774 | 0.0126 |
| Transformer − LR speed-only [AUC] | +0.0750 | 0.0126 |
| LR box-only − LR speed-only [AUC] | +0.0730 | 0.0126 |
| BiLSTM − LR speed-only [AUC] | +0.0650 | 0.0126 |
| Vanilla RNN − LR speed-only [AUC] | +0.0645 | 0.0126 |
| LR box-only − LR speed-only [PR-AUC] | +0.1524 | 0.0228 |
| GRU − LR speed-only [AUC] | +0.0610 | 0.0228 |

> Under the standard protocol 9 of 30 contrasts survive Holm and the families demonstrably
> differ. Under the control, **0 of 18 family contrasts survive**. The architecture ranking
> does not merely reorder — it stops being detectable.

### Arm B — standard vs phase-matched, per model. **10 of 21 survive Holm.**

Unpaired: phase-matching drops 115 negative pedestrians, so the two protocols have different
test sets (1,873 / 476 pedestrians vs 2,094 / 541). Each arm is bootstrapped over its own
pedestrians and the difference of the two independent distributions is reported — valid, but
less powerful than pairing.

| model | metric | standard → matched | Δ | 95 % CI | p_Holm |
|---|---|---|---|---|---|
| Vanilla RNN | AUC | 0.9545 → 0.8902 | −0.0642 | [−0.0986, −0.0300] | **0.0076** |
| Vanilla RNN | PR-AUC | 0.9050 → 0.7807 | −0.1243 | [−0.2128, −0.0400] | **0.0480** |
| Vanilla RNN | F1 | 0.8634 → 0.7725 | −0.0909 | [−0.1480, −0.0345] | **0.0420** |
| Transformer | AUC | 0.9550 → 0.9007 | −0.0543 | [−0.0880, −0.0217] | **0.0170** |
| GRU | AUC | 0.9423 → 0.8867 | −0.0556 | [−0.0924, −0.0182] | **0.0468** |
| GRU | PR-AUC | 0.8924 → 0.7603 | −0.1322 | [−0.2238, −0.0425] | **0.0448** |
| BiLSTM | AUC | 0.9320 → 0.8907 | −0.0413 | [−0.0784, −0.0042] | 0.1728 |
| **LR speed-only** | AUC | 0.9335 → 0.8257 | **−0.1077** | [−0.1485, −0.0675] | **0.0042** |
| **LR speed-only** | PR-AUC | 0.8538 → 0.6594 | **−0.1945** | [−0.2770, −0.0944] | **0.0042** |
| **LR speed-only** | F1 | 0.8199 → 0.7158 | **−0.1041** | [−0.1667, −0.0428] | **0.0144** |
| **LR box-only** | AUC | 0.9129 → 0.8987 | −0.0141 | [−0.0501, +0.0216] | **1.0000** |
| **LR box-only** | PR-AUC | 0.8035 → 0.8118 | +0.0083 | [−0.0838, +0.0999] | **1.0000** |
| **LR box-only** | F1 | 0.7812 → 0.7578 | −0.0235 | [−0.0841, +0.0368] | **1.0000** |

> The reversal is now tested, not asserted: **ego-speed-only degrades significantly on all
> three metrics, box-only on none.** Every box-only interval contains zero.

The Vanilla RNN — the family the standard protocol ranks first — is the only model whose
degradation is significant on all three metrics, which is what one expects if its advantage
came from the artefact.

## Caveats — read before quoting these numbers

- **Residual bias remains**: `to_end` still scores 0.7779, not 0.5. Negative tracks are simply
  shorter than positive ones. A stricter floor (e.g. 158) would tighten this at the cost of
  ~40 % of the negative class.
- **A competing explanation is not excluded.** Negatives sampled earlier are further from the
  camera, so their boxes are smaller and noisier. Part of the drop may be increased difficulty
  rather than a removed shortcut. Separating these would need a distance-matched design.
- Negative anchors are drawn once with seed 42; the draw was not repeated.
