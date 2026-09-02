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
- 114 negative pedestrians dropped: track too short to place an early window
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

## Caveats — read before quoting these numbers

- **The two conditions have different test sets** (1,873 vs 2,094 windows), so the drops are
  not a paired comparison and no significance test is reported for them.
- **Residual bias remains**: `to_end` still scores 0.7779, not 0.5. Negative tracks are simply
  shorter than positive ones. A stricter floor (e.g. 158) would tighten this at the cost of
  ~40 % of the negative class.
- **A competing explanation is not excluded.** Negatives sampled earlier are further from the
  camera, so their boxes are smaller and noisier. Part of the drop may be increased difficulty
  rather than a removed shortcut. Separating these would need a distance-matched design.
- Negative anchors are drawn once with seed 42; the draw was not repeated.
