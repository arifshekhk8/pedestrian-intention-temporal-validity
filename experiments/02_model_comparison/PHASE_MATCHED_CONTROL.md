# Phase-matched negative sampling — a control for class-dependent timing bias

Produced by `phase_matched_control.py --phase-source train`.

> **Revised.** The first version of this control estimated its target timing
> distribution from positives in every split, so validation and test timing helped place
> validation and test negatives. The rule is now estimated from training positives only
> and frozen before any negative is drawn. All numbers below are the corrected ones.
> See [`PHASE_RULE_LEAK_FIX.md`](PHASE_RULE_LEAK_FIX.md) for the defect, the fix, and the
> old-vs-new comparison. Superseded artefacts are kept under `data/pie_phase_matched/`,
> `runs/phase_matched/`, `phase_matched_results.json` and `phase_matched_stats.json`.

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
88 and clipped to what each track allows. Features, splits, engine, seeds, selection rule are
unchanged; `pos_weight` is recomputed for the new train split.

The target distribution and the floor are estimated from **training positives only**
(n = 812 of 1,648; median 268, minimum 88) and frozen to `phase_rule.json` before any
negative is drawn. The frozen rule is then applied unchanged to train, validation and
test. Each negative's placement uses only that rule and its own pedestrian's track bounds,
so no validation or test timing enters the construction.

- windows 4520 (pos 1648, neg 2872) vs 4,906 originally
- 115 negative pedestrians dropped. The script's `dropped` counter reports 114 — it only counts the
  `hi < min_to_end` branch (track too short to place an early window). One further pedestrian lost
  every window to the 16-consecutive-frames check, which `continue`s without incrementing the
  counter. Counting pedestrians present in `data/pie_clean/` but absent from the matched set
  gives 833 − 718 = **115**.
- splits 2084/563/1873, pos_weight 1.5665
- **AUC of `to_end` alone: 1.0000 → 0.7919** overall; per split, train 0.7664, val 0.8007,
  **test 0.8241**. Reduced, not eliminated — negative tracks are shorter than positive ones,
  so the distributions cannot be fully matched.

## Results

Per-seed means over five seeds. The significance tests below use the five-seed **ensemble**
on the same checkpoints, so their point estimates differ from this table — never take a
value from here and a *p*-value from there in the same sentence.

| model | AUC (orig → matched) | PR-AUC | F1 |
|---|---|---|---|
| BiLSTM | 0.9242 → **0.8923** (-0.0319) | 0.8688 → 0.7897 | 0.8276 → 0.7698 |
| Transformer | 0.9447 → **0.8971** (-0.0476) | 0.8964 → 0.7947 | 0.8250 → 0.7544 |
| GRU | 0.9375 → **0.8817** (-0.0558) | 0.8890 → 0.7684 | 0.8419 → 0.7401 |
| Vanilla RNN | 0.9481 → **0.8872** (-0.0609) | 0.8925 → 0.7749 | 0.8487 → 0.7553 |
| LR bbox + ego-speed (80) | 0.9488 → **0.9053** (-0.0435) | 0.9121 → 0.8283 | 0.8546 → 0.7621 |
| LR ego-speed only (16) | 0.9335 → **0.8309** (-0.1026) | 0.8538 → 0.6782 | 0.8199 → 0.7161 |
| LR bbox only (64) | 0.9129 → **0.8979** (-0.0150) | 0.8035 → 0.8216 | 0.7812 → 0.7407 |

## What it shows

**1. A real part of the original performance was the sampling artefact.** Every model loses
0.03–0.06 AUC, and the linear speed-only model loses 0.103.

**2. The ego-speed effect is the most affected — and its advantage reverses.**
Originally speed-only (0.9335) beat bbox-only (0.9129). Under phase-matched sampling
speed-only falls to 0.8309 while bbox-only barely moves (0.8979). **Bounding boxes are now the
more informative stream.** This directly supports the hypothesis that ego-speed was partly
encoding *"the car is about to pass this pedestrian"* rather than crossing behaviour.

**3. The linear baseline still matches every neural family** (LR 0.9053 vs best network
0.8971). That finding survives the control and is if anything stronger: no neural family
differs from the 80-feature linear model on any metric (all p_Holm = 1.0000).

**4. Architecture differences stop being resolvable.** Under the control no contrast between
the four families survives correction, against 7 of the same 18 under the standard protocol.
The per-seed AUC span narrows from 0.024 to 0.0154, but the statistical statement is the one
that carries the claim — see below.

## Significance testing (`phase_matched_stats.py`)

The table above is per-seed means. The tests below use the 5-seed **ensemble** on the same
checkpoints, with pedestrian-clustered bootstrap (B = 10,000) and Holm correction applied
separately within each arm.

### Arm A — between models, on phase-matched data. **9 of 63 survive Holm.**

**None of them is a family-vs-family contrast.** All 18 tests among BiLSTM / Transformer /
GRU / Vanilla RNN fail correction on every metric. Every survivor involves the speed-only
linear baseline:

| contrast | Δ | p_Holm |
|---|---|---|
| LR(80) − LR speed-only [PR-AUC] | +0.1500 | 0.0126 |
| LR(80) − LR speed-only [AUC] | +0.0744 | 0.0126 |
| Transformer − LR speed-only [AUC] | +0.0683 | 0.0126 |
| LR box-only − LR speed-only [AUC] | +0.0670 | 0.0126 |
| BiLSTM − LR speed-only [AUC] | +0.0635 | 0.0126 |
| Vanilla RNN − LR speed-only [AUC] | +0.0604 | 0.0126 |
| GRU − LR speed-only [AUC] | +0.0569 | 0.0126 |
| Transformer − LR speed-only [PR-AUC] | +0.1311 | 0.0224 |
| LR box-only − LR speed-only [PR-AUC] | +0.1434 | 0.0224 |

#### Arm A′ — the architecture claim, on a like-for-like basis

Arm A corrects across 63 tests over seven models; the standard-protocol run corrects across
30 over five. Survivor counts from the two are therefore not comparable as printed, and four
of the standard run's nine survivors are BiLSTM-h128 capacity contrasts rather than family
contrasts. `phase_matched_stats.py` reports `between_families_holm18`: Holm re-applied to the
**18 architecture contrasts alone**, using the same 18 tests under both protocols.

| protocol | family contrasts surviving Holm(18) |
|---|---|
| standard | **7 of 18** |
| phase-matched | **0 of 18** |

> Under the control the architecture ranking stops being resolvable. It does not vanish by a
> wide margin: the strongest remaining contrast is the Transformer over the GRU on AUC, at
> Δ = +0.0115 with p = 0.0032 against a first-step Holm threshold of 0.00278. Report it as a
> failure to resolve differences after correction, not as evidence that the families are
> equivalent.

### Arm B — standard vs phase-matched, per model. **10 of 21 survive Holm.**

Unpaired: phase-matching drops 115 negative pedestrians, so the two protocols have different
test sets (1,873 / 476 pedestrians vs 2,094 / 541). Each arm is bootstrapped over its own
pedestrians and the difference of the two independent distributions is reported — valid, but
less powerful than pairing.

All 21 tests, `*` marking the 10 that survive Holm:

| model | metric | standard → matched | Δ | 95 % CI | p_Holm | |
|---|---|---|---|---|---|---|
| Vanilla RNN | AUC | 0.9545 → 0.8913 | −0.0632 | [−0.0969, −0.0295] | 0.0042 | * |
| Vanilla RNN | PR-AUC | 0.9050 → 0.7835 | −0.1215 | [−0.2088, −0.0399] | 0.0416 | * |
| Vanilla RNN | F1 | 0.8634 → 0.7723 | −0.0911 | [−0.1490, −0.0342] | 0.0270 | * |
| Transformer | AUC | 0.9550 → 0.8992 | −0.0558 | [−0.0887, −0.0238] | 0.0192 | * |
| Transformer | PR-AUC | 0.9074 → 0.8093 | −0.0980 | [−0.1768, −0.0200] | 0.0840 | |
| Transformer | F1 | 0.8523 → 0.7480 | −0.1043 | [−0.1630, −0.0452] | 0.0108 | * |
| GRU | AUC | 0.9423 → 0.8877 | −0.0546 | [−0.0908, −0.0180] | 0.0432 | * |
| GRU | PR-AUC | 0.8924 → 0.7722 | −0.1202 | [−0.2097, −0.0326] | 0.0640 | |
| GRU | F1 | 0.8448 → 0.7629 | −0.0819 | [−0.1412, −0.0233] | 0.0572 | |
| BiLSTM | AUC | 0.9320 → 0.8944 | −0.0376 | [−0.0740, −0.0011] | 0.2180 | |
| BiLSTM | PR-AUC | 0.8768 → 0.7816 | −0.0952 | [−0.1798, −0.0080] | 0.1932 | |
| BiLSTM | F1 | 0.8313 → 0.7738 | −0.0575 | [−0.1179, +0.0025] | 0.2424 | |
| LR(80) | AUC | 0.9488 → 0.9053 | −0.0436 | [−0.0765, −0.0120] | 0.0640 | |
| LR(80) | PR-AUC | 0.9121 → 0.8283 | −0.0838 | [−0.1525, −0.0172] | 0.0816 | |
| LR(80) | F1 | 0.8546 → 0.7621 | −0.0925 | [−0.1511, −0.0336] | 0.0392 | * |
| **LR speed-only** | AUC | 0.9335 → 0.8309 | **−0.1026** | [−0.1425, −0.0632] | 0.0042 | * |
| **LR speed-only** | PR-AUC | 0.8538 → 0.6782 | **−0.1756** | [−0.2546, −0.0770] | 0.0076 | * |
| **LR speed-only** | F1 | 0.8199 → 0.7161 | **−0.1038** | [−0.1659, −0.0422] | 0.0136 | * |
| **LR box-only** | AUC | 0.9129 → 0.8979 | −0.0150 | [−0.0499, +0.0205] | 0.8143 | |
| **LR box-only** | PR-AUC | 0.8035 → 0.8216 | +0.0181 | [−0.0674, +0.1031] | 0.8143 | |
| **LR box-only** | F1 | 0.7812 → 0.7407 | −0.0405 | [−0.1011, +0.0201] | 0.5897 | |

> The reversal is tested, not asserted: **ego-speed-only degrades significantly on all
> three metrics, box-only on none.** Every box-only interval contains zero.

Two models degrade significantly on all three metrics: the speed-only baseline, and the
Vanilla RNN. Among the neural families the Vanilla RNN is the only one, and it is the family
the standard protocol ranks first — which is what one expects if its advantage came from the
artefact.

## Caveats — read before quoting these numbers

- **Residual bias remains**: `to_end` still scores 0.7919 overall and 0.8241 on the test
  split, not 0.5. Negative tracks are simply shorter than positive ones. A stricter floor
  (e.g. 158) would tighten this at the cost of ~40 % of the negative class.
- **A competing explanation is not excluded.** Negatives sampled earlier are further from the
  camera, so their boxes are smaller and noisier. Part of the drop may be increased difficulty
  rather than a removed shortcut. Separating these would need a distance-matched design.
- Negative anchors are drawn once with seed 42; the draw was not repeated.
