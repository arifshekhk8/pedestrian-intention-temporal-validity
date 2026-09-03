# Temporal validity in pedestrian crossing prediction

Code and evidence for a study of **temporal contamination** in the standard PIE/JAAD
crossing-prediction pipeline: how often the observation window already contains the crossing it is
supposed to predict, what that does to reported performance, and what changes once it is removed.

Everything here is reproducible from a **1.7 MB data file that ships in this repository**. You do not
need to download PIE to verify the headline numbers.

```bash
pip install -r requirements.txt
python src/engine.py --family bilstm --seed 42 --device cpu --select f1
# ~15 s on a laptop CPU → val F1 0.8271604938, val AUC 0.9516869823, best_epoch 10, 594,561 params
```

Those digits are not illustrative. They are bit-identical to the archived run this repository is
built from, reproduced on a different machine seven weeks later.

---

## The problem

The common way to build training windows for this task is to anchor them at
`(last annotated frame of the pedestrian track) − TTE`. That anchor has no connection to when the
pedestrian actually started crossing, because the parser that produces it never reads PIE's
`crossing_point` annotation.

Measured on the windows this produces (`results/leakage/pie_legacy_per_sequence.csv`):

| | |
|---|---|
| Crossing windows containing ≥1 frame where the pedestrian is **already crossing** | **387 / 570 = 67.9 %** |
| Crossing windows where **all 16** frames are mid-crossing | 369 / 570 = 64.7 % |
| Crossing windows with ≥30 frames of genuine look-ahead | **48 / 570 = 8.4 %** |
| Non-crossing windows affected | 0 / 819 |
| Median offset of the window anchor relative to crossing onset | **+182 frames (≈ 6 s after)** |

So roughly two thirds of the positive class is a **detection** problem wearing a prediction label,
and the contamination is one-sided — it falls entirely on the positive class.

This is not a newly discovered issue. Kotseruba et al. (PCPA, WACV 2021) name it and design a
protocol around it. What has not existed until now is a **measurement**: how prevalent it is, what it
costs, and whether fixing it changes any downstream conclusion.

## The fix, and how it is verified

`src/build_windows_clean.py` anchors every window on PIE's own `crossing_point`, truncates the track
there, and samples only windows ending 30–60 frames **before** onset.

The fix is checked rather than assumed, in two independent ways:

1. **The anchor is sound.** `crossing_point` equals the first frame in the `crossing` state for
   **516 / 519 crossers (99.42 %)** and is never earlier — so truncating there cannot leave crossing
   frames behind.
2. **The result is clean.** Joining the shipped windows against PIE's per-frame annotation gives
   **0 of 4,906 windows** containing a crossing frame, with every frame accounted for.

The dataset rebuilds **bit-exactly** from raw PIE (`np.array_equal` on `X`, `y` and `meta`).

## What changes once the leak is gone

**The geometric shortcut collapses.** Crossers were separable from non-crossers by box size alone.
Rank-biserial effect size at the anchor frame, legacy → clean:

| feature | legacy | clean |
|---|---|---|
| box area | +0.654 | +0.247 |
| box height | +0.629 | +0.211 |
| box bottom edge | +0.485 | +0.090 |
| box x-centre | −0.186 | −0.018 (n.s.) |

Temporal leakage and the "static shortcut" are the same phenomenon.

**Ego-speed dependence is architecture-dependent, and the biggest number is a fitting failure.**
Same family, same config, same protocol, ego-speed column removed (`ego_speed_ablation.py`):

| family | 5-D AUC | 4-D (bbox only) | drop |
|---|---|---|---|
| BiLSTM | 0.9242 | 0.7765 | +0.1477 |
| Vanilla RNN | 0.9481 | 0.8766 | +0.0715 |
| GRU | 0.9375 | 0.8836 | +0.0538 |
| Transformer | 0.9447 | **0.9291** | +0.0156 (n.s.) |

Read this against the linear baseline below: **logistic regression on the same bbox-only input scores
0.9129**. The BiLSTM's collapse to 0.7765 is therefore an *optimisation failure*, not evidence that
boxes lack signal — the information is plainly there, and a linear model extracts it.

## A second, separate sampling bias — and a control for it

The leakage fix does not touch a different problem inherited from PIE's windowing. For a
**non**-crosser, `crossing_point` is defined as the last annotated frame minus 2, so negative
windows land in the final 1–2 s of the track. For a crosser it is a real event, usually mid-track:

| class | frames from anchor to end of track (min / median / max) |
|---|---|
| non-crossers | 32 / 46 / 62 |
| crossers | 88 / 287 / 6606 |

**Zero overlap** — that nuisance variable alone separates the classes with **AUC = 1.0000**. The
model never sees it, but box growth and ego-speed encode "this pedestrian is about to be passed".

Re-sampling negatives earlier so their phase distribution matches the positives'
(`phase_matched_control.py`) reduces that separability to 0.7919 and gives:

| model | AUC: original → phase-matched |
|---|---|
| LR, ego-speed only (16 feats) | 0.9335 → **0.8309** (−0.103) |
| LR, bbox only (64 feats) | 0.9129 → **0.8979** (−0.015) |
| LR, bbox + ego-speed | 0.9488 → 0.9053 |
| Vanilla RNN | 0.9481 → 0.8872 |
| Transformer | 0.9447 → 0.8971 |

The target phase distribution is estimated from training positives only, then frozen and
applied to all three splits.

**The ego-speed advantage reverses.** Speed-only beat bbox-only on the original sampling; under
phase-matched sampling bounding boxes are clearly the stronger stream. So a substantial part of the
apparent ego-speed dominance was the *sampling artefact*, not crossing behaviour.

Full numbers and caveats — including that the two conditions have different test sets and that a
distance-based competing explanation is not excluded — in
[PHASE_MATCHED_CONTROL.md](experiments/02_model_comparison/PHASE_MATCHED_CONTROL.md).

## The result that matters most: a linear model matches everything

No published work on PIE/JAAD reports a genuinely trivial baseline — PCPA's weakest comparison is a
deep CNN on a single frame. On the leak-free protocol, fitted on train only, thresholded on
validation only, scored once on test (`trivial_baselines.py`):

| baseline | AUC | PR-AUC | F1 |
|---|---|---|---|
| majority class (always positive) | 0.5000 | 0.3252 | 0.4908 |
| **LR, bbox + ego-speed (80 raw features)** | **0.9488** | **0.9121** | **0.8546** |
| LR, ego-speed only (16 features) | 0.9335 | 0.8538 | 0.8199 |
| LR, last frame only (5 features) | 0.9251 | 0.8757 | 0.7903 |
| LR, bbox only (64 features) | 0.9129 | 0.8035 | 0.7812 |
| *best neural model (Vanilla RNN, 560k params)* | *0.9481* | *0.8925* | *0.8487* |

**Logistic regression is statistically indistinguishable from the best of four tuned neural
families** — AUC p = 0.27, PR-AUC p = 0.77, F1 p = 0.45 under a pedestrian-clustered bootstrap.

Two further facts follow. **Ego-speed alone — 16 numbers, linear — scores 0.9335**, beating the
2.24M-parameter BiLSTM on all inputs. And **five numbers from a single frame** score 0.9251, so the
temporal model is not carrying the result either.

So the leakage fix removes the shortcut that made the task look like detection, but what remains is
still largely a linear function of ego-vehicle dynamics. That is the honest state of the benchmark.

**Headline AUC does not collapse, but the honest comparison is not the flattering one.** Matched
5-seed against 5-seed the change is 0.948 → 0.932 (−0.016), and even that compares different test
sets and different class weights. The defensible statement is qualitative: the best validation epoch
moves from 3 to 17, usable windows grow 1,389 → 4,906, and the shortcut disappears.

## Model comparison under one engine

All four families train through **one training loop** (`src/engine.py`) on identical data, same
device, same seeds, same class weight, same checkpoint rule, one validation-fitted threshold each.
Test set touched once. Inference is pedestrian-clustered and Holm-corrected across all 30 tests
(`matched_comparison.py`).

| family | params | F1@tau | AUC | PR-AUC |
|---|---|---|---|---|
| BiLSTM (search winner, h256) | 2,237,313 | 0.8276 ± 0.0174 | 0.9242 ± 0.0086 | 0.8688 |
| Transformer (searched) | 794,241 | 0.8250 ± 0.0274 | 0.9447 ± 0.0090 | 0.8964 |
| GRU | 1,678,209 | 0.8419 ± 0.0083 | 0.9375 ± 0.0029 | 0.8890 |
| **Vanilla RNN (un-gated)** | **560,001** | **0.8487 ± 0.0154** | **0.9481 ± 0.0058** | 0.8925 |
| BiLSTM-h128 (baseline) | 594,561 | 0.8256 ± 0.0130 | 0.9349 ± 0.0053 | 0.8808 |

**The families do not tie.** Nine of thirty comparisons survive Holm correction, six of them with
the vanilla RNN as winner: it beats the BiLSTM (AUC, F1), the h128 baseline (AUC, PR-AUC) and the
GRU (AUC, F1). The smallest model tested is the best one.

**Transformer ≈ Vanilla RNN is a genuine tie** (AUC p = 0.83). Correction can only remove
differences, never create them, so this null is the most robust entry in the table.

Not established, and not claimed: Transformer > GRU (p_holm = 0.0546, just over the line).
Full table and caveats in [MATCHED_COMPARISON.md](experiments/02_model_comparison/MATCHED_COMPARISON.md).

**The transformer's advantage is not a search-budget artefact.** Its 78 configurations decompose
into 36 architecture + 42 recipe configs; the winner is an architecture config ranked #2 of 36, so a
matched 36-config budget selects the same model. The 42 recipe configs changed nothing.

## Does it generalise?

**JAAD — the leakage finding replicates independently.** Naive anchoring puts **428/460 (93.0 %)** of
crosser windows inside the crossing; the same geometric shortcut opens (+0.478) and the clean
protocol closes it (all |rb| < 0.17).
The JAAD *model* comparison is a **null result**: all 20 runs sit at chance (AUC 0.494–0.520) and
none beats a constant all-positive classifier. It is reported here as a negative finding, not as
evidence of family equivalence.

**IDD-PeD — leakage is worse, and the fix is not enough.** On unstructured Indian traffic the naive
anchor leaks on **81.3 %** of crossing windows. Anchoring on `crossing_point` — the rule that reaches
0 % on PIE — leaves **29.6 %** residual contamination, because IDD-PeD's `crossing_point` is a
measurably weaker onset marker. A strict `min(crossing_point, first_onset)` anchor is required to
reach 0 %.

**Zero-shot transfer fails, and is reported as such.** Test prevalence is 7.13 %, so an
always-positive classifier scores F1 = 0.1331. Every frozen PIE model scores **below** that
(0.1291–0.1311). Only *ranking* survives: AUC 0.675–0.720 against 0.940–0.948 in-domain.
Ego-speed dominance is a **PIE property, not a universal one** — it disappears when a model is
trained on IDD-PeD.

## Repository layout

```
data/pie_clean/     the 4,906-window leak-free dataset (1.7 MB, tracked) — X, y, meta
src/                the protocol and the training engine
  build_windows_clean.py   event-anchored builder (the fix)
  build_windows_legacy.py  track-end builder (the reference implementation of the bug)
  leakage_audit.py         measures contamination in any window set
  engine.py                one training loop, four families
  metrics.py               thresholds, bootstraps, split loading
experiments/        the studies, grouped by claim
results/            machine-written artifacts backing every number quoted above
demo/               YOLO26 + ByteTrack live demo (needs PIE clips; optional)
docs/               CONTRIBUTIONS, LIMITATIONS, PROTOCOL, REPRODUCE
```

## Reading order

1. [docs/PROTOCOL.md](docs/PROTOCOL.md) — the exact data contract; get this wrong and results break.
2. [docs/CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md) — each claim with the file that proves it.
3. [docs/LIMITATIONS.md](docs/LIMITATIONS.md) — what this work does **not** establish.
4. [docs/REPRODUCE.md](docs/REPRODUCE.md) — step by step, cheapest path first.

Also: [docs/bangla_walkthrough.pdf](docs/bangla_walkthrough.pdf) — the whole study explained end to
end in Bengali, with a verification log recording where every number comes from and four numbering
discrepancies found while writing it. Source: `docs/bangla_walkthrough.html`.

## A note on the task name

This work predicts PIE's per-track **`crossing`** attribute — a behavioural *outcome*. PIE also ships
`intention_prob`, a human-rated intention estimate, and the dataset authors warn that "models trained
on action labels will not be comparable to models trained on intention labels." Diving Deeper
(IV 2024) quantifies the difference: PIE intention AUC 0.65–0.70 versus action AUC 0.87–0.94.
The numbers here belong to the **action/outcome** task and should be compared against action
baselines (PCPA on PIE: Acc 0.86 / AUC 0.86 / F1 0.77), not against intention results.

## Licence and data

Code is MIT (see `LICENSE`). PIE, JAAD and IDD-PeD are redistributed by their own authors under their
own terms; this repository ships only derived window tensors from PIE's public annotations, plus the
code needed to rebuild them.
