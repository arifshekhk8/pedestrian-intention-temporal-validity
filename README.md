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

**Ego-speed dependence turns out to be far larger than the leaky protocol suggested.**

| protocol | 5-D (bbox + ego-speed) | 4-D (bbox only) | gap |
|---|---|---|---|
| legacy (leaky) | 0.948 ± 0.013 | 0.887 ± 0.011 | +0.06 |
| clean | **0.932 ± 0.011** | **0.753 ± 0.020** | **+0.179** |

That ego-speed dominates on PIE is already established (IntFormer 2021; CAPFI 2024; Diving Deeper
2024). What is new here is that **the leak was masking how much the model leans on it** — the
question of whether the ego-speed shortcut is itself a contamination artefact had not been asked.

**Headline AUC does not collapse, but the honest comparison is not the flattering one.** Matched
5-seed against 5-seed the change is 0.948 → 0.932 (−0.016), and even that compares different test
sets and different class weights. The defensible statement is qualitative: the best validation epoch
moves from 3 to 17, usable windows grow 1,389 → 4,906, and the shortcut disappears.

## Model comparison under one engine

All four families train through **one training loop** (`src/engine.py`) on identical data, so a
cross-family difference cannot be a protocol difference. Test set (`set03`) is touched once.

| family | params | test AUC | ensemble F1 | CPU latency (ms/window) |
|---|---|---|---|---|
| BiLSTM | 594,561 | 0.932 ± 0.011 | 0.8557 | 0.575 |
| Transformer (searched) | 794,241 | **0.9497 ± 0.0025** | 0.8565 | 0.459 |
| GRU | 446,081 | 0.941 ± 0.007 | **0.8628** | 0.721 |
| Vanilla RNN (un-gated) | 149,121 | 0.948 ± 0.001 | 0.8590 | **0.316** |

**The transformer's AUC win comes from the search, not from attention.** ΔAUC = +0.0135, 10k paired
bootstrap CI [+0.0097, +0.0174]; it survives a pedestrian-cluster bootstrap at [+0.0068, +0.0208].
But the *same encoder* trained with the BiLSTM's un-searched recipe ties the BiLSTM exactly:
Δ +0.0005, CI [−0.0034, +0.0043], p = 0.827. That control is the cleanest result in the project.

On **F1**, the families are indistinguishable. See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for why
"indistinguishable" is a weaker statement than it looks.

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
