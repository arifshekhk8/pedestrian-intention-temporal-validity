# The data contract

Get any of this wrong and results change silently. All of it was read out of the code, not the notes.

## Features

Exactly `[x1, y1, x2, y2, vehicle_speed]`, in **raw PIE pixel coordinates** (1920×1080) — *not*
normalised to image size. `vehicle_speed` is per-frame OBD speed in km/h, joined by frame id.

- window length: **16** consecutive frames
- standardisation: per-feature z-score using **train-split-only** mean/std computed over the
  flattened `(N*T, 5)` array, saved as `norm_mean.npy` / `norm_std.npy` in each run directory
- decision threshold: 0.5 on `sigmoid(logit)`, unless a validation-fitted τ\* is stated

## Label

PIE's per-pedestrian **`crossing`** attribute: 1 = crosses in front of the ego vehicle, 0 = does not,
−1 = not annotated for intent (dropped: 468 of 1,842 pedestrians). It is **track-level**, so every
window of a pedestrian carries the same label.

This is a behavioural *outcome* (action) label, not PIE's `intention_prob`. See the note at the end
of the main README.

## Windowing — the clean, event-anchored rule

For each pedestrian:

1. read `crossing_point` from `annotations_attributes/*_attributes.xml`
2. split the frame track into contiguous runs; keep only the run containing `crossing_point`
3. truncate that run at `crossing_point` **inclusive**
4. drop the pedestrian if the remaining length `L < obs_len + tte_min` (16 + 30 = 46)
5. slide 16-frame windows with stride `int((1 - overlap) * obs_len)` = 8 over
   `[L − (obs_len + tte_max), L − (obs_len + tte_min) + 1)`, falling back to `start = 0` when
   `L < obs_len + tte_max`

By construction `tte = crossing_point − anchor_frame ∈ [30, 60]`, so the window always ends at least
1 s before onset. This mirrors PIE's own `utilities/data_gen_utils.py::extract_tracks_tte`.

⚠ **`--out-dir` is required.** An earlier version defaulted it to the canonical dataset directory, so
running with a non-default `--obs-len` silently overwrote the 16-frame dataset that every published
result depends on. That default is removed here.

## The legacy (leaky) rule, for reference

`src/build_windows_legacy.py` anchors at `(last frame of contiguous segment) − TTE`, i.e. TTE means
*time to end of annotation*, not time to crossing. It never reads `crossing_point`. Kept as the
executable reference implementation of the bug, **not** for building new data.

## Splits

Fixed by PIE recording set, never random — a pedestrian belongs to exactly one set, so there is no
cross-split pedestrian leakage.

| split | sets | windows | pedestrians | positives |
|---|---|---|---|---|
| train | set01, set02, set04 | 2,178 | 562 | 812 (37.3 %) |
| val | set05, set06 | 634 | 164 | 155 (24.5 %) |
| test | **set03** | 2,094 | 541 | 681 (32.5 %) |

`pos_weight` = 1366 / 812 = **1.682** (the train split's own negative/positive ratio).

Note the shape of this split: test is 42.7 % of all windows, and `set05` contributes only 47 windows
from 13 pedestrians to validation — on which every early-stopping decision is made.

## Shipped dataset

`data/pie_clean/` — `X (4906, 16, 5) float32`, `y (4906,) int8` (3,258 neg / 1,648 pos),
`meta.pkl` (list of 4,906 dicts: `set_id`, `video_id`, `ped_id`, `anchor_frame`, `crossing_point`,
`tte`). 1,267 unique pedestrians, mean 3.87 windows each. TTE range [30, 60], median 44.

## Evaluation

Because windows overlap within a pedestrian, per-window metrics are not independent. Two rules follow:

1. **Bootstrap by pedestrian, not by window.** 2,094 test windows come from 541 pedestrians; a
   window-level bootstrap gives intervals roughly **1.8× too narrow**.
2. **Check per-pedestrian aggregation.** Verified here: per-window AUC 0.9131 vs per-pedestrian
   mean-probability AUC 0.9143 — a gap of −0.0012, so the overlap does **not** inflate the estimate.

Thresholds τ\* are fitted on **validation probabilities only** (`src/metrics.py::best_threshold`,
argmax F1 over unique validation probabilities, bounded [0.05, 0.95]). Never fitted on test.

## Reproducibility

- **CPU training is bit-reproducible** across processes and months. Use it.
- **`nn.LSTM` training on Apple MPS is process-history-dependent** — the same config and seed give
  different results depending on what ran earlier in the process. Recurrent runs that need exact
  reproduction must go on CPU.
- Checkpoints store numpy-scalar metrics beside the state dict, so `torch.load(..., weights_only=False)`
  is required on torch ≥ 2.6.
