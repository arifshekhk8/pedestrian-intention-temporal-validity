This folder ships two datasets. Both are small enough to track in git so the results can be
verified without downloading PIE.

---

# `pie_clean/` — the leak-free PIE window set

1.7 MB, tracked in git so the model results can be verified without downloading PIE.

| file | shape / type | contents |
|---|---|---|
| `X.npy` | `(4906, 16, 5)` float32 | `[x1, y1, x2, y2, vehicle_speed]`, raw 1920×1080 pixels, km/h |
| `y.npy` | `(4906,)` int8 | 3,258 negative / 1,648 positive (33.6 % positive) |
| `meta.pkl` | list of 4,906 dicts | `set_id, video_id, ped_id, anchor_frame, crossing_point, tte` |

1,267 unique pedestrians, mean 3.87 windows each. TTE ∈ [30, 60] frames, median 44.

## Provenance

Built by `src/build_windows_clean.py` from PIE's public annotations (spatial XML +
`annotations_attributes` + `annotations_vehicle` OBD speed). Every window is anchored on the
annotated `crossing_point` and ends 30–60 frames **before** crossing onset.

Independently verified: **0 of 4,906 windows** contain a frame in which the pedestrian is already
crossing, and the file rebuilds **bit-exactly** from raw PIE (`np.array_equal` on X, y and meta).

## Licence

Derived from the PIE dataset (Rasouli et al., ICCV 2019), redistributed by its authors at
<https://data.nvision2.eecs.yorku.ca/PIE_dataset/>. These are derived numeric window tensors from the
public annotation files — no imagery is included. Cite PIE if you use them.


---

# `pie_phase_matched/` — the class-dependent-timing control set

4,520 windows (1,648 positive / 2,872 negative), 1.4 MB. Same files, same 5-D
feature contract, same recording-set splits (2084/563/1873),
`pos_weight` 1.5665. `meta.pkl` carries an extra `to_end` field.

## Why it exists

In `pie_clean/`, negatives are anchored in the last 1–2 s of their track (PIE defines
`crossing_point` for a non-crosser as the last annotated frame minus 2) while positives sit
mid-track. Frames-to-track-end therefore separates the two classes with **AUC = 1.0000**, with zero
overlap — a class-dependent sampling bias distinct from the temporal leakage the clean protocol
fixes.

This set re-samples negatives earlier, drawing frames-to-track-end from the positive empirical
distribution (floor 88 = the positive minimum, clipped to each track's length, seed 42). Positives
are untouched. 115 negative pedestrians are dropped (833 → 718): 114 because their tracks are too
short to place an early window, and one more that loses every window to the 16-consecutive-frames
check. The script prints 114 because its counter tracks only the first cause.

Separability of `to_end` falls from 1.0000 to **0.7779** — reduced, not eliminated,
because negative tracks are simply shorter than positive ones.

Built by `experiments/02_model_comparison/phase_matched_control.py`. Results and caveats in
`experiments/02_model_comparison/PHASE_MATCHED_CONTROL.md`.
