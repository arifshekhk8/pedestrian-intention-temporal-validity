# `pie_clean/` — the shipped, leak-free PIE window set

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
