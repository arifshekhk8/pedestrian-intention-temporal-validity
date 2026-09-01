"""
02_build_sequences_clean.py  —  Journal-prep Issue 2: leak-free canonical extraction.

Replaces the leaky anchor in 02_build_sequences.py (last_annotated_frame - TTE,
arbitrary relative to the real crossing event — see ../issue1_leakage_audit/
01_leakage_report.md, 67.9% of crossers leaked) with the event-relative anchor
PIE's own benchmark code uses: PIE/utilities/data_gen_utils.py::extract_tracks_tte.

KEY DATA FACT: every pedestrian's *_attributes.xml carries a `crossing_point`
frame attribute — defined for crossers AND non-crossers alike (for non-crossers
it sits ~2 frames before the track's last annotated frame: "when the vehicle
reaches them"). Validated against the per-frame `cross` ground truth cached in
../issue1_leakage_audit/cross_state_map.pkl: crossing_point exactly equals the
first cross=="crossing" frame in 516/519 crossers (99.4%), and is never earlier
than true onset. So truncating each track at crossing_point and requiring
TTE >= 30 frames before it is leak-free by construction — this is the same
event-anchor PCPA/GTransPDM/PIP-Net-style baselines use, not a thesis invention.

ALGORITHM (mirrors extract_tracks_tte exactly, see that file for the original):
  1. Per pedestrian, take the contiguous frame segment containing crossing_point
     (drop earlier disjoint segments if the track has a gap — ~4% of peds).
  2. Truncate that segment at crossing_point (inclusive). Let L = segment length.
  3. If L < obs_len + TTE_MIN: exclude (track too short for any valid window).
  4. Else slide obs_len-frame windows with `overlap` stride, constrained so the
     last observed frame is between TTE_MIN and TTE_MAX frames before
     crossing_point (if L doesn't reach TTE_MAX, start from frame 0 instead —
     same fallback the official code uses).

Output:
  sequences_clean/X.npy    (N, obs_len, 5)   float32
  sequences_clean/y.npy    (N,)              int8
  sequences_clean/meta.pkl list of dicts {set_id, video_id, ped_id, anchor_frame, tte}
"""

import argparse
import glob
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

FEATURE_COLS = ["x1", "y1", "x2", "y2", "vehicle_speed"]


def parse_crossing_points(pie_root: Path) -> dict:
    """Return {(set_id, video_id, ped_id): crossing_point_frame:int}."""
    files = sorted(glob.glob(str(pie_root / "annotations_attributes" / "set*" / "*_attributes.xml")))
    if not files:
        raise FileNotFoundError(f"no attributes XML under {pie_root}/annotations_attributes/")
    pts = {}
    for path in files:
        m = re.search(r"(set\d+)[/\\](video_\d+)_attributes\.xml", path)
        if not m:
            continue
        set_id, video_id = m.group(1), m.group(2)
        root = ET.parse(path).getroot()
        for ped in root.findall("pedestrian"):
            pid = ped.attrib.get("id", "")
            cp = ped.attrib.get("crossing_point")
            if cp is not None:
                pts[(set_id, video_id, pid)] = int(cp)
    return pts


def split_contiguous(frames: np.ndarray) -> list[np.ndarray]:
    """Return list of index arrays, each a contiguous run (diff==1)."""
    if len(frames) == 0:
        return []
    gaps = np.where(np.diff(frames) != 1)[0] + 1
    return np.split(np.arange(len(frames)), gaps)


def build_sequences(df: pd.DataFrame, crossing_points: dict,
                     obs_len: int, tte_min: int, tte_max: int, overlap: float):
    features = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    frames_arr = df["frame"].to_numpy()

    stride = obs_len if overlap == 0 else int((1 - overlap) * obs_len)
    stride = max(stride, 1)

    X_list, y_list, meta_list = [], [], []
    skipped = defaultdict(int)
    n_multi_segment = 0

    groups = df.groupby(["set_id", "video_id", "ped_id"], sort=False)

    for (set_id, video_id, ped_id), grp in tqdm(groups, desc="Building sequences"):
        label = int(grp["crossing_label"].iloc[0])
        idx = grp.index.to_numpy()
        local_frames = frames_arr[idx]

        order = np.argsort(local_frames)
        idx_sorted = idx[order]
        frames_sorted = local_frames[order]

        cp = crossing_points.get((set_id, video_id, ped_id))
        if cp is None:
            skipped["no_crossing_point"] += 1
            continue

        segments = split_contiguous(frames_sorted)
        if len(segments) > 1:
            n_multi_segment += 1

        # Find the contiguous segment containing crossing_point; drop the rest.
        seg_pos = None
        for s in segments:
            if cp in frames_sorted[s]:
                seg_pos = s
                break
        if seg_pos is None:
            skipped["crossing_point_not_in_track"] += 1
            continue

        seg_global_idx = idx_sorted[seg_pos]
        seg_frames = frames_sorted[seg_pos]
        cp_local = int(np.where(seg_frames == cp)[0][0])

        track_idx = seg_global_idx[: cp_local + 1]   # truncate at crossing_point, inclusive
        track_frames = seg_frames[: cp_local + 1]
        L = len(track_frames)

        if L < obs_len + tte_min:
            skipped["track_too_short"] += 1
            continue

        if L < obs_len + tte_max:
            start, end = 0, L - (obs_len + tte_min) + 1
        else:
            start, end = L - (obs_len + tte_max), L - (obs_len + tte_min) + 1

        for i in range(start, end, stride):
            window_idx = track_idx[i : i + obs_len]
            window_frames = track_frames[i : i + obs_len]
            assert len(window_idx) == obs_len

            window = features[window_idx]
            assert window.shape == (obs_len, len(FEATURE_COLS))

            anchor_frame = int(window_frames[-1])
            tte = int(cp) - anchor_frame

            X_list.append(window)
            y_list.append(label)
            meta_list.append({
                "set_id": set_id,
                "video_id": video_id,
                "ped_id": ped_id,
                "anchor_frame": anchor_frame,
                "crossing_point": int(cp),
                "tte": tte,
            })

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, obs_len, 5), dtype=np.float32)
    y = np.array(y_list, dtype=np.int8)
    return X, y, meta_list, dict(skipped), n_multi_segment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", default=str(ROOT / "pie_annotations.pkl"))
    parser.add_argument("--pie-root", default=str(ROOT / "PIE"))
    parser.add_argument("--obs-len", type=int, default=16)
    parser.add_argument("--tte-min", type=int, default=30)
    parser.add_argument("--tte-max", type=int, default=60)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--out-dir", required=True,
                        help="Output directory. REQUIRED: earlier versions defaulted to the canonical\n                             dataset directory, so running with a non-default --obs-len silently\n                             overwrote the 16-frame dataset every published result depends on.")
    args = parser.parse_args()

    print(f"Loading {args.pkl} …")
    df = pd.read_pickle(args.pkl)
    print(f"  {len(df):,} rows, {df['ped_id'].nunique():,} unique pedestrians")

    print(f"Parsing crossing_point from {args.pie_root}/annotations_attributes/ …")
    crossing_points = parse_crossing_points(Path(args.pie_root))
    print(f"  {len(crossing_points):,} pedestrians with crossing_point")

    X, y, meta, skipped, n_multi_segment = build_sequences(
        df, crossing_points,
        obs_len=args.obs_len, tte_min=args.tte_min, tte_max=args.tte_max,
        overlap=args.overlap,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "X.npy"), X)
    np.save(os.path.join(args.out_dir, "y.npy"), y)
    with open(os.path.join(args.out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    total = len(y)

    print(f"\n=== Sequence Summary ===")
    print(f"X shape             : {X.shape}")
    print(f"y shape             : {y.shape}")
    print(f"y=0 (not-crossing)  : {n0:,}  ({100*n0/max(total,1):.1f}%)")
    print(f"y=1 (crossing)      : {n1:,}  ({100*n1/max(total,1):.1f}%)")
    print(f"Pedestrians with a gap in track (multi-segment): {n_multi_segment:,}")
    print(f"Saved to            : {args.out_dir}")

    if skipped:
        print(f"\nSkipped pedestrians :")
        for reason, count in skipped.items():
            print(f"  {reason}: {count:,}")
    else:
        print("\nNo pedestrians skipped.")


if __name__ == "__main__":
    main()
