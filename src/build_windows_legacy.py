"""
Build fixed-length observation windows from pie_annotations.pkl.

For each pedestrian (set_id, video_id, ped_id):
  - Split the frame series into contiguous segments (no gaps).
  - For each segment long enough, extract one 16-frame window ending
    exactly TTE=45 frames before the last annotated frame of that segment.

Output:
  sequences/X.npy    (N, obs_len, 5)   float32
  sequences/y.npy    (N,)              int8
  sequences/meta.pkl list of dicts {set_id, video_id, ped_id, anchor_frame}
"""

import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

FEATURE_COLS = ["x1", "y1", "x2", "y2", "vehicle_speed"]


def split_contiguous(frames: np.ndarray) -> list[np.ndarray]:
    """Return list of index arrays, each a contiguous run (diff==1)."""
    if len(frames) == 0:
        return []
    gaps = np.where(np.diff(frames) != 1)[0] + 1
    return np.split(np.arange(len(frames)), gaps)


def build_sequences(df: pd.DataFrame, obs_len: int, tte: int):
    features = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    frames_arr = df["frame"].to_numpy()

    X_list, y_list, meta_list = [], [], []
    skipped = defaultdict(int)  # reason → count

    groups = df.groupby(["set_id", "video_id", "ped_id"], sort=False)

    for (set_id, video_id, ped_id), grp in tqdm(groups, desc="Building sequences"):
        label = int(grp["crossing_label"].iloc[0])
        idx = grp.index.to_numpy()
        local_frames = frames_arr[idx]

        # Sort by frame within this pedestrian
        order = np.argsort(local_frames)
        idx_sorted = idx[order]
        frames_sorted = local_frames[order]

        segments = split_contiguous(frames_sorted)

        found_window = False
        for seg_pos in segments:
            if len(seg_pos) < obs_len + tte:
                # Segment too short for this TTE + obs window
                continue

            # Anchor: last frame of segment is at position len(seg_pos)-1.
            # Window ends at position (last - tte), inclusive.
            anchor_pos = len(seg_pos) - 1 - tte          # last frame of window
            window_start = anchor_pos - obs_len + 1       # first frame of window

            if window_start < 0:
                continue  # shouldn't happen given the length check, but guard

            seg_global_idx = idx_sorted[seg_pos]          # global df indices
            window_idx = seg_global_idx[window_start : anchor_pos + 1]

            window = features[window_idx]                 # (obs_len, 5)
            assert window.shape == (obs_len, len(FEATURE_COLS))

            anchor_frame = frames_sorted[seg_pos[anchor_pos]]

            X_list.append(window)
            y_list.append(label)
            meta_list.append(
                {
                    "set_id": set_id,
                    "video_id": video_id,
                    "ped_id": ped_id,
                    "anchor_frame": int(anchor_frame),
                }
            )
            found_window = True
            # One window per contiguous segment per pedestrian

        if not found_window:
            skipped["insufficient_frames"] += 1

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, obs_len, 5), dtype=np.float32)
    y = np.array(y_list, dtype=np.int8)
    return X, y, meta_list, dict(skipped)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", default="pie_annotations.pkl")
    parser.add_argument("--obs-len", type=int, default=16)
    parser.add_argument("--tte", type=int, default=45)
    parser.add_argument("--out-dir", default="sequences/")
    args = parser.parse_args()

    print(f"Loading {args.pkl} …")
    df = pd.read_pickle(args.pkl)
    print(f"  {len(df):,} rows, {df['ped_id'].nunique():,} unique pedestrians")

    X, y, meta, skipped = build_sequences(df, obs_len=args.obs_len, tte=args.tte)

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
    print(f"Saved to            : {args.out_dir}")

    if skipped:
        print(f"\nSkipped pedestrians :")
        for reason, count in skipped.items():
            print(f"  {reason}: {count:,}")
    else:
        print("\nNo pedestrians skipped.")


if __name__ == "__main__":
    main()
