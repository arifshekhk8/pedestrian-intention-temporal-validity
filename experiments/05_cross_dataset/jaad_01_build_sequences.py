"""
01_build_jaad_sequences.py  —  Cross-dataset validation, Track A1: JAAD sequence builder.

Mirrors journal_prep/issue2_clean_protocol/02_build_sequences_clean.py's event-anchored,
leak-free window algorithm (obs_len=16, TTE in [tte_min,tte_max], 50% overlap), adapted to
JAAD via its own `jaad_data.py` interface (generate_database()). Two build variants:

  --variant clean   Anchor each track at the first frame where the per-frame `cross`
                     behavior attribute equals "crossing" (the onset). A diagnostic run
                     against JAAD's own `crossing_point` ped-attribute (JAAD's nominal
                     analogue of PIE's crossing_point) found it does NOT reliably upper-
                     bound the true onset here: of 328 crossers with a valid crossing_point,
                     104 (31.7%) have onset < crossing_point (i.e. real crossing frames
                     occur BEFORE crossing_point, including two mismatches >80 frames) --
                     unlike PIE, where crossing_point matched true onset in 516/519 (99.4%).
                     So, unlike the PIE clean-protocol builder, we anchor directly at the
                     verified per-frame onset rather than trust the crossing_point field.
                     This is leak-free by construction for crossers. For non-crossers
                     (label 0) with no per-frame crossing tag at all, we fall back to
                     truncating 2 frames before the track's last annotated frame (mirrors
                     PIE's own non-crosser crossing_point convention: "~2 frames before the
                     track's last annotated frame").
  --variant naive   Mirrors the OLD leaky PIE anchor (last_annotated_frame - TTE, no
                     crossing-event awareness, one window per pedestrian) -- built only so
                     02_jaad_leakage_audit.py can show whether JAAD suffers the same
                     leakage class under a naive protocol, replicating Issue 1's finding on
                     a second dataset.

Feature set: bbox-only 4-D [x1,y1,x2,y2], raw pixel coords. JAAD videos are 1920x1080 (same
as PIE) -- no rescale needed. JAAD has no ego-vehicle speed (only 5 coarse vehicle-action
states: stopped/moving_slow/moving_fast/accelerating/decelerating), so there is no 5th
channel (see PLAN.md Track A).

Labels: JAAD_beh pedestrians only (id contains "b", i.e. behavior-annotated), track-level
`crossing` attribute:
  crossing == 1  -> label 1 (crosser)
  crossing == 0  -> label 0 (non-crosser)
  crossing == -1 -> DROPPED (annotator-marked irrelevant/ambiguous; mirrors PIE dropping
                    crossing_label == -1)

Split: JAAD's own official video-level split (split_ids/default/{train,val,test}.txt) --
never a random per-window split, matching PIE's set-level split discipline. All windows
from a pedestrian inherit their video's split.

Output (per variant, under <out-dir>):
  X.npy    (N, obs_len, 4)  float32
  y.npy    (N,)             int8
  meta.pkl list of dicts {video_id, ped_id, split, anchor_frame, event_frame, tte}
"""
import argparse
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
JAAD_ROOT = HERE / "JAAD"

sys.path.insert(0, str(JAAD_ROOT))
from jaad_data import JAAD  # noqa: E402

FEATURE_COLS = ["x1", "y1", "x2", "y2"]
NAIVE_TTE = 45  # mirrors sequences/ (leaky PIE) Day-5 default


def load_split_map(subset: str = "default") -> dict:
    """Return {video_id: 'train'|'val'|'test'} from JAAD's own split_ids/ files."""
    split_dir = JAAD_ROOT / "split_ids" / subset
    out = {}
    for split in ("train", "val", "test"):
        for line in (split_dir / f"{split}.txt").read_text().splitlines():
            vid = line.strip()
            if vid:
                out[vid] = split
    return out


def split_contiguous(frames: np.ndarray) -> list:
    """Return list of index arrays, each a contiguous run (diff==1)."""
    if len(frames) == 0:
        return []
    gaps = np.where(np.diff(frames) != 1)[0] + 1
    return np.split(np.arange(len(frames)), gaps)


def iter_behavior_pedestrians(db: dict):
    """Yield (video_id, ped_id, label, frames_sorted, bbox_sorted, cross_sorted) for every
    JAAD_beh pedestrian with a usable (non -1) crossing label, sorted by frame."""
    for vid, vdata in db.items():
        for pid, pdata in vdata["ped_annotations"].items():
            if "b" not in pid:
                continue
            attrs = pdata.get("attributes", {})
            crossing = attrs.get("crossing")
            if crossing is None or crossing == -1:
                continue
            frames = np.array(pdata["frames"])
            bbox = np.array(pdata["bbox"], dtype=np.float32)
            cross_beh = np.array(pdata["behavior"]["cross"])
            order = np.argsort(frames)
            yield vid, pid, int(crossing), frames[order], bbox[order], cross_beh[order]


def build_clean(db, split_map, obs_len, tte_min, tte_max, overlap):
    stride = max(1, int((1 - overlap) * obs_len)) if overlap else obs_len
    X_list, y_list, meta_list = [], [], []
    skipped = defaultdict(int)
    n_fallback_anchor = 0

    for vid, pid, label, frames_sorted, bbox_sorted, cross_sorted in tqdm(
            iter_behavior_pedestrians(db), desc="Building clean JAAD sequences"):
        split = split_map.get(vid)
        if split is None:
            skipped["video_not_in_split"] += 1
            continue

        crossing_idxs = np.where(cross_sorted == 1)[0]
        if len(crossing_idxs) > 0:
            event_frame = int(frames_sorted[crossing_idxs[0]])
        else:
            n_fallback_anchor += 1
            event_frame = int(frames_sorted[-1]) - 2

        segments = split_contiguous(frames_sorted)
        seg_pos = None
        for s in segments:
            if event_frame in frames_sorted[s]:
                seg_pos = s
                break
        if seg_pos is None:
            # fallback anchor landed outside every segment (can happen if the last
            # segment is <3 frames) -- use the last segment and re-anchor at its end.
            seg_pos = segments[-1]
            event_frame = int(frames_sorted[seg_pos][-1])

        seg_frames = frames_sorted[seg_pos]
        seg_bbox = bbox_sorted[seg_pos]
        ev_local = int(np.where(seg_frames == event_frame)[0][0])

        track_frames = seg_frames[: ev_local + 1]
        track_bbox = seg_bbox[: ev_local + 1]
        L = len(track_frames)

        if L < obs_len + tte_min:
            skipped["track_too_short"] += 1
            continue

        if L < obs_len + tte_max:
            start, end = 0, L - (obs_len + tte_min) + 1
        else:
            start, end = L - (obs_len + tte_max), L - (obs_len + tte_min) + 1

        for i in range(start, end, stride):
            window_bbox = track_bbox[i : i + obs_len]
            window_frames = track_frames[i : i + obs_len]
            if len(window_bbox) != obs_len:
                continue
            anchor_frame = int(window_frames[-1])
            tte = event_frame - anchor_frame

            X_list.append(window_bbox)
            y_list.append(label)
            meta_list.append({
                "video_id": vid, "ped_id": pid, "split": split,
                "anchor_frame": anchor_frame, "event_frame": event_frame, "tte": tte,
            })

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, obs_len, 4), dtype=np.float32)
    y = np.array(y_list, dtype=np.int8)
    return X, y, meta_list, dict(skipped), n_fallback_anchor


def build_naive(db, split_map, obs_len, tte):
    """OLD-style leaky anchor: one window per pedestrian, ending at last_frame - tte,
    with NO awareness of the crossing event. Exists only to feed the leakage audit."""
    X_list, y_list, meta_list = [], [], []
    skipped = defaultdict(int)

    for vid, pid, label, frames_sorted, bbox_sorted, _cross_sorted in tqdm(
            iter_behavior_pedestrians(db), desc="Building naive JAAD sequences"):
        split = split_map.get(vid)
        if split is None:
            skipped["video_not_in_split"] += 1
            continue

        L = len(frames_sorted)
        if L < obs_len + tte:
            skipped["track_too_short"] += 1
            continue

        end = L - tte  # exclusive end of the observation window
        start = end - obs_len
        if start < 0:
            skipped["track_too_short"] += 1
            continue

        window_bbox = bbox_sorted[start:end]
        window_frames = frames_sorted[start:end]
        anchor_frame = int(window_frames[-1])

        X_list.append(window_bbox)
        y_list.append(label)
        meta_list.append({
            "video_id": vid, "ped_id": pid, "split": split,
            "anchor_frame": anchor_frame, "event_frame": None, "tte": tte,
        })

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, obs_len, 4), dtype=np.float32)
    y = np.array(y_list, dtype=np.int8)
    return X, y, meta_list, dict(skipped)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["clean", "naive"], required=True)
    parser.add_argument("--jaad-root", default=str(JAAD_ROOT))
    parser.add_argument("--subset", default="default", choices=["default", "all_videos", "high_visibility"])
    parser.add_argument("--obs-len", type=int, default=16)
    parser.add_argument("--tte-min", type=int, default=30)
    parser.add_argument("--tte-max", type=int, default=60)
    parser.add_argument("--tte-naive", type=int, default=NAIVE_TTE)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else HERE / f"sequences_jaad_{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading JAAD database from {args.jaad_root} ...")
    imdb = JAAD(data_path=args.jaad_root)
    db = imdb.generate_database()
    split_map = load_split_map(args.subset)
    print(f"  {len(db):,} videos, {len(split_map):,} videos in the '{args.subset}' split")

    if args.variant == "clean":
        X, y, meta, skipped, n_fallback = build_clean(
            db, split_map, args.obs_len, args.tte_min, args.tte_max, args.overlap)
    else:
        X, y, meta, skipped = build_naive(db, split_map, args.obs_len, args.tte_naive)
        n_fallback = None

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    n0, n1, total = int((y == 0).sum()), int((y == 1).sum()), len(y)
    split_counts = defaultdict(lambda: {"n": 0, "pos": 0})
    for m, label in zip(meta, y):
        split_counts[m["split"]]["n"] += 1
        split_counts[m["split"]]["pos"] += int(label)

    print(f"\n=== JAAD Sequence Summary ({args.variant}) ===")
    print(f"X shape             : {X.shape}")
    print(f"y=0 (not-crossing)  : {n0:,}  ({100*n0/max(total,1):.1f}%)")
    print(f"y=1 (crossing)      : {n1:,}  ({100*n1/max(total,1):.1f}%)")
    if args.variant == "clean":
        print(f"Pedestrians using fallback anchor (no per-frame crossing tag): {n_fallback:,}")
    for split in ("train", "val", "test"):
        c = split_counts[split]
        pos_rate = 100 * c["pos"] / max(c["n"], 1)
        print(f"  {split:5s}: N={c['n']:5,d}  pos={c['pos']:5,d} ({pos_rate:.1f}%)")
    print(f"Saved to            : {out_dir}")
    if skipped:
        print("\nSkipped pedestrians :")
        for reason, count in skipped.items():
            print(f"  {reason}: {count:,}")


if __name__ == "__main__":
    main()
