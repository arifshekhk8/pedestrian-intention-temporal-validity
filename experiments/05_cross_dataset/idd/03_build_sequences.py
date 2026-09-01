"""03_build_sequences.py — Phase 5: the IDD-PeD dataset adapter.

Builds leak-free observation windows from the IDD-PeD database using the SAME algorithm as
`journal_prep/issue2_clean_protocol/02_build_sequences_clean.py` (which is read but never
modified). The steps below are a line-for-line port; only the data source differs.

  1. Per pedestrian, take the contiguous frame segment containing the EVENT FRAME.
  2. Truncate that segment at the event frame (inclusive). Let L = segment length.
  3. Exclude if L < obs_len + TTE_MIN.
  4. Slide obs_len-frame windows at stride (1-overlap)*obs_len, constrained so the last
     observed frame lies TTE_MIN..TTE_MAX frames before the event frame; if L does not
     reach TTE_MAX, start from frame 0 instead (the official PIE fallback).

TWO EVENT DEFINITIONS ARE SUPPORTED (`--anchor`), because IDD-PeD's `crossing_point` is a
weaker onset marker than PIE's (audit 04: it equals the first crossing-tagged frame in
68.9 % of crossers vs PIE's 99.4 %, and is LATE in 19.0 %):

  --anchor crossing_point : event = `crossing_point`. The literal PIE port. Leaves 29.6 %
                            of crossing windows contaminated on IDD-PeD, so it is built
                            only as the disclosed sensitivity variant.
  --anchor strict         : event = min(`crossing_point`, first frame tagged CU/CFU/CD/CFD)
                            [DEFAULT]. Guarantees, BY CONSTRUCTION, that no frame at or
                            after crossing onset can enter an observation window. This is
                            the closest scientifically defensible equivalent of PIE's rule
                            — on PIE the two definitions coincide for 99.4 % of crossers,
                            so this restores the semantics PIE's `crossing_point` already
                            had rather than inventing a new one.

Features are the PIE contract in PIE order: [x1, y1, x2, y2, ego_speed], RAW pixel
coordinates (no image-size normalization) and raw km/h speed. Per-video image width and
height are carried in `meta.pkl` so any downstream coordinate rescaling (needed only for
zero-shot transfer into PIE's 1920x1080 frame) is explicit and auditable rather than
baked in here.

Writes  data/sequences_iddped_clean/{X.npy, y.npy, meta.pkl}
        results/IDD_PeD_exclusions.csv        (every excluded track + the reason)

Run from the repo root:
    python idd_ped_crossdataset/scripts/03_build_sequences.py
"""
import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FOLDER = HERE.parent
sys.path.insert(0, str(FOLDER / "src"))
from iddped_parser import CROSSING_BEHAVIOR_SCALARS  # noqa: E402

DB_PATH = FOLDER / "data" / "iddped_database.pkl"
OUT_DIR = FOLDER / "data" / "sequences_iddped_clean"
EXCL = FOLDER / "results" / "IDD_PeD_exclusions.csv"
CFG_OUT = FOLDER / "configs" / "protocol.json"

# --- splits -----------------------------------------------------------------
# The authors define train/test only. We carve validation out of the TRAINING sets at set
# granularity (never a random window split), chosen on usable-track counts alone — no model
# was run and the test sets were not consulted. See reports/temporal_protocol_IDD_PeD.md §7.
TRAIN_SETS = {"gp_set_0001", "gp_set_0004", "gp_set_0007"}
VAL_SETS = {"gp_set_0002", "gp_set_0006"}
TEST_SETS = {"gp_set_0003", "gp_set_0005", "gp_set_0008", "gp_set_0009"}


def split_of(set_id):
    if set_id in TRAIN_SETS:
        return "train"
    if set_id in VAL_SETS:
        return "val"
    if set_id in TEST_SETS:
        return "test"
    return "unassigned"


def split_contiguous(frames):
    """Return list of index arrays, each a contiguous run (diff==1). Port of the PIE helper."""
    if len(frames) == 0:
        return []
    gaps = np.where(np.diff(frames) != 1)[0] + 1
    return np.split(np.arange(len(frames)), gaps)


def event_frame(attrs, fs, cb, anchor):
    """Return the frame index the observation window must precede, or None.

    'crossing_point' — the literal PIE port.
    'strict'         — min(crossing_point, first crossing-tagged frame). Cannot be later
                       than true onset, so post-onset contamination is impossible.
    """
    cp = attrs.get("crossing_point")
    if cp is None:
        return None
    cp = int(cp)
    if anchor == "crossing_point":
        return cp
    mask = np.isin(cb, list(CROSSING_BEHAVIOR_SCALARS))
    if not mask.any():
        return cp                      # never crosses -> nothing earlier to anchor on
    return min(cp, int(fs[int(np.argmax(mask))]))


def build(db, obs_len, tte_min, tte_max, overlap, anchor="strict"):
    stride = max(obs_len if overlap == 0 else int((1 - overlap) * obs_len), 1)
    X, y, meta = [], [], []
    skipped = defaultdict(int)
    excl_rows = []
    n_multi_segment = 0

    for set_id in sorted(db):
        for vid in sorted(db[set_id]):
            v = db[set_id][vid]
            obd = v["vehicle_annotations"]
            W, H = v["width"], v["height"]

            for pid, rec in sorted(v["pedestrian_annotations"].items()):
                def drop(reason):
                    skipped[reason] += 1
                    excl_rows.append(dict(set_id=set_id, video_id=vid, ped_id=pid,
                                          split=split_of(set_id), reason=reason))

                attrs = rec.get("attributes")
                if not attrs:
                    drop("no_POI_attributes"); continue
                if "crossing" not in attrs or attrs["crossing"] is None:
                    drop("no_crossing_label"); continue
                if "crossing_point" not in attrs or attrs["crossing_point"] is None:
                    drop("no_crossing_point"); continue

                label = int(attrs["crossing"])

                frames = np.array(rec["frames"])
                order = np.argsort(frames)
                fs = frames[order]
                boxes = np.array(rec["bbox"], dtype=np.float32)[order]
                cb = np.array([-99 if c is None else c
                               for c in rec["behavior"]["CrossingBehavior"]])[order]

                cp = event_frame(attrs, fs, cb, anchor)

                segments = split_contiguous(fs)
                if len(segments) > 1:
                    n_multi_segment += 1

                seg = None
                for s in segments:
                    if cp in fs[s]:
                        seg = s
                        break
                if seg is None:
                    # covers both "crossing_point outside the annotated range" (the dataset
                    # contains a handful of corrupt values, e.g. -8506 and 65963) and
                    # "crossing_point lands inside a track gap".
                    drop("crossing_point_not_in_track"); continue

                fs_s, boxes_s, cb_s = fs[seg], boxes[seg], cb[seg]
                cp_local = int(np.where(fs_s == cp)[0][0])

                t_frames = fs_s[: cp_local + 1]
                t_boxes = boxes_s[: cp_local + 1]
                t_cb = cb_s[: cp_local + 1]
                L = len(t_frames)

                if L < obs_len + tte_min:
                    drop("track_too_short"); continue

                # ego speed must exist for every frame we might observe
                if any(int(f) not in obd for f in t_frames):
                    drop("missing_ego_speed"); continue

                # degenerate boxes are excluded rather than repaired
                if np.any((t_boxes[:, 2] <= t_boxes[:, 0]) | (t_boxes[:, 3] <= t_boxes[:, 1])):
                    drop("degenerate_bbox"); continue

                speed = np.array([obd[int(f)]["OBD_speed"] for f in t_frames], dtype=np.float32)
                feats = np.concatenate([t_boxes, speed[:, None]], axis=1)  # (L, 5)

                if L < obs_len + tte_max:
                    start, end = 0, L - (obs_len + tte_min) + 1
                else:
                    start, end = L - (obs_len + tte_max), L - (obs_len + tte_min) + 1

                for i in range(start, end, stride):
                    wi = slice(i, i + obs_len)
                    window = feats[wi]
                    if window.shape[0] != obs_len:
                        continue
                    wf = t_frames[wi]
                    anchor_f = int(wf[-1])
                    X.append(window)
                    y.append(label)
                    meta.append(dict(
                        set_id=set_id, video_id=vid, ped_id=pid,
                        split=split_of(set_id),
                        anchor_frame=anchor_f,
                        crossing_point=int(attrs["crossing_point"]),
                        event_frame=int(cp),
                        tte=int(cp) - anchor_f,
                        width=W, height=H,
                        # per-frame crossing-behaviour tags inside the window, kept so the
                        # temporal audit (04) can be run without re-parsing the XML
                        window_cb=t_cb[wi].tolist(),
                        window_first_frame=int(wf[0]),
                    ))

    X = (np.stack(X).astype(np.float32) if X
         else np.empty((0, obs_len, 5), dtype=np.float32))
    return X, np.array(y, dtype=np.int8), meta, dict(skipped), n_multi_segment, excl_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-len", type=int, default=16)
    ap.add_argument("--tte-min", type=int, default=30)
    ap.add_argument("--tte-max", type=int, default=60)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--anchor", default="strict", choices=["strict", "crossing_point"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or (str(OUT_DIR) if args.anchor == "strict"
                               else str(OUT_DIR) + "_cp_anchor")
    cfg_out = (CFG_OUT if args.anchor == "strict"
               else CFG_OUT.with_name("protocol_cp_anchor.json"))
    excl_out = (EXCL if args.anchor == "strict"
                else EXCL.with_name("IDD_PeD_exclusions_cp_anchor.csv"))

    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)["database"]

    X, y, meta, skipped, n_multi, excl_rows = build(
        db, args.obs_len, args.tte_min, args.tte_max, args.overlap, anchor=args.anchor)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", X)
    np.save(out / "y.npy", y)
    with open(out / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    excl_out.parent.mkdir(parents=True, exist_ok=True)
    with open(excl_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set_id", "video_id", "ped_id", "split", "reason"])
        w.writeheader(); w.writerows(excl_rows)

    splits = np.array([m["split"] for m in meta])
    peds = np.array([f"{m['video_id']}/{m['ped_id']}" for m in meta])
    print(f"anchor = {args.anchor}")
    print(f"X {X.shape}  y {y.shape}")
    print(f"tracks with a gap (multi-segment): {n_multi}")
    print("\nexclusions (pedestrian tracks):")
    for k, v in sorted(skipped.items(), key=lambda kv: -kv[1]):
        print(f"  {k:32s} {v:,}")
    print("\nwindows per split:")
    summary = {}
    for s in ("train", "val", "test"):
        m = splits == s
        n1 = int(y[m].sum()); n0 = int((y[m] == 0).sum())
        npd = len(np.unique(peds[m]))
        summary[s] = dict(windows=int(m.sum()), pos=n1, neg=n0,
                          pos_rate=round(n1 / max(m.sum(), 1), 4), pedestrians=npd)
        print(f"  {s:5s} windows {m.sum():6,}  pos {n1:5,}  neg {n0:6,}  "
              f"pos-rate {100*n1/max(m.sum(),1):5.1f}%  pedestrians {npd:,}")

    pw = summary["train"]["neg"] / max(summary["train"]["pos"], 1)
    print(f"\npos_weight (train neg/pos) = {pw:.4f}")

    cfg_out.parent.mkdir(parents=True, exist_ok=True)
    cfg_out.write_text(json.dumps(dict(
        anchor=args.anchor,
        obs_len=args.obs_len, tte_min=args.tte_min, tte_max=args.tte_max,
        overlap=args.overlap, feature_order=["x1", "y1", "x2", "y2", "ego_speed"],
        coordinate_space="raw pixels, per-video resolution (see meta.width/height)",
        train_sets=sorted(TRAIN_SETS), val_sets=sorted(VAL_SETS), test_sets=sorted(TEST_SETS),
        splits=summary, pos_weight=round(pw, 4),
        exclusions=skipped,
    ), indent=2))
    print(f"\nWrote {out}/  and {cfg_out}  and {excl_out}")


if __name__ == "__main__":
    main()
