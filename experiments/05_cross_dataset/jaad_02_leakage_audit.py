"""
02_jaad_leakage_audit.py  —  Cross-dataset validation, Track A1: JAAD leakage audit.

Mirrors journal_prep/issue1_leakage_audit/01_leakage_audit.py's method, adapted to JAAD.
Unlike that script, we don't need to re-parse raw XML to recover the per-frame `cross`
ground truth -- JAAD's own `jaad_data.py` interface already preserves it (PIE's ORIGINAL
parser was the one that dropped it; that bug never existed here).

QUESTION: does a naively-anchored (last_frame - TTE) 16-frame window already contain
frames where the pedestrian is mid-crossing? If so, the task is detection of an
in-progress crossing, not prediction -- the exact Issue-1 finding on PIE. We check this
on both build variants from 01_build_jaad_sequences.py:
  - naive : expected to leak, replicating Issue 1's finding on a second dataset.
  - clean : expected to be 0% by construction (anchored at the verified per-frame onset).

WHAT WE CHECK (identical to the PIE audit)
  1. LEAKAGE: for every window, does [anchor-15 .. anchor] contain a frame with
     cross == "crossing" (scalar 1)?
  2. STATIC SHORTCUT: can crosser vs non-crosser be read off the anchor-frame bbox alone
     (bottom-y, height, x-center, area), Mann-Whitney U + rank-biserial effect size.

OUTPUTS (all written next to this script)
  <out-dir>/<variant>_leakage_per_sequence.csv
  <out-dir>/<variant>_leakage_report.md
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
JAAD_ROOT = HERE / "JAAD"
sys.path.insert(0, str(JAAD_ROOT))
from jaad_data import JAAD  # noqa: E402

OBS_LEN = 16


def build_cross_state_map(jaad_root: str) -> dict:
    """Return {(video_id, ped_id): {frame:int -> cross_state:int}} for every JAAD_beh
    pedestrian, straight from jaad_data.py's own parsed database (cross: 0=not-crossing,
    1=crossing, -1=irrelevant)."""
    imdb = JAAD(data_path=jaad_root)
    db = imdb.generate_database()
    cmap = {}
    for vid, vdata in db.items():
        for pid, pdata in vdata["ped_annotations"].items():
            if "b" not in pid:
                continue
            frames = pdata["frames"]
            cross_beh = pdata["behavior"]["cross"]
            cmap[(vid, pid)] = dict(zip(frames, cross_beh))
    return cmap


def audit_sequences(meta, y, X, cmap) -> pd.DataFrame:
    rows = []
    for i, m in enumerate(meta):
        key = (m["video_id"], m["ped_id"])
        anchor = int(m["anchor_frame"])
        label = int(y[i])
        states = cmap.get(key, {})

        window_frames = list(range(anchor - OBS_LEN + 1, anchor + 1))
        win_states = [states.get(f) for f in window_frames]
        n_crossing_in_window = sum(s == 1 for s in win_states)
        anchor_is_crossing = states.get(anchor) == 1

        crossing_frames = [f for f, s in states.items() if s == 1]
        onset = min(crossing_frames) if crossing_frames else None
        gap = (anchor - onset) if onset is not None else None

        x1, y1, x2, y2 = (float(v) for v in X[i, -1, :4])

        rows.append({
            "idx": i, "video_id": m["video_id"], "ped_id": m["ped_id"],
            "label": label, "anchor_frame": anchor,
            "n_frames_annotated_in_window": sum(s is not None for s in win_states),
            "n_crossing_in_window": n_crossing_in_window,
            "anchor_is_crossing": anchor_is_crossing,
            "window_has_leakage": n_crossing_in_window > 0,
            "crossing_onset_frame": onset,
            "anchor_minus_onset": gap,
            "bbox_bottom_y": y2, "bbox_height": y2 - y1,
            "bbox_xcenter": 0.5 * (x1 + x2),
            "bbox_area": max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)),
        })
    return pd.DataFrame(rows)


def shortcut_tests(df: pd.DataFrame) -> list:
    pos, neg = df[df.label == 1], df[df.label == 0]
    out = []
    for feat in ["bbox_bottom_y", "bbox_height", "bbox_xcenter", "bbox_area"]:
        a, b = pos[feat].to_numpy(), neg[feat].to_numpy()
        if len(a) == 0 or len(b) == 0:
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        rbc = 2.0 * u / (len(a) * len(b)) - 1.0
        out.append({"feature": feat, "crosser_median": float(np.median(a)),
                     "noncrosser_median": float(np.median(b)),
                     "mannwhitney_p": float(p), "rank_biserial": float(rbc)})
    return out


def write_report(df: pd.DataFrame, sc: list, out_path: Path, variant: str, tte_desc: str):
    n = len(df)
    pos, neg = df[df.label == 1], df[df.label == 0]
    leak_all, leak_pos, leak_neg = (int(df.window_has_leakage.sum()),
                                     int(pos.window_has_leakage.sum()),
                                     int(neg.window_has_leakage.sum()))
    anchor_cross_pos = int(pos.anchor_is_crossing.sum())
    pos_with_onset = pos.dropna(subset=["anchor_minus_onset"])
    onset_in_window = int((pos_with_onset.anchor_minus_onset >= 0).sum())
    fully_crossing = int((pos.n_crossing_in_window == OBS_LEN).sum())
    genuine_pred = int((pos.n_crossing_in_window == 0).sum())
    verdict = "CLEAN" if leak_pos == 0 else "LEAKAGE FOUND"

    lines = [f"# JAAD Leakage Audit ({variant} variant)\n",
             f"**Verdict: {verdict}**\n",
             "Mirrors `journal_prep/issue1_leakage_audit/01_leakage_audit.py`'s method; "
             "ground truth = JAAD's own per-frame `cross` behavior attribute (0=not-crossing, "
             "1=crossing, -1=irrelevant), read directly from `jaad_data.py`'s "
             "`generate_database()` -- no re-parsing hack needed (unlike PIE, JAAD's own "
             "interface never dropped this field).\n",
             "## Setup\n",
             f"- Sequences audited: **{n}** (crossers {len(pos)}, non-crossers {len(neg)})",
             f"- Observation window: **{OBS_LEN} frames** ending at `anchor_frame`",
             f"- Build: **{tte_desc}**",
             "- Leakage = >=1 frame inside the observation window with `cross == \"crossing\"`\n",
             "## 1. Window leakage\n",
             "| Group | N | sequences with >=1 crossing frame in window | % |",
             "|---|---|---|---|",
             f"| Crossers (label=1) | {len(pos)} | {leak_pos} | {100*leak_pos/max(len(pos),1):.1f}% |",
             f"| Non-crossers (label=0) | {len(neg)} | {leak_neg} | {100*leak_neg/max(len(neg),1):.1f}% |",
             f"| **All** | {n} | {leak_all} | {100*leak_all/max(n,1):.1f}% |\n",
             f"- Crossers with the **entire window already crossing**: **{fully_crossing}** "
             f"({100*fully_crossing/max(len(pos),1):.1f}% of crossers).",
             f"- Crossers with a **genuinely clean window** (0 crossing frames): "
             f"**{genuine_pred}** ({100*genuine_pred/max(len(pos),1):.1f}% of crossers).",
             f"- Crossers whose **anchor frame itself** is already crossing: **{anchor_cross_pos}** "
             f"({100*anchor_cross_pos/max(len(pos),1):.1f}% of crossers)",
             f"- Crossers with a labelled onset: {len(pos_with_onset)}; of those, onset at/before "
             f"window end: **{onset_in_window}** ({100*onset_in_window/max(len(pos_with_onset),1):.1f}%)\n"]

    if len(pos_with_onset):
        lines.append(f"- `anchor - onset` (frames): median {pos_with_onset.anchor_minus_onset.median():.0f}, "
                     f"mean {pos_with_onset.anchor_minus_onset.mean():.1f}, "
                     f"min {pos_with_onset.anchor_minus_onset.min():.0f}, "
                     f"max {pos_with_onset.anchor_minus_onset.max():.0f}.\n")

    lines.append("## 2. Static-shortcut test (anchor-frame bbox geometry)\n")
    lines.append("| Feature | crosser median | non-crosser median | p | rank-biserial |")
    lines.append("|---|---|---|---|---|")
    for r in sc:
        lines.append(f"| {r['feature']} | {r['crosser_median']:.1f} | {r['noncrosser_median']:.1f} | "
                     f"{r['mannwhitney_p']:.2e} | {r['rank_biserial']:+.3f} |")

    lines.append("\n## 3. Interpretation\n")
    if leak_pos == 0:
        lines.append("- **No window leakage.** By construction (event-anchored at the verified "
                     "per-frame onset), no observation window reaches into the crossing itself.")
    else:
        lines.append(f"- **Window leakage present** in {leak_pos}/{len(pos)} crossers "
                     f"({100*leak_pos/max(len(pos),1):.1f}%). The naive last-frame-minus-TTE anchor "
                     "reaches into the crossing for a large fraction of positives, replicating the "
                     "Issue-1 finding on a second dataset.")
    strong = [r for r in sc if abs(r["rank_biserial"]) >= 0.3]
    if strong:
        feats = ", ".join(f"{r['feature']} (r={r['rank_biserial']:+.2f})" for r in strong)
        lines.append(f"- **Static-geometry shortcut:** {feats} separate the classes at the anchor "
                     "frame alone.")
    else:
        lines.append("- **No strong static shortcut:** anchor-frame bbox geometry alone does not "
                     "separate the classes (all |rank-biserial| < 0.3).")

    out_path.write_text("\n".join(lines))
    print(f"[report] wrote {out_path}  —  VERDICT: {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["clean", "naive"], required=True)
    ap.add_argument("--seq-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=HERE)
    args = ap.parse_args()

    seq_dir = args.seq_dir or (HERE / f"sequences_jaad_{args.variant}")
    tte_desc = ("obs_len=16, TTE in [30,60], event-anchored (onset), 50% overlap" if args.variant == "clean"
                else "obs_len=16, naive anchor = last_frame - 45 (no crossing-event awareness)")

    print(f"[A] building JAAD per-frame cross-state map ...")
    cmap = build_cross_state_map(str(JAAD_ROOT))

    meta = pickle.load(open(seq_dir / "meta.pkl", "rb"))
    y = np.load(seq_dir / "y.npy")
    X = np.load(seq_dir / "X.npy")
    print(f"[B] auditing {len(meta)} sequences (X{X.shape}) from {seq_dir} ...")

    df = audit_sequences(meta, y, X, cmap)
    df.to_csv(args.out_dir / f"{args.variant}_leakage_per_sequence.csv", index=False)
    print(f"[B] wrote {args.variant}_leakage_per_sequence.csv "
          f"({int(df.window_has_leakage.sum())} flagged of {len(df)})")

    sc = shortcut_tests(df)
    write_report(df, sc, args.out_dir / f"{args.variant}_leakage_report.md", args.variant, tte_desc)


if __name__ == "__main__":
    main()
