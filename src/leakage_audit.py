"""
01_leakage_audit.py  —  Journal-prep Issue 1: temporal leakage audit.

QUESTION A reviewer will ask:
  "AUC 0.931 converging at epoch 3 is suspiciously easy. Does the 16-frame
   observation window already contain frames where the pedestrian is crossing?
   If so, the task is not *prediction*, it is *detection* of an in-progress
   crossing, and the headline number is invalid."

This script answers that with PIE's own per-frame ground truth.

KEY DATA FACT (discovered while building this):
  The PIE box annotations carry a per-frame `cross` attribute with values
  {not-crossing, crossing, crossing-irrelevant}. The original parser
  (01_parse_annotations.py) kept only `action` (walking/standing) and DROPPED
  `cross`. `cross == "crossing"` is the exact per-frame "is actively crossing
  right now" signal we need — so we re-parse the XML here to recover it.

WHAT WE CHECK
  1. LEAKAGE: for every training sequence, does its 16-frame observation window
     [anchor-15 .. anchor] contain any frame labelled cross=="crossing"?
     - Window reaching into the crossing event  => leakage (trivial task).
     - Crossing onset strictly AFTER the window  => genuine prediction.
     We report the gap = anchor_frame - crossing_onset_frame per crosser.
  2. TRIVIAL SHORTCUT: even with no crossing-frame leakage, can the label be
     read off the *static* anchor-frame bbox (screen position / size)? We test
     crosser vs non-crosser distributions of bottom-edge y, height, x-centre,
     area with a Mann-Whitney U test + rank-biserial effect size.

OUTPUTS (all written next to this script)
  cross_state_map.pkl           cached {(set,video,ped): {frame: cross_state}}
  leakage_per_sequence.csv      one row per sequence, all audit fields
  figures/leakage_gap_hist.png  histogram of (anchor - crossing_onset)
  figures/anchor_bbox_shortcut.png  crosser vs non-crosser anchor-bbox dists
  01_leakage_report.md          human-readable findings + VERDICT

Pure pandas/numpy/scipy/matplotlib — no GPU, runs in well under a minute on an
M4 Air. Re-run is instant once cross_state_map.pkl is cached.
"""

import argparse
import glob
import pickle
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# --------------------------------------------------------------------------- #
# Paths + contract constants (must match 02_build_sequences.py / training)
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent              # journal_prep/issue1_leakage_audit
ROOT = HERE.parent.parent                           # project root
PIE_ANNT_GLOB = str(ROOT / "PIE" / "annotations" / "set*" / "*_annt.xml")
SEQ_DIR = ROOT / "sequences"
FIG_DIR = HERE / "figures"

OBS_LEN = 16        # observation window length (frames)  — training contract
TTE = 45            # time-to-event used to build sequences/ (Day-5 default)
CROSSING = "crossing"   # the per-frame state that constitutes "already crossing"


# --------------------------------------------------------------------------- #
# Part A — recover the per-frame `cross` state the original parser dropped
# --------------------------------------------------------------------------- #
def build_cross_state_map(cache: Path) -> dict:
    """Return {(set_id, video_id, ped_id): {frame:int -> cross_state:str}}.

    Cached to `cache` so re-runs are instant.
    """
    if cache.exists():
        print(f"[A] loading cached cross-state map: {cache.name}")
        with open(cache, "rb") as f:
            return pickle.load(f)

    files = sorted(glob.glob(PIE_ANNT_GLOB))
    if not files:
        raise FileNotFoundError(f"no PIE annt XML under {PIE_ANNT_GLOB}")
    print(f"[A] parsing `cross` from {len(files)} videos ...")

    import re
    cmap: dict = {}
    for path in files:
        m = re.search(r"(set\d+)[/\\](video_\d+)_annt\.xml", path)
        if not m:
            continue
        set_id, video_id = m.group(1), m.group(2)
        root = ET.parse(path).getroot()
        for track in root.findall("track[@label='pedestrian']"):
            for box in track.findall("box"):
                if box.attrib.get("outside") == "1":
                    continue
                attrs = {a.attrib["name"]: a.text for a in box.findall("attribute")}
                ped_id = attrs.get("id", "")
                cross = attrs.get("cross")
                frame = int(box.attrib["frame"])
                cmap.setdefault((set_id, video_id, ped_id), {})[frame] = cross

    with open(cache, "wb") as f:
        pickle.dump(cmap, f)
    print(f"[A] cached cross-state for {len(cmap):,} pedestrian tracks -> {cache.name}")
    return cmap


# --------------------------------------------------------------------------- #
# Part B — per-sequence leakage audit
# --------------------------------------------------------------------------- #
def audit_sequences(meta, y, X, cmap) -> pd.DataFrame:
    """One row per sequence: how much of the observation window is mid-crossing."""
    rows = []
    for i, m in enumerate(meta):
        key = (m["set_id"], m["video_id"], m["ped_id"])
        anchor = int(m["anchor_frame"])
        label = int(y[i])
        states = cmap.get(key, {})

        # The 16 observed frames end at the anchor (last observed frame).
        window_frames = list(range(anchor - OBS_LEN + 1, anchor + 1))
        win_states = [states.get(f) for f in window_frames]
        n_crossing_in_window = sum(s == CROSSING for s in win_states)
        anchor_is_crossing = states.get(anchor) == CROSSING

        # Onset of crossing within this pedestrian's whole track (first crossing frame).
        crossing_frames = [f for f, s in states.items() if s == CROSSING]
        onset = min(crossing_frames) if crossing_frames else None
        # gap > 0  => observation window reaches at/after onset  => LEAKAGE
        # gap < 0  => crossing starts after the window           => genuine prediction
        gap = (anchor - onset) if onset is not None else None

        # anchor-frame bbox in RAW pixels (X.npy is unnormalised): [x1,y1,x2,y2]
        x1, y1, x2, y2 = (float(v) for v in X[i, -1, :4])

        rows.append({
            "idx": i,
            "set_id": m["set_id"], "video_id": m["video_id"], "ped_id": m["ped_id"],
            "label": label, "anchor_frame": anchor,
            "n_frames_annotated_in_window": sum(s is not None for s in win_states),
            "n_crossing_in_window": n_crossing_in_window,
            "anchor_is_crossing": anchor_is_crossing,
            "window_has_leakage": n_crossing_in_window > 0,
            "crossing_onset_frame": onset,
            "anchor_minus_onset": gap,
            # static-shortcut features at the anchor frame
            "bbox_bottom_y": y2,
            "bbox_height": y2 - y1,
            "bbox_xcenter": 0.5 * (x1 + x2),
            "bbox_area": max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Part C — trivial static-shortcut test (crosser vs non-crosser anchor bbox)
# --------------------------------------------------------------------------- #
def shortcut_tests(df: pd.DataFrame) -> list[dict]:
    pos = df[df.label == 1]
    neg = df[df.label == 0]
    out = []
    for feat in ["bbox_bottom_y", "bbox_height", "bbox_xcenter", "bbox_area"]:
        a, b = pos[feat].to_numpy(), neg[feat].to_numpy()
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        # rank-biserial effect size: 0 = no separation, +/-1 = perfect
        rbc = 2.0 * u / (len(a) * len(b)) - 1.0
        out.append({
            "feature": feat,
            "crosser_median": float(np.median(a)),
            "noncrosser_median": float(np.median(b)),
            "mannwhitney_p": float(p),
            "rank_biserial": float(rbc),
        })
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(df: pd.DataFrame, fig_dir: Path = FIG_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Fig 1 (the money figure): how many of the 16 observed frames are already
    # 'crossing', for crossers. Bimodal spike at 16 == pure detection.
    pos = df[df.label == 1]
    counts = pos.n_crossing_in_window.value_counts().reindex(
        range(OBS_LEN + 1), fill_value=0)
    colors = ["#27ae60"] + ["#e67e22"] * (OBS_LEN - 1) + ["#c0392b"]
    plt.figure(figsize=(9, 5))
    plt.bar(counts.index, counts.values, color=colors, edgecolor="white")
    plt.title("Leakage in the observation window (570 crossers)\n"
              "x = # of the 16 observed frames already labelled `crossing`")
    plt.xlabel("crossing frames inside the 16-frame observation window")
    plt.ylabel("# crossing sequences")
    plt.xticks(range(0, OBS_LEN + 1))
    plt.annotate(f"{int(counts[0])} genuine\nprediction\n(0 leak)",
                 (0, counts[0]), textcoords="offset points", xytext=(0, 6),
                 ha="center", color="#27ae60", fontsize=9)
    plt.annotate(f"{int(counts[OBS_LEN])} fully\nmid-crossing\n(detection)",
                 (OBS_LEN, counts[OBS_LEN]), textcoords="offset points",
                 xytext=(0, 6), ha="center", color="#c0392b", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "leakage_window_crossing_frames.png", dpi=130)
    plt.close()

    # Fig 1b: gap histogram (crossers), clipped to a readable window.
    g = df.loc[df.label == 1, "anchor_minus_onset"].dropna().clip(-90, 90)
    plt.figure(figsize=(8, 5))
    plt.hist(g, bins=36, color="#c0392b", alpha=0.85, edgecolor="white")
    plt.axvline(0, color="black", lw=2, ls="--",
                label="0 = window ends at crossing onset")
    plt.title("Observation-window end vs. crossing onset (crossers, clipped ±90f)\n"
              "x > 0  => window reaches INTO the crossing  => LEAKAGE")
    plt.xlabel("anchor_frame − crossing_onset_frame  (frames; clipped)")
    plt.ylabel("# crossing sequences")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "leakage_gap_hist.png", dpi=130)
    plt.close()

    # Fig 2: static-shortcut distributions at the anchor frame
    feats = ["bbox_bottom_y", "bbox_height", "bbox_xcenter", "bbox_area"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, feat in zip(axes, feats):
        ax.boxplot([df.loc[df.label == 0, feat], df.loc[df.label == 1, feat]],
                   tick_labels=["not-cross", "cross"], showfliers=False)
        ax.set_title(feat)
    fig.suptitle("Static anchor-frame bbox: can the label be read off geometry alone?")
    plt.tight_layout()
    plt.savefig(fig_dir / "anchor_bbox_shortcut.png", dpi=130)
    plt.close()
    print(f"[fig] wrote figures -> {fig_dir}")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(df: pd.DataFrame, sc: list[dict], out_dir: Path = HERE,
                  report_name: str = "01_leakage_report.md",
                  title: str = "# Issue 1 — Temporal Leakage Audit: Report",
                  tte_desc: str = f"{TTE} frames (sequences/ default)"):
    n = len(df)
    pos = df[df.label == 1]
    neg = df[df.label == 0]

    leak_all = int(df.window_has_leakage.sum())
    leak_pos = int(pos.window_has_leakage.sum())
    leak_neg = int(neg.window_has_leakage.sum())
    anchor_cross_pos = int(pos.anchor_is_crossing.sum())

    # of crossers that ever have a crossing frame, how many onset before window end?
    pos_with_onset = pos.dropna(subset=["anchor_minus_onset"])
    onset_in_window = int((pos_with_onset.anchor_minus_onset >= 0).sum())
    # most damning single number: window is ENTIRELY mid-crossing (all 16 frames)
    fully_crossing = int((pos.n_crossing_in_window == OBS_LEN).sum())
    genuine_pred = int((pos.n_crossing_in_window == 0).sum())

    verdict = "✅ CLEAN" if leak_pos == 0 else "🔴 LEAKAGE FOUND"

    lines = []
    lines.append(f"{title}\n")
    lines.append(f"**Verdict: {verdict}**\n")
    lines.append(
        "Generated by `01_leakage_audit.py`. Ground truth = PIE per-frame `cross` "
        "attribute (values: `not-crossing` / `crossing` / `crossing-irrelevant`), "
        "re-parsed from the annotation XML because the original "
        "`01_parse_annotations.py` kept only `action` and dropped `cross`.\n")
    lines.append("## Setup\n")
    lines.append(f"- Sequences audited: **{n}** "
                 f"(crossers {len(pos)}, non-crossers {len(neg)})")
    lines.append(f"- Observation window: **{OBS_LEN} frames** ending at `anchor_frame`")
    lines.append(f"- Build TTE: **{tte_desc}**")
    lines.append("- Leakage = ≥1 frame inside the observation window with "
                 "`cross == \"crossing\"` (pedestrian already crossing while observed)\n")

    lines.append("## 1. Window leakage\n")
    lines.append("| Group | N | sequences with ≥1 crossing frame in window | % |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Crossers (label=1) | {len(pos)} | {leak_pos} | "
                 f"{100*leak_pos/max(len(pos),1):.1f}% |")
    lines.append(f"| Non-crossers (label=0) | {len(neg)} | {leak_neg} | "
                 f"{100*leak_neg/max(len(neg),1):.1f}% |")
    lines.append(f"| **All** | {n} | {leak_all} | {100*leak_all/max(n,1):.1f}% |\n")
    lines.append(f"- Crossers with the **entire 16-frame window already crossing** "
                 f"(all frames `crossing`): **{fully_crossing}** "
                 f"({100*fully_crossing/max(len(pos),1):.1f}% of crossers) — pure "
                 f"detection, not prediction.")
    lines.append(f"- Crossers with a **genuinely clean window** (0 crossing frames): "
                 f"**{genuine_pred}** ({100*genuine_pred/max(len(pos),1):.1f}% of crossers).")
    lines.append(f"- Crossers whose **anchor frame itself** is already `crossing`: "
                 f"**{anchor_cross_pos}** ({100*anchor_cross_pos/max(len(pos),1):.1f}% of crossers)")
    lines.append(f"- Crossers with a labelled crossing onset: {len(pos_with_onset)}; "
                 f"of those, onset at/before window end (`anchor−onset ≥ 0`): "
                 f"**{onset_in_window}** "
                 f"({100*onset_in_window/max(len(pos_with_onset),1):.1f}%)\n")
    if len(pos_with_onset):
        lines.append(f"- `anchor − onset` (frames): median "
                     f"{pos_with_onset.anchor_minus_onset.median():.0f}, "
                     f"mean {pos_with_onset.anchor_minus_onset.mean():.1f}, "
                     f"min {pos_with_onset.anchor_minus_onset.min():.0f}, "
                     f"max {pos_with_onset.anchor_minus_onset.max():.0f}. "
                     "Positive ⇒ window reaches into the crossing (leakage); "
                     "negative ⇒ genuine look-ahead prediction.\n")
    lines.append("See `figures/leakage_window_crossing_frames.png` (the key figure) "
                 "and `figures/leakage_gap_hist.png`.\n")

    lines.append("## 2. Static-shortcut test (anchor-frame bbox geometry)\n")
    lines.append("Can crosser vs non-crosser be separated by the *static* bounding "
                 "box at the last observed frame, with no temporal info? "
                 "Mann–Whitney U (two-sided); rank-biserial |r|: 0=none, "
                 "0.1 small, 0.3 medium, 0.5 large.\n")
    lines.append("| Feature | crosser median | non-crosser median | p | rank-biserial |")
    lines.append("|---|---|---|---|---|")
    for r in sc:
        lines.append(f"| {r['feature']} | {r['crosser_median']:.1f} | "
                     f"{r['noncrosser_median']:.1f} | {r['mannwhitney_p']:.2e} | "
                     f"{r['rank_biserial']:+.3f} |")
    lines.append("\nSee `figures/anchor_bbox_shortcut.png`.\n")

    lines.append("## 3. Interpretation\n")
    if leak_pos == 0:
        lines.append(
            "- **No window leakage.** No observation window contains a `crossing` "
            "frame, so the model never observes the crossing itself — the task is "
            "genuine ahead-of-time prediction, not detection. This is the sentence "
            "to put in the paper's Experimental Setup.")
    else:
        lines.append(
            f"- **Window leakage present** in {leak_pos}/{len(pos)} crossers "
            f"({100*leak_pos/max(len(pos),1):.1f}%). For those sequences the model "
            "sees the pedestrian already crossing, so a high AUC is partly trivial "
            "detection, not prediction. This must be fixed in Issue 2 by anchoring "
            "the window strictly *before* crossing onset (event-relative TTE).")
    strong = [r for r in sc if abs(r["rank_biserial"]) >= 0.3]
    if strong:
        feats = ", ".join(f"{r['feature']} (r={r['rank_biserial']:+.2f})" for r in strong)
        lines.append(
            f"- **Static-geometry shortcut:** {feats} separate the classes at the "
            "anchor frame on their own. Position/size is a confound; report it and "
            "consider position-invariant features (Issue 7) so the model is not "
            "just reading screen location.")
    else:
        lines.append(
            "- **No strong static shortcut:** anchor-frame bbox geometry alone does "
            "not separate the classes (all |rank-biserial| < 0.3), so the signal is "
            "in the temporal dynamics, not a static position giveaway.")
    lines.append("")
    lines.append("## 4. Files\n")
    lines.append(f"- `leakage_per_sequence.csv` — full per-sequence audit (all {n} rows)")
    lines.append("- `cross_state_map.pkl` — cached per-frame `cross` ground truth")
    lines.append("- `figures/leakage_gap_hist.png`, `figures/anchor_bbox_shortcut.png`")

    out_path = out_dir / report_name
    out_path.write_text("\n".join(lines))
    print(f"[report] wrote {out_path}  —  VERDICT: {verdict}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", type=Path, default=SEQ_DIR,
                     help="sequences/ dir to audit (default: the original leaky sequences/)")
    ap.add_argument("--out-dir", type=Path, default=HERE,
                     help="where to write the report/CSV/figures (default: this script's dir)")
    ap.add_argument("--report-name", default="01_leakage_report.md")
    ap.add_argument("--title", default="# Issue 1 — Temporal Leakage Audit: Report")
    ap.add_argument("--tte-desc", default=f"{TTE} frames (sequences/ default)",
                     help="human-readable TTE description for the report's Setup section")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cmap = build_cross_state_map(HERE / "cross_state_map.pkl")

    meta = pickle.load(open(args.seq_dir / "meta.pkl", "rb"))
    y = np.load(args.seq_dir / "y.npy")
    X = np.load(args.seq_dir / "X.npy")
    print(f"[B] auditing {len(meta)} sequences (X{X.shape}) from {args.seq_dir} ...")

    df = audit_sequences(meta, y, X, cmap)
    df.to_csv(args.out_dir / "leakage_per_sequence.csv", index=False)
    print(f"[B] wrote leakage_per_sequence.csv "
          f"({int(df.window_has_leakage.sum())} flagged of {len(df)})")

    sc = shortcut_tests(df)
    make_figures(df, fig_dir=args.out_dir / "figures")
    write_report(df, sc, out_dir=args.out_dir, report_name=args.report_name,
                 title=args.title, tte_desc=args.tte_desc)


if __name__ == "__main__":
    main()
