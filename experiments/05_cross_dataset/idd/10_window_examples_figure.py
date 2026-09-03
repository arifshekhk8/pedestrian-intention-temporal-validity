"""10_window_examples_figure.py — Phase 12, figure 1: valid vs invalid observation windows.

A worked example on REAL IDD-PeD tracks showing, on a frame timeline, why the naive
track-end anchor and the literal `crossing_point` anchor admit contaminated windows and the
strict anchor does not. Every element is read from the annotations — nothing is schematic.

Panel (a): a crossing track where `crossing_point` is LATE (the 19 % case). The naive window
           sits inside the crossing; the crossing_point window still clips it; the strict
           window is clean.
Panel (b): a well-annotated crossing track (`crossing_point` == onset) — all three event
           anchors agree, only the naive one leaks.
Panel (c): the modal failure — a crossing track whose annotation BEGINS at the crossing
           point, so no valid pre-crossing window exists at all and the track is excluded.

Writes  figures/fig0_window_examples.png
        results/window_examples.json   (the exact tracks used, so the figure is auditable)

Run from the repo root:
    python idd_ped_crossdataset/scripts/10_window_examples_figure.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
FOLDER = HERE.parent
sys.path.insert(0, str(HERE / "lib"))
from iddped_parser import CROSSING_BEHAVIOR_SCALARS as CB  # noqa: E402

DB = FOLDER / "data" / "iddped_database.pkl"
OBS, TMIN, TMAX = 16, 30, 60


def tracks(db):
    for sid in sorted(db):
        for vid in sorted(db[sid]):
            for pid, rec in sorted(db[sid][vid]["pedestrian_annotations"].items()):
                a = rec.get("attributes")
                if not a or a.get("crossing") is None or a.get("crossing_point") is None:
                    continue
                fs = np.array(rec["frames"])
                o = np.argsort(fs)
                cb = np.array([-99 if c is None else c
                               for c in rec["behavior"]["CrossingBehavior"]])[o]
                yield dict(sid=sid, vid=vid, pid=pid, label=int(a["crossing"]),
                           cp=int(a["crossing_point"]), frames=fs[o], cb=cb)


def onset_of(t):
    m = np.isin(t["cb"], list(CB))
    return int(t["frames"][int(np.argmax(m))]) if m.any() else None


def pick(db):
    """Choose one real track for each of the three cases."""
    late = well = nolead = None
    for t in tracks(db):
        if t["label"] != 1:
            continue
        on = onset_of(t)
        if on is None:
            continue
        fs = t["frames"]
        lead_cp = int(np.sum(fs <= t["cp"]))
        lead_strict = int(np.sum(fs <= min(t["cp"], on)))
        d = t["cp"] - on
        if late is None and d >= 60 and lead_strict >= OBS + TMIN and len(fs) >= 200:
            late = t
        if well is None and d == 0 and lead_strict >= OBS + TMAX and len(fs) >= 200:
            well = t
        if nolead is None and t["cp"] == int(fs[0]) and len(fs) >= 120:
            nolead = t
        if late and well and nolead:
            break
    return late, well, nolead


def draw(ax, t, title):
    fs = t["frames"]
    on = onset_of(t)
    cp = t["cp"]
    strict = min(cp, on) if on is not None else cp
    f0, f1 = int(fs[0]), int(fs[-1])

    # crossing-state ribbon
    iscross = np.isin(t["cb"], list(CB))
    ax.add_patch(Rectangle((f0, 0.72), f1 - f0, 0.16, fc="#ecf0f1", ec="none"))
    runs = np.split(np.arange(len(fs)), np.where(np.diff(iscross.astype(int)) != 0)[0] + 1)
    for r in runs:
        if iscross[r[0]]:
            ax.add_patch(Rectangle((fs[r[0]], 0.72), fs[r[-1]] - fs[r[0]] + 1, 0.16,
                                   fc="#c0392b", ec="none"))
    ax.text(f0, 0.90, "pedestrian crossing state  (red = already crossing)", fontsize=7.5,
            va="bottom")

    # event markers — stagger the labels so coincident events stay legible
    marks = [(on, "#c0392b", "true onset"), (cp, "#8e44ad", "crossing_point"),
             (strict, "#27ae60", "strict event = min(cp, onset)")]
    span = max(f1 - f0, 1)
    for i, (f, c, lab) in enumerate(marks):
        if f is None:
            continue
        ax.axvline(f, color=c, ls="--", lw=1.4, ymin=0.05, ymax=0.95)
        ax.annotate(f"{lab} = {f}", xy=(f, 0.97),
                    xytext=(f + 0.012 * span, 1.155 - 0.052 * i),
                    fontsize=7.5, color=c, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.7, alpha=.6))

    # the three candidate windows (last observed frame = anchor)
    rows = []
    naive_anchor = f1 - 45
    if naive_anchor - OBS + 1 >= f0:
        rows.append(("naive (track-end − 45)", naive_anchor, "#c0392b"))
    for name, ev, col in (("crossing_point anchor", cp, "#8e44ad"),
                          ("strict anchor", strict, "#27ae60")):
        lead = int(np.sum(fs <= ev))
        if lead >= OBS + TMIN:
            anchor = int(fs[lead - 1 - TMIN])
            rows.append((name, anchor, col))
        else:
            rows.append((name + "  →  NO VALID WINDOW", None, col))

    for i, (name, anchor, col) in enumerate(rows):
        y = 0.52 - i * 0.20
        if anchor is None:
            ax.text(f0, y + 0.02, f"✗ {name}", fontsize=8, color=col, va="bottom")
            continue
        start = anchor - OBS + 1
        seg = (fs >= start) & (fs <= anchor)
        n_leak = int(np.isin(t["cb"][seg], list(CB)).sum())
        ok = n_leak == 0
        ax.add_patch(Rectangle((start, y), OBS, 0.13, fc=col, alpha=.85, ec="k", lw=.6))
        ax.text(f0, y + 0.145,
                f"{'✓' if ok else '✗'} {name} — window [{start}, {anchor}], "
                f"{n_leak}/16 frames already crossing",
                fontsize=8, color=("#1e8449" if ok else "#922b21"), va="bottom")

    ax.set_xlim(f0 - 5, f1 + 5)
    ax.set_ylim(-0.08, 1.30)
    ax.set_yticks([])
    ax.set_xlabel("video frame index")
    ax.set_title(f"{title}\n{t['vid']} · ped {t['pid']} · "
                 f"track [{f0}, {f1}] · crossing_point {cp} · onset {on}", fontsize=9)
    ax.grid(axis="x", alpha=.25)


def main():
    with open(DB, "rb") as f:
        db = pickle.load(f)["database"]
    late, well, nolead = pick(db)

    picked = {}
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    for ax, t, title in (
            (axes[0], late,
             "(a) LATE crossing_point (the 19 % case) — only the strict anchor is clean"),
            (axes[1], well,
             "(b) Well-annotated crossing track (crossing_point == onset) — event anchors agree"),
            (axes[2], nolead,
             "(c) The modal failure: annotation BEGINS at the crossing point — no valid "
             "pre-crossing window exists, track excluded")):
        if t is None:
            ax.text(0.5, 0.5, "no matching track found", ha="center")
            continue
        draw(ax, t, title)
        picked[title[:3]] = dict(video=t["vid"], ped=t["pid"], label=t["label"],
                                 first=int(t["frames"][0]), last=int(t["frames"][-1]),
                                 crossing_point=t["cp"], onset=onset_of(t))

    handles = [Patch(fc="#c0392b", label="naive (track-end) anchor"),
               Patch(fc="#8e44ad", label="crossing_point anchor (literal PIE port)"),
               Patch(fc="#27ae60", label="strict anchor (this study)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("IDD-PeD: valid vs invalid observation windows (obs_len 16, TTE ≥ 30)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.035, 1, 0.98])
    fig.savefig(FOLDER / "figures" / "fig0_window_examples.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    (FOLDER / "results" / "window_examples.json").write_text(json.dumps(picked, indent=2))
    print("Wrote figures/fig0_window_examples.png")
    print(json.dumps(picked, indent=2))


if __name__ == "__main__":
    main()
