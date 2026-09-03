# JAAD Leakage Audit (naive variant)

**Verdict: LEAKAGE FOUND**

Mirrors `src/leakage_audit.py`'s method; ground truth = JAAD's own per-frame `cross` behavior attribute (0=not-crossing, 1=crossing, -1=irrelevant), read directly from the JAAD repository's own `jaad_data.py::generate_database()` -- no re-parsing hack needed (unlike PIE, JAAD's own interface never dropped this field).

## Setup

- Sequences audited: **536** (crossers 460, non-crossers 76)
- Observation window: **16 frames** ending at `anchor_frame`
- Build: **obs_len=16, naive anchor = last_frame - 45 (no crossing-event awareness)**
- Leakage = >=1 frame inside the observation window with `cross == "crossing"`

## 1. Window leakage

| Group | N | sequences with >=1 crossing frame in window | % |
|---|---|---|---|
| Crossers (label=1) | 460 | 428 | 93.0% |
| Non-crossers (label=0) | 76 | 8 | 10.5% |
| **All** | 536 | 436 | 81.3% |

- Crossers with the **entire window already crossing**: **415** (90.2% of crossers).
- Crossers with a **genuinely clean window** (0 crossing frames): **32** (7.0% of crossers).
- Crossers whose **anchor frame itself** is already crossing: **421** (91.5% of crossers)
- Crossers with a labelled onset: 460; of those, onset at/before window end: **455** (98.9%)

- `anchor - onset` (frames): median 104, mean 115.3, min -30, max 535.

## 2. Static-shortcut test (anchor-frame bbox geometry)

| Feature | crosser median | non-crosser median | p | rank-biserial |
|---|---|---|---|---|
| bbox_bottom_y | 894.5 | 822.5 | 2.73e-07 | +0.368 |
| bbox_height | 272.0 | 167.0 | 7.96e-09 | +0.413 |
| bbox_xcenter | 846.0 | 1072.2 | 3.03e-01 | -0.074 |
| bbox_area | 32364.0 | 11050.0 | 2.50e-11 | +0.478 |

## 3. Interpretation

- **Window leakage present** in 428/460 crossers (93.0%). The naive last-frame-minus-TTE anchor reaches into the crossing for a large fraction of positives, replicating the Issue-1 finding on a second dataset.
- **Static-geometry shortcut:** bbox_bottom_y (r=+0.37), bbox_height (r=+0.41), bbox_area (r=+0.48) separate the classes at the anchor frame alone.