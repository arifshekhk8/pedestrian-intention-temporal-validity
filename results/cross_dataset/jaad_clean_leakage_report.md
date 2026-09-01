# JAAD Leakage Audit (clean variant)

**Verdict: CLEAN**

Mirrors `journal_prep/issue1_leakage_audit/01_leakage_audit.py`'s method; ground truth = JAAD's own per-frame `cross` behavior attribute (0=not-crossing, 1=crossing, -1=irrelevant), read directly from `jaad_data.py`'s `generate_database()` -- no re-parsing hack needed (unlike PIE, JAAD's own interface never dropped this field).

## Setup

- Sequences audited: **972** (crossers 715, non-crossers 257)
- Observation window: **16 frames** ending at `anchor_frame`
- Build: **obs_len=16, TTE in [30,60], event-anchored (onset), 50% overlap**
- Leakage = >=1 frame inside the observation window with `cross == "crossing"`

## 1. Window leakage

| Group | N | sequences with >=1 crossing frame in window | % |
|---|---|---|---|
| Crossers (label=1) | 715 | 0 | 0.0% |
| Non-crossers (label=0) | 257 | 0 | 0.0% |
| **All** | 972 | 0 | 0.0% |

- Crossers with the **entire window already crossing**: **0** (0.0% of crossers).
- Crossers with a **genuinely clean window** (0 crossing frames): **715** (100.0% of crossers).
- Crossers whose **anchor frame itself** is already crossing: **0** (0.0% of crossers)
- Crossers with a labelled onset: 715; of those, onset at/before window end: **0** (0.0%)

- `anchor - onset` (frames): median -44, mean -46.6, min -60, max -30.

## 2. Static-shortcut test (anchor-frame bbox geometry)

| Feature | crosser median | non-crosser median | p | rank-biserial |
|---|---|---|---|---|
| bbox_bottom_y | 789.0 | 818.0 | 8.89e-05 | -0.165 |
| bbox_height | 148.0 | 159.0 | 6.83e-02 | -0.077 |
| bbox_xcenter | 1179.5 | 1086.5 | 2.77e-04 | +0.153 |
| bbox_area | 8109.0 | 9672.0 | 1.01e-02 | -0.108 |

## 3. Interpretation

- **No window leakage.** By construction (event-anchored at the verified per-frame onset), no observation window reaches into the crossing itself.
- **No strong static shortcut:** anchor-frame bbox geometry alone does not separate the classes (all |rank-biserial| < 0.3).