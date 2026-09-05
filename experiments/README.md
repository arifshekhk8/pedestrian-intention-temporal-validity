# Experiments

Grouped by the claim each supports. Tier numbers refer to `docs/REPRODUCE.md`.

| folder | what it establishes | needs |
|---|---|---|
| `01_leakage/` | overlapping windows do not inflate the estimate; the clean 5-seed baseline | Tier 1 |
| `02_model_comparison/` | four-family comparison; ego-speed ablation; trivial baselines | Tier 1 for the three matched scripts; Tier 3 for the older per-study ones |
| `03_statistics/` | pedestrian-clustered bootstraps, LOSO, latency, detector-vs-GT robustness | Tier 3 |
| `04_observation_window/` | 32- and 64-frame windows — **read the caveat below** | Tier 3 + rebuilt tensors |
| `05_cross_dataset/` | JAAD and IDD-PeD replication and transfer | Tier 4 (external datasets) |

## The scripts that carry the headline results

These need **no checkpoint download** — they train from `data/pie_clean/` and cache into `runs/`:

| script | what it produces | runtime |
|---|---|---|
| `matched_comparison.py` | the four-family table, Holm-corrected → `MATCHED_COMPARISON.md` | ~15 min first run |
| `ego_speed_ablation.py` | 5-D vs 4-D per family → `EGO_SPEED_ABLATION.md` | ~20 min first run |
| `trivial_baselines.py` | linear baselines vs the best network → `TRIVIAL_BASELINES.md` | ~2 min |
| `tree_baselines.py` | decision tree / random forest / extra trees vs the linear reference → `TREE_BASELINES.md` | ~4 min |
| `phase_matched_control.py` | control for class-dependent timing bias → `PHASE_MATCHED_CONTROL.md` | ~20 min |
| `phase_matched_stats.py` | that control's bootstrap and Holm arms | ~8 min |

Run them in that order; each reuses the previous one's cached runs.

## Archival scripts — kept for provenance, **not runnable here**

These produced numbers that now ship as JSON and CSV under `results/`. They read source files or
cached probability tensors from the source project that were not copied into this repository, so
they cannot re-run. They are kept so the method behind each shipped number is readable.

| script | what it needs that is absent |
|---|---|
| `02_model_comparison/transformer_vs_bilstm.py` | `phase1_setup/02_train_transformer.py` |
| `02_model_comparison/gru_compare.py` | `gru/phase4_final/probs_cache/` |
| `02_model_comparison/rnn_compare.py` | `rnn/phase4_final/probs_cache/` |
| `02_model_comparison/generate_comparison_tables.py` | `pipeline/07_bilstm_attention.py` |
| `../demo/scene_demo.py` | `pipeline/11_demo_clean_ensemble.py` |

`03_statistics/*` and `04_observation_window/` import cleanly but still need the checkpoint tree
(`PCIP_CKPT_ROOT`), which is not in git and not in any release. Treat their outputs in `results/` as
the record. `05_cross_dataset/jaad_0{1,2}_*.py` need `jaad_data.py` from the JAAD repository — that
is expected, and is Tier 4.

Everything else imports and runs from this repository alone: 37 of 44 tracked scripts.

## Caveats attached to specific experiments

**`04_observation_window/`** — the window sweep is confounded with prediction horizon. The builder
ties the sliding stride to `obs_len`, so the share of windows at the maximum TTE = 60 rises
23.6 % → 46.7 % → 87.1 % across OW16/32/64, and the project's own matched-cohort ablation shows the
horizon alone costs 0.070 F1. Treat "F1 declines with window length" as unestablished; the supported
result is that the four families remain indistinguishable at OW32. See `docs/LIMITATIONS.md` §4.

To rebuild the tensors, pass an explicit output directory — never the canonical dataset path:

```bash
python src/build_windows_clean.py --obs-len 32 --out-dir build/ow32   # NOT data/pie_clean
```

**`05_cross_dataset/jaad_03_fourfamily.py`** — produces a documented null result: all 20 runs sit at
chance (AUC 0.494–0.520) and none beats a constant all-positive classifier. Kept because the negative
result is reportable; it is **not** evidence of architecture equivalence. The JAAD *leakage* audit
(`jaad_02_leakage_audit.py`) is the part that carries a positive finding.

**`02_model_comparison/generate_comparison_tables.py`** — regenerates the summary tables in
`results/model_comparison/`. One row of the shipped `model_comparison.csv` is known to be mislabelled:
the "Transformer (default, un-searched)" control cites the F1-program's B4 arm (AUC 0.942) rather
than the val-AUC-selected Stage-D default (0.9337) that actually carries the tie argument. The
authoritative number for that control is in `transformer_vs_bilstm.json`.
