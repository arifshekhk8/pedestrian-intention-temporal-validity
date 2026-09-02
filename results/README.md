# Results

Machine-written artifacts. Every number quoted in `README.md` and `docs/CONTRIBUTIONS.md` traces to a
file here; the mapping is tabulated at the end of `docs/REPRODUCE.md`.

Nothing in this folder is hand-edited, with one documented exception:

- `model_comparison/model_comparison.csv` — the "Transformer (default, un-searched)" row cites the
  wrong source arm (F1-program B4, AUC 0.942) instead of the val-AUC-selected Stage-D default
  (0.9337). Use `model_comparison/transformer_vs_bilstm.json` as authoritative for that control.

The **matched** four-family comparison and the ego-speed ablation live next to their scripts in
`experiments/02_model_comparison/` (`MATCHED_COMPARISON.md`, `EGO_SPEED_ABLATION.md`); those two
supersede the older per-study numbers in `model_comparison/` for any cross-family claim.

Folders map to claims: `leakage/` → C1–C3, `clean_protocol/` → C2/C4, `model_comparison/` → C5/C6,
`statistics/` → C6/C10, `observation_window/` → see LIMITATIONS §4, `cross_dataset/` → C7/C8.
