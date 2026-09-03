# Results

Machine-written artifacts. Every number quoted in `README.md` and `docs/CONTRIBUTIONS.md` traces to a
file here; the mapping is tabulated at the end of `docs/REPRODUCE.md`.

Nothing in this folder is hand-edited, with one documented exception:

- `model_comparison/model_comparison.csv` — the "Transformer (default, un-searched)" row cites the
  wrong source arm (F1-program B4, AUC 0.942) instead of the val-AUC-selected Stage-D default
  (0.9337). Use `model_comparison/transformer_vs_bilstm.json` as authoritative for that control.

The **matched** four-family comparison, the ego-speed ablation and the phase-matched control live
next to their scripts in `experiments/02_model_comparison/` (`MATCHED_COMPARISON.md`,
`EGO_SPEED_ABLATION.md`, `PHASE_MATCHED_CONTROL.md`); those supersede the older per-study numbers in
`model_comparison/` for any cross-family claim.

Two generations of the phase-matched control are kept. The corrected one is
`phase_matched_trainonly_{results,stats}.json`; `phase_matched_{results,stats}.json` is the
superseded first version, which estimated its target timing distribution from all three splits.
`experiments/02_model_comparison/PHASE_RULE_LEAK_FIX.md` records the defect and compares the two.

Reports in this folder were written by the source project and quote its file paths in places. Where
those paths named a file that exists here under a different name, the reference has been corrected;
where the file was never copied over, the report says so rather than linking into nothing.

Folders map to claims: `leakage/` → C1–C3, `clean_protocol/` → C2/C4, `model_comparison/` → C5/C6,
`statistics/` → C6/C10, `observation_window/` → see LIMITATIONS §4, `cross_dataset/` → C7/C8.
