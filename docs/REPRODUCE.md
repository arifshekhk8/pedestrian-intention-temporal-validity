# Reproducing the results

Ordered cheapest first. Tier 1 needs nothing but this repository.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Reference machine: Apple M4 (MacBook Air), Python 3.13.5, torch 2.12.0, CPU only.

---

## Tier 1 — verify the models (no dataset download, ~22 CPU-minutes)

The 4,906-window leak-free dataset ships in `data/pie_clean/`, so training runs immediately.

```bash
python src/engine.py --family bilstm --seed 42 --device cpu --select f1
```

Expected, **bit-exactly** (this is a regression test, not an approximation):

```
n_params    594561
best_epoch  10          auc_best_epoch 4
val.f1      0.8271604938271605
val.acc     0.9116719242902208
val.auc     0.9516869822883696
seconds     ~15
```

All four families, one seed each (~79 s total):

| `--family` | params | best epoch | val F1 | val AUC | seconds |
|---|---|---|---|---|---|
| `bilstm` | 594,561 | 10 | 0.827160 | 0.951687 | 15.0 |
| `transformer` | 794,241 | 15 | 0.876133 | 0.974719 | 41.7 |
| `gru` | 446,081 | 6 | 0.837920 | 0.959149 | 14.1 |
| `birnn` | 149,121 | 13 | 0.825083 | 0.966233 | 7.9 |

Five seeds of each reproduces the multi-seed table:

```bash
for fam in bilstm transformer gru birnn; do
  for s in 42 0 1 2 3; do
    python src/engine.py --family $fam --seed $s --device cpu --select f1 \
      --out_dir runs/$fam/seed$s
  done
done
```

⚠ Use `--device cpu`. On Apple MPS, `nn.LSTM` training is process-history-dependent and the numbers
will not reproduce (see `docs/PROTOCOL.md`).

### Tier 1b — the headline comparisons (still no download, ~35 CPU-minutes)

```bash
python experiments/02_model_comparison/matched_comparison.py    # four families, Holm-corrected
python experiments/02_model_comparison/ego_speed_ablation.py    # 5-D vs 4-D, per family
python experiments/02_model_comparison/trivial_baselines.py     # linear baselines
python experiments/02_model_comparison/tree_baselines.py        # tree ensembles vs linear
python experiments/02_model_comparison/phase_matched_control.py # timing-bias control*
python experiments/02_model_comparison/phase_matched_stats.py   # its bootstrap + Holm arms
```

*The control re-trains from the tracked `data/pie_phase_matched_trainonly/`, so it needs no download
either; only *rebuilding* that dataset requires `pie_annotations.pkl` (Tier 2), via `--annotations`.
Both scripts default to the corrected train-only artefacts. To rebuild the superseded first version,
pass `--phase-source all --out data/pie_phase_matched --runs-subdir phase_matched`; see
`experiments/02_model_comparison/PHASE_RULE_LEAK_FIX.md`.

These produce `MATCHED_COMPARISON.md`, `EGO_SPEED_ABLATION.md`, `TRIVIAL_BASELINES.md` and
`PHASE_MATCHED_CONTROL.md` beside themselves. They supersede the older per-study numbers in
`results/model_comparison/` for any cross-family claim.

## Tier 2 — verify the leakage claim (needs PIE annotations only, no video)

PIE's annotation XML is a small download; the ~1.5 GB video clips are **not** needed.

```bash
# 1. parse the annotation XML into a flat table  (→ pie_annotations.pkl, ~45 MB)
python src/parse_pie.py --pie-root /path/to/PIE

# 2. rebuild the leaky window set — the reference implementation of the bug
python src/build_windows_legacy.py --obs-len 16 --tte 45 --out-dir build/legacy

# 3. rebuild the clean window set; should be byte-identical to data/pie_clean/
python src/build_windows_clean.py --pie-root /path/to/PIE --out-dir build/clean

# 4. measure contamination in each
python src/leakage_audit.py --seq-dir build/legacy --out-dir build/audit_legacy
python src/leakage_audit.py --seq-dir build/clean  --out-dir build/audit_clean
```

Expected: legacy **387/570 (67.9 %)** crossing windows contaminated; clean **0/4906**.
Step 3 should reproduce `data/pie_clean/` exactly — verify with:

```bash
python -c "
import numpy as np
a=np.load('data/pie_clean/X.npy'); b=np.load('build/clean/X.npy')
print('identical:', np.array_equal(a,b))"
```

## Tier 3 — the older per-study analyses (**archival: not runnable here**)

⚠ **Read this before trying.** These scripts are kept as provenance for numbers that are already
shipped as JSON and CSV under `results/`. They are **not reproducible from this repository**, for two
independent reasons, and no release currently exists that would fix it:

1. **The checkpoints are not in git** — 757 MB of trained models, plus the cached probability
   tensors (`probs_cache/`) the comparison scripts read. Nothing here provides them.
2. **Five of the scripts read source files that were never copied over** from the source project.
   They are listed in `experiments/README.md` under "Archival scripts".

The scripts affected are `experiments/03_statistics/*`, `transformer_vs_bilstm.py`, `gru_compare.py`,
`rnn_compare.py`, `generate_comparison_tables.py` and `experiments/04_observation_window/`.

**What to do instead.** Every claim these scripts support is already backed by a machine-written file
in `results/`, listed in the table at the end of this document. For anything cross-family, the Tier 1b
scripts above are the authoritative, fully reproducible replacement — they retrain from
`data/pie_clean/` and need no checkpoints.

⚠ Separately: the published F1 arm could not be regenerated even with the sources, because all 65 of
its runs trained on Apple MPS, where recurrent training is not context-free. See
`docs/LIMITATIONS.md` §6.

## Tier 4 — cross-dataset (needs external datasets)

- **JAAD** — clone `ykotseruba/JAAD`, then
  `experiments/05_cross_dataset/jaad_01_build_sequences.py` → `jaad_02_leakage_audit.py`.
  The leakage audit is the part worth reproducing; the four-family runs are a documented null result.
- **IDD-PeD** — `experiments/05_cross_dataset/idd/00_download_iddped.sh` (CC BY 4.0, direct
  download), then `01_build_database.py` → `04_temporal_audit.py` → `05_zero_shot_transfer.py`.

## Tier 5 — live demo (needs PIE video clips + YOLO weights)

```bash
python demo/live_demo.py --stage verify     # parity gate: refuses to run unless the
                                            # ensemble matches the published table to 1e-4
python demo/live_demo.py --stage demo --video <clip.mp4> --video-id video_0012 \
       --start-frame 7676 --max-frames 900 --dump-csv
```

The `verify` stage is the useful part for a reviewer: it re-scores the clean test split through the
5-checkpoint ensemble and hard-fails unless AUC/F1/accuracy match the published values.

Note before quoting demo output: the deployed ensemble correlates with ego speed at r = −0.892 and
flags 96.2 % of pedestrians when the vehicle is stopped (`docs/LIMITATIONS.md` §9).

## What each results file backs

| file | claim |
|---|---|
| `results/leakage/pie_legacy_per_sequence.csv` | 67.9 % contamination, per-window |
| `results/leakage/pie_clean_leakage_report.md` | 0/4906 after the fix |
| `results/clean_protocol/bilstm_multiseed_results.csv` | clean 5-seed AUC 0.932 ± 0.011 |
| `results/clean_protocol/variants_multiseed_results.csv` | the *superseded* bbox-only arm (0.753 ± 0.020). Provenance is mixed — see EGO_SPEED_ABLATION.md |
| `experiments/02_model_comparison/ego_speed_ablation_results.json` | the matched ego-speed ablation (5-seed mean: +0.0156 to +0.1477 by family) |
| `experiments/02_model_comparison/trivial_baselines_results.json` | logistic regression matches all four neural families |
| `experiments/02_model_comparison/tree_baselines_results.json` | tree ensembles lose to the linear reference, even after a search it never got |
| `experiments/02_model_comparison/matched_comparison_results.json` | the matched four-family comparison, Holm-corrected |
| `experiments/02_model_comparison/phase_matched_trainonly_results.json` | the timing-bias control; ego-speed advantage reverses |
| `experiments/02_model_comparison/phase_matched_trainonly_stats.json` | its bootstrap and Holm arms; 0 of 18 family contrasts survive |
| `experiments/02_model_comparison/phase_matched_{results,stats}.json` | **superseded** first version of the control — see `PHASE_RULE_LEAK_FIX.md` |
| `results/clean_protocol/eval_parity_report.md` | overlapping windows do not inflate the estimate |
| `results/model_comparison/transformer_vs_bilstm.json` | ΔAUC +0.0135 and the un-searched control tie |
| `results/model_comparison/f1_final_arms.json` | the F1 headline arms and τ\* |
| `results/statistics/*_cluster_bootstrap.json` | pedestrian-clustered intervals |
| `results/statistics/matched_cohort_tte_ablation.csv` | the horizon effect that confounds the window sweep |
| `results/cross_dataset/jaad_naive_leakage_report.md` | 93.0 % contamination on JAAD |
| `results/cross_dataset/idd_results/table2_temporal_audit.csv` | 81.3 % → 29.6 % → 0.0 % on IDD-PeD |
| `results/cross_dataset/idd_results/expA_zero_shot.csv` | zero-shot below the always-positive baseline |
