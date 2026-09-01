# Contributions, and the evidence for each

Every claim below names the machine-written file that supports it and states what the surrounding
literature already established, so the novel part is separable from the confirming part.

Claims that did **not** survive audit are listed at the bottom — they are excluded from this
repository's conclusions on purpose.

---

## C1. A quantified audit of temporal contamination, on three datasets

**Claim.** Under track-end anchoring, most positive observation windows already contain the crossing
they are meant to predict; the rate is measured, not argued.

| dataset | crossing windows containing ≥1 mid-crossing frame |
|---|---|
| PIE (legacy anchor) | **387 / 570 = 67.9 %** (64.7 % fully inside; 8.4 % with ≥30 frames look-ahead) |
| JAAD (naive anchor) | **428 / 460 = 93.0 %** (90.2 % fully inside) |
| IDD-PeD (naive anchor) | **81.3 %** of crossing windows |

**Evidence.** `results/leakage/pie_legacy_per_sequence.csv`,
`results/cross_dataset/jaad_naive_leakage_report.md`,
`results/cross_dataset/idd_results/IDD_PeD_temporal_audit.csv`.

**Prior art.** PCPA (Kotseruba et al., WACV 2021) **names** this failure mode and designs a protocol
around it — so the *concept* is not new and must be cited as such. What did not exist is the
**measurement**: no published work quantifies contamination prevalence on these benchmarks, and none
measures its cost. The enabling detail is itself checkable — the standard parser reads only the
`crossing` attribute and never `crossing_point`, so the leak is structural rather than a bug.

**Also note.** PIE's official loader (`pie_data.py::_get_crossing`) already truncates tracks at the
crossing. A reviewer will raise this. The defensible residue: it is undocumented as a leakage
safeguard, it is not what the common track-end pipeline does, and **it does not fix the negative-class
anchor** (see LIMITATIONS §1).

## C2. An event-anchored protocol whose correctness is verified, not assumed

**Claim.** Anchoring on `crossing_point` with a 30–60 frame look-ahead makes contamination impossible
by construction — and the construction is checked two independent ways.

- The anchor is sound: `crossing_point` is the first `crossing`-state frame for **516/519 (99.42 %)**
  crossers and **never earlier** (min diff 0).
- The output is clean: **0 of 4,906** windows contain a crossing frame, every frame annotated.
- The dataset rebuilds **bit-exactly** from raw PIE (`np.array_equal` on X, y, meta).

**Evidence.** `src/build_windows_clean.py`, `data/pie_clean/`,
`results/leakage/pie_clean_leakage_report.md`.

## C3. Leakage and the "static geometry shortcut" are the same phenomenon

**Claim.** The box-size separability that makes this task look easy is a symptom of contamination.

Rank-biserial effect size at the anchor frame, legacy → clean: area **+0.654 → +0.247**, height
+0.629 → +0.211, bottom edge +0.485 → +0.090, x-centre −0.186 → −0.018 (n.s.). Replicated on JAAD:
+0.478 → all |rb| < 0.17.

**Evidence.** `results/leakage/*_leakage_report.md` (both produced by the same script on both datasets).

## C4. The ego-speed shortcut is partly a contamination artefact

**Claim.** The contribution of ego-speed is **three times larger** under the clean protocol than the
leaky one — the leak was masking how much the model leans on it.

| protocol | 5-D | 4-D bbox-only | gap |
|---|---|---|---|
| legacy | 0.948 ± 0.013 | 0.887 ± 0.011 | +0.06 |
| clean | 0.932 ± 0.011 | 0.753 ± 0.020 | **+0.179** |

**Evidence.** `results/clean_protocol/bilstm_multiseed_results.csv`,
`results/clean_protocol/variants_multiseed_results.csv`.

**Prior art — read carefully.** That ego-speed dominates on PIE is **settled**, reported
independently by IntFormer (2021, speed-only AUC 0.817), CAPFI (2024, speed-only AUC 0.83) and
Diving Deeper (2024). "Ego speed is a shortcut" is a quantified 2024 result with a failed fix.
**Do not claim the sign of the effect.** The unoccupied ground — and the actual contribution — is
asking whether that shortcut is a *temporal-contamination* artefact. Nobody had separated ego-speed's
predictive contribution from its leakage contribution.

## C5. The transformer's AUC advantage comes from the search, not from attention

**Claim.** Under one engine and identical data, a searched transformer beats the BiLSTM on AUC —
but the same encoder with the BiLSTM's un-searched recipe ties it exactly.

- searched vs BiLSTM: **ΔAUC +0.0135**, 10k paired CI [+0.0097, +0.0174], t-test p = 0.0249.
  Survives a pedestrian-cluster bootstrap (541 clusters): CI [+0.0068, +0.0208].
- un-searched control vs BiLSTM: **Δ +0.0005**, CI [−0.0034, +0.0043], p = 0.827.

**Evidence.** `results/model_comparison/transformer_vs_bilstm.json`.

This control is the cleanest experiment in the project: 102 search runs are provably test-free
(0/102 result files contain a `test` key), and the winner selection is re-derived from raw per-seed
files and cross-checked to 1e-9. See LIMITATIONS §3 for the capacity confound.

## C6. Cell-type isolation, structurally verified

**Claim.** LSTM, GRU and un-gated tanh RNN are indistinguishable on F1 at a 16-frame horizon.

The ablation is genuine rather than a re-implementation: parameter counts sit in an exact **4 : 3 : 1**
recurrent ratio (594,561 / 446,081 / 149,121 at h128) with identical non-recurrent parts and identical
state-dict keys — the difference really is one expression. The un-gated RNN's instability ledger is
empty (0 of 93 search runs diverged), and its search independently rediscovered the **same
configuration** the BiLSTM's own grid selected.

**Evidence.** `results/model_comparison/{gru,rnn}_comparison.json`,
`results/statistics/{gru,rnn}_cluster_bootstrap.json`.

**Prior art.** Chung et al. (2014) already showed un-gated RNNs tie gated cells over short sequences.
This is confirmation with modern statistics, not discovery — see LIMITATIONS §5.

## C7. External validity: leakage generalises; the fix does not, unmodified

**Claim (new).** On IDD-PeD the naive anchor leaks at 81.3 %. The `crossing_point` rule that reaches
0 % on PIE leaves **29.6 % residual contamination**, because IDD-PeD's `crossing_point` is a
measurably weaker onset marker. A strict `min(crossing_point, first_onset)` anchor is required.

**Evidence.** `results/cross_dataset/idd_results/IDD_PeD_temporal_audit.csv`, `table2_temporal_audit.csv`.

This is a defect the IDD-PeD paper (ICRA 2025) does not report, found by an independent parse that
recovers the authors' exact track count (4,916 = 3,284 + 1,632).

## C8. Zero-shot transfer fails, measured against trivial baselines

**Claim.** Frozen PIE models do not transfer to IDD-PeD at their operating point.

Test prevalence 7.13 % → always-positive F1 = **0.13312**. Every frozen model lands **below** it
(0.1291–0.1311). Only ranking survives: AUC 0.675–0.720 vs 0.940–0.948 in-domain; PR-AUC 0.139–0.176
against a 0.0713 chance line.

**Evidence.** `results/cross_dataset/idd_results/expA_zero_shot.{csv,json}`.

**Why this matters as novelty.** Every prior cross-dataset study on this task (including Gesnouin
et al., IV 2022) **drops ego speed**, because JAAD does not have it. The 5-D input contract had
therefore never been transfer-tested on a second dataset carrying real speed. IDD-PeD is that
dataset, and its own authors do not attempt zero-shot transfer, do not ablate their speed channel,
and do not audit their crossing annotations.

Corollary worth stating: **ego-speed dominance is a PIE property, not a universal one** — it
disappears when a model is trained on IDD-PeD.

## C9. Reproducibility as an artifact

**Claim.** A stranger regenerates the model results in ~22 CPU-minutes from a 1.7 MB tracked file,
without downloading PIE. One seed of each family takes ~79 s.

The engine is **bit-reproducible on CPU** across processes and months (verified here: val F1
0.8271604938271605 reproduced exactly seven weeks later on a different machine), and the training
function is **structurally incapable of touching the test split**.

**Evidence.** `src/engine.py`, `data/pie_clean/`, and the transcript in `docs/REPRODUCE.md`.

## C10. Negative results

Each of these is a plausible-sounding lever that does not work, measured rather than assumed:

- **F1-based checkpoint selection does not beat AUC-based** on test F1 (costs 0.008 LSTM, 0.002 transformer).
- **`pos_weight` tuning in 1.0–2.5 is not a usable lever** — validation differences (2.08e-5) are far
  below seed noise (0.012–0.019) and do not transfer to test.
- **Temporal attention adds nothing** once the leak is removed (0.925 ± 0.010 vs 0.932 ± 0.011).
- **The recipe half of a staged search bought nothing** — the winner used the default recipe
  verbatim, so 36 of 78 configs were wasted. It was purely an architecture search.
- **Window-level bootstrap CIs are anti-conservative** on this benchmark: 2,094 test windows come
  from only 541 pedestrians, and clustering widens the interval ~1.8×.

---

## Claims that did NOT survive audit

Excluded from this repository's conclusions. Listed so the exclusion is visible rather than silent.

| claim | why it fails |
|---|---|
| "The four families tie on JAAD" | All 20 runs at chance (AUC 0.494–0.520); a tie among chance models is not equivalence. |
| "Leakage inflates performance on JAAD" | The naive JAAD window set was built but never trained on. |
| "F1 declines with observation window" | Confounded with prediction horizon by the stride rule (TTE=60 share rises 23.6 % → 87.1 %). |
| "At OW64 the un-gated RNN alone falls behind" | Rests on an ensemble scoring below its own seed mean. |
| "GRU loses to the transformer on AUC but the RNN ties it" | Arm-set asymmetry — the GRU study never trained an AUC-selected large model. Not a cell effect. |
| "All verdicts survive the cluster bootstrap" | One RNN endpoint flips WIN → TIE under clustering. |
| "Removing the leakage cost nothing (Δ +0.001)" | Compares single-seed leaky to 5-seed clean. Like-for-like it is −0.016, and even that is unmatched. |
| PSI 2.0 cross-test | Abandoned: 2 of 3 scripts never written, dataset absent, and its checkpoints were trained with the test split folded in. |
