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

**External corroboration of the mechanism.** GTransPDM (Xie et al., IEEE SPL 2025), Fig. 4, sweeps
the time-to-event gap and reports that "as TTE decreased, the model's performance improved rapidly.
**Without a TTE interval (TTE = 0), the accuracy and F1 score reached 99.32 % and 98.73 %.**"
That is a paper with no stake in this argument showing the task collapses to near-perfect
solvability once the gap closes. The audit here explains why.

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

## C4. A linear model on raw features matches four tuned neural families

**Claim.** On the leak-free protocol, logistic regression on the 80 raw window features scores
AUC 0.9488 / PR-AUC 0.9121 / F1 0.8546 and is **statistically indistinguishable** from the best of
four searched neural families (pedestrian-clustered bootstrap: AUC p = 0.27, PR-AUC p = 0.77,
F1 p = 0.45). Ego-speed alone (16 features, linear) scores 0.9335 — above the 2.24M-parameter
BiLSTM. Five numbers from a single frame score 0.9251.

**Evidence.** `experiments/02_model_comparison/trivial_baselines_results.json`.

**Why it is novel.** The field reports no trivial baseline: PCPA's weakest comparison is a deep CNN
on a single frame, and no paper reports majority-class, speed-only, or a linear model on the raw
window. This is the gap, measured.

**Why it matters.** It bounds every architecture claim in this literature, including our own. It
also reframes the leakage result: removing the contamination removes the *detection* shortcut, but
what remains is still largely a linear function of ego-vehicle dynamics.

## C4b. Ego-speed's contribution is architecture-dependent

**Claim.** Removing ego-speed costs +0.0126 AUC (Transformer, n.s.) to +0.1420 (BiLSTM) on the
bootstrap-tested ensemble delta; on the 5-seed mean the same range is +0.0156 to +0.1477. Quote one
convention or the other — do not mix them, as an earlier version of this line did. A single
"ego speed is worth +0.18 AUC" figure is wrong; that figure describes the BiLSTM.

**Evidence.** `experiments/02_model_comparison/EGO_SPEED_ABLATION.md`.

⚠ **Do not claim that ego-speed "masks architectural differences."** The 4-D spread across families
(0.7765–0.9291) looks like architectures separating, but logistic regression on the same bbox-only
input scores 0.9129 — above three of the four networks. The spread is the BiLSTM (and to a lesser
degree GRU/RNN) failing to fit, not architectures revealing capability. A plausible mechanism is the
recurrent last-timestep readout discarding trajectory information that ego-speed otherwise
substitutes for, but that mechanism is **untested here** and should be stated as a hypothesis.

**Prior art.** Ego-speed dominance is settled (IntFormer 2021; CAPFI 2024; Diving Deeper 2024).
Do not claim the sign of the effect.

⚠ **And now qualify the magnitude too.** Under phase-matched negative sampling
(`PHASE_MATCHED_CONTROL.md`) the linear speed-only model falls 0.9335 → 0.8309 while bbox-only
barely moves (0.9129 → 0.8979), reversing which stream is stronger. Tested, not asserted:
speed-only degrades significantly on all three metrics (p_Holm ≤ 0.0136), box-only on none. A substantial share of the
apparent ego-speed dominance on this benchmark is a class-dependent *sampling* artefact rather than
a property of crossing behaviour. This is the single most important qualification in this document.

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

**Note on scope.** These numbers come from the original per-family study, not the matched protocol
of C6, so quote them only for the searched-vs-unsearched *control* — the point that an un-searched
transformer ties the BiLSTM. For any cross-family ranking use C6, which holds device, class weight
and checkpoint rule constant as well. The budget question is settled separately: the winner is an
architecture config ranked #2 of 36, so a matched 36-config budget selects the same model.

## C6. A matched four-family comparison — and the families do not tie

**Claim.** Under one engine, one device, one selection rule, matched class weight and matched search
budget, with pedestrian-clustered Holm-corrected inference, the **un-gated vanilla RNN (560,001
parameters — the smallest model tested) is the best**. It beats the BiLSTM on AUC and F1, the h128
baseline on AUC and PR-AUC, and the GRU on AUC and F1. It **ties the searched Transformer**
(AUC delta +0.0006, p = 0.83).

**Evidence.** `experiments/02_model_comparison/MATCHED_COMPARISON.md`. Nine of 30 tests survive Holm.

The ablation is structurally genuine: parameter counts sit in an exact **4 : 3 : 1** recurrent ratio
(594,561 / 446,081 / 149,121 at h128) with identical non-recurrent parts and identical state-dict
keys, so "only the cell changed" is verified, not asserted.

**Prior art.** Chung et al. (2014) already showed un-gated RNNs tie gated cells over short sequences.
This is confirmation with modern statistics — see LIMITATIONS §5.

**Superseded.** An earlier framing claimed the four families *tie* on F1. The matched experiment
refutes it: they differ, and the differences run in a consistent direction.

## C7. External validity: leakage generalises; the fix does not, unmodified

**Claim (new).** On IDD-PeD the naive anchor leaks at 81.3 %. The `crossing_point` rule that reaches
0 % on PIE leaves **29.6 % residual contamination**, because IDD-PeD's `crossing_point` is a
measurably weaker onset marker. A strict `min(crossing_point, first_onset)` anchor is required.

**Evidence.** `results/cross_dataset/idd_results/IDD_PeD_temporal_audit.csv`, `table2_temporal_audit.csv`.

Found by an independent parse that recovers the authors' exact track count (4,916 = 3,284 + 1,632).

⚠ **State this precisely — it contradicts an explicit claim in the dataset paper.** IDD-PeD asserts
that "for all the crossing cases, both the observation period and the time-to-event precede the
pedestrian's road crossing." Rebuilding windows from their released annotations does not reproduce
that. The disagreement is specifically that **their `crossing_point` does not reliably mark onset** —
a property of the annotation, reproducible from their public files. Frame it that way, not as an
accusation. Expect this to be the most contested claim in review.

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

## Note on the "statistical rigour" framing

Do **not** claim that this field reports no uncertainty. Three papers do:

- **BiPed** (Rasouli et al., ICCV 2021) — standard deviation over **20 random initialisations**
  (F1 std 0.006 vs PCPA's 0.01).
- **MFT** (Li et al., 2025) — ± on every metric (PIE acc 0.899 ± 0.005, AUC 0.885 ± 0.031).
- **Gesnouin et al.** (IV 2022) — 11 methods under one protocol with **Friedman + Wilcoxon–Holm**
  post-hoc testing (α = 0.1) and a critical-difference diagram. This is the closest methodological
  prior art to a "controlled, statistically rigorous comparison" and must be cited as such.

What remains genuinely unoccupied is narrower and should be stated narrowly:
**pedestrian-clustered resampling** (nobody respects the clustered structure of the test set),
**equivalence margins / reported power** for null results, and **search-budget control** across
compared families (Gesnouin uses each method's published configuration).

Likewise, do not claim observation-window length is never varied — **PIT** (TITS 2023, Table 4) and
**GTransPDM** (SPL 2025, Fig. 5) both sweep it. The unoccupied version is varying it *across
architecture families under one protocol*, with the horizon confound controlled (LIMITATIONS §4).

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
