# Limitations

Written against the code and the machine-written artifacts, not against the project's own notes.
Everything below was verified by re-reading the source or re-opening the result files.

---

## 1. Residual confound: the negative class is anchored at track termination

PIE annotates 1,374 pedestrians — 519 crossers and **855 non-crossers**. For a non-crosser,
`crossing_point` is not a behavioural event: it sits at the last annotated frame minus 2. Of the 833
non-crossers that reach `data/pie_clean/`, the gap is **exactly 2 for 824 (98.9 %)** and never
exceeds 2 (range 0–2).

The clean builder truncates at `crossing_point` for both classes. The consequence:

| | anchor position relative to end of track |
|---|---|
| negatives | 32–62 frames before track end (median 46) |
| positives | 158–741 frames before track end (median 287) |

Negatives are therefore **always observed as the ego vehicle is about to reach them**, while
positives are observed 1–2 s before a mid-track crossing. Any cue correlated with imminent passing —
ego deceleration, box growth rate — becomes a class signal produced by the *annotation termination
rule*, not by pedestrian behaviour.

This is inherited from PIE's own `extract_tracks_tte` and is not specific to this work — MFT (2025)
states the same convention explicitly, so it is field-wide. **The field has never questioned this
asymmetric anchor.**

Quantified: frames-from-anchor-to-track-end separates the classes with **AUC = 1.0000** (negatives
32–62, positives 88–6606, zero overlap).

**A control now exists** (`phase_matched_control.py`, LIMITATIONS is not the last word here):
re-sampling negatives earlier drops that separability to 0.7919 and costs every model 0.03–0.06 AUC,
with the linear speed-only model losing 0.103 and the ego-speed-vs-bbox advantage reversing. So the
bias is real and material. It is reduced, **not eliminated** — negative tracks are simply shorter,
and `to_end` still scores 0.8241 on the test split — and a distance-based competing explanation has
not been excluded. Under the control no contrast between the four families survives correction,
against 7 of the same 18 under the standard protocol; the strongest survivor misses by a hair
(p = 0.0032 against a threshold of 0.00278), so this is a failure to resolve differences, not
evidence of equivalence. See `experiments/02_model_comparison/PHASE_MATCHED_CONTROL.md`.

## 2. What "tie" can and cannot mean here

The earlier framing — that all four families tie — is **refuted by this project's own matched
experiment**: nine of thirty comparisons survive Holm correction. Two limitations remain around the
ties that *do* survive:

- **No equivalence margin was pre-specified.** The surviving null (Transformer ≈ Vanilla RNN,
  AUC p = 0.83) rests on a confidence interval containing zero. A defensible equivalence claim needs
  a margin δ agreed in advance plus a demonstration that the whole interval lies inside [−δ, +δ]
  (a TOST procedure). Bouthillier et al. (MLSys 2021) is the standing prior art against
  "CI includes 0 ⇒ equivalent".
- **Power is not reported.** Koehn (EMNLP 2004) found a genuine 0.5-point gap was detected on only
  12 % of 300-item test sets. At this sample size "not significant" is a likely outcome either way,
  so every null here should be read as *we could not detect a difference*, not *there is none*.
- **Hyperparameter-search variance is not randomised** — one search, then k seeds, which Bouthillier
  identifies as the biased `FixHOptEst` estimator.

## 2b. Every architecture claim is bounded by a linear baseline

Logistic regression on the 80 raw window features is statistically indistinguishable from the best
of the four searched neural families (AUC p = 0.27, PR-AUC p = 0.77, F1 p = 0.45). Ego-speed alone —
16 features, linear — scores 0.9335, above the 2.24M-parameter BiLSTM; a single frame scores 0.9251.

This does not invalidate the family comparison, but it does bound its importance: the differences
being measured sit at or below the level a linear model already reaches. Any statement of the form
"architecture X is better for this task" must be read against that ceiling.

## 2c. The 4-D BiLSTM result is a fitting failure, and its cause is untested

Removing ego-speed drops the BiLSTM to 0.7765 AUC while logistic regression on the same bbox-only
input reaches 0.9129. The information is present; the BiLSTM does not extract it. A plausible
mechanism is the recurrent last-timestep readout (`out[:, -1, :]`) discarding trajectory information
that ego-speed otherwise substitutes for, while attention and an explicit flattened linear model both
retain it. **This mechanism has not been tested here** — doing so would require re-running the
recurrent families with mean-pooled readout. Until then it is a hypothesis, and the 4-D per-family
numbers should not be read as a capability ranking.

## 3. The transformer's AUC win is not capacity-matched

`transformer_searched` has 794,241 parameters against the BiLSTM's 594,561 (1.34×), and received
2.2× the search budget (78 configs vs 36). The transformer that *ties* the BiLSTM has 268,417.
No capacity-matched transformer was ever evaluated on test, so **"attention wins" and "more
parameters win" are not separated**.

Related: the win does not survive leave-one-set-out cleanly. 6-fold LOSO gives 0.9392 vs 0.9285, but
only **+0.0016** once the degenerate 47-window `set05` fold (AUC 1.0000) is dropped.

## 4. The observation-window result is confounded with prediction horizon

The window builder ties the sliding stride to `obs_len`, so the fraction of windows sitting at the
maximum TTE = 60 rises **23.6 % → 46.7 % → 87.1 %** across OW16/32/64.

The project's own matched-cohort ablation
(`results/statistics/matched_cohort_tte_ablation.csv`) shows moving the horizon 45 → 60 costs
**0.070 F1** — larger than the entire window effect being attributed to window length. Within OW32
alone, far-TTE windows score 0.034–0.065 lower F1 than near-TTE windows.

**"F1 declines with observation window" is therefore not established by these runs.** What *is*
supported is that the four families remain indistinguishable at OW32.

The related claim that the un-gated RNN alone falls behind at OW64 rests on a 5-seed ensemble that
scores *worse than its own seed mean* (0.7841 vs 0.8015), with τ\* fitted on 138 validation windows
containing 31 positives.

## 5. Cell-type equivalence largely restates Chung (2014)

Chung et al. already showed that an un-gated tanh RNN ties gated cells over **short** sequences and
falls behind as the horizon grows. A 16-frame window is exactly that short-horizon regime, so
"gating buys nothing over 16 steps" is confirmation, not discovery. Likewise, "equalising the search
budget makes architecture differences vanish" is Melis et al. (2017) Table 1 in a new domain.

Note also that Chung fixes parameter count while Greff et al. (2017) explicitly refuse to. This work
compares at matched *hidden width*, which at h128 means a 149,121-parameter RNN against a
594,561-parameter BiLSTM — a 4× difference. That convention needs to be declared and defended.

## 6. The headline F1 model cannot be reproduced by re-running the code

All 65 runs in the F1-optimisation program record `device: mps`, and `nn.LSTM` training on Apple MPS
is process-history-dependent (same config + same seed gives different results depending on what ran
earlier in the process). **The published arm reproduces only from the saved checkpoints**, which is
why they are published as a release asset rather than regenerated on demand.

CPU training *is* bit-reproducible — that is the path `src/engine.py` uses and the one this
repository recommends.

## 7. A pre-registered gate failed and its documented fallback was not executed

The F1-first program pre-registered gate G1; it failed for the LSTM (2/5) and the stated fallback
("fall back to AUC checkpointing and document the amendment") was never run. An independent CPU
replication later showed the discarded alternative was **better** on test (ensemble F1 0.8550 with
AUC-checkpointing vs 0.8468 with F1-checkpointing). The shipped headline model embodies a lever its
own gate rejected.

Relatedly, `--select f1` is only half F1-first: early stopping and the LR scheduler still step on
validation **AUC**, and validation F1 is scored at a hardcoded 0.5 while the reported metric uses a
validation-fitted τ\*.

## 8. The JAAD model comparison is a null result

All 20 runs are at chance (AUC 0.494–0.520; 8 of 20 below 0.5) and every one is beaten by a constant
all-positive classifier (F1 0.8143 vs best 0.8006). The JAAD *leakage* replication is sound and
independent; the JAAD *four-family comparison* establishes nothing about architecture equivalence.

The naive JAAD window set was built but never trained on, so the **consequence** of leakage was never
tested on JAAD — only its prevalence.

## 9. The deployed model behaves close to a pure function of ego speed

Recomputed over the four tracked demo prediction files (`pipeline/demo_out/*_predictions.csv` in the
source project, 19,799 rows), the deployed ensemble gives **Pearson r(ego_speed, p_cross) = −0.892**.
At 0 km/h it flags **96.2 %** of all tracked pedestrians; above 20 km/h it flags **1.00 %**.

⚠ An earlier version of this section reported 3,652 predictions, r = −0.908, 96.8 % and 0.45 %, over
"sixteen segments". Those figures do not reproduce from the tracked CSVs and no subset of them yields
3,652 rows, so they have been replaced by the recomputed values above. The conclusion is unchanged.

This corroborates "the input signal dominates" — and independently matches Diving Deeper (IV 2024)
Table IV, where PIE action mAP falls from 0.950 at 0 km/h to 0.163 above 30 km/h. But it also means
the deployed system has almost **no within-frame discrimination between pedestrians**. Any demo
narrative describing per-pedestrian judgement is describing a speed threshold.

## 10. Scope

- Single dataset for the main protocol (PIE), single test split (`set03`, 2,094 windows from 541
  pedestrians). External validity rests on JAAD and IDD-PeD, one of which is a null result.
- Validation is 634 windows / 164 pedestrians, and one of its two recording sets contributes only 13
  pedestrians. Best epochs scatter 5–17 across seeds.
- The same 634 validation windows are reused for four sequential selections plus the threshold fit,
  so every validation number should be read as an upper bound. The val→test gap is visible: the
  h256-vs-baseline gap is +0.0137 on validation but +0.0051 on test.
- Labels are a track-level attribute broadcast to every window of that track.
- Daytime, dry-weather, single-city driving data.
