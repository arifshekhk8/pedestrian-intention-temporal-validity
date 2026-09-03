# Fixing a split leak in the phase-matched control

The phase-matched control estimated its target timing distribution from positives in
**every** split, then used it to place negatives in every split. Validation and test
positive timing therefore helped decide where validation and test negatives were drawn.
This note records the defect, the fix, and what changed. The superseded artefacts are
kept.

## The defect

`phase_matched_control.py` built the target distribution before the split mask existed:

```python
for m, lab in zip(meta, y):          # every window, all recording sets
    te = bounds[k][1] - m["anchor_frame"]
    if lab == 1:
        pos_to_end.append(te)        # <- no split filter
...
target = int(rng.choice(pos_to_end)) # <- draws test negatives from test positives
```

The split mask was applied ~60 lines later, after every negative anchor had been drawn.
`MIN_TO_END = 88` was likewise the minimum over pooled positives.

Measured, on `data/pie_clean/`:

| source of the sampled distribution | positive windows | share |
|---|---|---|
| train (set01/02/04) | 812 | 49.3 % |
| **val + test (set05/06, set03)** | **836** | **50.7 %** |

Not a harmless pooling. The splits have visibly different positive timing:

| split | n | min | median | max |
|---|---|---|---|---|
| train | 812 | 88 | 268 | 6606 |
| val | 155 | 134 | 288 | 1165 |
| test | 681 | 116 | 307 | 1642 |

For a paper whose contribution is temporal validity, a control that consults test-set
timing to build its own test set is not defensible, whatever the effect size turns out
to be.

## The fix

One rule change, in `build()`:

1. Estimate the target distribution from **training positives only** (n = 812).
2. Derive the floor from that same source. It is still 88 — the global minimum happens
   to sit in `set01`, a training set — but it is now *derived*, not read off the pooled
   data.
3. Freeze both to `<out>/phase_rule.json` before any negative is drawn.
4. Apply the frozen rule unchanged to train, validation and test. Each negative's
   placement uses only the frozen rule and that pedestrian's own track bounds.

Everything else is untouched: recording-set splits, positive windows, features, the
16-consecutive-frame check, engine, configs, seeds `{42,0,1,2,3}`, checkpoint rule,
threshold procedure, B = 10,000 pedestrian-clustered bootstrap, Holm.

`--phase-source all` reproduces the old behaviour for provenance. It is not used for
any reported number.

### Artefacts

| | superseded (v1) | corrected (v2) |
|---|---|---|
| data | `data/pie_phase_matched/` | `data/pie_phase_matched_trainonly/` |
| checkpoints | `runs/phase_matched/` | `runs/phase_matched_trainonly/` |
| point estimates | `phase_matched_results.json` | `phase_matched_trainonly_results.json` |
| statistics | `phase_matched_stats.json` | `phase_matched_trainonly_stats.json` |

## What changed in the data

The dataset is the same size — 4,520 windows, 1,648 positive, 2,872 negative, splits
2,084 / 563 / 1,873, `pos_weight` 1.5665 — but **1,295 of 2,872 negative windows (45 %)
moved to a different anchor**. Positives are byte-identical, as they must be.

Residual phase separability rose slightly:

| `to_end` alone, AUC | v1 | v2 |
|---|---|---|
| overall | 0.7779 | 0.7919 |
| train | 0.7516 | 0.7664 |
| val | 0.7832 | 0.8007 |
| **test** | **0.8134** | **0.8241** |

This is the expected direction and it is worth stating plainly. The train-only positive
distribution sits earlier than the pooled one (median 268 vs 287), so negatives placed
in the test split land slightly further from the test positives. The v1 figure was
flattered by the leak: it looked like a better match partly because it had seen the test
timing. The honest number is 0.8241.

## What changed in the results

Point estimates barely moved. Five-seed ensemble, phase-matched test split:

| model | AUC v1 → v2 | PR-AUC v1 → v2 | F1 v1 → v2 |
|---|---|---|---|
| BiLSTM | 0.8907 → 0.8944 | 0.7774 → 0.7816 | 0.7771 → 0.7738 |
| Transformer | 0.9007 → 0.8992 | 0.7807 → 0.8093 | 0.7697 → 0.7480 |
| GRU | 0.8867 → 0.8877 | 0.7603 → 0.7722 | 0.7680 → 0.7629 |
| Vanilla RNN | 0.8902 → 0.8913 | 0.7807 → 0.7835 | 0.7725 → 0.7723 |
| LR bbox + ego-speed (80) | 0.9031 → 0.9053 | 0.8184 → 0.8283 | 0.7561 → 0.7621 |
| LR ego-speed only (16) | 0.8257 → 0.8309 | 0.6594 → 0.6782 | 0.7158 → 0.7161 |
| LR bbox only (64) | 0.8987 → 0.8979 | 0.8118 → 0.8216 | 0.7578 → 0.7407 |

No AUC moves by more than 0.0051. Every qualitative conclusion survives.

## The four questions

**1. Does the main conclusion hold?** Yes.

Corrected for the confound noted below, using the same 18 architecture contrasts under
both protocols:

| protocol | family-vs-family contrasts surviving Holm(18) |
|---|---|
| standard | **7 of 18** |
| phase-matched (v2) | **0 of 18** |

Under the control the architecture ranking stops being resolvable. The strongest
remaining contrast is the Transformer over the GRU on AUC: Δ = +0.0115, p = 0.0032,
against a first-step Holm threshold of 0.00278. It is a near-miss, not a wide margin,
and the paper should say so rather than claim the differences vanish.

**2. Does ego-speed-only still degrade more than bbox-only?** Yes — more cleanly than
in v1.

| baseline | AUC standard → matched | Δ | p_Holm | metrics significant |
|---|---|---|---|---|
| LR ego-speed only (16) | 0.9335 → 0.8309 | **−0.1026** | **0.0042** | **3 / 3** |
| LR bbox only (64) | 0.9129 → 0.8979 | −0.0150 | 0.8143 | 0 / 3 |

Every bbox-only interval still contains zero. Under the control, bbox-only now beats
ego-speed-only significantly on AUC (+0.0670, p_Holm 0.0126) and PR-AUC (+0.1434,
p_Holm 0.0224). The reversal is intact.

**3. Do neural-family differences remain unresolved?** Yes. 0 of 18, as above. No neural
family differs significantly from any other, and none differs from the 80-feature linear
model on any metric (all p_Holm = 1.0000).

**4. Numbers needing updating.** Listed in the next section.

## A separate defect found in the same document

`PHASE_MATCHED_CONTROL.md` compared "9 of 30 survive under the standard protocol" with
"0 of 18 under the control". Those are not comparable. The standard run corrects across
30 tests over 5 models (four families plus the h128 capacity guard); the control's arm A
corrects across 63 tests over 7 models (four families plus three linear baselines). The
numerator also changed definition mid-sentence: 4 of those 9 survivors are h128 capacity
contrasts, not family contrasts.

`phase_matched_stats.py` now reports `between_families_holm18` — Holm re-applied to the
18 architecture contrasts alone — so both protocols can be quoted on the same basis.
That is the 7-vs-0 comparison above.

Three further corrections to that document, all verified against the JSON:

- "the only model whose degradation is significant on all three metrics" was wrong. In
  v2, Vanilla RNN is 3/3 **and** LR ego-speed-only is 3/3. The claim holds only for
  neural families.
- The arm B table listed 9 rows while the count said 10. `LR bbox + ego-speed (80)
  [F1@tau]` was missing.
- The "115 negative pedestrians dropped" figure was hard-coded into the output note. It
  is now computed.

## Manuscript updates

Replace throughout. Left column is what the current draft carries.

| quantity | old | new |
|---|---|---|
| `to_end` AUC after matching | 0.7779 | **0.7919** (test split: **0.8241**) |
| BiLSTM AUC (per-seed mean) | 0.8842 | **0.8923** |
| Transformer AUC | 0.8928 | **0.8971** |
| GRU AUC | 0.8828 | **0.8817** |
| Vanilla RNN AUC | 0.8852 | **0.8872** |
| LR(80) AUC | 0.9031 | **0.9053** |
| LR speed-only AUC | 0.8257 | **0.8309** |
| LR box-only AUC | 0.8987 | **0.8979** |
| four-family AUC span | 0.010 | **0.0154** (standard protocol: 0.024) |
| family contrasts surviving | "0 of 18" vs "9 of 30" | **0 of 18** vs **7 of 18**, same 18 tests |
| arm A survivors | 8 of 63 | **9 of 63** |
| arm B survivors | 10 of 21 | **10 of 21** (unchanged) |
| speed-only degradation | −0.1077, p 0.0042 | **−0.1026, p_Holm 0.0042** |
| box-only degradation | −0.0141, p 1.0000 | **−0.0150, p_Holm 0.8143** |
| Vanilla RNN AUC drop | −0.0642 | **−0.0632** |

§4 must also state that the phase rule is estimated from training positives only. The
statistics paragraph should declare the 18-contrast family for the architecture claim,
since that is now the basis on which both protocols are quoted.

## Caveats, unchanged by this fix

- Residual bias remains: `to_end` still scores 0.7919 overall and 0.8241 on test, not
  0.5. Negative tracks are simply shorter than positive ones, so the distributions
  cannot be fully matched.
- A competing explanation is still not excluded. Negatives sampled earlier are further
  from the camera, so their boxes are smaller and noisier. Part of the drop may be
  increased difficulty rather than a removed shortcut. Separating these needs a
  distance-matched design.
- Negative anchors are drawn once with seed 42. The draw was not repeated.
