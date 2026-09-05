# Tree ensembles on the leak-free data

Produced by `tree_baselines.py`. Decision tree, random forest and extra trees under the same
contract as the linear baselines, on `data/pie_clean/` (4,906 event-anchored windows).

## Why run this

`TRIVIAL_BASELINES.md` showed logistic regression on the 80 raw window values is
statistically indistinguishable from the best of four tuned neural families. That invites an
obvious objection: a linear model is one kind of simple model, and axis-aligned tree ensembles
are the other obvious family for tabular input of this shape. If a random forest also matched
the networks, the finding would be "any simple model works". If it did not, the finding is
narrower and more interesting.

## Protocol

Held identical to the linear baselines: same splits, same four flattened feature views,
`pos_weight` 1.682, one τ per model fitted on pooled **validation** probabilities by argmax
F1, test scored once. Tree models are stochastic, so each is fitted with the same five seeds
the neural families use and reported as mean ± sd; logistic regression is deterministic.
Inference is a pedestrian-clustered bootstrap, B = 10,000, Holm-corrected.

## Results — library defaults, no search

Five-seed mean ± sd. No model here received a hyperparameter search, matching how the linear
baselines are treated.

| model | AUC | PR-AUC | F1 |
|---|---|---|---|
| Decision tree | 0.8170 ± 0.0202 | 0.6489 ± 0.0218 | 0.7516 ± 0.0257 |
| Random forest | 0.9154 ± 0.0018 | 0.8664 ± 0.0034 | 0.7588 ± 0.0032 |
| Extra trees | 0.9252 ± 0.0015 | 0.8721 ± 0.0035 | 0.7786 ± 0.0026 |
| **Logistic regression** | **0.9488** | **0.9121** | **0.8546** |
| *Vanilla RNN (best neural)* | *0.9481 ± 0.0058* | *0.8925 ± 0.0062* | *0.8487 ± 0.0154* |

All nine tree contrasts against the linear reference survive Holm correction:

| contrast | Δ AUC | Δ PR-AUC | Δ F1 | worst p_Holm |
|---|---|---|---|---|
| Decision tree − LR(80) | −0.0944 | −0.2100 | −0.1025 | 0.0024 |
| Random forest − LR(80) | −0.0321 | −0.0447 | −0.0989 | 0.0028 |
| Extra trees − LR(80) | −0.0221 | −0.0396 | −0.0739 | 0.0216 |
| *Vanilla RNN − LR(80)* | *+0.0056* | *−0.0072* | *+0.0088* | *0.8105 (n.s.)* |

The neural family ties the linear model; every tree model loses to it, significantly, on every
metric.

## Results — after a search the linear model never got

The obvious objection is that trees with library defaults are undertuned. So the trees were
given a validation search that logistic regression does not receive. This is deliberately
asymmetric **in the trees' favour**: if they still lose, the gap is not a tuning artefact.
Selection is by validation AUC; the test split stays untouched.

| model | configs | selected | AUC | PR-AUC | F1 |
|---|---|---|---|---|---|
| Decision tree | 15 | depth 8, leaf 20 | 0.9165 ± 0.0013 | 0.8238 | 0.7994 |
| Random forest | 12 | 100 trees, depth 16, leaf 5 | 0.9254 ± 0.0022 | 0.8807 | 0.7666 |
| Extra trees | 12 | 100 trees, depth 16, leaf 5 | 0.9329 ± 0.0013 | 0.8877 | 0.8057 |
| **Logistic regression** | **0** | **defaults** | **0.9488** | **0.9121** | **0.8546** |

Searching helps the decision tree a great deal (0.8170 → 0.9165 AUC) and the ensembles very
little. **All nine contrasts still survive Holm.** The strongest tree, searched extra trees,
is −0.0152 AUC (p_Holm 0.0320), −0.0278 PR-AUC (0.0320) and −0.0511 F1 (0.0056) against an
unsearched linear model.

## The pattern worth reporting

Test AUC by feature view, defaults:

| model | 80 (box + speed) | 64 (box) | 16 (speed) | 5 (last frame) | best |
|---|---|---|---|---|---|
| Decision tree | 0.8170 | 0.6498 | **0.8626** | 0.7727 | 16 |
| Random forest | 0.9154 | 0.8177 | 0.9102 | **0.9208** | 5 |
| Extra trees | 0.9252 | 0.8199 | 0.8928 | **0.9291** | 5 |
| Logistic regression | **0.9488** | 0.9129 | 0.9335 | 0.9251 | **80** |

**Logistic regression is the only model that benefits from seeing the whole window.** Both
forests do better on five numbers from a single frame than on all eighty. Giving a tree
ensemble the flattened sequence makes it worse.

This is mechanical rather than mysterious. The flattened window is sixteen highly correlated
copies of the same five quantities. A linear model can form a weighted combination across
time — effectively a learned temporal filter — so the redundancy is useful to it. An
axis-aligned tree must pick one timestep at a time and cannot combine them, so the extra
columns are mostly noise that dilutes each split.

## What this establishes

**1. The linear result is not "any simple model works".** It is specific to a *linear* model
over the flattened sequence. Three standard tree learners, one of them given a search the
linear model never had, all lose significantly on all three metrics.

**2. It strengthens rather than weakens the trivial-baseline argument.** The claim that bounds
every architecture result in this literature is that a linear model on raw features matches
tuned networks. A reviewer asking "did you try a random forest?" now has an answer, and the
answer does not soften the finding.

**3. The failure mode is informative about the benchmark.** That tree ensembles peak on a
single frame, and that a linear model peaks on the full window, both point at the same thing:
the usable signal in this input is a smooth low-dimensional function of position and ego
speed, not a set of thresholded interactions.

## Caveats

- **Search asymmetry runs one way only.** Trees were searched, logistic regression and the
  reported neural configurations were not re-searched here. A *tuned* linear model was never
  tried either, so the comparison bounds the trees from above, not the linear model.
- Gradient boosting was not run. XGBoost or LightGBM would be the natural next family, and
  they are not equivalent to random forest on this kind of input.
- The searched trees select on the same 634-window validation split every other decision in
  this project uses, so the usual selection-bias caveat applies (Cawley & Talbot 2010).
- The decision-tree default (`max_depth=None`) grows to purity and is a known pathology; its
  untuned row should be read as a floor, not as a fair characterisation of the model class.
