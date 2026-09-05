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

## Under the phase-matched control

Re-run on `data/pie_phase_matched_trainonly/` (`--data ... --runs-subdir phase_matched_trainonly`),
with the class weight recomputed from that split's own training data (1.5665) and the neural
reference taken from its own checkpoints. Test partition 1,873 windows from 476 pedestrians.

| model, 80 features | AUC | PR-AUC | F1 |
|---|---|---|---|
| Decision tree | 0.7050 ± 0.0059 | 0.5542 | 0.6217 |
| Extra trees | 0.8337 ± 0.0031 | 0.7326 | 0.6890 |
| Random forest | 0.8565 ± 0.0036 | 0.7682 | 0.7041 |
| *Vanilla RNN* | *0.8872 ± 0.0113* | *0.7749* | *0.7553* |
| **Logistic regression** | **0.9053** | **0.8283** | **0.7621** |

Searched trees, same protocol:

| model | selected | AUC | PR-AUC | F1 |
|---|---|---|---|---|
| Decision tree | depth 5, leaf 20 | 0.8625 ± 0.0038 | 0.7560 | 0.7362 |
| Random forest | 300 trees, depth 8, leaf 1 | 0.8724 ± 0.0015 | 0.7775 | 0.7312 |
| Extra trees | 300 trees, depth 8, leaf 5 | 0.8786 ± 0.0009 | 0.7824 | 0.7313 |

Three things change, and they are worth separating.

**1. The ranking does not change.** Logistic regression is still first on every metric, and every
tree contrast on AUC and PR-AUC still survives Holm, searched or not. The headline conclusion is
protocol-independent.

**2. On F1, the searched trees stop being distinguishable from it.** Decision tree −0.0142
(p_Holm 0.4108), random forest −0.0269 (0.3020), extra trees −0.0318 (0.1578). On the
event-anchored data all nine contrasts survived; here only the six AUC and PR-AUC ones do. The
F1 gap narrows because every model's F1 falls and the spread widens.

**3. The linear model pulls slightly ahead of the best network.** On the event-anchored data the
vanilla RNN and logistic regression were indistinguishable (AUC p_Holm 0.8105). Under phase
matching the RNN is *numerically behind*: ΔAUC −0.0140 (p_Holm 0.0876) and ΔPR-AUC −0.0448
(p_Holm 0.0540). Both are significant before correction and neither survives it, so this is a
trend and not a result — but it runs the opposite way from what the architecture literature
would predict.

Every model degrades, and the linear one degrades least:

| model | AUC, event-anchored → phase-matched | Δ |
|---|---|---|
| **Logistic regression** | 0.9488 → 0.9053 | **−0.0436** |
| Random forest | 0.9154 → 0.8565 | −0.0589 |
| Vanilla RNN | 0.9481 → 0.8872 | −0.0609 |
| Extra trees | 0.9252 → 0.8337 | −0.0915 |
| Decision tree | 0.8170 → 0.7050 | −0.1121 |

Searched, the three trees lose an almost identical amount (−0.0541, −0.0530, −0.0543), which is
what one expects if they were all exploiting the same sampling artefact to the same degree.

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
