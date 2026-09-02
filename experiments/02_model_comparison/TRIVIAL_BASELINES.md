# Trivial baselines — how much of this task needs a neural network?

Produced by `trivial_baselines.py`. Fitted on train only, standardised with train statistics
only, thresholded on validation only, scored once on set03 — the same contract the neural
models obey. `class_weight` matches the neural `pos_weight` (1.682). No baseline
hyperparameter is tuned, which if anything handicaps them.

## Why this exists

No published work on PIE/JAAD reports a genuinely trivial baseline. The weakest comparison in
PCPA (WACV 2021) is a deep CNN on a single frame; nobody reports majority-class, speed-only, or
a linear model on the raw window. So nobody knows how much of this benchmark a straight line
solves.

## Results

| baseline | AUC | PR-AUC | F1 |
|---|---|---|---|
| majority class (always positive) | 0.5000 | 0.3252 | 0.4908 |
| LR bbox + ego-speed (80 feats) | 0.9488 | 0.9121 | 0.8546 |
| LR ego-speed only (16 feats) | 0.9335 | 0.8538 | 0.8199 |
| LR last frame only (5 feats) | 0.9251 | 0.8757 | 0.7903 |
| LR bbox only (64 feats) | 0.9129 | 0.8035 | 0.7812 |
| *best neural model — Vanilla RNN, 560,001 params* | *0.9481* | *0.8925* | *0.8487* |
| *Transformer, 794,241 params* | *0.9447* | *0.8964* | *0.8250* |
| *BiLSTM, 2,237,313 params* | *0.9242* | *0.8688* | *0.8276* |

## Is the linear model distinguishable from the best network?

Pedestrian-clustered bootstrap, B = 10,000, against the Vanilla RNN:

| metric | delta | 95% CI | p |
|---|---|---|---|
| AUC | -0.0056 | [-0.0164, +0.0041] | 0.2702 |
| PR-AUC | +0.0072 | [-0.0202, +0.0299] | 0.7731 |
| F1 | -0.0088 | [-0.0319, +0.0139] | 0.4470 |

**No.** On all three metrics the linear model and the best of four searched neural
families are statistically indistinguishable.

## What this establishes

1. **A linear model on 80 raw numbers matches every neural family tested.** Any architecture
   claim on this benchmark must be read against that ceiling.
2. **Ego-speed alone — 16 numbers, linear — scores 0.9335**, above the 2.24M-parameter BiLSTM
   using all inputs. This is the strongest available statement of ego-speed dominance.
3. **A single frame (5 numbers) scores 0.9251.** The temporal model is not carrying the result.
4. **The 4-D BiLSTM's 0.7765 is a fitting failure, not an information limit** — logistic
   regression on the same bbox-only input reaches 0.9129.

Taken with the leakage result: fixing the contamination removes the *detection* shortcut, but
what remains is still largely a linear function of ego-vehicle dynamics.
