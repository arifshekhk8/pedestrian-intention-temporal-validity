# Which input channel survives the PIE → IDD-PeD domain shift?

Inference-only ablation on the **frozen PIE checkpoints** (no retraining, no new modality). A channel is *neutralised* by replacing it with the PIE training mean, so after PIE standardization it is exactly 0 — carrying no information and introducing no distribution shift of its own. 5-seed probability ensembles, IDD-PeD strict test (2,357 windows, 168 positive, 757 pedestrian clusters), pre-registered `rescale` coordinates.

| model | full | − ego-speed | − all boxes | − y only | − x only |
|---|---|---|---|---|---|
| BiLSTM | **0.723** | 0.568 (-0.156) | 0.741 (+0.018) | 0.764 (+0.040) | 0.677 (-0.047) |
| Transformer | **0.752** | 0.565 (-0.187) | 0.754 (+0.001) | 0.774 (+0.022) | 0.693 (-0.059) |
| GRU | **0.722** | 0.572 (-0.150) | 0.751 (+0.029) | 0.764 (+0.042) | 0.663 (-0.059) |
| Vanilla RNN | **0.713** | 0.548 (-0.165) | 0.753 (+0.040) | 0.771 (+0.058) | 0.683 (-0.030) |

Mean AUC change across the four families: removing **ego-speed -0.165**, removing **all box channels +0.022**.

**Ego-speed is the transferable stream.** Neutralising it costs more than neutralising the entire box geometry — consistent with the PIE finding that the ego-speed signal, not the box trajectory, carries this task, and with the schema audit's measurement that IDD-PeD's speed arrives on PIE's own scale (z = −0.002) while its box coordinates do not.
