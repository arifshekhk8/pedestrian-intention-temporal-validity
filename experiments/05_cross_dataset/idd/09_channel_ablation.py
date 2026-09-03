"""09_channel_ablation.py — which input channel carries the transferable signal?

Experiment A shows the PIE-trained models retain some ranking ability on IDD-PeD
(AUC ~0.72) even though their F1 collapses. This script asks *which part of the input*
survives the domain shift, by neutralising channels at inference time on the SAME frozen
PIE checkpoints — no retraining, no new modality, inference only.

"Neutralising" a channel means replacing it with the **PIE training mean** for that channel,
so after PIE standardization the channel is exactly 0: it carries no information AND
introduces no distribution shift of its own. Comparing the AUC drop across channels
isolates each stream's contribution to the transfer.

Motivated by the schema audit's finding that the ego-speed channel arrives on PIE's own
scale (standardized mean z = -0.002) while the box y-channels are far outside PIE's training
distribution (z = +1.23 / +1.77 raw, -2.78 / -2.98 rescaled). If ego-speed is the
transferable stream, neutralising it should cost much more than neutralising the boxes.

Writes  results/channel_ablation.json / .md
        figures/fig5_channel_ablation.png

Run from the repo root (after 05):
    python idd_ped_crossdataset/scripts/09_channel_ablation.py
"""
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FOLDER = HERE.parent
sys.path.insert(0, str(HERE / "lib"))
from pie_bridge import (PIE_ARMS, SEEDS, arm_run_dir, load_common, load_engine,  # noqa: E402
                        load_iddped, ped_clusters, split_masks, to_pie_frame)

RES = FOLDER / "results"
SEQ = FOLDER / "data" / "sequences_iddped_clean"
MODELS = list(PIE_ARMS)
NICE = {"BiLSTM-F1": "BiLSTM", "Transformer-F1": "Transformer",
        "GRU-F1": "GRU", "Vanilla_RNN-F1": "Vanilla RNN"}

# channel index -> label
CH = {0: "x1", 1: "y1", 2: "x2", 3: "y2", 4: "ego_speed"}
ABLATIONS = {
    "full": [],
    "no_ego_speed": [4],
    "no_boxes": [0, 1, 2, 3],
    "no_y_only": [1, 3],
    "no_x_only": [0, 2],
}
B, RNG_SEED = 10_000, 42


def cluster_ci(y, v, stat, groups, B=B, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    k = len(groups)
    vals = np.empty(B)
    for b in range(B):
        idx = np.concatenate([groups[i] for i in rng.integers(0, k, k)])
        vals[b] = stat(y[idx], v[idx])
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def main():
    common = load_common()
    engine = load_engine()

    X_raw, y_all, meta = load_iddped(SEQ)
    te = split_masks(meta)["test"]
    meta_te = [m for m, k in zip(meta, te) if k]
    y = y_all[te]
    groups = ped_clusters(meta_te)
    X_te = to_pie_frame(X_raw[te], meta_te, mode="rescale")   # the pre-registered mapping

    out = {"n_test": int(te.sum()), "n_pos": int(y.sum()), "n_clusters": len(groups),
           "coord": "rescale", "results": {}}

    for arm in MODELS:
        spec = PIE_ARMS[arm]
        per_ab = {k: [] for k in ABLATIONS}
        for seed in SEEDS:
            rd = arm_run_dir(arm, seed)
            mean = np.load(rd / "norm_mean.npy")
            model = engine.MODEL_REGISTRY[spec["family"]](spec["cfg"])
            pf = common.prob_fn_from_run_dir(rd, model)
            for name, chans in ABLATIONS.items():
                Xa = X_te.copy()
                for c in chans:
                    Xa[:, :, c] = mean[c]          # -> standardized value exactly 0
                per_ab[name].append(pf(Xa))

        block = {}
        base = None
        for name in ABLATIONS:
            P = np.mean(np.stack(per_ab[name]), axis=0)
            auc = float(common.auc_fast(y, P))
            pr = float(common.metrics_at(y, P, 0.5)["pr_auc"])
            lo, hi = cluster_ci(y, P, common.auc_fast, groups)
            if name == "full":
                base = auc
            block[name] = dict(auc=auc, pr_auc=pr, ci95=[float(lo), float(hi)],
                               delta_vs_full=float(auc - base))
        out["results"][arm] = block
        print(f"{arm:16s} " + "  ".join(
            f"{n}={block[n]['auc']:.3f}({block[n]['delta_vs_full']:+.3f})" for n in ABLATIONS))

    (RES / "channel_ablation.json").write_text(json.dumps(out, indent=2))

    # ---------------------------------------------------------------- report
    md = ["# Which input channel survives the PIE → IDD-PeD domain shift?\n",
          "Inference-only ablation on the **frozen PIE checkpoints** (no retraining, no new "
          "modality). A channel is *neutralised* by replacing it with the PIE training mean, so "
          "after PIE standardization it is exactly 0 — carrying no information and introducing "
          "no distribution shift of its own. 5-seed probability ensembles, "
          f"IDD-PeD strict test ({out['n_test']:,} windows, {out['n_pos']:,} positive, "
          f"{out['n_clusters']:,} pedestrian clusters), pre-registered `rescale` coordinates.\n",
          "| model | full | − ego-speed | − all boxes | − y only | − x only |",
          "|---|---|---|---|---|---|"]
    for arm in MODELS:
        b = out["results"][arm]
        md.append(f"| {NICE[arm]} | **{b['full']['auc']:.3f}** | "
                  f"{b['no_ego_speed']['auc']:.3f} ({b['no_ego_speed']['delta_vs_full']:+.3f}) | "
                  f"{b['no_boxes']['auc']:.3f} ({b['no_boxes']['delta_vs_full']:+.3f}) | "
                  f"{b['no_y_only']['auc']:.3f} ({b['no_y_only']['delta_vs_full']:+.3f}) | "
                  f"{b['no_x_only']['auc']:.3f} ({b['no_x_only']['delta_vs_full']:+.3f}) |")
    md.append("")
    d_speed = np.mean([out["results"][a]["no_ego_speed"]["delta_vs_full"] for a in MODELS])
    d_box = np.mean([out["results"][a]["no_boxes"]["delta_vs_full"] for a in MODELS])
    md.append(f"Mean AUC change across the four families: removing **ego-speed "
              f"{d_speed:+.3f}**, removing **all box channels {d_box:+.3f}**.\n")
    if d_speed < d_box:
        md.append("**Ego-speed is the transferable stream.** Neutralising it costs more than "
                  "neutralising the entire box geometry — consistent with the PIE finding that "
                  "the ego-speed signal, not the box trajectory, carries this task, and with "
                  "the schema audit's measurement that IDD-PeD's speed arrives on PIE's own "
                  "scale (z = −0.002) while its box coordinates do not.\n")
    else:
        md.append("**The box channels, not ego-speed, carry what transfers here.** This does "
                  "*not* reproduce the PIE ego-speed-dominance pattern and must be reported as "
                  "such.\n")
    (RES / "channel_ablation.md").write_text("\n".join(md))

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(figsize=(9, 4.6))
    names = list(ABLATIONS)
    x = np.arange(len(MODELS)); wd = 0.16
    colors = ["#2c3e50", "#c0392b", "#7f8c8d", "#e67e22", "#2980b9"]
    for i, n in enumerate(names):
        v = [out["results"][a][n]["auc"] for a in MODELS]
        ax.bar(x + (i - 2) * wd, v, wd, label=n.replace("_", " "), color=colors[i])
    ax.axhline(0.5, ls="--", c="grey", lw=1)
    ax.text(len(MODELS) - 0.5, 0.505, "chance", fontsize=8, color="grey")
    ax.set_xticks(x); ax.set_xticklabels([NICE[a] for a in MODELS], fontsize=9)
    ax.set_ylabel("zero-shot IDD-PeD test AUC (5-seed ensemble)")
    ax.set_ylim(0.4, 0.85)
    ax.set_title("Channel ablation on frozen PIE models, evaluated on IDD-PeD")
    ax.legend(fontsize=8, ncol=3); ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(FOLDER / "figures" / "fig5_channel_ablation.png", dpi=200)
    plt.close(fig)

    print(f"\nWrote {RES/'channel_ablation.json'}\nWrote {RES/'channel_ablation.md'}")
    print(f"mean Δ: -ego_speed {d_speed:+.3f}   -boxes {d_box:+.3f}")


if __name__ == "__main__":
    main()
