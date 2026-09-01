"""Issue 4 — Bootstrap 95% CIs on test AUC (and PR-AUC) for all 3 model variants.

Most PIE crossing-prediction papers report a single point AUC with no confidence
interval; with only 2,094 test windows (set03) the test-set sampling uncertainty is
non-trivial. This regenerates each saved checkpoint's raw test probabilities and
percentile-bootstraps (probs, labels) to put a 95% CI on ROC-AUC and PR-AUC.

Checkpoints (clean protocol, 5 seeds each):
  baseline 5-D   : ../issue2_clean_protocol/runs_clean/multiseed/seed{S}/
  bbox-only 4-D  : ../issue2_clean_protocol/kaggle_result/runs_multiseed_clean/bilstm_bbox_only_seed{S}/
  attention 5-D  : ../issue2_clean_protocol/kaggle_result/runs_multiseed_clean/bilstm_attention_seed{S}/

Run locally (sklearn + scipy present in .venv); CPU is fine.
"""
import argparse, importlib.util, json, pickle
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
I2 = HERE.parent / "issue2_clean_protocol"
SEQ_DIR = I2 / "sequences_clean"
SEEDS = [42, 0, 1, 2, 3]

# reuse the verbatim model classes from the local cross-check script
_spec = importlib.util.spec_from_file_location("v6b", I2 / "06b_local_verify_seed42.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)

MODELS = {
    "baseline":  dict(dir=I2 / "runs_clean/multiseed",
                      sub="seed{S}", build=lambda: _m.BiLSTMIntentPredictor(input_dim=5), dim=5),
    "bbox_only": dict(dir=I2 / "kaggle_result/runs_multiseed_clean",
                      sub="bilstm_bbox_only_seed{S}", build=lambda: _m.BiLSTMIntentPredictorFlex(input_dim=4), dim=4),
    "attention": dict(dir=I2 / "kaggle_result/runs_multiseed_clean",
                      sub="bilstm_attention_seed{S}", build=lambda: _m.BiLSTMAttentionIntentPredictor(input_dim=5), dim=5),
}


def load_test():
    X = np.load(SEQ_DIR / "X.npy").astype(np.float32)
    y = np.load(SEQ_DIR / "y.npy").astype(np.float32)
    meta = pickle.load(open(SEQ_DIR / "meta.pkl", "rb"))
    sid = np.array([r["set_id"] for r in meta])
    te = sid == "set03"
    return X[te], y[te]


def auc_fast(y, s):
    """Tie-corrected ROC-AUC via the Mann-Whitney rank formula (fast for bootstrap)."""
    n_pos = y.sum(); n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@torch.no_grad()
def get_probs(model_name, seed, Xte):
    cfg = MODELS[model_name]
    d = cfg["dir"] / cfg["sub"].format(S=seed)
    mean = np.load(d / "norm_mean.npy"); std = np.load(d / "norm_std.npy")
    Xn = (Xte[:, :, :cfg["dim"]] - mean) / std
    model = cfg["build"]()
    ck = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck["model_state_dict"])
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(Xn)).squeeze(-1)).numpy()


def bootstrap_ci(y, p, B, rng):
    n = len(y)
    rocs = np.empty(B); prs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        rocs[b] = auc_fast(yb, pb)
        prs[b] = average_precision_score(yb, pb)
    return (np.nanpercentile(rocs, [2.5, 97.5]), np.nanpercentile(prs, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10000, help="bootstrap resamples")
    ap.add_argument("--rng", type=int, default=42)
    args = ap.parse_args()

    Xte, yte = load_test()
    print(f"test set03: N={len(yte)}, positives={int(yte.sum())} ({yte.mean():.3f})  | B={args.B}\n")

    rows = []
    for name in MODELS:
        for seed in SEEDS:
            rng = np.random.default_rng(args.rng)  # same resamples across runs/models -> paired
            p = get_probs(name, seed, Xte)
            roc = float(roc_auc_score(yte, p)); pr = float(average_precision_score(yte, p))
            (rlo, rhi), (plo, phi) = bootstrap_ci(yte, p, args.B, rng)
            rows.append(dict(model=name, seed=seed, roc_auc=roc, roc_lo=rlo, roc_hi=rhi,
                             pr_auc=pr, pr_lo=plo, pr_hi=phi))
            print(f"{name:10s} seed {seed:2d} | ROC {roc:.3f} [{rlo:.3f}, {rhi:.3f}] "
                  f"| PR {pr:.3f} [{plo:.3f}, {phi:.3f}]")

    # CSV
    import csv
    with open(HERE / "04_bootstrap_ci_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # Markdown
    def agg(name, key):
        v = np.array([r[key] for r in rows if r["model"] == name])
        return v.mean(), v.std()
    lines = [f"# Issue 4 — Bootstrap 95% CIs on test AUC (clean protocol)", "",
             f"Test = PIE set03, N={len(yte)} windows ({yte.mean():.1%} positive). "
             f"{args.B:,} percentile bootstrap resamples of (probs, labels); fixed RNG "
             f"(seed {args.rng}) so resamples are paired across models. ROC-AUC via "
             f"tie-corrected Mann-Whitney; PR-AUC = average precision.", "",
             "## Per-model summary (5 training seeds)", "",
             "| Model | ROC-AUC (seed mean ± std) | typical 95% CI width | PR-AUC (mean ± std) |",
             "|---|---|---|---|"]
    for name in MODELS:
        rm, rs = agg(name, "roc_auc"); pm, ps = agg(name, "pr_auc")
        widths = np.array([r["roc_hi"] - r["roc_lo"] for r in rows if r["model"] == name])
        lines.append(f"| {name} | {rm:.3f} ± {rs:.3f} | ±{widths.mean()/2:.3f} (≈[{rm-widths.mean()/2:.3f}, {rm+widths.mean()/2:.3f}]) | {pm:.3f} ± {ps:.3f} |")
    lines += ["", "## Per-seed detail (point estimate + 95% bootstrap CI)", "",
              "| Model | Seed | ROC-AUC | ROC 95% CI | PR-AUC | PR 95% CI |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['seed']} | {r['roc_auc']:.3f} | "
                     f"[{r['roc_lo']:.3f}, {r['roc_hi']:.3f}] | {r['pr_auc']:.3f} | "
                     f"[{r['pr_lo']:.3f}, {r['pr_hi']:.3f}] |")
    (HERE / "04_bootstrap_ci_results.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote 04_bootstrap_ci_results.csv + .md")


if __name__ == "__main__":
    main()
