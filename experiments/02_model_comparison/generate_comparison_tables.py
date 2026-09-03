"""00_generate_analysis.py — cross-model analysis pack for EVERY model in the four families.

Regenerates every table + figure in this folder. Most models are read from cached 5-seed
probability ensembles (no retraining); the two older BiLSTM variants without a cache
(bbox-only 4-D, additive-attention) are re-scored on the fly from their multiseed checkpoints.
`y_test` is verified identical across all caches before anything is computed.

Model families & sources of the probability vectors:
  BiLSTM        journal_prep/ + f1_optimization/ + pipeline/ :
                  baseline (lstm_frozen), bbox-only (regen), attention (regen), BiLSTM-F1 (lstm_a3)
  Transformer   transformer/ + f1_optimization/ :
                  searched (tf_frozen), default/un-searched (tf_b4), Transformer-F1 (tf_b3)
  GRU           gru/phase4_final/ : F1-winner, default-F1, default-AUC
  RNN           rnn/phase4_final/ : F1-winner, winner-AUC, default-F1, default-AUC

Confusion matrices use the deployable **5-seed probability ensemble** at each model's operating
threshold; the comparison tables' Acc/AUC/F1 are the **per-seed mean** (the paper numbers). Both
are labelled everywhere (the project's 0.932-vs-0.942 discipline).

Outputs (this folder):
  model_comparison.csv / .md    master metrics table (all models)
  latency_comparison.csv / .md  M4 CPU/GPU latency (measured families)
  hyperparameters.csv / .md      full hyperparameter table (all models)
  figures/confusion_matrix_<key>.png             one per model
  figures/confusion_grid_<family>.png            per-family grid (BiLSTM/Transformer/GRU/RNN)
  figures/metrics_bar.png / roc_curves.png / pr_curves.png
  figures/efficiency_frontier.png / latency_bar.png

Run from the repo root:  python journal_prep/Analysis/00_generate_analysis.py
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import (confusion_matrix, roc_curve, precision_recall_curve,
                             roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, precision_score, recall_score)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

F1OPT = ROOT / "f1_optimization" / "probs_cache"
GRUC = ROOT / "gru" / "phase4_final" / "probs_cache"
RNNC = ROOT / "rnn" / "phase4_final" / "probs_cache"
MULTISEED = ROOT / "journal_prep" / "issue2_clean_protocol" / "kaggle_result" / "runs_multiseed_clean"
SEEDS = [42, 0, 1, 2, 3]

FAM_COLOR = {"BiLSTM": "#0072B2", "Transformer": "#D55E00", "GRU": "#009E73", "RNN": "#CC79A7"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


BiLSTM = _load("m03", ROOT / "src" / "model_bilstm_legacy.py").BiLSTMIntentPredictor
Attn = _load("m07", ROOT / "pipeline" / "07_bilstm_attention.py").BiLSTMAttentionIntentPredictor


class BBoxBiLSTM(nn.Module):
    """The bbox-only (4-D) clean-protocol variant's exact architecture: a plain
    Linear(4,64)+ReLU input projection (module key `input_proj.weight`, ReLU applied
    functionally) → 2-layer BiLSTM(h128) → last-step → head. Verified bit-exact against the
    stored per-seed test AUC (0.777052, seed 42) before use."""

    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(4, 64)
        self.bilstm = nn.LSTM(64, 128, 2, dropout=0.3, bidirectional=True, batch_first=True)
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        out, _ = self.bilstm(torch.relu(self.input_proj(x)))
        return self.head(out[:, -1, :])
eng = _load("eng", ROOT / "journal_prep" / "issue12_unified_pipeline" / "12_unified_engine.py")
Xtr, ytr, Xva, yva, Xte, yte = eng.load_splits()
yte = yte.astype(int)


def _tau(js, arm, ek, tk):
    return float(json.loads(Path(js).read_text())["arms"][arm][ek][tk])


F1A, GRUA, RNNA = (ROOT / "f1_optimization" / "05_final_arms.json",
                   ROOT / "gru" / "phase4_final" / "05_final_arms.json",
                   ROOT / "rnn" / "phase4_final" / "05_final_arms.json")


@torch.no_grad()
def regen_ens(prefix, builder, in_dim):
    """5-seed ensemble probs for a multiseed variant scored from its checkpoints (CPU)."""
    X = Xte[:, :, :in_dim].astype(np.float32)
    acc = []
    for s in SEEDS:
        rd = MULTISEED / f"{prefix}_seed{s}"
        mean, std = np.load(rd / "norm_mean.npy"), np.load(rd / "norm_std.npy")
        ck = torch.load(rd / "best.pt", map_location="cpu", weights_only=False)
        model = builder(); model.load_state_dict(ck["model"]); model.eval()
        Xn = ((X - mean) / std).astype(np.float32)
        out = model(torch.from_numpy(Xn))
        out = out[0] if isinstance(out, tuple) else out
        acc.append(torch.sigmoid(out.squeeze(-1)).numpy())
    return np.mean(acc, axis=0)


def seedmean_from_finaljson(prefix):
    """per-seed-mean acc/auc/f1 (±std) from the 5 stored final.json test blocks."""
    A = {k: [] for k in ("acc", "auc", "f1")}
    for s in SEEDS:
        t = json.loads((MULTISEED / f"{prefix}_seed{s}" / "final.json").read_text())["test"]
        for k in A:
            A[k].append(t[k])
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1))) for k, v in A.items()}


# ---------------------------------------------------------------- model registry
# canon = per-seed-mean (acc,auc,f1) with std; for regen models it is filled from final.json.
BB = seedmean_from_finaljson("bilstm_bbox_only")
AT = seedmean_from_finaljson("bilstm_attention")

MODELS = [
    # ---- BiLSTM family ----
    dict(key="bilstm", family="BiLSTM", name="BiLSTM (baseline)", variant="clean baseline",
         probs=("cache", F1OPT / "lstm_frozen_ens_test.npy"), tau=0.5, params=594_561,
         latency=0.575, select="val AUC", overlay=True, primary=True,
         arch="LSTM", cite="hochreiter1997",
         canon=dict(acc=0.883, acc_sd=0.009, auc=0.932, auc_sd=0.011, f1=0.828, f1_sd=0.012)),
    dict(key="bilstm_bbox", family="BiLSTM", name="BiLSTM bbox-only (4-D)", variant="ablation: no ego-speed",
         probs=("regen", "bilstm_bbox_only", BBoxBiLSTM, 4), tau=0.5,
         params=594_497, latency=None, select="val AUC", overlay=True, primary=False,
         arch="LSTM", cite="hochreiter1997",
         canon=dict(acc=BB["acc"][0], acc_sd=BB["acc"][1], auc=BB["auc"][0], auc_sd=BB["auc"][1],
                    f1=BB["f1"][0], f1_sd=BB["f1"][1])),
    dict(key="bilstm_attn", family="BiLSTM", name="BiLSTM + attention", variant="variant: additive attention",
         probs=("regen", "bilstm_attention", lambda: Attn(input_dim=5), 5), tau=0.5,
         params=611_265, latency=None, select="val AUC", overlay=False, primary=False,
         arch="LSTM + additive attention", cite="bahdanau2015",
         canon=dict(acc=AT["acc"][0], acc_sd=AT["acc"][1], auc=AT["auc"][0], auc_sd=AT["auc"][1],
                    f1=AT["f1"][0], f1_sd=AT["f1"][1])),
    dict(key="bilstm_f1", family="BiLSTM", name="BiLSTM-F1 (h256)", variant="F1-optimised",
         probs=("cache", F1OPT / "lstm_a3_ens_test.npy"), tau=_tau(F1A, "A3", "ens", "tau"),
         params=2_237_313, latency=None, select="val F1 (hybrid)", overlay=True, primary=False,
         arch="LSTM", cite="hochreiter1997",
         canon=dict(acc=0.897, acc_sd=0.006, auc=0.940, auc_sd=0.004, f1=0.844, f1_sd=0.008)),
    # ---- Transformer family ----
    dict(key="transformer", family="Transformer", name="Transformer (searched)", variant="searched (AUC headline)",
         probs=("cache", F1OPT / "tf_frozen_ens_test.npy"), tau=0.5, params=794_241,
         latency=0.459, select="val AUC", overlay=True, primary=True,
         arch="pre-LN Transformer encoder", cite="vaswani2017",
         canon=dict(acc=0.894, acc_sd=0.009, auc=0.950, auc_sd=0.003, f1=0.845, f1_sd=0.013)),
    dict(key="transformer_def", family="Transformer", name="Transformer (default, un-searched)",
         variant="architecture control", probs=("cache", F1OPT / "tf_b4_ens_test.npy"),
         tau=_tau(F1A, "B4", "ens", "tau"), params=268_417, latency=None, select="val F1 (hybrid)",
         overlay=False, primary=False, arch="pre-LN Transformer encoder", cite="vaswani2017",
         canon=dict(acc=0.878, acc_sd=0.006, auc=0.942, auc_sd=0.004, f1=0.821, f1_sd=0.006)),
    dict(key="transformer_f1", family="Transformer", name="Transformer-F1", variant="F1-optimised",
         probs=("cache", F1OPT / "tf_b3_ens_test.npy"), tau=_tau(F1A, "B3", "ens", "tau"),
         params=794_241, latency=0.459, select="val F1 (hybrid)", overlay=True, primary=False,
         arch="pre-LN Transformer encoder", cite="vaswani2017",
         canon=dict(acc=0.896, acc_sd=0.011, auc=0.947, auc_sd=0.003, f1=0.847, f1_sd=0.017)),
    # ---- GRU family ----
    dict(key="gru", family="GRU", name="GRU-F1 (h256)", variant="F1-winner",
         probs=("cache", GRUC / "gru_f1_winner_ens_test.npy"),
         tau=_tau(GRUA, "gru_f1_winner", "ensemble", "tau"), params=1_678_209, latency=0.721,
         select="val F1 (hybrid)", overlay=True, primary=True, arch="GRU", cite="cho2014",
         canon=dict(acc=0.901, acc_sd=0.010, auc=0.941, auc_sd=0.007, f1=0.849, f1_sd=0.011)),
    dict(key="gru_def_f1", family="GRU", name="GRU (default h128, F1)", variant="un-searched control",
         probs=("cache", GRUC / "gru_default_f1_ens_test.npy"),
         tau=_tau(GRUA, "gru_default_f1", "ensemble", "tau"), params=446_081, latency=None,
         select="val F1 (hybrid)", overlay=False, primary=False, arch="GRU", cite="cho2014",
         canon=dict(acc=0.898, acc_sd=0.010, auc=0.939, auc_sd=0.007, f1=0.844, f1_sd=0.020)),
    dict(key="gru_def_auc", family="GRU", name="GRU (default h128, AUC)", variant="matched-size AUC twin",
         probs=("cache", GRUC / "gru_default_auc_ens_test.npy"),
         tau=_tau(GRUA, "gru_default_auc", "ensemble", "tau"), params=446_081, latency=None,
         select="val AUC", overlay=False, primary=False, arch="GRU", cite="cho2014",
         canon=dict(acc=0.898, acc_sd=0.007, auc=0.933, auc_sd=0.010, f1=0.840, f1_sd=0.012)),
    # ---- RNN family ----
    dict(key="rnn", family="RNN", name="Vanilla RNN-F1 (h256)", variant="F1-winner",
         probs=("cache", RNNC / "rnn_f1_winner_ens_test.npy"),
         tau=_tau(RNNA, "rnn_f1_winner", "ensemble", "tau"), params=560_001, latency=0.316,
         select="val F1 (hybrid)", overlay=True, primary=True,
         arch="vanilla (Elman) RNN, tanh", cite="elman1990",
         canon=dict(acc=0.902, acc_sd=0.008, auc=0.948, auc_sd=0.002, f1=0.852, f1_sd=0.012)),
    dict(key="rnn_win_auc", family="RNN", name="Vanilla RNN (winner h256, AUC)", variant="AUC-selected winner",
         probs=("cache", RNNC / "rnn_winner_auc_ens_test.npy"),
         tau=_tau(RNNA, "rnn_winner_auc", "ensemble", "tau"), params=560_001, latency=None,
         select="val AUC", overlay=False, primary=False,
         arch="vanilla (Elman) RNN, tanh", cite="elman1990",
         canon=dict(acc=0.910, acc_sd=0.006, auc=0.948, auc_sd=0.006, f1=0.845, f1_sd=0.022)),
    dict(key="rnn_def_f1", family="RNN", name="Vanilla RNN (default h128, F1)", variant="un-searched control",
         probs=("cache", RNNC / "rnn_default_f1_ens_test.npy"),
         tau=_tau(RNNA, "rnn_default_f1", "ensemble", "tau"), params=149_121, latency=None,
         select="val F1 (hybrid)", overlay=False, primary=False,
         arch="vanilla (Elman) RNN, tanh", cite="elman1990",
         canon=dict(acc=0.897, acc_sd=0.007, auc=0.942, auc_sd=0.007, f1=0.844, f1_sd=0.013)),
    dict(key="rnn_def_auc", family="RNN", name="Vanilla RNN (default h128, AUC)", variant="matched-size AUC twin",
         probs=("cache", RNNC / "rnn_default_auc_ens_test.npy"),
         tau=_tau(RNNA, "rnn_default_auc", "ensemble", "tau"), params=149_121, latency=None,
         select="val AUC", overlay=False, primary=False,
         arch="vanilla (Elman) RNN, tanh", cite="elman1990",
         canon=dict(acc=0.889, acc_sd=0.010, auc=0.942, auc_sd=0.008, f1=0.836, f1_sd=0.021)),
]

HYPER = {
    "bilstm": dict(cell="BiLSTM", w="h128", L=2, do=0.3, lr="1e-3", pw=1.682, ck="best val AUC", thr="0.50"),
    "bilstm_bbox": dict(cell="BiLSTM (4-D input)", w="h128", L=2, do=0.3, lr="1e-3", pw=1.682, ck="best val AUC", thr="0.50"),
    "bilstm_attn": dict(cell="BiLSTM + additive attn", w="h128, attn64", L=2, do=0.3, lr="1e-3", pw=1.682, ck="best val AUC", thr="0.50"),
    "bilstm_f1": dict(cell="BiLSTM", w="h256", L=2, do=0.3, lr="1e-3", pw=1.682, ck="best val F1", thr="val-τ* ≈0.52"),
    "transformer": dict(cell="Transformer (4 heads)", w="d128/ff512", L=4, do=0.1, lr="1e-3", pw=1.682, ck="best val AUC", thr="0.50"),
    "transformer_def": dict(cell="Transformer (4 heads)", w="d128/ff256", L=2, do=0.1, lr="1e-3", pw=1.682, ck="best val F1", thr="val-τ* ≈0.50"),
    "transformer_f1": dict(cell="Transformer (4 heads)", w="d128/ff512", L=4, do=0.1, lr="1e-3", pw="2.5→1.682", ck="best val F1", thr="val-τ* ≈0.65"),
    "gru": dict(cell="BiGRU", w="h256", L=2, do=0.3, lr="5e-4", pw=1.682, ck="best val F1", thr="val-τ* ≈0.53"),
    "gru_def_f1": dict(cell="BiGRU", w="h128", L=2, do=0.3, lr="1e-3", pw=1.682, ck="best val F1", thr="val-τ* ≈0.50"),
    "gru_def_auc": dict(cell="BiGRU", w="h128", L=2, do=0.3, lr="1e-3", pw=1.682, ck="best val AUC", thr="val-τ* ≈0.49"),
    "rnn": dict(cell="Bi vanilla RNN (tanh)", w="h256", L=2, do=0.2, lr="1e-4", pw=1.682, ck="best val F1", thr="val-τ* ≈0.53"),
    "rnn_win_auc": dict(cell="Bi vanilla RNN (tanh)", w="h256", L=2, do=0.2, lr="1e-4", pw=1.682, ck="best val AUC", thr="val-τ* ≈0.50"),
    "rnn_def_f1": dict(cell="Bi vanilla RNN (tanh)", w="h128", L=2, do=0.3, lr="1e-3", pw=1.682, ck="best val F1", thr="val-τ* ≈0.61"),
    "rnn_def_auc": dict(cell="Bi vanilla RNN (tanh)", w="h128", L=2, do=0.3, lr="1e-3", pw=1.682, ck="best val AUC", thr="val-τ* ≈0.55"),
}

LATENCY = {  # family -> measured M4 latency (ms/window)
    "BiLSTM": dict(cpu_b1=0.575, gpu_b1=1.647, cpu_b32=0.135, note="h128", src="Issue 9"),
    "Transformer": dict(cpu_b1=0.459, gpu_b1=1.388, cpu_b32=0.084, note="d128/ff512/L4", src="transformer/phase5"),
    "GRU": dict(cpu_b1=0.721, gpu_b1=None, cpu_b32=None, note="h256", src="gru/phase5"),
    "RNN": dict(cpu_b1=0.316, gpu_b1=4.293, cpu_b32=0.065, note="h256", src="rnn/phase5"),
}


def get_probs(m):
    kind = m["probs"][0]
    if kind == "cache":
        return np.load(m["probs"][1])
    _, prefix, builder, in_dim = m["probs"]
    return regen_ens(prefix, builder, in_dim)


def ens_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(acc=accuracy_score(y, pred), auc=roc_auc_score(y, p),
                pr_auc=average_precision_score(y, p), f1=f1_score(y, pred, zero_division=0),
                prec=precision_score(y, pred, zero_division=0),
                rec=recall_score(y, pred, zero_division=0),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


# ----------------------------------------------------------------------- tables
def build_tables(rows):
    with open(HERE / "model_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "model", "variant", "params", "arch_source", "selection", "tau",
                    "seedmean_acc", "seedmean_auc", "seedmean_f1",
                    "ens_acc", "ens_auc", "ens_f1", "ens_prec", "ens_rec", "ens_pr_auc",
                    "TN", "FP", "FN", "TP"])
        for m in MODELS:
            r, c = rows[m["key"]], m["canon"]
            w.writerow([m["family"], m["name"], m["variant"], m["params"], m["arch"], m["select"],
                        round(m["tau"], 3), c["acc"], c["auc"], c["f1"],
                        round(r["acc"], 4), round(r["auc"], 4), round(r["f1"], 4),
                        round(r["prec"], 4), round(r["rec"], 4), round(r["pr_auc"], 4),
                        r["tn"], r["fp"], r["fn"], r["tp"]])

    L = ["# Model comparison — every model in the four families (clean PIE protocol)", "",
         "Test = PIE **set03**, 2,094 windows (32.5% positive), obs_len 16, TTE∈[30,60]. Two-stream "
         "input (bounding box + ego-speed) unless noted. Every model trained under the identical "
         "frozen protocol (train set01/02/04, val set05/06, pos_weight 1.682, 5 seeds); selection "
         "on validation only, test touched once. **All models are custom architectures trained "
         "from scratch — none are pretrained** (see `README.md` for the per-model academic source).",
         "", "**Per-seed-mean** = mean±std over the 5 seeds (the paper numbers). **Ensemble** = the "
         "5 seeds' averaged probabilities (one deployable predictor; source of the confusion "
         "matrices) — a different, slightly higher statistic.", "",
         "| family | model | params | source (cite) | selection | Acc | AUC | **F1** |",
         "|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        c = m["canon"]; star = " ⭐" if m["primary"] else ""
        L.append(f"| {m['family']} | **{m['name']}**{star} | {m['params']:,} | {m['arch']} | "
                 f"{m['select']} | {c['acc']:.3f} ± {c['acc_sd']:.3f} | {c['auc']:.3f} ± "
                 f"{c['auc_sd']:.3f} | **{c['f1']:.3f} ± {c['f1_sd']:.3f}** |")
    L += ["", "⭐ = the four headline models (one per family). Ablation sweeps (window/TTE/"
          "hidden-size/depth/grid — same architecture, swept settings) are catalogued in "
          "`README.md`, not repeated here.", "",
          "## Ensemble @ operating threshold τ (source of the confusion matrices)", "",
          "| model | τ | Acc | AUC | F1 | Prec | Rec | PR-AUC | TN / FP / FN / TP |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        r = rows[m["key"]]
        L.append(f"| {m['name']} | {m['tau']:.3f} | {r['acc']:.3f} | {r['auc']:.3f} | {r['f1']:.3f} | "
                 f"{r['prec']:.3f} | {r['rec']:.3f} | {r['pr_auc']:.3f} | "
                 f"{r['tn']} / {r['fp']} / {r['fn']} / {r['tp']} |")
    L += [""]
    (HERE / "model_comparison.md").write_text("\n".join(L))

    # latency
    with open(HERE / "latency_comparison.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["family", "cpu_b1_ms", "gpu_b1_ms", "cpu_b32_ms", "note", "src"])
        for fam, d in LATENCY.items():
            w.writerow([fam, d["cpu_b1"], d["gpu_b1"], d["cpu_b32"], d["note"], d["src"]])
    L = ["# Latency comparison (Apple M4, classifier forward only)", "",
         "Issue-9 protocol: 50 warmup + 1000 timed forwards, `torch.mps.synchronize()` inside each "
         "timed MPS call. **Quote CPU batch-1** — the honest single-window latency (GPU launch "
         "overhead dominates a sub-million-param model at batch 1). 30 fps budget = 33.3 ms/frame.",
         "", "| family (measured model) | **CPU batch-1** (ms/win) | GPU batch-1 | CPU batch-32 | "
         "×inside 30 fps | source |", "|---|---|---|---|---|---|"]
    for fam, d in LATENCY.items():
        g = f"{d['gpu_b1']:.3f}" if d["gpu_b1"] else "—"
        c32 = f"{d['cpu_b32']:.3f}" if d["cpu_b32"] else "—"
        L.append(f"| {fam} ({d['note']}) | **{d['cpu_b1']:.3f}** | {g} | {c32} | "
                 f"~{33.3/d['cpu_b1']:.0f}× | {d['src']} |")
    L += ["", "One representative model per family was timed (latency is weight-driven, so it is "
          "reported per family, not per variant). All four are ~2 orders of magnitude inside the "
          "frame budget — latency is not a deployment discriminator; the live YOLO+ByteTrack "
          "pipeline is detection-bound (Issue 9). The **vanilla RNN is fastest** (un-gated cell = "
          "smallest); the GRU is slowest only because its F1-winner is the largest model.", ""]
    (HERE / "latency_comparison.md").write_text("\n".join(L))

    # hyperparameters
    cols = ["cell", "w", "L", "do", "lr", "pw", "ck", "thr"]
    hdr = {"cell": "cell", "w": "width", "L": "layers", "do": "dropout", "lr": "lr",
           "pw": "pos_weight", "ck": "checkpoint", "thr": "threshold"}
    with open(HERE / "hyperparameters.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model"] + [hdr[c] for c in cols])
        for m in MODELS:
            h = HYPER[m["key"]]; w.writerow([m["name"]] + [h[c] for c in cols])
    L = ["# Hyperparameters — every model", "",
         "Frozen-identical across families (batch 32, ≤100 epochs, patience-15 early stop on val "
         "AUC, train-only z-score, Adam, weight-decay 1e-5, pos_weight-1.682 anchor). Below are "
         "each model's *searched* knobs. `width` = recurrent read-out (2×hidden) or transformer "
         "d_model/dim_ff.", "",
         "| model | cell | width | layers | dropout | lr | pos_weight | checkpoint | threshold |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        h = HYPER[m["key"]]
        L.append(f"| **{m['name']}** | {h['cell']} | {h['w']} | {h['L']} | {h['do']} | {h['lr']} | "
                 f"{h['pw']} | {h['ck']} | {h['thr']} |")
    L += ["", "The GRU and vanilla RNN received the **identical** Issue-8 grid search + pos_weight "
          "sweep the BiLSTM did — same budget, so the only thing differing across the recurrent "
          "families is the cell type.", ""]
    (HERE / "hyperparameters.md").write_text("\n".join(L))


# ----------------------------------------------------------------------- figures
def _cm_panel(ax, m, y, p, small=False):
    thr = m["tau"]; pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1]); total = cm.sum()
    im = ax.imshow(cm, cmap="Blues")
    fs = 10 if small else 12
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{v}\n({v/total*100:.1f}%)", ha="center", va="center",
                    fontsize=fs, fontweight="bold",
                    color="white" if v > cm.max() * 0.6 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No-cross", "Cross"]); ax.set_yticklabels(["No-cross", "Cross"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    f1 = f1_score(y, pred, zero_division=0)
    ax.set_title(f"{m['name']}\nF1={f1:.3f}  ·  τ={thr:.2f}  ·  5-seed ensemble",
                 fontsize=9 if small else 10, fontweight="bold", color=FAM_COLOR[m["family"]])
    return im


def fig_confusion(m, y, p):
    fig, ax = plt.subplots(figsize=(4.3, 3.9), facecolor="white")
    im = _cm_panel(ax, m, y, p)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(FIG / f"confusion_matrix_{m['key']}.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig_family_grid(fam, y, probs):
    fam_models = [m for m in MODELS if m["family"] == fam]
    n = len(fam_models); cols = 2; rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(8, 3.7 * rows), facecolor="white")
    axes = np.atleast_1d(axes).ravel()
    for m, ax in zip(fam_models, axes):
        _cm_panel(ax, m, y, probs[m["key"]], small=True)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{fam} family — confusion matrices (test set03, 5-seed ensemble)",
                 fontsize=12, fontweight="bold", color=FAM_COLOR[fam])
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIG / f"confusion_grid_{fam.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig_metrics_bar():
    prim = [m for m in MODELS if m["primary"]]
    labels = [m["name"].split(" (")[0] for m in prim]
    metrics = [("Accuracy", "acc"), ("AUC", "auc"), ("F1", "f1")]
    x = np.arange(len(prim)); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    shades = {"Accuracy": 0.55, "AUC": 0.78, "F1": 1.0}
    for gi, (ml, mk) in enumerate(metrics):
        vals = [m["canon"][mk] for m in prim]; cols = [FAM_COLOR[m["family"]] for m in prim]
        bars = ax.bar(x + (gi - 1) * w, vals, w, color=cols, alpha=shades[ml],
                      edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center",
                    va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.80, 0.97); ax.set_ylabel("score (per-seed mean)")
    ax.set_title("Per-seed-mean metrics — four headline families (test set03)\n"
                 "hue = family; opacity = metric (light→Acc, mid→AUC, solid→F1)",
                 fontweight="bold", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=[Patch(facecolor=FAM_COLOR[m["family"]], label=m["family"]) for m in prim],
              loc="upper left", ncol=4, fontsize=9)
    plt.tight_layout(); plt.savefig(FIG / "metrics_bar.png", dpi=150, bbox_inches="tight"); plt.close()


def fig_roc(y, probs):
    over = [m for m in MODELS if m["overlay"]]
    fig, ax = plt.subplots(figsize=(6.4, 6), facecolor="white")
    for m in over:
        fpr, tpr, _ = roc_curve(y, probs[m["key"]]); auc = roc_auc_score(y, probs[m["key"]])
        ls = "-" if m["primary"] else ("-." if "bbox" in m["key"] else "--")
        ax.plot(fpr, tpr, ls, color=FAM_COLOR[m["family"]], lw=1.8,
                label=f"{m['name']} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="gray", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — representative models (test set03, 5-seed ensemble)",
                 fontweight="bold", fontsize=11)
    ax.legend(loc="lower right", fontsize=7.5); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG / "roc_curves.png", dpi=150, bbox_inches="tight"); plt.close()


def fig_pr(y, probs):
    over = [m for m in MODELS if m["overlay"]]
    fig, ax = plt.subplots(figsize=(6.4, 6), facecolor="white")
    for m in over:
        prec, rec, _ = precision_recall_curve(y, probs[m["key"]])
        ap = average_precision_score(y, probs[m["key"]])
        ls = "-" if m["primary"] else ("-." if "bbox" in m["key"] else "--")
        ax.plot(rec, prec, ls, color=FAM_COLOR[m["family"]], lw=1.8,
                label=f"{m['name']} (PR-AUC {ap:.3f})")
    ax.axhline(y.mean(), ls=":", color="gray", lw=1, label=f"chance ({y.mean():.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves — representative models (test set03)",
                 fontweight="bold", fontsize=11)
    ax.legend(loc="lower left", fontsize=7.5); ax.grid(alpha=0.3); ax.set_ylim(0.3, 1.0)
    plt.tight_layout(); plt.savefig(FIG / "pr_curves.png", dpi=150, bbox_inches="tight"); plt.close()


def fig_efficiency():
    prim = [m for m in MODELS if m["primary"]]
    fig, ax = plt.subplots(figsize=(8.2, 5.6), facecolor="white")
    for m in prim:
        lat = m["latency"]; size = 320 if lat is None else lat * 900
        ax.scatter(m["params"] / 1e6, m["canon"]["f1"], s=size, color=FAM_COLOR[m["family"]],
                   alpha=0.75, edgecolor="black", linewidth=0.8, zorder=3)
        latlbl = "n/a" if lat is None else f"{lat:.2f} ms"
        ax.annotate(f"{m['name'].split(' (')[0]}\n{latlbl}", (m["params"] / 1e6, m["canon"]["f1"]),
                    textcoords="offset points", xytext=(9, 6), fontsize=9)
    ax.set_xscale("log"); ax.set_xlabel("parameters (millions, log scale)")
    ax.set_ylabel("test F1 (per-seed mean)")
    ax.set_title("Efficiency frontier: parameters vs. F1  (marker size ∝ CPU batch-1 latency)",
                 fontweight="bold", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(handles=[Patch(facecolor=c, label=f) for f, c in FAM_COLOR.items()],
              loc="lower right", fontsize=9, title="family")
    plt.tight_layout(); plt.savefig(FIG / "efficiency_frontier.png", dpi=150, bbox_inches="tight"); plt.close()


def fig_latency_bar():
    fams = list(LATENCY.keys()); cpu = [LATENCY[f]["cpu_b1"] for f in fams]
    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor="white")
    bars = ax.bar(fams, cpu, color=[FAM_COLOR[f] for f in fams], edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, cpu):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f} ms", ha="center",
                va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("CPU batch-1 latency (ms/window)")
    ax.set_title("Single-window inference latency by family (Apple M4, CPU)\n"
                 "all ~100× inside a 30 fps budget — the vanilla RNN is fastest",
                 fontweight="bold", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG / "latency_bar.png", dpi=150, bbox_inches="tight"); plt.close()


def main():
    assert np.array_equal(yte, np.load(F1OPT / "y_test.npy").astype(int))
    assert np.array_equal(yte, np.load(GRUC / "y_test.npy").astype(int))
    assert np.array_equal(yte, np.load(RNNC / "y_test.npy").astype(int))
    probs = {m["key"]: get_probs(m) for m in MODELS}
    for k, p in probs.items():
        assert p.shape == yte.shape, f"{k}: {p.shape} != {yte.shape}"
    rows = {m["key"]: ens_metrics(yte, probs[m["key"]], m["tau"]) for m in MODELS}

    print(f"{'model':<34} {'τ':>5} {'ensF1':>6} {'ensAUC':>7} {'Acc':>6} {'Prec':>6} {'Rec':>6}")
    for m in MODELS:
        r = rows[m["key"]]
        print(f"  {m['name']:<32} {m['tau']:.2f}  {r['f1']:.3f}  {r['auc']:.3f}  "
              f"{r['acc']:.3f} {r['prec']:.3f} {r['rec']:.3f}")

    build_tables(rows)
    for m in MODELS:
        fig_confusion(m, yte, probs[m["key"]])
    for fam in FAM_COLOR:
        fig_family_grid(fam, yte, probs)
    fig_metrics_bar(); fig_roc(yte, probs); fig_pr(yte, probs)
    fig_efficiency(); fig_latency_bar()
    nfig = len(MODELS) + len(FAM_COLOR) + 5
    print(f"\nwrote 3 tables + {nfig} figures ({len(MODELS)} confusion matrices + "
          f"{len(FAM_COLOR)} family grids + 5 overlays)")


if __name__ == "__main__":
    main()
