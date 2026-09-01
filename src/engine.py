"""12_unified_engine.py — Issue 12: THE single, model-agnostic training pipeline.

One training loop for every sequence-model family compared in this thesis, so the
comparison is fair by construction: the entire protocol below is FROZEN and identical
for every family; only the model builder and the cfg-driven recipe differ.

FROZEN (identical for all families — do not parameterize):
  data          journal_prep/issue2_clean_protocol/sequences_clean/ (N=4906)
  splits        train {set01,02,04} 2178 / val {set05,06} 634 / test {set03} 2094
  normalization train-only per-feature z-score (std + 1e-6), saved per run
  loss          BCEWithLogitsLoss(pos_weight=1.682 unless a study sweeps it)
  batch         32 (shuffle from the seeded global torch RNG, num_workers=0)
  epochs        <=100, early stop patience 15 on val AUC
  scheduler     ReduceLROnPlateau(max, factor 0.5, patience 5) on val AUC
                (or cfg schedule="warmup_cosine": 5-epoch warmup + cosine to 1%)
  checkpoint    select="f1": best val F1 (tie: acc, then AUC) — the F1-first rule
                (f1_optimization/PLAN.md §3.3); select="auc": best val AUC (the
                original frozen rule). Both record `val_at_auc_best` for the G1 audit.
  metrics       sklearn f1/acc/auc/pr_auc/prec/rec at threshold 0.5, full-batch eval
  seeds         [42, 0, 1, 2, 3]; test set03 touched only by a designated final script

PARAMETERIZED (per family / per study):
  model         MODEL_REGISTRY[family](cfg) — bilstm | transformer | gru | birnn
  recipe        cfg: lr (required); weight_decay (default 1e-5); optimizer
                "adam" | "adamw" (AdamW uses no-decay groups for bias/norm/pos/cls);
                schedule "plateau" | "warmup_cosine"

Provenance: refactored from f1_optimization/00_train_engines.py (itself forked from
issue8's grid train() and transformer/phase1_setup/02_train_transformer.py, which are
kept frozen as the artifacts that produced their published runs). Bit-equivalence with
the f1_optimization runs is PROVEN by 12_equivalence_check.py — this engine reproduces
those cells exactly, so it is the same pipeline, now with one code path for every
family. GRU / bidirectional-RNN builders are registered and smoke-tested but not yet
part of any published result (planned follow-up).

CLI (val-only; never evaluates test):
  python journal_prep/issue12_unified_pipeline/12_unified_engine.py \
      --family gru --seed 42 --device mps --select f1 --out_dir <dir>
"""
import argparse
import importlib.util
import json
import math
import pickle
import random
import time
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Shipped, leak-free PIE windows (1.7 MB, tracked in git). Override with --seq_dir.
SEQ_DIR = Path(os.environ.get("PCIP_SEQ_DIR", ROOT / "data" / "pie_clean"))

TRAIN_SETS = {"set01", "set02", "set04"}
VAL_SETS = {"set05", "set06"}
TEST_SETS = {"set03"}
POS_WEIGHT = 1.682
SEEDS = [42, 0, 1, 2, 3]
EPOCHS = 100
BATCH = 32
PATIENCE = 15
PLATEAU_PATIENCE = 5
THRESHOLD = 0.5
WARMUP_EPOCHS = 5
COSINE_MIN_FRAC = 0.01


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BiLSTM = _load_module("m_bilstm", HERE / "model_bilstm_legacy.py").BiLSTMIntentPredictor
_tfm = _load_module("m_transformer", HERE / "model_transformer.py")
TransformerIntentPredictor = _tfm.TransformerIntentPredictor


class RecurrentIntentPredictor(nn.Module):
    """GRU / vanilla-RNN twin of BiLSTMIntentPredictor (pipeline/03): identical
    input projection, bidirectional recurrence, last-step readout, linear head —
    only the recurrent cell differs, so a future GRU/RNN comparison isolates the
    cell type exactly the way the transformer comparison isolated attention."""

    def __init__(self, cell="gru", input_dim=5, proj_dim=64, hidden_dim=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, proj_dim), nn.ReLU())
        rnn_cls = {"gru": nn.GRU, "rnn": nn.RNN}[cell]
        kwargs = dict(input_size=proj_dim, hidden_size=hidden_dim, num_layers=num_layers,
                      dropout=dropout, bidirectional=True, batch_first=True)
        if cell == "rnn":
            kwargs["nonlinearity"] = "tanh"
        self.rnn = rnn_cls(**kwargs)
        self.head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.rnn(self.input_proj(x))
        return self.head(out[:, -1, :])


# ---------------------------------------------------------------- registry
def _build_bilstm(cfg):
    return BiLSTM(input_dim=5, hidden_dim=cfg["hidden"],
                  num_layers=cfg["num_layers"], dropout=cfg["dropout"])


def _build_transformer(cfg):
    return TransformerIntentPredictor(
        input_dim=5, d_model=cfg["d_model"], nhead=cfg.get("nhead", 4),
        num_layers=cfg["num_layers"], dim_ff=cfg["dim_ff"], dropout=cfg["dropout"],
        pool=cfg["pool"], pos=cfg["pos"])


def _build_gru(cfg):
    return RecurrentIntentPredictor("gru", hidden_dim=cfg["hidden"],
                                    num_layers=cfg["num_layers"], dropout=cfg["dropout"])


def _build_birnn(cfg):
    return RecurrentIntentPredictor("rnn", hidden_dim=cfg["hidden"],
                                    num_layers=cfg["num_layers"], dropout=cfg["dropout"])


MODEL_REGISTRY = {
    "bilstm": _build_bilstm,
    "transformer": _build_transformer,
    "gru": _build_gru,        # registered, smoke-tested; no published result yet
    "birnn": _build_birnn,    # registered, smoke-tested; no published result yet
}

# reference configs (the two published headline families)
BILSTM_BASELINE_CFG = dict(lr=1e-3, dropout=0.3, hidden=128, num_layers=2)
BILSTM_F1_CFG = dict(lr=1e-3, dropout=0.3, hidden=256, num_layers=2)  # f1_optimization A3
TRANSFORMER_SEARCHED_CFG = dict(d_model=128, nhead=4, num_layers=4, dim_ff=512,
                                dropout=0.1, pool="last", pos="sin",
                                lr=1e-3, schedule="plateau", weight_decay=1e-5,
                                optimizer="adam")
GRU_DEFAULT_CFG = dict(lr=1e-3, dropout=0.3, hidden=128, num_layers=2)
PRESETS = {"bilstm": BILSTM_BASELINE_CFG, "transformer": TRANSFORMER_SEARCHED_CFG,
           "gru": GRU_DEFAULT_CFG, "birnn": GRU_DEFAULT_CFG}


def set_seed(s):
    """THE one seeding function (audit finding SEEDING-DIVERGENCE): seeds every RNG and
    sets the cuDNN determinism flags unconditionally (no-ops off-CUDA), so the same
    call is fully deterministic on CUDA too — never redefine this per program."""
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_splits():
    X = np.load(SEQ_DIR / "X.npy").astype(np.float32)
    y = np.load(SEQ_DIR / "y.npy").astype(np.float32)
    with open(SEQ_DIR / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    sid = np.array([m["set_id"] for m in meta])
    assert X.shape == (4906, 16, 5), f"unexpected X shape {X.shape}"
    tr, va, te = (np.isin(sid, sorted(s)) for s in (TRAIN_SETS, VAL_SETS, TEST_SETS))
    out = (X[tr], y[tr], X[va], y[va], X[te], y[te])
    assert len(out[1]) == 2178 and len(out[3]) == 634 and len(out[5]) == 2094
    return out


def metrics_at(y, probs, thr=THRESHOLD):
    pred = (probs >= thr).astype(int)
    return dict(f1=f1_score(y, pred, zero_division=0),
                acc=accuracy_score(y, pred),
                auc=roc_auc_score(y, probs),
                pr_auc=average_precision_score(y, probs),
                prec=precision_score(y, pred, zero_division=0),
                rec=recall_score(y, pred, zero_division=0))


@torch.no_grad()
def _val_metrics(model, Xva_n, yva, device):
    model.eval()
    p = torch.sigmoid(model(torch.from_numpy(Xva_n).to(device)).squeeze(-1)).cpu().numpy()
    return metrics_at(yva, p)


def _f1_key(m):
    return (round(m["f1"], 10), round(m["acc"], 10), m["auc"])


def _build_optimizer(model, cfg):
    lr = cfg["lr"]
    wd = cfg.get("weight_decay", 1e-5)
    if cfg.get("optimizer", "adam") == "adamw":
        no_decay_keys = ("bias", "norm", "pos_embed", "cls_token")
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if any(k in name.lower() for k in no_decay_keys) else decay).append(p)
        return torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                                  {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def _lr_for_epoch(epoch, base_lr):
    if epoch <= WARMUP_EPOCHS:
        return base_lr * epoch / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return base_lr * (COSINE_MIN_FRAC + (1 - COSINE_MIN_FRAC) * 0.5
                      * (1 + math.cos(math.pi * progress)))


def train_run(family, cfg, seed, device, data, pos_weight=None, select="f1",
              out_dir=None):
    """The one loop. VAL-ONLY: this function cannot evaluate test (no test code path);
    final test evaluation belongs to a designated single script per study."""
    assert family in MODEL_REGISTRY, f"unknown family {family!r}"
    assert select in ("f1", "auc")
    set_seed(seed)
    Xtr, ytr, Xva, yva, _, _ = data
    pw = POS_WEIGHT if pos_weight is None else pos_weight

    flat = Xtr.reshape(-1, Xtr.shape[-1])
    mean, std = flat.mean(axis=0), flat.std(axis=0) + 1e-6
    Xtr_n, Xva_n = (Xtr - mean) / std, (Xva - mean) / std

    model = MODEL_REGISTRY[family](cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    opt = _build_optimizer(model, cfg)
    schedule = cfg.get("schedule", "plateau")
    plateau = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=PLATEAU_PATIENCE)
        if schedule == "plateau" else None)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr_n), torch.from_numpy(ytr)),
                        batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=False)

    best = {"key": None, "state": None, "epoch": 0, "metrics": None}
    auc_best = {"auc": -1.0, "epoch": 0, "metrics": None}
    noimp, history, t0 = 0, [], time.time()
    for ep in range(1, EPOCHS + 1):
        if schedule == "warmup_cosine":
            new_lr = _lr_for_epoch(ep, cfg["lr"])
            for g in opt.param_groups:
                g["lr"] = new_lr
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb).squeeze(-1), yb).backward()
            opt.step()

        m = _val_metrics(model, Xva_n, yva, device)
        if plateau is not None:
            plateau.step(m["auc"])
        history.append({"epoch": ep, **{f"val_{k}": v for k, v in m.items()}})

        track_f1 = (select == "f1")
        key = _f1_key(m)
        if track_f1 and (best["key"] is None or key > best["key"]):
            best.update(key=key, epoch=ep, metrics=m,
                        state={k: v.detach().clone() for k, v in model.state_dict().items()})
        if m["auc"] > auc_best["auc"]:
            auc_best.update(auc=m["auc"], epoch=ep, metrics=m)
            if not track_f1:
                best.update(key=None, epoch=ep, metrics=m,
                            state={k: v.detach().clone() for k, v in model.state_dict().items()})
            noimp = 0
        else:
            noimp += 1
            if noimp >= PATIENCE:
                break

    model.load_state_dict(best["state"])
    result = dict(family=family, cfg=cfg, seed=seed, pos_weight=pw, device=str(device),
                  torch_version=torch.__version__,
                  select=select, n_params=int(n_params), best_epoch=best["epoch"],
                  val=best["metrics"], auc_best_epoch=auc_best["epoch"],
                  val_at_auc_best=auc_best["metrics"],
                  seconds=round(time.time() - t0, 1))
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": best["state"], "epoch": best["epoch"],
                    "val_metrics": best["metrics"]}, out_dir / "best.pt")
        np.save(out_dir / "norm_mean.npy", mean)
        np.save(out_dir / "norm_std.npy", std)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        (out_dir / "final.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--select", default="f1", choices=["f1", "auc"])
    ap.add_argument("--pos_weight", type=float, default=None)
    ap.add_argument("--cfg_json", default=None,
                    help="JSON cfg dict; defaults to the family preset")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    cfg = json.loads(args.cfg_json) if args.cfg_json else PRESETS[args.family]
    data = load_splits()
    r = train_run(args.family, cfg, args.seed, torch.device(args.device), data,
                  pos_weight=args.pos_weight, select=args.select, out_dir=args.out_dir)
    print(json.dumps({k: v for k, v in r.items() if k != "cfg"}, indent=2))


if __name__ == "__main__":
    main()
