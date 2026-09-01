"""Issue 5 — Leave-One-Set-Out cross-validation (baseline 5-D BiLSTM).

The main result uses one fixed split (test=set03). PIE has 6 recording sets;
rotating the held-out set across all 6 and averaging is a much stronger
generalization claim ("set03 isn't just an easy fold"). For each fold:
  test = set_i ; train+val = the other 5 sets, split 85/15 by **pedestrian**
  (grouped — no ped's windows span train and val), early-stop on val AUC,
  per-fold train-only normalization, per-fold pos_weight = n_neg/n_pos.
Identical architecture/hyperparameters to the baseline. Report per-fold test
metrics + unweighted mean ± std over the 6 folds (sets differ in size, so the
unweighted mean is the headline; per-fold N is reported alongside).

Local, CPU, ~3-5 min (baseline is 0.6M params; no backend effect for the 5-D model).
"""
import argparse, csv, importlib.util, pickle, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

HERE = Path(__file__).resolve().parent
I2 = HERE.parent / "issue2_clean_protocol"
SEQ_DIR = I2 / "sequences_clean"
_spec = importlib.util.spec_from_file_location("v6b", I2 / "06b_local_verify_seed42.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
BiLSTM = _m.BiLSTMIntentPredictor   # the locked 5-D baseline architecture

SETS = ["set01", "set02", "set03", "set04", "set05", "set06"]
EPOCHS, BATCH, LR, WD, PATIENCE, THR, VAL_FRAC = 100, 32, 1e-3, 1e-5, 15, 0.5, 0.15


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def load():
    X = np.load(SEQ_DIR / "X.npy").astype(np.float32)
    y = np.load(SEQ_DIR / "y.npy").astype(np.float32)
    meta = pickle.load(open(SEQ_DIR / "meta.pkl", "rb"))
    sid = np.array([r["set_id"] for r in meta])
    pid = np.array([f"{r['set_id']}/{r['ped_id']}" for r in meta])  # globally-unique ped key
    return X, y, sid, pid


@torch.no_grad()
def evaluate(model, X, y, device):
    model.eval()
    p = torch.sigmoid(model(torch.from_numpy(X).to(device)).squeeze(-1)).cpu().numpy()
    pred = (p >= THR).astype(int)
    return dict(auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                f1=f1_score(y, pred, zero_division=0), acc=accuracy_score(y, pred),
                prec=precision_score(y, pred, zero_division=0),
                rec=recall_score(y, pred, zero_division=0))


def train_fold(test_set, X, y, sid, pid, seed, device):
    set_seed(seed)
    te = sid == test_set
    pool = ~te
    # pedestrian-grouped 85/15 train/val split within the 5 non-test sets
    pool_peds = np.unique(pid[pool])
    rng = np.random.default_rng(seed); rng.shuffle(pool_peds)
    n_val = max(1, int(round(len(pool_peds) * VAL_FRAC)))
    val_peds = set(pool_peds[:n_val].tolist())
    is_val = np.array([p in val_peds for p in pid])
    va, tr = pool & is_val, pool & ~is_val

    Xtr, ytr, Xva, yva, Xte, yte = X[tr], y[tr], X[va], y[va], X[te], y[te]
    mean = Xtr.reshape(-1, 5).mean(0); std = Xtr.reshape(-1, 5).std(0) + 1e-6
    Xtr, Xva, Xte = (Xtr - mean) / std, (Xva - mean) / std, (Xte - mean) / std
    pw = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))

    model = BiLSTM(input_dim=5).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=BATCH, shuffle=True)

    best_auc, best_state, best_ep, noimp = -1.0, None, 0, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb).squeeze(-1), yb).backward(); opt.step()
        vauc = evaluate(model, Xva, yva, device)["auc"]
        sched.step(vauc)
        if vauc > best_auc:
            best_auc, best_ep, noimp = vauc, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            noimp += 1
            if noimp >= PATIENCE:
                break

    model.load_state_dict(best_state)
    m = evaluate(model, Xte, yte, device)
    m.update(test_set=test_set, test_n=int(te.sum()), test_pos=float(yte.mean()),
             train_n=int(tr.sum()), val_n=int(va.sum()), pos_weight=round(pw, 3),
             best_epoch=best_ep, val_auc=round(best_auc, 4))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = torch.device(args.device)

    X, y, sid, pid = load()
    print(f"loaded {len(X)} windows | device {device} | seed {args.seed}\n")
    print(f"{'fold(test)':12s} {'testN':>6s} {'pos%':>5s} {'AUC':>6s} {'PR':>6s} "
          f"{'F1':>6s} {'Acc':>6s} {'pw':>5s} {'ep':>3s}")
    print("-" * 64)
    rows = []
    for s in SETS:
        m = train_fold(s, X, y, sid, pid, args.seed, device)
        rows.append(m)
        print(f"{s:12s} {m['test_n']:6d} {m['test_pos']*100:5.1f} {m['auc']:6.3f} "
              f"{m['pr_auc']:6.3f} {m['f1']:6.3f} {m['acc']:6.3f} {m['pos_weight']:5.2f} "
              f"{m['best_epoch']:3d}")

    aucs = np.array([m["auc"] for m in rows])
    prs = np.array([m["pr_auc"] for m in rows])
    f1s = np.array([m["f1"] for m in rows])
    print("-" * 64)
    print(f"{'MEAN±STD':12s} {'':>6s} {'':>5s} {aucs.mean():6.3f} {prs.mean():6.3f} "
          f"{f1s.mean():6.3f}   (AUC std {aucs.std():.3f})")

    # CSV
    cols = ["test_set", "test_n", "test_pos", "train_n", "val_n", "pos_weight",
            "best_epoch", "val_auc", "auc", "pr_auc", "f1", "acc", "prec", "rec"]
    with open(HERE / "05_loso_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for m in rows:
            w.writerow({c: m[c] for c in cols})

    # Markdown
    big = [m for m in rows if m["test_n"] >= 100]
    abig = np.array([m["auc"] for m in big])
    L = [f"# Issue 5 — Leave-One-Set-Out CV (baseline 5-D BiLSTM)", "",
         f"Seed {args.seed}; per-fold test = one PIE set, train+val = other 5 "
         f"(85/15 pedestrian-grouped split, early-stop on val AUC), per-fold "
         f"normalization + pos_weight. Identical architecture/hyperparameters to the "
         f"baseline.", "",
         "| Fold (test set) | test N | pos % | AUC | PR-AUC | F1 | Acc | pos_w | best ep |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in rows:
        L.append(f"| {m['test_set']} | {m['test_n']} | {m['test_pos']*100:.1f} | "
                 f"{m['auc']:.3f} | {m['pr_auc']:.3f} | {m['f1']:.3f} | {m['acc']:.3f} | "
                 f"{m['pos_weight']:.2f} | {m['best_epoch']} |")
    L += ["",
          f"**Mean ± std over 6 folds (unweighted): AUC {aucs.mean():.3f} ± {aucs.std():.3f}, "
          f"PR-AUC {prs.mean():.3f} ± {prs.std():.3f}, F1 {f1s.mean():.3f} ± {f1s.std():.3f}.**",
          "",
          f"Excluding only the tiny, uninterpretable set05 fold (N=47), the "
          f"{len(big)} folds with N≥100: AUC {abig.mean():.3f} ± {abig.std():.3f}.", ""]
    (HERE / "05_loso_results.md").write_text("\n".join(L) + "\n")
    print("\nwrote 05_loso_results.csv + .md")


if __name__ == "__main__":
    main()
