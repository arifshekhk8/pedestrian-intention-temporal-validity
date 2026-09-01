"""
04_multiseed_baseline.py  —  Journal-prep Issue 2 hardening: multi-seed the clean
5-D baseline so the headline is 0.913 ± σ, not a single lucky seed.

A reviewer will not accept a single-seed test number. This re-runs
`04_train_bilstm.py` on the clean sequences across N seeds (seed 42 is the
canonical run and should reproduce the existing 0.9131 exactly — also a
determinism check), parses each run's final.json, and reports mean ± std for
every test metric.

Outputs (next to this script):
  runs_clean/multiseed/seed<k>/        one training run per seed
  04_multiseed_results.csv             per-seed test metrics
  04_multiseed_summary.md              mean ± std table

Run AFTER the bbox-only/attention retrains finish (CPU contention). ~2 min/seed.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SEQ_DIR = HERE / "sequences_clean"
OUT_ROOT = HERE / "runs_clean" / "multiseed"
TRAIN = ROOT / "pipeline" / "04_train_bilstm.py"  # moved in the GitHub reorg
POS_WEIGHT = 1.682   # clean train split neg/pos = 1366/812

METRICS = ["auc", "f1", "acc", "prec", "rec"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse a seed's final.json if it already exists")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        out_dir = OUT_ROOT / f"seed{seed}"
        final = out_dir / "final.json"
        if not (args.skip_existing and final.exists()):
            print(f"\n===== training seed {seed} =====")
            subprocess.run([
                sys.executable, str(TRAIN),
                "--seq_dir", str(SEQ_DIR),
                "--out_dir", str(out_dir),
                "--pos_weight", str(POS_WEIGHT),
                "--epochs", str(args.epochs),
                "--seed", str(seed),
            ], check=True, cwd=str(ROOT))
        with open(final) as f:
            d = json.load(f)
        row = {"seed": seed, "best_epoch": d["best_epoch"]}
        row.update({m: d["test"][m] for m in METRICS})
        rows.append(row)

    # ---- write CSV ----
    import csv
    csv_path = HERE / "04_multiseed_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "best_epoch"] + METRICS)
        w.writeheader()
        w.writerows(rows)

    # ---- summary ----
    def ms(metric):
        vals = np.array([r[metric] for r in rows])
        return vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0

    lines = [
        "# Issue 2 hardening — Multi-seed clean baseline (5-D)\n",
        f"`04_train_bilstm.py` on `sequences_clean/`, pos_weight {POS_WEIGHT}, "
        f"seeds {args.seeds}. Test split = set03 (2,094 windows), touched once per "
        "seed on the best-val checkpoint.\n",
        "## Per-seed test metrics\n",
        "| seed | best epoch | AUC | F1 | Acc | P | R |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['best_epoch']} | {r['auc']:.4f} | "
                     f"{r['f1']:.4f} | {r['acc']:.4f} | {r['prec']:.4f} | {r['rec']:.4f} |")
    lines.append("\n## Mean ± std\n")
    lines.append("| metric | mean | std |")
    lines.append("|---|---|---|")
    for m in METRICS:
        mu, sd = ms(m)
        lines.append(f"| {m} | {mu:.4f} | {sd:.4f} |")
    auc_mu, auc_sd = ms("auc")
    lines.append(f"\n**Headline: test AUC = {auc_mu:.3f} ± {auc_sd:.3f}** across "
                 f"{len(rows)} seeds on the clean, leak-free protocol.")
    (HERE / "04_multiseed_summary.md").write_text("\n".join(lines))

    print("\n================ MULTI-SEED SUMMARY ================")
    for m in METRICS:
        mu, sd = ms(m)
        print(f"{m:5s}: {mu:.4f} ± {sd:.4f}")
    print(f"\nwrote {csv_path.name} and 04_multiseed_summary.md")


if __name__ == "__main__":
    main()
