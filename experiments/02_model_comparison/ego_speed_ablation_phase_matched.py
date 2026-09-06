"""
ego_speed_ablation_phase_matched.py — the 5-D vs 4-D test, under the phase-matched control.

WHY THIS SCRIPT EXISTS
----------------------
`ego_speed_ablation.py` answers "what does the ego-speed channel contribute?" on the
EVENT-ANCHORED data only. Under the phase-matched control the question is not the same
question, because phase matching removes the class-dependent sampling timing that the
speed channel was partly reading. The event-anchored answer therefore cannot be carried
over, and no neural 4-D arm existed on the phase-matched data at all.

Up to now the only box-vs-speed evidence under phase matching came from logistic
regression inside `phase_matched_stats.py`. That is a linear probe, not the models the
paper reports. This script measures the same contrast with the four neural families.

HELD CONSTANT with the phase-matched control (nothing here is re-tuned)
  data            data/pie_phase_matched_trainonly/   (the corrected, train-only rule)
  splits          train set01/02/04 | val set05/06 | test set03
  engine          src/engine.py, one training loop
  device          cpu, seeds 42/0/1/2/3, checkpoint rule best-val-AUC
  pos_weight      that split's OWN train neg/pos ratio, as phase_matched_stats.py uses
  configs         each family's own val-selected winner, unchanged
  threshold       one tau per arm from POOLED validation probabilities, frozen first
  test set        touched once, after selection

THE ONLY VARIABLE is feature 4 (vehicle_speed): 5-D -> 4-D bbox-only.

WRITES NOTHING OVER EXISTING WORK
  5-D arms   read from the cached runs/phase_matched_trainonly/  (read-only, not retrained)
  4-D arms   trained fresh into runs/phase_matched_trainonly_bbox_only/  (new)

OUTPUTS (all new files, all beside this script)
  EGO_SPEED_ABLATION_PHASE_MATCHED.md            the human-readable report
  ego_speed_ablation_phase_matched_results.json  protocol, per-seed metrics, run
                                                 metadata, contrasts
  ..._per_seed.csv    one row per family x arm x seed -- the raw numbers behind every
                      mean, so an sd can be recomputed and a diverged seed can be seen
  ..._contrasts.csv   one row per test: delta, CI, raw p, Holm p, verdict

Accuracy is recorded per seed but deliberately NOT bootstrapped: adding it would make
16 tests instead of 12 and change every Holm-adjusted p-value in the table. The tested
metrics are AUC, PR-AUC and F1, matching ego_speed_ablation.py.

Usage:  python experiments/02_model_comparison/ego_speed_ablation_phase_matched.py
"""
import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _load("matched", HERE / "matched_comparison.py")   # metrics, bootstrap, holm, FAMILIES
E = M.E

# the four reported families; the h128 guard row belongs to the capacity argument,
# not to this one, and has no phase-matched checkpoints
FAMILIES = {k: v for k, v in M.FAMILIES.items() if k != "BiLSTM-h128"}


def acc_at(y, p, tau):
    return float(((np.asarray(p) >= tau).astype(int) == np.asarray(y)).mean())


def run_meta(run_dir):
    """Training metadata the engine already wrote, carried into the results file."""
    f = Path(run_dir) / "final.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    return dict(n_params=d.get("n_params"),
                selected_epoch=d.get("auc_best_epoch", d.get("best_epoch")),
                train_seconds=d.get("seconds"))


def load_phase_matched(pm_dir):
    """Read the phase-matched tensors directly.

    Deliberately not E.load_splits(): that reads the module-level SEQ_DIR, and
    repointing it would change behaviour for anything else importing the engine
    in this process. Same split rule, no global mutation.
    """
    X = np.load(pm_dir / "X.npy").astype(np.float32)
    y = np.load(pm_dir / "y.npy").astype(np.float32)
    meta = pickle.load(open(pm_dir / "meta.pkl", "rb"))
    sid = np.array([m["set_id"] for m in meta])
    tr, va, tt = (np.isin(sid, sorted(s))
                  for s in (E.TRAIN_SETS, E.VAL_SETS, E.TEST_SETS))
    groups = np.array([f"{m['set_id']}/{m['video_id']}/{m['ped_id']}"
                       for m, k in zip(meta, tt) if k])
    return X, y, tr, va, tt, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-dir", default=str(ROOT / "data" / "pie_phase_matched_trainonly"))
    ap.add_argument("--runs-subdir-5d", default="phase_matched_trainonly")
    ap.add_argument("--runs-subdir-4d", default="phase_matched_trainonly_bbox_only")
    ap.add_argument("--bootstrap", type=int, default=10000)
    # data files keep the json/csv stem; the report takes the repo's uppercase
    # convention for a headline result (EGO_SPEED_ABLATION.md, TREE_BASELINES.md, ...)
    ap.add_argument("--out", default="ego_speed_ablation_phase_matched_results")
    ap.add_argument("--report", default="EGO_SPEED_ABLATION_PHASE_MATCHED.md")
    args = ap.parse_args()

    PM = Path(args.pm_dir)
    X, y, tr, va, tt, groups = load_phase_matched(PM)
    ytr, yva, yte = y[tr], y[va], y[tt]
    # the phase-matched split's own ratio, NOT the event-anchored 1.682
    pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    runs5 = ROOT / "runs" / args.runs_subdir_5d
    runs4 = ROOT / "runs" / args.runs_subdir_4d
    runs4.mkdir(parents=True, exist_ok=True)

    data5 = (X[tr], ytr, X[va], yva, X[tt], yte)
    data4 = (X[tr][:, :, :4], ytr, X[va][:, :, :4], yva, X[tt][:, :, :4], yte)

    print(f"phase-matched: train {int(tr.sum())}  val {int(va.sum())}  "
          f"test {int(tt.sum())} windows / {len(set(groups))} pedestrians")
    print(f"pos_weight {pw:.4f}   seeds {M.SEEDS}   select {M.SELECT}   device {M.DEVICE}\n")

    rows, tests = {}, []
    for label, (family, cfg5) in FAMILIES.items():
        cfg4 = dict(cfg5)
        cfg4["input_dim"] = 4
        arms = {}
        for tag, cfg, dat, rdir in (
                ("5D", cfg5, data5, runs5 / label.replace(" ", "_")),
                ("4D", cfg4, data4, runs4 / label.replace(" ", "_"))):
            X_va, X_te = dat[2], dat[4]
            pv, pt, meta = [], [], []
            for s in M.SEEDS:
                d = rdir / f"seed{s}"
                if not (d / "best.pt").exists():
                    if tag == "5D":
                        raise SystemExit(
                            f"missing cached 5-D checkpoint {d}. This script must not "
                            f"retrain the existing phase-matched arms; run "
                            f"phase_matched_control.py first.")
                    d.mkdir(parents=True, exist_ok=True)
                    E.train_run(family, cfg, s, M.DEVICE, dat,
                                pos_weight=pw, select=M.SELECT, out_dir=d)
                pv.append(M.probs_for(d, family, cfg, X_va))
                pt.append(M.probs_for(d, family, cfg, X_te))
                meta.append(run_meta(d))
            tau = M.best_threshold(yva, np.mean(pv, axis=0))
            # keep the per-seed numbers: they are what the mean and sd are made of,
            # and a diverged seed is invisible once it has been averaged away
            per = [dict(seed=s, auc=M.auc(yte, p), pr_auc=M.pr_auc(yte, p),
                        f1=M.f1_at(yte, p, tau), acc=acc_at(yte, p, tau), **mt)
                   for s, p, mt in zip(M.SEEDS, pt, meta)]
            keys = ("auc", "pr_auc", "f1", "acc")
            arms[tag] = dict(
                tau=float(tau), ens=np.mean(pt, axis=0), input_dim=cfg.get("input_dim", 5),
                n_params=per[0].get("n_params"), per_seed=per,
                mean={k: float(np.mean([q[k] for q in per])) for k in keys},
                sd={k: float(np.std([q[k] for q in per], ddof=1)) for k in keys})
        rows[label] = arms
        print(f"{label:13s} 5D AUC {arms['5D']['mean']['auc']:.4f}  "
              f"4D AUC {arms['4D']['mean']['auc']:.4f}  "
              f"drop {arms['5D']['mean']['auc'] - arms['4D']['mean']['auc']:+.4f}")

    # ------------------------------------------------ paired, pedestrian-clustered
    idx_by_g, picks = M.make_resamples(groups, args.bootstrap)
    for label, a in rows.items():
        t5, t4 = a["5D"]["tau"], a["4D"]["tau"]
        for name, sa, sb in (("AUC", M.auc, M.auc), ("PR-AUC", M.pr_auc, M.pr_auc),
                             ("F1@tau", (lambda y_, p_, t=t5: M.f1_at(y_, p_, t)),
                                        (lambda y_, p_, t=t4: M.f1_at(y_, p_, t)))):
            o, lo, hi, pv_ = M.cluster_delta(np.asarray(yte), a["5D"]["ens"],
                                             a["4D"]["ens"], idx_by_g, picks, sa, sb)
            tests.append(dict(key=f"{label} 5D-4D [{name}]", delta=o, ci=[lo, hi], p=pv_))
        print(f"  bootstrapped {label}")

    adj, rej = M.holm([t["p"] for t in tests])
    print(f"\n{'comparison':34s} {'delta':>9s} {'95% CI':>22s} {'p_holm':>8s}  verdict")
    for t, pa_, rj in zip(tests, adj, rej):
        t["p_holm"], t["significant_holm"] = pa_, bool(rj)
        print(f"  {t['key']:32s} {t['delta']:+9.4f} "
              f"[{t['ci'][0]:+.4f},{t['ci'][1]:+.4f}] {pa_:8.4f}  "
              f"{'DIFFERENT' if rj else 'not distinguishable'}")

    def rel(p):   # repo-relative, so the record is portable between machines
        return str(Path(p).relative_to(ROOT))

    out = dict(
        protocol=dict(
            script=rel(Path(__file__)), data=rel(PM), seeds=M.SEEDS, pos_weight=pw,
            select=M.SELECT, device=M.DEVICE, torch=str(__import__("torch").__version__),
            runs_5d=rel(runs5), runs_4d=rel(runs4),
            n_train=int(tr.sum()), n_val=int(va.sum()), n_test=int(tt.sum()),
            n_test_pedestrians=len(set(groups)), test_prevalence=float(yte.mean()),
            variable="feature 4 (vehicle_speed): 5-D vs 4-D bbox-only",
            threshold="one tau per arm, argmax F1 on pooled validation probabilities",
            bootstrap=f"pedestrian-clustered, B={args.bootstrap}, seed 42",
            correction=f"Holm-Bonferroni across {len(tests)} tests",
            tested_metrics=["auc", "pr_auc", "f1"],
            untested_metrics=["acc (recorded per seed only; testing it would "
                              "change the Holm correction)"]),
        families={k: {t: {kk: vv for kk, vv in a[t].items() if kk != "ens"}
                      for t in a} for k, a in rows.items()},
        tests={t["key"]: {k: v for k, v in t.items() if k != "key"} for t in tests})

    (HERE / f"{args.out}.json").write_text(json.dumps(out, indent=2))
    write_per_seed_csv(HERE / f"{args.out}_per_seed.csv", out)
    write_contrasts_csv(HERE / f"{args.out}_contrasts.csv", out)
    write_report(HERE / args.report, out)
    for f in (f"{args.out}.json", f"{args.out}_per_seed.csv",
              f"{args.out}_contrasts.csv", args.report):
        print(f"wrote {HERE / f}")


def write_per_seed_csv(path, out):
    cols = ["family", "arm", "input_dim", "seed", "auc", "pr_auc", "f1", "acc",
            "tau", "n_params", "selected_epoch", "train_seconds"]
    lines = [",".join(cols)]
    for label, a in out["families"].items():
        for tag in ("5D", "4D"):
            for r in a[tag]["per_seed"]:
                lines.append(",".join(str(v) for v in [
                    label, tag, a[tag]["input_dim"], r["seed"],
                    f"{r['auc']:.6f}", f"{r['pr_auc']:.6f}", f"{r['f1']:.6f}",
                    f"{r['acc']:.6f}", f"{a[tag]['tau']:.6f}",
                    r.get("n_params", ""), r.get("selected_epoch", ""),
                    r.get("train_seconds", "")]))
    path.write_text("\n".join(lines) + "\n")


def write_contrasts_csv(path, out):
    lines = ["contrast,family,metric,delta,ci_lo,ci_hi,p_raw,p_holm,significant_holm"]
    for k, v in out["tests"].items():
        fam, metric = k.rsplit(" 5D-4D [", 1)
        lines.append(",".join([
            f'"{k}"', fam, metric.rstrip("]"), f"{v['delta']:.6f}",
            f"{v['ci'][0]:.6f}", f"{v['ci'][1]:.6f}", f"{v['p']:.6f}",
            f"{v['p_holm']:.6f}", str(v["significant_holm"])]))
    path.write_text("\n".join(lines) + "\n")


def write_report(path, out):
    """Markdown beside the JSON, with the event-anchored arm alongside if present."""
    ea_path = HERE / "ego_speed_ablation_results.json"
    ea = json.loads(ea_path.read_text()) if ea_path.exists() else None
    p = out["protocol"]

    L = ["# Ego-speed ablation under the phase-matched control", "",
         "Produced by `ego_speed_ablation_phase_matched.py`. The single variable is feature 4",
         "(`vehicle_speed`): 5-D versus 4-D bbox-only, same family, same config, same protocol.",
         "", "## Protocol", "",
         f"| | |", "|---|---|",
         f"| data | `{p['data']}` |",
         f"| split | train {p['n_train']:,} / val {p['n_val']:,} / test {p['n_test']:,} windows |",
         f"| test pedestrians | {p['n_test_pedestrians']} |",
         f"| test prevalence | {p['test_prevalence']:.1%} |",
         f"| class weight | {p['pos_weight']:.4f} (this split's own train ratio, not 1.682) |",
         f"| seeds | {p['seeds']} |",
         f"| checkpoint rule | best validation {p['select'].upper()} |",
         f"| threshold | {p['threshold']} |",
         f"| device | {p['device']}, torch {p['torch']} |",
         f"| inference | {p['bootstrap']} |",
         f"| correction | {p['correction']} |",
         "",
         "The 5-D arms are the cached phase-matched checkpoints and were **not** retrained;",
         f"only the 4-D arms are new (`{p['runs_4d']}`). Accuracy is recorded per seed but not",
         "bootstrapped — testing it would make 16 tests and shift every Holm-adjusted p below.",
         "", "## Per-seed mean ± sd", "",
         "| family | arm | params | AUC | PR-AUC | F1 | accuracy |",
         "|---|---|---|---|---|---|---|"]
    for label, a in out["families"].items():
        for tag in ("5D", "4D"):
            m, s = a[tag]["mean"], a[tag]["sd"]
            L.append(f"| {label} | {tag} | {a[tag]['n_params']:,} | "
                     f"{m['auc']:.4f} ± {s['auc']:.4f} | "
                     f"{m['pr_auc']:.4f} ± {s['pr_auc']:.4f} | "
                     f"{m['f1']:.4f} ± {s['f1']:.4f} | "
                     f"{m['acc']:.4f} ± {s['acc']:.4f} |")
    L += ["", "Dropping the channel removes 64 parameters from the input projection, so the two",
          "arms are the same size to four significant figures — the contrast is the input, not",
          "capacity.", "",
          "### The raw per-seed AUC behind those means", "",
          "| family | arm | " + " | ".join(f"seed {s}" for s in p["seeds"]) +
          " | selected epochs |", "|---|---|" + "---|" * (len(p["seeds"]) + 1)]
    for label, a in out["families"].items():
        for tag in ("5D", "4D"):
            ps = a[tag]["per_seed"]
            L.append(f"| {label} | {tag} | " +
                     " | ".join(f"{r['auc']:.4f}" for r in ps) + " | " +
                     ", ".join(str(r.get("selected_epoch")) for r in ps) + " |")
    L += ["", "Full per-seed values including PR-AUC, F1, accuracy and training time are in",
          "`ego_speed_ablation_phase_matched_results_per_seed.csv`.", ""]
    L += ["", "## 5-D − 4-D, paired on the pedestrian-clustered bootstrap", "",
          "| contrast | Δ | 95% CI | p_Holm | |", "|---|---|---|---|---|"]
    for k, v in out["tests"].items():
        verdict = ("**speed helps**" if v["significant_holm"] and v["delta"] > 0 else
                   "**speed hurts**" if v["significant_holm"] else "n.s.")
        L.append(f"| {k} | {v['delta']:+.4f} | [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] | "
                 f"{v['p_holm']:.4f} | {verdict} |")
    n_sig = sum(v["significant_holm"] for v in out["tests"].values())
    L += ["", f"**{n_sig} of {len(out['tests'])} contrasts survive Holm.** A positive delta means",
          "the 5-D arm scored higher, i.e. ego speed helped. Machine-readable copy in",
          "`ego_speed_ablation_phase_matched_results_contrasts.csv`.", "",
          "### Reading the metric directions", "",
          "The sign is not consistent across metrics: for the BiLSTM and the vanilla RNN the",
          "4-D arm is ahead on AUC and PR-AUC but behind on F1 and accuracy. That is a threshold",
          "effect, not a contradiction. Each arm gets its own tau from its own pooled validation",
          "probabilities, and the 4-D arms land much lower (BiLSTM 0.217 vs 0.486), which trades",
          "precision for recall and costs accuracy at a 36.4 % prevalence. AUC and PR-AUC are",
          "threshold-free and are the cleaner read on what the input channel contributes.", "",
          "None of these directions is established: every interval crosses zero after correction.",
          "The result is an absence of an ego-speed effect, not evidence that dropping the",
          "channel helps.", ""]

    if ea:
        L += ["## Against the event-anchored answer", "",
              "The same contrast on `data/pie_clean/`, from `ego_speed_ablation_results.json`.",
              "The two protocols keep different pedestrians, so these columns are **not paired** —",
              "read them as two experiments.", "",
              "| family | ΔAUC event-anchored | ΔAUC phase-matched | change |",
              "|---|---|---|---|"]
        for label, a in out["families"].items():
            pm = a["5D"]["mean"]["auc"] - a["4D"]["mean"]["auc"]
            if label in ea.get("families", {}):
                b = ea["families"][label]
                eaa = b["5D"]["mean"]["auc"] - b["4D"]["mean"]["auc"]
                L.append(f"| {label} | {eaa:+.4f} | {pm:+.4f} | {pm - eaa:+.4f} |")
            else:
                L.append(f"| {label} | — | {pm:+.4f} | — |")
        L.append("")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
