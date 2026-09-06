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
  results    ego_speed_ablation_phase_matched_results.json + .md  (new files)

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
    ap.add_argument("--out", default="ego_speed_ablation_phase_matched_results")
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
            pv, pt = [], []
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
            tau = M.best_threshold(yva, np.mean(pv, axis=0))
            per = [dict(auc=M.auc(yte, p), pr_auc=M.pr_auc(yte, p),
                        f1=M.f1_at(yte, p, tau)) for p in pt]
            arms[tag] = dict(
                tau=float(tau), ens=np.mean(pt, axis=0),
                mean={k: float(np.mean([q[k] for q in per])) for k in per[0]},
                sd={k: float(np.std([q[k] for q in per], ddof=1)) for k in per[0]})
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

    out = dict(
        protocol=dict(data=str(PM), seeds=M.SEEDS, pos_weight=pw, select=M.SELECT,
                      device=M.DEVICE, runs_5d=str(runs5), runs_4d=str(runs4),
                      n_test=int(tt.sum()), n_test_pedestrians=len(set(groups)),
                      bootstrap=f"pedestrian-clustered, B={args.bootstrap}",
                      correction=f"Holm-Bonferroni across {len(tests)} tests"),
        families={k: {t: {kk: vv for kk, vv in a[t].items() if kk != "ens"}
                      for t in a} for k, a in rows.items()},
        tests={t["key"]: {k: v for k, v in t.items() if k != "key"} for t in tests})
    (HERE / f"{args.out}.json").write_text(json.dumps(out, indent=2))
    write_report(HERE / f"{args.out}.md", out)
    print(f"\nwrote {HERE / (args.out + '.json')}")
    print(f"wrote {HERE / (args.out + '.md')}")


def write_report(path, out):
    """Markdown beside the JSON, with the event-anchored arm alongside if present."""
    ea_path = HERE / "ego_speed_ablation_results.json"
    ea = json.loads(ea_path.read_text()) if ea_path.exists() else None
    p = out["protocol"]

    L = ["# Ego-speed ablation under the phase-matched control", "",
         "Produced by `ego_speed_ablation_phase_matched.py`. The single variable is feature 4",
         "(`vehicle_speed`): 5-D versus 4-D bbox-only, same family, same config, same protocol.", "",
         f"- data `{Path(p['data']).name}`, {p['n_test']:,} test windows from "
         f"{p['n_test_pedestrians']} pedestrians",
         f"- class weight {p['pos_weight']:.4f} (this split's own train ratio), "
         f"seeds {p['seeds']}, select {p['select']}, {p['device']}",
         f"- {p['bootstrap']}, {p['correction']}",
         "- the 5-D arms are the cached phase-matched checkpoints, not retrained", "",
         "## Per-seed mean ± sd", "",
         "| family | arm | AUC | PR-AUC | F1 |", "|---|---|---|---|---|"]
    for label, a in out["families"].items():
        for tag in ("5D", "4D"):
            m, s = a[tag]["mean"], a[tag]["sd"]
            L.append(f"| {label} | {tag} | {m['auc']:.4f} ± {s['auc']:.4f} | "
                     f"{m['pr_auc']:.4f} ± {s['pr_auc']:.4f} | "
                     f"{m['f1']:.4f} ± {s['f1']:.4f} |")
    L += ["", "## 5-D − 4-D, paired on the pedestrian-clustered bootstrap", "",
          "| contrast | Δ | 95% CI | p_Holm | |", "|---|---|---|---|---|"]
    for k, v in out["tests"].items():
        verdict = ("**speed helps**" if v["significant_holm"] and v["delta"] > 0 else
                   "**speed hurts**" if v["significant_holm"] else "n.s.")
        L.append(f"| {k} | {v['delta']:+.4f} | [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] | "
                 f"{v['p_holm']:.4f} | {verdict} |")
    n_sig = sum(v["significant_holm"] for v in out["tests"].values())
    L += ["", f"**{n_sig} of {len(out['tests'])} contrasts survive Holm.**", ""]

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
