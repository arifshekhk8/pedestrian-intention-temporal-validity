"""
ego_speed_ablation.py — what does the ego-speed channel actually contribute?

Single-variable test: identical family, identical config, identical protocol; the
only change is dropping feature 4 (vehicle_speed), i.e. 5-D -> 4-D bbox-only.

WHY THIS SCRIPT EXISTS
----------------------
The project's published +0.179 AUC ego-speed gap compared a locally-trained CPU
5-D baseline (pos_weight 1.682) against Kaggle-GPU 4-D runs whose own shipped
summary records pos_weight 1.44 while the notebook that supposedly produced them
sets 1.682. Device, class weight and provenance all differ, so the comparison was
not single-variable. This rebuilds it under the matched protocol.

Everything is inherited from matched_comparison.py so the 5-D arms are literally
the same cached runs used in the model comparison -- only the 4-D arms are new.

Usage:  python experiments/02_model_comparison/ego_speed_ablation.py
"""
import importlib.util, json, pickle, sys
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

# the four families exactly as in the matched comparison (drop the h128 guard row)
FAMILIES = {k: v for k, v in M.FAMILIES.items() if k != "BiLSTM-h128"}


def main():
    B = 10000
    runs5 = ROOT / "runs" / "matched"        # reuse the cached 5-D runs
    runs4 = ROOT / "runs" / "bbox_only"
    runs4.mkdir(parents=True, exist_ok=True)

    data5 = E.load_splits()
    Xtr, ytr, Xva, yva, Xte, yte = data5
    # 4-D = same windows, ego-speed column removed
    data4 = (Xtr[:, :, :4], ytr, Xva[:, :, :4], yva, Xte[:, :, :4], yte)

    with open(E.SEQ_DIR / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    tm = [m for m in meta if m["set_id"] in E.TEST_SETS]
    groups = np.array([f"{m['set_id']}/{m['video_id']}/{m['ped_id']}" for m in tm])

    rows, tests = {}, []
    for label, (family, cfg5) in FAMILIES.items():
        cfg4 = dict(cfg5); cfg4["input_dim"] = 4
        arms = {}
        for tag, cfg, dat, X_va, X_te, rdir in (
                ("5D", cfg5, data5, Xva, Xte, runs5 / label.replace(" ", "_")),
                ("4D", cfg4, data4, data4[2], data4[4], runs4 / label.replace(" ", "_"))):
            pv, pt = [], []
            for s in M.SEEDS:
                d = rdir / f"seed{s}"
                if not (d / "best.pt").exists():
                    d.mkdir(parents=True, exist_ok=True)
                    E.train_run(family, cfg, s, M.DEVICE, dat,
                                pos_weight=M.POS_WEIGHT, select=M.SELECT, out_dir=d)
                pv.append(M.probs_for(d, family, cfg, X_va))
                pt.append(M.probs_for(d, family, cfg, X_te))
            tau = M.best_threshold(yva, np.mean(pv, axis=0))
            per = [dict(auc=M.auc(yte, p), pr_auc=M.pr_auc(yte, p),
                        f1=M.f1_at(yte, p, tau)) for p in pt]
            arms[tag] = dict(tau=float(tau), ens=np.mean(pt, axis=0),
                             mean={k: float(np.mean([q[k] for q in per])) for k in per[0]},
                             sd={k: float(np.std([q[k] for q in per], ddof=1)) for k in per[0]})
        rows[label] = arms
        print(f"{label:13s} 5D AUC {arms['5D']['mean']['auc']:.4f}  "
              f"4D AUC {arms['4D']['mean']['auc']:.4f}  "
              f"drop {arms['5D']['mean']['auc'] - arms['4D']['mean']['auc']:+.4f}")

    idx_by_g, picks = M.make_resamples(groups, B)
    for label, a in rows.items():
        t5, t4 = a["5D"]["tau"], a["4D"]["tau"]
        for name, sa, sb in (("AUC", M.auc, M.auc), ("PR-AUC", M.pr_auc, M.pr_auc),
                             ("F1@tau", (lambda y_, p_, t=t5: M.f1_at(y_, p_, t)),
                                        (lambda y_, p_, t=t4: M.f1_at(y_, p_, t)))):
            o, lo, hi, pv_ = M.cluster_delta(np.asarray(yte), a["5D"]["ens"], a["4D"]["ens"],
                                             idx_by_g, picks, sa, sb)
            tests.append(dict(key=f"{label} 5D-4D [{name}]", delta=o, ci=[lo, hi], p=pv_))
        print(f"  bootstrapped {label}")

    adj, rej = M.holm([t["p"] for t in tests])
    print(f"\n{'comparison':34s} {'delta':>9s} {'95% CI':>22s} {'p_holm':>8s}  verdict")
    for t, pa_, rj in zip(tests, adj, rej):
        t["p_holm"], t["significant_holm"] = pa_, bool(rj)
        print(f"  {t['key']:32s} {t['delta']:+9.4f} [{t['ci'][0]:+.4f},{t['ci'][1]:+.4f}] "
              f"{pa_:8.4f}  {'DIFFERENT' if rj else 'not distinguishable'}")

    out = dict(protocol=dict(seeds=M.SEEDS, pos_weight=M.POS_WEIGHT, select=M.SELECT,
                             device=M.DEVICE, bootstrap=f"pedestrian-clustered, B={B}",
                             correction=f"Holm-Bonferroni across {len(tests)} tests"),
               families={k: {t: {kk: vv for kk, vv in a[t].items() if kk != "ens"}
                             for t in a} for k, a in rows.items()},
               tests={t["key"]: {k: v for k, v in t.items() if k != "key"} for t in tests})
    (HERE / "ego_speed_ablation_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE/'ego_speed_ablation_results.json'}")


if __name__ == "__main__":
    main()
