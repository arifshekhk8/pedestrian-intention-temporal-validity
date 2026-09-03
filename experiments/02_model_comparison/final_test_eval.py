"""05_final_test_eval.py — Phase F5: THE single test-touching script (PLAN.md §8).

Evaluates every arm A0-B4 (PLAN.md §4) on test set03, exactly once, after all selection
was fixed on val by 01-04:
- model groups (a group = one set of 5 checkpoints) are loaded once each; per seed the
  VAL and TEST probabilities are computed on CPU (exactness) and cached to probs_cache/
  as .npy for 06;
- per-arm operating point: 0.5 for A0/B0 (the frozen restatement), otherwise tau*
  selected on that seed's own VAL probabilities (ensemble rows use tau* from the
  averaged VAL probabilities) — the pre-registered rule (00_common.best_threshold);
- PARITY GATE first: recomputed frozen-model test metrics must match the stored
  final.json (LSTM exact <1e-6; transformer <1e-4 Kaggle-GPU->CPU drift, per
  transformer/PLAN.md §10). Aborts on failure — nothing downstream is trustworthy then.

Outputs: 05_final_arms.csv, 05_final_arms.json, probs_cache/*.npy
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("f1_common", HERE.parent.parent / "src" / "metrics.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

RUNS = HERE / "runs_f1"
PCACHE = HERE / "probs_cache"


def pw_tag(pw):
    return f"pw{pw:g}"


def group_specs():
    sel = json.loads((HERE / "04_selection.json").read_text())
    ls, ts = sel["lstm"], sel["transformer"]
    a2_cfg = C.BASELINE_LSTM_CFG
    a3_cfg = ls["headline_cfg"]
    groups = {
        "lstm_frozen": dict(dirs=lambda s: C.LSTM_MULTISEED / f"seed{s}",
                            build=lambda: C.build_lstm(C.BASELINE_LSTM_CFG)),
        "lstm_a2": dict(dirs=lambda s: RUNS / f"lstm_{ls['baseline_cfg_id']}" / pw_tag(1.682) / f"seed{s}",
                        build=lambda: C.build_lstm(a2_cfg)),
        "lstm_a3": dict(dirs=lambda s: RUNS / f"lstm_{ls['headline_cfg_id']}" / pw_tag(ls["pw"]) / f"seed{s}",
                        build=lambda: C.build_lstm(a3_cfg)),
        "tf_frozen": dict(dirs=lambda s: C.TF_RUNS_FINAL / "transformer_searched" / f"seed{s}",
                          build=lambda: C.build_transformer(C.SEARCHED_TF_CFG)),
        "tf_b2": dict(dirs=lambda s: RUNS / "transformer_searched" / pw_tag(1.682) / f"seed{s}",
                      build=lambda: C.build_transformer(C.SEARCHED_TF_CFG)),
        "tf_b3": dict(dirs=lambda s: RUNS / "transformer_searched" / pw_tag(ts["pw"]) / f"seed{s}",
                      build=lambda: C.build_transformer(C.SEARCHED_TF_CFG)),
        "tf_b4": dict(dirs=lambda s: RUNS / "transformer_default" / pw_tag(1.682) / f"seed{s}",
                      build=lambda: C.build_transformer(C.DEFAULT_TF_CFG)),
    }
    arms = [
        dict(arm="A0", group="lstm_frozen", rule="0.5",
             desc=f"LSTM {ls['baseline_cfg_id']} frozen @0.5 (restated)"),
        dict(arm="A1", group="lstm_frozen", rule="tau",
             desc=f"LSTM {ls['baseline_cfg_id']} frozen @tau*"),
        dict(arm="A2", group="lstm_a2", rule="tau",
             desc=f"LSTM {ls['baseline_cfg_id']} F1-protocol pw1.682 @tau*"),
        dict(arm="A3", group="lstm_a3", rule="tau",
             desc=f"LSTM {ls['headline_cfg_id']} F1-protocol pw{ls['pw']:g} @tau* (headline)"),
        dict(arm="B0", group="tf_frozen", rule="0.5",
             desc="Transformer searched frozen @0.5 (restated)"),
        dict(arm="B1", group="tf_frozen", rule="tau", desc="Transformer searched frozen @tau*"),
        dict(arm="B2", group="tf_b2", rule="tau",
             desc="Transformer searched F1-protocol pw1.682 @tau*"),
        dict(arm="B3", group="tf_b3", rule="tau",
             desc=f"Transformer searched F1-protocol pw{ts['pw']:g} @tau* (headline)"),
        dict(arm="B4", group="tf_b4", rule="tau",
             desc="Transformer default F1-protocol pw1.682 @tau* (architecture control)"),
    ]
    return groups, arms, sel


def parity_gate(probs, yte):
    print("=" * 70)
    print("PARITY GATE — recomputed frozen test metrics vs stored final.json")
    print("=" * 70)
    from sklearn.metrics import roc_auc_score
    checks = [("lstm_frozen", C.LSTM_MULTISEED, 1e-6),
              ("tf_frozen", C.TF_RUNS_FINAL / "transformer_searched", 1e-4)]
    for gname, root, bound in checks:
        max_d = 0.0
        for s in C.SEEDS:
            stored = json.loads((root / f"seed{s}" / "final.json").read_text())["test"]["auc"]
            rec = roc_auc_score(yte, probs[gname][s]["test"])
            d = abs(rec - stored)
            max_d = max(max_d, d)
            print(f"  {gname:12s} seed{s:<3d} stored={stored:.6f} recomputed={rec:.6f} d={d:.2e}")
        if max_d >= bound:
            raise RuntimeError(f"PARITY GATE FAILED for {gname}: max delta {max_d:.2e} >= {bound:g}. "
                               f"Nothing downstream is trustworthy — aborting.")
        print(f"  {gname}: PASS (max delta {max_d:.2e} < {bound:g})\n")


def main():
    Xtr, ytr, Xva, yva, Xte, yte = C.load_splits()
    groups, arms, sel = group_specs()
    PCACHE.mkdir(exist_ok=True)

    # ---- compute + cache val/test probs per group (CPU, one checkpoint load each) ----
    probs = {}
    for gname, spec in groups.items():
        probs[gname] = {}
        for s in C.SEEDS:
            fv, ft = PCACHE / f"{gname}_seed{s}_val.npy", PCACHE / f"{gname}_seed{s}_test.npy"
            if fv.exists() and ft.exists():
                probs[gname][s] = dict(val=np.load(fv), test=np.load(ft))
                continue
            pf = C.prob_fn_from_run_dir(spec["dirs"](s), spec["build"]())
            pv, pt = pf(Xva), pf(Xte)
            np.save(fv, pv)
            np.save(ft, pt)
            probs[gname][s] = dict(val=pv, test=pt)
        print(f"probs ready: {gname}")
    print()

    parity_gate(probs, yte)

    np.save(PCACHE / "y_val.npy", yva)
    np.save(PCACHE / "y_test.npy", yte)

    # ---- evaluate arms ----
    rows, arm_json = [], {}
    for a in arms:
        g = a["group"]
        per_seed = []
        for s in C.SEEDS:
            pv, pt = probs[g][s]["val"], probs[g][s]["test"]
            tau = 0.5 if a["rule"] == "0.5" else C.best_threshold(yva, pv)
            m_tau = C.metrics_at(yte, pt, tau)
            m_05 = C.metrics_at(yte, pt, 0.5)
            mv_tau = C.metrics_at(yva, pv, tau)
            per_seed.append(dict(seed=s, tau=tau, test=m_tau, test_at_05=m_05,
                                 val_at_tau=mv_tau))
            rows.append(dict(arm=a["arm"], seed=s, tau=round(tau, 6),
                             **{f"test_{k}": round(v, 6) for k, v in m_tau.items()},
                             **{f"test05_{k}": round(v, 6) for k, v in m_05.items()}))
        # ensemble
        pv_e = np.mean([probs[g][s]["val"] for s in C.SEEDS], axis=0)
        pt_e = np.mean([probs[g][s]["test"] for s in C.SEEDS], axis=0)
        np.save(PCACHE / f"{g}_ens_val.npy", pv_e)
        np.save(PCACHE / f"{g}_ens_test.npy", pt_e)
        tau_e = 0.5 if a["rule"] == "0.5" else C.best_threshold(yva, pv_e)
        me = C.metrics_at(yte, pt_e, tau_e)
        me05 = C.metrics_at(yte, pt_e, 0.5)
        pred_e = (pt_e >= tau_e).astype(int)
        cm = [[int(((yte == 0) & (pred_e == 0)).sum()), int(((yte == 0) & (pred_e == 1)).sum())],
              [int(((yte == 1) & (pred_e == 0)).sum()), int(((yte == 1) & (pred_e == 1)).sum())]]
        rows.append(dict(arm=a["arm"], seed="ens", tau=round(tau_e, 6),
                         **{f"test_{k}": round(v, 6) for k, v in me.items()},
                         **{f"test05_{k}": round(v, 6) for k, v in me05.items()}))
        f1s = np.array([p["test"]["f1"] for p in per_seed])
        accs = np.array([p["test"]["acc"] for p in per_seed])
        arm_json[a["arm"]] = dict(desc=a["desc"], group=g, rule=a["rule"],
                                  per_seed=per_seed, ens=dict(tau=tau_e, test=me,
                                                              test_at_05=me05,
                                                              test_confusion_matrix=cm),
                                  test_f1_mean=float(f1s.mean()),
                                  test_f1_std=float(f1s.std(ddof=1)),
                                  test_acc_mean=float(accs.mean()))
        print(f"{a['arm']}: test F1 {f1s.mean():.4f} ± {f1s.std(ddof=1):.4f} "
              f"(ens {me['f1']:.4f} @tau {tau_e:.3f}) acc {accs.mean():.4f} — {a['desc']}")

    with open(HERE / "05_final_arms.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (HERE / "05_final_arms.json").write_text(json.dumps(dict(
        selection=sel, arms=arm_json), indent=2))
    print("\nwrote 05_final_arms.csv, 05_final_arms.json, probs_cache/")


if __name__ == "__main__":
    main()
