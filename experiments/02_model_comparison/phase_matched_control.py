"""
phase_matched_control.py — is the result driven by a class-dependent sampling bias?

THE PROBLEM
-----------
In PIE's standard windowing (and in this project's clean protocol, which inherits it
from `extract_tracks_tte`), `crossing_point` for a NON-crosser is defined as the last
annotated frame minus 2. Windows are then placed 30-60 frames before that. For a
crosser, `crossing_point` is a real behavioural event, typically mid-track.

The consequence, measured on data/pie_clean/:

    frames from anchor to end of track      min    median    max
      non-crossers                           32       46      62
      crossers                               88      287    6606

Zero overlap. That nuisance variable alone separates the classes with AUC = 1.0000.
Every negative is observed in the last ~1-2 s of its track; every positive at least
88 frames before its track ends. A model cannot see "frames to track end", but it can
see box growth and ego-speed, which encode "this pedestrian is about to be passed".

This is NOT temporal leakage -- the windows genuinely precede the crossing (verified
0/4906 contain a crossing frame). It is a separate, class-dependent sampling bias.

THE CONTROL
-----------
Positives cannot move: their anchor is pinned by the crossing event. So negatives are
re-sampled EARLIER, with their frames-to-track-end drawn from the positive empirical
distribution (clipped to what each track allows, floor = the source minimum).
Everything else -- features, splits, engine, seeds, class weight, selection rule --
is unchanged.

Then the four families and the linear baselines are retrained on the matched data and
compared against the same models on the original data.

WHERE THE TARGET DISTRIBUTION COMES FROM  (--phase-source)
----------------------------------------------------------
  train   DEFAULT AND CORRECT. The target distribution and the floor are estimated
          from TRAINING positives only (set01/02/04), frozen, and then applied to
          construct negatives in all three splits. No validation or test label or
          timing informs the matching rule.

  all     The original v1 behaviour, kept only so the superseded artefact can be
          rebuilt. It pools positives from every split, so 50.7% of the sampled
          distribution came from validation and test. For a paper about temporal
          validity that is not defensible: test-set positive timing shaped how test
          negatives were drawn. Do not use for new results.

The frozen rule is written to <out>/phase_rule.json.

Requires pie_annotations.pkl (Tier 2 of docs/REPRODUCE.md) because track start/end
frames are needed, and meta.pkl does not carry them.

Usage:
  python experiments/02_model_comparison/phase_matched_control.py \
      --annotations /path/to/pie_annotations.pkl
"""
import argparse, importlib.util, json, pickle
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FEATURE_COLS = ["x1", "y1", "x2", "y2", "vehicle_speed"]
OBS_LEN = 16
SEED = 42
# The floor was a hard-coded 88 in v1 -- the minimum over POOLED positives. It is now
# derived from whichever source --phase-source selects, so the rule is estimated from
# one split rather than read off the whole dataset. (Both happen to give 88: the global
# minimum is owned by set01, which is a training set.)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E = _load("engine", ROOT / "src" / "engine.py")
M = _load("matched", HERE / "matched_comparison.py")


def build(annotations, out_dir, phase_source="train"):
    df = pd.read_pickle(annotations)
    meta = pickle.load(open(E.SEQ_DIR / "meta.pkl", "rb"))
    y = np.load(E.SEQ_DIR / "y.npy")
    X = np.load(E.SEQ_DIR / "X.npy")

    feats = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    frames = df["frame"].to_numpy()
    bounds, index = {}, {}
    for key, grp in df.groupby(["set_id", "video_id", "ped_id"], sort=False):
        idx = grp.index.to_numpy()
        fr = frames[idx]
        o = np.argsort(fr)
        bounds[key] = (int(fr[o][0]), int(fr[o][-1]))
        index[key] = (idx[o], fr[o])

    # The target phase distribution is estimated from ONE split (training, by default)
    # and then frozen. Drawing it from every split would let validation and test
    # positive timing decide where validation and test negatives are placed.
    pos_to_end, pos_all, per_ped = [], [], {}
    for m, lab in zip(meta, y):
        k = (m["set_id"], m["video_id"], m["ped_id"])
        if k not in bounds:
            continue
        te = bounds[k][1] - m["anchor_frame"]
        per_ped.setdefault((k, int(lab)), []).append((m, te))
        if lab == 1:
            pos_all.append(te)
            if phase_source == "all" or m["set_id"] in E.TRAIN_SETS:
                pos_to_end.append(te)
    pos_to_end = np.array(pos_to_end)
    min_to_end = int(pos_to_end.min())
    print(f"phase rule from {phase_source!r}: n={len(pos_to_end)} of {len(pos_all)} "
          f"positives, floor={min_to_end}, median={int(np.median(pos_to_end))}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase_rule.json").write_text(json.dumps(dict(
        phase_source=phase_source,
        splits_used=sorted(E.TRAIN_SETS) if phase_source == "train" else "all",
        n_source_positives=int(len(pos_to_end)),
        n_all_positives=int(len(pos_all)),
        min_to_end=min_to_end,
        quantiles={q: float(np.percentile(pos_to_end, q)) for q in (0, 25, 50, 75, 100)},
        seed=SEED, obs_len=OBS_LEN,
        note="Frozen before any negative is drawn; applied unchanged to train, val "
             "and test. Negative placement uses only this rule and the pedestrian's "
             "own track bounds."), indent=2))

    rng = np.random.default_rng(SEED)
    Xn, yn, mn, dropped = [], [], [], 0
    for (k, lab), items in per_ped.items():
        if lab == 1:                                   # positives unchanged
            for m, _ in items:
                j = next(i for i, mm in enumerate(meta) if mm is m)
                Xn.append(X[j]); yn.append(1); mn.append(dict(m, to_end=bounds[k][1] - m["anchor_frame"]))
            continue
        f0, fN = bounds[k]
        idx_sorted, fr_sorted = index[k]
        hi = fN - f0 - (OBS_LEN - 1)                   # largest achievable to_end
        if hi < min_to_end:
            dropped += 1
            continue
        for _ in items:                                # same window count as before
            target = int(rng.choice(pos_to_end))
            te = int(np.clip(target, min_to_end, hi))
            anchor = fN - te
            pos = np.searchsorted(fr_sorted, anchor)
            if pos < OBS_LEN - 1 or pos >= len(fr_sorted):
                continue
            sel = idx_sorted[pos - OBS_LEN + 1: pos + 1]
            if len(sel) != OBS_LEN or (fr_sorted[pos] - fr_sorted[pos - OBS_LEN + 1]) != OBS_LEN - 1:
                continue                               # require 16 CONSECUTIVE frames
            Xn.append(feats[sel]); yn.append(0)
            mn.append(dict(set_id=k[0], video_id=k[1], ped_id=k[2],
                           anchor_frame=int(fr_sorted[pos]), crossing_point=None, tte=None,
                           to_end=int(te)))
    Xn = np.asarray(Xn, dtype=np.float32); yn = np.asarray(yn, dtype=np.int8)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", Xn); np.save(out_dir / "y.npy", yn)
    pickle.dump(mn, open(out_dir / "meta.pkl", "wb"))
    return Xn, yn, mn, dropped


def auc_of(y, s):
    return M.auc(np.asarray(y), np.asarray(s, dtype=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations",
                    default="/Users/arif/Developer/pedestrian-thesis/pie_annotations.pkl")
    ap.add_argument("--out", default=str(ROOT / "data" / "pie_phase_matched_trainonly"))
    ap.add_argument("--phase-source", choices=["train", "all"], default="train",
                    help="which positives estimate the target phase distribution; "
                         "'train' is correct, 'all' rebuilds the superseded v1 artefact")
    ap.add_argument("--runs-subdir", default="phase_matched_trainonly",
                    help="subdirectory of runs/ for this control's checkpoints")
    ap.add_argument("--results", default=None,
                    help="output json (default derives from --runs-subdir)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    res_path = HERE / (args.results or f"{args.runs_subdir}_results.json")
    if (out_dir / "X.npy").exists():
        Xn = np.load(out_dir / "X.npy"); yn = np.load(out_dir / "y.npy")
        mn = pickle.load(open(out_dir / "meta.pkl", "rb")); dropped = None
    else:
        Xn, yn, mn, dropped = build(Path(args.annotations), out_dir, args.phase_source)

    te = np.array([m["to_end"] for m in mn])
    print(f"phase-matched set: X{Xn.shape}  pos={int(yn.sum())} neg={int((yn==0).sum())}"
          + (f"  negatives dropped as too short: {dropped}" if dropped is not None else ""))
    print(f"  to_end  negatives: med {np.median(te[yn==0]):.0f} [{te[yn==0].min()}, {te[yn==0].max()}]")
    print(f"  to_end  positives: med {np.median(te[yn==1]):.0f} [{te[yn==1].min()}, {te[yn==1].max()}]")
    print(f"  AUC of 'to_end' alone:  ORIGINAL 1.0000  ->  MATCHED {auc_of(yn, te):.4f}")

    sid = np.array([m["set_id"] for m in mn])
    tr = np.isin(sid, sorted(E.TRAIN_SETS)); va = np.isin(sid, sorted(E.VAL_SETS))
    tt = np.isin(sid, sorted(E.TEST_SETS))

    # phase separability per split: the test-split value is the one that matters, and
    # under the corrected rule it is no longer propped up by test timing information.
    sep = {}
    for nm, msk in (("train", tr), ("val", va), ("test", tt)):
        sep[nm] = float(auc_of(yn[msk], te[msk]))
        print(f"  to_end AUC [{nm:5s}] = {sep[nm]:.4f}   "
              f"(n={int(msk.sum())}, pos={int(yn[msk].sum())})")
    data = (Xn[tr], yn[tr].astype(np.float32), Xn[va], yn[va].astype(np.float32),
            Xn[tt], yn[tt].astype(np.float32))
    pw = float((yn[tr] == 0).sum() / max((yn[tr] == 1).sum(), 1))
    print(f"  splits train/val/test = {tr.sum()}/{va.sum()}/{tt.sum()}  pos_weight={pw:.4f}")
    yte = data[5]
    groups = np.array([f"{m['set_id']}/{m['video_id']}/{m['ped_id']}"
                       for m, keep in zip(mn, tt) if keep])

    results = {}
    print(f"\n{'model':34s} {'AUC':>8s} {'PR-AUC':>8s} {'F1':>8s}")
    for label, (family, cfg) in M.FAMILIES.items():
        if label == "BiLSTM-h128":
            continue
        pv, pt = [], []
        for s in M.SEEDS:
            d = ROOT / "runs" / args.runs_subdir / label.replace(" ", "_") / f"seed{s}"
            if not (d / "best.pt").exists():
                d.mkdir(parents=True, exist_ok=True)
                E.train_run(family, cfg, s, M.DEVICE, data, pos_weight=pw,
                            select=M.SELECT, out_dir=d)
            pv.append(M.probs_for(d, family, cfg, data[2]))
            pt.append(M.probs_for(d, family, cfg, data[4]))
        tau = M.best_threshold(data[3], np.mean(pv, axis=0))
        per = [dict(auc=M.auc(yte, p), pr_auc=M.pr_auc(yte, p), f1=M.f1_at(yte, p, tau)) for p in pt]
        agg = {k: (float(np.mean([q[k] for q in per])), float(np.std([q[k] for q in per], ddof=1)))
               for k in per[0]}
        results[label] = dict(tau=float(tau), mean_sd=agg)
        print(f"{label:34s} {agg['auc'][0]:8.4f} {agg['pr_auc'][0]:8.4f} {agg['f1'][0]:8.4f}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    for name, sel in (("LR bbox + ego-speed (80)", lambda Z: Z),
                      ("LR ego-speed only (16)", lambda Z: Z[:, :, 4:5]),
                      ("LR bbox only (64)", lambda Z: Z[:, :, :4])):
        A, V, B = (sel(Z).reshape(len(Z), -1) for Z in (data[0], data[2], data[4]))
        sc = StandardScaler().fit(A)
        lr = LogisticRegression(max_iter=5000, class_weight={0: 1.0, 1: pw}).fit(sc.transform(A), data[1])
        pvv = lr.predict_proba(sc.transform(V))[:, 1]; ptt = lr.predict_proba(sc.transform(B))[:, 1]
        t = M.best_threshold(data[3], pvv)
        results[name] = dict(tau=float(t), mean_sd=dict(
            auc=(M.auc(yte, ptt), 0.0), pr_auc=(M.pr_auc(yte, ptt), 0.0), f1=(M.f1_at(yte, ptt, t), 0.0)))
        print(f"{name:34s} {M.auc(yte,ptt):8.4f} {M.pr_auc(yte,ptt):8.4f} {M.f1_at(yte,ptt,t):8.4f}")

    rule = json.loads((out_dir / "phase_rule.json").read_text()) \
        if (out_dir / "phase_rule.json").exists() else dict(phase_source="unknown")
    res_path.write_text(json.dumps(dict(
        phase_rule=rule,
        dataset=dict(n=int(len(yn)), pos=int(yn.sum()), neg=int((yn == 0).sum()),
                     pos_weight=pw, to_end_auc=float(auc_of(yn, te)),
                     to_end_auc_by_split=sep,
                     n_train=int(tr.sum()), n_val=int(va.sum()), n_test=int(tt.sum()),
                     n_test_pedestrians=int(len(set(groups)))),
        results=results), indent=2))
    print(f"\nwrote {res_path}")


if __name__ == "__main__":
    main()
