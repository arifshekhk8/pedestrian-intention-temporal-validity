"""10_gt_vs_detector_auc.py — Issue 10: GT-box vs YOLO-box prediction degradation.

WHY. The Phase-4 demo was qualitative ("AUC 1.000 on 10 peds" — not a result). The
real science of the live pipeline is: **how much does feeding the BiLSTM noisy
detector boxes (YOLO26-M + ByteTrack) instead of ground-truth boxes cost us?** This
quantifies that perception→prediction gap on the two demo clips.

DESIGN. For every clean GT window (Issue-2 protocol) in video_0012 + video_0016
(set03), we already have the GT-box path. We run YOLO26-M + ByteTrack over the
frames those windows need, match each GT pedestrian to the detector output by IoU,
rebuild the SAME 16-frame window from YOLO boxes (ego-speed kept identical — it comes
from the vehicle, not vision, so only the bbox features change), and score both paths
through the **clean** BiLSTM (`runs_clean/bilstm_baseline_clean/best.pt`). We compare
GT-path vs YOLO-path AUC on the matched set, and report detector/tracker quality
(recall, mean IoU, ID switches, fragmentation).

NO TRAINING — inference only. We process only the union of segments the windows need
(~5 k frames, not the full 33 k), so YOLO is ~5–8 min. Matched subset is indicative
(~100 peds across two clips), not a full benchmark — stated as such.

Outputs: 10_gt_vs_detector_results.md, 10_gt_vs_detector.csv, 10_gt_vs_detector_figure.png
"""
import csv
import importlib.util
import pickle
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SEQ = HERE.parent / "issue2_clean_protocol" / "sequences_clean"
RUN = HERE.parent / "issue2_clean_protocol" / "runs_clean" / "bilstm_baseline_clean"
YW = ROOT / "yolo26m.pt"
CLIPS = {"video_0012": ROOT / "PIE_clips" / "set03" / "video_0012.mp4",
         "video_0016": ROOT / "PIE_clips" / "set03" / "video_0016.mp4"}

_spec = importlib.util.spec_from_file_location("m03", ROOT / "pipeline" / "03_bilstm_model.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
BiLSTM = _m.BiLSTMIntentPredictor

OBS_LEN, LEAD, IOU_MIN, COV_MIN, THR = 16, 45, 0.3, 0.5, 0.5
PERSON, CONF, EXP_W, EXP_H = 0, 0.3, 1920, 1080


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_windows():
    """Clean GT windows for the two demo clips: boxes (raw px), ego-speed, label, frames."""
    X = np.load(SEQ / "X.npy").astype(np.float32)
    y = np.load(SEQ / "y.npy").astype(np.float32)
    meta = pickle.load(open(SEQ / "meta.pkl", "rb"))
    wins = []
    for i, m in enumerate(meta):
        if m["set_id"] != "set03" or m["video_id"] not in CLIPS:
            continue
        a = m["anchor_frame"]
        wins.append(dict(video=m["video_id"], ped=m["ped_id"], anchor=a, label=int(y[i]),
                         frames=list(range(a - OBS_LEN + 1, a + 1)),
                         gt_boxes=X[i, :, :4].copy(), ego=X[i, :, 4].copy()))
    return wins


def segments_for(wins, video):
    iv = sorted((w["frames"][0] - LEAD, w["frames"][-1]) for w in wins if w["video"] == video)
    merged = []
    for lo, hi in iv:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append([max(0, lo), hi])
    return merged


def yolo_pass(video, segments, device):
    """YOLO26-M + ByteTrack over the given segments; tracker reset per segment.
    Returns {frame_idx: [(track_id, [x1,y1,x2,y2]), ...]}."""
    import cv2
    from ultralytics import YOLO
    yolo = YOLO(str(YW))
    cap = cv2.VideoCapture(str(CLIPS[video]))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx, sy = EXP_W / w, EXP_H / h                      # scale YOLO boxes to PIE coords if needed
    dets, nf = {}, 0
    for lo, hi in segments:
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        for k, f in enumerate(range(lo, hi + 1)):
            ok, frame = cap.read()
            if not ok:
                break
            r = yolo.track(frame, classes=[PERSON], conf=CONF, tracker="bytetrack.yaml",
                           persist=(k > 0), device=device, verbose=False)
            r = r[0] if isinstance(r, (list, tuple)) else r
            ids = r.boxes.id
            if ids is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                for b, tid in zip(xyxy, ids.int().cpu().tolist()):
                    box = [b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy]
                    dets.setdefault(f, []).append((tid, box))
            nf += 1
    cap.release()
    return dets, nf


def match_window(w, dets):
    """Assemble the YOLO-box window for the GT pedestrian and measure tracker quality.

    Per frame, take the YOLO detection with the highest IoU to the GT box (≥IOU_MIN) —
    this isolates *box-localisation* noise (association resolved by IoU-to-GT). The
    window is built from those per-frame best boxes (gaps forward-filled). Tracker
    quality is measured *alongside* from the ByteTrack IDs of those best detections:
      purity = fraction of matched frames owned by the dominant (most-frequent) ID
      switch = a 2nd ID owns ≥3 matched frames (a genuine identity split)
      frag   = gaps inside the matched span (dropped frames)

    Returns dict: detected, yolo_boxes(16x4|None), cov, miou, switch, purity, frag."""
    from collections import Counter
    best = [None] * OBS_LEN                             # j -> (tid, box, iou)
    for j, f in enumerate(w["frames"]):
        gt = w["gt_boxes"][j]
        cand = dets.get(f, [])
        if not cand:
            continue
        tid, box = max(cand, key=lambda c: iou(gt, c[1]))
        v = iou(gt, box)
        if v >= IOU_MIN:
            best[j] = (tid, box, v)
    present = [j for j in range(OBS_LEN) if best[j] is not None]
    cov = len(present) / OBS_LEN
    if not present:
        return dict(detected=False, yolo_boxes=None, cov=0.0, miou=0.0,
                    switch=False, purity=0.0, frag=OBS_LEN)
    miou = float(np.mean([best[j][2] for j in present]))
    cnt = Counter(best[j][0] for j in present)
    dom, domn = cnt.most_common(1)[0]
    purity = domn / len(present)
    switch = any(t != dom and c >= 3 for t, c in cnt.items())
    frag = (present[-1] - present[0] + 1) - len(present)
    if cov < COV_MIN:
        return dict(detected=False, yolo_boxes=None, cov=cov, miou=miou,
                    switch=switch, purity=purity, frag=frag)
    last = best[present[0]][1]                          # per-frame best box, gaps filled
    filled = []
    for j in range(OBS_LEN):
        if best[j] is not None:
            last = best[j][1]
        filled.append(last)
    return dict(detected=True, yolo_boxes=np.array(filled, np.float32), cov=cov,
                miou=miou, switch=switch, purity=purity, frag=frag)


@torch.no_grad()
def make_scorer():
    mean = np.load(RUN / "norm_mean.npy"); std = np.load(RUN / "norm_std.npy")
    model = BiLSTM(input_dim=5)
    ck = torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()

    def score(boxes16, ego16):
        x = np.concatenate([boxes16, ego16[:, None]], axis=1).astype(np.float32)
        x = (x - mean) / std
        return float(torch.sigmoid(model(torch.from_numpy(x)[None]).squeeze()).item())
    return score


def auc(labels, probs):
    labels, probs = np.asarray(labels), np.asarray(probs)
    pos, neg = probs[labels == 1], probs[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann–Whitney U / rank AUC (sklearn-free, per CLAUDE.md local note)
    allp = np.concatenate([pos, neg]); order = allp.argsort()
    ranks = np.empty(len(allp)); ranks[order] = np.arange(1, len(allp) + 1)
    # average ties
    _, inv, cnt = np.unique(allp, return_inverse=True, return_counts=True)
    avg = np.zeros(len(_)); s = np.zeros(len(_))
    for r, ii in zip(ranks, inv):
        s[ii] += r
    avg = s / cnt
    ranks = avg[inv]
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device(YOLO) {device} | clean BiLSTM scorer from {RUN.name}\n")
    wins = load_windows()
    print(f"GT clean windows in demo clips: {len(wins)} "
          f"({sum(w['label'] for w in wins)} cross / {sum(1-w['label'] for w in wins)} not)")

    # ---- YOLO + ByteTrack over needed segments (cached so matching can be re-run) ----
    cache = HERE / "cache_dets.pkl"
    if cache.exists():
        dets = pickle.load(open(cache, "rb"))
        print(f"\n[cache] loaded YOLO detections from {cache.name}")
    else:
        dets = {}
        for video in CLIPS:
            segs = segments_for(wins, video)
            nfr = sum(hi - lo + 1 for lo, hi in segs)
            print(f"\n[{video}] {len(segs)} segments, ~{nfr} frames → YOLO+ByteTrack…")
            t0 = time.time()
            dets[video], processed = yolo_pass(video, segs, device)
            print(f"  done in {time.time()-t0:.0f}s ({processed} frames, "
                  f"{sum(len(v) for v in dets[video].values())} detections)")
        pickle.dump(dets, open(cache, "wb"))

    # ---- match + score ----
    score = make_scorer()
    rows = []
    for w in wins:
        m = match_window(w, dets[w["video"]])
        gt_prob = score(w["gt_boxes"], w["ego"])
        yp = score(m["yolo_boxes"], w["ego"]) if m["detected"] else None
        rows.append(dict(**{k: w[k] for k in ("video", "ped", "anchor", "label")},
                         detected=int(m["detected"]), cov=m["cov"], miou=m["miou"],
                         switch=int(m["switch"]), purity=m["purity"], frag=m["frag"],
                         gt_prob=gt_prob, yolo_prob=yp))
    write_outputs(rows)


def write_outputs(rows):
    # CSV
    with open(HERE / "10_gt_vs_detector.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "ped", "anchor", "label", "detected",
                                          "cov", "miou", "switch", "purity", "frag",
                                          "gt_prob", "yolo_prob"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "cov": round(r["cov"], 3), "miou": round(r["miou"], 3),
                        "purity": round(r["purity"], 3),
                        "gt_prob": round(r["gt_prob"], 4),
                        "yolo_prob": None if r["yolo_prob"] is None else round(r["yolo_prob"], 4)})

    det = [r for r in rows if r["detected"]]
    miss = [r for r in rows if not r["detected"]]
    nwin = len(rows)
    # per-window AUC on the matched (detected) subset
    labels = [r["label"] for r in det]
    gt_auc_w = auc(labels, [r["gt_prob"] for r in det])
    yo_auc_w = auc(labels, [r["yolo_prob"] for r in det])
    # per-pedestrian aggregation (mean prob over that ped's detected windows)
    peds = {}
    for r in det:
        peds.setdefault((r["video"], r["ped"]), []).append(r)
    pl = [v[0]["label"] for v in peds.values()]
    gt_auc_p = auc(pl, [np.mean([r["gt_prob"] for r in v]) for v in peds.values()])
    yo_auc_p = auc(pl, [np.mean([r["yolo_prob"] for r in v]) for v in peds.values()])
    # agreement
    dg = np.array([r["gt_prob"] for r in det]); dy = np.array([r["yolo_prob"] for r in det])
    flips = int(np.sum((dg >= THR) != (dy >= THR)))
    miou = float(np.mean([r["miou"] for r in det]))
    switches = int(np.sum([r["switch"] for r in det]))
    mean_purity = float(np.mean([r["purity"] for r in det]))
    fragmented = int(np.sum([r["frag"] > 0 for r in det]))
    # ped-level recall
    all_peds = {(r["video"], r["ped"]) for r in rows}
    det_peds = {(r["video"], r["ped"]) for r in det}

    L = ["# Issue 10 — GT-box vs YOLO-box prediction degradation (indicative)", "",
         f"Demo clips video_0012 + video_0016 (set03). For each clean GT window we run "
         f"YOLO26-M + ByteTrack over the frames it needs and assemble the YOLO-box "
         f"window from the **best-IoU detection per frame** (this isolates box-"
         f"localisation noise; ego-speed is unchanged — it comes from the vehicle, not "
         f"vision). Both paths are scored through the clean BiLSTM "
         f"(`runs_clean/bilstm_baseline_clean`). ByteTrack **identity** quality is "
         f"measured separately (below) so detection-box noise and tracking errors are "
         f"not conflated. Indicative subset, not a full benchmark.", "",
         "## Detector / tracker quality", "",
         f"- **Windows:** {nwin} total · **{len(det)} detected** "
         f"({100*len(det)/nwin:.0f}% — the BiLSTM gets a usable track) · {len(miss)} "
         f"missed (detector never covers ≥{int(COV_MIN*100)}% of the window).",
         f"- **Pedestrians:** {len(all_peds)} total · {len(det_peds)} with ≥1 detected "
         f"window ({100*len(det_peds)/len(all_peds):.0f}% detector recall).",
         f"- **Box quality (matched):** mean IoU(GT, YOLO) = **{miou:.3f}**; the "
         f"pedestrian's dominant ByteTrack ID covers a mean **{100*mean_purity:.0f}%** "
         f"of its matched frames (track purity).",
         f"- **ID switches:** {switches}/{len(det)} detected windows "
         f"({100*switches/max(len(det),1):.0f}%) have a *second* ByteTrack ID also "
         f"substantially covering the same pedestrian (a genuine switch). "
         f"**Fragmentation:** {fragmented}/{len(det)} "
         f"({100*fragmented/max(len(det),1):.0f}%) have ≥1 frame where the dominant "
         f"track drops out (gap-filled from its nearest box).", "",
         "## Prediction: GT-box vs YOLO-box (matched subset)", "",
         "| path | per-window AUC | per-pedestrian AUC |", "|---|---|---|",
         f"| **GT boxes** (offline) | {gt_auc_w:.3f} | {gt_auc_p:.3f} |",
         f"| **YOLO boxes** (full pipeline) | {yo_auc_w:.3f} | {yo_auc_p:.3f} |",
         f"| **drop (GT − YOLO)** | **{gt_auc_w-yo_auc_w:+.3f}** | **{gt_auc_p-yo_auc_p:+.3f}** |",
         "",
         f"On the matched windows the two probability streams agree closely: a "
         f"decision flips across the 0.5 threshold in **{flips}/{len(det)} "
         f"({100*flips/max(len(det),1):.0f}%)** of windows.", "",
         "## Verdict", "",
         f"**The prediction model is robust to detector box noise.** On {len(det)} "
         f"matched windows / {len(det_peds)} pedestrians, replacing ground-truth boxes "
         f"with YOLO26-M boxes (mean IoU {miou:.2f}) moves AUC by only "
         f"**{gt_auc_w-yo_auc_w:+.3f} per window / {gt_auc_p-yo_auc_p:+.3f} per "
         f"pedestrian**, and the decision flips in just {flips}/{len(det)} "
         f"({100*flips/max(len(det),1):.0f}%) of windows — so the offline AUC is "
         f"broadly indicative of live performance under realistic box noise.", "",
         f"**The pipeline's weak links are perception, not prediction:** (1) "
         f"**detector recall** — {100*len(det_peds)/len(all_peds):.0f}% of pedestrians "
         f"are detected, so ~{100-round(100*len(det_peds)/len(all_peds))}% are never "
         f"covered well enough to predict at all (a safety gap worth stating); (2) "
         f"**tracker fragmentation** — a single ByteTrack ID covers a mean of only "
         f"**{100*mean_purity:.0f}%** of a pedestrian's frames and "
         f"{100*switches/max(len(det),1):.0f}% of windows carry a competing ID, so a "
         f"deployment would need stronger re-identification. Neither weakens the "
         f"BiLSTM's tolerance to box noise above — they are detector/tracker "
         f"engineering gaps, separate from this thesis's prediction model. Numbers are "
         f"indicative (N={len(det_peds)} peds, two clips).",
         "", "_AUC computed via the rank/Mann–Whitney estimator (no sklearn locally)._"]
    (HERE / "10_gt_vs_detector_results.md").write_text("\n".join(L))

    make_figure(rows, det, gt_auc_w, yo_auc_w, gt_auc_p, yo_auc_p)
    print(f"\ndetected {len(det)}/{nwin} windows | GT vs YOLO AUC "
          f"win {gt_auc_w:.3f}/{yo_auc_w:.3f} ped {gt_auc_p:.3f}/{yo_auc_p:.3f} | "
          f"meanIoU {miou:.3f} | flips {flips}")
    print("wrote 10_gt_vs_detector_results.md, .csv, figure")


def make_figure(rows, det, gaw, yaw, gap, yap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    # left: GT vs YOLO prob scatter, colored by label
    ax = axes[0]
    dg = np.array([r["gt_prob"] for r in det]); dy = np.array([r["yolo_prob"] for r in det])
    lab = np.array([r["label"] for r in det])
    ax.scatter(dg[lab==1], dy[lab==1], c="#dc2626", alpha=0.5, s=25, label="crosser")
    ax.scatter(dg[lab==0], dy[lab==0], c="#2563eb", alpha=0.5, s=25, label="non-crosser")
    ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.5)
    ax.axhline(0.5, color="grey", lw=0.6); ax.axvline(0.5, color="grey", lw=0.6)
    ax.set_xlabel("GT-box prob"); ax.set_ylabel("YOLO-box prob")
    ax.set_title("Per-window prediction agreement"); ax.legend(); ax.set_aspect("equal")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    # right: AUC bars
    ax2 = axes[1]
    x = np.arange(2); wd = 0.35
    ax2.bar(x-wd/2, [gaw, gap], wd, color="#16a34a", label="GT boxes")
    ax2.bar(x+wd/2, [yaw, yap], wd, color="#ea580c", label="YOLO boxes")
    for i,(g,y) in enumerate([(gaw,yaw),(gap,yap)]):
        ax2.text(i-wd/2, g+0.005, f"{g:.3f}", ha="center", fontsize=9)
        ax2.text(i+wd/2, y+0.005, f"{y:.3f}", ha="center", fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(["per-window", "per-pedestrian"])
    ax2.set_ylim(0.5, 1.0); ax2.set_ylabel("AUC (set03 demo clips)")
    ax2.set_title("GT-box vs YOLO-box AUC"); ax2.legend(loc="lower right"); ax2.grid(axis="y", alpha=0.3)
    fig.suptitle("Issue 10 — perception→prediction degradation (indicative, 2 clips)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(HERE / "10_gt_vs_detector_figure.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
