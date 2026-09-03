"""
11_demo_clean_ensemble.py — live demo on the CLEAN protocol, headline model.

Why this exists alongside 10_yolo_bytetrack_demo.py
---------------------------------------------------
`10_` is the Phase-4 demo and is kept as the historical record. It loads a single
h128 checkpoint from `paper_and_artifacts/runs/bilstm_baseline`, which is the
**legacy leaky-protocol model** — the one the manuscript retracts. Anything it
renders cannot appear in the journal paper beside clean-protocol results.

This script runs the same YOLO26 + ByteTrack front end but drives it with the
paper's actual headline predictor:

    BiLSTM-F1  =  BiLSTMIntentPredictor(hidden_dim=256, num_layers=2, dropout=0.3)
                  trained on the crossing-point-anchored protocol,
                  5-seed probability ensemble, operating threshold tau* = 0.5164

Three differences from `10_` that matter:

  1. `--hidden` is settable, so the h256 headline model loads.
  2. `--weights-dirs` takes a comma-separated list; probabilities are averaged
     across seeds. This is the deployable predictor and the one the confusion
     matrices in journal_prep/Analysis/ already report, so the demo and the
     tables describe the same system.
  3. `--stage verify` scores the clean test set through the assembled ensemble
     and checks it reproduces the published numbers before any frame is
     rendered. Run it first. A silent feature-order or norm-stats mismatch would
     otherwise poison every downstream figure.

It also saves RAW frames next to the annotated video. The burned-in OpenCV
overlay is fine for a video but too coarse for a journal page; the publication
figure is drawn in matplotlib from the raw frame plus the CSV.

Usage
-----
    # 0. gate (do this first, takes ~20 s, no video needed)
    python pipeline/11_demo_clean_ensemble.py --stage verify

    # 1. render a segment
    python pipeline/11_demo_clean_ensemble.py --stage demo \
        --video PIE_clips/set03/video_0012.mp4 --video-id video_0012 \
        --start-frame 7676 --max-frames 240 \
        --out-dir pipeline/demo_out_clean --dump-csv --save-raw-frames
"""

import argparse
import csv
import json
from collections import deque
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np
import torch

OBS_LEN = 16
PERSON_CLASS = 0
EXPECTED_W, EXPECTED_H = 1920, 1080

ROOT = Path(__file__).resolve().parent.parent


def _load(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BiLSTM = _load("m_bilstm_legacy", ROOT / "src" / "model_bilstm_legacy.py").BiLSTMIntentPredictor

# The published headline arm: f1_optimization/05_final_arms.json -> arms.A3
F1_RUN_ROOT = ROOT / "f1_optimization" / "runs_f1" / "lstm_lr1e-03_do0.3_h256_nl2" / "pw1.682"
F1_SEEDS = [42, 0, 1, 2, 3]
F1_HIDDEN = 256
F1_TAU = 0.5164303779602051

# What --stage verify must reproduce (same file, arms.A3.ens.test)
EXPECTED = {"auc": 0.9467385396564105, "f1": 0.8556851311953353, "acc": 0.9054441260744985}
TOL = 1e-4


def default_weights_dirs() -> str:
    return ",".join(str(F1_RUN_ROOT / f"seed{s}") for s in F1_SEEDS)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ultralytics_device(device: torch.device):
    return {"cuda": 0, "mps": "mps", "cpu": "cpu"}[device.type]


# ---------------------------------------------------------------------------
# Ensemble loading
# ---------------------------------------------------------------------------

def load_ensemble(weights_dirs, hidden: int, device: torch.device):
    """Load N checkpoints that share one normalization. Returns (models, mean, std).

    The seeds differ only in initialization, so their train-split z-score stats
    must be identical; we assert that rather than assume it, because a mismatched
    pair would shift the input distribution for part of the ensemble and the
    error would be invisible in the output.
    """
    models, mean0, std0 = [], None, None
    for d in weights_dirs:
        d = Path(d)
        mean = np.load(d / "norm_mean.npy").astype(np.float32)
        std = np.load(d / "norm_std.npy").astype(np.float32)
        assert mean.shape == (5,) and std.shape == (5,), \
            f"{d}: expected (5,) norm stats, got mean{mean.shape} std{std.shape}"
        if mean0 is None:
            mean0, std0 = mean, std
        else:
            assert np.allclose(mean, mean0, atol=1e-6) and np.allclose(std, std0, atol=1e-6), \
                (f"{d}: norm stats differ from the first checkpoint. The ensemble "
                 f"members were not trained on the same split; refusing to average them.")

        model = BiLSTM(input_dim=5, hidden_dim=hidden, num_layers=2, dropout=0.3).to(device)
        ckpt = torch.load(d / "best.pt", map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()
        models.append(model)
        ep = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
        print(f"[model] {d.name}: loaded (best epoch {ep})")

    n_par = sum(p.numel() for p in models[0].parameters())
    print(f"[model] ensemble of {len(models)} x h{hidden} BiLSTM "
          f"({n_par:,} params each) on {device}")
    return models, mean0, std0


@torch.no_grad()
def predict_batch(models, X: np.ndarray, mean, std, device) -> np.ndarray:
    """X: (N, OBS_LEN, 5) raw -> mean sigmoid over the ensemble, shape (N,)."""
    x = (X.astype(np.float32) - mean) / std
    xt = torch.from_numpy(x).to(device)
    acc = None
    for m in models:
        out = m(xt)
        out = out[0] if isinstance(out, tuple) else out
        p = torch.sigmoid(out.squeeze(-1))
        acc = p if acc is None else acc + p
    return (acc / len(models)).cpu().numpy()


# ---------------------------------------------------------------------------
# Stage: verify  — the gate
# ---------------------------------------------------------------------------

def stage_verify(args, device):
    """Score the clean test set through the assembled ensemble and require it to
    reproduce the published arm. Nothing downstream is trustworthy until this passes."""
    eng_path = ROOT / "src" / "engine.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("eng", eng_path)
    eng = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eng)
    *_, X_test, y_test = eng.load_splits()
    y_test = y_test.astype(int)
    print(f"[verify] clean test set: {len(y_test)} windows, {y_test.mean():.4f} positive")

    models, mean, std = load_ensemble(args.weights_dirs, args.hidden, device)
    p = predict_batch(models, X_test, mean, std, device)

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    got = {
        "auc": roc_auc_score(y_test, p),
        "f1": f1_score(y_test, (p >= args.threshold).astype(int)),
        "acc": accuracy_score(y_test, (p >= args.threshold).astype(int)),
    }

    print(f"[verify] threshold tau = {args.threshold:.10f}")
    ok = True
    for k in ("auc", "f1", "acc"):
        d = abs(got[k] - EXPECTED[k])
        flag = "OK " if d <= TOL else "FAIL"
        ok &= d <= TOL
        print(f"[verify] {flag} {k:4s} got {got[k]:.6f}  expected {EXPECTED[k]:.6f}  |d| {d:.2e}")

    if not ok:
        raise SystemExit(
            "\n[verify] PARITY GATE FAILED. The ensemble does not reproduce the published\n"
            "         arm (f1_optimization/05_final_arms.json -> arms.A3.ens.test).\n"
            "         Likely causes: wrong --hidden, wrong seed set, feature order, or\n"
            "         mismatched norm stats. Do NOT render frames until this passes."
        )
    print("[verify] PASS — this ensemble is the paper's headline model. Safe to render.")


# ---------------------------------------------------------------------------
# Ego-speed / video helpers (same contract as 10_)
# ---------------------------------------------------------------------------

def build_speed_map(args) -> dict:
    if args.ego_source == "obd":
        parse_obd = import_module("01_parse_annotations").parse_obd
        smap = parse_obd(args.obd_xml)
        print(f"[speed] {len(smap)} frames from OBD xml {args.obd_xml}")
        return smap
    import pandas as pd
    df = pd.read_pickle(args.annotations_pkl)
    sub = df[(df["set_id"] == args.set_id) & (df["video_id"] == args.video_id)]
    if sub.empty:
        raise ValueError(f"No rows in {args.annotations_pkl} for {args.set_id}/{args.video_id}")
    smap = sub.groupby("frame")["vehicle_speed"].first().to_dict()
    smap = {int(k): float(v) for k, v in smap.items()}
    print(f"[speed] {len(smap)} annotated frames "
          f"({args.set_id}/{args.video_id}, range {min(smap)}..{max(smap)})")
    return smap


def video_meta(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, w, h, n


def frame_reader(path: str, start: int, maxn):
    cap = cv2.VideoCapture(path)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    i = 0
    while maxn is None or i < maxn:
        ok, frame = cap.read()
        if not ok:
            break
        yield start + i, frame
        i += 1
    cap.release()


def load_yolo():
    import ultralytics
    from ultralytics import YOLO
    print(f"[yolo] ultralytics {ultralytics.__version__}")
    model = YOLO("yolo26m.pt")
    print("[yolo] loaded yolo26m.pt")
    return model


def blur_heads_inplace(img, yolo, dev, conf=0.05):
    """Blur the head region of every person in the frame, for the published video.

    A separate, deliberately low-confidence pass: the tracking pass runs at the
    conf used for every result in the paper and must not be perturbed, but for
    privacy we would rather blur a patch of pavement than miss a bystander.
    """
    r = yolo.predict(img, classes=[PERSON_CLASS], conf=conf, device=dev, verbose=False)[0]
    for b in r.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        hy2 = y1 + max(10, int(0.24 * (y2 - y1)))
        x1, y1 = max(0, x1), max(0, y1)
        x2, hy2 = min(img.shape[1], x2), min(img.shape[0], hy2)
        if x2 - x1 < 4 or hy2 - y1 < 4:
            continue
        roi = img[y1:hy2, x1:x2]
        k = max(11, (min(roi.shape[:2]) // 2) * 2 + 1)
        img[y1:hy2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return img


def draw_box(img, xyxy, label, prob, tau):
    """Video overlay only. The publication figure is drawn in matplotlib."""
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    if prob is None:
        color = (190, 190, 190)                 # warming up, < 16 frames buffered
    elif prob >= tau:
        color = (214, 120, 42)                  # BGR of the palette accent blue
    else:
        color = (178, 184, 185)                 # de-emphasis grey
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Stage: demo
# ---------------------------------------------------------------------------

def stage_demo(args, device):
    yolo = load_yolo()
    dev = ultralytics_device(device)
    models, mean, std = load_ensemble(args.weights_dirs, args.hidden, device)
    smap = build_speed_map(args)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    if args.save_raw_frames:
        raw_dir.mkdir(parents=True, exist_ok=True)

    fps, w, h, n = video_meta(args.video)
    if (w, h) != (EXPECTED_W, EXPECTED_H):
        print(f"[warn] video is {w}x{h}, training coords are {EXPECTED_W}x{EXPECTED_H}")
    print(f"[demo] {args.video} {w}x{h} @ {fps:.1f}fps | "
          f"start={args.start_frame} max={args.max_frames} tau={args.threshold:.4f}")

    writer = None
    if args.write_video:
        writer = cv2.VideoWriter(str(out_dir / f"demo_{args.video_id}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    buffers, last_frame, last_prob = {}, {}, {}
    rows, last_speed, seen, n_pred = [], 0.0, 0, 0

    for fidx, frame in frame_reader(args.video, args.start_frame, args.max_frames):
        seen += 1
        ego = smap.get(fidx, last_speed)
        if not np.isfinite(ego):
            ego = last_speed
        last_speed = ego
        img = frame.copy() if writer is not None else None
        if img is not None and args.blur_faces:
            blur_heads_inplace(img, yolo, dev)

        r = yolo.track(frame, classes=[PERSON_CLASS], conf=args.conf,
                       tracker="bytetrack.yaml", persist=True,
                       device=dev, verbose=False)[0]
        ids = r.boxes.id

        if ids is not None:
            pend_tids, pend_wins, pend_xyxy = [], [], []
            for b, tid in zip(r.boxes, ids.int().tolist()):
                xyxy = b.xyxy[0].tolist()
                if tid in last_frame and fidx - last_frame[tid] > 1:
                    buffers.pop(tid, None)      # gap in the track: restart the window
                last_frame[tid] = fidx
                buf = buffers.setdefault(tid, deque(maxlen=OBS_LEN))
                buf.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], ego])
                if len(buf) == OBS_LEN:
                    pend_tids.append(tid); pend_wins.append(np.asarray(buf)); pend_xyxy.append(xyxy)

            if pend_tids:                        # one batched forward per frame
                probs = predict_batch(models, np.stack(pend_wins), mean, std, device)
                n_pred += len(probs)
                for tid, prob, xyxy in zip(pend_tids, probs, pend_xyxy):
                    last_prob[tid] = float(prob)
                    if args.dump_csv:
                        rows.append([fidx, tid, *[f"{v:.1f}" for v in xyxy],
                                     f"{ego:.3f}", f"{prob:.4f}",
                                     int(prob >= args.threshold)])

            if img is not None:
                for b, tid in zip(r.boxes, ids.int().tolist()):
                    p = last_prob.get(tid)
                    lab = f"ID{tid} {p:.2f}" if p is not None else f"ID{tid} ..."
                    draw_box(img, b.xyxy[0].tolist(), lab, p, args.threshold)

        if img is not None:
            cv2.putText(img, f"frame {fidx} | ego {ego:.1f} km/h", (20, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(img)

        if args.save_raw_frames and (seen - 1) % args.sample_every == 0:
            cv2.imwrite(str(raw_dir / f"{args.video_id}_f{fidx:05d}.png"), frame)

    if writer is not None:
        writer.release()
        print(f"[demo] wrote {out_dir / f'demo_{args.video_id}.mp4'}")
    print(f"[demo] {seen} frames, {n_pred} window predictions")

    if args.dump_csv:
        csv_path = out_dir / f"demo_{args.video_id}_predictions.csv"
        with open(csv_path, "w", newline="") as f:
            wc = csv.writer(f)
            wc.writerow(["frame", "track_id", "x1", "y1", "x2", "y2",
                         "ego_speed", "prob_cross", "pred"])
            wc.writerows(rows)
        print(f"[demo] wrote {len(rows)} rows -> {csv_path}")

    meta = {"video_id": args.video_id, "start_frame": args.start_frame,
            "max_frames": args.max_frames, "threshold": args.threshold,
            "hidden": args.hidden, "weights_dirs": [str(d) for d in args.weights_dirs],
            "model": "BiLSTM-F1 (clean crossing-point protocol), 5-seed ensemble",
            "yolo_conf": args.conf, "n_predictions": n_pred}
    (out_dir / f"demo_{args.video_id}_meta.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Clean-protocol YOLO26 + ByteTrack + BiLSTM-F1 ensemble demo")
    ap.add_argument("--stage", choices=["verify", "demo"], default="demo")
    ap.add_argument("--video")
    ap.add_argument("--set-id", default="set03")
    ap.add_argument("--video-id", default="video_0012")
    ap.add_argument("--weights-dirs", default=default_weights_dirs(),
                    help="comma-separated checkpoint dirs; probabilities are averaged")
    ap.add_argument("--hidden", type=int, default=F1_HIDDEN)
    ap.add_argument("--threshold", type=float, default=F1_TAU)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--ego-source", choices=["pkl", "obd"], default="pkl")
    ap.add_argument("--annotations-pkl", default="pie_annotations.pkl")
    ap.add_argument("--obd-xml", default=None)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--out-dir", default="pipeline/demo_out_clean")
    ap.add_argument("--sample-every", type=int, default=10,
                    help="save a raw frame every N processed frames")
    ap.add_argument("--dump-csv", action="store_true")
    ap.add_argument("--save-raw-frames", action="store_true",
                    help="save unannotated frames for the publication figure")
    ap.add_argument("--write-video", action="store_true",
                    help="also write the burned-in overlay mp4 (supplementary video)")
    ap.add_argument("--blur-faces", action="store_true",
                    help="blur every detected head in the written video (publication)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = ap.parse_args()

    args.weights_dirs = [Path(d.strip()) for d in args.weights_dirs.split(",") if d.strip()]
    for d in args.weights_dirs:
        if not (d / "best.pt").exists():
            ap.error(f"no best.pt in {d}")
    if args.stage == "demo" and not args.video:
        ap.error("--stage demo requires --video")
    if args.ego_source == "obd" and not args.obd_xml:
        ap.error("--ego-source obd requires --obd-xml")

    device = pick_device(args.device)
    print(f"[init] device: {device} | stage: {args.stage}")
    if args.stage == "verify":
        stage_verify(args, device)
    else:
        stage_demo(args, device)


if __name__ == "__main__":
    main()
