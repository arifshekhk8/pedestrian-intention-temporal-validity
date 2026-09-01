"""
12_supervisor_demo.py — presentation-grade demo of the clean-protocol pipeline.

What this adds over `11_demo_clean_ensemble.py`
-----------------------------------------------
`11_` is the workhorse: it renders frames and dumps the CSV the publication figure
is drawn from, and its overlay ("ID5 0.78") is a debugging overlay. This script
runs the identical model through the identical front end and spends the extra
effort on the presentation:

  * a readable verdict per pedestrian ("WILL CROSS 0.78") instead of a bare number,
    with a probability bar and a tick marking the operating threshold, so a viewer
    can see how close a call was;
  * a header showing which model is running, the threshold, the ego speed the model
    is actually being fed, and the measured throughput split into detector time and
    predictor time, which is the evidence for the latency claim in the paper;
  * `--live`, which plays the result in a window as it is computed, so the pipeline
    can be demonstrated rather than described;
  * `--scene`, a set of named presets so a demo is one command and not a hunt
    through frame numbers.

It shares every model path with `11_`: same checkpoints, same threshold, same
feature order, same normalization. It imports them rather than redefining them,
so the two cannot drift apart.

Run the parity gate first. It takes about 20 s, needs no video, and refuses to
proceed if the assembled ensemble is not the published model:

    python pipeline/11_demo_clean_ensemble.py --stage verify

Then, for example:

    python pipeline/12_supervisor_demo.py --scene anticipation --live
    python pipeline/12_supervisor_demo.py --scene anticipation --write-video

See HOW_TO_RUN_THE_DEMO.md for the full walkthrough.
"""

import argparse
import importlib.util
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def _load_sibling(name: str, filename: str):
    """Import a sibling script whose name starts with a digit."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "pipeline" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _load_sibling("demo11", "11_demo_clean_ensemble.py")

OBS_LEN = D.OBS_LEN
PERSON = D.PERSON_CLASS

# ---------------------------------------------------------------------------
# Scene presets
#
# Every one of these was picked from the ranked candidate table, not by eye, and
# each is labelled with what it is meant to show. The 'uncertainty' scene is
# included on purpose: a demo reel of nothing but successes invites the question
# of what was left out, and it is better to answer it before it is asked.
# ---------------------------------------------------------------------------

SCENES = {
    "anticipation": dict(
        video_id="video_0016", start=4270, frames=190,
        title="Anticipating a crossing",
        blurb="Pedestrian flagged while still on the kerb, 1.5 s before stepping out. "
              "This is the scene in Figure 9a."),
    "bystander": dict(
        video_id="video_0012", start=460, frames=200,
        title="Correctly ignoring a bystander",
        blurb="A worker stands at the kerb beside a marked crosswalk and does not "
              "cross. Same position as a crosser, opposite verdict. Figure 9b."),
    "busy": dict(
        video_id="video_0012", start=5560, frames=780,
        title="A busy corner, several tracks at once",
        blurb="Multiple pedestrians tracked simultaneously; watch probabilities "
              "firm up as each approach to the kerb resolves."),
    "driving": dict(
        video_id="video_0016", start=11950, frames=350,
        title="Driving, and staying quiet",
        blurb="The one scene where the ego vehicle is actually moving, 6 to 24 km/h. "
              "Pedestrians on the pavement are correctly left unflagged, which is the "
              "behaviour that matters operationally: no false alarms while cruising. "
              "The other scenes have the car stopped at a light, which is when "
              "pedestrians cross."),
    "uncertainty": dict(
        video_id="video_0016", start=4415, frames=340,
        title="Where the model is least sure",
        blurb="A group waiting at a kerb. Probabilities sit near the threshold and "
              "several non-crossers cross it. Shown deliberately, not hidden."),
}

# BGR, matching the figure palette in figures/figstyle.py
C_CROSS = (214, 120, 42)      # accent blue: predicted to cross
C_STAY = (170, 176, 178)      # recessive grey: predicted not to cross
C_WARM = (150, 150, 150)      # still filling the 16-frame buffer
C_PANEL = (26, 26, 24)        # header plate
C_TEXT = (255, 255, 255)
C_DIM = (186, 186, 182)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def _plate(img, x, y, w, h, color, alpha=0.82):
    """Alpha-blended filled rectangle. Text over raw video is unreadable."""
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = cv2.addWeighted(
        np.full_like(roi, color, dtype=np.uint8), alpha, roi, 1 - alpha, 0)


def _overlaps(a, b):
    return not (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0]
                or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])


def draw_pedestrian(img, xyxy, prob, tau, n_buf, scale, occupied, label_min_h=0):
    """One tracked pedestrian: box, verdict plate, probability bar.

    `occupied` is the list of plates already placed this frame. On a busy corner
    half a dozen pedestrians stand shoulder to shoulder and their labels land on
    top of each other, which makes the overlay unreadable exactly when there is
    most to read. Each plate is nudged upward until it finds clear space.

    `label_min_h` suppresses the plate (not the box) for pedestrians too far away
    for a label to be legible. Down the street the detector picks up a dozen
    people whose tracks keep breaking, and a screen of "buffering 2/16" plates
    buries the pedestrian the viewer is meant to be looking at. The boxes stay, so
    nothing is hidden; only the unreadable text goes.
    """
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    if prob is None:
        color, verdict = C_WARM, f"buffering {n_buf}/{OBS_LEN}"
    elif prob >= tau:
        color, verdict = C_CROSS, "WILL CROSS"
    else:
        color, verdict = C_STAY, "not crossing"

    # Two thresholds. A verdict is the point of the demo, so it survives on a
    # fairly small box. "buffering 2/16" only says a track is new, and a dozen of
    # them at once buries the pedestrian being pointed at, so it needs a much
    # bigger box to earn its plate. Boxes are always drawn either way.
    h = y2 - y1
    small = h < (label_min_h * 1.8 if prob is None else label_min_h)
    th = max(1, int((1.5 if small else 3) * scale))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, th)
    if small:
        return

    fs = 0.62 * scale
    ft = max(1, int(2 * scale))
    txt = verdict if prob is None else f"{verdict}  {prob:.2f}"
    (tw, tht), _ = cv2.getTextSize(txt, FONT, fs, ft)
    pad = int(8 * scale)
    bar_h = int(9 * scale) if prob is not None else 0
    ph = tht + 2 * pad + bar_h
    pw = max(tw + 2 * pad, min(x2 - x1, int(260 * scale)))
    px = min(max(0, x1), W - pw)
    py = max(0, y1 - ph - int(5 * scale))

    step = ph + int(4 * scale)
    for _ in range(8):                       # nudge up, then give up and overlap
        if not any(_overlaps((px, py, pw, ph), o) for o in occupied):
            break
        py -= step
        if py < 0:
            py = max(0, y1 - ph - int(5 * scale))
            break
    occupied.append((px, py, pw, ph))

    _plate(img, px, py, pw, ph, color, alpha=0.92)
    cv2.putText(img, txt, (px + pad, py + pad + tht), FONT, fs, C_TEXT, ft, cv2.LINE_AA)
    if py + ph < y1 - int(2 * scale):        # leader line back to a nudged plate
        cv2.line(img, (px + pw // 2, py + ph), (x1 + (x2 - x1) // 2, y1),
                 color, max(1, int(1.5 * scale)))

    if prob is not None:
        # Probability bar with a tick at the operating threshold, so a viewer can
        # see whether a call was comfortable or marginal.
        bx, by = px + pad, py + pad + tht + int(5 * scale)
        bw, bh = pw - 2 * pad, max(3, int(5 * scale))
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
        cv2.rectangle(img, (bx, by), (bx + int(bw * float(prob)), by + bh), C_TEXT, -1)
        tick = bx + int(bw * tau)
        cv2.line(img, (tick, by - int(3 * scale)), (tick, by + bh + int(3 * scale)),
                 (20, 20, 20), max(2, int(2 * scale)))


def draw_header(img, lines, scale):
    """Top-left status plate. Line 0 is the title, the rest are dim."""
    fs_t, fs = 0.78 * scale, 0.60 * scale
    ft = max(1, int(2 * scale))
    pad = int(16 * scale)
    gap = int(11 * scale)
    sizes = [cv2.getTextSize(t, FONT, fs_t if i == 0 else fs, ft)[0]
             for i, t in enumerate(lines)]
    w = max(s[0] for s in sizes) + 2 * pad
    h = sum(s[1] for s in sizes) + gap * (len(lines) - 1) + 2 * pad
    _plate(img, int(14 * scale), int(14 * scale), w, h, C_PANEL)
    y = int(14 * scale) + pad
    for i, (t, (tw, tht)) in enumerate(zip(lines, sizes)):
        y += tht
        cv2.putText(img, t, (int(14 * scale) + pad, y), FONT,
                    fs_t if i == 0 else fs, C_TEXT if i == 0 else C_DIM,
                    ft, cv2.LINE_AA)
        y += gap


def draw_legend(img, tau, scale):
    """Bottom-left key. Without it the colours are just colours."""
    items = [(C_CROSS, f"predicted to cross  (p >= {tau:.2f})"),
             (C_STAY, "predicted not to cross"),
             (C_WARM, f"filling the {OBS_LEN}-frame window")]
    fs = 0.56 * scale
    ft = max(1, int(2 * scale))
    pad, gap, sw = int(14 * scale), int(10 * scale), int(26 * scale)
    sizes = [cv2.getTextSize(t, FONT, fs, ft)[0] for _, t in items]
    w = max(s[0] for s in sizes) + sw + 3 * pad
    h = sum(s[1] for s in sizes) + gap * (len(items) - 1) + 2 * pad
    x0 = int(14 * scale)
    y0 = img.shape[0] - h - int(14 * scale)
    _plate(img, x0, y0, w, h, C_PANEL)
    y = y0 + pad
    for (color, t), (tw, tht) in zip(items, sizes):
        y += tht
        cv2.rectangle(img, (x0 + pad, y - tht), (x0 + pad + sw, y), color, -1)
        cv2.putText(img, t, (x0 + pad + sw + pad, y), FONT, fs, C_DIM, ft, cv2.LINE_AA)
        y += gap


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

WINDOW = "crossing intention  |  q quit  space pause"


def run(args, device):
    if args.live:
        # Fail here with something readable rather than a cv2.error 200 frames in,
        # which is what happens over SSH or in a headless shell.
        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        except cv2.error as e:
            raise SystemExit(
                f"cannot open a display window ({e.err.strip() if e.err else e}).\n"
                "--live needs a desktop session. Over SSH or in a headless shell, "
                "drop --live and use --write-video instead.")

    yolo = D.load_yolo()
    dev = D.ultralytics_device(device)
    models, mean, std = D.load_ensemble(args.weights_dirs, args.hidden, device)
    smap = D.build_speed_map(args)

    fps, W, H, _ = D.video_meta(args.video)
    scale = W / 1920.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    out_path = out_dir / f"{args.name}.mp4"
    if args.write_video:
        # mp4v is what OpenCV can always emit; --transcode converts to H.264 after,
        # which is what actually plays in Keynote, PowerPoint and browsers.
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (W, H))
        if not writer.isOpened():
            raise SystemExit(f"cannot open writer for {out_path}")

    print(f"\n[demo] {args.title}")
    if getattr(args, "blurb", None):
        print(f"[demo] {args.blurb}")
    print(f"[demo] {args.video} {W}x{H} @ {fps:.1f}fps  frames "
          f"{args.start_frame}..{args.start_frame + (args.max_frames or 0)}")

    buffers, last_frame, last_prob = {}, {}, {}
    t_det = t_pred = t_blur = t_draw = 0.0
    n_pred = seen = n_held = 0
    last_speed = 0.0
    recent = deque(maxlen=30)
    t_wall = time.perf_counter()

    for fidx, frame in D.frame_reader(args.video, args.start_frame, args.max_frames):
        t_frame = time.perf_counter()
        seen += 1
        # PIE annotates the ego speed on a subset of frames. On an unannotated
        # frame the last known value is held forward, which is what a real vehicle
        # bus would do between CAN messages. The overlay says which it is, so a
        # held value is never mistaken for a reading.
        ego_live = fidx in smap and np.isfinite(smap[fidx])
        ego = float(smap[fidx]) if ego_live else last_speed
        last_speed = ego
        n_held += 0 if ego_live else 1

        img = frame.copy()
        if args.blur_faces:
            # A second detector pass, purely so faces can be published. It is not
            # part of the system being measured, so it is timed separately and
            # kept out of the headline throughput.
            t0 = time.perf_counter()
            D.blur_heads_inplace(img, yolo, dev)
            t_blur += time.perf_counter() - t0

        t0 = time.perf_counter()
        r = yolo.track(frame, classes=[PERSON], conf=args.conf,
                       tracker="bytetrack.yaml", persist=True,
                       device=dev, verbose=False)[0]
        t_det += time.perf_counter() - t0

        ids = r.boxes.id
        if ids is not None:
            tids = ids.int().tolist()
            pend_t, pend_w, pend_b = [], [], []
            for b, tid in zip(r.boxes, tids):
                xyxy = b.xyxy[0].tolist()
                if tid in last_frame and fidx - last_frame[tid] > 1:
                    buffers.pop(tid, None)          # track gap: restart the window
                last_frame[tid] = fidx
                buf = buffers.setdefault(tid, deque(maxlen=OBS_LEN))
                buf.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], ego])
                if len(buf) == OBS_LEN:
                    pend_t.append(tid); pend_w.append(np.asarray(buf)); pend_b.append(xyxy)

            if pend_t:
                t0 = time.perf_counter()
                probs = D.predict_batch(models, np.stack(pend_w), mean, std, device)
                t_pred += time.perf_counter() - t0
                n_pred += len(probs)
                for tid, p in zip(pend_t, probs):
                    last_prob[tid] = float(p)

            t0 = time.perf_counter()
            occupied = []
            order = sorted(zip(r.boxes, tids), key=lambda bt: bt[0].xyxy[0][1].item())
            for b, tid in order:             # top of frame first, so nudges go up
                draw_pedestrian(img, b.xyxy[0].tolist(), last_prob.get(tid),
                                args.threshold, len(buffers.get(tid, ())), scale,
                                occupied, args.label_min_height)
            t_draw += time.perf_counter() - t0

        dt = time.perf_counter() - t_frame
        recent.append(dt)
        # Throughput of the system itself: detect, track, predict. The privacy
        # blur, the overlay and the file write are demo scaffolding and are
        # excluded, otherwise the number measures the presentation, not the model.
        pipe_s = (t_det + t_pred) / max(seen, 1)
        pipe_fps = 1.0 / max(pipe_s, 1e-9)

        draw_header(img, [
            args.title,
            "BiLSTM-F1, 5-seed ensemble  |  threshold %.3f" % args.threshold,
            f"{args.video_id}  frame {fidx}  |  ego speed {ego:.1f} km/h{'' if ego_live else '  (held)'}",
            f"detect + track + predict: {pipe_fps:4.1f} FPS "
            f"({pipe_fps / fps:.2f}x real time)",
            f"detector {1000 * t_det / seen:5.1f} ms/frame  |  "
            f"intention {1000 * t_pred / max(n_pred, 1):.2f} ms/window",
        ], scale)
        draw_legend(img, args.threshold, scale)

        if writer is not None:
            writer.write(img)

        if args.live:
            disp = img if args.display_width >= W else cv2.resize(
                img, (args.display_width, int(H * args.display_width / W)))
            cv2.imshow(WINDOW, disp)
            wait = 1 if args.fast else max(1, int(1000 / fps - dt * 1000))
            k = cv2.waitKey(wait) & 0xFF
            if k == ord("q"):
                print("[demo] stopped by user")
                break
            if k == ord(" "):
                while (cv2.waitKey(30) & 0xFF) != ord(" "):
                    pass

    wall = time.perf_counter() - t_wall
    if writer is not None:
        writer.release()
    if args.live:
        cv2.destroyAllWindows()

    pipe = (t_det + t_pred) / max(seen, 1)
    print(f"\n[demo] {seen} frames, {n_pred} window predictions in {wall:.1f} s wall clock")
    print(f"[demo] --- the system ---")
    print(f"[demo]   detector + tracker  {1000 * t_det / max(seen, 1):7.1f} ms/frame")
    print(f"[demo]   intention ensemble  {1000 * t_pred / max(n_pred, 1):7.2f} ms/window "
          f"({1000 * t_pred / max(n_pred, 1) / 5:.2f} ms per single model)")
    print(f"[demo]   together            {1 / max(pipe, 1e-9):7.1f} FPS  "
          f"= {1 / max(pipe, 1e-9) / fps:.2f}x real time at {fps:.0f} fps")
    pct = 100 * t_det / max(t_det + t_pred, 1e-9)
    print(f"[demo]   the detector is {pct:.0f}% of that, so the pipeline is "
          f"detection-bound, as Section 4.9 reports")
    print(f"[demo] --- demo scaffolding, not part of the system ---")
    if t_blur:
        print(f"[demo]   privacy blur pass   {1000 * t_blur / max(seen, 1):7.1f} ms/frame "
              f"(a second detector pass; use --no-blur-faces to drop it)")
    print(f"[demo]   overlay drawing     {1000 * t_draw / max(seen, 1):7.1f} ms/frame")
    print(f"[demo]   decode + write etc  "
          f"{1000 * (wall - t_det - t_pred - t_blur - t_draw) / max(seen, 1):7.1f} ms/frame")

    if writer is not None:
        print(f"[demo] wrote {out_path}")
        if args.transcode:
            _transcode(out_path)
        meta = dict(scene=args.name, title=args.title, video_id=args.video_id,
                    start_frame=args.start_frame, frames=seen,
                    threshold=args.threshold, hidden=args.hidden,
                    model="BiLSTM-F1 (clean crossing-point protocol), 5-seed ensemble",
                    yolo_conf=args.conf, blur_faces=bool(args.blur_faces),
                    ego_frames_held=n_held,
                    pipeline_fps=round(1 / max(pipe, 1e-9), 2),
                    ms_per_frame_detector=round(1000 * t_det / max(seen, 1), 2),
                    ms_per_window_ensemble=round(1000 * t_pred / max(n_pred, 1), 4),
                    ms_per_frame_privacy_blur=round(1000 * t_blur / max(seen, 1), 2),
                    wall_seconds=round(wall, 1))
        out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def _transcode(path: Path):
    """mp4v -> H.264. OpenCV cannot write H.264, and mp4v will not play in
    PowerPoint, Keynote or a browser, which is where a demo video ends up."""
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        print("[demo] ffmpeg not found; leaving the mp4v file as is")
        return
    tmp = path.with_name(path.stem + "_h264.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
           "-c:v", "libx264", "-preset", "slow", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           "-vf", "scale=1280:-2", str(tmp)]
    subprocess.run(cmd, check=True)
    tmp.replace(path)
    mb = path.stat().st_size / 1e6
    print(f"[demo] transcoded to H.264, 1280 px wide, {mb:.1f} MB -> {path}")


def main():
    ap = argparse.ArgumentParser(
        description="Presentation demo: YOLO26 + ByteTrack + BiLSTM-F1 ensemble",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="scenes: " + ", ".join(SCENES))
    ap.add_argument("--scene", choices=sorted(SCENES),
                    help="named preset (sets video, start frame and length)")
    ap.add_argument("--video", help="override: path to an mp4")
    ap.add_argument("--video-id", help="override: e.g. video_0012")
    ap.add_argument("--set-id", default="set03")
    ap.add_argument("--start-frame", type=int)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--name", help="output basename (defaults to the scene name)")
    ap.add_argument("--title", help="override the on-screen title")

    ap.add_argument("--live", action="store_true", help="show a window while processing")
    ap.add_argument("--fast", action="store_true",
                    help="with --live, run flat out instead of pacing to video speed")
    ap.add_argument("--display-width", type=int, default=1280)
    ap.add_argument("--label-min-height", type=int, default=70,
                    help="hide the text plate (not the box) for pedestrians shorter than this many pixels; 0 labels everything")
    ap.add_argument("--write-video", action="store_true")
    ap.add_argument("--transcode", action="store_true", default=True,
                    help="convert the written file to H.264 (default on)")
    ap.add_argument("--no-transcode", dest="transcode", action="store_false")
    ap.add_argument("--blur-faces", action="store_true", default=True,
                    help="blur every detected head (default on)")
    ap.add_argument("--no-blur-faces", dest="blur_faces", action="store_false")

    ap.add_argument("--weights-dirs", default=D.default_weights_dirs())
    ap.add_argument("--hidden", type=int, default=D.F1_HIDDEN)
    ap.add_argument("--threshold", type=float, default=D.F1_TAU)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--ego-source", choices=["pkl", "obd"], default="pkl")
    ap.add_argument("--annotations-pkl", default="pie_annotations.pkl")
    ap.add_argument("--obd-xml", default=None)
    ap.add_argument("--out-dir", default="pipeline/demo_videos")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = ap.parse_args()

    if args.scene:
        s = SCENES[args.scene]
        args.video_id = args.video_id or s["video_id"]
        args.start_frame = s["start"] if args.start_frame is None else args.start_frame
        args.max_frames = s["frames"] if args.max_frames is None else args.max_frames
        args.name = args.name or args.scene
        args.title = args.title or s["title"]
        args.blurb = s["blurb"]
    if not args.video:
        if not args.video_id:
            raise SystemExit("give --scene, or --video-id (and optionally --video)")
        args.video = str(ROOT / "PIE_clips" / args.set_id / f"{args.video_id}.mp4")
    args.name = args.name or args.video_id
    args.title = args.title or args.video_id
    args.start_frame = args.start_frame or 0
    if not args.live and not args.write_video:
        raise SystemExit("nothing to do: pass --live, --write-video, or both")

    args.weights_dirs = [Path(p) for p in str(args.weights_dirs).split(",")]
    run(args, D.pick_device(args.device))


if __name__ == "__main__":
    main()
