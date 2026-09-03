"""09_inference_latency.py — Issue 9: isolated inference-latency benchmark.

WHY. The only timing number in the project is "~900 frames in ~50 s on MPS" for the
*full* pipeline, dominated by YOLO26-M. The BiLSTM's own latency (the part this thesis
contributes) was never measured in isolation, so "real-time / ~18 fps" conflates three
very different components. This measures each piece on the actual M4 hardware.

NO TRAINING. Loads the existing `runs/bilstm_baseline/best.pt` and `yolo26m.pt` and
times forward passes only. ~3–5 min total.

What it measures (all on this M4):
  1. Isolated BiLSTM latency — CPU and MPS × batch {1, 8, 32} (1 window vs many
     parallel tracks). Reports ms/forward, ms/window, windows/s.
  2. obs_len {8, 16, 30} latency at batch=1 — tests "shorter window = lower latency".
  3. Pipeline breakdown — YOLO26-M ms/frame (measured on real PIE frames), ByteTrack
     (estimated), BiLSTM ms/window (measured) → where the time actually goes.

Correctness: MPS is asynchronous, so every timed region calls torch.mps.synchronize()
before stopping the clock — otherwise we'd time kernel *dispatch*, not compute. Warmup
passes absorb first-call compilation. time.perf_counter throughout.

Outputs: 09_latency_results.json, 09_latency_report.md, 09_latency_figure.png
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CKPT = ROOT / "paper_and_artifacts" / "runs" / "bilstm_baseline" / "best.pt"
YOLO_WEIGHTS = ROOT / "yolo26m.pt"
CLIP = ROOT / "PIE_clips" / "set03" / "video_0012.mp4"

_spec = importlib.util.spec_from_file_location("m03", ROOT / "src" / "model_bilstm_legacy.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
BiLSTM = _m.BiLSTMIntentPredictor

WARMUP, TIMED = 50, 1000
BATCHES = [1, 8, 32]
OBS_LENS = [8, 16, 30]


def devices():
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def load_model(device):
    model = BiLSTM().to(device)                       # baseline arch: hidden 128, 2 layers
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def time_forward(model, x, device, warmup=WARMUP, timed=TIMED):
    """Per-call latency (ms): warm up, then time `timed` forwards individually with
    an MPS sync inside each so we measure compute, not async dispatch."""
    for _ in range(warmup):
        model(x)
    sync(device)
    ts = np.empty(timed)
    for i in range(timed):
        t0 = time.perf_counter()
        model(x)
        sync(device)
        ts[i] = (time.perf_counter() - t0) * 1e3      # ms
    return ts


def bench_bilstm():
    out = {}
    for dev in devices():
        model = load_model(dev)
        out[dev] = {}
        # batch sweep at obs_len=16
        out[dev]["batch"] = {}
        for b in BATCHES:
            x = torch.randn(b, 16, 5, device=dev)
            ts = time_forward(model, x, dev)
            out[dev]["batch"][b] = dict(
                ms_forward=float(ts.mean()), ms_std=float(ts.std()),
                ms_p50=float(np.percentile(ts, 50)), ms_p99=float(np.percentile(ts, 99)),
                ms_per_window=float(ts.mean() / b),
                windows_per_s=float(b / (ts.mean() / 1e3)))
            print(f"  [bilstm/{dev}] batch={b:2d}: {ts.mean():.3f} ms/forward "
                  f"({ts.mean()/b:.4f} ms/window, {b/(ts.mean()/1e3):,.0f} win/s)")
        # obs_len sweep at batch=1
        out[dev]["obs_len"] = {}
        for ol in OBS_LENS:
            x = torch.randn(1, ol, 5, device=dev)
            ts = time_forward(model, x, dev)
            out[dev]["obs_len"][ol] = dict(ms_forward=float(ts.mean()), ms_std=float(ts.std()))
            print(f"  [bilstm/{dev}] obs_len={ol:2d} (b=1): {ts.mean():.3f} ± {ts.std():.3f} ms")
    return out


def bench_yolo():
    """Measure YOLO26-M ms/frame on real PIE frames (MPS primary, CPU sample)."""
    try:
        import cv2
        from ultralytics import YOLO
    except Exception as e:
        print(f"  [yolo] skipped ({e})")
        return None
    if not (YOLO_WEIGHTS.exists() and CLIP.exists()):
        print(f"  [yolo] skipped (weights or clip missing)")
        return None

    cap = cv2.VideoCapture(str(CLIP))
    frames = []
    while len(frames) < 120:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        print("  [yolo] no frames decoded"); return None
    h, w = frames[0].shape[:2]

    out = {"frame_hw": [h, w], "imgsz": 640}
    for dev, n in [("mps", 100), ("cpu", 25)]:
        if dev == "mps" and not torch.backends.mps.is_available():
            continue
        model = YOLO(str(YOLO_WEIGHTS))
        # warmup
        for fr in frames[:5]:
            model.predict(fr, device=dev, imgsz=640, classes=[0], verbose=False)
        ts = []
        for fr in frames[:n]:
            t0 = time.perf_counter()
            model.predict(fr, device=dev, imgsz=640, classes=[0], verbose=False)
            ts.append((time.perf_counter() - t0) * 1e3)
        ts = np.array(ts)
        out[dev] = dict(ms_frame=float(ts.mean()), ms_std=float(ts.std()),
                        fps=float(1e3 / ts.mean()), n=int(n))
        print(f"  [yolo/{dev}] {ts.mean():.1f} ± {ts.std():.1f} ms/frame "
              f"({1e3/ts.mean():.1f} fps) over {n} frames")
    return out


def write_outputs(bilstm, yolo):
    json.dump({"bilstm": bilstm, "yolo": yolo}, open(HERE / "09_latency_results.json", "w"), indent=2)

    has_mps = "mps" in bilstm
    BUDGET = 33.3                                       # ms per frame at 30 fps
    # derived headline numbers
    single = {d: bilstm[d]["batch"][1]["ms_per_window"] for d in bilstm}
    best_dev = min(single, key=single.get); best_single = single[best_dev]
    batched = {d: bilstm[d]["batch"][32]["ms_per_window"] for d in bilstm}
    bbat_dev = min(batched, key=batched.get); best_batched = batched[bbat_dev]
    cpu1 = bilstm["cpu"]["batch"][1]["ms_per_window"]
    mps1 = bilstm["mps"]["batch"][1]["ms_per_window"] if has_mps else None

    L = ["# Issue 9 — Isolated inference-latency benchmark (Apple M4)", "",
         "All numbers measured on this MacBook Air **M4**; inference only (no "
         "training). MPS timings synchronise inside each timed call "
         f"(torch.mps.synchronize), {WARMUP} warmup + {TIMED} timed forwards per cell. "
         "BiLSTM = the locked baseline (hidden 128, 2 layers, 0.6 M params, input "
         "16×5); latency is weight-independent, but we load the real "
         "`bilstm_baseline/best.pt` for fidelity.", "",
         "## 1. Isolated BiLSTM latency (the previously-unreported number)", "",
         "| device | batch | ms / forward | ms / **window** | windows / s | p99 ms |",
         "|---|---|---|---|---|---|"]
    for dev in bilstm:
        for b in BATCHES:
            c = bilstm[dev]["batch"][b]
            L.append(f"| {dev.upper()} | {b} | {c['ms_forward']:.3f} | "
                     f"**{c['ms_per_window']:.4f}** | {c['windows_per_s']:,.0f} | "
                     f"{c['ms_p99']:.3f} |")
    insight = ""
    if has_mps and mps1 > cpu1:
        insight = (f" A notable detail: **CPU beats MPS at batch 1** "
                   f"({cpu1:.3f} vs {mps1:.3f} ms) — GPU kernel-dispatch overhead "
                   f"dominates a model this small, so the GPU only pays off once many "
                   f"parallel tracks are batched. At batch 32 MPS reaches "
                   f"{bilstm['mps']['batch'][32]['ms_per_window']:.4f} ms/window vs CPU "
                   f"{bilstm['cpu']['batch'][32]['ms_per_window']:.4f}. For single-track "
                   f"real-time use, CPU is the better backend for this network.")
    L += ["",
          f"**The BiLSTM is effectively free.** Fastest single-window latency is "
          f"**{best_single:.3f} ms** ({best_dev.upper()}, batch 1) = "
          f"{1e3/best_single:,.0f} windows/s — about **{BUDGET/best_single:.0f}× "
          f"inside** a 30 fps frame budget ({BUDGET:.1f} ms). Batching 32 parallel "
          f"tracks amortises to {best_batched:.4f} ms/window ({bbat_dev.upper()}).{insight} "
          f"The 0.6 M-param model is not a latency concern on any backend.", "",
          "## 2. Observation-window length vs latency (batch=1)", "",
          "| obs_len | " + " | ".join(f"{d.upper()} ms" for d in bilstm) + " |",
          "|---|" + "---|" * len(bilstm)]
    for ol in OBS_LENS:
        row = " | ".join(f"{bilstm[d]['obs_len'][ol]['ms_forward']:.3f}" for d in bilstm)
        L.append(f"| {ol} | {row} |")
    pd = "cpu"                                           # cleanest signal (no GPU jitter)
    o = bilstm[pd]["obs_len"]
    monotonic = o[8]["ms_forward"] <= o[16]["ms_forward"] <= o[30]["ms_forward"]
    L += ["",
          f"\"Shorter window = lower latency\" is "
          f"**{'confirmed' if monotonic else 'not monotonic'}**: on {pd.upper()}, "
          f"obs_len 8/16/30 → {o[8]['ms_forward']:.3f} / {o[16]['ms_forward']:.3f} / "
          f"{o[30]['ms_forward']:.3f} ms (the LSTM unrolls over the sequence, so cost "
          f"scales with length). The effect is real but ≤1 ms in absolute terms — "
          f"immaterial next to detection (below)."]

    # ---- pipeline breakdown ----
    L += ["", "## 3. Pipeline breakdown — where the time actually goes", ""]
    if yolo and ("mps" in yolo or "cpu" in yolo):
        ydev = "mps" if "mps" in yolo else "cpu"
        y = yolo[ydev]
        bw = mps1 if has_mps else cpu1                   # single-device (MPS) pipeline
        bt_est = 1.0                                     # ByteTrack assoc.: lightweight, ~1 ms/frame
        total = y["ms_frame"] + bt_est + bw
        L += [f"Measured on real PIE frames ({yolo['frame_hw'][1]}×{yolo['frame_hw'][0]}, "
              f"imgsz {yolo['imgsz']}, person class). BiLSTM shown at its on-MPS "
              f"single-track cost (conservative; on CPU it is {cpu1:.3f} ms):", "",
              "| stage | ms / frame | share |", "|---|---|---|",
              f"| YOLO26-M detect ({ydev.upper()}) | {y['ms_frame']:.1f} | "
              f"{100*y['ms_frame']/total:.1f}% |",
              f"| ByteTrack assoc. (est.) | ~{bt_est:.1f} | {100*bt_est/total:.1f}% |",
              f"| **BiLSTM intent** (per track) | **{bw:.3f}** | "
              f"**{100*bw/total:.2f}%** |",
              f"| **total / frame** | **{total:.1f}** | → **{1e3/total:.1f} fps** |", "",
              f"**The pipeline is detection-bound, not prediction-bound.** YOLO26-M is "
              f"~{y['ms_frame']/max(bw,1e-6):.0f}× the BiLSTM; the intent model is only "
              f"**{100*bw/total:.1f}%** of per-frame cost. Honest framing: the BiLSTM "
              f"adds negligible latency on top of whatever detector is used, and the "
              f"headline ~{1e3/total:.0f} fps is set by YOLO26-M — a lighter detector "
              f"would raise it without touching this thesis's contribution."]
        if "cpu" in yolo:
            L.append(f"\n(YOLO26-M on CPU: {yolo['cpu']['ms_frame']:.0f} ms/frame "
                     f"= {yolo['cpu']['fps']:.1f} fps — MPS is the deployment target.)")
    else:
        L.append("_YOLO not measured (weights/clip unavailable); see the demo's "
                 "~55 ms/frame full-pipeline figure. BiLSTM numbers above stand alone._")

    L += ["", "## Verdict", "",
          f"The previously-missing number: **isolated BiLSTM latency = "
          f"{best_single:.3f} ms/window** ({best_dev.upper()}, batch 1"
          + (f"; {mps1:.3f} ms on MPS — slower for a model this small because of GPU "
             f"dispatch overhead" if has_mps else "") + f"). That is "
          f"**~{BUDGET/best_single:.0f}× inside** a 30 fps frame budget (33.3 ms) and "
          + (f"a negligible part of the full perception→prediction pipeline, which is "
             f"detection-bound (BiLSTM ≈ "
             f"{100*(mps1 if has_mps else cpu1)/(yolo[('mps' if 'mps' in yolo else 'cpu')]['ms_frame']+1.0+(mps1 if has_mps else cpu1)):.0f}% "
             f"of per-frame cost)" if yolo and ("mps" in yolo or "cpu" in yolo)
             else "a negligible cost") + f". Real-time operation is justified for the "
          f"prediction model itself; the pipeline fps is a property of the chosen "
          f"detector (YOLO26-M), not the BiLSTM."]
    (HERE / "09_latency_report.md").write_text("\n".join(L))

    make_figure(bilstm, yolo)
    print("\nwrote 09_latency_results.json, 09_latency_report.md, 09_latency_figure.png")


def make_figure(bilstm, yolo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")

    # left: BiLSTM ms/window by batch, per device (log y)
    ax = axes[0]
    for dev, color in [("cpu", "#6b7280"), ("mps", "#0891b2")]:
        if dev not in bilstm:
            continue
        ys = [bilstm[dev]["batch"][b]["ms_per_window"] for b in BATCHES]
        ax.plot(BATCHES, ys, marker="o", ms=8, lw=2, color=color, label=dev.upper())
        for b, y in zip(BATCHES, ys):
            ax.annotate(f"{y:.3f}", (b, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=color)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(BATCHES); ax.set_xticklabels(BATCHES)
    ax.set_xlabel("batch (parallel tracks)"); ax.set_ylabel("ms / window (log)")
    ax.set_title("Isolated BiLSTM latency"); ax.grid(alpha=0.3, which="both")
    ax.legend()

    # right: pipeline breakdown (stacked bar) if YOLO measured
    ax2 = axes[1]
    if yolo and ("mps" in yolo or "cpu" in yolo):
        ydev = "mps" if "mps" in yolo else "cpu"
        bw = bilstm.get("mps", bilstm["cpu"])["batch"][1]["ms_per_window"]
        parts = [("YOLO26-M", yolo[ydev]["ms_frame"], "#dc2626"),
                 ("ByteTrack~", 1.0, "#f59e0b"),
                 ("BiLSTM", bw, "#0891b2")]
        bottom = 0
        for name, val, c in parts:
            ax2.bar(0, val, bottom=bottom, color=c, width=0.5,
                    label=f"{name}: {val:.2f} ms")
            bottom += val
        ax2.set_xticks([]); ax2.set_ylabel("ms / frame")
        ax2.set_title(f"Pipeline breakdown ({ydev.upper()}) → {1e3/bottom:.0f} fps")
        ax2.legend(loc="upper right")
        ax2.set_xlim(-0.6, 1.2)
    else:
        ax2.text(0.5, 0.5, "YOLO not measured", ha="center", va="center")
        ax2.axis("off")
    fig.suptitle("Issue 9 — Inference latency (Apple M4): BiLSTM is not the bottleneck",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(HERE / "09_latency_figure.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    import sys
    if "--report-only" in sys.argv:                      # regenerate report/figure from saved JSON
        d = json.load(open(HERE / "09_latency_results.json"))
        bilstm = {dev: {**v, "batch": {int(k): vv for k, vv in v["batch"].items()},
                        "obs_len": {int(k): vv for k, vv in v["obs_len"].items()}}
                  for dev, v in d["bilstm"].items()}
        write_outputs(bilstm, d["yolo"])
        return
    print(f"M4 latency benchmark | devices {devices()} | "
          f"{WARMUP} warmup + {TIMED} timed\n")
    print("=== BiLSTM ===")
    bilstm = bench_bilstm()
    print("\n=== YOLO26-M + pipeline ===")
    yolo = bench_yolo()
    write_outputs(bilstm, yolo)


if __name__ == "__main__":
    main()
