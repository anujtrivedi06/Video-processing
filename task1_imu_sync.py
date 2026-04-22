"""
task1_imu_sync.py
-----------------
Task 1: IMU Data Parsing and Synchronized Visualization

Reads recording2.mp4, recording2.imu, and recording2.vts, then produces
an output video (imu_sync_output.mp4) that shows:
  - Top half:  original camera feed
  - Bottom half: scrolling real-time plots for accel / gyro / magnetometer
  - HUD overlay: frame #, timestamp, current sensor values, rates, temp, sync stats

Usage:
    python task1_imu_sync.py \
        --video   recording2.mp4 \
        --imu     recording2.imu \
        --vts     recording2.vts \
        --output  imu_sync_output.mp4 \
        [--window 2.0]   # seconds of IMU data shown in rolling window
        [--skip 1]       # process every Nth frame (1 = all)
"""

import argparse
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")           # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from collections import deque

from parse_imu import parse_imu_file, parse_vts_file, sync_imu_to_frames


# ── Colour palette (dark HUD theme) ───────────────────────────────────────────
BG_COLOR   = (15,  15,  20)    # near-black BGR
ACCEL_CLR  = "#00d4ff"         # cyan
GYRO_CLR   = "#ff6b35"         # orange
MAG_CLR    = "#a8ff3e"         # lime
AXIS_CLRS  = ["#ff4d6d", "#ffd60a", "#4cc9f0"]   # X=red  Y=yellow  Z=blue


def fig_to_bgr(fig, w, h):
    """Render a matplotlib figure to a BGR numpy array of size (h, w, 3)."""
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    return bgr


def draw_hud(frame, fi, ts_s, imu_row, sync_delay_us,
             imu_rate, cam_fps, mean_d, med_d, max_d):
    """Burn telemetry text onto the top-left corner of frame (in-place)."""
    lines = [
        f"Frame: {fi:05d}",
        f"Time : {ts_s:.4f} s",
        f"──── IMU ────",
        f"Ax:{imu_row.accel_x:+.3f}  Ay:{imu_row.accel_y:+.3f}  Az:{imu_row.accel_z:+.3f}  m/s²",
        f"Gx:{imu_row.gyro_x:+.3f}  Gy:{imu_row.gyro_y:+.3f}  Gz:{imu_row.gyro_z:+.3f}  °/s",
        f"Mx:{imu_row.mag_x:+.1f}  My:{imu_row.mag_y:+.1f}  Mz:{imu_row.mag_z:+.1f}  µT",
        f"Temp: {imu_row.temperature:.1f} °C",
        f"IMU rate: {imu_rate:.1f} Hz   Cam FPS: {cam_fps:.2f}",
        f"──── Sync ───",
        f"Delay: {sync_delay_us:.1f} µs",
        f"Mean:{mean_d:.1f}  Med:{med_d:.1f}  Max:{max_d:.1f}  µs",
    ]

    x0, y0 = 10, 10
    pad = 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.42, 1
    line_h = 17

    # semi-transparent backing rectangle
    h_box = len(lines) * line_h + 2 * pad
    w_box = 430
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - pad, y0 - pad),
                  (x0 + w_box, y0 + h_box), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, txt in enumerate(lines):
        y = y0 + i * line_h + line_h
        color = (220, 220, 220)
        if "Ax" in txt or "Ay" in txt or "Az" in txt:
            color = (255, 200, 100)
        elif "Gx" in txt:
            color = (100, 200, 255)
        elif "Mx" in txt:
            color = (100, 255, 160)
        elif "Temp" in txt:
            color = (180, 140, 255)
        elif "Delay" in txt or "Mean" in txt:
            color = (80, 255, 200)
        cv2.putText(frame, txt, (x0, y), font, scale, color, thick, cv2.LINE_AA)


def build_plot_panel(t_win, ax_win, ay_win, az_win,
                     gx_win, gy_win, gz_win,
                     mx_win, my_win, mz_win,
                     current_t, out_w, out_h):
    """Render the 3-row scrolling IMU plot panel as BGR image."""
    dpi = 100
    fig_w = out_w / dpi
    fig_h = out_h / dpi

    fig, axes = plt.subplots(3, 1, figsize=(fig_w, fig_h), dpi=dpi,
                              facecolor="#0d0d14")
    fig.subplots_adjust(hspace=0.35, left=0.06, right=0.98,
                        top=0.92, bottom=0.08)

    datasets = [
        ("Accelerometer (m/s²)", [ax_win, ay_win, az_win]),
        ("Gyroscope (deg/s)",    [gx_win, gy_win, gz_win]),
        ("Magnetometer (µT)",    [mx_win, my_win, mz_win]),
    ]
    labels = ["X", "Y", "Z"]

    for ax, (title, channels) in zip(axes, datasets):
        ax.set_facecolor("#13131f")
        ax.set_title(title, color="#cccccc", fontsize=7, pad=3, loc="left",
                     fontweight="bold")
        ax.tick_params(colors="#666666", labelsize=6)
        ax.spines[:].set_color("#2a2a3a")
        ax.grid(color="#1e1e2e", linewidth=0.5)

        t_arr = np.array(t_win)
        for ch, lbl, clr in zip(channels, labels, AXIS_CLRS):
            if len(ch) > 1:
                ax.plot(t_arr, np.array(ch), color=clr, linewidth=0.8,
                        label=lbl, alpha=0.9)

        # vertical cursor at current_t
        ax.axvline(current_t, color="#ffffff", linewidth=0.6, alpha=0.5,
                   linestyle="--")
        ax.legend(loc="upper right", fontsize=5, framealpha=0.3,
                  facecolor="#1a1a2a", edgecolor="#333",
                  labelcolor="white")

    bgr = fig_to_bgr(fig, out_w, out_h)
    plt.close(fig)
    return bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",  default="recording2.mp4")
    ap.add_argument("--imu",    default="recording2.imu")
    ap.add_argument("--vts",    default="recording2.vts")
    ap.add_argument("--output", default="imu_sync_output.mp4")
    ap.add_argument("--window", type=float, default=2.0,
                    help="Rolling IMU window in seconds")
    ap.add_argument("--skip",   type=int, default=1,
                    help="Process every Nth frame (1=all)")
    args = ap.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Parsing IMU …")
    imu_df = parse_imu_file(args.imu)
    print("Parsing VTS …")
    vts_df = parse_vts_file(args.vts)
    print("Syncing …")
    sync   = sync_imu_to_frames(imu_df, vts_df)

    # Pre-compute sync delay stats (µs)
    delays_us = np.array([v["sync_delay_ns"] for v in sync.values()]) / 1e3
    mean_d  = float(delays_us.mean())
    med_d   = float(np.median(delays_us))
    max_d   = float(delays_us.max())

    # ── Open video ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open {args.video}")

    cam_fps   = cap.get(cv2.CAP_PROP_FPS)
    vid_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output layout: video on top (full width), plots on bottom (same width)
    # Downscale video to 960×540 to keep file size reasonable
    OUT_W = 960
    OUT_H_VIDEO = int(OUT_W * vid_h / vid_w)   # preserve aspect ratio
    OUT_H_PLOT  = 300                            # plot panel height
    OUT_H_TOTAL = OUT_H_VIDEO + OUT_H_PLOT

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, cam_fps / args.skip,
                             (OUT_W, OUT_H_TOTAL))

    print(f"\nVideo: {vid_w}×{vid_h} @ {cam_fps:.2f} fps  →  "
          f"output {OUT_W}×{OUT_H_TOTAL}  ({n_frames} frames)\n")

    # Rolling deques for plot
    WIN_SAMPLES = int(args.window * 600)   # ~600 Hz max
    t_win   = deque(maxlen=WIN_SAMPLES)
    ax_win  = deque(maxlen=WIN_SAMPLES)
    ay_win  = deque(maxlen=WIN_SAMPLES)
    az_win  = deque(maxlen=WIN_SAMPLES)
    gx_win  = deque(maxlen=WIN_SAMPLES)
    gy_win  = deque(maxlen=WIN_SAMPLES)
    gz_win  = deque(maxlen=WIN_SAMPLES)
    mx_win  = deque(maxlen=WIN_SAMPLES)
    my_win  = deque(maxlen=WIN_SAMPLES)
    mz_win  = deque(maxlen=WIN_SAMPLES)

    # ── Build VTS frame→timestamp lookup ──────────────────────────────────────
    vts_lookup = dict(zip(vts_df["frame_index"].values,
                          vts_df["timestamp_s"].values))

    frame_idx = 0
    frames_written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.skip != 0:
            frame_idx += 1
            continue

        # ── Get VTS timestamp for this frame ───────────────────────────────
        ts_s = vts_lookup.get(frame_idx, None)
        if ts_s is None:
            # frame not in VTS – use linear interpolation from FPS
            ts_s = (imu_df["timestamp_s"].iloc[0] +
                    frame_idx / cam_fps)

        # ── Get nearest IMU sample ─────────────────────────────────────────
        sync_info  = sync.get(frame_idx, None)
        if sync_info:
            imu_idx    = sync_info["nearest_imu_idx"]
            delay_us   = sync_info["sync_delay_ns"] / 1e3
        else:
            imu_idx  = 0
            delay_us = 0.0

        imu_row = imu_df.iloc[imu_idx]

        # ── Populate rolling window with ALL IMU samples up to current ts ──
        # find range of IMU indices in the rolling window
        win_start_ts = ts_s - args.window
        i_start = max(0, int(np.searchsorted(
            imu_df["timestamp_s"].values, win_start_ts)) - 1)
        i_end   = min(len(imu_df), imu_idx + 1)

        t_win.clear(); ax_win.clear(); ay_win.clear(); az_win.clear()
        gx_win.clear(); gy_win.clear(); gz_win.clear()
        mx_win.clear(); my_win.clear(); mz_win.clear()

        chunk = imu_df.iloc[i_start:i_end]
        for _, r in chunk.iterrows():
            t_win.append(r.timestamp_s)
            ax_win.append(r.accel_x); ay_win.append(r.accel_y); az_win.append(r.accel_z)
            gx_win.append(r.gyro_x);  gy_win.append(r.gyro_y);  gz_win.append(r.gyro_z)
            mx_win.append(r.mag_x);   my_win.append(r.mag_y);   mz_win.append(r.mag_z)

        # ── Render video frame ─────────────────────────────────────────────
        vid_small = cv2.resize(frame, (OUT_W, OUT_H_VIDEO),
                               interpolation=cv2.INTER_AREA)

        # Compute actual IMU rate from nearby samples
        if imu_idx > 10:
            dt_arr = np.diff(imu_df["timestamp_ns"].values[imu_idx-10:imu_idx])
            imu_rate = 1e9 / dt_arr.mean() if dt_arr.mean() > 0 else 0
        else:
            imu_rate = 568.0

        draw_hud(vid_small, frame_idx, ts_s, imu_row, delay_us,
                 imu_rate, cam_fps, mean_d, med_d, max_d)

        # ── Render plot panel ──────────────────────────────────────────────
        plot_panel = build_plot_panel(
            t_win, ax_win, ay_win, az_win,
            gx_win, gy_win, gz_win,
            mx_win, my_win, mz_win,
            ts_s, OUT_W, OUT_H_PLOT
        )

        # ── Stack and write ────────────────────────────────────────────────
        combined = np.vstack([vid_small, plot_panel])
        writer.write(combined)

        frames_written += 1
        if frames_written % 30 == 0:
            pct = 100 * frame_idx / max(n_frames, 1)
            print(f"  [{pct:5.1f}%] frame {frame_idx}/{n_frames}", end="\r")

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"\nDone. Wrote {frames_written} frames → {args.output}")


if __name__ == "__main__":
    main()
