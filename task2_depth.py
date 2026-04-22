"""
task2_depth.py
--------------
Task 2: Monocular Dense Depth Estimation

Uses Depth Anything V2 (via HuggingFace transformers) to estimate per-frame
depth maps from the video, and produces a side-by-side output video:
    left  – original RGB frame
    right – colorized depth map (inferno colormap)

Notes on fisheye / barrel distortion:
    The camera uses a wide-angle fisheye lens. True undistortion requires
    the camera intrinsic matrix and distortion coefficients (not provided).
    We note the distortion but do NOT apply correction, as guessing the
    coefficients would produce worse results than leaving the image as-is.
    Depth Anything V2 is trained on diverse in-the-wild images and handles
    mild-to-moderate barrel distortion gracefully. Edge regions will have
    slightly reduced depth accuracy.

Usage:
    python task2_depth.py \
        --video   recording2.mp4 \
        --output  depth_output.mp4 \
        [--model  depth-anything/Depth-Anything-V2-Small-hf] \
        [--skip   2]      # process every Nth frame (speeds things up on CPU)
        [--half_res]      # process at half resolution (saves memory/time)
"""

import argparse
import sys
import numpy as np
import cv2

# ── Model loading (lazy import so the script is importable without torch) ──────
def load_model(model_name: str):
    try:
        from transformers import pipeline as hf_pipeline
        print(f"Loading depth model: {model_name} …")
        pipe = hf_pipeline(
            task="depth-estimation",
            model=model_name,
            device=0 if _cuda_available() else -1,
        )
        print(f"  device: {'GPU (CUDA)' if _cuda_available() else 'CPU'}")
        return pipe, "hf"
    except Exception as e:
        print(f"[warn] HuggingFace pipeline failed ({e}), falling back to MiDaS …")
        return load_midas()


def load_midas():
    import torch
    print("Loading MiDaS DPT-Large via torch.hub …")
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True)
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform
    device = torch.device("cuda" if _cuda_available() else "cpu")
    model.to(device).eval()
    print(f"  device: {device}")
    return (model, transform, device), "midas"


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def estimate_depth_hf(pipe, frame_bgr):
    """Run HF depth pipeline on a BGR frame, return float32 depth array."""
    from PIL import Image
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    result = pipe(pil_img)
    depth = np.array(result["depth"], dtype=np.float32)
    return depth


def estimate_depth_midas(model_bundle, frame_bgr):
    """Run MiDaS on a BGR frame, return float32 depth array."""
    import torch
    model, transform, device = model_bundle
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(rgb).to(device)
    with torch.no_grad():
        pred = model(input_batch)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=frame_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return pred.cpu().numpy().astype(np.float32)


def colorize_depth(depth: np.ndarray, colormap=cv2.COLORMAP_INFERNO) -> np.ndarray:
    """
    Normalize depth to 0-255 and apply a perceptual colormap.
    Returns a BGR uint8 image.
    """
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-6:
        norm = np.zeros_like(depth, dtype=np.uint8)
    else:
        norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, colormap)
    return colored


def add_depth_label(img, text, pos=(10, 25)):
    """Burn a small label onto an image corner."""
    cv2.putText(img, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def draw_colorbar(panel_h, panel_w=40):
    """Create a vertical inferno colorbar strip."""
    bar = np.linspace(255, 0, panel_h, dtype=np.uint8).reshape(-1, 1)
    bar = cv2.applyColorMap(bar, cv2.COLORMAP_INFERNO)
    bar = cv2.resize(bar, (panel_w, panel_h))
    # add near/far labels
    cv2.putText(bar, "N", (5, 15),  cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
    cv2.putText(bar, "F", (5, panel_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
    return bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",    default="recording2.mp4")
    ap.add_argument("--output",   default="depth_output.mp4")
    ap.add_argument("--model",    default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--skip",     type=int, default=1,
                    help="Process every Nth frame (1=all). Use 2 or 3 on CPU.")
    ap.add_argument("--half_res", action="store_true",
                    help="Halve input resolution before depth inference (faster).")
    ap.add_argument("--colormap", default="inferno",
                    choices=["inferno", "magma", "turbo", "plasma"])
    args = ap.parse_args()

    CMAP = {
        "inferno": cv2.COLORMAP_INFERNO,
        "magma":   cv2.COLORMAP_MAGMA,
        "turbo":   cv2.COLORMAP_TURBO,
        "plasma":  cv2.COLORMAP_PLASMA,
    }[args.colormap]

    # ── Load model ─────────────────────────────────────────────────────────────
    model_bundle, backend = load_model(args.model)

    # ── Open video ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open {args.video}")

    fps      = cap.get(cv2.CAP_PROP_FPS)
    vid_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process at half resolution if requested (or if original is very large)
    proc_w = vid_w // 2 if args.half_res else vid_w
    proc_h = vid_h // 2 if args.half_res else vid_h

    # Output is side-by-side: [original | depth]  + narrow colorbar
    cb_w    = 40
    out_w   = proc_w * 2 + cb_w
    out_h   = proc_h
    out_fps = fps / args.skip

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, out_fps, (out_w, out_h))

    print(f"\nVideo: {vid_w}×{vid_h} @ {fps:.2f} fps, {n_frames} frames")
    print(f"Processing: {proc_w}×{proc_h}, every {args.skip} frame(s)")
    print(f"Output: {out_w}×{out_h}\n")

    colorbar = draw_colorbar(proc_h, cb_w)
    frame_idx = 0
    frames_written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.skip != 0:
            frame_idx += 1
            continue

        # Resize to processing resolution
        if args.half_res or (proc_w != vid_w):
            proc_frame = cv2.resize(frame, (proc_w, proc_h),
                                    interpolation=cv2.INTER_AREA)
        else:
            proc_frame = frame

        # ── Depth estimation ───────────────────────────────────────────────
        if backend == "hf":
            depth = estimate_depth_hf(model_bundle, proc_frame)
        else:
            depth = estimate_depth_midas(model_bundle, proc_frame)

        # Resize depth map to match processing frame if needed
        if depth.shape[:2] != (proc_h, proc_w):
            depth = cv2.resize(depth, (proc_w, proc_h),
                               interpolation=cv2.INTER_LINEAR)

        depth_colored = colorize_depth(depth, CMAP)

        # ── Labels ────────────────────────────────────────────────────────
        orig_labeled = proc_frame.copy()
        add_depth_label(orig_labeled, f"RGB  Frame {frame_idx:05d}")

        depth_labeled = depth_colored.copy()
        add_depth_label(depth_labeled,
                        f"Depth ({args.colormap})  "
                        f"[{depth.min():.1f} – {depth.max():.1f}]")

        # ── Compose side-by-side + colorbar ───────────────────────────────
        combined = np.hstack([orig_labeled, depth_labeled, colorbar])
        writer.write(combined)

        frames_written += 1
        if frames_written % 10 == 0:
            pct = 100 * frame_idx / max(n_frames, 1)
            print(f"  [{pct:5.1f}%] frame {frame_idx}/{n_frames}", end="\r")

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"\nDone. Wrote {frames_written} frames → {args.output}")


if __name__ == "__main__":
    main()
