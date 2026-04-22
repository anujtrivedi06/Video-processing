"""
task3_segmentation.py
---------------------
Task 3: Object Detection and Scene Segmentation

Uses YOLOv8-seg (instance segmentation) to detect and label objects in each
video frame.  Optionally runs MediaPipe Hands for hand detection / landmark
overlay (bonus requirement).

Output video shows:
  - Coloured segmentation masks blended onto the frame
  - Bounding boxes with class labels and confidence scores
  - Hand skeleton landmarks (if --hands flag is set and MediaPipe is installed)
  - Per-frame stats HUD (detections count, FPS)

Usage:
    pip install ultralytics mediapipe

    python task3_segmentation.py \
        --video   recording2.mp4 \
        --output  segmentation_output.mp4 \
        [--model  yolov8n-seg]   # nano=fast, s/m/l/x=better quality
        [--conf   0.35]          # confidence threshold
        [--iou    0.45]          # NMS IoU threshold
        [--skip   1]             # every Nth frame
        [--hands]                # enable MediaPipe hand detection (bonus)
        [--half_res]             # process at half resolution
"""

import argparse
import sys
import random
import os
import urllib.request
import numpy as np
import cv2


# ── Colour palette for classes (consistent across frames) ─────────────────────
random.seed(42)
CLASS_COLORS = {}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

TASKS_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def get_class_color(cls_id: int):
    if cls_id not in CLASS_COLORS:
        hue = (cls_id * 47) % 180          # spread hues evenly
        hsv = np.uint8([[[hue, 220, 230]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
        CLASS_COLORS[cls_id] = tuple(int(x) for x in bgr)
    return CLASS_COLORS[cls_id]


def blend_mask(frame, mask, color, alpha=0.45):
    """Blend a boolean mask onto frame with the given BGR color."""
    overlay = frame.copy()
    overlay[mask] = color
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_box_label(frame, x1, y1, x2, y2, label, conf, color):
    """Draw a bounding box and label badge."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    badge_y1 = max(y1 - th - 6, 0)
    cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_stats_hud(frame, n_dets, fps_proc, frame_idx):
    """Burn a small stats overlay onto the top-right corner."""
    lines = [
        f"Frame: {frame_idx:05d}",
        f"Dets : {n_dets}",
        f"Speed: {fps_proc:.1f} fps",
    ]
    fh, fw = frame.shape[:2]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    line_h = 18
    x0 = fw - 160
    y0 = 10
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (x0, y0 + (i + 1) * line_h),
                    font, scale, (200, 240, 200), thick, cv2.LINE_AA)


def load_yolo(model_name: str):
    from ultralytics import YOLO
    print(f"Loading YOLO model: {model_name} …")
    model = YOLO(f"{model_name}.pt")
    print(f"  classes: {len(model.names)}")
    return model


def load_mediapipe_hands():
    try:
        import mediapipe as mp
        print("Loading MediaPipe Hands …")
        if hasattr(mp, "solutions"):
            hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=4,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            drawing = mp.solutions.drawing_utils
            styles = mp.solutions.drawing_styles
            return {
                "backend": "solutions",
                "model": hands,
                "drawing": drawing,
                "styles": styles,
            }

        # Newer MediaPipe builds may only expose the Tasks API.
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.hand_landmarker import (
            HandLandmarker,
            HandLandmarkerOptions,
        )
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )

        model_dir = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "hand_landmarker.task")
        if not os.path.exists(model_path):
            print("Downloading MediaPipe hand_landmarker.task …")
            urllib.request.urlretrieve(TASKS_HAND_MODEL_URL, model_path)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_hands=4,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        hand_landmarker = HandLandmarker.create_from_options(options)
        return {
            "backend": "tasks",
            "model": hand_landmarker,
            "drawing": None,
            "styles": None,
        }
    except ImportError:
        print("[warn] mediapipe not installed. Skipping hand detection.")
        return None


def run_hand_detection(frame_rgb, hands_bundle):
    """
    Run MediaPipe Hands on an RGB frame.
    Draws landmarks directly onto a copy and returns it.
    Returns (annotated_frame_bgr, n_hands_detected)
    """
    import mediapipe as mp
    out_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if hands_bundle["backend"] == "solutions":
        hands_model = hands_bundle["model"]
        drawing = hands_bundle["drawing"]
        styles = hands_bundle["styles"]
        results = hands_model.process(frame_rgb)

        n_hands = 0
        if results.multi_hand_landmarks:
            n_hands = len(results.multi_hand_landmarks)
            for hand_lms in results.multi_hand_landmarks:
                drawing.draw_landmarks(
                    out_bgr,
                    hand_lms,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    styles.get_default_hand_landmarks_style(),
                    styles.get_default_hand_connections_style(),
                )
        return out_bgr, n_hands

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = hands_bundle["model"].detect(mp_image)
    hand_landmarks = result.hand_landmarks if result and result.hand_landmarks else []

    for lms in hand_landmarks:
        points = []
        for lm in lms:
            x = int(lm.x * out_bgr.shape[1])
            y = int(lm.y * out_bgr.shape[0])
            points.append((x, y))

        for a, b in HAND_CONNECTIONS:
            if a < len(points) and b < len(points):
                cv2.line(out_bgr, points[a], points[b], (255, 215, 0), 1, cv2.LINE_AA)
        for p in points:
            cv2.circle(out_bgr, p, 2, (0, 255, 255), -1, cv2.LINE_AA)

    return out_bgr, len(hand_landmarks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",    default="recording2.mp4")
    ap.add_argument("--output",   default="segmentation_output.mp4")
    ap.add_argument("--model",    default="yolov8n-seg")
    ap.add_argument("--conf",     type=float, default=0.35)
    ap.add_argument("--iou",      type=float, default=0.45)
    ap.add_argument("--skip",     type=int,   default=1)
    ap.add_argument("--hands",    action="store_true")
    ap.add_argument("--half_res", action="store_true")
    args = ap.parse_args()

    # ── Load models ────────────────────────────────────────────────────────────
    yolo = load_yolo(args.model)
    if args.hands:
        hands_bundle = load_mediapipe_hands()
    else:
        hands_bundle = None

    # ── Open video ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open {args.video}")

    fps      = cap.get(cv2.CAP_PROP_FPS)
    vid_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    proc_w = vid_w // 2 if args.half_res else vid_w
    proc_h = vid_h // 2 if args.half_res else vid_h
    out_fps = fps / args.skip

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, out_fps, (proc_w, proc_h))

    print(f"\nVideo: {vid_w}×{vid_h} @ {fps:.2f} fps, {n_frames} frames")
    print(f"Processing: {proc_w}×{proc_h}, every {args.skip} frame(s)")
    print(f"YOLO conf={args.conf}  iou={args.iou}")
    print(f"Hand detection: {'ON' if hands_bundle else 'OFF'}\n")

    import time
    frame_idx     = 0
    frames_written = 0
    fps_proc       = 0.0
    t_prev         = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.skip != 0:
            frame_idx += 1
            continue

        if args.half_res:
            frame = cv2.resize(frame, (proc_w, proc_h),
                               interpolation=cv2.INTER_AREA)

        t0 = time.time()

        # ── YOLO inference ────────────────────────────────────────────────
        results = yolo(frame,
                       conf=args.conf,
                       iou=args.iou,
                       verbose=False)[0]

        annotated = frame.copy()
        n_dets = 0

        if results.boxes is not None and len(results.boxes):
            boxes   = results.boxes.xyxy.cpu().numpy().astype(int)
            confs   = results.boxes.conf.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)
            masks   = results.masks.data.cpu().numpy() if results.masks else None

            n_dets = len(boxes)

            for i in range(n_dets):
                cls_id = cls_ids[i]
                color  = get_class_color(cls_id)
                label  = yolo.names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = boxes[i]

                # Draw segmentation mask
                if masks is not None and i < len(masks):
                    raw_mask = masks[i]
                    # resize mask to frame size
                    m = cv2.resize(raw_mask, (proc_w, proc_h),
                                   interpolation=cv2.INTER_NEAREST)
                    blend_mask(annotated, m > 0.5, color)

                # Draw bounding box + label
                draw_box_label(annotated, x1, y1, x2, y2, label, confs[i], color)

        # ── Hand detection (bonus) ────────────────────────────────────────
        if hands_bundle:
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            annotated, n_hands = run_hand_detection(rgb, hands_bundle)
            if n_hands > 0:
                cv2.putText(annotated, f"Hands: {n_hands}",
                            (10, proc_h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 200), 1, cv2.LINE_AA)

        # ── HUD ───────────────────────────────────────────────────────────
        fps_proc = 1.0 / max(time.time() - t0, 1e-6)
        draw_stats_hud(annotated, n_dets, fps_proc, frame_idx)

        writer.write(annotated)
        frames_written += 1

        if frames_written % 30 == 0:
            pct = 100 * frame_idx / max(n_frames, 1)
            print(f"  [{pct:5.1f}%] frame {frame_idx}/{n_frames}  "
                  f"{fps_proc:.1f} fps", end="\r")

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"\nDone. Wrote {frames_written} frames → {args.output}")


if __name__ == "__main__":
    main()
