# Trekion.ai Robotics Data Engineering Intern — Technical Assessment

## Overview

This repository contains the solution to the Trekion.ai technical assessment. It processes multi-modal sensor data from a robotic camera rig, including a video stream, IMU binary data, and a video timestamp file, to produce three annotated output videos.

---

## Repository Structure

```
.
├── parse_imu.py            # Binary parser for .imu and .vts formats (shared module)
├── task1_imu_sync.py       # Task 1 – IMU synchronized visualization video
├── task2_depth.py          # Task 2 – Monocular dense depth estimation video
├── task3_segmentation.py   # Task 3 – Object detection & segmentation video
├── writeup.md              # Brief technical write-up
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Input Files Required

Place these files in the same directory as the scripts before running:

| File | Description |
|------|-------------|
| `recording2.mp4` | Raw camera video (1920×1080, 30 fps, ~44 seconds) |
| `recording2.imu` | Proprietary binary IMU file (magic: `TRIMU001`) |
| `recording2.vts` | Proprietary binary video timestamp file (magic: `TRIVTS01`) |

---

## Setup

### 1. Python version
Python 3.9 or later is recommended.

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

For hand detection (bonus – Task 3 `--hands` flag):
```bash
pip install mediapipe
```

> **GPU note:** Tasks 2 and 3 will automatically use CUDA if a compatible GPU is available. On CPU, use `--skip 2` or `--skip 3` to reduce processing time. Google Colab (free tier) is recommended for GPU access.

---

## Usage

### Task 1 — IMU Synchronized Visualization

Parses `.imu` and `.vts` binary files, syncs IMU data to video frames, and renders a stacked output video with scrolling sensor plots and a telemetry HUD.

```bash
python task1_imu_sync.py \
    --video  recording2.mp4 \
    --imu    recording2.imu \
    --vts    recording2.vts \
    --output imu_sync_output.mp4 \
    --window 2.0 \
    --skip   1
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | `recording2.mp4` | Input video path |
| `--imu` | `recording2.imu` | IMU binary file path |
| `--vts` | `recording2.vts` | VTS binary file path |
| `--output` | `imu_sync_output.mp4` | Output video path |
| `--window` | `2.0` | Rolling time window (seconds) shown in plots |
| `--skip` | `1` | Process every Nth frame (1 = all frames) |

**Output:** `imu_sync_output.mp4` — original video on top, scrolling accel/gyro/mag plots on the bottom, telemetry HUD top-left.

---

### Task 2 — Monocular Depth Estimation

Runs Depth Anything V2 (or MiDaS as fallback) on each frame to produce dense depth maps.

```bash
python task2_depth.py \
    --video recording2.mp4 \
    --output depth_output.mp4 \
    --half_res \
    --skip 5
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | `recording2.mp4` | Input video path |
| `--output` | `depth_output.mp4` | Output video path |
| `--model` | `depth-anything/Depth-Anything-V2-Small-hf` | HuggingFace model ID |
| `--skip` | `1` | Process every Nth frame |
| `--half_res` | `False` | Process at half resolution (faster, less memory) |
| `--colormap` | `inferno` | Colormap: `inferno`, `magma`, `turbo`, or `plasma` |

**Output:** `depth_output.mp4` — side-by-side video: original RGB left, colorized depth map right, with colorbar.

**Model fallback:** If the HuggingFace `transformers` pipeline fails, the script automatically falls back to MiDaS DPT-Large via `torch.hub`.

---

### Task 3 — Object Detection and Segmentation

Runs YOLOv8 instance segmentation on each frame. Optionally runs MediaPipe for hand landmark detection.

```bash
python task3_segmentation.py \
    --video recording2.mp4 \
    --output seg_output.mp4 \
    --model yolov8n-seg \
    --half_res \
    --skip 3 \
    --hands
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | `recording2.mp4` | Input video path |
| `--output` | `segmentation_output.mp4` | Output video path |
| `--model` | `yolov8n-seg` | YOLO model: `yolov8n-seg` (fast) through `yolov8x-seg` (best quality) |
| `--skip` | `1` | Process every Nth frame |
| `--hands` | `False` | Enable MediaPipe hand landmark detection |
| `--half_res` | `False` | Process at half resolution |

**Output:** `seg_output.mp4` — annotated video with colored segmentation masks, bounding boxes, class labels, confidence scores, and (if `--hands`) hand skeleton landmarks.

---

## Module: `parse_imu.py`

This module is used internally by `task1_imu_sync.py` and can also be run standalone to validate binary files:

```bash
python parse_imu.py recording2.imu recording2.vts
```

It exposes three functions:
- `parse_imu_file(path)` → `pd.DataFrame` — parses the `.imu` binary format
- `parse_vts_file(path)` → `pd.DataFrame` — parses the `.vts` binary format
- `sync_imu_to_frames(imu_df, vts_df)` → `dict` — nearest-neighbor timestamp sync

---

## Binary Format Reference

### IMU File (`TRIMU001`)

```
Header (64 bytes):
  [0:8]   ASCII magic "TRIMU001"
  [8:12]  uint32 LE — number of sensor types (3)
  [12:16] uint32 LE — nominal sample rate (~568 Hz)
  [16:64] metadata / padding

Record (80 bytes, little-endian):
  [0:8]   uint64  — hardware timestamp (nanoseconds)
  [8:44]  10× float32 — accel XYZ, gyro XYZ, mag XYZ, temperature
  [44:80] reserved (zeros)
```

### VTS File (`TRIVTS01`)

```
Header (32 bytes):
  [0:8]   ASCII magic "TRIVTS01"
  [8:12]  uint32 LE — version (2)
  [12:32] padding

Record (24 bytes, little-endian):
  [0:4]   uint32  — record index (0-based)
  [4:12]  uint64  — hardware timestamp (nanoseconds, same clock as IMU)
  [12:16] uint32  — video frame index
  [16:24] uint64  — timestamp in microseconds
```

---

## Notes

- **Fisheye lens:** The camera uses a wide-angle fisheye lens. True undistortion requires calibration parameters not included with the data. The depth and segmentation scripts process the distorted image directly; Depth Anything V2 and YOLOv8 are both robust to mild barrel distortion.
- **Performance:** On CPU, Tasks 2 and 3 are slow. Use `--skip 2` or `--skip 3` to speed up, or run on Colab/Kaggle with a free GPU.
- **Reproducibility:** All random seeds are fixed (segmentation color palette uses `random.seed(42)`).

---
