# Trekion.ai – Robotics Data Engineering Intern Technical Assignment

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

Place these three files in the same directory as the scripts (or pass full paths via CLI flags):

| File | Description |
|---|---|
| `recording2.mp4` | Raw video, 1920×1080 @ 30fps, ~44s, fisheye lens |
| `recording2.imu` | Proprietary binary IMU data (TRIMU001 format) |
| `recording2.vts` | Proprietary binary video timestamp file (TRIVTS01 format) |

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

### 4. GPU (optional but strongly recommended for Tasks 2 & 3)
If you have a CUDA-capable GPU, install PyTorch with CUDA support from https://pytorch.org/get-started/locally/.
The scripts will automatically use the GPU if available.

For CPU-only machines, use `--skip 2` or `--skip 3` on Tasks 2 and 3 to reduce processing time, and consider running on **Google Colab** (free GPU):

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## Running the Scripts

### Task 1 – IMU Synchronized Visualization
```bash
python task1_imu_sync.py \
    --video  recording2.mp4 \
    --imu    recording2.imu \
    --vts    recording2.vts \
    --output imu_sync_output.mp4
```

Optional flags:
- `--window 2.0`  – scrolling time window in seconds (default: 2.0)
- `--skip 1`      – process every Nth frame; use `--skip 2` or `3` to speed up

### Task 2 – Depth Estimation
```bash
python task2_depth.py \
    --video   recording2.mp4 \
    --output  depth_output.mp4
```

Optional flags:
- `--model depth-anything/Depth-Anything-V2-Small-hf`  – change model (default)
- `--skip 2`        – process every 2nd frame (recommended on CPU)
- `--half_res`      – halve input resolution before inference (faster)
- `--colormap turbo` – choose colormap: `inferno`, `magma`, `turbo`, `plasma`

### Task 3 – Object Detection & Segmentation
```bash
python task3_segmentation.py \
    --video   recording2.mp4 \
    --output  segmentation_output.mp4
```

Optional flags:
- `--model yolov8n-seg`  – YOLO variant: `n` (fast) → `s` → `m` → `l` → `x` (accurate)
- `--conf 0.35`          – detection confidence threshold
- `--iou 0.45`           – NMS IoU threshold
- `--skip 1`             – process every Nth frame
- `--half_res`           – halve resolution for speed
- `--hands`              – enable MediaPipe hand detection (requires `pip install mediapipe`)

---

## Dependencies

```
opencv-python>=4.8
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
transformers>=4.40        # for Depth Anything V2 (Task 2)
torch>=2.0                # PyTorch backend
torchvision>=0.15
Pillow>=10.0
ultralytics>=8.0          # YOLOv8 (Task 3)
```

---

## Notes on Binary Format Parsing

See `parse_imu.py` for the fully documented format specifications reverse-engineered from the binary files. Key findings:

- Both `.imu` and `.vts` files use **little-endian** encoding throughout.
- Timestamps are **uint64 nanoseconds** on a shared hardware monotonic clock, enabling direct cross-file synchronization.
- IMU records are **80 bytes** each (10 floats + padding), with a 64-byte file header.
- VTS records are **24 bytes** each (1× uint32 + 1× uint64 + 1× uint32 + 1× uint64), with a 32-byte file header.
- Effective IMU sample rate measured from timestamps: **568.6 Hz**.
- VTS covers frames **28–1343** (1316 frames, 43.8 s), matching the video duration.

---

## Reproducibility

All models are downloaded automatically on first run from official model hubs (HuggingFace, Ultralytics). No manual model download is required.

To fully reproduce, run the three scripts in order on a machine with internet access and at least 4 GB RAM (8 GB+ recommended for Tasks 2 & 3).
