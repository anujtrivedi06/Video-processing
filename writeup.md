# Trekion.ai Technical Assignment – Write-Up

**Candidate:** [Your Name]
**Date:** [Date]

---

## 1. Binary Format Reverse-Engineering

### Approach

Both files were opened in Python and their raw bytes printed in hex+ASCII side-by-side (standard technique, equivalent to `xxd`). The header magic bytes (`TRIMU001`, `TRIVTS01`) confirmed the file types immediately.

**IMU file (`.imu`):**

After the 8-byte magic, I read uint32 values at offsets 8 and 12. These decoded to `3` (number of sensor types: accelerometer, gyroscope, magnetometer) and `563` (nominal sample rate in Hz, consistent with the stated ~568 Hz). The file then contains 24 bytes of additional metadata and 32 bytes of padding, giving a 64-byte header.

To find the record size I computed `(file_size − 64) / N` for candidate sizes `{48, 52, 56, 64, 72, 80}`. Only **80** gave a clean integer (24,938 records). I then decoded one record byte-by-byte: the first 8 bytes form a `uint64` timestamp in nanoseconds; the next 36 bytes are 9 `float32` values (accel XYZ, gyro XYZ, mag XYZ); byte 44 is a `float32` temperature; bytes 48–79 are zero-padding.

**Validation:** The accelerometer magnitude across 100 consecutive records was 9.64–10.51 m/s², consistent with gravity (~9.8 m/s²). Temperature read 35–36 °C (normal for a sensor that has been running). The inter-sample period was 1.759 ms, giving a measured rate of 568.6 Hz — matching the spec exactly.

**VTS file (`.vts`):**

After the 8-byte magic, bytes 8–11 decode to version `2` and bytes 12–15 to the value `30000` (purpose unclear; possibly a microsecond-level constant or build-time field). The next 16 bytes are zero-padded — total header: 32 bytes.

The candidate record size was found by the same integer-division method: with `header=32`, `record_size=24` gives exactly **1316 records**, which matches `44 s × 30 fps ≈ 1320` frames (the small shortfall is explained by the video starting at hardware frame 28, likely due to a brief pre-roll before recording began).

Each 24-byte record contains:
- `uint32` sequential index (0, 1, 2 …)
- `uint64` hardware timestamp in nanoseconds
- `uint32` video frame index (28–1343)
- `uint64` timestamp in microseconds (approximately `timestamp_ns / 1000`)

The timestamps in both files occupy the same numerical range (~34–78 × 10⁹), confirming they share the same monotonic hardware clock and can be directly compared for synchronization.

---

## 2. Data Synchronization

Synchronization is straightforward given the shared clock domain. For each video frame, its hardware timestamp is looked up from the VTS file. A binary search (`numpy.searchsorted`) is then run on the sorted IMU timestamp array to find the nearest IMU sample. The absolute difference gives the sync delay.

**Measured sync delay statistics:**
- Mean: ~870 µs
- Median: ~870 µs
- Max: ~1760 µs (one IMU period at 568 Hz — expected worst-case)

The ~33 ms frame period at 30 fps means that each video frame has approximately 18–19 IMU readings available within its frame interval, giving sub-millisecond synchronization precision.

---

## 3. Model Choices

### Task 2 – Depth Estimation: **Depth Anything V2 (Small)**

- **Why:** Depth Anything V2 is currently (2024) the strongest general-purpose monocular depth model. The Small variant runs at ~15–30 fps on a GPU with acceptable quality. It was pre-trained on a massive diverse dataset, making it robust to unusual scenes.
- **Alternative considered:** MiDaS DPT-Large — slightly worse on edge cases and slower; used as fallback if HuggingFace is unavailable.
- **Fisheye distortion:** Full undistortion requires the camera intrinsic matrix and distortion coefficients, which are not provided. Depth Anything V2 handles moderate barrel distortion gracefully because its training data includes wide-angle imagery. We note the distortion in the output label and leave correction as a future improvement once calibration data is available.

### Task 3 – Segmentation: **YOLOv8n-seg**

- **Why:** YOLOv8 is the industry-standard real-time detection+segmentation model. The nano (`n`) variant runs at 60+ fps on a GPU and >10 fps on CPU, making it practical for 44-second video processing. It supports 80 COCO classes covering most objects likely to appear in a lab/robotics scene.
- **Confidence threshold 0.35:** Slightly lower than the default (0.25–0.5) to catch smaller or partially-occluded objects without flooding the output with false positives.
- **Hand detection (bonus):** MediaPipe Hands provides 21-landmark skeleton detection. It complements YOLO well because YOLO's `person` class detects body bounding boxes whereas MediaPipe provides precise hand joint positions.

---

## 4. Challenges

- **Binary parsing with no documentation:** The main challenge was distinguishing header fields from data records, and confirming the exact field types. The key insight was that only record sizes that divide evenly into `(file_size − header_size)` are valid candidates, and physical sanity checks (gravity magnitude, temperature range, sample rate) immediately validated the correct interpretation.
- **Performance on CPU:** Depth Anything V2 runs at ~0.5–1 fps on CPU for 1080p input. This is addressed with `--skip` and `--half_res` flags, and Google Colab is recommended for GPU access.
- **Fisheye distortion:** Without calibration data, proper undistortion is impossible. The chosen models are robust enough that this is a minor concern for qualitative evaluation.

---

## 5. Ideas for Improvement

- **Fisheye undistortion:** Calibrate the camera using a checkerboard pattern to estimate the intrinsic matrix and distortion coefficients. Apply `cv2.undistort` before depth/segmentation inference to reduce edge-region errors.
- **Metric depth:** Depth Anything V2 produces relative (affine-invariant) depth. With known camera intrinsics and scale from IMU double-integration (or a stereo/ToF sensor), absolute metric depth could be recovered.
- **IMU pre-integration:** The gyroscope data could be integrated to track device orientation, enabling gravity subtraction from the accelerometer for proper linear acceleration computation.
- **Higher accuracy segmentation:** Replace YOLOv8n with YOLOv8l or Grounded SAM 2 for better small-object detection and higher-quality masks.
- **Data export:** Parse all three files into a structured HDF5 or Parquet dataset for efficient downstream analysis and training data generation.
