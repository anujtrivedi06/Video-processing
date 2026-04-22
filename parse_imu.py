"""
parse_imu.py
------------
Parses the proprietary .imu and .vts binary formats from Trekion recording sessions.

Format reverse-engineered by inspection:

IMU File (TRIMU001):
  Header: 64 bytes
    [0:8]   ASCII magic "TRIMU001"
    [8:12]  uint32 LE  – number of sensor types (3: accel, gyro, mag)
    [12:16] uint32 LE  – nominal sample rate (563 ≈ 568 Hz)
    [16:64] metadata / padding
  Records: N × 80 bytes (little-endian)
    [0:8]   uint64  hardware timestamp (nanoseconds, monotonic clock)
    [8:12]  float32 accel_x  (m/s²)
    [12:16] float32 accel_y  (m/s²)
    [16:20] float32 accel_z  (m/s²)
    [20:24] float32 gyro_x   (deg/s)
    [24:28] float32 gyro_y   (deg/s)
    [28:32] float32 gyro_z   (deg/s)
    [32:36] float32 mag_x    (µT)
    [36:40] float32 mag_y    (µT)
    [40:44] float32 mag_z    (µT)
    [44:48] float32 temperature (°C)
    [48:80] reserved / zeros

VTS File (TRIVTS01):
  Header: 32 bytes
    [0:8]   ASCII magic "TRIVTS01"
    [8:12]  uint32 LE  – version (2)
    [12:16] uint32 LE  – field (30000, purpose unclear)
    [16:32] padding
  Records: N × 24 bytes (little-endian)
    [0:4]   uint32  record_index (0-based sequential)
    [4:12]  uint64  hardware timestamp (nanoseconds, same clock as IMU)
    [12:16] uint32  video frame index
    [16:24] uint64  timestamp in microseconds (≈ timestamp_ns / 1000)
"""

import struct
import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────
IMU_MAGIC     = b"TRIMU001"
VTS_MAGIC     = b"TRIVTS01"
IMU_HEADER_SZ = 64
VTS_HEADER_SZ = 32
IMU_RECORD_SZ = 80
VTS_RECORD_SZ = 24

# struct format strings (little-endian)
IMU_HEADER_FMT = "<8sIIxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # 64 bytes
IMU_RECORD_FMT = "<Qffffffffffff" + "8x"   # 8+10*4+padding = 80 bytes total
# Actually simpler to just slice manually – see parse_imu_file()

VTS_RECORD_FMT = "<IQI4xQ"  # 4+8+4+4_pad+8 = 28? Let's do manual slicing too


def parse_imu_file(path: str) -> pd.DataFrame:
    """
    Parse a .imu binary file and return a DataFrame with columns:
        timestamp_ns, timestamp_s,
        accel_x, accel_y, accel_z,
        gyro_x,  gyro_y,  gyro_z,
        mag_x,   mag_y,   mag_z,
        temperature
    """
    with open(path, "rb") as f:
        raw = f.read()

    # ── Validate magic ─────────────────────────────────────────────────────────
    magic = raw[:8]
    if magic != IMU_MAGIC:
        raise ValueError(f"Bad IMU magic: {magic!r}, expected {IMU_MAGIC!r}")

    # ── Parse header fields ────────────────────────────────────────────────────
    num_sensor_types = struct.unpack_from("<I", raw, 8)[0]   # = 3
    nominal_rate_hz  = struct.unpack_from("<I", raw, 12)[0]  # = 563

    # ── Parse records ──────────────────────────────────────────────────────────
    data_bytes   = raw[IMU_HEADER_SZ:]
    n_records    = len(data_bytes) // IMU_RECORD_SZ
    remainder    = len(data_bytes) %  IMU_RECORD_SZ
    if remainder:
        print(f"[warn] IMU: {remainder} trailing bytes ignored")

    # Pre-allocate arrays for speed
    timestamps   = np.empty(n_records, dtype=np.uint64)
    accel        = np.empty((n_records, 3), dtype=np.float32)
    gyro         = np.empty((n_records, 3), dtype=np.float32)
    mag          = np.empty((n_records, 3), dtype=np.float32)
    temperature  = np.empty(n_records, dtype=np.float32)

    for i in range(n_records):
        off = i * IMU_RECORD_SZ
        timestamps[i]  = struct.unpack_from("<Q",   data_bytes, off)[0]
        accel[i]       = struct.unpack_from("<fff", data_bytes, off +  8)
        gyro[i]        = struct.unpack_from("<fff", data_bytes, off + 20)
        mag[i]         = struct.unpack_from("<fff", data_bytes, off + 32)
        temperature[i] = struct.unpack_from("<f",   data_bytes, off + 44)[0]

    df = pd.DataFrame({
        "timestamp_ns" : timestamps.astype(np.int64),
        "timestamp_s"  : timestamps.astype(np.float64) / 1e9,
        "accel_x"      : accel[:, 0],
        "accel_y"      : accel[:, 1],
        "accel_z"      : accel[:, 2],
        "gyro_x"       : gyro[:, 0],
        "gyro_y"       : gyro[:, 1],
        "gyro_z"       : gyro[:, 2],
        "mag_x"        : mag[:, 0],
        "mag_y"        : mag[:, 1],
        "mag_z"        : mag[:, 2],
        "temperature"  : temperature,
    })

    print(f"[IMU] Parsed {n_records} records  |  "
          f"duration={df['timestamp_s'].iloc[-1]-df['timestamp_s'].iloc[0]:.2f}s  |  "
          f"nominal_rate={nominal_rate_hz} Hz  |  "
          f"num_sensor_types={num_sensor_types}")
    return df


def parse_vts_file(path: str) -> pd.DataFrame:
    """
    Parse a .vts binary file and return a DataFrame with columns:
        record_index, frame_index, timestamp_ns, timestamp_s, timestamp_us
    """
    with open(path, "rb") as f:
        raw = f.read()

    # ── Validate magic ─────────────────────────────────────────────────────────
    magic = raw[:8]
    if magic != VTS_MAGIC:
        raise ValueError(f"Bad VTS magic: {magic!r}, expected {VTS_MAGIC!r}")

    version   = struct.unpack_from("<I", raw, 8)[0]
    field_30k = struct.unpack_from("<I", raw, 12)[0]

    # ── Parse records ──────────────────────────────────────────────────────────
    data_bytes = raw[VTS_HEADER_SZ:]
    n_records  = len(data_bytes) // VTS_RECORD_SZ
    remainder  = len(data_bytes) %  VTS_RECORD_SZ
    if remainder:
        print(f"[warn] VTS: {remainder} trailing bytes ignored")

    rec_indices  = np.empty(n_records, dtype=np.uint32)
    frame_indices = np.empty(n_records, dtype=np.uint32)
    timestamps   = np.empty(n_records, dtype=np.uint64)
    timestamps_us = np.empty(n_records, dtype=np.uint64)

    for i in range(n_records):
        off = i * VTS_RECORD_SZ
        rec_indices[i]   = struct.unpack_from("<I", data_bytes, off)[0]
        timestamps[i]    = struct.unpack_from("<Q", data_bytes, off + 4)[0]
        frame_indices[i] = struct.unpack_from("<I", data_bytes, off + 12)[0]
        timestamps_us[i] = struct.unpack_from("<Q", data_bytes, off + 16)[0]

    df = pd.DataFrame({
        "record_index"  : rec_indices.astype(np.int32),
        "frame_index"   : frame_indices.astype(np.int32),
        "timestamp_ns"  : timestamps.astype(np.int64),
        "timestamp_s"   : timestamps.astype(np.float64) / 1e9,
        "timestamp_us"  : timestamps_us.astype(np.int64),
    })

    print(f"[VTS] Parsed {n_records} records  |  "
          f"frames {df['frame_index'].iloc[0]}–{df['frame_index'].iloc[-1]}  |  "
          f"duration={df['timestamp_s'].iloc[-1]-df['timestamp_s'].iloc[0]:.2f}s  |  "
          f"version={version}")
    return df


def sync_imu_to_frames(imu_df: pd.DataFrame, vts_df: pd.DataFrame) -> dict:
    """
    For each video frame, find the nearest IMU sample in the shared timestamp domain.

    Returns a dict keyed by frame_index with values:
        {
          'nearest_imu_idx': int,
          'sync_delay_ns':   int   (|frame_ts - imu_ts|)
        }
    """
    imu_ts = imu_df["timestamp_ns"].values
    sync   = {}

    for _, row in vts_df.iterrows():
        frame_ts = row["timestamp_ns"]
        idx      = np.searchsorted(imu_ts, frame_ts, side="left")
        # Compare neighbours
        if idx == 0:
            best = 0
        elif idx >= len(imu_ts):
            best = len(imu_ts) - 1
        else:
            left_d  = abs(int(frame_ts) - int(imu_ts[idx - 1]))
            right_d = abs(int(frame_ts) - int(imu_ts[idx]))
            best    = idx - 1 if left_d <= right_d else idx

        sync[int(row["frame_index"])] = {
            "nearest_imu_idx": best,
            "sync_delay_ns":   abs(int(frame_ts) - int(imu_ts[best])),
        }

    delays = np.array([v["sync_delay_ns"] for v in sync.values()])
    print(f"[SYNC] Sync delay (µs) — mean={delays.mean()/1e3:.1f}  "
          f"median={np.median(delays)/1e3:.1f}  max={delays.max()/1e3:.1f}")
    return sync


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    imu_path = sys.argv[1] if len(sys.argv) > 1 else "recording2.imu"
    vts_path = sys.argv[2] if len(sys.argv) > 2 else "recording2.vts"

    imu_df = parse_imu_file(imu_path)
    vts_df = parse_vts_file(vts_path)

    print("\nIMU sample:\n", imu_df.head(3).to_string())
    print("\nVTS sample:\n", vts_df.head(3).to_string())

    sync = sync_imu_to_frames(imu_df, vts_df)
    print(f"\nSync map has {len(sync)} entries (one per video frame)")
