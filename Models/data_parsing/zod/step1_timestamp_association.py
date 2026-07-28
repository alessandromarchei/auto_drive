#!/usr/bin/env python3
"""
Step 1: Timestamp Association for ZOD/Zenesact Dataset

For each camera image:
  1) Closest radar timestamp (16 Hz)
  2) Closest vehicle control timestamp (100 Hz), with optional steering average

Output: associations saved as JSON for use in step 2 (cipo_radar).
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from zod_utils import get_images_blur_dir


def parse_image_timestamp(fname: Path) -> int:
    """Parse ISO timestamp from image filename -> nanoseconds since epoch."""
    # e.g. 000000_quebec_2022-02-14T13:23:32.140954Z.jpg
    stem = fname.stem
    parts = stem.split("_")
    ts_str = "_".join(parts[2:])  # 2022-02-14T13:23:32.140954Z
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1e9)


def load_radar_timestamps(radar_path: Path) -> np.ndarray:
    """Load unique radar timestamps from the radar_front .npy file."""
    npy_files = list(radar_path.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files in {radar_path}")
    data = np.load(npy_files[0], allow_pickle=True)
    return np.unique(data["timestamp"])


def load_vehicle_controls(vehicle_hdf5: Path):
    """Load ego_vehicle_controls timestamps and steering from HDF5."""
    with h5py.File(vehicle_hdf5, "r") as f:
        ts = f["ego_vehicle_controls/timestamp/nanoseconds/value"][:]
        steering_rad = f["ego_vehicle_controls/steering_wheel_angle/angle/radians/value"][:]
    return ts, steering_rad


def load_ego_velocity(vehicle_hdf5: Path):
    """Load ego_vehicle_data timestamps and longitudinal velocity (m/s) from HDF5."""
    with h5py.File(vehicle_hdf5, "r") as f:
        ts = f["ego_vehicle_data/timestamp/nanoseconds/value"][:]
        vel = f["ego_vehicle_data/lon_vel_data/velocity/meters_per_second/value"][:]
    return ts, vel


def find_closest_idx(query_ts: int, ref_ts: np.ndarray) -> int:
    """Return index of ref_ts closest to query_ts."""
    return int(np.argmin(np.abs(ref_ts.astype(np.int64) - query_ts)))


# Volvo XC90 (ZOD collection vehicle) - https://zenseact.com/data-collection
STEERING_COLUMN_RATIO = 16.8  # steering wheel deg / tyre deg
WHEELBASE_M = 2.984


def curvature_from_steering(steering_wheel_rad: float) -> float:
    """
    Curvature (1/m) from steering wheel angle via Ackermann bicycle model.
    tyre_angle = steering_wheel_angle / steering_column_ratio
    curvature = tan(tyre_angle) / wheelbase
    """
    tyre_angle_rad = steering_wheel_rad / STEERING_COLUMN_RATIO
    return np.tan(tyre_angle_rad) / WHEELBASE_M


def main():
    parser = argparse.ArgumentParser(description="Step 1: Timestamp association")
    parser.add_argument("--sequence", type=str, default="000000", help="Sequence ID (e.g. 000490)")
    parser.add_argument("--zod-root", type=str, required=True, help="Path to ZOD dataset root")
    parser.add_argument(
        "--steering-avg-n",
        type=int,
        default=10,
        help="Number of steering samples to average around the matched control timestamp.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: <zod-root>/associations/<seq>_associations.json).",
    )
    args = parser.parse_args()

    zod = Path(args.zod_root)
    seq = args.sequence

    img_dir = get_images_blur_dir(zod, seq)
    radar_dir = zod / "radar_front" / "sequences" / seq / "radar_front"
    vehicle_hdf5 = zod / "vehicle_data" / "sequences" / seq / "vehicle_data.hdf5"

    if not img_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {img_dir}")
    if not radar_dir.exists():
        raise FileNotFoundError(f"Radar dir not found: {radar_dir}")
    if not vehicle_hdf5.exists():
        raise FileNotFoundError(f"Vehicle HDF5 not found: {vehicle_hdf5}")

    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images found in {img_dir}")
    img_timestamps = [parse_image_timestamp(p) for p in images]

    radar_ts = load_radar_timestamps(radar_dir)
    ctrl_ts, steering_rad = load_vehicle_controls(vehicle_hdf5)
    try:
        ego_vel_ts, ego_vel_ms = load_ego_velocity(vehicle_hdf5)
    except (KeyError, OSError) as e:
        print(f"  Warning: ego velocity not available ({e}), using 0", flush=True)
        ego_vel_ts = ctrl_ts
        ego_vel_ms = np.zeros(len(ctrl_ts))

    radar_npy_files = list(radar_dir.glob("*.npy"))
    if not radar_npy_files:
        raise FileNotFoundError(f"No radar .npy found in {radar_dir}")
    radar_npy = radar_npy_files[0]
    radar_data = np.load(radar_npy, allow_pickle=True)
    _ = radar_data  # ensure file readable (and for radar_npy_path metadata)

    associations = []
    half_n = args.steering_avg_n // 2

    for img_path, img_ts in zip(images, img_timestamps):
        radar_idx = find_closest_idx(img_ts, radar_ts)
        radar_ts_matched = int(radar_ts[radar_idx])
        dt_radar_ms = (img_ts - radar_ts_matched) / 1e6

        ctrl_idx = find_closest_idx(img_ts, ctrl_ts)
        ctrl_ts_matched = int(ctrl_ts[ctrl_idx])
        dt_ctrl_ms = (img_ts - ctrl_ts_matched) / 1e6

        ego_vel_idx = find_closest_idx(img_ts, ego_vel_ts)
        ego_speed_ms = float(ego_vel_ms[ego_vel_idx])

        lo = max(0, ctrl_idx - half_n)
        hi = min(len(steering_rad), ctrl_idx + half_n + 1)
        steering_window = steering_rad[lo:hi]
        steering_avg_rad = float(np.mean(steering_window))

        curvature_inv_m = curvature_from_steering(steering_avg_rad)

        associations.append(
            {
                "image": img_path.name,
                "image_timestamp_ns": img_ts,
                "radar_timestamp_ns": radar_ts_matched,
                "radar_dt_ms": round(dt_radar_ms, 2),
                "vehicle_timestamp_ns": ctrl_ts_matched,
                "vehicle_dt_ms": round(dt_ctrl_ms, 2),
                "steering_angle_rad": round(steering_avg_rad, 6),
                "curvature_inv_m": round(curvature_inv_m, 6),
                "ego_speed_ms": round(ego_speed_ms, 3),
                "radar_idx": int(radar_idx),
                "ctrl_idx": int(ctrl_idx),
            }
        )

    out_path = args.output or (zod / "associations" / f"{seq}_associations.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "sequence": seq,
        "num_images": len(associations),
        "radar_timestamps_count": len(radar_ts),
        "vehicle_controls_count": len(ctrl_ts),
        "associations": associations,
        "radar_npy_path": str(radar_npy),
        "vehicle_hdf5_path": str(vehicle_hdf5),
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(associations)} associations to {out_path}", flush=True)
    print(f"  Radar dt (ms): mean={np.mean([a['radar_dt_ms'] for a in associations]):.2f}", flush=True)
    print(f"  Vehicle dt (ms): mean={np.mean([a['vehicle_dt_ms'] for a in associations]):.2f}", flush=True)


if __name__ == "__main__":
    # Ensure local imports work when executed from this directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()

