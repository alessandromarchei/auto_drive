#!/usr/bin/env python3
"""
Full pipeline: step1 -> CIPO-radar -> labels -> 2x2 debug grid (single PNGs).

All artifacts under --zod-root:
  - {zod_root}/associations/{seq}_associations.json
  - {zod_root}/output/{seq}/cipo_radar.json
  - {zod_root}/labels/{seq}/*.json
  - {zod_root}/output/{seq}/debug/images/grid_*.png

Override the output parent with --output-base (default: {zod_root}/output).
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[4]

# Allow zod_utils import
sys.path.insert(0, str(Path(__file__).parent))
from zod_utils import sequence_output_dir


def run_step1(seq: str, zod_root: Path) -> bool:
    """Run step1 timestamp association."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "step1_timestamp_association.py"),
        "--sequence", seq,
        "--zod-root", str(zod_root),
    ]
    print(f"[1/4] Running step1 associations for {seq}...")
    r = subprocess.run(cmd, cwd=str(zod_root))
    return r.returncode == 0


def run_cipo_radar(seq: str, zod_root: Path, output_dir: Path, model_path: Optional[str] = None) -> bool:
    """Run CIPO-radar association (clustering + tracking)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "cipo_radar.json"
    script = Path(__file__).parent / "run_cipo_radar.py"
    cmd = [sys.executable, str(script), "--sequence", seq, "--zod-root", str(zod_root), "--output", str(out_path)]
    if model_path:
        cmd.extend(["--model-path", model_path])
    print(f"[2/4] Running CIPO-radar (cluster + track) for {seq}...")
    r = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return r.returncode == 0


_STEERING_COLUMN_RATIO = 16.8  # Volvo XC90: steering wheel deg / tyre deg


def generate_labels(seq: str, zod_root: Path, cipo_radar_path: Path) -> bool:
    """Generate all-in-one JSON labels from cipo_radar + associations."""
    assoc_path = zod_root / "associations" / f"{seq}_associations.json"
    out_dir = zod_root / "labels" / seq
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(assoc_path) as f:
        assoc = json.load(f)
    with open(cipo_radar_path) as f:
        cipo = json.load(f)

    # Build image -> full cipo_radar record
    cipo_map = {r["image"]: r for r in cipo["results"]}

    print(f"[3/4] Generating labels for {seq}...")
    n_labels = 0
    for rec in assoc["associations"]:
        img = rec["image"]
        cipo_rec = cipo_map.get(img, {})
        steering = rec.get("steering_angle_rad", 0.0) or 0.0
        tyre = float(steering) / _STEERING_COLUMN_RATIO

        # Exact mirror of the per-image record in `cipo_radar.json`.
        # This way labels/{seq}/*.json can be used directly for annotations.
        label = dict(cipo_rec)
        if label:
            # Ensure steering/tyre/aliases exist for older runs.
            label.setdefault("steering_angle_rad", rec.get("steering_angle_rad"))
            label.setdefault("tyre_angle_rad", round(tyre, 7))
            label.setdefault("ego_speed_ms", rec.get("ego_speed_ms"))
            if label.get("curvature") is None:
                label["curvature"] = label.get("curvature_inv_m", rec.get("curvature_inv_m"))
            label.setdefault("distance_to_in_path_object", label.get("distance_m"))
            label.setdefault("speed_of_in_path_object", label.get("speed_ms"))
        else:
            # Fallback minimal schema (should be rare if cipo_radar covers all frames).
            label = {
                "image": img,
                "image_timestamp_ns": rec.get("image_timestamp_ns"),
                "radar_timestamp_ns": rec.get("radar_timestamp_ns"),
                "steering_angle_rad": rec.get("steering_angle_rad"),
                "tyre_angle_rad": round(tyre, 7),
                "curvature_inv_m": rec.get("curvature_inv_m"),
                "curvature": rec.get("curvature_inv_m"),
                "ego_speed_ms": rec.get("ego_speed_ms"),
                "cipo_detected": False,
                "cipo_scenario": None,
                "cipo_from_path": None,
                "azimuth_radar_deg": None,
                "distance_m": None,
                "speed_ms": None,
                "pixel_point": None,
                "bbox": None,
                "bev_xy": None,
                "speed_ms_adjusted": None,
                "distance_to_in_path_object": None,
                "speed_of_in_path_object": None,
            }
        out_name = Path(img).stem + ".json"
        out_path = out_dir / out_name
        with open(out_path, "w") as f:
            json.dump(label, f, indent=2)
        n_labels += 1

    print(f"  Saved {n_labels} labels to {out_dir}")
    return True


def run_debug_grid(
    seq: str,
    zod_root: Path,
    output_dir: Path,
    cipo_path: Path,
    every: int = 20,
    model_path: Optional[str] = None,
) -> bool:
    """Single 2x2 debug image per sampled frame -> output_dir/debug/images/."""
    out_dir = output_dir / "debug" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).parent / "debug_zod_grid.py"
    cmd = [
        sys.executable,
        str(script),
        "--sequence", seq,
        "--zod-root", str(zod_root),
        "--every", str(every),
        "--labels-dir", str(zod_root / "labels" / seq),
        "--cipo-radar", str(cipo_path),
        "--output-dir", str(out_dir),
    ]
    if model_path:
        cmd.extend(["--model-path", model_path])
    print(f"[4/4] Running 2x2 debug grid for {seq} (every {every} frames) -> {out_dir}...")
    r = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return r.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Full pipeline: associations -> CIPO-radar -> labels -> debug viz")
    parser.add_argument("--sequence", type=str, default="000330")
    parser.add_argument("--zod-root", type=str, default=None, required=True, help="Path to the ZOD dataset root (contains associations/, radar_front/, vehicle_data/, images_blur_* folders, etc.)")
    parser.add_argument("--skip-step1", action="store_true", help="Skip step1 if associations exist")
    parser.add_argument("--skip-labels", action="store_true", help="Skip label generation")
    parser.add_argument("--skip-viz", action="store_true", help="Skip 2x2 debug grid (debug/images/)")
    parser.add_argument("--every", type=int, default=20, help="Debug grid: process every Nth frame")
    parser.add_argument("--skip-debug", action="store_true", help="Skip debug grid (for --all-sequences)")
    parser.add_argument(
        "--output-base",
        type=str,
        default=None,
        help="Parent directory for per-sequence folders (default: {zod_root}/output). "
             "Files go to {output_base}/{seq}/cipo_radar.json, etc.",
    )
    parser.add_argument("--all-sequences", action="store_true", help="Iterate over all 1473 sequences (000000-001472)")
    parser.add_argument("--start-seq", type=str, default=None, help="Start of sequence range, inclusive (e.g. 000000). Use with --all-sequences.")
    parser.add_argument("--end-seq", type=str, default=None, help="End of sequence range, inclusive (e.g. 000490). Use with --all-sequences.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for --all-sequences. Each worker runs a separate process. "
                             "WARNING: each worker loads the model independently - use 1 per GPU to avoid OOM.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="AutoSpeed checkpoint for run_cipo_radar / debug grid (default: {zod_root}/models/autospeed.pt)",
    )
    args = parser.parse_args()

    zod = Path(args.zod_root)

    if args.all_sequences:
        from zod_utils import iter_all_sequences
        sequences = list(iter_all_sequences())
        if args.start_seq:
            sequences = [s for s in sequences if s >= args.start_seq]
        if args.end_seq:
            sequences = [s for s in sequences if s <= args.end_seq]
        print(f"Running pipeline on {len(sequences)} sequences (workers={args.workers})...")
        if args.start_seq or args.end_seq:
            print(f"  Range: {args.start_seq or 'start'} → {args.end_seq or 'end'}")

        if args.workers <= 1:
            failed = []
            for i, seq in enumerate(sequences):
                print(f"\n{'='*60} [{i+1}/{len(sequences)}] Sequence {seq}")
                output_dir = _sequence_output_dir(zod, seq, args.output_base)
                if not _run_pipeline_for_seq(seq, zod, output_dir, args):
                    failed.append(seq)
        else:
            failed = _run_parallel(sequences, zod, args)

        print(f"\nDone. Failed: {len(failed)} sequences")
        if failed:
            print(" ".join(failed[:20]), "..." if len(failed) > 20 else "")
        return 1 if failed else 0

    seq = args.sequence
    output_dir = _sequence_output_dir(zod, seq, args.output_base)
    ok = _run_pipeline_for_seq(seq, zod, output_dir, args)
    return 0 if ok else 1


def _sequence_output_dir(zod: Path, seq: str, output_base: Optional[str]) -> Path:
    """Resolve per-sequence dir: {zod_root}/output/{seq} unless --output-base is set."""
    if output_base:
        return Path(output_base) / seq
    return sequence_output_dir(zod, seq)


def _worker_fn(args_tuple):
    """Worker entry point for parallel execution (subprocess-safe)."""
    seq, zod_root, output_base, args_dict = args_tuple
    zod = Path(zod_root)
    output_dir = _sequence_output_dir(zod, seq, output_base)

    class _Args:
        pass

    a = _Args()
    for k, v in args_dict.items():
        setattr(a, k, v)

    ok = _run_pipeline_for_seq(seq, zod, output_dir, a)
    return seq, ok


def _run_parallel(sequences, zod, args):
    """Run pipeline on multiple sequences in parallel using ProcessPoolExecutor."""
    args_dict = vars(args)
    out_parent = str(Path(args.output_base)) if args.output_base else str(zod / "output")
    tasks = [(seq, str(zod), out_parent, args_dict) for seq in sequences]
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker_fn, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futures):
            seq = futures[fut]
            done += 1
            try:
                _, ok = fut.result()
                status = "OK" if ok else "FAILED"
                if not ok:
                    failed.append(seq)
            except Exception as e:
                status = f"ERROR: {e}"
                failed.append(seq)
            print(f"  [{done}/{len(sequences)}] {seq} → {status}", flush=True)
    return failed


def _run_pipeline_for_seq(seq: str, zod: Path, output_dir: Path, args) -> bool:
    """Run full pipeline for a single sequence. Returns True on success."""
    # Step 1: associations
    assoc_path = zod / "associations" / f"{seq}_associations.json"
    if not args.skip_step1 or not assoc_path.exists():
        if not run_step1(seq, zod):
            print("Step1 failed")
            return False
    else:
        print(f"[1/4] Skipping step1 (associations exist)")

    # Step 2: CIPO-radar
    cipo_path = output_dir / "cipo_radar.json"
    if not run_cipo_radar(seq, zod, output_dir, model_path=args.model_path):
        print("CIPO-radar failed")
        return False

    # Step 3: labels (per-image JSONs) for training/annotation convenience.
    if not args.skip_labels:
        generate_labels(seq, zod, cipo_path)
    else:
        print("[3/4] Skipped label generation (--skip-labels).")

    # Step 4: 2x2 debug grid (works even without labels; falls back to curvature_inv_m).
    if not args.skip_viz and not getattr(args, "skip_debug", False):
        run_debug_grid(seq, zod, output_dir, cipo_path, every=args.every, model_path=args.model_path)
    else:
        print("[4/4] Skipping debug grid")

    print(f"\nDone. Sequence {seq}:")
    print(f"  Associations: {assoc_path}")
    print(f"  Output:      {output_dir}")
    print(f"    cipo_radar:  {cipo_path}")
    print(f"    debug grid:  {output_dir / 'debug' / 'images'}")
    return True


if __name__ == "__main__":
    sys.exit(main())
