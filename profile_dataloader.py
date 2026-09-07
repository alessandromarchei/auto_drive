#!/usr/bin/env python3
"""Standalone stage profiler for AutoDriveDataset workers.

It does not modify training. It monkey-patches AutoDriveDataset.__getitem__ in
the worker processes and reports per-worker averages plus the wall-clock time
spent waiting for DataLoader batches.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, default_collate, get_worker_info

_REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(_REPO_ROOT))

import Models.data_utils.load_data_auto_drive as dataset_module


STAGES = (
    "open_decode",
    "crop_resize",
    "augmentation",
    "to_tensor",
    "getitem_total",
)


def profiled_getitem(self, idx: int) -> dict:
    total_start = time.perf_counter_ns()
    path_prev, path_curr, d_norm, curvature, flag, dist_mask = self.pairs[idx]

    stage_start = time.perf_counter_ns()
    with Image.open(path_prev) as image:
        img_prev_pil = image.convert("RGB")
        img_prev_pil.load()
    with Image.open(path_curr) as image:
        img_curr_pil = image.convert("RGB")
        img_curr_pil.load()
    open_decode_ms = (time.perf_counter_ns() - stage_start) / 1e6

    stage_start = time.perf_counter_ns()
    img_prev = dataset_module._center_crop_50deg_resize(
        img_prev_pil, self.hfov_deg
    )
    img_curr = dataset_module._center_crop_50deg_resize(
        img_curr_pil, self.hfov_deg
    )
    crop_resize_ms = (time.perf_counter_ns() - stage_start) / 1e6

    stage_start = time.perf_counter_ns()
    if self.is_train:
        img_prev, img_curr, curvature = dataset_module._augment_pair(
            img_prev, img_curr, curvature
        )
    augmentation_ms = (time.perf_counter_ns() - stage_start) / 1e6

    stage_start = time.perf_counter_ns()
    img_prev_tensor = dataset_module._to_tensor(img_prev)
    img_curr_tensor = dataset_module._to_tensor(img_curr)
    to_tensor_ms = (time.perf_counter_ns() - stage_start) / 1e6

    curv_norm = curvature / dataset_module.CURV_SCALE
    worker = get_worker_info()
    worker_id = -1 if worker is None else worker.id
    total_ms = (time.perf_counter_ns() - total_start) / 1e6

    return {
        "img_prev": img_prev_tensor,
        "img_curr": img_curr_tensor,
        "d_norm": torch.tensor(d_norm, dtype=torch.float32),
        "curvature": torch.tensor(curv_norm, dtype=torch.float32),
        "flag": torch.tensor(flag, dtype=torch.float32),
        "dist_mask": torch.tensor(dist_mask, dtype=torch.bool),
        "_profile_ms": torch.tensor(
            [
                open_decode_ms,
                crop_resize_ms,
                augmentation_ms,
                to_tensor_ms,
                total_ms,
            ],
            dtype=torch.float64,
        ),
        "_worker_id": torch.tensor(worker_id, dtype=torch.int64),
    }


def profiled_collate(samples: list[dict]) -> dict:
    start = time.perf_counter_ns()
    batch = default_collate(samples)
    batch["_collate_ms"] = torch.tensor(
        (time.perf_counter_ns() - start) / 1e6,
        dtype=torch.float64,
    )
    return batch


def worker_init_fn(_: int) -> None:
    # Avoid every process creating its own CPU thread pool.
    torch.set_num_threads(1)
    try:
        import cv2
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except ImportError:
        pass


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def print_report(
    worker_samples: dict[int, list[np.ndarray]],
    worker_collate: dict[int, list[float]],
    waits_ms: list[float],
    elapsed: float,
    sample_count: int,
) -> None:
    print("\n" + "=" * 104)
    print(
        f"Measured samples: {sample_count:,} | elapsed: {elapsed:.2f} s | "
        f"loader throughput: {sample_count / max(elapsed, 1e-9):.1f} pairs/s"
    )
    print(
        "DataLoader next() wait: "
        f"mean={statistics.fmean(waits_ms) if waits_ms else 0.0:.2f} ms | "
        f"p50={percentile(waits_ms, 50):.2f} | "
        f"p95={percentile(waits_ms, 95):.2f} | "
        f"p99={percentile(waits_ms, 99):.2f}"
    )
    print("-" * 104)
    print(
        f"{'worker':>7} {'samples':>8} {'decode':>10} {'crop':>10} "
        f"{'augment':>10} {'tensor':>10} {'other':>10} {'total':>10} {'collate':>10}"
    )

    all_rows: list[np.ndarray] = []
    for worker_id in sorted(worker_samples):
        rows = np.stack(worker_samples[worker_id])
        all_rows.append(rows)
        means = rows.mean(axis=0)
        measured_sum = means[:4].sum()
        other = max(0.0, means[4] - measured_sum)
        collate_mean = (
            statistics.fmean(worker_collate[worker_id])
            if worker_collate[worker_id] else 0.0
        )
        print(
            f"{worker_id:>7} {len(rows):>8} "
            f"{means[0]:>9.2f} {means[1]:>9.2f} {means[2]:>9.2f} "
            f"{means[3]:>9.2f} {other:>9.2f} {means[4]:>9.2f} "
            f"{collate_mean:>9.2f}"
        )

    if all_rows:
        rows = np.concatenate(all_rows, axis=0)
        means = rows.mean(axis=0)
        stage_sum = max(means[:4].sum(), 1e-9)
        print("-" * 104)
        print(
            f"{'ALL':>7} {len(rows):>8} "
            f"{means[0]:>9.2f} {means[1]:>9.2f} {means[2]:>9.2f} "
            f"{means[3]:>9.2f} {max(0.0, means[4] - means[:4].sum()):>9.2f} "
            f"{means[4]:>9.2f} {'':>10}"
        )
        print(
            "Stage share (measured work): "
            + ", ".join(
                f"{name}={100.0 * means[index] / stage_sum:.1f}%"
                for index, name in enumerate(STAGES[:4])
            )
        )
    print("=" * 104)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="ZOD dataset root")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--batches", type=int, default=300)
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument(
        "--disable-augmentations",
        action="store_true",
        help="Profile train split with augmentation disabled for an A/B comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '<unset>')} | "
        f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', '<unset>')}"
    )

    # Ubuntu/Linux fork workers inherit this profiling replacement.
    dataset_module.AutoDriveDataset.__getitem__ = profiled_getitem
    data = dataset_module.LoadDataAutoDrive(args.root)
    dataset = getattr(data, args.split)
    if args.disable_augmentations:
        dataset.is_train = False

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=(args.split == "train"),
        num_workers=args.workers,
        pin_memory=not args.no_pin_memory,
        persistent_workers=(args.workers > 0),
        collate_fn=profiled_collate,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["multiprocessing_context"] = "fork"

    loader = DataLoader(**loader_kwargs)
    iterator = iter(loader)

    worker_samples: dict[int, list[np.ndarray]] = defaultdict(list)
    worker_collate: dict[int, list[float]] = defaultdict(list)
    waits_ms: list[float] = []
    sample_count = 0
    measurement_start = None

    for batch_index in range(args.batches + args.warmup_batches):
        wait_start = time.perf_counter_ns()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        wait_ms = (time.perf_counter_ns() - wait_start) / 1e6

        if batch_index < args.warmup_batches:
            continue
        if measurement_start is None:
            measurement_start = time.perf_counter()

        waits_ms.append(wait_ms)
        profiles = batch.pop("_profile_ms").numpy()
        worker_ids = batch.pop("_worker_id").numpy()
        collate_ms = float(batch.pop("_collate_ms").item())

        unique_workers = np.unique(worker_ids)
        for worker_id in unique_workers:
            mask = worker_ids == worker_id
            worker_samples[int(worker_id)].extend(profiles[mask])
        # Auto-batched DataLoader normally assigns a complete batch to one worker.
        worker_collate[int(worker_ids[0])].append(collate_ms)
        sample_count += len(worker_ids)

        measured_batches = batch_index - args.warmup_batches + 1
        if args.report_every > 0 and measured_batches % args.report_every == 0:
            elapsed = time.perf_counter() - measurement_start
            print_report(
                worker_samples, worker_collate, waits_ms, elapsed, sample_count
            )

    elapsed = 0.0 if measurement_start is None else time.perf_counter() - measurement_start
    print_report(worker_samples, worker_collate, waits_ms, elapsed, sample_count)


if __name__ == "__main__":
    main()
