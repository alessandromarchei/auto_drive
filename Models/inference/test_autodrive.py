"""
Test-only script for AutoDrive.

Loads a trained checkpoint and evaluates it on the ZOD test split.

Outputs:
    runs/tests/<run_name>/
        tensorboard/
        results.txt

Example
-------
python Models/training/test_auto_drive.py \
    --root /path/to/zod \
    --checkpoint runs/autodrive/my_run/checkpoints/AutoDrive_best.pth \
    --run-name my_test \
    --encoder-name tf_efficientnet_lite0 \
    --batch-size 16 \
    --workers 2 \
    --amp bf16
"""

import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Models.data_utils.load_data_auto_drive import LoadDataAutoDrive
from Models.training.auto_drive_trainer import (
    AutoDriveTrainer,
    TRAIN_MODE_JOINT,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _collate(batch: list[dict]) -> dict:
    return {
        k: torch.stack([b[k] for b in batch])
        for k in batch[0]
    }


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def _run_test(
    trainer: AutoDriveTrainer,
    loader: DataLoader,
):
    total = 0.0
    dist = 0.0
    curv = 0.0
    flag = 0.0
    acc = 0.0
    mae = 0.0
    steer_mae = 0.0
    n = 0

    pbar = tqdm.tqdm(
        loader,
        total=len(loader),
        desc="Testing",
    )

    with torch.inference_mode():

        for batch in pbar:

            t, d, c, f, a, m, s = trainer.validate(batch)

            total += t
            dist += d
            curv += c
            flag += f
            acc += a
            mae += m
            steer_mae += s

            n += 1

            pbar.set_postfix(
                loss=f"{total / n:.4f}",
                steer=f"{steer_mae / n:.2f}",
                acc=f"{acc / n:.1f}%",
            )

    if n == 0:
        raise RuntimeError("Test DataLoader is empty.")

    return (
        total / n,
        dist / n,
        curv / n,
        flag / n,
        acc / n,
        mae / n,
        steer_mae / n,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = ArgumentParser(
        description="Test a trained AutoDrive checkpoint."
    )

    # Required
    parser.add_argument(
        "--root",
        required=True,
        help="ZOD dataset root.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to AutoDrive checkpoint (.pth).",
    )

    # Output
    parser.add_argument(
        "--runs-dir",
        default="runs/tests",
        help="Directory containing test runs.",
    )

    parser.add_argument(
        "--run-name",
        required=True,
        help="Name of this test run.",
    )

    # Model
    parser.add_argument(
        "--encoder-name",
        default=None,
        help="Encoder used during training, "
             "e.g. tf_efficientnet_lite0.",
    )

    # Runtime
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--amp",
        choices=["fp16", "bf16", "off"],
        default="off",
    )

    parser.add_argument(
        "--tf32",
        action="store_true",
    )

    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Check arguments
    # ------------------------------------------------------------------

    checkpoint = Path(args.checkpoint)

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint}"
        )

    # ------------------------------------------------------------------
    # CUDA
    # ------------------------------------------------------------------

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

    if args.tf32:
        torch.set_float32_matmul_precision("high")

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------

    run_dir = Path(args.runs_dir) / args.run_name
    tb_dir = run_dir / "tensorboard"

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tb_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = run_dir / "results.txt"

    print("=" * 70)
    print("AutoDrive Test")
    print("=" * 70)
    print(f"Checkpoint : {checkpoint}")
    print(f"Dataset    : {args.root}")
    print(f"Output     : {run_dir}")
    print(f"Encoder    : {args.encoder_name}")
    print(f"Batch size : {args.batch_size}")
    print(f"Workers    : {args.workers}")
    print(f"AMP        : {args.amp}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    print("\nLoading dataset...")

    data = LoadDataAutoDrive(args.root)

    test_loader_options = dict(
        num_workers=args.workers,
        collate_fn=_collate,

        # Test doesn't need aggressive DataLoader configuration.
        # This avoids unnecessary RAM pressure.
        pin_memory=False,
        persistent_workers=False,
    )

    if args.workers > 0:
        test_loader_options["prefetch_factor"] = 1

    test_loader = DataLoader(
        data.test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **test_loader_options,
    )

    print(f"Test samples : {len(data.test):,}")
    print(f"Test batches : {len(test_loader):,}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    print("\nCreating model...")

    trainer = AutoDriveTrainer(
        tensorboard_dir=str(tb_dir),

        # Evaluation computes all three outputs/losses.
        train_mode=TRAIN_MODE_JOINT,

        # No pretrained initialization needed:
        # checkpoint will overwrite the model weights.
        autospeed_ckpt="",

        encoder_name=args.encoder_name,
        encoder_pretrained=False,

        amp=args.amp,

        # No reason to compile for a one-shot test.
        torch_compile=False,
    )

    trainer._apply_train_mode()

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------

    print("\nLoading checkpoint...")

    start_epoch, global_step, best_val_loss = (
        trainer.load_checkpoint(str(checkpoint))
    )

    print(f"Checkpoint epoch       : {start_epoch}")
    print(f"Checkpoint global step : {global_step}")
    print(f"Checkpoint best val    : {best_val_loss}")

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    trainer.set_eval_mode()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    print("\nRunning test...")

    start = time.perf_counter()

    results = _run_test(
        trainer,
        test_loader,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    (
        t_total,
        t_dist,
        t_curv,
        t_flag,
        t_acc,
        t_mae,
        t_steer_mae,
    ) = results

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------

    num_samples = len(data.test)

    samples_per_second = (
        num_samples / elapsed
        if elapsed > 0
        else 0.0
    )

    if torch.cuda.is_available():
        peak_allocated = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved()
            / 1024**3
        )
    else:
        peak_allocated = 0.0
        peak_reserved = 0.0

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(f"Total loss       : {t_total:.6f}")
    print(f"Distance loss    : {t_dist:.6f}")
    print(f"Curvature loss   : {t_curv:.6f}")
    print(f"Flag loss        : {t_flag:.6f}")

    print()

    print(f"Flag accuracy    : {t_acc:.2f} %")
    print(f"Distance MAE     : {t_mae:.3f} m")
    print(f"Steering MAE     : {t_steer_mae:.3f} deg")

    print()

    print(f"Test samples     : {num_samples:,}")
    print(f"Elapsed          : {elapsed:.2f} s")
    print(
        f"Throughput       : "
        f"{samples_per_second:.2f} samples/s"
    )

    if torch.cuda.is_available():
        print(
            f"CUDA peak alloc  : "
            f"{peak_allocated:.3f} GB"
        )
        print(
            f"CUDA peak reserv : "
            f"{peak_reserved:.3f} GB"
        )

    print("=" * 70)

    # ------------------------------------------------------------------
    # TensorBoard
    # ------------------------------------------------------------------

    trainer.log_test(
        t_total,
        t_dist,
        t_curv,
        t_flag,
        t_acc,
        t_mae,
        t_steer_mae,
    )

    trainer.writer.add_scalar(
        "Performance/test_seconds",
        elapsed,
        0,
    )

    trainer.writer.add_scalar(
        "Performance/test_samples_per_second",
        samples_per_second,
        0,
    )

    if torch.cuda.is_available():

        trainer.writer.add_scalar(
            "Memory/test_cuda_peak_allocated_GB",
            peak_allocated,
            0,
        )

        trainer.writer.add_scalar(
            "Memory/test_cuda_peak_reserved_GB",
            peak_reserved,
            0,
        )

    # Last processed sample visualization
    trainer.save_visualization(
        0,
        split="val",
    )

    # ------------------------------------------------------------------
    # results.txt
    # ------------------------------------------------------------------

    result_text = f"""AutoDrive Test Results
======================

Checkpoint
----------
path: {checkpoint}
epoch: {start_epoch}
global_step: {global_step}
best_val_loss: {best_val_loss}

Model
-----
encoder: {args.encoder_name}
amp: {args.amp}

Dataset
-------
root: {args.root}
test_samples: {num_samples}
batch_size: {args.batch_size}

Metrics
-------
total_loss: {t_total:.8f}
distance_loss: {t_dist:.8f}
curvature_loss: {t_curv:.8f}
flag_loss: {t_flag:.8f}

flag_accuracy_percent: {t_acc:.4f}
distance_mae_m: {t_mae:.6f}
steering_mae_deg: {t_steer_mae:.6f}

Performance
-----------
elapsed_seconds: {elapsed:.4f}
samples_per_second: {samples_per_second:.4f}
cuda_peak_allocated_GB: {peak_allocated:.4f}
cuda_peak_reserved_GB: {peak_reserved:.4f}
"""

    results_path.write_text(result_text)

    print(f"\nResults saved to:")
    print(f"  {results_path}")

    print("\nTensorBoard:")
    print(f"  tensorboard --logdir {tb_dir}")

    trainer.cleanup()


if __name__ == "__main__":
    main()