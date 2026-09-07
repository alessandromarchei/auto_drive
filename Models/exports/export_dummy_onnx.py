import sys
import time
import argparse
from argparse import ArgumentParser
from pathlib import Path
import math
import torch
import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Models.data_utils.load_data_auto_drive import LoadDataAutoDrive
from Models.training.auto_drive_trainer import (
    AutoDriveTrainer,
    TRAIN_MODE_CURVATURE,
    TRAIN_MODE_JOINT,
)


def _collate(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def _run_val(trainer: AutoDriveTrainer, loader: DataLoader):
    """Average validation metrics across all batches."""
    total = dist = curv = flag = acc = mae = steer_mae = 0.0
    n = 0
    for batch in loader:
        t, d, c, f, a, m, s = trainer.validate(batch)
        total += t; dist += d; curv += c; flag += f; acc += a; mae += m; steer_mae += s
        n += 1
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return total/n, dist/n, curv/n, flag/n, acc/n, mae/n, steer_mae/n


def main():
    parser = ArgumentParser()
    parser.add_argument("--onnx_name",   default="autodrive.onnx",
                        help="Explicit ONNX model name")
    parser.add_argument("--encoder-name", default=None,
                        help="Optional timm encoder, e.g. tf_efficientnet_lite0")
    parser.add_argument("--encoder-pretrained", action="store_true",
                        help="Load timm ImageNet weights before training")
    parser.add_argument("--onnx-opset", type=int, default=13)
    parser.add_argument("--no-onnx-simplify", action="store_true")
    args = parser.parse_args()


    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = AutoDriveTrainer(
        encoder_name=args.encoder_name,
        encoder_pretrained=args.encoder_pretrained,
        torch_compile=False
    )

    trainer._apply_train_mode()
    trainer.zero_grad()


    # ------------------------------------------------------------------
    # Export-only mode
    # ------------------------------------------------------------------
    export_ckpt = Path(args.onnx_name) 

    print("\n" + "=" * 60)
    print("EXPORT ONLY")
    print("=" * 60)
    print(f"Opset      : {args.onnx_opset}")
    print(
        f"Simplify   : "
        f"{'no' if args.no_onnx_simplify else 'yes'}"
    )

    trainer.export_onnx(
        str(export_ckpt),
        opset=args.onnx_opset,
        simplify=not args.no_onnx_simplify,
    )

    trainer.cleanup()

    print(f"\nONNX exported successfully:")
    print(f"  {export_ckpt.resolve()}")

    return


if __name__ == "__main__":
    main()
