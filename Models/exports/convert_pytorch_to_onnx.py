#!/usr/bin/env python3

import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Models.training.auto_drive_trainer import AutoDriveTrainer


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--pth",
        required=True,
        help="Input PyTorch checkpoint (.pth)",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output ONNX path",
    )

    parser.add_argument(
        "--encoder-name",
        default=None,
        help="Encoder used during training, e.g. tf_efficientnet_lite0",
    )

    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=13,
    )

    parser.add_argument(
        "--no-onnx-simplify",
        action="store_true",
    )

    args = parser.parse_args()

    trainer = AutoDriveTrainer(
        encoder_name=args.encoder_name,
        encoder_pretrained=False,
        torch_compile=False,
    )

    # Load weights into base_model
    trainer.load_checkpoint(args.pth)

    # Export using AutoDriveStreamingWrapper internally
    trainer.export_onnx(
        args.output,
        opset=args.onnx_opset,
        simplify=not args.no_onnx_simplify,
    )

    trainer.cleanup()


if __name__ == "__main__":
    main()