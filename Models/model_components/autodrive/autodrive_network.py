import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from Models.model_components.autodrive.autodrive_backbone import (
    AutoDriveBackbone,
)
from Models.model_components.autodrive.autodrive_head import (
    AutoDriveHead,
)
from Models.model_components.backbones import TimmFeatureEncoder


IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 512

_WIDTH = [3, 16, 32, 64, 128, 256]
_DEPTH = [1, 1, 1, 1, 1, 1]
_CSP = [False, True]


def _remove_wrapper_prefixes(key: str) -> str:
    """

    Examples:
        module.net.encoder...       -> net.encoder...
        _orig_mod.net.encoder...    -> net.encoder...
        module._orig_mod.net...     -> net...
    """

    prefixes = (
        "module.",
        "_orig_mod.",
    )

    changed = True

    while changed:
        changed = False

        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True

    return key


def _extract_state_dict(checkpoint):
    """
    Supporta i checkpoint salvati come:

        {"model": nn.Module}
        {"model": state_dict}
        {"model_state_dict": state_dict}
        {"state_dict": state_dict}
        state_dict
    """

    if not isinstance(checkpoint, dict):
        if hasattr(checkpoint, "state_dict"):
            return checkpoint.state_dict()

        raise TypeError(
            "Unsupported checkpoint format: expected a dictionary "
            "or a torch.nn.Module"
        )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    elif "model" in checkpoint:
        model_or_state_dict = checkpoint["model"]

        if hasattr(model_or_state_dict, "state_dict"):
            state_dict = model_or_state_dict.state_dict()
        else:
            state_dict = model_or_state_dict

    else:
        # Bare state dictionary.
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(
            "The extracted checkpoint object is not a state dictionary"
        )

    return {
        _remove_wrapper_prefixes(key): value
        for key, value in state_dict.items()
    }


class AutoDrive(nn.Module):
    """
    AutoDrive network with a backbone shared between previous/current images.

    Backbone:
        - encoder_name=None:
            original AutoDriveBackbone;
        - encoder_name provided:
            TimmFeatureEncoder, selecting the OS=32/P5 output.

    Head:
        concatenates the P5 maps extracted from the previous and current
        frames, then predicts:
            - normalized distance;
            - curvature;
            - flag logit.
    """

    def __init__(
        self,
        encoder_name: Optional[str] = None,
        encoder_pretrained: bool = False,
        autospeed_checkpoint_path: Optional[str] = None,
    ):
        super().__init__()

        self.encoder_name = encoder_name

        if encoder_name is None:
            self.backbone = AutoDriveBackbone(
                _WIDTH,
                _DEPTH,
                _CSP,
            )

            print(
                "[AutoDrive] Using original AutoDriveBackbone"
            )

        else:
            self.backbone = TimmFeatureEncoder(
                model_name=encoder_name,
                pretrained=encoder_pretrained,
                target_channels=[
                    _WIDTH[3],  # OS=4  -> 64
                    _WIDTH[4],  # OS=8  -> 128
                    _WIDTH[4],  # OS=16 -> 128
                    _WIDTH[5],  # OS=32 -> 256
                ],
            )

            print(
                f"[AutoDrive] Using timm encoder: {encoder_name}"
            )
            print(
                "[AutoDrive] Selected P5: "
                f"OS=32, channels={_WIDTH[5]}"
            )

        self.head = AutoDriveHead(
            in_channels=_WIDTH[5],
            p5_h=IMAGE_HEIGHT // 32,
            p5_w=IMAGE_WIDTH // 32,
        )

        if autospeed_checkpoint_path is not None:
            self.load_backbone_from_autospeed(
                autospeed_checkpoint_path
            )

    def _extract_p5(self, backbone_output):
        """
        Original AutoDriveBackbone returns P5 directly.

        TimmFeatureEncoder returns:
            [P2, P3, P4, P5]

        where the last tensor is the OS=32 feature.
        """

        if self.encoder_name is None:
            if not isinstance(backbone_output, torch.Tensor):
                raise TypeError(
                    "Original AutoDriveBackbone was expected to return "
                    f"a Tensor, got {type(backbone_output).__name__}"
                )

            return backbone_output

        if not isinstance(backbone_output, (tuple, list)):
            raise TypeError(
                "TimmFeatureEncoder was expected to return a list/tuple "
                f"of feature maps, got {type(backbone_output).__name__}"
            )

        if len(backbone_output) != 4:
            raise RuntimeError(
                "TimmFeatureEncoder was expected to return four feature "
                f"maps at OS=4,8,16,32, but returned {len(backbone_output)}"
            )

        p5 = backbone_output[-1]

        expected_channels = _WIDTH[5]
        expected_height = IMAGE_HEIGHT // 32
        expected_width = IMAGE_WIDTH // 32

        if p5.shape[1] != expected_channels:
            raise RuntimeError(
                "Invalid P5 channel count: "
                f"expected {expected_channels}, got {p5.shape[1]}"
            )

        if (
            p5.shape[-2] != expected_height
            or p5.shape[-1] != expected_width
        ):
            raise RuntimeError(
                "Invalid P5 spatial shape: "
                f"expected ({expected_height}, {expected_width}), "
                f"got {tuple(p5.shape[-2:])}"
            )

        return p5

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract the P5/OS=32 feature map.
        """

        backbone_output = self.backbone(image)
        return self._extract_p5(backbone_output)

    def forward(
        self,
        image_curr: torch.Tensor,
        feature_prev: torch.Tensor,
    ):
        """
        Args:
            image_curr:
                Current RGB frame.
                Shape: [B, 3, 512, 1024]

            feature_prev:
                Cached P5 feature map from the previous frame.
                Shape: [B, 256, 16, 32]

        Returns:
            distance_normalized: [B, 1]
            curvature:          [B, 1]
            flag_logit:         [B, 1]
            feature_curr:       [B, 256, 16, 32]
        """

        # only execute encoder on current image
        feature_curr = self.encode(image_curr)

        distance, curvature, flag_logit = self.head(
            feature_prev,
            feature_curr,
        )

        # feature_curr is returned so that it can be cached for the next frame
        return (
            distance,
            curvature,
            flag_logit,
            feature_curr,
        )


    def load_backbone_from_autospeed(
        self,
        autospeed_checkpoint_path: str,
        require_full_match: bool = True,
    ) -> None:
        """
        Load backbone `net.*` from checkpoint AutoSpeed.

        """

        checkpoint_path = Path(
            autospeed_checkpoint_path
        ).expanduser().resolve()

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"AutoSpeed checkpoint not found: {checkpoint_path}"
            )

        print(
            "[AutoDrive] Loading backbone from AutoSpeed checkpoint:"
        )
        print(f"  {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        autospeed_state_dict = _extract_state_dict(
            checkpoint
        )

        autospeed_backbone_state = {
            key[len("net."):]: value
            for key, value in autospeed_state_dict.items()
            if key.startswith("net.")
        }

        if not autospeed_backbone_state:
            autospeed_backbone_state = autospeed_state_dict

        target_state = self.backbone.state_dict()

        matched = {}
        missing = []
        mismatched = []

        for target_key, target_value in target_state.items():
            if target_key not in autospeed_backbone_state:
                missing.append(target_key)
                continue

            source_value = autospeed_backbone_state[target_key]

            if source_value.shape != target_value.shape:
                mismatched.append(
                    (
                        target_key,
                        tuple(source_value.shape),
                        tuple(target_value.shape),
                    )
                )
                continue

            matched[target_key] = source_value

        unexpected = [
            key
            for key in autospeed_backbone_state
            if key not in target_state
        ]

        print(
            f"  matched:    {len(matched)}/{len(target_state)}"
        )
        print(f"  missing:    {len(missing)}")
        print(f"  mismatched: {len(mismatched)}")
        print(f"  unexpected: {len(unexpected)}")

        for key in missing[:10]:
            print(f"    missing: {key}")

        for key, source_shape, target_shape in mismatched[:10]:
            print(
                f"    mismatch: {key}, "
                f"AutoSpeed={source_shape}, "
                f"AutoDrive={target_shape}"
            )

        if not matched:
            raise RuntimeError(
                "No AutoSpeed backbone parameters matched AutoDrive. "
                "Check that both networks use the same backbone type, "
                "encoder_name and target channels."
            )

        if require_full_match and (
            missing or mismatched
        ):
            raise RuntimeError(
                "AutoSpeed backbone is not fully compatible with "
                "AutoDrive. Set require_full_match=False only if "
                "partial loading is intentional."
            )

        load_result = self.backbone.load_state_dict(
            matched,
            strict=False,
        )

        print(
            "[AutoDrive] AutoSpeed backbone loaded successfully"
        )
        print(
            f"  load missing keys: "
            f"{len(load_result.missing_keys)}"
        )
        print(
            f"  load unexpected keys: "
            f"{len(load_result.unexpected_keys)}"
        )