"""
ZOD dataset path utilities. Images split across 3 folders by sequence range.
"""
from pathlib import Path

# images_blur_000000_000490, images_blur_000491_000981, images_blur_000982_001472
IMAGES_BLUR_RANGES = [
    (0, 490, "images_blur_000000_000490"),
    (491, 981, "images_blur_000491_000981"),
    (982, 1472, "images_blur_000982_001472"),
]


def get_images_blur_dir(zod_root: Path, seq: str) -> Path:
    """
    Return camera_front_blur path for sequence.
    seq 000666 -> images_blur_000491_000981/sequences/000666/camera_front_blur
    """
    zod_root = Path(zod_root)
    seq_int = int(seq)
    for lo, hi, folder in IMAGES_BLUR_RANGES:
        if lo <= seq_int <= hi:
            return zod_root / folder / "sequences" / seq / "camera_front_blur"
    # Fallback: try first folder
    return zod_root / "images_blur_000000_000490" / "sequences" / seq / "camera_front_blur"


def get_calibration_path(zod_root: Path, seq: str) -> Path:
    """Calibration: infos/sequences/{seq}/calibration.json"""
    return Path(zod_root) / "infos" / "sequences" / seq / "calibration.json"


def iter_all_sequences():
    """Yield sequence IDs 000000 through 001472."""
    for i in range(1473):
        yield f"{i:06d}"


def sequence_output_dir(zod_root: Path, seq: str) -> Path:
    """
    Per-sequence pipeline outputs live under the dataset root:
    {zod_root}/output/{seq}/  (cipo_radar.json, debug/images/, model_input_sample.png, …)
    """
    return Path(zod_root) / "output" / seq


def default_autospeed_checkpoint(zod_root: Path) -> Path:
    """AutoSpeed weights next to the dataset: {zod_root}/models/autospeed.pt"""
    return Path(zod_root) / "models" / "autospeed.pt"
