"""
AutoDrive ZOD dataset loader.

Labels:  {zod_root}/labels/{seq}/*.json  (one per frame, ISO-timestamp filename)
Images:  {zod_root}/images_blur_*/sequences/{seq}/camera_front_blur/

Image preprocessing — identical to AutoSpeed inference pipeline:
    ZOD camera has ~120° HFOV.  We center-crop to 50° HFOV then resize to
    1024×512 with bilinear interpolation.  This is the exact same function
    used in run_cipo_radar.py → center_crop_50deg_resize.

    Lazy cache sequence:
        1. If a cached 1024×512 JPEG exists, decode it directly.
        2. Otherwise open the raw full-resolution image, center-crop to 50° HFOV,
           resize to 1024×512 and save the result atomically in the cache.
        3. Training only: horizontal flip (negates curvature) + colour/noise augmentation
           applied identically to both frames via Albumentations additional targets
        4. Normalise and convert to CHW float32 tensor (ImageNet stats)

    The cache never contains random augmentation. During the first epoch missing
    entries are generated lazily by DataLoader workers; later epochs use the fast path.

Sequential pairs:
    (T-1, T) within each sequence only — no cross-sequence pairing.

Split:  85 / 10 / 5 at sequence level to avoid temporal leakage.

Distance GT:
    d_norm = (150 - min(d, 150)) / 150  →  ∈ [0, 1]
    dist_mask=True  only when cipo_detected=True AND distance is valid.
    dist_mask=False → distance loss is zero for that sample.

Curvature normalisation:
    Raw ZOD curvature spans ≈ ±0.21 (1/m) across the full dataset.
    CURV_SCALE = 0.21 maps that range onto [-1, 1], matching the Tanh output.
    The dataset returns curvature / CURV_SCALE so the L1 loss operates on the
    same ±1 scale as the head output.  To convert model predictions back to
    physical (1/m) units: pred_1_per_m = pred_normalised * CURV_SCALE.
"""

import json
import os
import random
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, get_worker_info

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from Models.data_parsing.zod.zod_utils import (
    get_images_blur_dir,
    get_calibration_path,
)

# ── Network input size (must match AutoSpeed training resolution) ──────────
_NET_W, _NET_H   = 1024, 512
_TARGET_FOV      = 50.0   # degrees — same as AutoSpeed/run_cipo_radar
_ZOD_HFOV_DEG   = 120.0  # fallback; overridden by calibration file
_D_MAX           = 150.0  # metres

# ── Curvature normalisation ────────────────────────────────────────────────
# Empirical max |curvature| across the full ZOD dataset (296 k frames):
#   min = -0.2099 (1/m)   max = +0.2090 (1/m)
# Dividing by CURV_SCALE maps GT onto [-1, 1] — same scale as Tanh output.
# Callers that need physical units: pred_1_per_m = pred_norm * CURV_SCALE
CURV_SCALE: float = 0.21

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# The cache stores only deterministic preprocessing (crop + resize), never
# random augmentation. JPEG keeps the cache reasonably small and is much
# faster to decode than the original high-resolution frames.
_CACHE_DIRNAME = "images_autodrive_1024x512"
_CACHE_JPEG_QUALITY = 90
_CACHE_CONFIG_NAME = "cache_config.json"

# ── Colour / noise augmentation — applied identically to both frames ───────
_COLOUR_AUG = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.4),
    A.GaussNoise(noise_scale_factor=0.2, p=0.3),
    A.ISONoise(color_shift=(0.05, 0.2), intensity=(0.1, 0.3), p=0.2),
    A.ToGray(num_output_channels=3, method='weighted_average', p=0.05),
], additional_targets={"image_curr": "image"})


# ── Image preprocessing ────────────────────────────────────────────────────

def _center_crop_50deg_resize(img: Image.Image, hfov_deg: float) -> np.ndarray:
    """
    Center-crop to 50° HFOV then resize to 1024×512.

    Exact same logic as center_crop_50deg_resize() in run_cipo_radar.py:
        crop_w = round(img_w * 50 / hfov_deg)   ← proportion of 50° from full FOV
        crop_h = crop_w // 2                      ← 2:1 aspect ratio
        Crop centered both horizontally and vertically.
        Resize with PIL BILINEAR to 1024×512.

    Uses actual image dims (img.size), not calibration W/H (may differ).
    Returns: numpy HWC uint8 array of shape (512, 1024, 3).
    """
    img_w, img_h  = img.size
    orig_crop_w   = int(round(img_w * _TARGET_FOV / hfov_deg))
    orig_crop_h   = orig_crop_w // 2                            # 2:1 ratio
    crop_x        = (img_w - orig_crop_w) // 2
    crop_y        = (img_h - orig_crop_h) // 2                 # centered vertically
    cropped       = img.crop((crop_x, crop_y,
                               crop_x + orig_crop_w,
                               crop_y + orig_crop_h))
    resampling = getattr(Image, "Resampling", Image)
    model_img = cropped.resize((_NET_W, _NET_H), resampling.BILINEAR)
    return np.asarray(model_img.convert("RGB"), dtype=np.uint8)


def _cache_config() -> dict:
    """Parameters that determine the bytes stored in the image cache."""
    return {
        "version": 1,
        "width": _NET_W,
        "height": _NET_H,
        "target_hfov_deg": _TARGET_FOV,
        "format": "JPEG",
        "jpeg_quality": _CACHE_JPEG_QUALITY,
        "jpeg_subsampling": 2,
    }


def _ensure_cache_config(cache_root: Path) -> None:
    """Create cache metadata, or reject a cache built with other settings."""
    cache_root.mkdir(parents=True, exist_ok=True)
    config_path = cache_root / _CACHE_CONFIG_NAME
    expected = _cache_config()

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            actual = json.load(fh)
        if actual != expected:
            raise RuntimeError(
                f"Incompatible AutoDrive image cache: {config_path}\n"
                f"Expected: {expected}\nFound: {actual}\n"
                f"Delete or rename '{cache_root}' before changing cache settings."
            )
        return

    tmp_path = config_path.with_name(
        f".{config_path.name}.{os.getpid()}.tmp"
    )
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(expected, fh, indent=2, sort_keys=True)
        os.replace(tmp_path, config_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _load_cached_rgb(cache_path: Path) -> np.ndarray:
    """Decode a cached image and verify that it has the expected dimensions."""
    with Image.open(cache_path) as image:
        image.load()
        if image.size != (_NET_W, _NET_H):
            raise ValueError(
                f"Wrong cached image size {image.size}, expected {(_NET_W, _NET_H)}"
            )
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _atomic_save_jpeg(image: np.ndarray, cache_path: Path) -> None:
    """Publish a complete JPEG atomically, safe with multiple DataLoader workers."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    worker = get_worker_info()
    worker_id = worker.id if worker is not None else 0
    tmp_path = cache_path.with_name(
        f".{cache_path.stem}.{os.getpid()}.{worker_id}.tmp.jpg"
    )

    try:
        Image.fromarray(image, mode="RGB").save(
            tmp_path,
            format="JPEG",
            quality=_CACHE_JPEG_QUALITY,
            subsampling=2,
            optimize=False,
            progressive=False,
        )
        os.replace(tmp_path, cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _load_or_create_cached(
    source_path: Path,
    cache_path: Path,
    hfov_deg: float,
) -> np.ndarray:
    """
    Fast path: decode the already cropped 1024x512 JPEG.
    Slow path: decode source, crop/resize once, atomically cache, then decode
    the cached JPEG so epoch 1 sees exactly the same representation as later epochs.
    """
    if cache_path.is_file():
        try:
            return _load_cached_rgb(cache_path)
        except (OSError, ValueError, UnidentifiedImageError):
            # A stale/corrupt entry is recoverable from the source image.
            cache_path.unlink(missing_ok=True)

    with Image.open(source_path) as source:
        processed = _center_crop_50deg_resize(source.convert("RGB"), hfov_deg)

    _atomic_save_jpeg(processed, cache_path)
    return _load_cached_rgb(cache_path)


def _read_hfov_deg(zod_root: Path, seq: str) -> float:
    """Read horizontal FOV (degrees) from the sequence calibration file."""
    calib_path = get_calibration_path(zod_root, seq)
    if calib_path.exists():
        with open(calib_path) as f:
            calib = json.load(f)["FC"]
        return float(calib["field_of_view"][0])
    return _ZOD_HFOV_DEG  # fallback


def _norm_distance(d_metres: float) -> float:
    return (_D_MAX - min(d_metres, _D_MAX)) / _D_MAX


def _to_tensor(img_np: np.ndarray) -> torch.Tensor:
    img = TF.to_tensor(img_np)
    img = TF.normalize(img, _IMAGENET_MEAN, _IMAGENET_STD)
    return img


# ── Augmentations ──────────────────────────────────────────────────────────

def _augment_pair(img_prev: np.ndarray, img_curr: np.ndarray,
                  curvature: float) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Training augmentations applied to a (prev, curr) image pair.

    Horizontal flip (p=0.5):
        Both frames flipped left-right simultaneously.
        Curvature sign negated  (right curve → left curve after flip).
        Distance and flag are unchanged (symmetric).

    Colour / noise:
        Random parameters are drawn once and applied identically to both frames
        with Albumentations additional_targets, preserving temporal consistency.
    """
    if random.random() < 0.5:
        img_prev  = np.ascontiguousarray(img_prev[:, ::-1, :])
        img_curr  = np.ascontiguousarray(img_curr[:, ::-1, :])
        curvature = -curvature

    result = _COLOUR_AUG(image=img_prev, image_curr=img_curr)
    img_prev = result["image"]
    img_curr = result["image_curr"]

    return img_prev, img_curr, curvature


# ── Dataset ────────────────────────────────────────────────────────────────

class AutoDriveDataset(Dataset):
    """
    Each __getitem__ returns:
        img_prev  : (3, 512, 1024) float tensor  — ImageNet-normalised
        img_curr  : (3, 512, 1024) float tensor
        d_norm    : scalar float ∈ [0, 1]
        curvature : scalar float ∈ [-1, 1]  — normalised by CURV_SCALE (0.21)
                    (sign flipped on horizontal flip; convert to 1/m: × CURV_SCALE)
        flag      : scalar float {0.0, 1.0}  — 1 = CIPO present
        dist_mask : bool  — True = distance loss active for this sample
    """

    def __init__(
        self,
        zod_root: str | Path,
        sequences: list[str],
        is_train: bool = True,
        cache_root: str | Path | None = None,
    ):
        self.is_train = is_train
        self.pairs: list[tuple] = []

        zod_root = Path(zod_root)
        self.cache_root = (
            Path(cache_root) if cache_root is not None
            else zod_root / _CACHE_DIRNAME
        )
        _ensure_cache_config(self.cache_root)

        print(
            f"  Lazy image cache: {self.cache_root}\n"
            f"  Preprocessing: 50° crop → {_NET_W}×{_NET_H} JPEG "
            f"(quality={_CACHE_JPEG_QUALITY})"
        )

        for seq in sequences:
            label_dir = zod_root / "labels" / seq
            if not label_dir.exists():
                continue

            label_files = sorted(label_dir.glob("*.json"))
            if len(label_files) < 2:
                continue

            img_dir = get_images_blur_dir(zod_root, seq)
            hfov_deg = _read_hfov_deg(zod_root, seq)
            records = []
            for lf in label_files:
                with open(lf) as fh:
                    rec = json.load(fh)
                image_rel = Path(rec["image"])
                source_path = img_dir / image_rel

                # Cache mirrors sequence/image identity, but always uses JPEG.
                # Reject absolute or parent-traversing label paths in the cache tree.
                if image_rel.is_absolute() or ".." in image_rel.parts:
                    cache_rel = Path(image_rel.name)
                else:
                    cache_rel = image_rel
                cache_path = (
                    self.cache_root
                    / "sequences"
                    / seq
                    / "camera_front_blur"
                    / cache_rel.with_suffix(".jpg")
                )

                if source_path.exists():
                    records.append(
                        (source_path, cache_path, hfov_deg, rec)
                    )

            for i in range(1, len(records)):
                source_prev, cache_prev, hfov_prev, _ = records[i - 1]
                source_curr, cache_curr, hfov_curr, lbl_curr = records[i]

                cipo      = bool(lbl_curr.get("cipo_detected", False))
                raw_dist  = lbl_curr.get("distance_to_in_path_object")
                curvature = float(lbl_curr.get("curvature") or 0.0)

                if cipo and raw_dist is not None:
                    d_norm    = _norm_distance(float(raw_dist))
                    dist_mask = True
                else:
                    d_norm    = 0.0
                    dist_mask = False

                flag = 1.0 if cipo else 0.0
                self.pairs.append(
                    (
                        source_prev, cache_prev, hfov_prev,
                        source_curr, cache_curr, hfov_curr,
                        d_norm, curvature, flag, dist_mask,
                    )
                )

        print(f"AutoDriveDataset ({'train' if is_train else 'val/test'}): "
              f"{len(self.pairs):,} pairs from {len(sequences)} sequences.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        (
            source_prev, cache_prev, hfov_prev,
            source_curr, cache_curr, hfov_curr,
            d_norm, curvature, flag, dist_mask,
        ) = self.pairs[idx]

        # Fast path after epoch 1: decode small cached JPEGs. Missing files are
        # produced lazily and atomically, so this is safe with many workers.
        img_prev = _load_or_create_cached(source_prev, cache_prev, hfov_prev)
        img_curr = _load_or_create_cached(source_curr, cache_curr, hfov_curr)

        # 3. Training augmentations (flip + colour/noise)
        if self.is_train:
            img_prev, img_curr, curvature = _augment_pair(img_prev, img_curr, curvature)

        # 4. Normalise and convert to tensor
        # Curvature: divide by CURV_SCALE so GT ∈ [-1, 1], matching Tanh output.
        curv_norm = curvature / CURV_SCALE
        return {
            "img_prev":  _to_tensor(img_prev),
            "img_curr":  _to_tensor(img_curr),
            "d_norm":    torch.tensor(d_norm,    dtype=torch.float32),
            "curvature": torch.tensor(curv_norm, dtype=torch.float32),
            "flag":      torch.tensor(flag,      dtype=torch.float32),
            "dist_mask": torch.tensor(dist_mask, dtype=torch.bool),
        }


# ── Splitter ───────────────────────────────────────────────────────────────

class LoadDataAutoDrive:
    """
    Splits all ZOD sequences 85 / 10 / 5 at sequence level to avoid
    temporal leakage.

        data = LoadDataAutoDrive("/path/to/zod")
        data.train / data.val / data.test  →  AutoDriveDataset
    """

    TRAIN_FRAC = 0.85
    VAL_FRAC   = 0.10

    def __init__(
        self,
        zod_root: str | Path,
        cache_root: str | Path | None = None,
    ):
        zod_root   = Path(zod_root)
        labels_dir = zod_root / "labels"

        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

        all_seqs = sorted([d.name for d in labels_dir.iterdir() if d.is_dir()])
        if not all_seqs:
            raise FileNotFoundError(f"No sequence folders found under {labels_dir}")

        n       = len(all_seqs)
        n_train = max(1, round(n * self.TRAIN_FRAC))
        n_val   = max(1, round(n * self.VAL_FRAC))

        train_seqs = all_seqs[:n_train]
        val_seqs   = all_seqs[n_train : n_train + n_val]
        test_seqs  = all_seqs[n_train + n_val :]

        print(f"Sequences — train: {len(train_seqs)}  val: {len(val_seqs)}  test: {len(test_seqs)}")

        self.train = AutoDriveDataset(
            zod_root, train_seqs, is_train=True, cache_root=cache_root
        )
        self.val = AutoDriveDataset(
            zod_root, val_seqs, is_train=False, cache_root=cache_root
        )
        self.test = AutoDriveDataset(
            zod_root, test_seqs, is_train=False, cache_root=cache_root
        )