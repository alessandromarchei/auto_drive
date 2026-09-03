# AutoDrive trainer

import sys
import math
import copy
from contextlib import nullcontext
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Models.model_components.autodrive.autodrive_network import AutoDrive
from Models.data_utils.load_data_auto_drive import CURV_SCALE


class AverageMeter:
    """Running average over an epoch."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum   = 0.0
        self.count = 0
        self.avg   = 0.0

    def update(self, val: float, n: int = 1):
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


# ---------------------------------------------------------------------------
# Training modes
# ---------------------------------------------------------------------------
TRAIN_MODE_CURVATURE = "curvature"   # freeze backbone + dist/flag heads
TRAIN_MODE_JOINT     = "joint"       # train everything end-to-end


class AutoDriveTrainer:
    """
    Losses
    ------
    curvature : L1( tanh_pred, curvature_gt )              — always
    distance  : L1( relu_pred, d_norm_gt )                 — only when dist_mask=True (CIPO present)
    flag      : BCEWithLogitsLoss( logits, flag_gt,
                  pos_weight≈0.418 )                        — always (joint mode only)

    Training modes
    --------------
    CURVATURE mode  — backbone frozen; only curvature_head updated.
                      Useful to verify the head can learn with pretrained features.
    JOINT mode      — backbone + all heads trained end-to-end.

    TensorBoard
    -----------
    Loss/train_*            per-step losses   (every log_every steps)
    Loss/train_avg_*        epoch-averaged losses
    Loss/val_*              validation losses
    Metrics/flag_acc_%      CIPO classification accuracy
    Metrics/dist_mae_m      distance MAE in metres (CIPO frames only)
    Metrics/lr
    Metrics/grad_norm       gradient norm before clipping
    Hist/flag_logits        histogram of flag logit distribution
    Hist/d_pred             histogram of distance predictions
    Hist/curv_pred          histogram of curvature predictions
    Visualization/sample    annotated frame image
    """

    _CIPO_POS_WEIGHT = torch.tensor([0.295 / 0.705])  # ≈ 0.418
    _CURV_LOSS_W = 2.0
    _DIST_LOSS_W = 2.0
    _FLAG_VAL_THRESHOLD = 0.65
    _WHEELBASE_M = 2.984
    _STEERING_COLUMN_RATIO = 16.8

    def __init__(self, tensorboard_dir: str = "runs",
                 train_mode: str = TRAIN_MODE_JOINT,
                 autospeed_ckpt: str = "",
                 encoder_name: str | None = None,
                 encoder_pretrained: bool = False,
                 amp: bool = False,
                 torch_compile: bool = False,
                 compile_mode: str = "default"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_mode = train_mode
        print(f"AutoDriveTrainer — device: {self.device}  mode: {train_mode}")

        self.amp_enabled = bool(amp and self.device.type == "cuda")
        self.compile_enabled = bool(torch_compile)

        # Keep an unwrapped model for checkpoints, freezing and ONNX export.
        self.base_model = AutoDrive(
            encoder_name=encoder_name,
            encoder_pretrained=encoder_pretrained,
        ).to(self.device)

        # Optionally load pretrained backbone from AutoSpeed
        if autospeed_ckpt:
            self.base_model.load_backbone_from_autospeed(autospeed_ckpt)

        self.model = self.base_model
        if self.compile_enabled:
            if not hasattr(torch, "compile"):
                raise RuntimeError("--torch-compile requires PyTorch >= 2.0")
            print(f"torch.compile: enabled (mode={compile_mode})")
            self.model = torch.compile(self.base_model, mode=compile_mode)
        else:
            print("torch.compile: disabled")

        print(f"AMP: {'enabled' if self.amp_enabled else 'disabled'}")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # LR: 1e-4 for joint (E2E), 3e-4 for curvature-only (head only)
        self.learning_rate = 3e-4 if train_mode == TRAIN_MODE_CURVATURE else 1e-4
        self._setup_optimizer()

        self._l1  = nn.L1Loss()
        self._bce = nn.BCEWithLogitsLoss(
            pos_weight=self._CIPO_POS_WEIGHT.to(self.device)
        )

        # Per-step state
        self.loss:           torch.Tensor | None = None
        self.loss_distance:  float = 0.0
        self.loss_curvature: float = 0.0
        self.loss_flag:      float = 0.0
        self._grad_norm:     float = 0.0
        self._grad_max_abs:  float = 0.0
        self._grad_nonfinite: int = 0
        self._param_norm:    float = 0.0

        # Epoch-level running averages
        self.avg_total     = AverageMeter()
        self.avg_distance  = AverageMeter()
        self.avg_curvature = AverageMeter()
        self.avg_flag      = AverageMeter()

        # Last-batch tensors for histogram logging
        self._flag_logits_cpu: torch.Tensor | None = None
        self._d_pred_cpu:      torch.Tensor | None = None
        self._curv_pred_cpu:   torch.Tensor | None = None

        # Visualization
        self._img_prev_vis:  torch.Tensor | None = None
        self._d_pred_val:    float = 0.0
        self._d_gt_val:      float = 0.0
        self._curv_pred_val: float = 0.0
        self._curv_gt_val:   float = 0.0
        self._flag_pred_val: float = 0.0
        self._flag_gt_val:   float = 0.0

    # ------------------------------------------------------------------
    # Optimizer / parameter groups
    # ------------------------------------------------------------------

    def _setup_optimizer(self):
        """Build optimizer over trainable parameters only."""
        trainable = filter(lambda p: p.requires_grad, self.base_model.parameters())
        self.optimizer = optim.Adam(trainable, lr=self.learning_rate, weight_decay=1e-5)

    def _apply_train_mode(self):
        """Freeze/unfreeze parameters according to train_mode."""
        if self.train_mode == TRAIN_MODE_CURVATURE:
            # Freeze backbone; keep full head trainable.
            # Only curvature loss is computed — distance and flag losses are 0.
            # This lets the head conv+FC layers adapt to the task while the
            # pretrained backbone features stay intact.
            for p in self.base_model.backbone.parameters():
                p.requires_grad_(False)
            for p in self.base_model.head.parameters():
                p.requires_grad_(True)
            n_frozen   = sum(p.numel() for p in self.base_model.backbone.parameters())
            n_trainable = sum(p.numel() for p in self.base_model.head.parameters())
            print(f"  [mode=curvature] backbone FROZEN ({n_frozen:,} params); "
                  f"full head trainable ({n_trainable:,} params). "
                  f"Curvature loss only.")
        else:
            # Everything trainable end-to-end
            for p in self.base_model.parameters():
                p.requires_grad_(True)
            n_total = sum(p.numel() for p in self.base_model.parameters())
            print(f"  [mode=joint] all {n_total:,} parameters trainable.")

        self._setup_optimizer()  # rebuild after changing requires_grad

    def freeze_backbone(self):
        for p in self.base_model.backbone.parameters():
            p.requires_grad_(False)
        self._setup_optimizer()
        print("  Backbone FROZEN.")

    def unfreeze_backbone(self):
        for p in self.base_model.backbone.parameters():
            p.requires_grad_(True)
        self._setup_optimizer()
        print("  Backbone UNFROZEN.")

    def set_learning_rate(self, lr: float):
        self.learning_rate = lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    # ------------------------------------------------------------------
    # Epoch management
    # ------------------------------------------------------------------

    def reset_averages(self):
        self.avg_total.reset()
        self.avg_distance.reset()
        self.avg_curvature.reset()
        self.avg_flag.reset()

    # ------------------------------------------------------------------
    # Batch management
    # ------------------------------------------------------------------

    def set_batch(self, batch: dict):
        self.img_prev     = batch["img_prev"].to(self.device, non_blocking=True)
        self.img_curr     = batch["img_curr"].to(self.device, non_blocking=True)
        self.d_norm_gt    = batch["d_norm"].unsqueeze(1).to(self.device, non_blocking=True)
        self.curvature_gt = batch["curvature"].unsqueeze(1).to(self.device, non_blocking=True)
        self.flag_gt      = batch["flag"].unsqueeze(1).to(self.device, non_blocking=True)
        self.dist_mask    = batch["dist_mask"].to(self.device, non_blocking=True)

        self._img_prev_vis = batch["img_prev"][0]
        self._d_gt_val     = batch["d_norm"][0].item()
        self._curv_gt_val  = batch["curvature"][0].item()
        self._flag_gt_val  = batch["flag"][0].item()

    # ------------------------------------------------------------------
    # Forward + loss
    # ------------------------------------------------------------------

    def run_model(self):
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.amp_enabled else nullcontext()
        )
        with amp_context:
            #at training time, encode the previous image first
            feature_prev = self.base_model.encode(self.img_prev)

            #once previous features are computed, run the model with the current image and previous features
            d_pred, curv_pred, flag_logits, feature_curr = self.model(
                self.img_curr,
                feature_prev,
            )

        # Keep reductions/loss accumulation in FP32 when AMP is active.
        # The objective itself is unchanged from the original trainer.
        d_pred_loss = d_pred.float()
        curv_pred_loss = curv_pred.float()
        flag_logits_loss = flag_logits.float()

        # Curvature — always active
        loss_c = self._l1(curv_pred_loss, self.curvature_gt)

        if self.train_mode == TRAIN_MODE_CURVATURE:
            # Curvature-only: distance and flag losses are zero
            loss_d = torch.tensor(0.0, device=self.device)
            loss_f = torch.tensor(0.0, device=self.device)
        else:
            # Distance — only when CIPO is detected with a valid distance
            if self.dist_mask.any():
                mask_idx = self.dist_mask.unsqueeze(1)
                loss_d   = self._l1(d_pred_loss[mask_idx], self.d_norm_gt[mask_idx])
            else:
                loss_d = torch.tensor(0.0, device=self.device)
            # Flag — always in joint mode
            loss_f = self._bce(flag_logits_loss, self.flag_gt)

        self.loss = (
            self._CURV_LOSS_W * loss_c +
            self._DIST_LOSS_W * loss_d +
            loss_f
        )
        self.loss_curvature = loss_c.item()
        self.loss_distance  = loss_d.item()
        self.loss_flag      = loss_f.item()

        n = self.img_prev.size(0)
        self.avg_total.update(self.loss.item(), n)
        self.avg_curvature.update(self.loss_curvature, n)
        self.avg_distance.update(self.loss_distance, n)
        self.avg_flag.update(self.loss_flag, n)

        # Store tensors for histogram logging (detach, move to CPU)
        with torch.no_grad():
            self._flag_logits_cpu = flag_logits.detach().cpu()
            self._d_pred_cpu      = d_pred.detach().cpu()
            self._curv_pred_cpu   = curv_pred.detach().cpu()

        self._d_pred_val    = d_pred[0].item()
        self._curv_pred_val = curv_pred[0].item()
        self._flag_pred_val = torch.sigmoid(flag_logits[0]).item()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, batch: dict) -> tuple:
        """Returns (total, dist, curv, flag, flag_acc_pct, dist_mae_m, steer_mae_deg)."""
        self.set_batch(batch)

        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.amp_enabled else nullcontext()
        )
        with amp_context:
            # At validation time, encode the previous image first
            feature_prev = self.base_model.encode(self.img_prev)
            d_pred, curv_pred, flag_logits = self.model(feature_prev=feature_prev, feature_curr=self.img_curr)

        d_pred = d_pred.float()
        curv_pred = curv_pred.float()
        flag_logits = flag_logits.float()

        loss_c = self._l1(curv_pred, self.curvature_gt)

        if self.dist_mask.any():
            mask_idx   = self.dist_mask.unsqueeze(1)
            loss_d     = self._l1(d_pred[mask_idx], self.d_norm_gt[mask_idx])
            dist_mae_m = (150.0 * (d_pred[mask_idx] - self.d_norm_gt[mask_idx]).abs()).mean().item()
        else:
            loss_d     = torch.tensor(0.0, device=self.device)
            dist_mae_m = 0.0

        loss_f = self._bce(flag_logits, self.flag_gt)
        total = (
            loss_c +
            loss_d +
            loss_f
        )

        # Validation/Test classification threshold on probability (not on raw logit)
        flag_prob = torch.sigmoid(flag_logits)
        pred_label = (flag_prob > self._FLAG_VAL_THRESHOLD).float()
        flag_acc   = (pred_label == self.flag_gt).float().mean().item() * 100.0

        # Curvature (normalised) -> physical (1/m) -> steering wheel angle (deg)
        curv_pred_m = curv_pred * CURV_SCALE
        curv_gt_m = self.curvature_gt * CURV_SCALE
        steer_scale = self._STEERING_COLUMN_RATIO * (180.0 / math.pi)
        steer_pred_deg = torch.atan(curv_pred_m * self._WHEELBASE_M) * steer_scale
        steer_gt_deg = torch.atan(curv_gt_m * self._WHEELBASE_M) * steer_scale
        steer_mae_deg = (steer_pred_deg - steer_gt_deg).abs().mean().item()

        return (total.item(), loss_d.item(), loss_c.item(), loss_f.item(),
                flag_acc, dist_mae_m, steer_mae_deg)

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    def loss_backward(self):
        self.scaler.scale(self.loss).backward()

    def run_optimizer(self):
        # AMP gradients must be unscaled before measuring/clipping. This keeps
        # the original max_norm=10 optimization rule mathematically intact.
        self.scaler.unscale_(self.optimizer)
        all_params = [p for p in self.base_model.parameters() if p.grad is not None]
        if all_params:
            with torch.no_grad():
                self._grad_max_abs = max(
                    p.grad.detach().abs().max().item() for p in all_params
                )
                self._grad_nonfinite = sum(
                    (~torch.isfinite(p.grad.detach())).sum().item() for p in all_params
                )
                self._param_norm = math.sqrt(sum(
                    p.detach().float().pow(2).sum().item()
                    for p in self.base_model.parameters()
                ))
            self._grad_norm = torch.nn.utils.clip_grad_norm_(
                self.base_model.parameters(), max_norm=10.0
            ).item()
        else:
            self._grad_norm = 0.0

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def get_loss(self) -> float:
        return self.loss.item()

    # ------------------------------------------------------------------
    # Mode helpers
    # ------------------------------------------------------------------

    def set_train_mode(self):
        self.model.train()

    def set_eval_mode(self):
        self.model.eval()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, epoch: int, global_step: int, best_val_loss: float):
        print(f"Saving checkpoint → {path}")
        torch.save({
            "epoch":         epoch,
            "global_step":   global_step,
            "best_val_loss": best_val_loss,
            "train_mode":    self.train_mode,
            "model":         self.base_model.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "scaler":        self.scaler.state_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> tuple[int, int, float]:
        """
        Load checkpoint. Returns (start_epoch, global_step, best_val_loss).
        Handles both full-state and weights-only formats.
        """
        print(f"Loading checkpoint ← {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(ckpt, dict) and "model" in ckpt:
            self.base_model.load_state_dict(ckpt["model"])
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                print("  Optimizer state not restored (parameter groups changed).")
            if "scaler" in ckpt:
                self.scaler.load_state_dict(ckpt["scaler"])
            start_epoch   = ckpt.get("epoch", 0)
            global_step   = ckpt.get("global_step", 0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            saved_mode    = ckpt.get("train_mode", "unknown")
            print(f"  Resuming epoch {start_epoch + 1}, step {global_step}, "
                  f"best_val {best_val_loss:.4f}  (saved mode: {saved_mode})")
        else:
            self.base_model.load_state_dict(ckpt)
            start_epoch = global_step = 0
            best_val_loss = float("inf")
            print("  Weights-only checkpoint — counters reset.")

        return start_epoch, global_step, best_val_loss

    def save_model(self, path: str):
        torch.save(self.base_model.state_dict(), path)

    def export_onnx(self, path: str, opset: int = 13,
                    input_shape=(1, 3, 512, 1024), simplify: bool = True):
        """Export the uncompiled FP32 inference graph with two image inputs."""
        import onnx

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        export_model = copy.deepcopy(self.base_model).cpu().float().eval()
        
        feature_prev = torch.zeros(1, 256, 16, 32, dtype=torch.float32,)
        image_curr = torch.randn(*input_shape, dtype=torch.float32)

        with torch.inference_mode():
            outputs = export_model(feature_prev=feature_prev, image_curr=image_curr)
            print("ONNX output shapes:", [tuple(x.shape) for x in outputs])
            torch.onnx.export(
                export_model,
                (image_curr, feature_prev),
                output_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=[
                    "image_curr",
                    "feature_prev",
                ],
                output_names=[
                    "distance",
                    "curvature",
                    "flag_logit",
                    "feature_curr",
                ],
                dynamic_axes=None,
                training=torch.onnx.TrainingMode.EVAL,
                external_data=False,
                dynamo=False,
            )

        graph = onnx.load(str(output_path))
        if simplify:
            try:
                import onnxsim
                graph, ok = onnxsim.simplify(
                    graph,
                    overwrite_input_shapes={
                        "feature_prev": list(input_shape),
                        "image_curr": list(input_shape),
                    },
                )
                if not ok:
                    raise RuntimeError("onnxsim output consistency check failed")
            except ImportError:
                print("onnxsim not installed: skipping simplification")

        # Reaction 3.47 accepts ONNX IR <= 9.
        if graph.ir_version > 9:
            graph.ir_version = 9
        onnx.checker.check_model(graph, full_check=True)
        onnx.save_model(graph, str(output_path), save_as_external_data=False)
        print(f"ONNX exported: {output_path}")
        del export_model

    # ------------------------------------------------------------------
    # TensorBoard — per step
    # ------------------------------------------------------------------

    def log_train_step(self, step: int):
        """Per-step raw losses + diagnostic scalars."""
        self.writer.add_scalar("Loss/train_total",     self.get_loss(),       step)
        self.writer.add_scalar("Loss/train_curvature", self.loss_curvature,   step)
        self.writer.add_scalar("Loss/train_distance",  self.loss_distance,    step)
        self.writer.add_scalar("Loss/train_flag",      self.loss_flag,        step)
        self.writer.add_scalar("Metrics/grad_norm",    self._grad_norm,       step)
        self.writer.add_scalar("Gradients/max_abs",    self._grad_max_abs,    step)
        self.writer.add_scalar("Gradients/nonfinite",  self._grad_nonfinite,  step)
        self.writer.add_scalar("Parameters/l2_norm",   self._param_norm,      step)
        if self._param_norm > 0:
            self.writer.add_scalar(
                "Optimization/lr_grad_over_param",
                self.learning_rate * self._grad_norm / self._param_norm,
                step,
            )
        self.writer.add_scalar("AMP/scale",            self.scaler.get_scale(), step)
        for index, group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f"LearningRate/group_{index}", group["lr"], step)
        if torch.cuda.is_available():
            self.writer.add_scalar("Memory/cuda_allocated_GB",
                                   torch.cuda.memory_allocated() / 1024**3, step)
            self.writer.add_scalar("Memory/cuda_reserved_GB",
                                   torch.cuda.memory_reserved() / 1024**3, step)

    def log_histograms(self, step: int):
        """Output distribution histograms — call every N steps."""
        if self._flag_logits_cpu is not None:
            self.writer.add_histogram("Hist/flag_logits", self._flag_logits_cpu, step)
            self.writer.add_histogram("Hist/flag_prob",
                                      torch.sigmoid(self._flag_logits_cpu), step)
        if self._d_pred_cpu is not None:
            self.writer.add_histogram("Hist/d_pred_norm", self._d_pred_cpu, step)
            # Convert to metres for readability
            self.writer.add_histogram("Hist/d_pred_m",
                                      150.0 * (1.0 - self._d_pred_cpu.clamp(0, 1)), step)
        if self._curv_pred_cpu is not None:
            self.writer.add_histogram("Hist/curv_pred", self._curv_pred_cpu, step)

    # ------------------------------------------------------------------
    # TensorBoard — per epoch
    # ------------------------------------------------------------------

    def log_train_epoch(self, epoch: int):
        self.writer.add_scalar("Loss/train_avg_total",     self.avg_total.avg,     epoch)
        self.writer.add_scalar("Loss/train_avg_curvature", self.avg_curvature.avg, epoch)
        self.writer.add_scalar("Loss/train_avg_distance",  self.avg_distance.avg,  epoch)
        self.writer.add_scalar("Loss/train_avg_flag",      self.avg_flag.avg,      epoch)
        self.writer.add_scalar("Metrics/lr",               self.learning_rate,     epoch)
        self.writer.flush()

    def log_val_epoch(self, total: float, dist: float, curv: float, flag: float,
                      flag_acc: float, dist_mae_m: float, steer_mae_deg: float, epoch: int):
        self.writer.add_scalar("Loss/val_total",      total,      epoch)
        self.writer.add_scalar("Loss/val_curvature",  curv,       epoch)
        self.writer.add_scalar("Loss/val_distance",   dist,       epoch)
        self.writer.add_scalar("Loss/val_flag",       flag,       epoch)
        self.writer.add_scalar("Metrics/flag_acc_%",  flag_acc,   epoch)
        self.writer.add_scalar("Metrics/dist_mae_m",  dist_mae_m, epoch)
        self.writer.add_scalar("Metrics/steer_mae_deg", steer_mae_deg, epoch)
        self.writer.flush()

    def log_test(self, total: float, dist: float, curv: float, flag: float,
                 flag_acc: float, dist_mae_m: float, steer_mae_deg: float):
        self.writer.add_scalar("Loss/test_total",         total,      0)
        self.writer.add_scalar("Loss/test_curvature",     curv,       0)
        self.writer.add_scalar("Loss/test_distance",      dist,       0)
        self.writer.add_scalar("Loss/test_flag",          flag,       0)
        self.writer.add_scalar("Metrics/test_flag_acc_%", flag_acc,   0)
        self.writer.add_scalar("Metrics/test_dist_mae_m", dist_mae_m, 0)
        self.writer.add_scalar("Metrics/test_steer_mae_deg", steer_mae_deg, 0)
        self.writer.flush()

    def log_train_loss(self, step: int):
        self.log_train_step(step)

    def _curvature_to_steer_deg(self, curvature_1_per_m: float) -> float:
        """Inverse Ackermann: curvature (1/m) -> steering wheel angle (deg)."""
        tyre_angle_rad = math.atan(curvature_1_per_m * self._WHEELBASE_M)
        return math.degrees(tyre_angle_rad * self._STEERING_COLUMN_RATIO)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def save_visualization(self, step: int, split: str = "train"):
        """Write an annotated sample image to TensorBoard for train/val."""
        if self._img_prev_vis is None:
            return

        d_pred_m = 150.0 * (1.0 - self._d_pred_val)
        d_gt_m   = 150.0 * (1.0 - self._d_gt_val)

        img_np = self._img_prev_vis.permute(1, 2, 0).numpy()
        img_np = (img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4),
                                 gridspec_kw={"width_ratios": [3, 1]})
        fig.patch.set_facecolor("#1e1e1e")

        # Left: camera image with overlay
        ax_img = axes[0]
        ax_img.imshow(img_np)
        ax_img.set_title(f"Step {step}", color="white", fontsize=9)
        ax_img.axis("off")

        # Overlay text — curvature displayed in physical units (1/m)
        curv_pred_m = self._curv_pred_val * CURV_SCALE
        curv_gt_m   = self._curv_gt_val   * CURV_SCALE
        steer_pred_deg = self._curvature_to_steer_deg(curv_pred_m)
        steer_gt_deg   = self._curvature_to_steer_deg(curv_gt_m)
        txt_lines = [
            f"dist pred : {d_pred_m:6.1f} m   GT: {d_gt_m:6.1f} m",
            f"curv pred : {curv_pred_m:+.5f} 1/m  GT: {curv_gt_m:+.5f} 1/m",
            f"steer pred: {steer_pred_deg:+6.1f} deg  GT: {steer_gt_deg:+6.1f} deg",
            f"flag prob : {self._flag_pred_val:.3f}        GT: {int(self._flag_gt_val)}",
        ]
        ax_img.text(
            10, 30, "\n".join(txt_lines),
            color="lime", fontsize=7, family="monospace",
            bbox=dict(facecolor="black", alpha=0.65, edgecolor="none", pad=4),
        )

        # Right: bar chart of three predictions vs GT
        # Curvature is already normalised ∈ [-1, 1] (GT÷CURV_SCALE, pred=Tanh).
        ax_bar = axes[1]
        ax_bar.set_facecolor("#2a2a2a")
        labels  = ["d_norm", "curv (norm)", "flag_prob"]
        pred_vals = [
            float(np.clip(self._d_pred_val,    0, 1)),
            float(np.clip(self._curv_pred_val, -1, 1)),
            float(self._flag_pred_val),
        ]
        gt_vals = [
            float(np.clip(self._d_gt_val,    0, 1)),
            float(np.clip(self._curv_gt_val, -1, 1)),
            float(self._flag_gt_val),
        ]
        x = np.arange(len(labels))
        w = 0.35
        ax_bar.bar(x - w/2, pred_vals, w, label="pred", color="#4fc3f7", alpha=0.85)
        ax_bar.bar(x + w/2, gt_vals,   w, label="GT",   color="#ef9a9a", alpha=0.85)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels, color="white", fontsize=7)
        ax_bar.set_ylim(-1.1, 1.1)
        ax_bar.axhline(0, color="white", linewidth=0.5, alpha=0.5)
        ax_bar.legend(fontsize=7, labelcolor="white",
                      facecolor="#333333", edgecolor="none")
        ax_bar.tick_params(colors="white")
        ax_bar.set_title("Pred vs GT", color="white", fontsize=9)

        plt.tight_layout()
        tag = "Visualization/val_sample" if split == "val" else "Visualization/sample"
        self.writer.add_figure(tag, fig, global_step=step)
        plt.close(fig)

    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print("AutoDriveTrainer: finished.")