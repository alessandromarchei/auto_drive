import torch
import sys
sys.path.append('../../../../')
from Models.model_components.auto_speed.auto_speed_backbone import AutoSpeedBackbone
from Models.model_components.auto_speed.auto_speed_neck import AutoSpeedNeck
from Models.model_components.auto_speed.auto_speed_head import AutoSpeedHead
from Models.model_components.common_layers import Conv
from Models.model_components.backbones import TimmFeatureEncoder
import onnx
import onnxsim
import pathlib

image_width = 1024
image_height = 512

def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class YOLO(torch.nn.Module):
    def __init__(self, width, depth, csp, num_classes, encoder_name=None, encoder_pretrained=False):
        super().__init__()

        self.encoder_name = encoder_name

        if encoder_name is None:
            self.net = AutoSpeedBackbone(
                width,
                depth,
                csp,
            )

            print(
                "[YOLO] Using original AutoSpeedBackbone"
            )

        else:
            self.net = TimmFeatureEncoder(
                model_name=encoder_name,
                pretrained=encoder_pretrained,
                target_channels=[
                    width[3],
                    width[4],
                    width[4],
                    width[5],
                ],
            )

            print(
                f"[YOLO] Using timm encoder: {encoder_name}"
            )

            print(
                f"[YOLO] Native AutoSpeed width preserved: {width}"
            )

        self.fpn = AutoSpeedNeck(width, depth, csp)

        img_dummy = torch.zeros(1, width[0], image_height, image_width)
        self.head = AutoSpeedHead(num_classes, (width[3], width[4], width[5]))
        self.head.stride = torch.tensor([img_dummy.shape[-2] / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


class AutoSpeedNetwork:
    def __init__(self):
        self.dynamic_weighting = {
            'n': {
                'csp': [False, True],
                'depth': [1, 1, 1, 1, 1, 1],
                'width': [3, 16, 32, 64, 128, 256],
            },
            's': {
                'csp': [False, True],
                'depth': [1, 1, 1, 1, 1, 1],
                'width': [3, 32, 64, 128, 256, 512],
            },
            'm': {
                'csp': [True, True],
                'depth': [1, 1, 1, 1, 1, 1],
                'width': [3, 64, 128, 256, 512, 512],
            },
            'l': {
                'csp': [True, True],
                'depth': [2, 2, 2, 2, 2, 2],
                'width': [3, 64, 128, 256, 512, 512],
            },
            'x': {
                'csp': [True, True],
                'depth': [2, 2, 2, 2, 2, 2],
                'width': [3, 96, 192, 384, 768, 768],
            },
        }

    def build_model(self, version, num_classes, encoder_name=None, encoder_pretrained=False):
        csp = self.dynamic_weighting[version]['csp']
        depth = self.dynamic_weighting[version]['depth']
        width = self.dynamic_weighting[version]['width']
        return YOLO(width, depth, csp, num_classes, encoder_name=encoder_name, encoder_pretrained=encoder_pretrained)

    def load_model(
        self,
        version,
        num_classes,
        checkpoint_path,
        encoder_name=None,
        encoder_pretrained=False,
    ):
        config = self.dynamic_weighting[version]

        model = YOLO(
            width=config['width'],
            depth=config['depth'],
            csp=config['csp'],
            encoder_name=encoder_name,
            encoder_pretrained=encoder_pretrained,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=False,
        )

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']

        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']

        elif 'model' in checkpoint:
            loaded_model = checkpoint['model']

            if isinstance(loaded_model, dict):
                state_dict = loaded_model
            else:
                state_dict = loaded_model.state_dict()

        else:
            state_dict = checkpoint

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        return model

    def export_onnx(
        self,
        model: torch.nn.Module,
        output_path,
        input_shape=(1, 3, image_height, image_width),
        device: str = "cpu",
        simplify: bool = True,
    ):

        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model = model.to(device=device, dtype=torch.float32)
        model.eval()

        input_data = torch.randn(
            *input_shape,
            dtype=torch.float32,
            device=device,
        )

        with torch.inference_mode():
            outputs = model(input_data)

            print("PyTorch output shapes:")
            for index, output in enumerate(outputs):
                print(f"  output[{index}]: {tuple(output.shape)}")

            torch.onnx.export(
                model,
                input_data,
                str(output_path),
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=None,
                training=torch.onnx.TrainingMode.EVAL,
                external_data=False,
                dynamo=False,
            )

        print(f"Exported ONNX: {output_path}")

        onnx_network = onnx.load(str(output_path))

        print(f"Exported ONNX IR version: {onnx_network.ir_version}")
        print(
            "Opsets:",
            [
                (opset.domain or "ai.onnx", opset.version)
                for opset in onnx_network.opset_import
            ],
        )

        if simplify:
            print("Running onnxsim simplification...")

            onnx_network_simplified, check_ok = onnxsim.simplify(
                onnx_network,
                overwrite_input_shapes={"input": list(input_shape)},
            )

            if not check_ok:
                raise RuntimeError(
                    "onnxsim simplification failed the consistency check "
                    "(output mismatch vs original graph)"
                )

            n_before = len(onnx_network.graph.node)
            n_after = len(onnx_network_simplified.graph.node)

            print(
                f"onnxsim: {n_before} -> {n_after} nodes "
                f"({n_before - n_after} removed)"
            )

            onnx_network = onnx_network_simplified

        if onnx_network.ir_version > 9:
            print(f"Setting IR version {onnx_network.ir_version} -> 9")
            onnx_network.ir_version = 9

        onnx.save_model(onnx_network, str(output_path), save_as_external_data=False)

        onnx_network = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_network, full_check=True)

        if onnx_network.ir_version > 9:
            raise RuntimeError(f"Unsupported IR version: {onnx_network.ir_version}")

        print(f"Final ONNX IR version: {onnx_network.ir_version}")
        print("Checks passed - static Reaction-compatible export complete")

        return output_path