import torch.nn as nn
import timm
import torch

class TimmFeatureEncoder(torch.nn.Module):
    TARGET_REDUCTIONS = (4, 8, 16, 32)

    def __init__(
        self,
        model_name,
        target_channels,
        pretrained=False,
    ):
        super().__init__()

        self.model_name = model_name

        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            exportable=True,
        )

        reductions = list(
            self.encoder.feature_info.reduction()
        )

        channels = list(
            self.encoder.feature_info.channels()
        )

        print(f"[TimmFeatureEncoder] model: {model_name}")
        print("[TimmFeatureEncoder] available features:")

        for index, (reduction, num_channels) in enumerate(
            zip(reductions, channels)
        ):
            print(
                f"  index={index}: "
                f"OS={reduction}, "
                f"channels={num_channels}"
            )

        self.selected_positions = []

        for target_reduction in self.TARGET_REDUCTIONS:
            matches = [
                index
                for index, reduction in enumerate(reductions)
                if reduction == target_reduction
            ]

            if not matches:
                raise ValueError(
                    f"Encoder '{model_name}' does not provide "
                    f"OS{target_reduction}. "
                    f"Available reductions: {reductions}"
                )

            # Se ce ne sono più di una, usa la feature
            # semanticamente più profonda.
            self.selected_positions.append(matches[-1])

        source_channels = [
            channels[index]
            for index in self.selected_positions
        ]

        if len(target_channels) != 4:
            raise ValueError(
                f"target_channels must contain four values, "
                f"got {target_channels}"
            )

        self.source_channels = source_channels
        self.target_channels = list(target_channels)

        self.adapters = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=source_channels,
                    out_channels=destination_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(
                    destination_channels
                ),
                torch.nn.ReLU(
                    inplace=False
                ),
            )
            for source_channels, destination_channels in zip(
                self.source_channels,
                self.target_channels,
            )
        ])

        print("[TimmFeatureEncoder] selected/adapted features:")

        for reduction, source, destination in zip(
            self.TARGET_REDUCTIONS,
            self.source_channels,
            self.target_channels,
        ):
            print(
                f"  OS={reduction}: "
                f"{source} -> {destination} channels"
            )

    def forward(self, x):
        all_features = self.encoder(x)

        selected_features = [
            all_features[position]
            for position in self.selected_positions
        ]

        adapted_features = [
            adapter(feature)
            for adapter, feature in zip(
                self.adapters,
                selected_features,
            )
        ]

        #return the latest 3 features for the neck
        adapted_features = adapted_features[-3:]

        return adapted_features