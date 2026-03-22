from collections import OrderedDict
import logging
from typing import Optional

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class ResidualConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int, padding: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x) + self.conv_skip(x)


class SqueezeExciteBlock(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, rate: list[int] | None = None):
        super().__init__()
        rate = rate or [6, 12, 18]
        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        return self.output(torch.cat([x1, x2, x3], dim=1))


class UpsampleBilinear(nn.Module):
    def __init__(self, scale: int = 2):
        super().__init__()
        self.upsample = nn.Upsample(
            mode="bilinear", scale_factor=scale, align_corners=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder: int, input_decoder: int, output_dim: int):
        super().__init__()
        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )
        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )
        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2


class Generator(nn.Module):
    def __init__(
        self,
        input_channels: int = 7,
        output_channels: int = 3,
        filters: list[int] | None = None,
    ):
        super().__init__()
        filters = filters or [72, 120, 200, 330, 512, 16]

        self.input_layer = nn.Sequential(
            nn.Conv2d(input_channels, filters[0], kernel_size=7, padding=3),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=7, padding=3),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(input_channels, filters[0], kernel_size=7, padding=3)
        )

        self.squeeze_excite1 = SqueezeExciteBlock(filters[0])
        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = SqueezeExciteBlock(filters[1])
        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = SqueezeExciteBlock(filters[2])
        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = UpsampleBilinear(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = UpsampleBilinear(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = UpsampleBilinear(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])
        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], filters[5], 1, padding=0),
            nn.Conv2d(filters[5], output_channels, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        return self.output_layer(x9)


def _normalize_state_dict_keys(state_dict: OrderedDict) -> OrderedDict:
    normalized_state_dict: OrderedDict = OrderedDict()
    for key, value in state_dict.items():
        normalized_key = key[7:] if key.startswith("module.") else key
        normalized_state_dict[normalized_key] = value
    return normalized_state_dict


def load_model(model_path: Optional[str], device: str = "cuda") -> Generator:
    if not model_path:
        raise RuntimeError("Model checkpoint path is required")

    model = Generator(input_channels=7, output_channels=3)

    try:
        raw_state = torch.load(model_path, map_location=device)
        state_dict = _normalize_state_dict_keys(raw_state)
        model.load_state_dict(state_dict, strict=True)
        logger.info("Model loaded: %s", model_path)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Model checkpoint not found at {model_path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Model checkpoint load failed: {exc}") from exc

    model = model.to(device)
    model.eval()
    return model
