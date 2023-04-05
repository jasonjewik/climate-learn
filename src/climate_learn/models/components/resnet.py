import torch
from torch import nn
from .cnn_blocks import PeriodicConv2D, ResidualBlock, Downsample, Upsample

# Large based on https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py
# MIT License


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        history=1,
        hidden_channels=128,
        activation="leaky",
        out_channels=None,
        norm: bool = True,
        dropout: float = 0.1,
        n_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "leaky":
            self.activation = nn.LeakyReLU(0.3)
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

        insize = self.in_channels * history
        # Project image into feature map
        self.image_proj = PeriodicConv2D(
            insize, hidden_channels, kernel_size=7, padding=3
        )

        blocks = []
        for i in range(n_blocks):
            blocks.append(
                ResidualBlock(
                    hidden_channels,
                    hidden_channels,
                    activation=activation,
                    norm=True,
                    dropout=dropout,
                )
            )

        self.blocks = nn.ModuleList(blocks)

        if norm:
            self.norm = nn.BatchNorm2d(hidden_channels)
        else:
            self.norm = nn.Identity()
        out_channels = self.out_channels
        self.final = PeriodicConv2D(
            hidden_channels, out_channels, kernel_size=7, padding=3
        )

    def predict(self, x):
        if len(x.shape) == 5:  # history
            x = x.flatten(1, 2)
        # x.shape [128, 1, 32, 64]
        x = self.image_proj(x)  # [128, 128, 32, 64]

        for m in self.blocks:
            x = m(x)

        pred = self.final(self.activation(self.norm(x)))  # pred.shape [128, 50, 32, 64]

        return pred

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, out_variables, metric, lat, log_postfix
    ):
        # B, C, H, W
        pred = self.predict(x)
        return (
            [
                m(pred, y, out_variables, lat=lat, log_postfix=log_postfix)
                for m in metric
            ],
            x,
        )

    def evaluate(
        self, x, y, variables, out_variables, transform, metrics, lat, clim, log_postfix
    ):
        pred = self.predict(x)
        return [
            m(pred, y, transform, out_variables, lat, clim, log_postfix)
            for m in metrics
        ], pred

class ResNetEncoder2(nn.Module):
    def __init__(
        self,
        in_channels=1,
        hidden_channels=[32,128,512],
        out_channels=1024,
        embedding_dim=1024,
        dropout=0.1,
        img_size=(32,64),
        mode="pretraining"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim
        self.activation = nn.LeakyReLU(0.3)
        self.mode = mode

        self.image_proj = PeriodicConv2D(
            self.in_channels,
            self.hidden_channels[0],
            kernel_size=2,
            stride=2,
            padding=0
        )

        blocks = []
        for i in range(len(self.hidden_channels)):
            in_ch = self.hidden_channels[i]
            if i < len(self.hidden_channels) - 1:
                out_ch = self.hidden_channels[i+1]
            else:
                out_ch = self.out_channels
            blocks.append(
                ResidualBlock(
                    in_ch,
                    out_ch,
                    activation="leaky",
                    norm=True,
                    dropout=dropout
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.BatchNorm2d(self.out_channels)
        if self.mode == "pretraining":
            self.final = nn.Sequential(
                PeriodicConv2D(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=img_size[0]//2,
                    stride=img_size[0]//2,
                    padding=0
                ),
                nn.Flatten(),
                nn.Linear(1*2*self.out_channels, self.embedding_dim)
            )
        else:
            self.final = nn.Identity()

    def forward(self, x):
        x = self.image_proj(x)
        for block in self.blocks:
            x = block(x)
        pred = self.final(self.activation(self.norm(x)))
        return pred


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        hidden_channels=128,
        out_channels=1,
        embedding_dim=512,
        dropout=0.1,
        n_blocks=16,
        downsample_every=5,
        img_size=(32,64),
        mode="pretraining"
    ):
        super().__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.hidden_channels = hidden_channels
        self.embedding_dim = embedding_dim
        self.activation = nn.LeakyReLU(0.3)
        self.mode = mode

        self.image_proj = PeriodicConv2D(
            self.in_channels,
            self.hidden_channels,
            kernel_size=7,
            padding=3
        )

        blocks = []
        num_downsample = 0
        for i in range(n_blocks):
            blocks.append(
                ResidualBlock(
                    self.hidden_channels,
                    self.hidden_channels,
                    activation="leaky",
                    norm=True,
                    dropout=dropout
                )
            )
            if divmod(i, downsample_every)[1] == downsample_every - 1:
                blocks.append(Downsample(self.hidden_channels))
                num_downsample += 1

        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.BatchNorm2d(self.hidden_channels)
        linear_in_dim = (
            self.hidden_channels
            * img_size[0]//2**num_downsample
            * img_size[1]//2**num_downsample
        )
        if self.mode == "pretraining":
            self.final = nn.Sequential(
                nn.Flatten(),
                nn.Linear(linear_in_dim, self.embedding_dim)
            )
        else:
            self.final = nn.Identity()

    def forward(self, x):
        x = self.image_proj(x)
        for block in self.blocks:
            x = block(x)
        pred = self.final(self.activation(self.norm(x)))
        return pred


class ResNetDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=128,
        activation="leaky",
        out_channels=None,
        norm=True,
        dropout=0.1,
        n_blocks=2,
        upsample_every=5
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "leaky":
            self.activation = nn.LeakyReLU(0.3)
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")
        
        self.image_proj = PeriodicConv2D(
            self.in_channels,
            self.hidden_channels,
            kernel_size=7,
            padding=3
        )

        blocks = []
        num_upsample = 0
        for i in range(n_blocks):
            blocks.append(
                ResidualBlock(
                    self.hidden_channels,
                    self.hidden_channels,
                    activation="leaky",
                    norm=True,
                    dropout=dropout
                )
            )
            if divmod(i, upsample_every)[1] == upsample_every - 1:
                blocks.append(Upsample(self.hidden_channels))
                num_upsample += 1
                
        self.blocks = nn.ModuleList(blocks)

        if norm:
            self.norm = nn.BatchNorm2d(hidden_channels)
        else:
            self.norm = nn.Identity()
        out_channels = self.out_channels
        self.final = PeriodicConv2D(
            hidden_channels,
            out_channels,
            kernel_size=7,
            padding=3
        )

    def predict(self, x):
        x = self.image_proj(x)
        for m in self.blocks:
            x = m(x)
        pred = self.final(self.activation(self.norm(x)))
        return pred

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, out_variables, metric, lat, log_postfix
    ):
        # B, C, H, W
        pred = self.predict(x)
        return (
            [
                m(pred, y, out_variables, lat=lat, log_postfix=log_postfix)
                for m in metric
            ],
            x,
        )

    def evaluate(
        self, x, y, variables, out_variables, transform, metrics, lat, clim, log_postfix
    ):
        pred = self.predict(x)
        return [
            m(pred, y, transform, out_variables, lat, clim, log_postfix)
            for m in metrics
        ], pred