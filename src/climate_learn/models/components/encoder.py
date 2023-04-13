import torch
from torch import nn
from .cnn_blocks import PeriodicConv2D, ResidualBlock, Downsample


class Encoder(nn.Module):
    def __init__(
        self,
        out_channels=512,
        embedding_dim=128,
        dropout=0.1,
        mode="pretraining"
    ):
        super().__init__()
        self.in_channels = 1
        self.hidden_channels = [64, 64, 128, 128, 512, 512]
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim
        self.activation = nn.LeakyReLU(0.3)
        self.mode = mode

        self.image_proj = PeriodicConv2D(
            self.in_channels,
            self.hidden_channels[0],
            kernel_size=7,
            padding=3
        )

        blocks = []
        for i in range(len(self.hidden_channels)-1):
            in_ch = self.hidden_channels[i]
            out_ch = self.hidden_channels[i+1]
            blocks.append(
                ResidualBlock(
                    in_ch,
                    out_ch,
                    activation="leaky",
                    norm=True,
                    dropout=dropout
                )
            )
            if divmod(i, 2)[1] == 1:
                blocks.append(Downsample(out_ch))

        self.blocks = nn.ModuleList(blocks)

        self.norm = nn.BatchNorm2d(self.out_channels)
        self.feature_proj = PeriodicConv2D(
            self.hidden_channels[-1],
            self.out_channels,
            kernel_size=7,
            padding=3
        )
        if mode == "pretraining":
            self.final = nn.Sequential(
                Downsample(self.out_channels),
                Downsample(self.out_channels),
                nn.Flatten(),
                nn.Linear(2*4*self.out_channels, self.embedding_dim)
            )
        else:
            self.final = nn.Identity()

    def forward(self, x):
        x = self.image_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.feature_proj(x)
        pred = self.final(self.activation(self.norm(x)))
        return pred