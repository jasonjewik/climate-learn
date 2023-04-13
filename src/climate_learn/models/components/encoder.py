import torch
from torch import nn
from .cnn_blocks import PeriodicConv2D, ResidualBlock, Downsample


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels=[128,128,256,256,512,512],
        out_channels=512,
        embedding_dim=128,
        dropout=0.1,
        mode="pretraining"
    ):
        super().__init__()
        self.in_channels = 1
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim
        self.activation = nn.LeakyReLU(0.3)
        self.mode = mode

        self.image_proj = PeriodicConv2D(
            self.in_channels,
            self.hidden_channels[0],
            kernel_size=3,
            stride=2,
            padding=1
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

        self.blocks = nn.ModuleList(blocks)

        self.final = PeriodicConv2D(
            self.hidden_channels[-1],
            self.out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
        if mode == "pretraining":
            self.projection_head = nn.Sequential(
                nn.AvgPool2d(8, 8),
                nn.Flatten(),
                nn.Linear(2*self.out_channels, self.embedding_dim)
            )
        else:
            self.projection_head = nn.Identity()

    def forward(self, x):
        x = self.image_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.final(self.activation(x))
        pred = self.projection_head(x)
        return pred