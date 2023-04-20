import torch
from torch import nn
from .cnn_blocks import PeriodicConv2D, ResidualBlock, ResNeXtBlock


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, 7, 3, 0)
        self.act = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(540, 3)
        )

    def forward(self, x):
        h = self.act(self.conv(x))
        h = self.fc(h)
        return h


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
    

class Encoder2(nn.Module):
    def __init__(self,
                 in_channels=1,
                 hidden_channels=[64, 64, 64, 128, 128, 128, 256, 256, 256],
                 out_channels=512,
                 out_dim=1024,
                 embed_dim=128,                 
                 activation="leaky-relu",
                 dropout=0.3,
                 mode="pretraining"
                ):
        super().__init__()
        self.image_proj = nn.Conv2d(
            in_channels,
            hidden_channels[0],
            kernel_size=7,
            stride=2,
            padding=3
        )
        res_blocks = []
        for i in range(len(hidden_channels)):
            in_ch = hidden_channels[i]
            if i < len(hidden_channels) - 1:
                out_ch = hidden_channels[i+1]
            else:
                out_ch = out_channels
            use_1x1conv = (in_ch != out_ch)
            stride = 2 if (in_ch != out_ch) else 1
            res_blocks.append(
                nn.Sequential(
                    ResNeXtBlock(in_ch, in_ch, 16, 1, False, 1, activation, dropout),
                    ResNeXtBlock(in_ch, in_ch, 16, 1, False, 1, activation, dropout),
                    ResNeXtBlock(in_ch, out_ch, 16, 1, use_1x1conv, stride, activation, dropout)
                )
            )
        self.res_blocks = nn.ModuleList(res_blocks)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky-relu":
            self.activation = nn.LeakyReLU()
        elif activation.startswith("leaky-relu"):
            neg_slope = float(activation.split("-")[-1])
            self.activation = nn.LeakyReLU(neg_slope)
        else:
            raise NotImplementedError()
        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels*2*4, out_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )
        if mode == "pretraining":
            self.proj_head = nn.Linear(out_dim, embed_dim)
        else:
            self.proj_head = nn.Identity()
    
    def forward(self, x):
        h = self.image_proj(x)
        for block in self.res_blocks:
            h = block(h)
        h = self.final(h)
        h = self.proj_head(h)
        return h