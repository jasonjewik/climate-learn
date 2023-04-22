import torch
from torch import nn
from .cnn_blocks import PeriodicConv2D, ResidualBlock, ResNeXtBlock

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


class ResNeXtEncoder(nn.Module):
    def __init__(self,
                 filters=[64,128,256],
                 downsample=[True,True,True],
                 out_channels=512,
                 proj_dim=512,
                 embed_dim=512,
                 dropout=0.3,
                 activation="leaky-relu",
                 mode="pretraining"
                ):
        super().__init__()
        assert len(filters) == len(downsample)
        self.image_proj = nn.Conv2d(1, filters[0], 7, 2, 3)
        res_blocks = {}
        for i in range(len(filters)):
            in_ch = filters[i]
            out_ch = filters[i+1] if i < len(filters)-1 else out_channels
            res_mode = "downsample" if downsample[i] else "preserve"
            res_blocks[f"res{i}"] = nn.Sequential(
                ResNeXtBlock(
                    in_ch,
                    in_ch,
                    proj_input=True,
                    activation=activation,
                    dropout=dropout
                ),
                ResNeXtBlock(
                    in_ch,
                    in_ch,
                    activation=activation,
                    dropout=dropout
                ),
                ResNeXtBlock(
                    in_ch,
                    out_ch, 
                    proj_input=True,
                    activation=activation,
                    dropout=dropout,
                    mode=res_mode
                )
            )
        self.res_blocks = nn.ModuleDict(res_blocks)
        self.num_res_blocks = len(filters)
        ftr_proj = nn.Sequential(
            nn.Conv2d(out_channels, proj_dim, 1),
            nn.BatchNorm2d(proj_dim)
        )
        num_downsample = downsample.count(True) + 1
        out_img_height = 32 // (2**num_downsample)
        out_img_width = 64 // (2**num_downsample)
        linear_in_dim = proj_dim * out_img_height * out_img_width
        mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_in_dim, embed_dim)
        )
        self.proj_head = nn.Sequential(ftr_proj, mlp)
        self.final = self.proj_head
        self.mode = mode
    
    def forward(self, x):
        h = self.image_proj(x)
        for i in range(self.num_res_blocks):
            block = self.res_blocks[f"res{i}"]
            h = block(h)
        if self.mode == "pretraining":
            h = self.final(h)
        return h
    

class ResNeXtDecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 filters=[512,512,512],
                 upsample=[True,True,True],
                 dropout=0.3,
                 activation="leaky-relu",
                ):
        super().__init__()
        assert len(filters) == len(upsample)
        self.ftr_proj = nn.ConvTranspose2d(in_channels, filters[0], 4, 2, 1)
        res_blocks = {}
        for i in range(len(filters)):
            in_ch = filters[i]
            out_ch = filters[i+1] if i < len(filters)-1 else out_channels
            res_mode = "upsample" if upsample[i] else "preserve"
            res_blocks[f"res{i}"] = nn.Sequential(
                ResNeXtBlock(
                    in_ch,
                    in_ch,
                    proj_input=True,
                    activation=activation,
                    dropout=dropout
                ),
                ResNeXtBlock(
                    in_ch,
                    in_ch,
                    activation=activation,
                    dropout=dropout
                ),
                ResNeXtBlock(
                    in_ch,
                    out_ch, 
                    proj_input=True,
                    activation=activation,
                    dropout=dropout,
                    mode=res_mode
                )
            )
        self.res_blocks = nn.ModuleDict(res_blocks)
        self.num_res_blocks = len(filters)        
    
    def forward(self, x):
        h = self.ftr_proj(x)
        for i in range(self.num_res_blocks):
            block = self.res_blocks[f"res{i}"]
            h = block(h)
        return h
    

class ResNeXtForecaster(nn.Module):
    def __init__(self, encoders, decoder, freeze_encoders=True):
        super().__init__()
        if freeze_encoders:
            for enc in encoders:
                for param in enc.parameters():
                    param.requires_grad = False
        self.encoders = encoders
        self.decoder = decoder

    def predict(self, x):
        if len(x.shape) == 4:
            h = self.get_embeddings(x)
        elif len(x.shape) == 5:
            embeds = []
            for i in range(x.shape[1]):
                embeds.append(self.get_embeddings(x[:,i]))
            h = torch.stack(embeds, 1)
            h = h.flatten(1, 2)
        yhat = self.decoder(h)
        return yhat
    
    def get_embeddings(self, x):
        embeds = []
        for i, enc in enumerate(self.encoders):
            embed = enc(x[:,i].unsqueeze(1))
            embeds.append(embed)
        h = torch.cat(embeds, 1)
        return h
        
    def forward(self, x, y, out_variables, metric, lat, log_postfix):
        pred = self.predict(x)
        return [
            m(pred, y, out_variables, lat=lat, log_postfix=log_postfix)
            for m in metric
        ], x
    
    def evaluate(
        self, x, y, variables, out_variables, transform, metrics, lat, clim, log_postfix
    ):
        pred = self.predict(x)
        return [
            m(pred, y, transform, out_variables, lat, clim, log_postfix)
            for m in metrics
        ], pred
