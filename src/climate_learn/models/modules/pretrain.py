from .utils.lr_scheduler import LinearWarmupCosineAnnealingLR

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.transforms import transforms
import torch.nn.functional as F


class PretrainLitModule(pl.LightningModule):
    def __init__(
        self,
        net,
        net2=None,
        optimizer=torch.optim.Adam,
        betas=(0.9,0.999),
        lr=0.01,
        weight_decay=0.1,
        warmup_epochs=5,
        max_epochs=100,
        warmup_start_lr=1e-8,
        eta_min=1e-8,
        logit_temp=torch.e,
        max_logit_temp=10,
        logit_temp_lr=0.1,
        logit_scaling=True,
        loss="clip"
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net", "net2", "optimizer"])
        self.net = net
        self.net2 = net2
        self.optim_cls = optimizer
        self.logit_temp = nn.parameter.Parameter(
            torch.tensor([logit_temp], requires_grad=True)
        )
        self.max_logit_temp = torch.tensor(
            [max_logit_temp],
            requires_grad=False
        )
        self.supported_losses = [
            "clip", "time-clip", "l1-time-clip", "softmax-time-clip",
            "cyclip", "time-cyclip", "l1-time-cyclip", "softmax-time-cyclip"
        ]
        if loss not in self.supported_losses:
            raise NotImplementedError(
                f"loss {loss} not supported"
            )
        self.loss = loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        yhat0 = self.net(x[:,0].unsqueeze(1))
        if self.net2 is None:
            yhat1 = self.net(x[:,1].unsqueeze(1))
        else:
            yhat1 = self.net2(x[:,1].unsqueeze(1))
        yhat0_norm = F.normalize(yhat0)
        yhat1_norm = F.normalize(yhat1)
        return torch.stack((yhat0_norm, yhat1_norm))
        
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=len(batch[0])
        )
        self.log(
            "logit_temp",
            self.logit_temp,
            on_step=True,
            on_epoch=False,
            batch_size=len(batch[0])
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=len(batch[0])
        )
        self.log(
            "logit_temp",
            self.logit_temp,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch[0])
        )
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch[0])
        )
        self.log(
            "logit_temp",
            self.logit_temp,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch[0])
        )
        return loss
    
    def compute_loss(self, batch):
        x, times = batch[0], batch[1].flatten()
        # Logit scaling
        if self.hparams.logit_scaling:
            logit_temp = self.logit_temp.clamp(max=self.max_logit_temp)
            logit_scale = logit_temp.exp()
        else:
            logit_scale = 1
        # Compute logits
        embeddings = self(x)
        cross_modal_logits = logit_scale * (embeddings[0] @ embeddings[1].T)
        in_modal_logits = logit_scale * torch.stack((
            embeddings[0] @ embeddings[0].T,
            embeddings[1] @ embeddings[1].T
        ))
        # Compute loss
        if self.loss in ("clip", "cyclip"):
            # Standard CLIP labels
            labels = torch.arange(x.shape[0]).to(device=self.device)
        elif "time" in self.loss:
            # Labels follow a normal distribution centered at 0, 
            # scaled so that at 0, the label is ~1,
            # and by +-30, the label is ~0
            t = times.clone().repeat((x.shape[0], 1))
            t_delta = t - t.T
            alpha, sigma = 20, 8
            k = alpha / (sigma * torch.sqrt(
                torch.tensor([2 * torch.pi], device=self.device)
            ))
            labels = k * torch.exp(-0.5 * torch.square(t_delta / sigma))
            # L1 normalization along rows
            if "l1" in self.loss:
                labels = F.normalize(labels, p=1, dim=1)                
            # Softmax along rows
            elif "softmax" in self.loss:
                labels = F.softmax(labels, dim=1)
            # Floor values close to 0
            labels = torch.where(labels > 1e-8, labels, 0)
        loss = (
            self.criterion(cross_modal_logits, labels)
            + self.criterion(cross_modal_logits.T, labels)
        ) / 2
        if "cyclip" in self.loss:
            # Unscaled CyCLIP loss
            cross_modal_loss = torch.mean(torch.square(
                cross_modal_logits[0] - cross_modal_logits[1]
            )) / 2
            in_modal_loss = torch.mean(torch.square(
                in_modal_logits[0] - in_modal_logits[1]
            )) / 2
            loss += (cross_modal_loss + in_modal_loss) / 2
        return loss
    
    def configure_optimizers(self):
        parameters = []
        parameters.append({
            "params": self.net.parameters(),
            "lr": self.hparams.lr,
            "weight_decay": self.hparams.weight_decay
        })
        if self.net2:
            parameters.append({
                "params": self.net2.parameters(),
                "lr": self.hparams.lr,
                "weight_decay": self.hparams.weight_decay
            })
        parameters.append({
            "params": self.logit_temp,
            "lr": self.hparams.logit_temp_lr
        })
        adam_opts = (torch.optim.Adam, torch.optim.Adamax, torch.optim.AdamW)
        if self.optim_cls in adam_opts:
            optimizer = self.optim_cls(parameters, betas=self.hparams.betas)
        else:
            optimizer = self.optim_cls(parameters)
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs,
            self.hparams.max_epochs,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def on_train_start(self):
        self.send_tensors_to_device()

    def on_validation_start(self):
        self.send_tensors_to_device()

    def on_test_start(self):
        self.send_tensors_to_device()
    
    def set_denormalization(self, mean, std):
        self.denormalization = transforms.Normalize(mean, std)

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def set_pred_range(self, r):
        self.pred_range = r

    def set_train_climatology(self, clim):
        self.train_clim = clim

    def set_val_climatology(self, clim):
        self.val_clim = clim

    def set_test_climatology(self, clim):
        self.test_clim = clim

    def send_tensors_to_device(self):
        self.logit_temp = self.logit_temp.to(device=self.device)
        self.max_logit_temp = self.max_logit_temp.to(device=self.device)