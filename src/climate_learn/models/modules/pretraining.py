from .utils.lr_scheduler import LinearWarmupCosineAnnealingLR

import numpy as np
import pytorch_lightning as pl
import torch
from torchvision.transforms import transforms
import torch.nn.functional as F

class PretrainingLitModule(pl.LightningModule):
    def __init__(
        self,
        net1,
        net2,
        optimizer=torch.optim.Adam,
        lr=5e-4,
        weight_decay=0.2,
        warmup_epochs=5,
        max_epochs=100,
        warmup_start_lr=1e-8,
        eta_min=1e-8,
        init_temp=np.log(0.07),
        max_temp=np.log(100),
        temp_lr=1e-2,
        logit_scaling=True
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net1", "net2"])
        self.net1 = net1
        self.net2 = net2
        self.optim_cls = optimizer
        self.logit_scaling = logit_scaling
        self.temperature = torch.tensor([init_temp], requires_grad=True)
        self.max_temp = torch.tensor([max_temp], requires_grad=False)
        self.temp_lr = temp_lr

    def forward(self, x):
        enc1 = F.normalize(self.net1(x[:,:,0,:,:]))
        enc2 = F.normalize(self.net2(x[:,:,1,:,:]))
        return torch.stack((enc1, enc2))
    
    def configure_optimizers(self):
        params = [
            {
                "params": self.net1.parameters(),
                "lr": self.hparams.lr,
                "weight_decay": self.hparams.weight_decay
            },
            {
                "params": self.net2.parameters(),
                "lr": self.hparams.lr,
                "weight_decay": self.hparams.weight_decay
            },
            {
                "params": self.temperature,
                "lr": self.temp_lr
            }
        ]
        optimizer = self.optim_cls(params)

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs,
            self.hparams.max_epochs,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    
    def training_step(self, batch, batch_idx):
        x = batch[0]
        encodings = self(x)
        if self.logit_scaling:
            logit_scale = torch.exp(torch.min(
                self.temperature,
                self.max_temp
            ))
        else:
            logit_scale = 1
        logits = logit_scale * torch.stack((
            torch.matmul(encodings[0], encodings[1].T),
            torch.matmul(encodings[1], encodings[0].T)
        ))
        labels = torch.arange(x.shape[0], device=self.device)
        loss = (
            F.cross_entropy(logits[0], labels)
            + F.cross_entropy(logits[1], labels)
        ) / 2
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=len(x)
        )
        self.log(
            "temperature",
            self.temperature,
            on_step=True,
            on_epoch=False
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[0]
        encodings = self(x)
        if self.logit_scaling:
            logit_scale = torch.exp(torch.min(
                self.temperature,
                self.max_temp
            ))
        else:
            logit_scale = 1
        logits = logit_scale * torch.stack((
            torch.matmul(encodings[0], encodings[1].T),
            torch.matmul(encodings[1], encodings[0].T)
        ))
        labels = torch.arange(x.shape[0], device=self.device)
        loss = (
            F.cross_entropy(logits[0], labels)
            + F.cross_entropy(logits[1], labels)
        ) / 2
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=len(x)
        )
        self.log(
            "temperature",
            self.temperature,
            on_step=False,
            on_epoch=True
        )
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch[0]
        encodings = self(x)
        if self.logit_scaling:
            logit_scale = torch.exp(torch.min(
                self.temperature,
                self.max_temp
            ))
        else:
            logit_scale = 1
        logits = logit_scale * torch.stack((
            torch.matmul(encodings[0], encodings[1].T),
            torch.matmul(encodings[1], encodings[0].T)
        ))
        labels = torch.arange(x.shape[0], device=self.device)
        loss = (
            F.cross_entropy(logits[0], labels)
            + F.cross_entropy(logits[1], labels)
        ) / 2
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(x)
        )
        self.log(
            "temperature",
            self.temperature,
            on_step=False,
            on_epoch=True
        )
        return loss

    def on_train_start(self):
        self.temperature = self.temperature.to(device=self.device)
        self.max_temp = self.max_temp.to(device=self.device)

    def on_validation_start(self):
        self.temperature = self.temperature.to(device=self.device)
        self.max_temp = self.max_temp.to(device=self.device)

    def on_test_start(self):
        self.temperature = self.temperature.to(device=self.device)
        self.max_temp = self.max_temp.to(device=self.device)
    
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


class PretrainingLitModule2(pl.LightningModule):
    def __init__(
        self,
        net,
        optimizer=torch.optim.Adam,
        lr=5e-4,
        weight_decay=0.2,
        warmup_epochs=5,
        max_epochs=100,
        warmup_start_lr=1e-8,
        eta_min=1e-8,
        init_temp=np.log(0.07),
        max_temp=np.log(100),
        temp_lr=1e-2,
        logit_scaling=True
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.optim_cls = optimizer
        self.logit_scaling = logit_scaling
        self.temperature = torch.tensor([init_temp], requires_grad=True)
        self.max_temp = torch.tensor([max_temp], requires_grad=False)
        self.temp_lr = temp_lr

    def forward(self, x):
        enc1 = F.normalize(self.net(x[:,:,0,:,:]))
        enc2 = F.normalize(self.net(x[:,:,1,:,:]))
        return torch.stack((enc1, enc2))
        
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
            "temperature",
            self.temperature,
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
            "temperature",
            self.temperature,
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
            "temperature",
            self.temperature,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch[0])
        )
        return loss
    
    def compute_loss(self, batch):
        x = batch[0]
        encodings = self(x)
        if self.logit_scaling:
            logit_scale = torch.exp(torch.min(
                self.temperature,
                self.max_temp
            ))
        else:
            logit_scale = 1
        logits = logit_scale * torch.stack((
            torch.matmul(encodings[0], encodings[1].T),
            torch.matmul(encodings[1], encodings[0].T)
        ))
        labels = torch.arange(x.shape[0], device=self.device)
        loss = (
            F.cross_entropy(logits[0], labels)
            + F.cross_entropy(logits[1], labels)
        ) / 2
        return loss
    
    def configure_optimizers(self):
        params = [
            {
                "params": self.net.parameters(),
                "lr": self.hparams.lr,
                "weight_decay": self.hparams.weight_decay
            },
            {
                "params": self.temperature,
                "lr": self.temp_lr
            }
        ]
        optimizer = self.optim_cls(params, betas=(0.9,0.99))

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs,
            self.hparams.max_epochs,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min
        )
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 10
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def on_train_start(self):
        self.temperature = self.temperature.to(device=self.device)
        self.max_temp = self.max_temp.to(device=self.device)

    def on_validation_start(self):
        self.temperature = self.temperature.to(device=self.device)
        self.max_temp = self.max_temp.to(device=self.device)

    def on_test_start(self):
        self.temperature = self.temperature.to(device=self.device)
        self.max_temp = self.max_temp.to(device=self.device)
    
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