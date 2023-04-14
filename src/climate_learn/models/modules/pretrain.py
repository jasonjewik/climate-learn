from .utils.lr_scheduler import LinearWarmupCosineAnnealingLR

import numpy as np
import pytorch_lightning as pl
import torch
from torchvision.transforms import transforms
import torch.nn.functional as F


class PretrainLitModule(pl.LightningModule):
    def __init__(
        self,
        net,
        net2=None,
        optimizer=torch.optim.Adam,
        betas=(0.9,0.999),
        lr=5e-4,
        weight_decay=0.2,
        warmup_epochs=5,
        max_epochs=100,
        warmup_start_lr=1e-8,
        eta_min=1e-8,
        logit_temp=np.log(0.07),
        max_logit_temp=np.log(100),
        logit_temp_lr=1e-3,
        logit_scaling=True,
        loss="clip"
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net", "net2", "optimizer"])
        self.net = net
        self.net2 = net2
        self.optim_cls = optimizer
        self.logit_temp = torch.tensor([logit_temp], requires_grad=True)
        self.max_logit_temp = torch.tensor([max_logit_temp], requires_grad=False)
        if loss not in ("clip", "cyclip", "time-clip", "time-cyclip"):
            raise NotImplementedError(
                f"loss {loss} not supported"
            )
        self.loss = loss

    def forward(self, x):
        x0, x1 = x[:,0].unsqueeze(1), x[:,1].unsqueeze(1)
        enc1 = F.normalize(self.net(x0))
        if self.net2:
            enc2 = F.normalize(self.net2(x1))
        else:
            enc2 = F.normalize(self.net(x1))
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
            logit_scale = torch.exp(logit_temp)
        else:
            logit_scale = 1
        # Compute logits
        encodings = self(x)
        cross_modal_logits = logit_scale * torch.stack((
            torch.matmul(encodings[0], encodings[1].T),
            torch.matmul(encodings[1], encodings[0].T)
        ))
        in_modal_logits = logit_scale * torch.stack((
            torch.matmul(encodings[0], encodings[0].T),
            torch.matmul(encodings[1], encodings[1].T)
        ))
        # Compute loss
        if self.loss in ("clip", "cyclip"):
            # Standard CLIP labels
            labels = torch.eye(x.shape[0]).to(device=self.device)
        elif self.loss in ("time-clip", "time-cyclip"):
            # Labels follow a normal distribution centered at 0, 
            # scaled so that at 0, the label is ~1,
            # and by +-30, the label is ~0
            t = times.clone().repeat((x.shape[0], 1))
            t_delta = t - t.T
            alpha, sigma = 20, 8
            labels = (
                alpha / (sigma * torch.sqrt(
                    torch.tensor([2 * torch.pi], device=self.device)
                )) * torch.exp(-0.5 * torch.square(t_delta / sigma))
            )
        clip_loss = (
            F.cross_entropy(cross_modal_logits[0], labels)
            + F.cross_entropy(cross_modal_logits[1], labels)
        ) / 2
        loss = clip_loss
        if self.loss in ("cyclip", "time-cyclip"):
            # Standard CyCLIP loss
            cross_modal_loss = torch.mean(torch.square(
                cross_modal_logits[0] - cross_modal_logits[1]
            )) / 2
            in_modal_loss = torch.mean(torch.square(
                in_modal_logits[0] - in_modal_logits[1]
            )) / 2
            cyclip_loss = (cross_modal_loss + in_modal_loss) / 2
            loss += cyclip_loss   
        return loss
    
    def configure_optimizers(self):
        params = [
            {
                "params": self.net.parameters(),
                "lr": self.hparams.lr,
                "weight_decay": self.hparams.weight_decay
            },
            {
                "params": self.logit_temp,
                "lr": self.hparams.logit_temp_lr
            }
        ]
        adam_opts = (torch.optim.Adam, torch.optim.Adamax, torch.optim.AdamW)
        if self.optim_cls in adam_opts:
            optimizer = self.optim_cls(params, betas=self.hparams.betas)
        else:
            optimizer = self.optim_cls(params)
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