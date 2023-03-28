from .components import *
from .modules import *
from climate_learn.data import IterDataModule, DataModule
from climate_learn.data.tasks.args import ForecastingArgs
from climate_learn.utils.datetime import Hours
import torch.nn as nn
import torch


def load_model(name, task, model_kwargs, optim_kwargs):
    if name == "vit":
        model_cls = VisionTransformer
    elif name == "resnet":
        model_cls = ResNet
    elif name == "unet":
        model_cls = Unet
    elif name.startswith("weap"):
        model_cls = UnetEncoder

    model = model_cls(**model_kwargs)
    
    if name.startswith("weap"):
        model = WEAPEncoder(model)
        if name == "weap.z500":
            path = "../capstone/encoders/z500_epoch12.pt"
        elif name == "weap.t850":
            path = "../capstone/encoders/t850_epoch12.pt"
        model.load_state_dict(torch.load(path))
        # for param in model.parameters():
        #     param.requires_grad = False
        model = WEAPForecast(model, UnetDecoder(**model_kwargs))

    if task == "forecasting":
        # if name == "weap.z500":
        #     path = "../capstone/z500/epoch_024.ckpt"
        # elif name == "weap.t850":
        #     path = "../capstone/t850/epoch_016.ckpt"
        # module = ForecastLitModule.load_from_checkpoint(path, model, **optim_kwargs)
        module = ForecastLitModule(model, **optim_kwargs)
    elif task == "downscaling":
        module = DownscaleLitModule(model, **optim_kwargs)
    else:
        raise NotImplementedError("Only support foreacasting and downscaling")

    return module


def set_climatology(model_module, data_module):
    normalization = data_module.get_out_transforms()
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    model_module.set_denormalization(mean_denorm, std_denorm)
    model_module.set_lat_lon(*data_module.get_lat_lon())
    if isinstance(data_module, IterDataModule):
        model_module.set_pred_range(data_module.hparams.pred_range)
    elif isinstance(data_module, DataModule):
        if isinstance(
            data_module.hparams.data_module_args.train_task_args, ForecastingArgs
        ):
            model_module.set_pred_range(
                Hours(data_module.hparams.data_module_args.train_task_args.pred_range)
            )
        else:
            model_module.set_pred_range(Hours(1))
    model_module.set_train_climatology(data_module.get_climatology(split="train"))
    model_module.set_val_climatology(data_module.get_climatology(split="val"))
    model_module.set_test_climatology(data_module.get_climatology(split="test"))


def fit_lin_reg_baseline(model_module, data_module, reg_hparam=1.0):
    model_module.fit_lin_reg_baseline(data_module.train_dataset, reg_hparam)


class WEAPEncoder(nn.Module):
    def __init__(self, ftr_extractor, device="cpu"):
        super().__init__()
        self.ftr_extractor = ftr_extractor.to(device)
        self.linear = nn.Linear(1024*4*8, 1024, device=device)

    def forward(self, x):
        x = self.ftr_extractor.predict(x)[1]
        x = torch.flatten(x, start_dim=1)
        return self.linear(x)
    

class WEAPForecast(nn.Module):
    def __init__(self, encoder, decoder, device="cpu"):
        super().__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        
    def predict(self, x):
        h, x = self.encoder.ftr_extractor.predict(x)
        pred = self.decoder.predict(x, h)
        return pred
    
    def forward(self, x, y, out_variables, metric, lat, log_postfix):
        pred = self.predict(x)
        return ([
            m(pred, y, out_variables, lat=lat, log_postfix=log_postfix)
            for m in metric
        ], x)
    
    def evaluate(
        self, x, y, variables, out_variables, transform, metrics, lat, clim, log_postfix
    ):
        pred = self.predict(x)
        return ([
            m(pred, y, transform, out_variables, lat, clim, log_postfix)
            for m in metrics
        ], pred)
