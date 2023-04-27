import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np


class RetrievalLitModule(pl.LightningModule):
    def __init__(self, nets):
        super().__init__()
        

    def forward(self, x):
        yhat0 = self.net(x[:,0].unsqueeze(1))
        if self.net2 is None:
            yhat1 = self.net(x[:,1].unsqueeze(1))
        else:
            yhat1 = self.net2(x[:,1].unsqueeze(1))
        yhat0_norm = F.normalize(yhat0)
        yhat1_norm = F.normalize(yhat1)
        return torch.stack((yhat0_norm, yhat1_norm))
    
    def test_step(self, batch, batch_idx):
        top_acc = [1, 3, 5, 10]
        ab_counts, ba_counts = self.compute_loss(batch, top_acc)
        for i, (ab_loss, ba_loss) in enumerate(zip(ab_counts, ba_counts)):
            self.log(
                f"ab_top{top_acc[i]}",
                ab_loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=len(batch[0])
            )
            self.log(
                f"ba_top{top_acc[i]}",
                ba_loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=len(batch[0])
            )
    
    def compute_loss(self, batch, top_acc):
        x, times = batch[0], batch[1].flatten()
        encodings = self(x)
        a, b = encodings[0], encodings[1]
        ab_counts = [0, 0, 0, 0]
        ba_counts = [0, 0, 0, 0]
        top_acc = [1, 3, 5, 10]
        for i in range(x.shape[0]):
            ab_ranking = torch.argsort(torch.mean(torch.square(a[i] - b), dim=1))
            ba_ranking = torch.argsort(torch.mean(torch.square(b[i] - a), dim=1))
            for j, acc in enumerate(top_acc):
                if i in ab_ranking[:acc]:
                    ab_counts[j] += 1
                if i in ba_ranking[:acc]:
                    ba_counts[j] += 1
        
        ab_counts = np.array(ab_counts) / x.shape[0]
        ba_counts = np.array(ba_counts) / x.shape[0]
        return ab_counts, ba_counts
        