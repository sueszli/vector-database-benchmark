import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
X = torch.randn(128, 3)
y = torch.randint(0, 2, (128,))
dataset = TensorDataset(X, y)
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class DummyModel(pl.LightningModule):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.layer = torch.nn.Linear(3, 1)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        if False:
            return 10
        (x, y) = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat.flatten(), y.float())
        self.log('train_loss', loss)
        self.log_dict({'metric_1': 1 / (batch_idx + 1), 'metric_2': batch_idx * 100})
        return loss

    def configure_optimizers(self):
        if False:
            while True:
                i = 10
        return torch.optim.Adam(self.parameters(), lr=0.001)