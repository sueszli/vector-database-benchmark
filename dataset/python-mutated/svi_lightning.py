import argparse
import lightning.pytorch as pl
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.nn import PyroModule

class Model(PyroModule):

    def __init__(self, size):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.size = size

    def forward(self, covariates, data=None):
        if False:
            while True:
                i = 10
        coeff = pyro.sample('coeff', dist.Normal(0, 1))
        bias = pyro.sample('bias', dist.Normal(0, 1))
        scale = pyro.sample('scale', dist.LogNormal(0, 1))
        with pyro.plate('data', self.size, len(covariates)):
            loc = bias + coeff * covariates
            return pyro.sample('obs', dist.Normal(loc, scale), obs=data)

class PyroLightningModule(pl.LightningModule):

    def __init__(self, loss_fn: pyro.infer.elbo.ELBOModule, lr: float):
        if False:
            while True:
                i = 10
        super().__init__()
        self.loss_fn = loss_fn
        self.model = loss_fn.model
        self.guide = loss_fn.guide
        self.lr = lr
        self.predictive = pyro.infer.Predictive(self.model, guide=self.guide, num_samples=1)

    def forward(self, *args):
        if False:
            return 10
        return self.predictive(*args)

    def training_step(self, batch, batch_idx):
        if False:
            while True:
                i = 10
        'Training step for Pyro training.'
        loss = self.loss_fn(*batch)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        if False:
            for i in range(10):
                print('nop')
        'Configure an optimizer.'
        return torch.optim.Adam(self.loss_fn.parameters(), lr=self.lr)

def main(args):
    if False:
        return 10
    pyro.set_rng_seed(args.seed)
    pyro.settings.set(module_local_params=True)
    model = Model(args.size)
    covariates = torch.randn(args.size)
    data = model(covariates)
    guide = AutoNormal(model)
    loss_fn = Trace_ELBO()(model, guide)
    training_plan = PyroLightningModule(loss_fn, args.learning_rate)
    dataset = torch.utils.data.TensorDataset(covariates, data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    mini_batch = dataset[:args.batch_size]
    loss_fn(*mini_batch)
    trainer = pl.Trainer(accelerator=args.accelerator, strategy=args.strategy, devices=args.devices, max_epochs=args.max_epochs)
    trainer.fit(training_plan, train_dataloaders=dataloader)
if __name__ == '__main__':
    assert pyro.__version__.startswith('1.8.6')
    parser = argparse.ArgumentParser(description='Distributed training via PyTorch Lightning')
    parser.add_argument('--size', default=1000000, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--seed', default=20200723, type=int)
    parser.add_argument('--accelerator', default='auto')
    parser.add_argument('--strategy', default='auto')
    parser.add_argument('--devices', default='auto')
    parser.add_argument('--max_epochs', default=None)
    args = parser.parse_args()
    main(args)