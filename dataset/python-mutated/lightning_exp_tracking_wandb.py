from lightning_exp_tracking_model_dl import DummyModel, dataloader
import os
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
import ray
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

def train_func(config):
    if False:
        for i in range(10):
            print('nop')
    logger = None
    if ray.train.get_context().get_world_rank() == 0:
        logger = WandbLogger(name='demo-run', project='demo-project')
    ptl_trainer = pl.Trainer(max_epochs=5, accelerator='cpu', logger=logger, log_every_n_steps=1)
    model = DummyModel()
    ptl_trainer.fit(model, train_dataloaders=dataloader)
    if ray.train.get_context().get_world_rank() == 0:
        wandb.finish()
scaling_config = ScalingConfig(num_workers=2, use_gpu=False)
assert 'WANDB_API_KEY' in os.environ, 'Please set WANDB_API_KEY="abcde" when running this script.'
ray.init(runtime_env={'env_vars': {'WANDB_API_KEY': os.environ['WANDB_API_KEY']}})
trainer = TorchTrainer(train_func, scaling_config=scaling_config)
trainer.fit()