from pathlib import Path
from ray import train
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune.logger import TBXLoggerCallback
from ray.tune.logger.mlflow import MLflowLoggerCallback

def train_func():
    if False:
        i = 10
        return i + 15
    for i in range(3):
        train.report(dict(epoch=i))
trainer = TorchTrainer(train_func, scaling_config=ScalingConfig(num_workers=2), run_config=RunConfig(callbacks=[MLflowLoggerCallback(experiment_name='train_experiment'), TBXLoggerCallback()]))
result = trainer.fit()
print('Run directory:', Path(result.path).parent)