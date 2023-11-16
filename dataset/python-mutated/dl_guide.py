MOCK = True
import os
import tempfile
from typing import Dict, Optional
import torch
import ray
from ray import train
from ray.train import Checkpoint
from ray.train.torch import TorchTrainer

def get_datasets() -> Dict[str, ray.data.Dataset]:
    if False:
        for i in range(10):
            print('nop')
    return {'train': ray.data.from_items([{'x': i, 'y': 2 * i} for i in range(10)])}

def train_loop_per_worker(config: dict):
    if False:
        for i in range(10):
            print('nop')
    from torchvision.models import resnet18
    model = resnet18()
    checkpoint: Optional[Checkpoint] = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            model_state_dict = torch.load(os.path.join(checkpoint_dir, 'model.pt'))
            model.load_state_dict(model_state_dict)
    model = train.torch.prepare_model(model)
    train_ds = train.get_dataset_shard('train')
    for epoch in range(5):
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(model.module.state_dict(), os.path.join(tmpdir, 'model.pt'))
            train.report({'epoch': epoch}, checkpoint=Checkpoint.from_directory(tmpdir))
trainer = TorchTrainer(train_loop_per_worker=train_loop_per_worker, datasets=get_datasets(), scaling_config=train.ScalingConfig(num_workers=2), run_config=train.RunConfig(name='dl_trainer_restore', storage_path=os.path.expanduser('~/ray_results')))
result = trainer.fit()
from ray.train.torch import TorchTrainer
restored_trainer = TorchTrainer.restore(path=os.path.expanduser('~/ray_results/dl_trainer_restore'), datasets=get_datasets())
if not MOCK:
    original_trainer = TorchTrainer(run_config=train.RunConfig(storage_path='s3://results-bucket', name='dl_trainer_restore'))
    result = trainer.fit()
    restored_trainer = TorchTrainer.restore('s3://results-bucket/dl_trainer_restore', datasets=get_datasets())
experiment_path = os.path.expanduser('~/ray_results/dl_restore_autoresume')
if TorchTrainer.can_restore(experiment_path):
    trainer = TorchTrainer.restore(experiment_path, datasets=get_datasets())
    result = trainer.fit()
else:
    trainer = TorchTrainer(train_loop_per_worker=train_loop_per_worker, datasets=get_datasets(), scaling_config=train.ScalingConfig(num_workers=2), run_config=train.RunConfig(storage_path=os.path.expanduser('~/ray_results'), name='dl_restore_autoresume'))
result = trainer.fit()