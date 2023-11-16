import os
import torch
from torch import nn
from ray import train, tune

class MyTrainableClass(tune.Trainable):

    def setup(self, config):
        if False:
            for i in range(10):
                print('nop')
        self.model = nn.Sequential(nn.Linear(config.get('input_size', 32), 32), nn.ReLU(), nn.Linear(32, 10))

    def step(self):
        if False:
            return 10
        return {}

    def save_checkpoint(self, tmp_checkpoint_dir):
        if False:
            print('Hello World!')
        checkpoint_path = os.path.join(tmp_checkpoint_dir, 'model.pth')
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        if False:
            i = 10
            return i + 15
        checkpoint_path = os.path.join(tmp_checkpoint_dir, 'model.pth')
        self.model.load_state_dict(torch.load(checkpoint_path))
tuner = tune.Tuner(MyTrainableClass, param_space={'input_size': 64}, run_config=train.RunConfig(stop={'training_iteration': 2}, checkpoint_config=train.CheckpointConfig(checkpoint_frequency=2)))
tuner.fit()
import random

def detect_instance_preemption():
    if False:
        return 10
    choice = random.randint(1, 100)
    return choice <= 1

def train_func(self):
    if False:
        print('Hello World!')
    result = {'mean_accuracy': 'my_accuracy'}
    if detect_instance_preemption():
        result.update(should_checkpoint=True)
    return result
tuner = tune.Tuner(MyTrainableClass, run_config=train.RunConfig(stop={'training_iteration': 2}, checkpoint_config=train.CheckpointConfig(checkpoint_frequency=10)))
tuner.fit()
tuner = tune.Tuner(MyTrainableClass, run_config=train.RunConfig(stop={'training_iteration': 2}, checkpoint_config=train.CheckpointConfig(checkpoint_frequency=10, checkpoint_at_end=True)))
tuner.fit()

class MyModel:

    def state_dict(self) -> dict:
        if False:
            print('Hello World!')
        return {}

    def load_state_dict(self, state_dict):
        if False:
            print('Hello World!')
        pass
import os
import tempfile
from ray import train, tune
from ray.train import Checkpoint

def train_func(config):
    if False:
        i = 10
        return i + 15
    start = 1
    my_model = MyModel()
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'))
            start = checkpoint_dict['epoch'] + 1
            my_model.load_state_dict(checkpoint_dict['model_state'])
    for epoch in range(start, config['epochs'] + 1):
        metrics = {'metric': 1}
        with tempfile.TemporaryDirectory() as tempdir:
            torch.save({'epoch': epoch, 'model_state': my_model.state_dict()}, os.path.join(tempdir, 'checkpoint.pt'))
            train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))
tuner = tune.Tuner(train_func, param_space={'epochs': 5})
result_grid = tuner.fit()
assert not result_grid.errors
NUM_EPOCHS = 12
CHECKPOINT_FREQ = 3

def train_func(config):
    if False:
        return 10
    for epoch in range(1, config['epochs'] + 1):
        metrics = {'metric': 'my_metric'}
        if epoch % CHECKPOINT_FREQ == 0:
            with tempfile.TemporaryDirectory() as tempdir:
                train.report(metrics, checkpoint=Checkpoint.from_directory(tempdir))
        else:
            train.report(metrics)
tuner = tune.Tuner(train_func, param_space={'epochs': NUM_EPOCHS})
result_grid = tuner.fit()
assert not result_grid.errors
assert len(result_grid[0].best_checkpoints) == NUM_EPOCHS // CHECKPOINT_FREQ