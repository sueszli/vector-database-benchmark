import json
import os
import tempfile
from ray import train, tune
from ray.train import Checkpoint

def trainable(config):
    if False:
        print('Hello World!')
    checkpoint = train.get_checkpoint()
    start = 1
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            with open(os.path.join(checkpoint_dir, 'checkpoint.json'), 'r') as f:
                state = json.load(f)
        start = state['epoch'] + 1
    for epoch in range(start, config['num_epochs']):
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            with open(os.path.join(temp_checkpoint_dir, 'checkpoint.json'), 'w') as f:
                json.dump({'epoch': epoch}, f)
            train.report({'epoch': epoch}, checkpoint=Checkpoint.from_directory(temp_checkpoint_dir))
tuner = tune.Tuner(trainable, param_space={'num_epochs': 10}, run_config=train.RunConfig(storage_path=os.path.expanduser('~/ray_results'), name='tune_fault_tolerance_guide'))
result_grid = tuner.fit()
assert not result_grid.errors
tuner = tune.Tuner.restore(os.path.expanduser('~/ray_results/tune_fault_tolerance_guide'), trainable=trainable, resume_errored=True)
tuner.fit()
tuner = tune.Tuner.restore(os.path.expanduser('~/ray_results/tune_fault_tolerance_guide'), trainable=trainable, resume_errored=True, restart_errored=False, resume_unfinished=True)
import os
from ray import train, tune
storage_path = os.path.expanduser('~/ray_results')
exp_name = 'tune_fault_tolerance_guide'
path = os.path.join(storage_path, exp_name)
if tune.Tuner.can_restore(path):
    tuner = tune.Tuner.restore(path, trainable=trainable, resume_errored=True)
else:
    tuner = tune.Tuner(trainable, param_space={'num_epochs': 10}, run_config=train.RunConfig(storage_path=storage_path, name=exp_name))
tuner.fit()
if tune.Tuner.can_restore(path):
    tuner = tune.Tuner.restore(path, trainable=trainable, resume_errored=True)
else:
    tuner = tune.Tuner(trainable, param_space={'num_epochs': 10}, run_config=train.RunConfig(storage_path=storage_path, name=exp_name))
assert tuner.get_results()
import ray
from ray import train, tune

class LargeModel:

    def __init__(self, model_id):
        if False:
            for i in range(10):
                print('nop')
        self.model_id = model_id

def train_fn(config):
    if False:
        for i in range(10):
            print('nop')
    model = ray.get(config['model_ref'])
    print(model.model_id)
model_refs = [ray.put(LargeModel(1)), ray.put(LargeModel(2))]
tuner = tune.Tuner(train_fn, param_space={'model_ref': tune.grid_search(model_refs)}, run_config=train.RunConfig(storage_path=os.path.expanduser('~/ray_results'), name='restore_object_refs'))
tuner.fit()
if ray.is_initialized():
    ray.shutdown()
param_space = {'model_ref': tune.grid_search([ray.put(LargeModel(1)), ray.put(LargeModel(2))])}
tuner = tune.Tuner.restore(os.path.expanduser('~/ray_results/restore_object_refs'), trainable=train_fn, param_space=param_space, resume_errored=True)
tuner.fit()
from ray import train, tune
tuner = tune.Tuner(trainable, param_space={'num_epochs': 10}, run_config=train.RunConfig(storage_path=os.path.expanduser('~/ray_results'), name='trial_fault_tolerance', failure_config=train.FailureConfig(max_failures=3)))
tuner.fit()