import numpy as np
from ray import train, tune
from ray.train import ScalingConfig

def train_func(config):
    if False:
        return 10
    np.random.seed(config['seed'])
    random_result = np.random.uniform(0, 100, size=1).item()
    train.report({'result': random_result})
np.random.seed(1234)
tuner = tune.Tuner(train_func, tune_config=tune.TuneConfig(num_samples=10, search_alg=tune.search.BasicVariantGenerator()), param_space={'seed': tune.randint(0, 1000)})
tuner.fit()
config = {'a': {'x': tune.uniform(0, 10)}, 'b': tune.choice([1, 2, 3])}
config = {'a': tune.randint(5, 10), 'b': tune.sample_from(lambda spec: np.random.randint(0, spec.config.a))}

def _iter():
    if False:
        print('Hello World!')
    for a in range(5, 10):
        for b in range(a):
            yield (a, b)
config = {'ab': tune.grid_search(list(_iter()))}

def train_func(config):
    if False:
        return 10
    random_result = np.random.uniform(0, 100, size=1).item()
    train.report({'result': random_result})
train_fn = train_func
MOCK = True
if not MOCK:
    tuner = tune.Tuner(tune.with_resources(train_fn, resources={'cpu': 2, 'gpu': 0.5, 'custom_resources': {'hdd': 80}}))
    tuner.fit()
    tuner = tune.Tuner(tune.with_resources(train_fn, resources=tune.PlacementGroupFactory([{'CPU': 2, 'GPU': 0.5, 'hdd': 80}, {'CPU': 1}, {'CPU': 1}], strategy='PACK')))
    tuner.fit()
    tuner = tune.Tuner(tune.with_resources(train_fn, resources=ScalingConfig(trainer_resources={'CPU': 2, 'GPU': 0.5, 'hdd': 80}, num_workers=2, resources_per_worker={'CPU': 1})))
    tuner.fit()
    tuner = tune.Tuner(tune.with_resources(train_fn, resources=lambda config: {'GPU': 1} if config['use_gpu'] else {'GPU': 0}), param_space={'use_gpu': True})
    tuner.fit()
    metric = None

    def train_fn(config):
        if False:
            return 10
        train.report({'metric': metric})
    tuner = tune.Tuner(tune.with_resources(train_fn, resources=tune.PlacementGroupFactory([{'CPU': 1}, {'CPU': 1}], strategy='PACK')))
    tuner.fit()
from ray import tune
import numpy as np

def train_func(config, num_epochs=5, data=None):
    if False:
        for i in range(10):
            print('nop')
    for i in range(num_epochs):
        for sample in data:
            pass
data = np.random.random(size=100000000)
tuner = tune.Tuner(tune.with_parameters(train_func, num_epochs=5, data=data))
tuner.fit()
import random
random.seed(1234)
output = [random.randint(0, 100) for _ in range(10)]
assert output == [99, 56, 14, 0, 11, 74, 4, 85, 88, 10]
import random
import numpy as np
random.seed(1234)
np.random.seed(5678)
import torch
torch.manual_seed(0)
import tensorflow as tf
tf.random.set_seed(0)
import random
import numpy as np
from ray import tune

def trainable(config):
    if False:
        return 10
    random.seed(config['seed'])
    np.random.seed(config['seed'])
config = {'seed': tune.randint(0, 10000)}
if __name__ == '__main__':
    random.seed(1234)
    np.random.seed(1234)
    tuner = tune.Tuner(trainable, param_space=config)
    tuner.fit()
from ray import train, tune
import numpy as np

def f(config, data=None):
    if False:
        while True:
            i = 10
    pass
data = np.random.random(size=100000000)
tuner = tune.Tuner(tune.with_parameters(f, data=data))
tuner.fit()
MyTrainableClass = None
if not MOCK:
    tuner = tune.Tuner(MyTrainableClass, run_config=train.RunConfig(storage_path='s3://my-log-dir'))
    tuner.fit()
if not MOCK:
    from ray import tune
    tuner = tune.Tuner(train_fn, run_config=train.RunConfig(storage_path='s3://your-s3-bucket/durable-trial/'))
    tuner.fit()
    from ray import train, tune
    tuner = tune.Tuner(train_fn, run_config=train.RunConfig(storage_path='/path/to/shared/storage'))
    tuner.fit()
import ray
ray.shutdown()
parameters = {'qux': tune.sample_from(lambda spec: 2 + 2), 'bar': tune.grid_search([True, False]), 'foo': tune.grid_search([1, 2, 3]), 'baz': 'asd'}
tuner = tune.Tuner(train_fn, param_space=parameters)
tuner.fit()
tuner = tune.Tuner(train_fn, run_config=train.RunConfig(name='my_trainable'), param_space={'alpha': tune.uniform(100, 200), 'beta': tune.sample_from(lambda spec: spec.config.alpha * np.random.normal()), 'nn_layers': [tune.grid_search([16, 64, 256]), tune.grid_search([16, 64, 256])]}, tune_config=tune.TuneConfig(num_samples=10))
if not MOCK:
    import os
    from pathlib import Path

    def train_func(config):
        if False:
            print('Hello World!')
        print(open('./read.txt').read())
        assert os.getcwd() == os.environ['TUNE_ORIG_WORKING_DIR']
        tune_trial_dir = Path(train.get_context().get_trial_dir())
        with open(tune_trial_dir / 'write.txt', 'w') as f:
            f.write('trial saved artifact')
    os.environ['RAY_CHDIR_TO_TRIAL_DIR'] = '0'
    tuner = tune.Tuner(train_func)
    tuner.fit()
import os
import tempfile
import torch
from ray import train, tune
from ray.train import Checkpoint
import random

def trainable(config):
    if False:
        for i in range(10):
            print('nop')
    for epoch in range(1, config['num_epochs']):
        with tempfile.TemporaryDirectory() as tempdir:
            torch.save({'model_state_dict': {'x': 1}}, os.path.join(tempdir, 'model.pt'))
            train.report({'score': random.random()}, checkpoint=Checkpoint.from_directory(tempdir))
tuner = tune.Tuner(trainable, param_space={'num_epochs': 10, 'hyperparam': tune.grid_search([1, 2, 3])}, tune_config=tune.TuneConfig(metric='score', mode='max'))
result_grid = tuner.fit()
best_result = result_grid.get_best_result()
best_checkpoint = best_result.checkpoint
import ray

def trainable(config):
    if False:
        i = 10
        return i + 15
    checkpoint: Checkpoint = config['start_from_checkpoint']
    with checkpoint.as_directory() as checkpoint_dir:
        model_state_dict = torch.load(os.path.join(checkpoint_dir, 'model.pt'))
    for epoch in range(1, config['num_epochs']):
        ...
        train.report({'score': random.random()})
new_tuner = tune.Tuner(trainable, param_space={'num_epochs': 10, 'hyperparam': tune.grid_search([4, 5, 6]), 'start_from_checkpoint': best_checkpoint}, tune_config=tune.TuneConfig(metric='score', mode='max'))
result_grid = new_tuner.fit()