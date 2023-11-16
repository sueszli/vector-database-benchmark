import json
import os
from typing import Dict, Union
import pytest
import ray
from ray import train, tune
from ray.train._internal.storage import StorageContext
from ray.tune.trainable import wrap_function
from ray.train.tests.util import create_dict_checkpoint

@pytest.fixture
def ray_start_2_cpus():
    if False:
        return 10
    address_info = ray.init(num_cpus=2)
    yield address_info
    ray.shutdown()

class SavingTrainable(tune.Trainable):

    def __init__(self, return_type: str, *args, **kwargs):
        if False:
            print('Hello World!')
        self.return_type = return_type
        super(SavingTrainable, self).__init__(*args, **kwargs)

    def step(self):
        if False:
            print('Hello World!')
        return {'iter': self.training_iteration}

    def save_checkpoint(self, tmp_checkpoint_dir: str):
        if False:
            i = 10
            return i + 15
        checkpoint_data = {'data': 1}
        if self.return_type == 'object':
            return checkpoint_data
        subdir = os.path.join(tmp_checkpoint_dir, 'subdir')
        os.makedirs(subdir, exist_ok=True)
        checkpoint_file = os.path.join(subdir, 'checkpoint.pkl')
        with open(checkpoint_file, 'w') as f:
            f.write(json.dumps(checkpoint_data))
        if self.return_type == 'root':
            return tmp_checkpoint_dir
        elif self.return_type == 'subdir':
            return subdir
        elif self.return_type == 'checkpoint':
            return checkpoint_file

    def load_checkpoint(self, checkpoint: Union[Dict, str]):
        if False:
            i = 10
            return i + 15
        if self.return_type == 'object':
            assert isinstance(checkpoint, dict)
            checkpoint_data = checkpoint
            checkpoint_file = None
        elif self.return_type == 'root':
            assert 'subdir' not in checkpoint
            checkpoint_file = os.path.join(checkpoint, 'subdir', 'checkpoint.pkl')
        elif self.return_type == 'subdir':
            assert 'subdir' in checkpoint
            assert 'checkpoint.pkl' not in checkpoint
            checkpoint_file = os.path.join(checkpoint, 'checkpoint.pkl')
        else:
            assert checkpoint.endswith('subdir/checkpoint.pkl')
            checkpoint_file = checkpoint
        if checkpoint_file:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = json.load(f)
        checkpoint_data = {key: value for (key, value) in checkpoint_data.items() if not key.startswith('_')}
        assert checkpoint_data == {'data': 1}, checkpoint_data

def function_trainable(config):
    if False:
        print('Hello World!')
    with create_dict_checkpoint({'checkpoint_data': 5}) as checkpoint:
        train.report({'metric': 4}, checkpoint=checkpoint)

@pytest.mark.parametrize('return_type', ['object', 'root'])
def test_save_load_checkpoint_path_class(ray_start_2_cpus, return_type, tmpdir):
    if False:
        print('Hello World!')
    'Assert that restoring from a Trainable.save() future works with\n    class trainables.\n\n    Needs Ray cluster so we get actual futures.\n    '
    trainable = ray.remote(SavingTrainable).remote(return_type=return_type)
    ray.get(trainable.train.remote())
    saving_future = trainable.save.remote()
    ray.get(saving_future)
    restoring_future = trainable.restore.remote(saving_future)
    ray.get(restoring_future)

def test_save_load_checkpoint_path_fn(ray_start_2_cpus, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    'Assert that restoring from a Trainable.save() future works with\n    function trainables.\n\n    Needs Ray cluster so we get actual futures.\n    '
    trainable_cls = wrap_function(function_trainable)
    trainable = ray.remote(trainable_cls).remote(storage=StorageContext(storage_path=str(tmp_path), experiment_dir_name='exp', trial_dir_name='trial'))
    ray.get(trainable.train.remote())
    saving_future = trainable.save.remote()
    ray.get(saving_future)
    restoring_future = trainable.restore.remote(saving_future)
    ray.get(restoring_future)

@pytest.mark.skip('Fallback to latest checkpoint is not implemented.')
def test_find_latest_checkpoint_local(tmpdir):
    if False:
        print('Hello World!')
    'Tests that we identify the latest available checkpoint correctly.\n\n    When new checkpoints are created, they should be the latest available ones.\n    When the latest checkpoint is deleted, we should go back to the previous one.\n    '

@pytest.mark.skip('Fallback to latest checkpoint is not implemented.')
def test_find_latest_checkpoint_remote(tmpdir):
    if False:
        i = 10
        return i + 15
    'Tests that we identify the latest available checkpoint correctly.\n\n    When new checkpoints are created, they should be the latest available ones.\n    When the latest checkpoint is deleted, we should go back to the previous one.\n    '

@pytest.mark.skip('Fallback to latest checkpoint is not implemented.')
@pytest.mark.parametrize('upload_uri', [None, 'memory:///test/location_recover_latest'])
@pytest.mark.parametrize('fetch_from_cloud', [False, True])
def test_recover_from_latest(tmpdir, upload_uri, fetch_from_cloud):
    if False:
        for i in range(10):
            print('nop')
    'Test that trainable recovery falls back to recovery from latest checkpoint.\n\n    Creates a trainable, saves a few checkpoints.\n\n    Asserts that restoring from a non-existing path falls back to the latest saved\n    checkpoint.\n\n    Asserts that restoring from a previously-existing path falls back to the latest\n    saved checkpoints.\n\n    If `fetch_from_cloud=True`, asserts that newer checkpoints on cloud are preferred\n    over older checkpoints on local disk.\n    '
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))