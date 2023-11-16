import os
import pickle
import shutil
import tempfile
import unittest
import pytest
import ray
from ray import tune
from ray.train import CheckpointConfig
from ray.rllib import _register_all
from ray.tune import Trainable
from ray.tune.utils import validate_save_restore

class SerialTuneRelativeLocalDirTest(unittest.TestCase):
    local_mode = True
    prefix = 'Serial'

    class MockTrainable(Trainable):
        _name = 'MockTrainable'

        def setup(self, config):
            if False:
                return 10
            self.state = {'hi': 1}

        def step(self):
            if False:
                for i in range(10):
                    print('nop')
            return {'timesteps_this_iter': 1, 'done': True}

        def save_checkpoint(self, checkpoint_dir):
            if False:
                for i in range(10):
                    print('nop')
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pkl')
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(self.state, f)

        def load_checkpoint(self, checkpoint_dir):
            if False:
                print('Hello World!')
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pkl')
            with open(checkpoint_path, 'rb') as f:
                extra_data = pickle.load(f)
            self.state.update(extra_data)

    @pytest.fixture(autouse=True)
    def setLocalDir(self, tmp_path, monkeypatch):
        if False:
            i = 10
            return i + 15
        monkeypatch.setenv('RAY_AIR_LOCAL_CACHE_DIR', str(tmp_path / 'ray_results'))

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.absolute_local_dir = None
        ray.init(num_cpus=1, num_gpus=0, local_mode=self.local_mode)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        if self.absolute_local_dir is not None:
            shutil.rmtree(self.absolute_local_dir, ignore_errors=True)
            self.absolute_local_dir = None
        ray.shutdown()
        _register_all()

    def _get_trial_dir(self, absoulte_exp_dir):
        if False:
            return 10
        print('looking for', self.MockTrainable._name)
        print('in', os.listdir(absoulte_exp_dir))
        trial_dirname = next((child_dir for child_dir in os.listdir(absoulte_exp_dir) if os.path.isdir(os.path.join(absoulte_exp_dir, child_dir)) and child_dir.startswith(self.MockTrainable._name)))
        trial_absolute_dir = os.path.join(absoulte_exp_dir, trial_dirname)
        return (trial_dirname, trial_absolute_dir)

    def _train(self, exp_name, local_dir, absolute_local_dir):
        if False:
            return 10
        (trial,) = tune.run(self.MockTrainable, name=exp_name, stop={'training_iteration': 1}, checkpoint_config=CheckpointConfig(checkpoint_frequency=1), storage_path=local_dir, config={'env': 'CartPole-v0', 'log_level': 'DEBUG'}).trials
        exp_dir = os.path.join(absolute_local_dir, exp_name)
        (_, abs_trial_dir) = self._get_trial_dir(exp_dir)
        self.assertIsNone(trial.error_file)
        self.assertEqual(trial.path, abs_trial_dir)
        self.assertTrue(os.path.isdir(absolute_local_dir), absolute_local_dir)
        self.assertTrue(os.path.isdir(exp_dir))
        self.assertTrue(os.path.isdir(abs_trial_dir))
        self.assertTrue(os.path.isfile(os.path.join(abs_trial_dir, 'checkpoint_000000/checkpoint.pkl')))

    def _restore(self, exp_name, local_dir, absolute_local_dir):
        if False:
            print('Hello World!')
        (trial_name, abs_trial_dir) = self._get_trial_dir(os.path.join(absolute_local_dir, exp_name))
        checkpoint_path = os.path.join(local_dir, exp_name, trial_name, 'checkpoint_000000')
        assert os.path.exists(os.path.expanduser(checkpoint_path))
        (trial,) = tune.run(self.MockTrainable, name=exp_name, stop={'training_iteration': 2}, restore=checkpoint_path, config={'env': 'CartPole-v0', 'log_level': 'DEBUG'}).trials
        self.assertIsNone(trial.error_file)

    def testTempfile(self):
        if False:
            print('Hello World!')
        local_dir = tempfile.mkdtemp()
        exp_name = self.prefix + 'Tempfile'
        self.absolute_local_dir = local_dir
        self._train(exp_name, local_dir, local_dir)
        self._restore(exp_name, local_dir, local_dir)

    def testCheckpointWithNoop(self):
        if False:
            i = 10
            return i + 15
        'Tests that passing the checkpoint_dir right back works.'

        class MockTrainable(Trainable):

            def setup(self, config):
                if False:
                    while True:
                        i = 10
                pass

            def step(self):
                if False:
                    i = 10
                    return i + 15
                return {'score': 1}

            def save_checkpoint(self, checkpoint_dir):
                if False:
                    print('Hello World!')
                with open(os.path.join(checkpoint_dir, 'test.txt'), 'wb') as f:
                    pickle.dump('test', f)

            def load_checkpoint(self, checkpoint_dir):
                if False:
                    i = 10
                    return i + 15
                with open(os.path.join(checkpoint_dir, 'test.txt'), 'rb') as f:
                    x = pickle.load(f)
                assert x == 'test'
        validate_save_restore(MockTrainable)
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))