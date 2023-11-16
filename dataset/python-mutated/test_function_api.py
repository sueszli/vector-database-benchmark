import json
import os
import sys
import shutil
import tempfile
import unittest
import ray
from ray.air.constants import TRAINING_ITERATION
from ray.rllib import _register_all
import ray.train
from ray import tune
from ray.train import Checkpoint, CheckpointConfig
from ray.tune.logger import NoopLogger
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.tune.trainable import with_parameters, wrap_function
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers import ResourceChangingScheduler
from ray.train.tests.util import mock_storage_context

def creator_generator(logdir):
    if False:
        print('Hello World!')

    def logger_creator(config):
        if False:
            print('Hello World!')
        return NoopLogger(config, logdir)
    return logger_creator

class FunctionCheckpointingTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.logdir = tempfile.mkdtemp()
        self.logger_creator = creator_generator(self.logdir)

    def create_trainable(self, train_fn):
        if False:
            for i in range(10):
                print('nop')
        return wrap_function(train_fn)(logger_creator=self.logger_creator, storage=mock_storage_context())

    def tearDown(self):
        if False:
            print('Hello World!')
        shutil.rmtree(self.logdir)

    def testCheckpointReuse(self):
        if False:
            return 10
        'Test that repeated save/restore never reuses same checkpoint dir.'

        def train_fn(config):
            if False:
                for i in range(10):
                    print('nop')
            checkpoint = ray.train.get_checkpoint()
            if checkpoint:
                with checkpoint.as_directory() as checkpoint_dir:
                    count = sum(('checkpoint-' in path for path in os.listdir(checkpoint_dir)))
                    assert count == 1, os.listdir(checkpoint_dir)
            for step in range(20):
                with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    path = os.path.join(temp_checkpoint_dir, 'checkpoint-{}'.format(step))
                    open(path, 'a').close()
                    ray.train.report(dict(test=step), checkpoint=Checkpoint.from_directory(temp_checkpoint_dir))
        checkpoint = None
        for i in range(5):
            new_trainable = self.create_trainable(train_fn)
            if checkpoint:
                new_trainable.restore(checkpoint)
            for i in range(2):
                result = new_trainable.train()
            checkpoint = new_trainable.save()
            new_trainable.stop()
        assert result[TRAINING_ITERATION] == 10

    def testFunctionRecurringSave(self):
        if False:
            i = 10
            return i + 15
        'This tests that save and restore are commutative.'

        def train_fn(config):
            if False:
                for i in range(10):
                    print('nop')
            for step in range(10):
                with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    if step % 3 == 0:
                        path = os.path.join(temp_checkpoint_dir, 'checkpoint.json')
                        with open(path, 'w') as f:
                            json.dump({'step': step}, f)
                    ray.train.report(dict(test=step), checkpoint=Checkpoint.from_directory(temp_checkpoint_dir))
        new_trainable = self.create_trainable(train_fn)
        new_trainable.train()
        checkpoint_obj = new_trainable.save()
        new_trainable.restore(checkpoint_obj)
        checkpoint = new_trainable.save()
        new_trainable.stop()
        new_trainable2 = self.create_trainable(train_fn)
        new_trainable2.restore(checkpoint)
        new_trainable2.train()
        new_trainable2.stop()

class FunctionApiTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        ray.init(num_cpus=4, num_gpus=0, object_store_memory=150 * 1024 * 1024)

    def tearDown(self):
        if False:
            return 10
        ray.shutdown()
        _register_all()

    def testCheckpointError(self):
        if False:
            for i in range(10):
                print('nop')

        def train_fn(config):
            if False:
                i = 10
                return i + 15
            pass
        with self.assertRaises(ValueError):
            tune.run(train_fn, checkpoint_config=CheckpointConfig(checkpoint_frequency=1))
        with self.assertRaises(ValueError):
            tune.run(train_fn, checkpoint_config=CheckpointConfig(checkpoint_at_end=True))

    def testWithParameters(self):
        if False:
            while True:
                i = 10

        class Data:

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.data = [0] * 500000
        data = Data()
        data.data[100] = 1

        def train_fn(config, data=None):
            if False:
                print('Hello World!')
            data.data[101] = 2
            ray.train.report(dict(metric=len(data.data), hundred=data.data[100]))
        (trial_1, trial_2) = tune.run(with_parameters(train_fn, data=data), num_samples=2).trials
        self.assertEqual(data.data[101], 0)
        self.assertEqual(trial_1.last_result['metric'], 500000)
        self.assertEqual(trial_1.last_result['hundred'], 1)
        self.assertEqual(trial_2.last_result['metric'], 500000)
        self.assertEqual(trial_2.last_result['hundred'], 1)
        self.assertTrue(str(trial_1).startswith('train_'))

        def train_fn(config, data=None):
            if False:
                return 10
            data.data[101] = 2
            ray.train.report(dict(metric=len(data.data)))
        (trial_1, trial_2) = tune.run(with_parameters(train_fn, data=data), num_samples=2).trials
        self.assertEqual(data.data[101], 0)
        self.assertEqual(trial_1.last_result['metric'], 500000)
        self.assertEqual(trial_2.last_result['metric'], 500000)
        self.assertTrue(str(trial_1).startswith('train_'))

    def testWithParameters2(self):
        if False:
            for i in range(10):
                print('nop')

        class Data:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                import numpy as np
                self.data = np.random.rand(2 * 1024 * 1024)

        def train_fn(config, data=None):
            if False:
                for i in range(10):
                    print('nop')
            pass
        trainable = tune.with_parameters(train_fn, data=Data())
        import cloudpickle as cp
        dumped = cp.dumps(trainable)
        assert sys.getsizeof(dumped) < 100 * 1024

    def testNewResources(self):
        if False:
            for i in range(10):
                print('nop')
        sched = ResourceChangingScheduler(resources_allocation_function=lambda a, b, c, d: PlacementGroupFactory([{'CPU': 2}]))

        def train_fn(config):
            if False:
                i = 10
                return i + 15
            ray.train.report(dict(metric=1, resources=ray.train.get_context().get_trial_resources()))
        analysis = tune.run(train_fn, scheduler=sched, stop={'training_iteration': 2}, resources_per_trial=PlacementGroupFactory([{'CPU': 1}]), num_samples=1)
        results_list = list(analysis.results.values())
        assert results_list[0]['resources'].head_cpus == 2.0

    def testWithParametersTwoRuns1(self):
        if False:
            for i in range(10):
                print('nop')

        def train_fn(config, extra=4):
            if False:
                return 10
            ray.train.report(dict(metric=extra))
        trainable = tune.with_parameters(train_fn, extra=8)
        out = tune.run(trainable, metric='metric', mode='max')
        self.assertEqual(out.best_result['metric'], 8)
        self.tearDown()
        self.setUp()

        def train_fn_2(config, extra=5):
            if False:
                return 10
            ray.train.report(dict(metric=extra))
        trainable = tune.with_parameters(train_fn_2, extra=9)
        out = tune.run(trainable, metric='metric', mode='max')
        self.assertEqual(out.best_result['metric'], 9)

    def testWithParametersTwoRuns2(self):
        if False:
            print('Hello World!')

        def train_fn(config, extra=4):
            if False:
                for i in range(10):
                    print('nop')
            ray.train.report(dict(metric=extra))

        def train_fn_2(config, extra=5):
            if False:
                print('Hello World!')
            ray.train.report(dict(metric=extra))
        trainable1 = tune.with_parameters(train_fn, extra=8)
        trainable2 = tune.with_parameters(train_fn_2, extra=9)
        out1 = tune.run(trainable1, metric='metric', mode='max')
        out2 = tune.run(trainable2, metric='metric', mode='max')
        self.assertEqual(out1.best_result['metric'], 8)
        self.assertEqual(out2.best_result['metric'], 9)

    def testReturnAnonymous(self):
        if False:
            return 10

        def train_fn(config):
            if False:
                return 10
            return config['a']
        (trial_1, trial_2) = tune.run(train_fn, config={'a': tune.grid_search([4, 8])}).trials
        self.assertEqual(trial_1.last_result[DEFAULT_METRIC], 4)
        self.assertEqual(trial_2.last_result[DEFAULT_METRIC], 8)

    def testReturnSpecific(self):
        if False:
            return 10

        def train_fn(config):
            if False:
                for i in range(10):
                    print('nop')
            return {'m': config['a']}
        (trial_1, trial_2) = tune.run(train_fn, config={'a': tune.grid_search([4, 8])}).trials
        self.assertEqual(trial_1.last_result['m'], 4)
        self.assertEqual(trial_2.last_result['m'], 8)

    def testYieldAnonymous(self):
        if False:
            i = 10
            return i + 15

        def train_fn(config):
            if False:
                while True:
                    i = 10
            for i in range(10):
                yield (config['a'] + i)
        (trial_1, trial_2) = tune.run(train_fn, config={'a': tune.grid_search([4, 8])}).trials
        self.assertEqual(trial_1.last_result[DEFAULT_METRIC], 4 + 9)
        self.assertEqual(trial_2.last_result[DEFAULT_METRIC], 8 + 9)

    def testYieldSpecific(self):
        if False:
            for i in range(10):
                print('nop')

        def train_fn(config):
            if False:
                return 10
            for i in range(10):
                yield {'m': config['a'] + i}
        (trial_1, trial_2) = tune.run(train_fn, config={'a': tune.grid_search([4, 8])}).trials
        self.assertEqual(trial_1.last_result['m'], 4 + 9)
        self.assertEqual(trial_2.last_result['m'], 8 + 9)
if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main(['-v', __file__]))