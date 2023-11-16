import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from collections import OrderedDict
import ray
from ray import tune
from ray.air._internal.checkpoint_manager import _TrackedCheckpoint, CheckpointStorage
from ray.air.constants import TRAINING_ITERATION
from ray.rllib import _register_all
from ray.tune.execution.ray_trial_executor import _ExecutorEvent, _ExecutorEventType, RayTrialExecutor
from ray.tune.callback import warnings
from ray.tune.experiment import Trial
from ray.tune.execution.trial_runner import TrialRunner
from ray.tune import Callback
from ray.tune.experiment import Experiment

class TestCallback(Callback):

    def __init__(self):
        if False:
            return 10
        self.state = OrderedDict()

    def setup(self, **info):
        if False:
            while True:
                i = 10
        self.state['setup'] = info

    def on_step_begin(self, **info):
        if False:
            return 10
        self.state['step_begin'] = info

    def on_step_end(self, **info):
        if False:
            for i in range(10):
                print('nop')
        self.state['step_end'] = info

    def on_trial_start(self, **info):
        if False:
            for i in range(10):
                print('nop')
        self.state['trial_start'] = info

    def on_trial_restore(self, **info):
        if False:
            while True:
                i = 10
        self.state['trial_restore'] = info

    def on_trial_save(self, **info):
        if False:
            print('Hello World!')
        self.state['trial_save'] = info

    def on_trial_result(self, **info):
        if False:
            print('Hello World!')
        self.state['trial_result'] = info
        result = info['result']
        trial = info['trial']
        assert result.get(TRAINING_ITERATION, None) != trial.last_result.get(TRAINING_ITERATION, None)

    def on_trial_complete(self, **info):
        if False:
            while True:
                i = 10
        self.state['trial_complete'] = info

    def on_trial_error(self, **info):
        if False:
            return 10
        self.state['trial_fail'] = info

    def on_experiment_end(self, **info):
        if False:
            while True:
                i = 10
        self.state['experiment_end'] = info

class _MockTrialExecutor(RayTrialExecutor):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.next_future_result = None

    def start_trial(self, trial: Trial):
        if False:
            print('Hello World!')
        trial.status = Trial.RUNNING
        return True

    def continue_training(self, trial: Trial):
        if False:
            for i in range(10):
                print('nop')
        pass

    def get_next_executor_event(self, live_trials, next_trial_exists):
        if False:
            while True:
                i = 10
        return self.next_future_result

class TrialRunnerCallbacks(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        ray.init()
        self.tmpdir = tempfile.mkdtemp()
        self.callback = TestCallback()
        self.executor = _MockTrialExecutor()
        self.trial_runner = TrialRunner(trial_executor=self.executor, callbacks=[self.callback])
        self.trial_runner.setup_experiments(experiments=[None], total_num_samples=1)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        ray.shutdown()
        _register_all()
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        shutil.rmtree(self.tmpdir)

    def testCallbackSteps(self):
        if False:
            while True:
                i = 10
        trials = [Trial('__fake', trial_id='one'), Trial('__fake', trial_id='two')]
        for t in trials:
            self.trial_runner.add_trial(t)
        self.executor.next_future_result = _ExecutorEvent(event_type=_ExecutorEventType.PG_READY)
        self.trial_runner.step()
        self.assertEqual(self.callback.state['trial_start']['iteration'], 0)
        self.assertEqual(self.callback.state['trial_start']['trial'].trial_id, 'one')
        self.assertTrue(all((k not in self.callback.state for k in ['trial_restore', 'trial_save', 'trial_result', 'trial_complete', 'trial_fail', 'experiment_end'])))
        self.executor.next_future_result = _ExecutorEvent(event_type=_ExecutorEventType.PG_READY)
        self.trial_runner.step()
        self.assertEqual(self.callback.state['step_begin']['iteration'], 1)
        self.assertEqual(self.callback.state['step_end']['iteration'], 2)
        self.assertEqual(self.callback.state['trial_start']['iteration'], 1)
        self.assertEqual(self.callback.state['trial_start']['trial'].trial_id, 'two')
        cp = _TrackedCheckpoint(dir_or_data=ray.put(1), storage_mode=CheckpointStorage.PERSISTENT, metrics={TRAINING_ITERATION: 0})
        trials[0].temporary_state.saving_to = cp
        self.executor.next_future_result = _ExecutorEvent(event_type=_ExecutorEventType.SAVING_RESULT, trial=trials[0], result={_ExecutorEvent.KEY_FUTURE_RESULT: '__checkpoint'})
        self.trial_runner.step()
        self.assertEqual(self.callback.state['trial_save']['iteration'], 2)
        self.assertEqual(self.callback.state['trial_save']['trial'].trial_id, 'one')
        result = {TRAINING_ITERATION: 1, 'metric': 800, 'done': False}
        self.executor.next_future_result = _ExecutorEvent(event_type=_ExecutorEventType.TRAINING_RESULT, trial=trials[1], result={_ExecutorEvent.KEY_FUTURE_RESULT: result})
        self.assertTrue(not trials[1].has_reported_at_least_once)
        self.trial_runner.step()
        self.assertEqual(self.callback.state['trial_result']['iteration'], 3)
        self.assertEqual(self.callback.state['trial_result']['trial'].trial_id, 'two')
        self.assertEqual(self.callback.state['trial_result']['result']['metric'], 800)
        self.assertEqual(trials[1].last_result['metric'], 800)
        trials[1].temporary_state.restoring_from = cp
        self.executor.next_future_result = _ExecutorEvent(event_type=_ExecutorEventType.RESTORING_RESULT, trial=trials[1], result={_ExecutorEvent.KEY_FUTURE_RESULT: None})
        self.trial_runner.step()
        self.assertEqual(self.callback.state['trial_restore']['iteration'], 4)
        self.assertEqual(self.callback.state['trial_restore']['trial'].trial_id, 'two')
        trials[1].temporary_state.restoring_from = None
        self.executor.next_future_result = _ExecutorEvent(event_type=_ExecutorEventType.TRAINING_RESULT, trial=trials[1], result={_ExecutorEvent.KEY_FUTURE_RESULT: {TRAINING_ITERATION: 2, 'metric': 900, 'done': True}})
        self.trial_runner.step()
        self.assertEqual(self.callback.state['trial_complete']['iteration'], 5)
        self.assertEqual(self.callback.state['trial_complete']['trial'].trial_id, 'two')
        self.executor.next_future_result = _ExecutorEvent(event_type=_ExecutorEventType.TRAINING_RESULT, trial=trials[0], result={_ExecutorEvent.KEY_EXCEPTION: Exception()})
        self.trial_runner.step()
        self.assertEqual(self.callback.state['trial_fail']['iteration'], 6)
        self.assertEqual(self.callback.state['trial_fail']['trial'].trial_id, 'one')

    def testCallbacksEndToEnd(self):
        if False:
            i = 10
            return i + 15

        def train_fn(config):
            if False:
                i = 10
                return i + 15
            if config['do'] == 'save':
                with tune.checkpoint_dir(0):
                    pass
                tune.report(metric=1)
            elif config['do'] == 'fail':
                raise RuntimeError('I am failing on purpose.')
            elif config['do'] == 'delay':
                time.sleep(2)
                tune.report(metric=20)
        config = {'do': tune.grid_search(['save', 'fail', 'delay'])}
        tune.run(train_fn, config=config, raise_on_failed_trial=False, callbacks=[self.callback])
        self.assertIn('setup', self.callback.state)
        self.assertTrue(self.callback.state['setup'] is not None)
        keys = Experiment.PUBLIC_KEYS.copy()
        keys.add('total_num_samples')
        for key in keys:
            self.assertIn(key, self.callback.state['setup'])
        self.assertTrue(list(self.callback.state)[0] == 'setup')
        self.assertEqual(self.callback.state['trial_fail']['trial'].config['do'], 'fail')
        self.assertEqual(self.callback.state['trial_save']['trial'].config['do'], 'save')
        self.assertEqual(self.callback.state['trial_result']['trial'].config['do'], 'delay')
        self.assertEqual(self.callback.state['trial_complete']['trial'].config['do'], 'delay')
        self.assertIn('experiment_end', self.callback.state)
        self.assertTrue(list(self.callback.state)[-1] == 'experiment_end')

    @patch.object(warnings, 'warn')
    def testCallbackSetupBackwardsCompatible(self, mocked_warning_method):
        if False:
            i = 10
            return i + 15

        class NoExperimentInSetupCallback(Callback):

            def setup(self):
                if False:
                    i = 10
                    return i + 15
                return
        callback = NoExperimentInSetupCallback()
        trial_runner = TrialRunner(callbacks=[callback])
        trial_runner.setup_experiments(experiments=[Experiment('', lambda x: x)], total_num_samples=1)
        mocked_warning_method.assert_called_once()
        self.assertIn('Please update', mocked_warning_method.call_args_list[0][0][0])
if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main(['-v', __file__]))