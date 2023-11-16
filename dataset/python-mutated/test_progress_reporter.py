import collections
import os
import regex as re
import unittest
from unittest.mock import MagicMock, Mock, patch
import pytest
import numpy as np
from ray import train, tune
from ray._private.test_utils import run_string_as_driver
from ray.tune.progress_reporter import CLIReporter, JupyterNotebookReporter, ProgressReporter, _fair_filter_trials, _best_trial_str, _detect_reporter, _time_passed_str, _trial_progress_str, TuneReporterBase, _max_len
from ray.tune.result import AUTO_RESULT_KEYS
from ray.tune.experiment.trial import Trial
EXPECTED_RESULT_1 = 'Result logdir: /foo\nNumber of trials: 5 (1 PENDING, 3 RUNNING, 1 TERMINATED)\n+--------------+------------+-------+-----+-----+------------+\n|   Trial name | status     | loc   |   a |   b |   metric_1 |\n|--------------+------------+-------+-----+-----+------------|\n|        00002 | RUNNING    | here  |   2 |   4 |        1   |\n|        00001 | PENDING    | here  |   1 |   2 |        0.5 |\n|        00000 | TERMINATED | here  |   0 |   0 |        0   |\n+--------------+------------+-------+-----+-----+------------+\n... 2 more trials not shown (2 RUNNING)'
EXPECTED_RESULT_2 = 'Result logdir: /foo\nNumber of trials: 5 (1 PENDING, 3 RUNNING, 1 TERMINATED)\n+--------------+------------+-------+-----+-----+---------+---------+\n|   Trial name | status     | loc   |   a |   b |   n/k/0 |   n/k/1 |\n|--------------+------------+-------+-----+-----+---------+---------|\n|        00002 | RUNNING    | here  |   2 |   4 |       2 |       4 |\n|        00003 | RUNNING    | here  |   3 |   6 |       3 |       6 |\n|        00004 | RUNNING    | here  |   4 |   8 |       4 |       8 |\n|        00001 | PENDING    | here  |   1 |   2 |       1 |       2 |\n|        00000 | TERMINATED | here  |   0 |   0 |       0 |       0 |\n+--------------+------------+-------+-----+-----+---------+---------+'
EXPECTED_RESULT_3 = 'Result logdir: /foo\nNumber of trials: 5 (1 PENDING, 3 RUNNING, 1 TERMINATED)\n+--------------+------------+-------+-----+-----------+------------+\n|   Trial name | status     | loc   |   A |   NestSub |   Metric 2 |\n|--------------+------------+-------+-----+-----------+------------|\n|        00002 | RUNNING    | here  |   2 |       1   |       0.5  |\n|        00001 | PENDING    | here  |   1 |       0.5 |       0.25 |\n|        00000 | TERMINATED | here  |   0 |       0   |       0    |\n+--------------+------------+-------+-----+-----------+------------+\n... 2 more trials not shown (2 RUNNING)'
EXPECTED_RESULT_4 = 'Result logdir: /foo\nNumber of trials: 5 (1 PENDING, 3 RUNNING, 1 TERMINATED)\n+--------------+------------+-------+-----+-----+------------+\n|   Trial name | status     | loc   |   a |   b |   metric_1 |\n|--------------+------------+-------+-----+-----+------------|\n|        00002 | RUNNING    | here  |   2 |   4 |        1   |\n|        00003 | RUNNING    | here  |   3 |   6 |        1.5 |\n|        00004 | RUNNING    | here  |   4 |   8 |        2   |\n|        00001 | PENDING    | here  |   1 |   2 |        0.5 |\n|        00000 | TERMINATED | here  |   0 |   0 |        0   |\n+--------------+------------+-------+-----+-----+------------+'
END_TO_END_COMMAND = '\nimport ray\nfrom ray import tune\nfrom ray.tune.experiment.trial import _Location\nfrom ray.tune.progress_reporter import _get_trial_location\nfrom unittest.mock import patch\n\n\ndef mock_get_trial_location(trial, result):\n    location = _get_trial_location(trial, result)\n    if location.pid:\n        return _Location("123.123.123.123", "1")\n    return location\n\n\nwith patch("ray.tune.progress_reporter._get_trial_location",\n           mock_get_trial_location):\n    reporter = tune.progress_reporter.CLIReporter(metric_columns=["done"])\n\n    def f(config):\n        return {"done": True}\n\n    ray.init(num_cpus=1)\n    tune.run_experiments(\n        {\n            "one": {\n                "run": f,\n                "config": {\n                    "a": tune.grid_search(list(range(10))),\n                },\n            },\n            "two": {\n                "run": f,\n                "config": {\n                    "b": tune.grid_search(list(range(10))),\n                },\n            },\n            "three": {\n                "run": f,\n                "config": {\n                    "c": tune.grid_search(list(range(10))),\n                },\n            },\n        },\n        verbose=3,\n        progress_reporter=reporter)'
EXPECTED_END_TO_END_START = 'Number of trials: 30/30 (29 PENDING, 1 RUNNING)\n+---------------+----------+-------------------+-----+-----+\n| Trial name    | status   | loc               |   a |   b |\n|---------------+----------+-------------------+-----+-----|\n| f_xxxxx_00000 | RUNNING  | 123.123.123.123:1 |   0 |     |\n| f_xxxxx_00001 | PENDING  |                   |   1 |     |'
EXPECTED_END_TO_END_END = 'Number of trials: 30/30 (30 TERMINATED)\n+---------------+------------+-------------------+-----+-----+-----+--------+\n| Trial name    | status     | loc               |   a |   b |   c | done   |\n|---------------+------------+-------------------+-----+-----+-----+--------|\n| f_xxxxx_00000 | TERMINATED | 123.123.123.123:1 |   0 |     |     | True   |\n| f_xxxxx_00001 | TERMINATED | 123.123.123.123:1 |   1 |     |     | True   |\n| f_xxxxx_00002 | TERMINATED | 123.123.123.123:1 |   2 |     |     | True   |\n| f_xxxxx_00003 | TERMINATED | 123.123.123.123:1 |   3 |     |     | True   |\n| f_xxxxx_00004 | TERMINATED | 123.123.123.123:1 |   4 |     |     | True   |\n| f_xxxxx_00005 | TERMINATED | 123.123.123.123:1 |   5 |     |     | True   |\n| f_xxxxx_00006 | TERMINATED | 123.123.123.123:1 |   6 |     |     | True   |\n| f_xxxxx_00007 | TERMINATED | 123.123.123.123:1 |   7 |     |     | True   |\n| f_xxxxx_00008 | TERMINATED | 123.123.123.123:1 |   8 |     |     | True   |\n| f_xxxxx_00009 | TERMINATED | 123.123.123.123:1 |   9 |     |     | True   |\n| f_xxxxx_00010 | TERMINATED | 123.123.123.123:1 |     |   0 |     | True   |\n| f_xxxxx_00011 | TERMINATED | 123.123.123.123:1 |     |   1 |     | True   |\n| f_xxxxx_00012 | TERMINATED | 123.123.123.123:1 |     |   2 |     | True   |\n| f_xxxxx_00013 | TERMINATED | 123.123.123.123:1 |     |   3 |     | True   |\n| f_xxxxx_00014 | TERMINATED | 123.123.123.123:1 |     |   4 |     | True   |\n| f_xxxxx_00015 | TERMINATED | 123.123.123.123:1 |     |   5 |     | True   |\n| f_xxxxx_00016 | TERMINATED | 123.123.123.123:1 |     |   6 |     | True   |\n| f_xxxxx_00017 | TERMINATED | 123.123.123.123:1 |     |   7 |     | True   |\n| f_xxxxx_00018 | TERMINATED | 123.123.123.123:1 |     |   8 |     | True   |\n| f_xxxxx_00019 | TERMINATED | 123.123.123.123:1 |     |   9 |     | True   |\n| f_xxxxx_00020 | TERMINATED | 123.123.123.123:1 |     |     |   0 | True   |\n| f_xxxxx_00021 | TERMINATED | 123.123.123.123:1 |     |     |   1 | True   |\n| f_xxxxx_00022 | TERMINATED | 123.123.123.123:1 |     |     |   2 | True   |\n| f_xxxxx_00023 | TERMINATED | 123.123.123.123:1 |     |     |   3 | True   |\n| f_xxxxx_00024 | TERMINATED | 123.123.123.123:1 |     |     |   4 | True   |\n| f_xxxxx_00025 | TERMINATED | 123.123.123.123:1 |     |     |   5 | True   |\n| f_xxxxx_00026 | TERMINATED | 123.123.123.123:1 |     |     |   6 | True   |\n| f_xxxxx_00027 | TERMINATED | 123.123.123.123:1 |     |     |   7 | True   |\n| f_xxxxx_00028 | TERMINATED | 123.123.123.123:1 |     |     |   8 | True   |\n| f_xxxxx_00029 | TERMINATED | 123.123.123.123:1 |     |     |   9 | True   |\n+---------------+------------+-------------------+-----+-----+-----+--------+'
EXPECTED_END_TO_END_AC = 'Number of trials: 30/30 (30 TERMINATED)\n+---------------+------------+-------+-----+-----+-----+\n| Trial name    | status     | loc   |   a |   b |   c |\n|---------------+------------+-------+-----+-----+-----|\n| f_xxxxx_00000 | TERMINATED |       |   0 |     |     |\n| f_xxxxx_00001 | TERMINATED |       |   1 |     |     |\n| f_xxxxx_00002 | TERMINATED |       |   2 |     |     |\n| f_xxxxx_00003 | TERMINATED |       |   3 |     |     |\n| f_xxxxx_00004 | TERMINATED |       |   4 |     |     |\n| f_xxxxx_00005 | TERMINATED |       |   5 |     |     |\n| f_xxxxx_00006 | TERMINATED |       |   6 |     |     |\n| f_xxxxx_00007 | TERMINATED |       |   7 |     |     |\n| f_xxxxx_00008 | TERMINATED |       |   8 |     |     |\n| f_xxxxx_00009 | TERMINATED |       |   9 |     |     |\n| f_xxxxx_00010 | TERMINATED |       |     |   0 |     |\n| f_xxxxx_00011 | TERMINATED |       |     |   1 |     |\n| f_xxxxx_00012 | TERMINATED |       |     |   2 |     |\n| f_xxxxx_00013 | TERMINATED |       |     |   3 |     |\n| f_xxxxx_00014 | TERMINATED |       |     |   4 |     |\n| f_xxxxx_00015 | TERMINATED |       |     |   5 |     |\n| f_xxxxx_00016 | TERMINATED |       |     |   6 |     |\n| f_xxxxx_00017 | TERMINATED |       |     |   7 |     |\n| f_xxxxx_00018 | TERMINATED |       |     |   8 |     |\n| f_xxxxx_00019 | TERMINATED |       |     |   9 |     |\n| f_xxxxx_00020 | TERMINATED |       |     |     |   0 |\n| f_xxxxx_00021 | TERMINATED |       |     |     |   1 |\n| f_xxxxx_00022 | TERMINATED |       |     |     |   2 |\n| f_xxxxx_00023 | TERMINATED |       |     |     |   3 |\n| f_xxxxx_00024 | TERMINATED |       |     |     |   4 |\n| f_xxxxx_00025 | TERMINATED |       |     |     |   5 |\n| f_xxxxx_00026 | TERMINATED |       |     |     |   6 |\n| f_xxxxx_00027 | TERMINATED |       |     |     |   7 |\n| f_xxxxx_00028 | TERMINATED |       |     |     |   8 |\n| f_xxxxx_00029 | TERMINATED |       |     |     |   9 |\n+---------------+------------+-------+-----+-----+-----+'
EXPECTED_BEST_1 = "Current best trial: 00001 with metric_1=0.5 and parameters={'a': 1, 'b': 2, 'n': {'k': [1, 2]}}"
EXPECTED_BEST_2 = "Current best trial: 00004 with metric_1=2.0 and parameters={'a': 4}"
EXPECTED_SORT_RESULT_UNSORTED = 'Number of trials: 5 (1 PENDING, 1 RUNNING, 3 TERMINATED)\n+--------------+------------+-------+-----+------------+\n|   Trial name | status     | loc   |   a |   metric_1 |\n|--------------+------------+-------+-----+------------|\n|        00004 | RUNNING    | here  |   4 |            |\n|        00003 | PENDING    | here  |   3 |            |\n|        00000 | TERMINATED | here  |   0 |        0.3 |\n|        00001 | TERMINATED | here  |   1 |        0.2 |\n+--------------+------------+-------+-----+------------+\n... 1 more trials not shown (1 TERMINATED)'
EXPECTED_SORT_RESULT_ASC = 'Number of trials: 5 (1 PENDING, 1 RUNNING, 3 TERMINATED)\n+--------------+------------+-------+-----+------------+\n|   Trial name | status     | loc   |   a |   metric_1 |\n|--------------+------------+-------+-----+------------|\n|        00004 | RUNNING    | here  |   4 |            |\n|        00003 | PENDING    | here  |   3 |            |\n|        00001 | TERMINATED | here  |   1 |        0.2 |\n|        00000 | TERMINATED | here  |   0 |        0.3 |\n+--------------+------------+-------+-----+------------+\n... 1 more trials not shown (1 TERMINATED)'
EXPECTED_NESTED_SORT_RESULT = 'Number of trials: 5 (1 PENDING, 1 RUNNING, 3 TERMINATED)\n+--------------+------------+-------+-----+-------------------+\n|   Trial name | status     | loc   |   a |   nested/metric_2 |\n|--------------+------------+-------+-----+-------------------|\n|        00004 | RUNNING    | here  |   4 |                   |\n|        00003 | PENDING    | here  |   3 |                   |\n|        00001 | TERMINATED | here  |   1 |               0.2 |\n|        00000 | TERMINATED | here  |   0 |               0.3 |\n+--------------+------------+-------+-----+-------------------+\n... 1 more trials not shown (1 TERMINATED)'
EXPECTED_SORT_RESULT_DESC = 'Number of trials: 5 (1 PENDING, 1 RUNNING, 3 TERMINATED)\n+--------------+------------+-------+-----+------------+\n|   Trial name | status     | loc   |   a |   metric_1 |\n|--------------+------------+-------+-----+------------|\n|        00004 | RUNNING    | here  |   4 |            |\n|        00003 | PENDING    | here  |   3 |            |\n|        00002 | TERMINATED | here  |   2 |        0.4 |\n|        00000 | TERMINATED | here  |   0 |        0.3 |\n+--------------+------------+-------+-----+------------+\n... 1 more trials not shown (1 TERMINATED)'
VERBOSE_EXP_OUT_1 = 'Number of trials: 3/3 (2 PENDING, 1 RUNNING)'
VERBOSE_EXP_OUT_2 = 'Number of trials: 3/3 (3 TERMINATED)'
VERBOSE_TRIAL_NORM_1 = "Trial train_fn_xxxxx_00000 reported acc=5 with parameters={'do': 'complete'}. This trial completed.\n"
VERBOSE_TRIAL_NORM_2_PATTERN = "Trial train_fn_xxxxx_00001 reported _metric=6 with parameters=\\{'do': 'once'\\}\\.\\n(?s).*Trial train_fn_xxxxx_00001 completed\\. Last result: _metric=6\\n"
VERBOSE_TRIAL_NORM_3 = "Trial train_fn_xxxxx_00002 reported acc=7 with parameters={'do': 'twice'}.\n"
VERBOSE_TRIAL_NORM_4 = "Trial train_fn_xxxxx_00002 reported acc=8 with parameters={'do': 'twice'}. This trial completed.\n"
VERBOSE_TRIAL_WITH_ONCE_RESULT = 'Result for train_fn_xxxxx_00001'
VERBOSE_TRIAL_WITH_ONCE_COMPLETED = 'Trial train_fn_xxxxx_00001 completed.'
VERBOSE_TRIAL_DETAIL = '+-------------------+----------+-------------------+----------+\n| Trial name        | status   | loc               | do       |\n|-------------------+----------+-------------------+----------|\n| train_fn_xxxxx_00000 | RUNNING  | 123.123.123.123:1 | complete |'
VERBOSE_CMD = 'from ray import train as ray_train, tune\nimport random\nimport numpy as np\nimport time\nfrom ray.tune.experiment.trial import _Location\nfrom ray.tune.progress_reporter import _get_trial_location\nfrom unittest.mock import patch\n\n\ndef mock_get_trial_location(trial, result):\n    location = _get_trial_location(trial, result)\n    if location.pid:\n        return _Location("123.123.123.123", "1")\n    return location\n\n\ndef train_fn(config):\n    if config["do"] == "complete":\n        time.sleep(0.1)\n        ray_train.report(dict(acc=5, done=True))\n    elif config["do"] == "once":\n        time.sleep(0.5)\n        return 6\n    else:\n        time.sleep(1.0)\n        ray_train.report(dict(acc=7))\n        ray_train.report(dict(acc=8))\n\nrandom.seed(1234)\nnp.random.seed(1234)\n\n\nwith patch("ray.tune.progress_reporter._get_trial_location",\n           mock_get_trial_location):\n    tune.run(\n        train_fn,\n        config={\n            "do": tune.grid_search(["complete", "once", "twice"])\n        },'

class ProgressReporterTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            return 10
        os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = 'auto'
        os.environ['RAY_AIR_NEW_OUTPUT'] = '0'

    def mock_trial(self, status, i):
        if False:
            for i in range(10):
                print('nop')
        mock = MagicMock()
        mock.status = status
        mock.trial_id = '%05d' % i
        return mock

    def testFairFilterTrials(self):
        if False:
            while True:
                i = 10
        'Tests that trials are represented fairly.'
        trials_by_state = collections.defaultdict(list)
        states_under = (Trial.PAUSED, Trial.ERROR)
        states_over = (Trial.PENDING, Trial.RUNNING, Trial.TERMINATED)
        max_trials = 13
        num_trials_under = 2
        num_trials_over = 10
        i = 0
        for state in states_under:
            for _ in range(num_trials_under):
                trials_by_state[state].append(self.mock_trial(state, i))
                i += 1
        for state in states_over:
            for _ in range(num_trials_over):
                trials_by_state[state].append(self.mock_trial(state, i))
                i += 1
        filtered_trials_by_state = _fair_filter_trials(trials_by_state, max_trials=max_trials)
        for state in trials_by_state:
            if state in states_under:
                expected_num_trials = num_trials_under
            else:
                expected_num_trials = (max_trials - num_trials_under * len(states_under)) / len(states_over)
            state_trials = filtered_trials_by_state[state]
            self.assertEqual(len(state_trials), expected_num_trials)
            for i in range(len(state_trials) - 1):
                assert state_trials[i].trial_id < state_trials[i + 1].trial_id

    def testAddMetricColumn(self):
        if False:
            return 10
        'Tests edge cases of add_metric_column.'
        reporter = CLIReporter(metric_columns=['foo', 'bar'])
        with self.assertRaises(ValueError):
            reporter.add_metric_column('bar')
        with self.assertRaises(ValueError):
            reporter.add_metric_column('baz', 'qux')
        reporter.add_metric_column('baz')
        self.assertIn('baz', reporter._metric_columns)
        reporter = CLIReporter()
        reporter.add_metric_column('foo', 'bar')
        self.assertIn('foo', reporter._metric_columns)

    def testInfer(self):
        if False:
            while True:
                i = 10
        reporter = CLIReporter()
        test_result = dict(foo_result=1, baz_result=4123, bar_result='testme')

        def test(config):
            if False:
                i = 10
                return i + 15
            for i in range(3):
                train.report(test_result)
        analysis = tune.run(test, num_samples=3, verbose=3)
        all_trials = analysis.trials
        inferred_results = reporter._infer_user_metrics(all_trials)
        for metric in inferred_results:
            self.assertNotIn(metric, AUTO_RESULT_KEYS)
            self.assertTrue(metric in test_result)

        class TestReporter(CLIReporter):
            _output = []

            def __init__(self, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                super().__init__(*args, **kwargs)
                self._max_report_freqency = 0

            def report(self, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                progress_str = self._progress_str(*args, **kwargs)
                self._output.append(progress_str)
        reporter = TestReporter()
        analysis = tune.run(test, num_samples=3, progress_reporter=reporter, verbose=3)
        found = {k: False for k in test_result}
        for output in reporter._output:
            for key in test_result:
                if key in output:
                    found[key] = True
        assert found['foo_result']
        assert found['baz_result']
        assert not found['bar_result']

    def testProgressStr(self):
        if False:
            print('Hello World!')
        trials = []
        for i in range(5):
            t = Mock()
            if i == 0:
                t.status = 'TERMINATED'
            elif i == 1:
                t.status = 'PENDING'
            else:
                t.status = 'RUNNING'
            t.trial_id = '%05d' % i
            t.local_experiment_path = '/foo'
            t.temporary_state = Mock()
            t.temporary_state.location = 'here'
            t.config = {'a': i, 'b': i * 2, 'n': {'k': [i, 2 * i]}}
            t.evaluated_params = {'a': i, 'b': i * 2, 'n/k/0': i, 'n/k/1': 2 * i}
            t.last_result = {'config': {'a': i, 'b': i * 2, 'n': {'k': [i, 2 * i]}}, 'metric_1': i / 2, 'metric_2': i / 4, 'nested': {'sub': i / 2}}
            t.__str__ = lambda self: self.trial_id
            trials.append(t)
        prog1 = _trial_progress_str(trials, ['metric_1'], ['a', 'b'], fmt='psql', max_rows=3, force_table=True)
        print(prog1)
        assert prog1 == EXPECTED_RESULT_1
        prog2 = _trial_progress_str(trials, [], None, fmt='psql', max_rows=None, force_table=True)
        print(prog2)
        assert prog2 == EXPECTED_RESULT_2
        prog3 = _trial_progress_str(trials, {'nested/sub': 'NestSub', 'metric_2': 'Metric 2'}, {'a': 'A'}, fmt='psql', max_rows=3, force_table=True)
        print(prog3)
        assert prog3 == EXPECTED_RESULT_3
        best1 = _best_trial_str(trials[1], 'metric_1')
        assert best1 == EXPECTED_BEST_1

    def testBestTrialStr(self):
        if False:
            i = 10
            return i + 15
        'Assert that custom nested parameter columns are printed correctly'
        config = {'nested': {'conf': 'nested_value'}, 'toplevel': 'toplevel_value'}
        trial = Trial('', config=config, stub=True)
        trial.run_metadata.last_result = {'metric': 1, 'config': config, 'nested': {'metric': 2}}
        result = _best_trial_str(trial, 'metric')
        self.assertIn('nested_value', result)
        result = _best_trial_str(trial, 'metric', parameter_columns=['nested/conf'])
        self.assertIn('nested_value', result)
        result = _best_trial_str(trial, 'nested/metric', parameter_columns=['nested/conf'])
        self.assertIn('nested_value', result)

    def testBestTrialZero(self):
        if False:
            while True:
                i = 10
        trial1 = Trial('', config={}, stub=True)
        trial1.run_metadata.last_result = {'metric': 7, 'config': {}}
        trial2 = Trial('', config={}, stub=True)
        trial2.run_metadata.last_result = {'metric': 0, 'config': {}}
        trial3 = Trial('', config={}, stub=True)
        trial3.run_metadata.last_result = {'metric': 2, 'config': {}}
        reporter = TuneReporterBase(metric='metric', mode='min')
        (best_trial, metric) = reporter._current_best_trial([trial1, trial2, trial3])
        assert best_trial == trial2

    def testBestTrialNan(self):
        if False:
            while True:
                i = 10
        trial1 = Trial('', config={}, stub=True)
        trial1.run_metadata.last_result = {'metric': np.nan, 'config': {}}
        trial2 = Trial('', config={}, stub=True)
        trial2.run_metadata.last_result = {'metric': 0, 'config': {}}
        trial3 = Trial('', config={}, stub=True)
        trial3.run_metadata.last_result = {'metric': 2, 'config': {}}
        reporter = TuneReporterBase(metric='metric', mode='min')
        (best_trial, metric) = reporter._current_best_trial([trial1, trial2, trial3])
        assert best_trial == trial2
        trial1 = Trial('', config={}, stub=True)
        trial1.run_metadata.last_result = {'metric': np.nan, 'config': {}}
        trial2 = Trial('', config={}, stub=True)
        trial2.run_metadata.last_result = {'metric': 0, 'config': {}}
        trial3 = Trial('', config={}, stub=True)
        trial3.run_metadata.last_result = {'metric': 2, 'config': {}}
        reporter = TuneReporterBase(metric='metric', mode='max')
        (best_trial, metric) = reporter._current_best_trial([trial1, trial2, trial3])
        assert best_trial == trial3

    def testTimeElapsed(self):
        if False:
            while True:
                i = 10
        time_start = 1454825920
        time_now = time_start + 1 * 60 * 60 + 31 * 60 + 22
        output = _time_passed_str(time_start, time_now)
        self.assertIn('Current time: 2016-02-', output)
        self.assertIn(':50:02 (running for 01:31:22.00)', output)
        time_now += 2 * 60 * 60 * 24
        output = _time_passed_str(time_start, time_now)
        self.assertIn('Current time: 2016-02-', output)
        self.assertIn(':50:02 (running for 2 days, 01:31:22.00)', output)

    def testCurrentBestTrial(self):
        if False:
            while True:
                i = 10
        trials = []
        for i in range(5):
            t = Mock()
            t.status = 'RUNNING'
            t.trial_id = '%05d' % i
            t.local_experiment_path = '/foo'
            t.temporary_state = Mock()
            t.temporary_state.location = 'here'
            t.config = {'a': i, 'b': i * 2, 'n': {'k': [i, 2 * i]}}
            t.evaluated_params = {'a': i}
            t.last_result = {'config': {'a': i}, 'metric_1': i / 2}
            t.__str__ = lambda self: self.trial_id
            trials.append(t)

        class TestReporter(CLIReporter):
            _output = []

            def __init__(self, *args, **kwargs):
                if False:
                    print('Hello World!')
                super().__init__(*args, **kwargs)
                self._max_report_freqency = 0

            def report(self, *args, **kwargs):
                if False:
                    print('Hello World!')
                progress_str = self._progress_str(*args, **kwargs)
                self._output.append(progress_str)
        reporter = TestReporter(mode='max')
        reporter.report(trials, done=False)
        assert EXPECTED_BEST_2 in reporter._output[0]

    def testSortByMetric(self):
        if False:
            i = 10
            return i + 15
        trials = []
        for i in range(5):
            t = Mock()
            if i < 3:
                t.status = 'TERMINATED'
            elif i == 3:
                t.status = 'PENDING'
            else:
                t.status = 'RUNNING'
            t.trial_id = '%05d' % i
            t.local_experiment_path = '/foo'
            t.temporary_state = Mock()
            t.temporary_state.location = 'here'
            t.run_metadata = Mock()
            t.config = {'a': i}
            t.evaluated_params = {'a': i}
            t.last_result = {'config': {'a': i}}
            t.__str__ = lambda self: self.trial_id
            trials.append(t)
        trials[0].last_result['metric_1'] = 0.3
        trials[0].last_result['nested'] = {'metric_2': 0.3}
        trials[1].last_result['metric_1'] = 0.2
        trials[1].last_result['nested'] = {'metric_2': 0.2}
        trials[2].last_result['metric_1'] = 0.4
        trials[2].last_result['nested'] = {'metric_2': 0.4}

        class TestReporter(CLIReporter):

            def __init__(self, *args, **kwargs):
                if False:
                    return 10
                super().__init__(*args, **kwargs)
                self._max_report_freqency = 0
                self._output = ''

            def report(self, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                progress_str = self._progress_str(*args, **kwargs)
                self._output = progress_str
        reporter1 = TestReporter(max_progress_rows=4, mode='max', metric='metric_1')
        reporter1.report(trials, done=False)
        assert EXPECTED_SORT_RESULT_UNSORTED in reporter1._output
        reporter2 = TestReporter(max_progress_rows=4, mode='min', metric='metric_1', sort_by_metric=True)
        reporter2.report(trials, done=False)
        assert EXPECTED_SORT_RESULT_ASC in reporter2._output
        reporter3 = TestReporter(max_progress_rows=4, mode='max', metric='metric_1', sort_by_metric=True)
        reporter3.report(trials, done=False)
        assert EXPECTED_SORT_RESULT_DESC in reporter3._output
        reporter4 = TestReporter(max_progress_rows=4, metric='metric_1', sort_by_metric=True)
        reporter4.report(trials, done=False)
        assert EXPECTED_SORT_RESULT_UNSORTED in reporter4._output
        reporter5 = TestReporter(max_progress_rows=4, mode='max', sort_by_metric=True)
        reporter5.report(trials, done=False)
        assert EXPECTED_SORT_RESULT_UNSORTED in reporter5._output
        reporter6 = TestReporter(max_progress_rows=4, sort_by_metric=True)
        reporter6.set_search_properties(metric='metric_1', mode='max')
        reporter6.report(trials, done=False)
        assert EXPECTED_SORT_RESULT_DESC in reporter6._output
        reporter7 = TestReporter(max_progress_rows=4, mode='min', metric='nested/metric_2', sort_by_metric=True, metric_columns=['nested/metric_2'])
        reporter7.report(trials, done=False)
        assert EXPECTED_NESTED_SORT_RESULT in reporter7._output

    def testEndToEndReporting(self):
        if False:
            i = 10
            return i + 15
        try:
            os.environ['_TEST_TUNE_TRIAL_UUID'] = 'xxxxx'
            os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = '100'
            output = run_string_as_driver(END_TO_END_COMMAND)
            try:
                if os.environ.get('TUNE_NEW_EXECUTION') == '0':
                    assert EXPECTED_END_TO_END_START in output
                assert EXPECTED_END_TO_END_END in output
                for line in output.splitlines():
                    if '(raylet)' in line:
                        assert 'cluster ID' in line, 'Unexpected raylet log messages'
            except Exception:
                print('*** BEGIN OUTPUT ***')
                print(output)
                print('*** END OUTPUT ***')
                raise
        finally:
            del os.environ['_TEST_TUNE_TRIAL_UUID']

    def testVerboseReporting(self):
        if False:
            return 10
        try:
            os.environ['_TEST_TUNE_TRIAL_UUID'] = 'xxxxx'
            verbose_0_cmd = VERBOSE_CMD + 'verbose=0)'
            output = run_string_as_driver(verbose_0_cmd)
            try:
                self.assertNotIn(VERBOSE_EXP_OUT_1, output)
                self.assertNotIn(VERBOSE_EXP_OUT_2, output)
                self.assertNotIn(VERBOSE_TRIAL_NORM_1, output)
                self.assertIsNone(re.search(VERBOSE_TRIAL_NORM_2_PATTERN, output))
                self.assertNotIn(VERBOSE_TRIAL_NORM_3, output)
                self.assertNotIn(VERBOSE_TRIAL_NORM_4, output)
                if os.environ.get('TUNE_NEW_EXECUTION') == '0':
                    self.assertNotIn(VERBOSE_TRIAL_DETAIL, output)
            except Exception:
                print('*** BEGIN OUTPUT ***')
                print(output)
                print('*** END OUTPUT ***')
                raise
            verbose_1_cmd = VERBOSE_CMD + 'verbose=1)'
            output = run_string_as_driver(verbose_1_cmd)
            try:
                if os.environ.get('TUNE_NEW_EXECUTION') == '0':
                    self.assertIn(VERBOSE_EXP_OUT_1, output)
                self.assertIn(VERBOSE_EXP_OUT_2, output)
                self.assertNotIn(VERBOSE_TRIAL_NORM_1, output)
                self.assertIsNone(re.search(VERBOSE_TRIAL_NORM_2_PATTERN, output))
                self.assertNotIn(VERBOSE_TRIAL_NORM_3, output)
                self.assertNotIn(VERBOSE_TRIAL_NORM_4, output)
                if os.environ.get('TUNE_NEW_EXECUTION') == '0':
                    self.assertNotIn(VERBOSE_TRIAL_DETAIL, output)
            except Exception:
                print('*** BEGIN OUTPUT ***')
                print(output)
                print('*** END OUTPUT ***')
                raise
            verbose_2_cmd = VERBOSE_CMD + 'verbose=2)'
            output = run_string_as_driver(verbose_2_cmd)
            try:
                if os.environ.get('TUNE_NEW_EXECUTION') == '0':
                    self.assertIn(VERBOSE_EXP_OUT_1, output)
                self.assertIn(VERBOSE_EXP_OUT_2, output)
                self.assertIn(VERBOSE_TRIAL_NORM_1, output)
                self.assertIsNotNone(re.search(VERBOSE_TRIAL_NORM_2_PATTERN, output))
                self.assertIn(VERBOSE_TRIAL_NORM_3, output)
                self.assertIn(VERBOSE_TRIAL_NORM_4, output)
                self.assertNotIn(VERBOSE_TRIAL_DETAIL, output)
            except Exception:
                print('*** BEGIN OUTPUT ***')
                print(output)
                print('*** END OUTPUT ***')
                raise
            verbose_3_cmd = VERBOSE_CMD + 'verbose=3)'
            output = run_string_as_driver(verbose_3_cmd)
            try:
                if os.environ.get('TUNE_NEW_EXECUTION') == '0':
                    self.assertIn(VERBOSE_EXP_OUT_1, output)
                self.assertIn(VERBOSE_EXP_OUT_2, output)
                self.assertNotIn(VERBOSE_TRIAL_NORM_1, output)
                self.assertIsNone(re.search(VERBOSE_TRIAL_NORM_2_PATTERN, output))
                self.assertNotIn(VERBOSE_TRIAL_NORM_3, output)
                self.assertNotIn(VERBOSE_TRIAL_NORM_4, output)
                if os.environ.get('TUNE_NEW_EXECUTION') == '0':
                    self.assertIn(VERBOSE_TRIAL_DETAIL, output)
                self.assertTrue(output.count(VERBOSE_TRIAL_WITH_ONCE_RESULT) == 1)
                self.assertIn(VERBOSE_TRIAL_WITH_ONCE_COMPLETED, output)
            except Exception:
                print('*** BEGIN OUTPUT ***')
                print(output)
                print('*** END OUTPUT ***')
                raise
        finally:
            del os.environ['_TEST_TUNE_TRIAL_UUID']

    def testReporterDetection(self):
        if False:
            for i in range(10):
                print('nop')
        'Test if correct reporter is returned from ``detect_reporter()``'
        reporter = _detect_reporter()
        self.assertTrue(isinstance(reporter, CLIReporter))
        self.assertFalse(isinstance(reporter, JupyterNotebookReporter))
        with patch('ray.tune.progress_reporter.IS_NOTEBOOK', True):
            reporter = _detect_reporter()
            self.assertFalse(isinstance(reporter, CLIReporter))
            self.assertTrue(isinstance(reporter, JupyterNotebookReporter))

    def testProgressReporterAPI(self):
        if False:
            return 10

        class CustomReporter(ProgressReporter):

            def should_report(self, trials, done=False):
                if False:
                    return 10
                return True

            def report(self, trials, done, *sys_info):
                if False:
                    while True:
                        i = 10
                pass
        tune.run(lambda config: 2, num_samples=1, progress_reporter=CustomReporter(), verbose=3)

    def testMaxLen(self):
        if False:
            print('Hello World!')
        trials = []
        for i in range(5):
            t = Mock()
            t.status = 'TERMINATED'
            t.trial_id = '%05d' % i
            t.local_experiment_path = '/foo'
            t.temporary_state = Mock()
            t.temporary_state.location = 'here'
            t.config = {'verylong' * 20: i}
            t.evaluated_params = {'verylong' * 20: i}
            t.last_result = {'some_metric': 'evenlonger' * 100}
            t.__str__ = lambda self: self.trial_id
            trials.append(t)
        progress_str = _trial_progress_str(trials, metric_columns=['some_metric'], force_table=True)
        assert any((len(row) <= 90 for row in progress_str.split('\n')))

def test_max_len():
    if False:
        i = 10
        return i + 15
    assert _max_len('some_long_string/even_longer', max_len=28) == 'some_long_string/even_longer'
    assert _max_len('some_long_string/even_longer', max_len=15) == '.../even_longer'
    assert _max_len('19_character_string/19_character_string/too_long', max_len=20, wrap=True) == '...r_string/19_chara\ncter_string/too_long'
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))