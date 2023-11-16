import os
import sys
import json
import pytest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from recommenders.tuning.nni.nni_utils import get_experiment_status, check_experiment_status, check_stopped, check_metrics_written, get_trials, NNI_STATUS_URL, NNI_TRIAL_JOBS_URL

class MockResponse:

    def __init__(self, content, error):
        if False:
            for i in range(10):
                print('nop')
        self._content = content
        self._error = error

    def json(self):
        if False:
            return 10
        return {'status': self._content, 'errors': [self._error]}

def mocked_status_get(url, content, error):
    if False:
        i = 10
        return i + 15
    assert url.startswith(NNI_STATUS_URL)
    return MockResponse(content, error)

class MockResponseTrials:

    def __init__(self, content):
        if False:
            return 10
        self._content = content

    def json(self):
        if False:
            while True:
                i = 10
        return self._content

def mocked_trials_get(url, content):
    if False:
        print('Hello World!')
    assert url.startswith(NNI_TRIAL_JOBS_URL)
    return MockResponseTrials(content)

def mock_exception():
    if False:
        for i in range(10):
            print('nop')
    raise Exception()

@pytest.mark.experimental
@pytest.mark.skipif(sys.platform == 'win32', reason='nni not installable on windows')
def test_get_experiment_status():
    if False:
        i = 10
        return i + 15
    content = 'some_status'
    error = ''
    with patch('requests.get', side_effect=lambda url: mocked_status_get(url, content, error)):
        nni_status = get_experiment_status(NNI_STATUS_URL)
        assert nni_status['status'] == 'some_status'
        assert nni_status['errors'] == ['']

@pytest.mark.experimental
@pytest.mark.skipif(sys.platform == 'win32', reason='nni not installable on windows')
def test_check_experiment_status_done():
    if False:
        i = 10
        return i + 15
    content = 'DONE'
    error = ''
    with patch('requests.get', side_effect=lambda url: mocked_status_get(url, content, error)):
        check_experiment_status(wait=0.1, max_retries=1)

@pytest.mark.experimental
@pytest.mark.skipif(sys.platform == 'win32', reason='nni not installable on windows')
def test_check_experiment_status_tuner_no_more_trial():
    if False:
        return 10
    content = 'TUNER_NO_MORE_TRIAL'
    error = ''
    with patch('requests.get', side_effect=lambda url: mocked_status_get(url, content, error)):
        check_experiment_status(wait=0.1, max_retries=1)

@pytest.mark.experimental
@pytest.mark.skipif(sys.platform == 'win32', reason='nni not installable on windows')
def test_check_experiment_status_running():
    if False:
        i = 10
        return i + 15
    content = 'RUNNING'
    error = ''
    with pytest.raises(TimeoutError) as excinfo:
        with patch('requests.get', side_effect=lambda url: mocked_status_get(url, content, error)):
            check_experiment_status(wait=0.1, max_retries=1)
    assert 'check_experiment_status() timed out' == str(excinfo.value)

@pytest.mark.experimental
@pytest.mark.skipif(sys.platform == 'win32', reason='nni not installable on windows')
def test_check_experiment_status_no_more_trial():
    if False:
        while True:
            i = 10
    content = 'NO_MORE_TRIAL'
    error = ''
    with pytest.raises(TimeoutError) as excinfo:
        with patch('requests.get', side_effect=lambda url: mocked_status_get(url, content, error)):
            check_experiment_status(wait=0.1, max_retries=1)
    assert 'check_experiment_status() timed out' == str(excinfo.value)

@pytest.mark.experimental
@pytest.mark.skipif(sys.platform == 'win32', reason='nni not installable on windows')
def test_check_experiment_status_failed():
    if False:
        for i in range(10):
            print('nop')
    content = 'some_failed_status'
    error = 'NNI_ERROR'
    with pytest.raises(RuntimeError) as excinfo:
        with patch('requests.get', side_effect=lambda url: mocked_status_get(url, content, error)):
            check_experiment_status(wait=0.1, max_retries=1)
    assert 'NNI experiment failed to complete with status some_failed_status - NNI_ERROR' == str(excinfo.value)

@pytest.mark.experimental
@pytest.mark.skipif(sys.platform == 'win32', reason='nni not installable on windows')
def test_check_stopped_timeout():
    if False:
        for i in range(10):
            print('nop')
    content = 'some_status'
    error = ''
    with pytest.raises(TimeoutError) as excinfo:
        with patch('requests.get', side_effect=lambda url: mocked_status_get(url, content, error)):
            check_stopped(wait=0.1, max_retries=1)
    assert 'check_stopped() timed out' == str(excinfo.value)

@pytest.mark.experimental
@pytest.mark.skipif(sys.platform == 'win32', reason='nni not installable on windows')
def test_check_stopped():
    if False:
        return 10
    with patch('requests.get', side_effect=mock_exception):
        check_stopped(wait=0.1, max_retries=1)

@pytest.mark.experimental
@pytest.mark.skipif(sys.platform == 'win32', reason='nni not installable on windows')
def test_check_metrics_written():
    if False:
        i = 10
        return i + 15
    content = [{'finalMetricData': None}, {'finalMetricData': None}]
    with patch('requests.get', side_effect=lambda url: mocked_trials_get(url, content)):
        check_metrics_written(wait=0.1, max_retries=1)

@pytest.mark.experimental
@pytest.mark.skipif(sys.platform == 'win32', reason='nni not installable on windows')
def test_check_metrics_written_timeout():
    if False:
        for i in range(10):
            print('nop')
    content = [{'logPath': '/p'}, {'logPath': '/q'}]
    with pytest.raises(TimeoutError) as excinfo:
        with patch('requests.get', side_effect=lambda url: mocked_trials_get(url, content)):
            check_metrics_written(wait=0.1, max_retries=1)
    assert 'check_metrics_written() timed out' == str(excinfo.value)

@pytest.mark.experimental
@pytest.mark.skipif(sys.platform == 'win32', reason='nni not installable on windows')
def test_get_trials():
    if False:
        for i in range(10):
            print('nop')
    with TemporaryDirectory() as tmp_dir1, TemporaryDirectory() as tmp_dir2:
        mock_trials = [{'finalMetricData': [{'data': '"{\\"rmse\\": 0.8, \\"default\\": 0.3}"'}], 'logPath': 'file://localhost:{}'.format(tmp_dir1)}, {'finalMetricData': [{'data': '"{\\"rmse\\": 0.9, \\"default\\": 0.2}"'}], 'logPath': 'file://localhost:{}'.format(tmp_dir2)}]
        metrics1 = {'rmse': 0.8, 'precision_at_k': 0.3}
        with open(os.path.join(tmp_dir1, 'metrics.json'), 'w') as f:
            json.dump(metrics1, f)
        params1 = {'parameter_id': 1, 'parameter_source': 'algorithm', 'parameters': {'n_factors': 100, 'reg': 0.1}}
        with open(os.path.join(tmp_dir1, 'parameter.cfg'), 'w') as f:
            json.dump(params1, f)
        metrics2 = {'rmse': 0.9, 'precision_at_k': 0.2}
        with open(os.path.join(tmp_dir2, 'metrics.json'), 'w') as f:
            json.dump(metrics2, f)
        params2 = {'parameter_id': 2, 'parameter_source': 'algorithm', 'parameters': {'n_factors': 50, 'reg': 0.02}}
        with open(os.path.join(tmp_dir2, 'parameter.cfg'), 'w') as f:
            json.dump(params2, f)
        with patch('requests.get', side_effect=lambda url: mocked_trials_get(url, mock_trials)):
            (trials, best_metrics, best_params, best_trial_path) = get_trials(optimize_mode='maximize')
        expected_trials = [({'rmse': 0.8, 'default': 0.3}, tmp_dir1), ({'rmse': 0.9, 'default': 0.2}, tmp_dir2)]
        assert trials == expected_trials
        assert best_metrics == metrics1
        assert best_params == params1
        assert best_trial_path == tmp_dir1