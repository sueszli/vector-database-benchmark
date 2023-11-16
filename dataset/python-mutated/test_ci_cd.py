import mock
import pytest
from api.v1.api_pb2 import SDK_JAVA
from ci_cd import _check_envs, _run_ci_cd
from config import Origin, Config

@pytest.mark.parametrize('step', ['CI', 'CD'])
@mock.patch('ci_cd.DatastoreClient')
@mock.patch('ci_cd.find_examples')
@mock.patch('verify.Verifier._run_and_verify')
def test_ci_step(mock_run_and_verify, mock_find_examples, mock_datastore, create_test_example, step):
    if False:
        while True:
            i = 10
    mock_find_examples.return_value = [create_test_example(tag_meta=dict(name='Default', default_example=True)), create_test_example(tag_meta=dict(name='Single', multifile=False)), create_test_example(is_multifile=True, tag_meta=dict(name='Multi'))]
    _run_ci_cd(step, 'SDK_JAVA', Origin.PG_EXAMPLES, 'test', Config.DEFAULT_NAMESPACE, ['../../examples'])
    mock_run_and_verify.assert_called_once()
    if step == 'CD':
        mock_datastore.assert_called_once()

def test__check_envs():
    if False:
        return 10
    with pytest.raises(KeyError):
        _check_envs()