"""
Test ext_nodes master_tops module
"""
import subprocess
import textwrap
import pytest
import salt.tops.ext_nodes as ext_nodes
import salt.utils.stringutils
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {ext_nodes: {'__opts__': {'master_tops': {'ext_nodes': 'echo'}}}}

def test_ext_nodes():
    if False:
        return 10
    '\n    Confirm that subprocess.Popen works as expected and does not raise an\n    exception (see https://github.com/saltstack/salt/pull/46863).\n    '
    stdout = salt.utils.stringutils.to_bytes(textwrap.dedent('        classes:\n            - one\n            - two'))
    run_mock = MagicMock()
    run_mock.return_value.stdout = stdout
    with patch.object(subprocess, 'run', run_mock):
        ret = ext_nodes.top(opts={'id': 'foo'})
    assert ret == {'base': ['one', 'two']}
    run_mock.assert_called_once_with(['echo', 'foo'], check=True, stdout=-1)

def test_ext_nodes_with_environment():
    if False:
        for i in range(10):
            print('nop')
    '\n    Same as above, but also tests that the matches are assigned to the proper\n    environment if one is returned by the ext_nodes command.\n    '
    stdout = salt.utils.stringutils.to_bytes(textwrap.dedent('        classes:\n            - one\n            - two\n        environment: dev'))
    run_mock = MagicMock()
    run_mock.return_value.stdout = stdout
    with patch.object(subprocess, 'run', run_mock):
        ret = ext_nodes.top(opts={'id': 'foo'})
    assert ret == {'dev': ['one', 'two']}
    run_mock.assert_called_once_with(['echo', 'foo'], check=True, stdout=-1)