from __future__ import annotations
import pytest
from ansible.executor.module_common import modify_module
FAKE_OLD_MODULE = b'#!/usr/bin/python\nimport sys\nprint(\'{"result": "%s"}\' % sys.executable)\n'

@pytest.fixture
def fake_old_module_open(mocker):
    if False:
        i = 10
        return i + 15
    m = mocker.mock_open(read_data=FAKE_OLD_MODULE)
    mocker.patch('builtins.open', m)

def test_shebang_task_vars(fake_old_module_open, templar):
    if False:
        while True:
            i = 10
    task_vars = {'ansible_python_interpreter': '/usr/bin/python3'}
    (data, style, shebang) = modify_module('fake_module', 'fake_path', {}, templar, task_vars=task_vars)
    assert shebang == '#!/usr/bin/python3'