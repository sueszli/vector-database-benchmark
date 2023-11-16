import subprocess
from unittest import mock
import pytest
from tools.bump_action import main

@pytest.fixture
def workflow_and_action(tmp_path):
    if False:
        i = 10
        return i + 15
    base = tmp_path.joinpath('root')
    base.joinpath('.github/workflows').mkdir(parents=True)
    workflow = base.joinpath('.github/workflows/main.yml')
    base.joinpath('.github/actions/myaction').mkdir(parents=True)
    action = base.joinpath('.github/actions/myaction/action.yml')
    yield (base, workflow, action)

def test_main_noop(workflow_and_action, capsys):
    if False:
        return 10
    (base, workflow, action) = workflow_and_action
    workflow_src = 'name: main\non:\n  push:\njobs:\n  main:\n    runs-on: ubuntu-latest\n    steps:\n    - run: echo hi\n'
    action_src = 'name: my-action\ninputs:\n  arg:\n    required: true\n\nruns:\n  using: composite\n  steps:\n  - run: echo hi\n    shell: bash\n'
    workflow.write_text(workflow_src)
    action.write_text(action_src)
    assert main(('actions/whatever', 'v1.2.3', f'--base-dir={base}')) == 0
    (out, err) = capsys.readouterr()
    assert out == err == ''

def test_main_upgrades_action(workflow_and_action, capsys):
    if False:
        return 10
    (base, workflow, action) = workflow_and_action
    workflow_src = 'name: main\non:\n  push:\njobs:\n  main:\n    runs-on: ubuntu-latest\n    steps:\n    - uses: actions/whatever@v0.1.2\n'
    workflow_expected = 'name: main\non:\n  push:\njobs:\n  main:\n    runs-on: ubuntu-latest\n    steps:\n    - uses: actions/whatever@v1.2.3\n'
    action_src = 'name: my-action\ninputs:\n  arg:\n    required: true\n\nruns:\n  using: composite\n  steps:\n  - uses: actions/whatever@v0.1.2\n'
    action_expected = 'name: my-action\ninputs:\n  arg:\n    required: true\n\nruns:\n  using: composite\n  steps:\n  - uses: actions/whatever@v1.2.3\n'
    workflow.write_text(workflow_src)
    action.write_text(action_src)
    with mock.patch.object(subprocess, 'call', return_value=123):
        assert main(('actions/whatever', 'v1.2.3', f'--base-dir={base}')) == 123
    (out, err) = capsys.readouterr()
    assert out == f'{workflow} upgrading actions/whatever...\n{action} upgrading actions/whatever...\nfreezing...\n'
    assert workflow.read_text() == workflow_expected
    assert action.read_text() == action_expected