from __future__ import annotations
import subprocess
from typing import TYPE_CHECKING
import pytest
from poetry.utils._compat import WINDOWS
if TYPE_CHECKING:
    from cleo.testers.application_tester import ApplicationTester
    from cleo.testers.command_tester import CommandTester
    from pytest_mock import MockerFixture
    from poetry.poetry import Poetry
    from poetry.utils.env import MockEnv
    from poetry.utils.env import VirtualEnv
    from tests.types import CommandTesterFactory
    from tests.types import FixtureDirGetter
    from tests.types import ProjectFactory

@pytest.fixture
def tester(command_tester_factory: CommandTesterFactory) -> CommandTester:
    if False:
        i = 10
        return i + 15
    return command_tester_factory('run')

@pytest.fixture(autouse=True)
def patches(mocker: MockerFixture, env: MockEnv) -> None:
    if False:
        for i in range(10):
            print('nop')
    mocker.patch('poetry.utils.env.EnvManager.get', return_value=env)

@pytest.fixture
def poetry_with_scripts(project_factory: ProjectFactory, fixture_dir: FixtureDirGetter) -> Poetry:
    if False:
        print('Hello World!')
    source = fixture_dir('scripts')
    return project_factory(name='scripts', pyproject_content=(source / 'pyproject.toml').read_text(encoding='utf-8'), source=source)

def test_run_passes_all_args(app_tester: ApplicationTester, env: MockEnv) -> None:
    if False:
        i = 10
        return i + 15
    app_tester.execute('run python -V')
    assert [['python', '-V']] == env.executed

def test_run_keeps_options_passed_before_command(app_tester: ApplicationTester, env: MockEnv) -> None:
    if False:
        while True:
            i = 10
    app_tester.execute('-V --no-ansi run python', decorated=True)
    assert not app_tester.io.is_decorated()
    assert app_tester.io.fetch_output() == app_tester.io.remove_format(app_tester.application.long_version + '\n')
    assert [] == env.executed

def test_run_has_helpful_error_when_command_not_found(app_tester: ApplicationTester, env: MockEnv, capfd: pytest.CaptureFixture[str]) -> None:
    if False:
        for i in range(10):
            print('nop')
    nonexistent_command = 'nonexistent-command'
    env._execute = True
    app_tester.execute(f'run {nonexistent_command}')
    assert env.executed == [[nonexistent_command]]
    assert app_tester.status_code == 1
    if WINDOWS:
        assert nonexistent_command in capfd.readouterr().err
    else:
        assert app_tester.io.fetch_error() == f'Command not found: {nonexistent_command}\n'

@pytest.mark.skipif(not WINDOWS, reason='Poetry only installs CMD script files for console scripts of editable dependencies on Windows')
def test_run_console_scripts_of_editable_dependencies_on_windows(tmp_venv: VirtualEnv, command_tester_factory: CommandTesterFactory) -> None:
    if False:
        return 10
    "\n    On Windows, Poetry installs console scripts of editable dependencies by creating\n    in the environment's `Scripts/` directory both:\n\n        A) a Python file named after the console script (no `.py` extension) which\n            imports and calls the console script using Python code\n        B) a CMD script file also named after the console script\n            (with `.cmd` extension) which calls `python.exe` to execute (A)\n\n    This configuration enables calling the console script by name from `cmd.exe`\n    because the `.cmd` file extension appears by default in the PATHEXT environment\n    variable that `cmd.exe` uses to determine which file should be executed if a\n    filename without an extension is executed as a command.\n\n    This test validates that you can also run such a CMD script file via `poetry run`\n    just by providing the script's name without the `.cmd` extension.\n    "
    tester = command_tester_factory('run', environment=tmp_venv)
    cmd_script_file = tmp_venv._bin_dir / 'quix.cmd'
    cmd_script_file.write_text('exit /b 123')
    assert tester.execute('quix') == 123

def test_run_script_exit_code(poetry_with_scripts: Poetry, command_tester_factory: CommandTesterFactory, tmp_venv: VirtualEnv, mocker: MockerFixture) -> None:
    if False:
        return 10
    mocker.patch('os.execvpe', lambda file, args, env: subprocess.call([file] + args[1:], env=env))
    install_tester = command_tester_factory('install', poetry=poetry_with_scripts, environment=tmp_venv)
    assert install_tester.execute() == 0
    tester = command_tester_factory('run', poetry=poetry_with_scripts, environment=tmp_venv)
    assert tester.execute('exit-code') == 42
    assert tester.execute('return-code') == 42

@pytest.mark.parametrize('installed_script', [False, True], ids=['not installed', 'installed'])
def test_run_script_sys_argv0(installed_script: bool, poetry_with_scripts: Poetry, command_tester_factory: CommandTesterFactory, tmp_venv: VirtualEnv, mocker: MockerFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    If RunCommand calls an installed script defined in pyproject.toml,\n    sys.argv[0] must be set to the full path of the script.\n    '
    mocker.patch('poetry.utils.env.EnvManager.get', return_value=tmp_venv)
    mocker.patch('os.execvpe', lambda file, args, env: subprocess.call([file] + args[1:], env=env))
    install_tester = command_tester_factory('install', poetry=poetry_with_scripts, environment=tmp_venv)
    assert install_tester.execute() == 0
    if not installed_script:
        for path in tmp_venv.script_dirs[0].glob('check-argv0*'):
            path.unlink()
    tester = command_tester_factory('run', poetry=poetry_with_scripts, environment=tmp_venv)
    argv1 = 'absolute' if installed_script else 'relative'
    assert tester.execute(f'check-argv0 {argv1}') == 0
    if installed_script:
        expected_message = ''
    else:
        expected_message = "Warning: 'check-argv0' is an entry point defined in pyproject.toml, but it's not installed as a script. You may get improper `sys.argv[0]`.\n\nThe support to run uninstalled scripts will be removed in a future release.\n\nRun `poetry install` to resolve and get rid of this message.\n\n"
    assert tester.io.fetch_error() == expected_message