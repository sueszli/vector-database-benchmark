import os
from pathlib import Path
from pip import __version__
from tests.lib import PipTestEnvironment

def test_runner_work_in_environments_with_no_pip(script: PipTestEnvironment, pip_src: Path) -> None:
    if False:
        i = 10
        return i + 15
    runner = pip_src / 'src' / 'pip' / '__pip-runner__.py'
    script.pip('uninstall', 'pip', '--yes', use_module=True)
    script.run('python', '-c', 'import pip', expect_error=True)
    result = script.run('python', os.fspath(runner), '--version')
    assert __version__ in result.stdout