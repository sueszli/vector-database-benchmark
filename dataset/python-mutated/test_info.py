from __future__ import annotations
import sys
from pathlib import Path
from typing import TYPE_CHECKING
import pytest
from poetry.utils.env import MockEnv
if TYPE_CHECKING:
    from cleo.testers.command_tester import CommandTester
    from pytest_mock import MockerFixture
    from tests.types import CommandTesterFactory

@pytest.fixture(autouse=True)
def setup(mocker: MockerFixture) -> None:
    if False:
        while True:
            i = 10
    mocker.patch('poetry.utils.env.EnvManager.get', return_value=MockEnv(path=Path('/prefix'), base=Path('/base/prefix'), is_venv=True))

@pytest.fixture
def tester(command_tester_factory: CommandTesterFactory) -> CommandTester:
    if False:
        return 10
    return command_tester_factory('env info')

def test_env_info_displays_complete_info(tester: CommandTester) -> None:
    if False:
        for i in range(10):
            print('nop')
    tester.execute()
    expected = f"\nVirtualenv\nPython:         3.7.0\nImplementation: CPython\nPath:           {Path('/prefix')}\nExecutable:     {sys.executable}\nValid:          True\n\nSystem\nPlatform:   darwin\nOS:         posix\nPython:     {'.'.join((str(v) for v in sys.version_info[:3]))}\nPath:       {Path('/base/prefix')}\nExecutable: python\n"
    assert tester.io.fetch_output() == expected

def test_env_info_displays_path_only(tester: CommandTester) -> None:
    if False:
        while True:
            i = 10
    tester.execute('--path')
    expected = str(Path('/prefix')) + '\n'
    assert tester.io.fetch_output() == expected

def test_env_info_displays_executable_only(tester: CommandTester) -> None:
    if False:
        return 10
    tester.execute('--executable')
    expected = str(sys.executable) + '\n'
    assert tester.io.fetch_output() == expected