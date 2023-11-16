from __future__ import annotations
from typing import TYPE_CHECKING
import pytest
if TYPE_CHECKING:
    from pathlib import Path
    from cleo.testers.command_tester import CommandTester
    from tests.types import CommandTesterFactory

@pytest.fixture
def tester(command_tester_factory: CommandTesterFactory) -> CommandTester:
    if False:
        for i in range(10):
            print('nop')
    return command_tester_factory('cache list')

def test_cache_list(tester: CommandTester, mock_caches: None, repository_one: str, repository_two: str) -> None:
    if False:
        while True:
            i = 10
    tester.execute()
    expected = f'{repository_one}\n{repository_two}\n'
    assert tester.io.fetch_output() == expected

def test_cache_list_empty(tester: CommandTester, repository_cache_dir: Path) -> None:
    if False:
        i = 10
        return i + 15
    tester.execute()
    expected = 'No caches found\n'
    assert tester.io.fetch_error() == expected