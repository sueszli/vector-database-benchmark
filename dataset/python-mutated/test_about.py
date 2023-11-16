from __future__ import annotations
from typing import TYPE_CHECKING
import pytest
if TYPE_CHECKING:
    from cleo.testers.command_tester import CommandTester
    from tests.types import CommandTesterFactory

@pytest.fixture()
def tester(command_tester_factory: CommandTesterFactory) -> CommandTester:
    if False:
        return 10
    return command_tester_factory('about')

def test_about(tester: CommandTester) -> None:
    if False:
        print('Hello World!')
    from poetry.utils._compat import metadata
    tester.execute()
    expected = f"Poetry - Package Management for Python\n\nVersion: {metadata.version('poetry')}\nPoetry-Core Version: {metadata.version('poetry-core')}\n\nPoetry is a dependency manager tracking local dependencies of your projects and libraries.\nSee https://github.com/python-poetry/poetry for more information.\n"
    assert tester.io.fetch_output() == expected