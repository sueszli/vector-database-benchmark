from __future__ import annotations
from typing import TYPE_CHECKING
import pytest
from poetry.factory import Factory
from tests.helpers import get_package
if TYPE_CHECKING:
    from cleo.testers.command_tester import CommandTester
    from tests.helpers import TestRepository
    from tests.types import CommandTesterFactory

@pytest.fixture()
def tester(command_tester_factory: CommandTesterFactory) -> CommandTester:
    if False:
        for i in range(10):
            print('nop')
    return command_tester_factory('debug resolve')

@pytest.fixture(autouse=True)
def __add_packages(repo: TestRepository) -> None:
    if False:
        i = 10
        return i + 15
    cachy020 = get_package('cachy', '0.2.0')
    cachy020.add_dependency(Factory.create_dependency('msgpack-python', '>=0.5 <0.6'))
    repo.add_package(get_package('cachy', '0.1.0'))
    repo.add_package(cachy020)
    repo.add_package(get_package('msgpack-python', '0.5.3'))
    repo.add_package(get_package('pendulum', '2.0.3'))
    repo.add_package(get_package('cleo', '0.6.5'))

def test_debug_resolve_gives_resolution_results(tester: CommandTester) -> None:
    if False:
        for i in range(10):
            print('nop')
    tester.execute('cachy')
    expected = 'Resolving dependencies...\n\nResolution results:\n\nmsgpack-python 0.5.3\ncachy          0.2.0\n'
    assert tester.io.fetch_output() == expected

def test_debug_resolve_tree_option_gives_the_dependency_tree(tester: CommandTester) -> None:
    if False:
        print('Hello World!')
    tester.execute('cachy --tree')
    expected = 'Resolving dependencies...\n\nResolution results:\n\ncachy 0.2.0\n└── msgpack-python >=0.5 <0.6\n'
    assert tester.io.fetch_output() == expected

def test_debug_resolve_git_dependency(tester: CommandTester) -> None:
    if False:
        while True:
            i = 10
    tester.execute('git+https://github.com/demo/demo.git')
    expected = 'Resolving dependencies...\n\nResolution results:\n\npendulum 2.0.3\ndemo     0.1.2\n'
    assert tester.io.fetch_output() == expected