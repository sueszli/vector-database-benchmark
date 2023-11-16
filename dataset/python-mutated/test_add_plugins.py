from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
if TYPE_CHECKING:
    from collections.abc import Mapping
import pytest
from poetry.core.packages.package import Package
from poetry.console.commands.add import AddCommand
from poetry.console.commands.self.self_command import SelfCommand
from poetry.factory import Factory
from tests.console.commands.self.utils import get_self_command_dependencies
if TYPE_CHECKING:
    from cleo.testers.command_tester import CommandTester
    from tests.helpers import TestRepository
    from tests.types import CommandTesterFactory

@pytest.fixture()
def tester(command_tester_factory: CommandTesterFactory) -> CommandTester:
    if False:
        return 10
    return command_tester_factory('self add')

def assert_plugin_add_result(tester: CommandTester, expected: str, constraint: str | Mapping[str, str | list[str]]) -> None:
    if False:
        print('Hello World!')
    assert tester.io.fetch_output() == expected
    dependencies: dict[str, Any] = get_self_command_dependencies()
    assert 'poetry-plugin' in dependencies
    assert dependencies['poetry-plugin'] == constraint

def test_add_no_constraint(tester: CommandTester, repo: TestRepository) -> None:
    if False:
        while True:
            i = 10
    repo.add_package(Package('poetry-plugin', '0.1.0'))
    tester.execute('poetry-plugin')
    expected = 'Using version ^0.1.0 for poetry-plugin\n\nUpdating dependencies\nResolving dependencies...\n\nPackage operations: 1 install, 0 updates, 0 removals\n\n  - Installing poetry-plugin (0.1.0)\n\nWriting lock file\n'
    assert_plugin_add_result(tester, expected, '^0.1.0')

def test_add_with_constraint(tester: CommandTester, repo: TestRepository) -> None:
    if False:
        i = 10
        return i + 15
    repo.add_package(Package('poetry-plugin', '0.1.0'))
    repo.add_package(Package('poetry-plugin', '0.2.0'))
    tester.execute('poetry-plugin@^0.2.0')
    expected = '\nUpdating dependencies\nResolving dependencies...\n\nPackage operations: 1 install, 0 updates, 0 removals\n\n  - Installing poetry-plugin (0.2.0)\n\nWriting lock file\n'
    assert_plugin_add_result(tester, expected, '^0.2.0')

def test_add_with_git_constraint(tester: CommandTester, repo: TestRepository) -> None:
    if False:
        for i in range(10):
            print('nop')
    repo.add_package(Package('pendulum', '2.0.5'))
    tester.execute('git+https://github.com/demo/poetry-plugin.git')
    expected = '\nUpdating dependencies\nResolving dependencies...\n\nPackage operations: 2 installs, 0 updates, 0 removals\n\n  - Installing pendulum (2.0.5)\n  - Installing poetry-plugin (0.1.2 9cf87a2)\n\nWriting lock file\n'
    assert_plugin_add_result(tester, expected, {'git': 'https://github.com/demo/poetry-plugin.git'})

def test_add_with_git_constraint_with_extras(tester: CommandTester, repo: TestRepository) -> None:
    if False:
        i = 10
        return i + 15
    repo.add_package(Package('pendulum', '2.0.5'))
    repo.add_package(Package('tomlkit', '0.7.0'))
    tester.execute('git+https://github.com/demo/poetry-plugin.git[foo]')
    expected = '\nUpdating dependencies\nResolving dependencies...\n\nPackage operations: 3 installs, 0 updates, 0 removals\n\n  - Installing pendulum (2.0.5)\n  - Installing tomlkit (0.7.0)\n  - Installing poetry-plugin (0.1.2 9cf87a2)\n\nWriting lock file\n'
    constraint: dict[str, str | list[str]] = {'git': 'https://github.com/demo/poetry-plugin.git', 'extras': ['foo']}
    assert_plugin_add_result(tester, expected, constraint)

@pytest.mark.parametrize('url, rev', [('git+https://github.com/demo/poetry-plugin2.git#subdirectory=subdir', None), ('git+https://github.com/demo/poetry-plugin2.git@master#subdirectory=subdir', 'master')])
def test_add_with_git_constraint_with_subdirectory(url: str, rev: str | None, tester: CommandTester, repo: TestRepository) -> None:
    if False:
        i = 10
        return i + 15
    repo.add_package(Package('pendulum', '2.0.5'))
    tester.execute(url)
    expected = '\nUpdating dependencies\nResolving dependencies...\n\nPackage operations: 2 installs, 0 updates, 0 removals\n\n  - Installing pendulum (2.0.5)\n  - Installing poetry-plugin (0.1.2 9cf87a2)\n\nWriting lock file\n'
    constraint = {'git': 'https://github.com/demo/poetry-plugin2.git', 'subdirectory': 'subdir'}
    if rev:
        constraint['rev'] = rev
    assert_plugin_add_result(tester, expected, constraint)

def test_add_existing_plugin_warns_about_no_operation(tester: CommandTester, repo: TestRepository, installed: TestRepository) -> None:
    if False:
        i = 10
        return i + 15
    pyproject = SelfCommand.get_default_system_pyproject_file()
    with open(pyproject, 'w', encoding='utf-8', newline='') as f:
        f.write(f'[tool.poetry]\nname = "poetry-instance"\nversion = "1.2.0"\ndescription = "Python dependency management and packaging made easy."\nauthors = []\n\n[tool.poetry.dependencies]\npython = "^3.6"\n\n[tool.poetry.group.{SelfCommand.ADDITIONAL_PACKAGE_GROUP}.dependencies]\npoetry-plugin = "^1.2.3"\n')
    installed.add_package(Package('poetry-plugin', '1.2.3'))
    repo.add_package(Package('poetry-plugin', '1.2.3'))
    tester.execute('poetry-plugin')
    assert isinstance(tester.command, AddCommand)
    expected = f'The following packages are already present in the pyproject.toml and will be skipped:\n\n  - poetry-plugin\n{tester.command._hint_update_packages}\nNothing to add.\n'
    assert tester.io.fetch_output() == expected

def test_add_existing_plugin_updates_if_requested(tester: CommandTester, repo: TestRepository, installed: TestRepository) -> None:
    if False:
        for i in range(10):
            print('nop')
    pyproject = SelfCommand.get_default_system_pyproject_file()
    with open(pyproject, 'w', encoding='utf-8', newline='') as f:
        f.write(f'[tool.poetry]\nname = "poetry-instance"\nversion = "1.2.0"\ndescription = "Python dependency management and packaging made easy."\nauthors = []\n\n[tool.poetry.dependencies]\npython = "^3.6"\n\n[tool.poetry.group.{SelfCommand.ADDITIONAL_PACKAGE_GROUP}.dependencies]\npoetry-plugin = "^1.2.3"\n')
    installed.add_package(Package('poetry-plugin', '1.2.3'))
    repo.add_package(Package('poetry-plugin', '1.2.3'))
    repo.add_package(Package('poetry-plugin', '2.3.4'))
    tester.execute('poetry-plugin@latest')
    expected = 'Using version ^2.3.4 for poetry-plugin\n\nUpdating dependencies\nResolving dependencies...\n\nPackage operations: 0 installs, 1 update, 0 removals\n\n  - Updating poetry-plugin (1.2.3 -> 2.3.4)\n\nWriting lock file\n'
    assert_plugin_add_result(tester, expected, '^2.3.4')

def test_adding_a_plugin_can_update_poetry_dependencies_if_needed(tester: CommandTester, repo: TestRepository, installed: TestRepository) -> None:
    if False:
        for i in range(10):
            print('nop')
    poetry_package = Package('poetry', '1.2.0')
    poetry_package.add_dependency(Factory.create_dependency('tomlkit', '^0.7.0'))
    plugin_package = Package('poetry-plugin', '1.2.3')
    plugin_package.add_dependency(Factory.create_dependency('tomlkit', '^0.7.2'))
    installed.add_package(poetry_package)
    installed.add_package(Package('tomlkit', '0.7.1'))
    repo.add_package(plugin_package)
    repo.add_package(Package('tomlkit', '0.7.1'))
    repo.add_package(Package('tomlkit', '0.7.2'))
    tester.execute('poetry-plugin')
    expected = 'Using version ^1.2.3 for poetry-plugin\n\nUpdating dependencies\nResolving dependencies...\n\nPackage operations: 1 install, 1 update, 0 removals\n\n  - Updating tomlkit (0.7.1 -> 0.7.2)\n  - Installing poetry-plugin (1.2.3)\n\nWriting lock file\n'
    assert_plugin_add_result(tester, expected, '^1.2.3')