from __future__ import annotations
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
import pytest
from cleo.testers.command_tester import CommandTester
from packaging.utils import canonicalize_name
from poetry.core.utils.helpers import module_name
from poetry.console.application import Application
from poetry.console.commands.init import InitCommand
from poetry.repositories import RepositoryPool
from tests.helpers import get_package
if TYPE_CHECKING:
    from collections.abc import Iterator
    from _pytest.fixtures import FixtureRequest
    from poetry.core.packages.package import Package
    from pytest_mock import MockerFixture
    from poetry.config.config import Config
    from poetry.poetry import Poetry
    from tests.helpers import PoetryTestApplication
    from tests.helpers import TestRepository
    from tests.types import FixtureDirGetter

@pytest.fixture
def source_dir(tmp_path: Path) -> Iterator[Path]:
    if False:
        while True:
            i = 10
    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        yield tmp_path
    finally:
        os.chdir(cwd)

@pytest.fixture
def patches(mocker: MockerFixture, source_dir: Path, repo: TestRepository) -> None:
    if False:
        while True:
            i = 10
    mocker.patch('pathlib.Path.cwd', return_value=source_dir)
    mocker.patch('poetry.console.commands.init.InitCommand._get_pool', return_value=RepositoryPool([repo]))

@pytest.fixture
def tester(patches: None) -> CommandTester:
    if False:
        print('Hello World!')
    app = Application()
    return CommandTester(app.find('init'))

@pytest.fixture
def init_basic_inputs() -> str:
    if False:
        i = 10
        return i + 15
    return '\n'.join(['my-package', '1.2.3', 'This is a description', 'n', 'MIT', '~2.7 || ^3.6', 'n', 'n', '\n'])

@pytest.fixture()
def init_basic_toml() -> str:
    if False:
        print('Hello World!')
    return '[tool.poetry]\nname = "my-package"\nversion = "1.2.3"\ndescription = "This is a description"\nauthors = ["Your Name <you@example.com>"]\nlicense = "MIT"\nreadme = "README.md"\n\n[tool.poetry.dependencies]\npython = "~2.7 || ^3.6"\n'

def test_basic_interactive(tester: CommandTester, init_basic_inputs: str, init_basic_toml: str) -> None:
    if False:
        i = 10
        return i + 15
    tester.execute(inputs=init_basic_inputs)
    assert init_basic_toml in tester.io.fetch_output()

def test_noninteractive(app: PoetryTestApplication, mocker: MockerFixture, poetry: Poetry, repo: TestRepository, tmp_path: Path) -> None:
    if False:
        print('Hello World!')
    command = app.find('init')
    assert isinstance(command, InitCommand)
    command._pool = poetry.pool
    repo.add_package(get_package('pytest', '3.6.0'))
    p = mocker.patch('pathlib.Path.cwd')
    p.return_value = tmp_path
    tester = CommandTester(command)
    args = '--name my-package --dependency pytest'
    tester.execute(args=args, interactive=False)
    expected = 'Using version ^3.6.0 for pytest\n'
    assert tester.io.fetch_output() == expected
    assert tester.io.fetch_error() == ''
    toml_content = (tmp_path / 'pyproject.toml').read_text()
    assert 'name = "my-package"' in toml_content
    assert 'pytest = "^3.6.0"' in toml_content

def test_interactive_with_dependencies(tester: CommandTester, repo: TestRepository) -> None:
    if False:
        while True:
            i = 10
    repo.add_package(get_package('django-pendulum', '0.1.6-pre4'))
    repo.add_package(get_package('pendulum', '2.0.0'))
    repo.add_package(get_package('pytest', '3.6.0'))
    repo.add_package(get_package('flask', '2.0.0'))
    inputs = ['my-package', '1.2.3', 'This is a description', 'n', 'MIT', '~2.7 || ^3.6', '', 'pendulu', '1', '', 'Flask', '0', '', '', '', 'pytest', '0', '', '', '\n']
    tester.execute(inputs='\n'.join(inputs))
    expected = '[tool.poetry]\nname = "my-package"\nversion = "1.2.3"\ndescription = "This is a description"\nauthors = ["Your Name <you@example.com>"]\nlicense = "MIT"\nreadme = "README.md"\n\n[tool.poetry.dependencies]\npython = "~2.7 || ^3.6"\npendulum = "^2.0.0"\nflask = "^2.0.0"\n\n[tool.poetry.group.dev.dependencies]\npytest = "^3.6.0"\n'
    assert expected in tester.io.fetch_output()

def test_interactive_with_dependencies_and_no_selection(tester: CommandTester, repo: TestRepository) -> None:
    if False:
        while True:
            i = 10
    repo.add_package(get_package('django-pendulum', '0.1.6-pre4'))
    repo.add_package(get_package('pendulum', '2.0.0'))
    repo.add_package(get_package('pytest', '3.6.0'))
    inputs = ['my-package', '1.2.3', 'This is a description', 'n', 'MIT', '~2.7 || ^3.6', '', 'pendulu', '', '', '', 'pytest', '', '', '', '\n']
    tester.execute(inputs='\n'.join(inputs))
    expected = '[tool.poetry]\nname = "my-package"\nversion = "1.2.3"\ndescription = "This is a description"\nauthors = ["Your Name <you@example.com>"]\nlicense = "MIT"\nreadme = "README.md"\n\n[tool.poetry.dependencies]\npython = "~2.7 || ^3.6"\n'
    assert expected in tester.io.fetch_output()

def test_empty_license(tester: CommandTester) -> None:
    if False:
        print('Hello World!')
    inputs = ['my-package', '1.2.3', '', 'n', '', '', 'n', 'n', '\n']
    tester.execute(inputs='\n'.join(inputs))
    python = '.'.join((str(c) for c in sys.version_info[:2]))
    expected = f'[tool.poetry]\nname = "my-package"\nversion = "1.2.3"\ndescription = ""\nauthors = ["Your Name <you@example.com>"]\nreadme = "README.md"\n\n[tool.poetry.dependencies]\npython = "^{python}"\n'
    assert expected in tester.io.fetch_output()

def test_interactive_with_git_dependencies(tester: CommandTester, repo: TestRepository) -> None:
    if False:
        return 10
    repo.add_package(get_package('pendulum', '2.0.0'))
    repo.add_package(get_package('pytest', '3.6.0'))
    inputs = ['my-package', '1.2.3', 'This is a description', 'n', 'MIT', '~2.7 || ^3.6', '', 'git+https://github.com/demo/demo.git', '', '', 'pytest', '0', '', '', '\n']
    tester.execute(inputs='\n'.join(inputs))
    expected = '[tool.poetry]\nname = "my-package"\nversion = "1.2.3"\ndescription = "This is a description"\nauthors = ["Your Name <you@example.com>"]\nlicense = "MIT"\nreadme = "README.md"\n\n[tool.poetry.dependencies]\npython = "~2.7 || ^3.6"\ndemo = {git = "https://github.com/demo/demo.git"}\n\n[tool.poetry.group.dev.dependencies]\npytest = "^3.6.0"\n'
    assert expected in tester.io.fetch_output()
_generate_choice_list_packages_params: list[list[Package]] = [[get_package('flask-blacklist', '1.0.0'), get_package('Flask-Shelve', '1.0.0'), get_package('flask-pwa', '1.0.0'), get_package('Flask-test1', '1.0.0'), get_package('Flask-test2', '1.0.0'), get_package('Flask-test3', '1.0.0'), get_package('Flask-test4', '1.0.0'), get_package('Flask-test5', '1.0.0'), get_package('Flask', '1.0.0'), get_package('Flask-test6', '1.0.0'), get_package('Flask-test7', '1.0.0')], [get_package('flask-blacklist', '1.0.0'), get_package('Flask-Shelve', '1.0.0'), get_package('flask-pwa', '1.0.0'), get_package('Flask-test1', '1.0.0'), get_package('Flask', '1.0.0')]]

@pytest.fixture(params=_generate_choice_list_packages_params)
def _generate_choice_list_packages(request: FixtureRequest) -> list[Package]:
    if False:
        i = 10
        return i + 15
    packages: list[Package] = request.param
    return packages

@pytest.mark.parametrize('package_name', ['flask', 'Flask', 'flAsK'])
def test_generate_choice_list(tester: CommandTester, package_name: str, _generate_choice_list_packages: list[Package]) -> None:
    if False:
        for i in range(10):
            print('nop')
    init_command = tester.command
    assert isinstance(init_command, InitCommand)
    packages = _generate_choice_list_packages
    choices = init_command._generate_choice_list(packages, canonicalize_name(package_name))
    assert choices[0] == 'Flask'

def test_interactive_with_git_dependencies_with_reference(tester: CommandTester, repo: TestRepository) -> None:
    if False:
        return 10
    repo.add_package(get_package('pendulum', '2.0.0'))
    repo.add_package(get_package('pytest', '3.6.0'))
    inputs = ['my-package', '1.2.3', 'This is a description', 'n', 'MIT', '~2.7 || ^3.6', '', 'git+https://github.com/demo/demo.git@develop', '', '', 'pytest', '0', '', '', '\n']
    tester.execute(inputs='\n'.join(inputs))
    expected = '[tool.poetry]\nname = "my-package"\nversion = "1.2.3"\ndescription = "This is a description"\nauthors = ["Your Name <you@example.com>"]\nlicense = "MIT"\nreadme = "README.md"\n\n[tool.poetry.dependencies]\npython = "~2.7 || ^3.6"\ndemo = {git = "https://github.com/demo/demo.git", rev = "develop"}\n\n[tool.poetry.group.dev.dependencies]\npytest = "^3.6.0"\n'
    assert expected in tester.io.fetch_output()

def test_interactive_with_git_dependencies_and_other_name(tester: CommandTester, repo: TestRepository) -> None:
    if False:
        return 10
    repo.add_package(get_package('pendulum', '2.0.0'))
    repo.add_package(get_package('pytest', '3.6.0'))
    inputs = ['my-package', '1.2.3', 'This is a description', 'n', 'MIT', '~2.7 || ^3.6', '', 'git+https://github.com/demo/pyproject-demo.git', '', '', 'pytest', '0', '', '', '\n']
    tester.execute(inputs='\n'.join(inputs))
    expected = '[tool.poetry]\nname = "my-package"\nversion = "1.2.3"\ndescription = "This is a description"\nauthors = ["Your Name <you@example.com>"]\nlicense = "MIT"\nreadme = "README.md"\n\n[tool.poetry.dependencies]\npython = "~2.7 || ^3.6"\ndemo = {git = "https://github.com/demo/pyproject-demo.git"}\n\n[tool.poetry.group.dev.dependencies]\npytest = "^3.6.0"\n'
    assert expected in tester.io.fetch_output()

def test_interactive_with_directory_dependency(tester: CommandTester, repo: TestRepository, source_dir: Path, fixture_dir: FixtureDirGetter) -> None:
    if False:
        for i in range(10):
            print('nop')
    repo.add_package(get_package('pendulum', '2.0.0'))
    repo.add_package(get_package('pytest', '3.6.0'))
    demo = fixture_dir('git') / 'github.com' / 'demo' / 'demo'
    shutil.copytree(str(demo), str(source_dir / 'demo'))
    inputs = ['my-package', '1.2.3', 'This is a description', 'n', 'MIT', '~2.7 || ^3.6', '', './demo', '', '', 'pytest', '0', '', '', '\n']
    tester.execute(inputs='\n'.join(inputs))
    expected = '[tool.poetry]\nname = "my-package"\nversion = "1.2.3"\ndescription = "This is a description"\nauthors = ["Your Name <you@example.com>"]\nlicense = "MIT"\nreadme = "README.md"\n\n[tool.poetry.dependencies]\npython = "~2.7 || ^3.6"\ndemo = {path = "demo"}\n\n[tool.poetry.group.dev.dependencies]\npytest = "^3.6.0"\n'
    assert expected in tester.io.fetch_output()

def test_interactive_with_directory_dependency_and_other_name(tester: CommandTester, repo: TestRepository, source_dir: Path, fixture_dir: FixtureDirGetter) -> None:
    if False:
        for i in range(10):
            print('nop')
    repo.add_package(get_package('pendulum', '2.0.0'))
    repo.add_package(get_package('pytest', '3.6.0'))
    demo = fixture_dir('git') / 'github.com' / 'demo' / 'pyproject-demo'
    shutil.copytree(str(demo), str(source_dir / 'pyproject-demo'))
    inputs = ['my-package', '1.2.3', 'This is a description', 'n', 'MIT', '~2.7 || ^3.6', '', './pyproject-demo', '', '', 'pytest', '0', '', '', '\n']
    tester.execute(inputs='\n'.join(inputs))
    expected = '[tool.poetry]\nname = "my-package"\nversion = "1.2.3"\ndescription = "This is a description"\nauthors = ["Your Name <you@example.com>"]\nlicense = "MIT"\nreadme = "README.md"\n\n[tool.poetry.dependencies]\npython = "~2.7 || ^3.6"\ndemo = {path = "pyproject-demo"}\n\n[tool.poetry.group.dev.dependencies]\npytest = "^3.6.0"\n'
    assert expected in tester.io.fetch_output()

def test_interactive_with_file_dependency(tester: CommandTester, repo: TestRepository, source_dir: Path, fixture_dir: FixtureDirGetter) -> None:
    if False:
        for i in range(10):
            print('nop')
    repo.add_package(get_package('pendulum', '2.0.0'))
    repo.add_package(get_package('pytest', '3.6.0'))
    demo = fixture_dir('distributions') / 'demo-0.1.0-py2.py3-none-any.whl'
    shutil.copyfile(str(demo), str(source_dir / demo.name))
    inputs = ['my-package', '1.2.3', 'This is a description', 'n', 'MIT', '~2.7 || ^3.6', '', './demo-0.1.0-py2.py3-none-any.whl', '', '', 'pytest', '0', '', '', '\n']
    tester.execute(inputs='\n'.join(inputs))
    expected = '[tool.poetry]\nname = "my-package"\nversion = "1.2.3"\ndescription = "This is a description"\nauthors = ["Your Name <you@example.com>"]\nlicense = "MIT"\nreadme = "README.md"\n\n[tool.poetry.dependencies]\npython = "~2.7 || ^3.6"\ndemo = {path = "demo-0.1.0-py2.py3-none-any.whl"}\n\n[tool.poetry.group.dev.dependencies]\npytest = "^3.6.0"\n'
    assert expected in tester.io.fetch_output()

def test_interactive_with_wrong_dependency_inputs(tester: CommandTester, repo: TestRepository) -> None:
    if False:
        for i in range(10):
            print('nop')
    inputs = ['my-package', '1.2.3', 'This is a description', 'n', 'MIT', '^3.8', '', 'foo 1.19.2', 'pendulum 2.0.0 foo', 'pendulum@^2.0.0', '', '', 'pytest 3.6.0 foo', 'pytest 3.6.0', 'pytest@3.6.0', '', '\n']
    tester.execute(inputs='\n'.join(inputs))
    expected = '[tool.poetry]\nname = "my-package"\nversion = "1.2.3"\ndescription = "This is a description"\nauthors = ["Your Name <you@example.com>"]\nlicense = "MIT"\nreadme = "README.md"\n\n[tool.poetry.dependencies]\npython = "^3.8"\nfoo = "1.19.2"\npendulum = "^2.0.0"\n\n[tool.poetry.group.dev.dependencies]\npytest = "3.6.0"\n'
    assert expected in tester.io.fetch_output()

def test_python_option(tester: CommandTester) -> None:
    if False:
        while True:
            i = 10
    inputs = ['my-package', '1.2.3', 'This is a description', 'n', 'MIT', 'n', 'n', '\n']
    tester.execute("--python '~2.7 || ^3.6'", inputs='\n'.join(inputs))
    expected = '[tool.poetry]\nname = "my-package"\nversion = "1.2.3"\ndescription = "This is a description"\nauthors = ["Your Name <you@example.com>"]\nlicense = "MIT"\nreadme = "README.md"\n\n[tool.poetry.dependencies]\npython = "~2.7 || ^3.6"\n'
    assert expected in tester.io.fetch_output()

def test_predefined_dependency(tester: CommandTester, repo: TestRepository) -> None:
    if False:
        while True:
            i = 10
    repo.add_package(get_package('pendulum', '2.0.0'))
    inputs = ['my-package', '1.2.3', 'This is a description', 'n', 'MIT', '~2.7 || ^3.6', 'n', 'n', '\n']
    tester.execute('--dependency pendulum', inputs='\n'.join(inputs))
    expected = '[tool.poetry]\nname = "my-package"\nversion = "1.2.3"\ndescription = "This is a description"\nauthors = ["Your Name <you@example.com>"]\nlicense = "MIT"\nreadme = "README.md"\n\n[tool.poetry.dependencies]\npython = "~2.7 || ^3.6"\npendulum = "^2.0.0"\n'
    assert expected in tester.io.fetch_output()

def test_predefined_and_interactive_dependencies(tester: CommandTester, repo: TestRepository) -> None:
    if False:
        i = 10
        return i + 15
    repo.add_package(get_package('pendulum', '2.0.0'))
    repo.add_package(get_package('pyramid', '1.10'))
    inputs = ['my-package', '1.2.3', 'This is a description', 'n', 'MIT', '~2.7 || ^3.6', '', 'pyramid', '0', '', '', 'n', '\n']
    tester.execute('--dependency pendulum', inputs='\n'.join(inputs))
    expected = '[tool.poetry]\nname = "my-package"\nversion = "1.2.3"\ndescription = "This is a description"\nauthors = ["Your Name <you@example.com>"]\nlicense = "MIT"\nreadme = "README.md"\n\n[tool.poetry.dependencies]\npython = "~2.7 || ^3.6"\n'
    output = tester.io.fetch_output()
    assert expected in output
    assert 'pendulum = "^2.0.0"' in output
    assert 'pyramid = "^1.10"' in output

def test_predefined_dev_dependency(tester: CommandTester, repo: TestRepository) -> None:
    if False:
        i = 10
        return i + 15
    repo.add_package(get_package('pytest', '3.6.0'))
    inputs = ['my-package', '1.2.3', 'This is a description', 'n', 'MIT', '~2.7 || ^3.6', 'n', 'n', '\n']
    tester.execute('--dev-dependency pytest', inputs='\n'.join(inputs))
    expected = '[tool.poetry]\nname = "my-package"\nversion = "1.2.3"\ndescription = "This is a description"\nauthors = ["Your Name <you@example.com>"]\nlicense = "MIT"\nreadme = "README.md"\n\n[tool.poetry.dependencies]\npython = "~2.7 || ^3.6"\n\n[tool.poetry.group.dev.dependencies]\npytest = "^3.6.0"\n'
    assert expected in tester.io.fetch_output()

def test_predefined_and_interactive_dev_dependencies(tester: CommandTester, repo: TestRepository) -> None:
    if False:
        return 10
    repo.add_package(get_package('pytest', '3.6.0'))
    repo.add_package(get_package('pytest-requests', '0.2.0'))
    inputs = ['my-package', '1.2.3', 'This is a description', 'n', 'MIT', '~2.7 || ^3.6', 'n', '', 'pytest-requests', '0', '', '', '\n']
    tester.execute('--dev-dependency pytest', inputs='\n'.join(inputs))
    expected = '[tool.poetry]\nname = "my-package"\nversion = "1.2.3"\ndescription = "This is a description"\nauthors = ["Your Name <you@example.com>"]\nlicense = "MIT"\nreadme = "README.md"\n\n[tool.poetry.dependencies]\npython = "~2.7 || ^3.6"\n\n[tool.poetry.group.dev.dependencies]\npytest = "^3.6.0"\npytest-requests = "^0.2.0"\n'
    output = tester.io.fetch_output()
    assert expected in output
    assert 'pytest-requests = "^0.2.0"' in output
    assert 'pytest = "^3.6.0"' in output

def test_predefined_all_options(tester: CommandTester, repo: TestRepository) -> None:
    if False:
        return 10
    repo.add_package(get_package('pendulum', '2.0.0'))
    repo.add_package(get_package('pytest', '3.6.0'))
    inputs = ['1.2.3', '', 'n', 'n', '\n']
    tester.execute("--name my-package --description 'This is a description' --author 'Foo Bar <foo@example.com>' --python '^3.8' --license MIT --dependency pendulum --dev-dependency pytest", inputs='\n'.join(inputs))
    expected = '[tool.poetry]\nname = "my-package"\nversion = "1.2.3"\ndescription = "This is a description"\nauthors = ["Foo Bar <foo@example.com>"]\nlicense = "MIT"\nreadme = "README.md"\n\n[tool.poetry.dependencies]\npython = "^3.8"\npendulum = "^2.0.0"\n\n[tool.poetry.group.dev.dependencies]\npytest = "^3.6.0"\n'
    output = tester.io.fetch_output()
    assert expected in output

def test_add_package_with_extras_and_whitespace(tester: CommandTester) -> None:
    if False:
        return 10
    command = tester.command
    assert isinstance(command, InitCommand)
    result = command._parse_requirements(['databases[postgresql, sqlite]'])
    assert result[0]['name'] == 'databases'
    assert len(result[0]['extras']) == 2
    assert 'postgresql' in result[0]['extras']
    assert 'sqlite' in result[0]['extras']

def test_init_existing_pyproject_simple(tester: CommandTester, source_dir: Path, init_basic_inputs: str, init_basic_toml: str) -> None:
    if False:
        return 10
    pyproject_file = source_dir / 'pyproject.toml'
    existing_section = '\n[tool.black]\nline-length = 88\n'
    pyproject_file.write_text(existing_section)
    tester.execute(inputs=init_basic_inputs)
    assert f'{existing_section}\n{init_basic_toml}' in pyproject_file.read_text()

@pytest.mark.parametrize('linesep', ['\n', '\r\n'])
def test_init_existing_pyproject_consistent_linesep(tester: CommandTester, source_dir: Path, init_basic_inputs: str, init_basic_toml: str, linesep: str) -> None:
    if False:
        i = 10
        return i + 15
    pyproject_file = source_dir / 'pyproject.toml'
    existing_section = '\n[tool.black]\nline-length = 88\n'.replace('\n', linesep)
    with open(pyproject_file, 'w', newline='') as f:
        f.write(existing_section)
    tester.execute(inputs=init_basic_inputs)
    with open(pyproject_file, newline='') as f:
        content = f.read()
    init_basic_toml = init_basic_toml.replace('\n', linesep)
    assert f'{existing_section}{linesep}{init_basic_toml}' in content

def test_init_non_interactive_existing_pyproject_add_dependency(tester: CommandTester, source_dir: Path, init_basic_inputs: str, repo: TestRepository) -> None:
    if False:
        return 10
    pyproject_file = source_dir / 'pyproject.toml'
    existing_section = '\n[tool.black]\nline-length = 88\n'
    pyproject_file.write_text(existing_section)
    repo.add_package(get_package('foo', '1.19.2'))
    tester.execute("--author 'Your Name <you@example.com>' --name 'my-package' --python '^3.6' --dependency foo", interactive=False)
    expected = '[tool.poetry]\nname = "my-package"\nversion = "0.1.0"\ndescription = ""\nauthors = ["Your Name <you@example.com>"]\nreadme = "README.md"\n\n[tool.poetry.dependencies]\npython = "^3.6"\nfoo = "^1.19.2"\n'
    assert f'{existing_section}\n{expected}' in pyproject_file.read_text()

def test_init_existing_pyproject_with_build_system_fails(tester: CommandTester, source_dir: Path, init_basic_inputs: str) -> None:
    if False:
        print('Hello World!')
    pyproject_file = source_dir / 'pyproject.toml'
    existing_section = '\n[build-system]\nrequires = ["setuptools >= 40.6.0", "wheel"]\nbuild-backend = "setuptools.build_meta"\n'
    pyproject_file.write_text(existing_section)
    tester.execute(inputs=init_basic_inputs)
    assert tester.io.fetch_error().strip() == 'A pyproject.toml file with a defined build-system already exists.'
    assert existing_section in pyproject_file.read_text()

@pytest.mark.parametrize('name', [None, '', 'foo', '   foo  ', 'foo==2.0', 'foo@2.0', '  foo@2.0   ', 'foo 2.0', '   foo 2.0  '])
def test_validate_package_valid(name: str | None) -> None:
    if False:
        return 10
    assert InitCommand._validate_package(name) == name

@pytest.mark.parametrize('name', ['foo bar 2.0', '   foo bar 2.0   ', 'foo bar foobar 2.0'])
def test_validate_package_invalid(name: str) -> None:
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError):
        assert InitCommand._validate_package(name)

@pytest.mark.parametrize('package_name, include', (('mypackage', None), ('my-package', 'my_package'), ('my.package', 'my'), ('my-awesome-package', 'my_awesome_package'), ('my.awesome.package', 'my')))
def test_package_include(tester: CommandTester, package_name: str, include: str | None) -> None:
    if False:
        i = 10
        return i + 15
    tester.execute(inputs='\n'.join((package_name, '', '', 'poetry', '', '^3.10', 'n', 'n', '\n')))
    packages = ''
    if include and module_name(package_name) != include:
        packages = f'packages = [{{include = "{include}"}}]\n'
    expected = f'''[tool.poetry]\nname = "{package_name.replace('.', '-')}"\nversion = "0.1.0"\ndescription = ""\nauthors = ["poetry"]\nreadme = "README.md"\n{packages}\n[tool.poetry.dependencies]\npython = "^3.10"\n'''
    assert expected in tester.io.fetch_output()

@pytest.mark.parametrize(['prefer_active', 'python'], [(True, '1.1'), (False, f'{sys.version_info[0]}.{sys.version_info[1]}')])
def test_respect_prefer_active_on_init(prefer_active: bool, python: str, config: Config, mocker: MockerFixture, tester: CommandTester, source_dir: Path) -> None:
    if False:
        i = 10
        return i + 15
    from poetry.utils.env import GET_PYTHON_VERSION_ONELINER
    orig_check_output = subprocess.check_output

    def mock_check_output(cmd: str, *_: Any, **__: Any) -> str:
        if False:
            i = 10
            return i + 15
        if GET_PYTHON_VERSION_ONELINER in cmd:
            return '1.1.1'
        result: str = orig_check_output(cmd, *_, **__)
        return result
    mocker.patch('subprocess.check_output', side_effect=mock_check_output)
    config.config['virtualenvs']['prefer-active-python'] = prefer_active
    pyproject_file = source_dir / 'pyproject.toml'
    tester.execute("--author 'Your Name <you@example.com>' --name 'my-package'", interactive=False)
    expected = f'[tool.poetry.dependencies]\npython = "^{python}"\n'
    assert expected in pyproject_file.read_text()

def test_get_pool(mocker: MockerFixture, source_dir: Path) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Since we are mocking _get_pool() in the other tests, we at least should make\n    sure it works in general. See https://github.com/python-poetry/poetry/issues/8634.\n    '
    mocker.patch('pathlib.Path.cwd', return_value=source_dir)
    app = Application()
    command = app.find('init')
    assert isinstance(command, InitCommand)
    pool = command._get_pool()
    assert pool.repositories