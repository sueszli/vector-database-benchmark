from __future__ import annotations
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
from textwrap import dedent
from unittest import mock
from unittest.mock import MagicMock
import pytest
from pip._internal.req.constructors import install_req_from_line
from pip._internal.utils.hashes import FAVORITE_HASH
from pip._internal.utils.urls import path_to_url
from pip._vendor.packaging.version import Version
from piptools.build import ProjectMetadata
from piptools.scripts.compile import cli
from piptools.utils import COMPILE_EXCLUDE_OPTIONS, get_pip_version_for_python_executable
from .constants import MINIMAL_WHEELS_PATH, PACKAGES_PATH
legacy_resolver_only = pytest.mark.parametrize('current_resolver', ('legacy',), indirect=('current_resolver',))
backtracking_resolver_only = pytest.mark.parametrize('current_resolver', ('backtracking',), indirect=('current_resolver',))

@pytest.fixture(autouse=True, params=[pytest.param('legacy', id='legacy resolver'), pytest.param('backtracking', id='backtracking resolver')])
def current_resolver(request, monkeypatch):
    if False:
        while True:
            i = 10
    exclude_options = COMPILE_EXCLUDE_OPTIONS | {'--resolver'}
    monkeypatch.setattr('piptools.utils.COMPILE_EXCLUDE_OPTIONS', exclude_options)
    resolver_name = request.param
    monkeypatch.setenv('PIP_TOOLS_RESOLVER', resolver_name)
    return resolver_name

@pytest.fixture(autouse=True)
def _temp_dep_cache(tmpdir, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setenv('PIP_TOOLS_CACHE_DIR', str(tmpdir / 'cache'))

def test_default_pip_conf_read(pip_with_index_conf, runner):
    if False:
        return 10
    with open('requirements.in', 'w'):
        pass
    out = runner.invoke(cli, ['-v'])
    assert 'Using indexes:\n  http://example.com' in out.stderr
    assert '--index-url http://example.com' in out.stderr

def test_command_line_overrides_pip_conf(pip_with_index_conf, runner):
    if False:
        print('Hello World!')
    with open('requirements.in', 'w'):
        pass
    out = runner.invoke(cli, ['-v', '-i', 'http://override.com'])
    assert 'Using indexes:\n  http://override.com' in out.stderr

@pytest.mark.network
@pytest.mark.parametrize(('install_requires', 'expected_output'), (pytest.param('small-fake-a==0.1', 'small-fake-a==0.1', id='regular'), pytest.param('pip-tools @ https://github.com/jazzband/pip-tools/archive/7d86c8d3.zip', 'pip-tools @ https://github.com/jazzband/pip-tools/archive/7d86c8d3.zip', id='zip URL'), pytest.param('pip-tools @ git+https://github.com/jazzband/pip-tools@7d86c8d3', 'pip-tools @ git+https://github.com/jazzband/pip-tools@7d86c8d3', id='scm URL'), pytest.param('pip-tools @ https://files.pythonhosted.org/packages/06/96/89872db07ae70770fba97205b0737c17ef013d0d1c790899c16bb8bac419/pip_tools-3.6.1-py2.py3-none-any.whl', 'pip-tools @ https://files.pythonhosted.org/packages/06/96/89872db07ae70770fba97205b0737c17ef013d0d1c790899c16bb8bac419/pip_tools-3.6.1-py2.py3-none-any.whl', id='wheel URL')))
def test_command_line_setuptools_read(runner, make_package, install_requires, expected_output):
    if False:
        return 10
    package_dir = make_package(name='fake-setuptools-a', install_requires=(install_requires,))
    out = runner.invoke(cli, (str(package_dir / 'setup.py'), '--find-links', MINIMAL_WHEELS_PATH, '--no-build-isolation'))
    assert out.exit_code == 0
    output_file = package_dir / 'requirements.txt'
    assert output_file.exists()
    assert expected_output in output_file.read_text().splitlines()

@pytest.mark.network
@pytest.mark.parametrize(('options', 'expected_output_file'), (([], 'requirements.txt'), (['--output-file', 'output.txt'], 'output.txt'), (['setup.py'], 'requirements.txt'), (['setup.py', '--output-file', 'output.txt'], 'output.txt')))
def test_command_line_setuptools_output_file(runner, options, expected_output_file):
    if False:
        while True:
            i = 10
    '\n    Test the output files for setup.py as a requirement file.\n    '
    with open('setup.py', 'w') as package:
        package.write(dedent('                from setuptools import setup\n                setup(install_requires=[])\n                '))
    out = runner.invoke(cli, ['--no-build-isolation'] + options)
    assert out.exit_code == 0
    assert os.path.exists(expected_output_file)

@pytest.mark.network
def test_command_line_setuptools_nested_output_file(tmpdir, runner):
    if False:
        i = 10
        return i + 15
    '\n    Test the output file for setup.py in nested folder as a requirement file.\n    '
    proj_dir = tmpdir.mkdir('proj')
    with open(str(proj_dir / 'setup.py'), 'w') as package:
        package.write(dedent('                from setuptools import setup\n                setup(install_requires=[])\n                '))
    out = runner.invoke(cli, [str(proj_dir / 'setup.py'), '--no-build-isolation'])
    assert out.exit_code == 0
    assert (proj_dir / 'requirements.txt').exists()

@pytest.mark.network
def test_setuptools_preserves_environment_markers(runner, make_package, make_wheel, make_pip_conf, tmpdir):
    if False:
        return 10
    make_pip_conf(dedent('            [global]\n            disable-pip-version-check = True\n            '))
    dists_dir = tmpdir / 'dists'
    foo_dir = make_package(name='foo', version='1.0')
    make_wheel(foo_dir, dists_dir)
    bar_dir = make_package(name='bar', version='2.0', install_requires=['foo ; python_version >= "1"'])
    out = runner.invoke(cli, [str(bar_dir / 'setup.py'), '--output-file', '-', '--no-header', '--no-annotate', '--no-emit-find-links', '--no-build-isolation', '--find-links', str(dists_dir)])
    assert out.exit_code == 0, out.stderr
    assert out.stdout == 'foo==1.0 ; python_version >= "1"\n'

def test_no_index_option(runner, tmp_path):
    if False:
        while True:
            i = 10
    req_in = tmp_path / 'requirements.in'
    req_in.touch()
    out = runner.invoke(cli, [req_in.as_posix(), '--no-index', '--verbose'])
    assert out.exit_code == 0
    assert 'Ignoring indexes.' in out.stderr

def test_find_links_option(runner):
    if False:
        for i in range(10):
            print('nop')
    with open('requirements.in', 'w') as req_in:
        req_in.write('-f ./libs3')
    out = runner.invoke(cli, ['-v', '-f', './libs1', '-f', './libs2'])
    assert 'Using links:\n  ./libs1\n  ./libs2\n  ./libs3\n' in out.stderr
    with open('requirements.txt') as req_txt:
        assert '--find-links ./libs1\n--find-links ./libs2\n--find-links ./libs3\n' in req_txt.read()

def test_find_links_envvar(monkeypatch, runner):
    if False:
        print('Hello World!')
    with open('requirements.in', 'w') as req_in:
        req_in.write('-f ./libs3')
    monkeypatch.setenv('PIP_FIND_LINKS', './libs1 ./libs2')
    out = runner.invoke(cli, ['-v'])
    assert 'Using links:\n  ./libs1\n  ./libs2\n  ./libs3\n' in out.stderr
    with open('requirements.txt') as req_txt:
        assert '--find-links ./libs1\n--find-links ./libs2\n--find-links ./libs3\n' in req_txt.read()

def test_extra_index_option(pip_with_index_conf, runner):
    if False:
        print('Hello World!')
    with open('requirements.in', 'w'):
        pass
    out = runner.invoke(cli, ['-v', '--extra-index-url', 'http://extraindex1.com', '--extra-index-url', 'http://extraindex2.com'])
    assert 'Using indexes:\n  http://example.com\n  http://extraindex1.com\n  http://extraindex2.com' in out.stderr
    assert '--index-url http://example.com\n--extra-index-url http://extraindex1.com\n--extra-index-url http://extraindex2.com' in out.stderr

def test_extra_index_envvar(monkeypatch, runner):
    if False:
        i = 10
        return i + 15
    with open('requirements.in', 'w'):
        pass
    monkeypatch.setenv('PIP_INDEX_URL', 'http://example.com')
    monkeypatch.setenv('PIP_EXTRA_INDEX_URL', 'http://extraindex1.com http://extraindex2.com')
    out = runner.invoke(cli, ['-v'])
    assert 'Using indexes:\n  http://example.com\n  http://extraindex1.com\n  http://extraindex2.com' in out.stderr
    assert '--index-url http://example.com\n--extra-index-url http://extraindex1.com\n--extra-index-url http://extraindex2.com' in out.stderr

@pytest.mark.parametrize('option', ('--extra-index-url', '--find-links'))
def test_redacted_urls_in_verbose_output(runner, option):
    if False:
        while True:
            i = 10
    "\n    Test that URLs with sensitive data don't leak to the output.\n    "
    with open('requirements.in', 'w'):
        pass
    out = runner.invoke(cli, ['--no-header', '--no-emit-index-url', '--no-emit-find-links', '--verbose', option, 'http://username:password@example.com'])
    assert 'http://username:****@example.com' in out.stderr
    assert 'password' not in out.stderr

def test_trusted_host_option(pip_conf, runner):
    if False:
        for i in range(10):
            print('nop')
    with open('requirements.in', 'w'):
        pass
    out = runner.invoke(cli, ['-v', '--trusted-host', 'example.com', '--trusted-host', 'example2.com'])
    assert '--trusted-host example.com\n--trusted-host example2.com\n' in out.stderr

def test_trusted_host_envvar(monkeypatch, pip_conf, runner):
    if False:
        i = 10
        return i + 15
    with open('requirements.in', 'w'):
        pass
    monkeypatch.setenv('PIP_TRUSTED_HOST', 'example.com example2.com')
    out = runner.invoke(cli, ['-v'])
    assert '--trusted-host example.com\n--trusted-host example2.com\n' in out.stderr

@pytest.mark.parametrize('options', (pytest.param(['--trusted-host', 'example.com', '--no-emit-trusted-host'], id='trusted host'), pytest.param(['--find-links', 'wheels', '--no-emit-find-links'], id='find links'), pytest.param(['--index-url', 'https://index-url', '--no-emit-index-url'], id='index url')))
def test_all_no_emit_options(runner, options):
    if False:
        i = 10
        return i + 15
    with open('requirements.in', 'w'):
        pass
    out = runner.invoke(cli, ['--output-file', '-', '--no-header', *options])
    assert out.stdout.strip().splitlines() == []

@pytest.mark.parametrize(('option', 'expected_output'), (pytest.param('--emit-index-url', ['--index-url https://index-url'], id='index url'), pytest.param('--no-emit-index-url', [], id='no index')))
def test_emit_index_url_option(runner, option, expected_output):
    if False:
        i = 10
        return i + 15
    with open('requirements.in', 'w'):
        pass
    out = runner.invoke(cli, ['--output-file', '-', '--quiet', '--no-header', '--index-url', 'https://index-url', option])
    assert out.stdout.strip().splitlines() == expected_output

@pytest.mark.network
def test_realistic_complex_sub_dependencies(runner):
    if False:
        for i in range(10):
            print('nop')
    wheels_dir = 'wheels'
    subprocess.run(['pip', 'wheel', '--no-deps', '-w', wheels_dir, os.path.join(PACKAGES_PATH, 'fake_with_deps', '.')], check=True)
    with open('requirements.in', 'w') as req_in:
        req_in.write('fake_with_deps')
    out = runner.invoke(cli, ['-n', '--rebuild', '-f', wheels_dir])
    assert out.exit_code == 0

def test_run_as_module_compile():
    if False:
        for i in range(10):
            print('nop')
    'piptools can be run as ``python -m piptools ...``.'
    result = subprocess.run([sys.executable, '-m', 'piptools', 'compile', '--help'], stdout=subprocess.PIPE, check=True)
    assert result.stdout.startswith(b'Usage:')
    assert b'Compiles requirements.txt from requirements.in' in result.stdout

def test_editable_package(pip_conf, runner):
    if False:
        i = 10
        return i + 15
    'piptools can compile an editable'
    fake_package_dir = os.path.join(PACKAGES_PATH, 'small_fake_with_deps')
    fake_package_dir = path_to_url(fake_package_dir)
    with open('requirements.in', 'w') as req_in:
        req_in.write('-e ' + fake_package_dir)
    out = runner.invoke(cli, ['-n'])
    assert out.exit_code == 0
    assert fake_package_dir in out.stderr
    assert 'small-fake-a==0.1' in out.stderr

def test_editable_package_without_non_editable_duplicate(pip_conf, runner):
    if False:
        i = 10
        return i + 15
    '\n    piptools keeps editable requirement,\n    without also adding a duplicate "non-editable" requirement variation\n    '
    fake_package_dir = os.path.join(PACKAGES_PATH, 'small_fake_a')
    fake_package_dir = path_to_url(fake_package_dir)
    with open('requirements.in', 'w') as req_in:
        req_in.write('-e ' + fake_package_dir + '\nsmall_fake_with_unpinned_deps')
    out = runner.invoke(cli, ['-n'])
    assert out.exit_code == 0
    assert fake_package_dir in out.stderr
    assert 'small-fake-a==' not in out.stderr

@legacy_resolver_only
def test_editable_package_constraint_without_non_editable_duplicate(pip_conf, runner):
    if False:
        while True:
            i = 10
    '\n    piptools keeps editable constraint,\n    without also adding a duplicate "non-editable" requirement variation\n    '
    fake_package_dir = os.path.join(PACKAGES_PATH, 'small_fake_a')
    fake_package_dir = path_to_url(fake_package_dir)
    with open('constraints.txt', 'w') as constraints:
        constraints.write('-e ' + fake_package_dir)
    with open('requirements.in', 'w') as req_in:
        req_in.write('-c constraints.txt\nsmall_fake_with_unpinned_deps')
    out = runner.invoke(cli, ['--output-file', '-', '--quiet'])
    assert out.exit_code == 0
    assert fake_package_dir in out.stdout
    assert 'small-fake-a==' not in out.stdout

@legacy_resolver_only
@pytest.mark.parametrize('req_editable', ((True,), (False,)))
def test_editable_package_in_constraints(pip_conf, runner, req_editable):
    if False:
        for i in range(10):
            print('nop')
    '\n    piptools can compile an editable that appears in both primary requirements\n    and constraints\n    '
    fake_package_dir = os.path.join(PACKAGES_PATH, 'small_fake_with_deps')
    fake_package_dir = path_to_url(fake_package_dir)
    with open('constraints.txt', 'w') as constraints_in:
        constraints_in.write('-e ' + fake_package_dir)
    with open('requirements.in', 'w') as req_in:
        prefix = '-e ' if req_editable else ''
        req_in.write(prefix + fake_package_dir + '\n-c constraints.txt')
    out = runner.invoke(cli, ['-n'])
    assert out.exit_code == 0
    assert fake_package_dir in out.stderr
    assert 'small-fake-a==0.1' in out.stderr

@pytest.mark.network
def test_editable_package_vcs(runner):
    if False:
        while True:
            i = 10
    vcs_package = 'git+https://github.com/jazzband/pip-tools@f97e62ecb0d9b70965c8eff952c001d8e2722e94#egg=pip-tools'
    with open('requirements.in', 'w') as req_in:
        req_in.write('-e ' + vcs_package)
    out = runner.invoke(cli, ['-n', '--rebuild'])
    assert out.exit_code == 0
    assert vcs_package in out.stderr
    assert 'click' in out.stderr

@pytest.mark.network
def test_compile_cached_vcs_package(runner, venv):
    if False:
        return 10
    "\n    Test pip-compile doesn't write local paths for cached wheels of VCS packages.\n\n    Regression test for issue GH-1647.\n    "
    vcs_package = 'typing-extensions @ git+https://github.com/python/typing_extensions@9c0759a260fe126210a1e2026720000a3c40a919'
    vcs_wheel_prefix = 'typing_extensions-4.3.0-py3'
    subprocess.run([os.fspath(venv / 'python'), '-mpip', 'install', vcs_package], check=True)
    assert vcs_wheel_prefix in subprocess.run([sys.executable, '-mpip', 'cache', 'list', '--format=abspath', vcs_wheel_prefix], check=True, capture_output=True, text=True).stdout
    with open('requirements.in', 'w') as req_in:
        req_in.write(vcs_package)
    out = runner.invoke(cli, ['--output-file', '-', '--quiet', '--no-header', '--no-emit-options', '--no-annotate', '--no-build-isolation'])
    assert out.exit_code == 0, out
    assert vcs_package == out.stdout.strip()

@legacy_resolver_only
def test_locally_available_editable_package_is_not_archived_in_cache_dir(pip_conf, tmpdir, runner):
    if False:
        while True:
            i = 10
    '\n    piptools will not create an archive for a locally available editable requirement\n    '
    cache_dir = tmpdir.mkdir('cache_dir')
    fake_package_dir = os.path.join(PACKAGES_PATH, 'small_fake_with_deps')
    fake_package_dir = path_to_url(fake_package_dir)
    with open('requirements.in', 'w') as req_in:
        req_in.write('-e ' + fake_package_dir)
    out = runner.invoke(cli, ['-n', '--rebuild', '--cache-dir', str(cache_dir)])
    assert out.exit_code == 0
    assert fake_package_dir in out.stderr
    assert 'small-fake-a==0.1' in out.stderr
    assert not os.listdir(os.path.join(str(cache_dir), 'pkgs'))

@pytest.mark.parametrize(('line', 'dependency'), (pytest.param('https://github.com/jazzband/pip-tools/archive/7d86c8d3ecd1faa6be11c7ddc6b29a30ffd1dae3.zip', '\nclick==', id='Zip URL'), pytest.param('git+https://github.com/jazzband/pip-tools@7d86c8d3ecd1faa6be11c7ddc6b29a30ffd1dae3', '\nclick==', id='VCS URL'), pytest.param('https://files.pythonhosted.org/packages/06/96/89872db07ae70770fba97205b0737c17ef013d0d1c790899c16bb8bac419/pip_tools-3.6.1-py2.py3-none-any.whl', '\nclick==', id='Wheel URL'), pytest.param('pytest-django @ git+https://github.com/pytest-dev/pytest-django@21492afc88a19d4ca01cd0ac392a5325b14f95c7#egg=pytest-django', 'pytest-django @ git+https://github.com/pytest-dev/pytest-django@21492afc88a19d4ca01cd0ac392a5325b14f95c7', id='VCS with direct reference and egg')))
@pytest.mark.parametrize('generate_hashes', ((True,), (False,)))
@pytest.mark.network
def test_url_package(runner, line, dependency, generate_hashes):
    if False:
        while True:
            i = 10
    with open('requirements.in', 'w') as req_in:
        req_in.write(line)
    out = runner.invoke(cli, ['-n', '--rebuild', '--no-build-isolation'] + (['--generate-hashes'] if generate_hashes else []))
    assert out.exit_code == 0
    assert dependency in out.stderr

@pytest.mark.parametrize(('line', 'dependency', 'rewritten_line'), (pytest.param(path_to_url(os.path.join(MINIMAL_WHEELS_PATH, 'small_fake_with_deps-0.1-py2.py3-none-any.whl')), '\nsmall-fake-a==0.1', None, id='Wheel URI'), pytest.param(path_to_url(os.path.join(PACKAGES_PATH, 'small_fake_with_deps')), '\nsmall-fake-a==0.1', None, id='Local project URI'), pytest.param(os.path.join(MINIMAL_WHEELS_PATH, 'small_fake_with_deps-0.1-py2.py3-none-any.whl'), '\nsmall-fake-a==0.1', path_to_url(os.path.join(MINIMAL_WHEELS_PATH, 'small_fake_with_deps-0.1-py2.py3-none-any.whl')), id='Bare path to file URI'), pytest.param(os.path.join(MINIMAL_WHEELS_PATH, 'small_fake_with_deps-0.1-py2.py3-none-any.whl'), '\nsmall-fake-with-deps @ ' + path_to_url(os.path.join(MINIMAL_WHEELS_PATH, 'small_fake_with_deps-0.1-py2.py3-none-any.whl')), '\nsmall-fake-with-deps @ ' + path_to_url(os.path.join(MINIMAL_WHEELS_PATH, 'small_fake_with_deps-0.1-py2.py3-none-any.whl')), id='Local project with absolute URI'), pytest.param(path_to_url(os.path.join(PACKAGES_PATH, 'small_fake_with_subdir')) + '#subdirectory=subdir&egg=small-fake-a', 'small-fake-a @ ' + path_to_url(os.path.join(PACKAGES_PATH, 'small_fake_with_subdir')) + '#subdirectory=subdir', 'small-fake-a @ ' + path_to_url(os.path.join(PACKAGES_PATH, 'small_fake_with_subdir')) + '#subdirectory=subdir', id='Local project with subdirectory')))
@pytest.mark.parametrize('generate_hashes', ((True,), (False,)))
def test_local_file_uri_package(pip_conf, runner, line, dependency, rewritten_line, generate_hashes):
    if False:
        i = 10
        return i + 15
    if rewritten_line is None:
        rewritten_line = line
    with open('requirements.in', 'w') as req_in:
        req_in.write(line)
    out = runner.invoke(cli, ['-n', '--rebuild'] + (['--generate-hashes'] if generate_hashes else []))
    assert out.exit_code == 0
    assert rewritten_line in out.stderr
    assert dependency in out.stderr

def test_relative_file_uri_package(pip_conf, runner):
    if False:
        return 10
    shutil.copy(os.path.join(MINIMAL_WHEELS_PATH, 'small_fake_with_deps-0.1-py2.py3-none-any.whl'), '.')
    with open('requirements.in', 'w') as req_in:
        req_in.write('file:small_fake_with_deps-0.1-py2.py3-none-any.whl')
    out = runner.invoke(cli, ['-n', '--rebuild'])
    assert out.exit_code == 0
    assert 'file:small_fake_with_deps-0.1-py2.py3-none-any.whl' in out.stderr

def test_direct_reference_with_extras(runner):
    if False:
        while True:
            i = 10
    with open('requirements.in', 'w') as req_in:
        req_in.write('pip-tools[testing,coverage] @ git+https://github.com/jazzband/pip-tools@6.2.0')
    out = runner.invoke(cli, ['-n', '--rebuild', '--no-build-isolation'])
    assert out.exit_code == 0
    assert 'pip-tools[coverage,testing] @ git+https://github.com/jazzband/pip-tools@6.2.0' in out.stderr
    assert 'pytest==' in out.stderr
    assert 'pytest-cov==' in out.stderr

def test_input_file_without_extension(pip_conf, runner):
    if False:
        return 10
    '\n    piptools can compile a file without an extension,\n    and add .txt as the default output file extension.\n    '
    with open('requirements', 'w') as req_in:
        req_in.write('small-fake-a==0.1')
    out = runner.invoke(cli, ['requirements'])
    assert out.exit_code == 0
    assert 'small-fake-a==0.1' in out.stderr
    assert os.path.exists('requirements.txt')

def test_ignore_incompatible_existing_pins(pip_conf, runner):
    if False:
        i = 10
        return i + 15
    '\n    Successfully compile when existing output pins conflict with input.\n    '
    with open('requirements.txt', 'w') as req_txt:
        req_txt.write('small-fake-a==0.2\nsmall-fake-b==0.2')
    with open('requirements.in', 'w') as req_in:
        req_in.write('small-fake-with-deps\nsmall-fake-b<0.2')
    out = runner.invoke(cli, [])
    assert out.exit_code == 0

def test_upgrade_packages_option(pip_conf, runner):
    if False:
        i = 10
        return i + 15
    '\n    piptools respects --upgrade-package/-P inline list.\n    '
    with open('requirements.in', 'w') as req_in:
        req_in.write('small-fake-a\nsmall-fake-b')
    with open('requirements.txt', 'w') as req_in:
        req_in.write('small-fake-a==0.1\nsmall-fake-b==0.1')
    out = runner.invoke(cli, ['--no-annotate', '-P', 'small-fake-b'])
    assert out.exit_code == 0
    assert 'small-fake-a==0.1' in out.stderr.splitlines()
    assert 'small-fake-b==0.3' in out.stderr.splitlines()

def test_upgrade_packages_option_irrelevant(pip_conf, runner):
    if False:
        i = 10
        return i + 15
    '\n    piptools ignores --upgrade-package/-P items not already constrained.\n    '
    with open('requirements.in', 'w') as req_in:
        req_in.write('small-fake-a')
    with open('requirements.txt', 'w') as req_in:
        req_in.write('small-fake-a==0.1')
    out = runner.invoke(cli, ['--no-annotate', '--upgrade-package', 'small-fake-b'])
    assert out.exit_code == 0
    assert 'small-fake-a==0.1' in out.stderr.splitlines()
    assert 'small-fake-b==0.3' not in out.stderr.splitlines()

def test_upgrade_packages_option_no_existing_file(pip_conf, runner):
    if False:
        print('Hello World!')
    "\n    piptools respects --upgrade-package/-P inline list when the output file\n    doesn't exist.\n    "
    with open('requirements.in', 'w') as req_in:
        req_in.write('small-fake-a\nsmall-fake-b')
    out = runner.invoke(cli, ['--no-annotate', '-P', 'small-fake-b'])
    assert out.exit_code == 0
    assert 'small-fake-a==0.2' in out.stderr.splitlines()
    assert 'small-fake-b==0.3' in out.stderr.splitlines()
    assert 'WARNING: the output file requirements.txt exists but is empty' not in out.stderr

def test_upgrade_packages_empty_target_file_warning(pip_conf, runner):
    if False:
        for i in range(10):
            print('nop')
    '\n    piptools warns the user if --upgrade-package/-P is specified and the\n    output file exists, but is empty.\n    '
    with open('requirements.in', 'w') as req_in:
        req_in.write('small-fake-a==0.2')
    with open('requirements.txt', 'w') as req_txt:
        req_txt.write('')
    out = runner.invoke(cli, ['--no-annotate', '-P', 'small-fake-a'])
    assert out.exit_code == 0
    assert 'small-fake-a==0.2' in out.stderr.splitlines()
    assert 'WARNING: the output file requirements.txt exists but is empty' in out.stderr

@pytest.mark.parametrize(('current_package', 'upgraded_package'), (pytest.param('small-fake-b==0.1', 'small-fake-b==0.3', id='upgrade'), pytest.param('small-fake-b==0.3', 'small-fake-b==0.1', id='downgrade')))
def test_upgrade_packages_version_option(pip_conf, runner, current_package, upgraded_package):
    if False:
        while True:
            i = 10
    '\n    piptools respects --upgrade-package/-P inline list with specified versions.\n    '
    with open('requirements.in', 'w') as req_in:
        req_in.write('small-fake-a\nsmall-fake-b')
    with open('requirements.txt', 'w') as req_in:
        req_in.write('small-fake-a==0.1\n' + current_package)
    out = runner.invoke(cli, ['--no-annotate', '--upgrade-package', upgraded_package])
    assert out.exit_code == 0
    stderr_lines = out.stderr.splitlines()
    assert 'small-fake-a==0.1' in stderr_lines
    assert upgraded_package in stderr_lines

def test_upgrade_packages_version_option_no_existing_file(pip_conf, runner):
    if False:
        while True:
            i = 10
    '\n    piptools respects --upgrade-package/-P inline list with specified versions.\n    '
    with open('requirements.in', 'w') as req_in:
        req_in.write('small-fake-a\nsmall-fake-b')
    out = runner.invoke(cli, ['-P', 'small-fake-b==0.2'])
    assert out.exit_code == 0
    assert 'small-fake-a==0.2' in out.stderr
    assert 'small-fake-b==0.2' in out.stderr

@pytest.mark.parametrize('reqs_in', (pytest.param('small-fake-a\nsmall-fake-b', id='direct reqs'), pytest.param('small-fake-with-unpinned-deps', id='parent req')))
def test_upgrade_packages_version_option_and_upgrade(pip_conf, runner, reqs_in):
    if False:
        i = 10
        return i + 15
    '\n    piptools respects --upgrade-package/-P inline list with specified versions\n    whilst also doing --upgrade.\n    '
    with open('requirements.in', 'w') as req_in:
        req_in.write(reqs_in)
    with open('requirements.txt', 'w') as req_in:
        req_in.write('small-fake-a==0.1\nsmall-fake-b==0.1')
    out = runner.invoke(cli, ['--upgrade', '-P', 'small-fake-b==0.1'])
    assert out.exit_code == 0
    assert 'small-fake-a==0.2' in out.stderr
    assert 'small-fake-b==0.1' in out.stderr

def test_upgrade_packages_version_option_and_upgrade_no_existing_file(pip_conf, runner):
    if False:
        while True:
            i = 10
    "\n    piptools respects --upgrade-package/-P inline list with specified versions\n    whilst also doing --upgrade and the output file doesn't exist.\n    "
    with open('requirements.in', 'w') as req_in:
        req_in.write('small-fake-a\nsmall-fake-b')
    out = runner.invoke(cli, ['--upgrade', '-P', 'small-fake-b==0.1'])
    assert out.exit_code == 0
    assert 'small-fake-a==0.2' in out.stderr
    assert 'small-fake-b==0.1' in out.stderr

def test_upgrade_package_with_extra(runner, make_package, make_sdist, tmpdir):
    if False:
        i = 10
        return i + 15
    '\n    piptools ignores extras on --upgrade-package/-P items if already constrained.\n    '
    test_package_1 = make_package('test_package_1', version='0.1', extras_require={'more': 'test_package_2'})
    test_package_2 = make_package('test_package_2', version='0.1')
    dists_dir = tmpdir / 'dists'
    for pkg in (test_package_1, test_package_2):
        make_sdist(pkg, dists_dir)
    with open('requirements.in', 'w') as req_in:
        req_in.write('test-package-1[more]')
    out = runner.invoke(cli, ['--output-file', '-', '--quiet', '--find-links', str(dists_dir), '--no-annotate', '--no-header', '--no-emit-options', '--no-build-isolation', '--upgrade-package', 'test-package-1[more]'])
    assert out.exit_code == 0, out
    assert dedent('            test-package-1[more]==0.1\n            test-package-2==0.1\n            ') == out.stdout

def test_quiet_option(pip_conf, runner):
    if False:
        i = 10
        return i + 15
    with open('requirements.in', 'w') as req_in:
        req_in.write('small-fake-a')
    out = runner.invoke(cli, ['--quiet'])
    assert b'small-fake-a' not in out.stderr_bytes

def test_dry_run_noisy_option(runner):
    if False:
        for i in range(10):
            print('nop')
    with open('requirements.in', 'w'):
        pass
    out = runner.invoke(cli, ['--dry-run'])
    assert 'Dry-run, so nothing updated.' in out.stderr.splitlines()

def test_dry_run_quiet_option(runner):
    if False:
        return 10
    with open('requirements.in', 'w'):
        pass
    out = runner.invoke(cli, ['--output-file', '-', '--dry-run', '--quiet'])
    assert not out.stdout_bytes
    assert 'dry-run' not in out.stderr.lower()
    assert '# ' in out.stderr

def test_generate_hashes_with_editable(pip_conf, runner):
    if False:
        while True:
            i = 10
    small_fake_package_dir = os.path.join(PACKAGES_PATH, 'small_fake_with_deps')
    small_fake_package_url = path_to_url(small_fake_package_dir)
    with open('requirements.in', 'w') as fp:
        fp.write(f'-e {small_fake_package_url}\n')
    out = runner.invoke(cli, ['--no-annotate', '--generate-hashes'])
    expected = '-e {}\nsmall-fake-a==0.1 \\\n    --hash=sha256:5e6071ee6e4c59e0d0408d366fe9b66781d2cf01be9a6e19a2433bb3c5336330\nsmall-fake-b==0.1 \\\n    --hash=sha256:acdba8f8b8a816213c30d5310c3fe296c0107b16ed452062f7f994a5672e3b3f\n'.format(small_fake_package_url)
    assert out.exit_code == 0
    assert expected in out.stderr

@pytest.mark.network
def test_generate_hashes_with_url(runner):
    if False:
        while True:
            i = 10
    with open('requirements.in', 'w') as fp:
        fp.write('https://github.com/jazzband/pip-tools/archive/7d86c8d3ecd1faa6be11c7ddc6b29a30ffd1dae3.zip#egg=pip-tools\n')
    out = runner.invoke(cli, ['--no-annotate', '--generate-hashes'])
    expected = 'pip-tools @ https://github.com/jazzband/pip-tools/archive/7d86c8d3ecd1faa6be11c7ddc6b29a30ffd1dae3.zip \\\n    --hash=sha256:d24de92e18ad5bf291f25cfcdcf0171be6fa70d01d0bef9eeda356b8549715e7\n'
    assert out.exit_code == 0
    assert expected in out.stderr

def test_generate_hashes_verbose(pip_conf, runner):
    if False:
        for i in range(10):
            print('nop')
    '\n    The hashes generation process should show a progress.\n    '
    with open('requirements.in', 'w') as fp:
        fp.write('small-fake-a==0.1')
    out = runner.invoke(cli, ['--generate-hashes', '-v'])
    expected_verbose_text = 'Generating hashes:\n  small-fake-a\n'
    assert expected_verbose_text in out.stderr

@pytest.mark.network
def test_generate_hashes_with_annotations(runner):
    if False:
        print('Hello World!')
    with open('requirements.in', 'w') as fp:
        fp.write('six==1.15.0')
    out = runner.invoke(cli, ['--output-file', '-', '--quiet', '--no-header', '--generate-hashes'])
    assert out.stdout == dedent('        six==1.15.0 \\\n            --hash=sha256:30639c035cdb23534cd4aa2dd52c3bf48f06e5f4a941509c8bafd8ce11080259 \\\n            --hash=sha256:8b74bedcbbbaca38ff6d7491d76f2b06b3592611af620f8426e82dddb04a5ced\n            # via -r requirements.in\n        ')

@pytest.mark.network
@pytest.mark.parametrize('gen_hashes', (True, False))
@pytest.mark.parametrize('annotate_options', (('--no-annotate',), ('--annotation-style', 'line'), ('--annotation-style', 'split')))
@pytest.mark.parametrize(('nl_options', 'must_include', 'must_exclude'), (pytest.param(('--newline', 'lf'), b'\n', b'\r\n', id='LF'), pytest.param(('--newline', 'crlf'), b'\r\n', b'\n', id='CRLF'), pytest.param(('--newline', 'native'), os.linesep.encode(), {'\n': b'\r\n', '\r\n': b'\n'}[os.linesep], id='native')))
def test_override_newline(pip_conf, runner, gen_hashes, annotate_options, nl_options, must_include, must_exclude, tmp_path):
    if False:
        while True:
            i = 10
    opts = annotate_options + nl_options
    if gen_hashes:
        opts += ('--generate-hashes',)
    example_dir = tmp_path / 'example_dir'
    example_dir.mkdir()
    in_path = example_dir / 'requirements.in'
    out_path = example_dir / 'requirements.txt'
    in_path.write_bytes(b'small-fake-a==0.1\nsmall-fake-b\n')
    runner.invoke(cli, [*opts, f'--output-file={os.fsdecode(out_path)}', os.fsdecode(in_path)])
    txt = out_path.read_bytes()
    assert must_include in txt
    if must_exclude in must_include:
        txt = txt.replace(must_include, b'')
    assert must_exclude not in txt
    opts = annotate_options + ('--newline', 'preserve')
    if gen_hashes:
        opts += ('--generate-hashes',)
    runner.invoke(cli, [*opts, f'--output-file={os.fsdecode(out_path)}', os.fsdecode(in_path)])
    txt = out_path.read_bytes()
    assert must_include in txt
    if must_exclude in must_include:
        txt = txt.replace(must_include, b'')
    assert must_exclude not in txt

@pytest.mark.network
@pytest.mark.parametrize(('linesep', 'must_exclude'), (pytest.param('\n', '\r\n', id='LF'), pytest.param('\r\n', '\n', id='CRLF')))
def test_preserve_newline_from_input(runner, linesep, must_exclude):
    if False:
        while True:
            i = 10
    with open('requirements.in', 'wb') as req_in:
        req_in.write(f'six{linesep}'.encode())
    runner.invoke(cli, ['--newline=preserve', 'requirements.in'])
    with open('requirements.txt', 'rb') as req_txt:
        txt = req_txt.read().decode()
    assert linesep in txt
    if must_exclude in linesep:
        txt = txt.replace(linesep, '')
    assert must_exclude not in txt

def test_generate_hashes_with_split_style_annotations(pip_conf, runner, tmpdir_cwd):
    if False:
        i = 10
        return i + 15
    reqs_in = tmpdir_cwd / 'requirements.in'
    reqs_in.write_text(dedent('            small_fake_with_deps\n            small-fake-a\n            '))
    out = runner.invoke(cli, ['--output-file', '-', '--quiet', '--no-header', '--no-emit-options', '--generate-hashes', '--annotation-style', 'split'])
    assert out.stdout == dedent('        small-fake-a==0.1 \\\n            --hash=sha256:5e6071ee6e4c59e0d0408d366fe9b66781d2cf01be9a6e19a2433bb3c5336330\n            # via\n            #   -r requirements.in\n            #   small-fake-with-deps\n        small-fake-with-deps==0.1 \\\n            --hash=sha256:71403033c0545516cc5066c9196d9490affae65a865af3198438be6923e4762e\n            # via -r requirements.in\n        ')

def test_generate_hashes_with_line_style_annotations(pip_conf, runner, tmpdir_cwd):
    if False:
        return 10
    reqs_in = tmpdir_cwd / 'requirements.in'
    reqs_in.write_text(dedent('            small_fake_with_deps\n            small-fake-a\n            '))
    out = runner.invoke(cli, ['--output-file', '-', '--quiet', '--no-header', '--no-emit-options', '--generate-hashes', '--annotation-style', 'line'])
    assert out.stdout == dedent('        small-fake-a==0.1 \\\n            --hash=sha256:5e6071ee6e4c59e0d0408d366fe9b66781d2cf01be9a6e19a2433bb3c5336330\n            # via -r requirements.in, small-fake-with-deps\n        small-fake-with-deps==0.1 \\\n            --hash=sha256:71403033c0545516cc5066c9196d9490affae65a865af3198438be6923e4762e\n            # via -r requirements.in\n        ')

@pytest.mark.network
def test_generate_hashes_with_mixed_sources(runner, make_package, make_wheel, make_sdist, tmp_path):
    if False:
        i = 10
        return i + 15
    '\n    Test that pip-compile generate hashes for every file from all given sources:\n    PyPI and/or --find-links.\n    '
    wheels_dir = tmp_path / 'wheels'
    wheels_dir.mkdir()
    dummy_six_pkg = make_package(name='six', version='1.16.0')
    make_wheel(dummy_six_pkg, wheels_dir, '--build-number', '123')
    fav_hasher = hashlib.new(FAVORITE_HASH)
    fav_hasher.update((wheels_dir / 'six-1.16.0-123-py3-none-any.whl').read_bytes())
    dummy_six_wheel_digest = fav_hasher.hexdigest()
    with open('requirements.in', 'w') as fp:
        fp.write('six==1.16.0\n')
    out = runner.invoke(cli, ['--output-file', '-', '--quiet', '--no-header', '--generate-hashes', '--no-emit-options', '--no-annotate', '--find-links', wheels_dir.as_uri()])
    expected_digests = sorted(('1e61c37477a1626458e36f7b1d82aa5c9b094fa4802892072e49de9c60c4c926', '8abb2f1d86890a2dfb989f9a77cfcfd3e47c2a354b01111771326f8aa26e0254', dummy_six_wheel_digest))
    expected_output = dedent(f'        six==1.16.0 \\\n            --hash=sha256:{expected_digests[0]} \\\n            --hash=sha256:{expected_digests[1]} \\\n            --hash=sha256:{expected_digests[2]}\n        ')
    assert out.stdout == expected_output

def test_filter_pip_markers(pip_conf, runner):
    if False:
        return 10
    '\n    Check that pip-compile works with pip environment markers (PEP496)\n    '
    with open('requirements.in', 'w') as req_in:
        req_in.write("small-fake-a==0.1\nunknown_package==0.1; python_version == '1'")
    out = runner.invoke(cli, ['--output-file', '-', '--quiet'])
    assert out.exit_code == 0
    assert 'small-fake-a==0.1' in out.stdout
    assert 'unknown_package' not in out.stdout

def test_bad_setup_file(runner):
    if False:
        for i in range(10):
            print('nop')
    with open('setup.py', 'w') as package:
        package.write('BAD SYNTAX')
    out = runner.invoke(cli, ['--no-build-isolation'])
    assert out.exit_code == 2
    assert f"Failed to parse {os.path.abspath('setup.py')}" in out.stderr

@legacy_resolver_only
def test_no_candidates(pip_conf, runner):
    if False:
        i = 10
        return i + 15
    with open('requirements', 'w') as req_in:
        req_in.write('small-fake-a>0.3b1,<0.3b2')
    out = runner.invoke(cli, ['-n', 'requirements'])
    assert out.exit_code == 2
    assert 'Skipped pre-versions:' in out.stderr

@legacy_resolver_only
def test_no_candidates_pre(pip_conf, runner):
    if False:
        return 10
    with open('requirements', 'w') as req_in:
        req_in.write('small-fake-a>0.3b1,<0.3b1')
    out = runner.invoke(cli, ['-n', 'requirements', '--pre'])
    assert out.exit_code == 2
    assert 'Tried pre-versions:' in out.stderr

@pytest.mark.parametrize(('url', 'expected_url'), (pytest.param('https://example.com', b'https://example.com', id='regular url'), pytest.param('https://username:password@example.com', b'https://username:****@example.com', id='url with credentials')))
def test_default_index_url(make_pip_conf, url, expected_url):
    if False:
        i = 10
        return i + 15
    "\n    Test help's output with default index URL.\n    "
    make_pip_conf(dedent(f'            [global]\n            index-url = {url}\n            '))
    result = subprocess.run([sys.executable, '-m', 'piptools', 'compile', '--help'], stdout=subprocess.PIPE, check=True)
    assert expected_url in result.stdout

def test_stdin_without_output_file(runner):
    if False:
        for i in range(10):
            print('nop')
    '\n    The --output-file option is required for STDIN.\n    '
    out = runner.invoke(cli, ['-n', '-'])
    assert out.exit_code == 2
    assert '--output-file is required if input is from stdin' in out.stderr

def test_stdin(pip_conf, runner):
    if False:
        return 10
    '\n    Test compile requirements from STDIN.\n    '
    out = runner.invoke(cli, ['-', '--output-file', '-', '--quiet', '--no-emit-options', '--no-header'], input='small-fake-a==0.1')
    assert out.stdout == dedent('        small-fake-a==0.1\n            # via -r -\n        ')

def test_multiple_input_files_without_output_file(runner):
    if False:
        print('Hello World!')
    '\n    The --output-file option is required for multiple requirement input files.\n    '
    with open('src_file1.in', 'w') as req_in:
        req_in.write('six==1.10.0')
    with open('src_file2.in', 'w') as req_in:
        req_in.write('django==2.1')
    out = runner.invoke(cli, ['src_file1.in', 'src_file2.in'])
    assert '--output-file is required if two or more input files are given' in out.stderr
    assert out.exit_code == 2

@pytest.mark.parametrize(('options', 'expected'), (pytest.param(('--annotate',), '            small-fake-a==0.1\n                # via\n                #   -c constraints.txt\n                #   small-fake-with-deps\n            small-fake-with-deps==0.1\n                # via -r requirements.in\n            ', id='annotate'), pytest.param(('--annotate', '--annotation-style', 'line'), '            small-fake-a==0.1         # via -c constraints.txt, small-fake-with-deps\n            small-fake-with-deps==0.1  # via -r requirements.in\n            ', id='annotate line style'), pytest.param(('--no-annotate',), '            small-fake-a==0.1\n            small-fake-with-deps==0.1\n            ', id='no annotate')))
def test_annotate_option(pip_conf, runner, options, expected):
    if False:
        for i in range(10):
            print('nop')
    '\n    The output lines have annotations if the option is turned on.\n    '
    with open('constraints.txt', 'w') as constraints_in:
        constraints_in.write('small-fake-a==0.1')
    with open('requirements.in', 'w') as req_in:
        req_in.write('-c constraints.txt\n')
        req_in.write('small_fake_with_deps')
    out = runner.invoke(cli, [*options, '--output-file', '-', '--quiet', '--no-emit-options', '--no-header'])
    assert out.exit_code == 0, out
    assert out.stdout == dedent(expected)

@pytest.mark.parametrize(('option', 'expected'), (pytest.param('--allow-unsafe', dedent('                small-fake-a==0.1\n                small-fake-b==0.3\n\n                # The following packages are considered to be unsafe in a requirements file:\n                small-fake-with-deps==0.1\n                '), id='allow all packages'), pytest.param('--no-allow-unsafe', dedent('                small-fake-a==0.1\n                small-fake-b==0.3\n\n                # The following packages are considered to be unsafe in a requirements file:\n                # small-fake-with-deps\n                '), id='comment out small-fake-with-deps and its dependencies'), pytest.param(None, dedent('                small-fake-a==0.1\n                small-fake-b==0.3\n\n                # The following packages are considered to be unsafe in a requirements file:\n                # small-fake-with-deps\n                '), id='allow unsafe is default option')))
def test_allow_unsafe_option(pip_conf, monkeypatch, runner, option, expected):
    if False:
        return 10
    '\n    Unsafe packages are printed as expected with and without --allow-unsafe.\n    '
    monkeypatch.setattr('piptools.resolver.UNSAFE_PACKAGES', {'small-fake-with-deps'})
    with open('requirements.in', 'w') as req_in:
        req_in.write('small-fake-b\n')
        req_in.write('small-fake-with-deps')
    out = runner.invoke(cli, ['--output-file', '-', '--quiet', '--no-header', '--no-emit-options', '--no-annotate', *([option] if option else [])])
    assert out.exit_code == 0, out
    assert out.stdout == expected

@pytest.mark.parametrize(('unsafe_package', 'expected'), (('small-fake-with-deps', dedent('                small-fake-a==0.1\n                small-fake-b==0.3\n\n                # The following packages are considered to be unsafe in a requirements file:\n                # small-fake-with-deps\n                ')), ('small-fake-a', dedent('                small-fake-b==0.3\n                small-fake-with-deps==0.1\n\n                # The following packages are considered to be unsafe in a requirements file:\n                # small-fake-a\n                '))))
def test_unsafe_package_option(pip_conf, monkeypatch, runner, unsafe_package, expected):
    if False:
        while True:
            i = 10
    monkeypatch.setattr('piptools.resolver.UNSAFE_PACKAGES', {'small-fake-with-deps'})
    with open('requirements.in', 'w') as req_in:
        req_in.write('small-fake-b\n')
        req_in.write('small-fake-with-deps')
    out = runner.invoke(cli, ['--output-file', '-', '--quiet', '--no-header', '--no-emit-options', '--no-annotate', '--unsafe-package', unsafe_package])
    assert out.exit_code == 0, out
    assert out.stdout == expected

@pytest.mark.parametrize(('option', 'attr', 'expected'), (('--cert', 'cert', 'foo.crt'), ('--client-cert', 'client_cert', 'bar.pem')))
@mock.patch('piptools.scripts.compile.parse_requirements')
def test_cert_option(parse_requirements, runner, option, attr, expected):
    if False:
        print('Hello World!')
    '\n    The options --cert and --client-cert have to be passed to the PyPIRepository.\n    '
    with open('requirements.in', 'w'):
        pass
    runner.invoke(cli, [option, expected])
    (args, kwargs) = parse_requirements.call_args
    assert getattr(kwargs['options'], attr) == expected

@pytest.mark.parametrize(('option', 'expected'), (('--build-isolation', True), ('--no-build-isolation', False)))
@mock.patch('piptools.scripts.compile.parse_requirements')
def test_parse_requirements_build_isolation_option(parse_requirements, runner, option, expected):
    if False:
        while True:
            i = 10
    '\n    A value of the --build-isolation/--no-build-isolation flag\n    must be passed to parse_requirements().\n    '
    with open('requirements.in', 'w'):
        pass
    runner.invoke(cli, [option])
    (args, kwargs) = parse_requirements.call_args
    assert kwargs['options'].build_isolation is expected

@pytest.mark.parametrize(('option', 'expected'), (('--build-isolation', True), ('--no-build-isolation', False)))
@mock.patch('piptools.scripts.compile.build_project_metadata')
def test_build_project_metadata_isolation_option(build_project_metadata, runner, option, expected):
    if False:
        print('Hello World!')
    '\n    A value of the --build-isolation/--no-build-isolation flag\n    must be passed to build_project_metadata().\n    '
    with open('setup.py', 'w') as package:
        package.write(dedent('                from setuptools import setup\n                setup(install_requires=[])\n                '))
    runner.invoke(cli, [option])
    (_, kwargs) = build_project_metadata.call_args
    assert kwargs['isolated'] is expected

@mock.patch('piptools.scripts.compile.PyPIRepository')
def test_forwarded_args(PyPIRepository, runner):
    if False:
        while True:
            i = 10
    "\n    Test the forwarded cli args (--pip-args 'arg...') are passed to the pip command.\n    "
    with open('requirements.in', 'w'):
        pass
    cli_args = ('--no-annotate', '--generate-hashes')
    pip_args = ('--no-color', '--isolated', '--disable-pip-version-check')
    runner.invoke(cli, [*cli_args, '--pip-args', ' '.join(pip_args)])
    (args, kwargs) = PyPIRepository.call_args
    assert set(pip_args).issubset(set(args[0]))

@pytest.mark.parametrize(('cli_option', 'infile_option', 'expected_package'), ((False, False, 'small-fake-a==0.2'), (True, False, 'small-fake-a==0.3b1'), (False, True, 'small-fake-a==0.3b1'), (True, True, 'small-fake-a==0.3b1')))
def test_pre_option(pip_conf, runner, cli_option, infile_option, expected_package):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests pip-compile respects --pre option.\n    '
    with open('requirements.in', 'w') as req_in:
        if infile_option:
            req_in.write('--pre\n')
        req_in.write('small-fake-a\n')
    out = runner.invoke(cli, ['--no-annotate', '-n'] + (['-p'] if cli_option else []))
    assert out.exit_code == 0, out.stderr
    assert expected_package in out.stderr.splitlines(), out.stderr

@pytest.mark.parametrize('add_options', ([], ['--output-file', 'requirements.txt'], ['--upgrade'], ['--upgrade', '--output-file', 'requirements.txt'], ['--upgrade-package', 'small-fake-a'], ['--upgrade-package', 'small-fake-a', '--output-file', 'requirements.txt']))
def test_dry_run_option(pip_conf, runner, add_options):
    if False:
        return 10
    "\n    Tests pip-compile doesn't create requirements.txt file on dry-run.\n    "
    with open('requirements.in', 'w') as req_in:
        req_in.write('small-fake-a\n')
    out = runner.invoke(cli, ['--no-annotate', '--dry-run', *add_options])
    assert out.exit_code == 0, out.stderr
    assert 'small-fake-a==0.2' in out.stderr.splitlines()
    assert not os.path.exists('requirements.txt')

@pytest.mark.parametrize(('add_options', 'expected_cli_output_package'), (([], 'small-fake-a==0.1'), (['--output-file', 'requirements.txt'], 'small-fake-a==0.1'), (['--upgrade'], 'small-fake-a==0.2'), (['--upgrade', '--output-file', 'requirements.txt'], 'small-fake-a==0.2'), (['--upgrade-package', 'small-fake-a'], 'small-fake-a==0.2'), (['--upgrade-package', 'small-fake-a', '--output-file', 'requirements.txt'], 'small-fake-a==0.2')))
def test_dry_run_doesnt_touch_output_file(pip_conf, runner, add_options, expected_cli_output_package):
    if False:
        for i in range(10):
            print('nop')
    "\n    Tests pip-compile doesn't touch requirements.txt file on dry-run.\n    "
    with open('requirements.in', 'w') as req_in:
        req_in.write('small-fake-a\n')
    with open('requirements.txt', 'w') as req_txt:
        req_txt.write('small-fake-a==0.1\n')
    before_compile_mtime = os.stat('requirements.txt').st_mtime
    out = runner.invoke(cli, ['--no-annotate', '--dry-run', *add_options])
    assert out.exit_code == 0, out.stderr
    assert expected_cli_output_package in out.stderr.splitlines()
    with open('requirements.txt') as req_txt:
        assert 'small-fake-a==0.1' in req_txt.read().splitlines()
    after_compile_mtime = os.stat('requirements.txt').st_mtime
    assert after_compile_mtime == before_compile_mtime

@pytest.mark.parametrize(('empty_input_pkg', 'prior_output_pkg'), (('', ''), ('', 'small-fake-a==0.1\n'), ('# Nothing to see here', ''), ('# Nothing to see here', 'small-fake-a==0.1\n')))
def test_empty_input_file_no_header(runner, empty_input_pkg, prior_output_pkg):
    if False:
        i = 10
        return i + 15
    '\n    Tests pip-compile creates an empty requirements.txt file,\n    given --no-header and empty requirements.in\n    '
    with open('requirements.in', 'w') as req_in:
        req_in.write(empty_input_pkg)
    with open('requirements.txt', 'w') as req_txt:
        req_txt.write(prior_output_pkg)
    runner.invoke(cli, ['--no-header', 'requirements.in'])
    with open('requirements.txt') as req_txt:
        assert req_txt.read().strip() == ''

def test_upgrade_package_doesnt_remove_annotation(pip_conf, runner):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests pip-compile --upgrade-package shouldn\'t remove "via" annotation.\n    See: GH-929\n    '
    with open('requirements.in', 'w') as req_in:
        req_in.write('small-fake-with-deps\n')
    runner.invoke(cli)
    with open('requirements.txt', 'w') as req_txt:
        req_txt.write('small-fake-with-deps==0.1\nsmall-fake-a==0.1         # via small-fake-with-deps\n')
    runner.invoke(cli, ['-P', 'small-fake-a', '--no-emit-options', '--no-header'])
    with open('requirements.txt') as req_txt:
        assert req_txt.read() == dedent('            small-fake-a==0.1\n                # via small-fake-with-deps\n            small-fake-with-deps==0.1\n                # via -r requirements.in\n            ')

@pytest.mark.parametrize('num_inputs', (2, 3, 10))
def test_many_inputs_includes_all_annotations(pip_conf, runner, tmp_path, num_inputs):
    if False:
        return 10
    '\n    Tests that an entry required by multiple input files is attributed to all of them in the\n    annotation.\n    See: https://github.com/jazzband/pip-tools/issues/1853\n    '
    req_ins = [tmp_path / f'requirements{n:02d}.in' for n in range(num_inputs)]
    for req_in in req_ins:
        req_in.write_text('small-fake-a==0.1\n')
    out = runner.invoke(cli, ['--output-file', '-', '--quiet', '--no-header', '--no-emit-find-links'] + [str(r) for r in req_ins])
    assert out.exit_code == 0, out.stderr
    assert out.stdout == '\n'.join(['small-fake-a==0.1', '    # via'] + [f'    #   -r {req_in}' for req_in in req_ins]) + '\n'

@pytest.mark.parametrize('options', ('--index-url https://example.com', '--extra-index-url https://example.com', '--find-links ./libs1', '--trusted-host example.com', '--no-binary :all:', '--only-binary :all:'))
def test_options_in_requirements_file(runner, options):
    if False:
        print('Hello World!')
    '\n    Test the options from requirements.in is copied to requirements.txt.\n    '
    with open('requirements.in', 'w') as reqs_in:
        reqs_in.write(options)
    out = runner.invoke(cli)
    assert out.exit_code == 0, out
    with open('requirements.txt') as reqs_txt:
        assert options in reqs_txt.read().splitlines()

@pytest.mark.parametrize(('cli_options', 'expected_message'), (pytest.param(['--index-url', 'scheme://foo'], 'Was scheme://foo reachable?', id='single index url'), pytest.param(['--index-url', 'scheme://foo', '--extra-index-url', 'scheme://bar'], 'Were scheme://foo or scheme://bar reachable?', id='multiple index urls'), pytest.param(['--index-url', 'scheme://username:password@host'], 'Was scheme://username:****@host reachable?', id='index url with credentials')))
@legacy_resolver_only
def test_unreachable_index_urls(runner, cli_options, expected_message):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test pip-compile raises an error if index URLs are not reachable.\n    '
    with open('requirements.in', 'w') as reqs_in:
        reqs_in.write('some-package')
    out = runner.invoke(cli, cli_options)
    assert out.exit_code == 2, out
    stderr_lines = out.stderr.splitlines()
    assert 'No versions found' in stderr_lines
    assert expected_message in stderr_lines

@pytest.mark.parametrize('subdep_already_pinned', (True, False))
@pytest.mark.parametrize(('current_package', 'upgraded_package'), (pytest.param('small-fake-b==0.1', 'small-fake-b==0.2', id='upgrade'), pytest.param('small-fake-b==0.2', 'small-fake-b==0.1', id='downgrade')))
def test_upgrade_packages_option_subdependency(pip_conf, runner, current_package, upgraded_package, subdep_already_pinned):
    if False:
        print('Hello World!')
    '\n    Test that pip-compile --upgrade-package/-P upgrades/downgrades subdependencies.\n    '
    with open('requirements.in', 'w') as reqs:
        reqs.write('small-fake-with-unpinned-deps\n')
    with open('requirements.txt', 'w') as reqs:
        reqs.write('small-fake-a==0.1\n')
        if subdep_already_pinned:
            reqs.write(current_package + '\n')
        reqs.write('small-fake-with-unpinned-deps==0.1\n')
    out = runner.invoke(cli, ['--no-annotate', '--dry-run', '--upgrade-package', upgraded_package])
    stderr_lines = out.stderr.splitlines()
    assert 'small-fake-a==0.1' in stderr_lines, 'small-fake-a must keep its version'
    assert upgraded_package in stderr_lines, f'{current_package} must be upgraded/downgraded to {upgraded_package}'

@pytest.mark.parametrize(('input_opts', 'output_opts'), (pytest.param('--index-url https://index-url', '--index-url https://another-index-url', id='index url'), pytest.param('--extra-index-url https://extra-index-url', '--extra-index-url https://another-extra-index-url', id='extra index url'), pytest.param('--find-links dir', '--find-links another-dir', id='find links'), pytest.param('--trusted-host hostname', '--trusted-host another-hostname', id='trusted host'), pytest.param('--no-binary :package:', '--no-binary :another-package:', id='no binary'), pytest.param('--only-binary :package:', '--only-binary :another-package:', id='only binary'), pytest.param('', '--index-url https://index-url', id='empty input options'), pytest.param('--index-url https://index-url', '--index-url https://index-url\n--extra-index-url https://another-extra-index-url', id='partially matched options')))
def test_remove_outdated_options(runner, input_opts, output_opts):
    if False:
        print('Hello World!')
    "\n    Test that the options from the current requirements.txt wouldn't stay\n    after compile if they were removed from requirements.in file.\n    "
    with open('requirements.in', 'w') as req_in:
        req_in.write(input_opts)
    with open('requirements.txt', 'w') as req_txt:
        req_txt.write(output_opts)
    out = runner.invoke(cli, ['--output-file', '-', '--quiet', '--no-header'])
    assert out.exit_code == 0, out
    assert out.stdout.strip() == input_opts

def test_sub_dependencies_with_constraints(pip_conf, runner):
    if False:
        return 10
    with open('constraints.txt', 'w') as constraints_in:
        constraints_in.write('small-fake-a==0.1\n')
        constraints_in.write('small-fake-b==0.2\n')
        constraints_in.write('small-fake-with-unpinned-deps==0.1')
    with open('requirements.in', 'w') as req_in:
        req_in.write('-c constraints.txt\n')
        req_in.write('small_fake_with_deps_and_sub_deps')
    out = runner.invoke(cli, ['--no-annotate'])
    assert out.exit_code == 0
    req_out_lines = set(out.stderr.splitlines())
    assert {'small-fake-a==0.1', 'small-fake-b==0.2', 'small-fake-with-deps-and-sub-deps==0.1', 'small-fake-with-unpinned-deps==0.1'}.issubset(req_out_lines)

def test_preserve_compiled_prerelease_version(pip_conf, runner):
    if False:
        i = 10
        return i + 15
    with open('requirements.in', 'w') as req_in:
        req_in.write('small-fake-a')
    with open('requirements.txt', 'w') as req_txt:
        req_txt.write('small-fake-a==0.3b1')
    out = runner.invoke(cli, ['--no-annotate', '--no-header'])
    assert out.exit_code == 0, out
    assert 'small-fake-a==0.3b1' in out.stderr.splitlines()

@backtracking_resolver_only
def test_ignore_compiled_unavailable_version(pip_conf, runner, current_resolver):
    if False:
        while True:
            i = 10
    with open('requirements.in', 'w') as req_in:
        req_in.write('small-fake-a')
    with open('requirements.txt', 'w') as req_txt:
        req_txt.write('small-fake-a==9999')
    out = runner.invoke(cli, ['--no-annotate', '--no-header'])
    assert out.exit_code == 0, out
    assert 'small-fake-a==' in out.stderr
    assert 'small-fake-a==9999' not in out.stderr.splitlines()
    assert 'Discarding small-fake-a==9999 (from -r requirements.txt (line 1)) to proceed the resolution' in out.stderr

def test_prefer_binary_dist(pip_conf, make_package, make_sdist, make_wheel, tmpdir, runner):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test pip-compile chooses a correct version of a package with\n    a binary distribution when PIP_PREFER_BINARY environment variable is on.\n    '
    dists_dir = tmpdir / 'dists'
    first_package_v1 = make_package(name='first-package', version='1.0')
    make_wheel(first_package_v1, dists_dir)
    first_package_v2 = make_package(name='first-package', version='2.0')
    make_sdist(first_package_v2, dists_dir)
    second_package_v1 = make_package(name='second-package', version='1.0', install_requires=['first-package'])
    make_wheel(second_package_v1, dists_dir)
    with open('requirements.in', 'w') as req_in:
        req_in.write('second-package')
    out = runner.invoke(cli, ['--no-annotate', '--find-links', str(dists_dir)], env={'PIP_PREFER_BINARY': '1'})
    assert out.exit_code == 0, out
    assert 'first-package==1.0' in out.stderr.splitlines(), out.stderr
    assert 'second-package==1.0' in out.stderr.splitlines(), out.stderr

@pytest.mark.parametrize('prefer_binary', (True, False))
def test_prefer_binary_dist_even_there_is_source_dists(pip_conf, make_package, make_sdist, make_wheel, tmpdir, runner, prefer_binary):
    if False:
        i = 10
        return i + 15
    '\n    Test pip-compile chooses a correct version of a package with a binary distribution\n    (despite a source dist existing) when PIP_PREFER_BINARY environment variable is on\n    or off.\n\n    Regression test for issue GH-1118.\n    '
    dists_dir = tmpdir / 'dists'
    package_v1 = make_package(name='test-package', version='1.0')
    make_wheel(package_v1, dists_dir)
    package_v2 = make_package(name='test-package', version='2.0')
    make_wheel(package_v2, dists_dir)
    make_sdist(package_v2, dists_dir)
    with open('requirements.in', 'w') as req_in:
        req_in.write('test-package')
    out = runner.invoke(cli, ['--no-annotate', '--find-links', str(dists_dir)], env={'PIP_PREFER_BINARY': str(int(prefer_binary))})
    assert out.exit_code == 0, out
    assert 'test-package==2.0' in out.stderr.splitlines(), out.stderr

@pytest.mark.parametrize('output_content', ('test-package-1==0.1', ''))
def test_duplicate_reqs_combined(pip_conf, make_package, make_sdist, tmpdir, runner, output_content):
    if False:
        i = 10
        return i + 15
    '\n    Test pip-compile tracks dependencies properly when install requirements are\n    combined, especially when an output file already exists.\n\n    Regression test for issue GH-1154.\n    '
    test_package_1 = make_package('test_package_1', version='0.1')
    test_package_2 = make_package('test_package_2', version='0.1', install_requires=['test-package-1'])
    dists_dir = tmpdir / 'dists'
    for pkg in (test_package_1, test_package_2):
        make_sdist(pkg, dists_dir)
    with open('requirements.in', 'w') as reqs_in:
        reqs_in.write(f'file:{test_package_2}\n')
        reqs_in.write(f'file:{test_package_2}#egg=test-package-2\n')
    if output_content:
        with open('requirements.txt', 'w') as reqs_out:
            reqs_out.write(output_content)
    out = runner.invoke(cli, ['--find-links', str(dists_dir)])
    assert out.exit_code == 0, out
    assert str(test_package_2) in out.stderr
    assert 'test-package-1==0.1' in out.stderr

def test_local_duplicate_subdependency_combined(runner, make_package):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test pip-compile tracks subdependencies properly when install requirements\n    are combined, especially when local paths are passed as urls, and those reqs\n    are combined after getting dependencies.\n\n    Regression test for issue GH-1505.\n    '
    package_a = make_package('project-a', install_requires=['pip-tools==6.3.0'])
    package_b = make_package('project-b', install_requires=['project-a'])
    with open('requirements.in', 'w') as req_in:
        req_in.writelines([f'file://{package_a}#egg=project-a\n', f'file://{package_b}#egg=project-b'])
    out = runner.invoke(cli, ['-n'])
    assert out.exit_code == 0
    assert 'project-b' in out.stderr
    assert 'project-a' in out.stderr
    assert 'pip-tools==6.3.0' in out.stderr
    assert 'click' in out.stderr

def test_combine_extras(pip_conf, runner, make_package):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensure that multiple declarations of a dependency that specify different\n    extras produces a requirement for that package with the union of the extras\n    '
    package_with_extras = make_package('package_with_extras', extras_require={'extra1': ['small-fake-a==0.1'], 'extra2': ['small-fake-b==0.1']})
    with open('requirements.in', 'w') as req_in:
        req_in.writelines(['-r ./requirements-second.in\n', f'{package_with_extras}[extra1]'])
    with open('requirements-second.in', 'w') as req_sec_in:
        req_sec_in.write(f'{package_with_extras}[extra2]')
    out = runner.invoke(cli, ['-n'])
    assert out.exit_code == 0
    assert 'package-with-extras' in out.stderr
    assert 'small-fake-a==' in out.stderr
    assert 'small-fake-b==' in out.stderr

def test_combine_different_extras_of_the_same_package(pip_conf, runner, tmpdir, make_package, make_wheel):
    if False:
        for i in range(10):
            print('nop')
    '\n    Loosely based on the example from https://github.com/jazzband/pip-tools/issues/1511.\n    '
    pkgs = [make_package('fake-colorful', version='0.3'), make_package('fake-tensorboardX', version='0.5'), make_package('fake-ray', version='0.1', extras_require={'default': ['fake-colorful==0.3'], 'tune': ['fake-tensorboardX==0.5']}), make_package('fake-tune-sklearn', version='0.7', install_requires=['fake-ray[tune]==0.1'])]
    dists_dir = tmpdir / 'dists'
    for pkg in pkgs:
        make_wheel(pkg, dists_dir)
    with open('requirements.in', 'w') as req_in:
        req_in.writelines(['fake-ray[default]==0.1\n', 'fake-tune-sklearn==0.7\n'])
    out = runner.invoke(cli, ['--output-file', '-', '--quiet', '--find-links', str(dists_dir), '--no-header', '--no-emit-options'])
    assert out.exit_code == 0
    assert dedent('        fake-colorful==0.3\n            # via fake-ray\n        fake-ray[default,tune]==0.1\n            # via\n            #   -r requirements.in\n            #   fake-tune-sklearn\n        fake-tensorboardx==0.5\n            # via fake-ray\n        fake-tune-sklearn==0.7\n            # via -r requirements.in\n        ') == out.stdout

@pytest.mark.parametrize(('pkg2_install_requires', 'req_in_content', 'out_expected_content'), (pytest.param('', ['test-package-1===0.1.0\n'], ['test-package-1===0.1.0'], id='pin package with ==='), pytest.param('', ['test-package-1==0.1.0\n'], ['test-package-1==0.1.0'], id='pin package with =='), pytest.param('test-package-1==0.1.0', ['test-package-1===0.1.0\n', 'test-package-2==0.1.0\n'], ['test-package-1===0.1.0', 'test-package-2==0.1.0'], id='dep === pin preferred over == pin, main package == pin'), pytest.param('test-package-1==0.1.0', ['test-package-1===0.1.0\n', 'test-package-2===0.1.0\n'], ['test-package-1===0.1.0', 'test-package-2===0.1.0'], id='dep === pin preferred over == pin, main package === pin'), pytest.param('test-package-1==0.1.0', ['test-package-2===0.1.0\n'], ['test-package-1==0.1.0', 'test-package-2===0.1.0'], id='dep == pin conserved, main package === pin')))
def test_triple_equal_pinned_dependency_is_used(runner, make_package, make_wheel, tmpdir, pkg2_install_requires, req_in_content, out_expected_content):
    if False:
        while True:
            i = 10
    '\n    Test that pip-compile properly emits the pinned requirement with ===\n    torchvision 0.8.2 requires torch==1.7.1 which can resolve to versions with\n    patches (e.g. torch 1.7.1+cu110), we want torch===1.7.1 without patches\n    '
    dists_dir = tmpdir / 'dists'
    test_package_1 = make_package('test_package_1', version='0.1.0')
    make_wheel(test_package_1, dists_dir)
    test_package_2 = make_package('test_package_2', version='0.1.0', install_requires=[pkg2_install_requires])
    make_wheel(test_package_2, dists_dir)
    with open('requirements.in', 'w') as reqs_in:
        for line in req_in_content:
            reqs_in.write(line)
    out = runner.invoke(cli, ['--find-links', str(dists_dir)])
    assert out.exit_code == 0, out
    for line in out_expected_content:
        assert line in out.stderr
METADATA_TEST_CASES = (pytest.param('setup.cfg', '\n            [metadata]\n            name = sample_lib\n            author = Vincent Driessen\n            author_email = me@nvie.com\n\n            [options]\n            packages = find:\n            install_requires = small-fake-a==0.1\n\n            [options.extras_require]\n            dev = small-fake-b==0.2\n            test = small-fake-c==0.3\n        ', id='setup.cfg'), pytest.param('setup.py', '\n            from setuptools import setup, find_packages\n\n            setup(\n                name="sample_lib",\n                version=0.1,\n                install_requires=["small-fake-a==0.1"],\n                packages=find_packages(),\n                extras_require={\n                    "dev": ["small-fake-b==0.2"],\n                    "test": ["small-fake-c==0.3"],\n                },\n            )\n        ', id='setup.py'), pytest.param('pyproject.toml', '\n            [build-system]\n            requires = ["flit_core >=2,<4"]\n            build-backend = "flit_core.buildapi"\n\n            [tool.flit.metadata]\n            module = "sample_lib"\n            author = "Vincent Driessen"\n            author-email = "me@nvie.com"\n\n            requires = ["small-fake-a==0.1"]\n\n            [tool.flit.metadata.requires-extra]\n            dev  = ["small-fake-b==0.2"]\n            test = ["small-fake-c==0.3"]\n        ', id='flit'), pytest.param('pyproject.toml', '\n            [build-system]\n            requires = ["poetry_core>=1.0.0"]\n            build-backend = "poetry.core.masonry.api"\n\n            [tool.poetry]\n            name = "sample_lib"\n            version = "0.1.0"\n            description = ""\n            authors = ["Vincent Driessen <me@nvie.com>"]\n\n            [tool.poetry.dependencies]\n            python = "*"\n            small-fake-a = "0.1"\n            small-fake-b = "0.2"\n            small-fake-c = "0.3"\n\n            [tool.poetry.extras]\n            dev  = ["small-fake-b"]\n            test = ["small-fake-c"]\n        ', id='poetry'))

@pytest.mark.network
@pytest.mark.parametrize(('fname', 'content'), METADATA_TEST_CASES)
def test_not_specified_input_file(fake_dists, runner, make_module, fname, content, monkeypatch):
    if False:
        i = 10
        return i + 15
    '\n    Test that a default-named file is parsed if present.\n    '
    meta_path = make_module(fname=fname, content=content)
    monkeypatch.chdir(os.path.dirname(meta_path))
    out = runner.invoke(cli, ['--output-file', '-', '--no-header', '--no-emit-options', '--no-annotate', '--no-build-isolation', '--find-links', fake_dists])
    monkeypatch.undo()
    assert out.exit_code == 0, out.stderr
    assert 'small-fake-a==0.1\n' == out.stdout

def test_not_specified_input_file_without_allowed_files(runner):
    if False:
        return 10
    '\n    It should raise an error if there are no input files or default input files\n    such as "setup.py" or "requirements.in".\n    '
    out = runner.invoke(cli)
    assert out.exit_code == 2
    expected_error = 'Error: Invalid value: If you do not specify an input file, the default is one of: requirements.in, setup.py, pyproject.toml, setup.cfg'
    assert expected_error in out.stderr.splitlines()

@pytest.mark.network
@pytest.mark.parametrize(('fname', 'content'), METADATA_TEST_CASES)
def test_input_formats(fake_dists, runner, make_module, fname, content):
    if False:
        while True:
            i = 10
    '\n    Test different dependency formats as input file.\n    '
    meta_path = make_module(fname=fname, content=content)
    out = runner.invoke(cli, ['-n', '--no-build-isolation', '--find-links', fake_dists, meta_path])
    assert out.exit_code == 0, out.stderr
    assert 'small-fake-a==0.1' in out.stderr
    assert 'small-fake-b' not in out.stderr
    assert 'small-fake-c' not in out.stderr
    assert 'extra ==' not in out.stderr

@pytest.mark.parametrize('verbose_option', (True, False))
def test_error_in_pyproject_toml(fake_dists, runner, make_module, capfd, verbose_option):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that an error in pyproject.toml is reported.\n    '
    fname = 'pyproject.toml'
    invalid_content = dedent('        [project]\n        invalid = "metadata"\n        ')
    meta_path = make_module(fname=fname, content=invalid_content)
    options = []
    if verbose_option:
        options = ['--verbose']
    options.extend(['-n', '--no-build-isolation', '--find-links', fake_dists, meta_path])
    out = runner.invoke(cli, options)
    assert out.exit_code == 2, out.stderr
    captured = capfd.readouterr()
    assert ("`project` must contain ['name'] properties" in captured.err) is verbose_option

@pytest.mark.network
@pytest.mark.parametrize(('fname', 'content'), METADATA_TEST_CASES)
def test_one_extra(fake_dists, runner, make_module, fname, content):
    if False:
        print('Hello World!')
    '\n    Test one ``--extra`` (dev) passed, other extras (test) must be ignored.\n    '
    meta_path = make_module(fname=fname, content=content)
    out = runner.invoke(cli, ['-n', '--extra', 'dev', '--no-build-isolation', '--find-links', fake_dists, meta_path])
    assert out.exit_code == 0, out.stderr
    assert 'small-fake-a==0.1' in out.stderr
    assert 'small-fake-b==0.2' in out.stderr
    assert 'extra ==' not in out.stderr

@pytest.mark.network
@pytest.mark.parametrize('extra_opts', (pytest.param(('--extra', 'dev', '--extra', 'test'), id='singular'), pytest.param(('--extra', 'dev,test'), id='comma-separated')))
@pytest.mark.parametrize(('fname', 'content'), METADATA_TEST_CASES)
def test_multiple_extras(fake_dists, runner, make_module, fname, content, extra_opts):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test passing multiple ``--extra`` params.\n    '
    meta_path = make_module(fname=fname, content=content)
    out = runner.invoke(cli, ['-n', *extra_opts, '--no-build-isolation', '--find-links', fake_dists, meta_path])
    assert out.exit_code == 0, out.stderr
    assert 'small-fake-a==0.1' in out.stderr
    assert 'small-fake-b==0.2' in out.stderr
    assert 'extra ==' not in out.stderr

@pytest.mark.network
@pytest.mark.parametrize(('fname', 'content'), METADATA_TEST_CASES)
def test_all_extras(fake_dists, runner, make_module, fname, content):
    if False:
        while True:
            i = 10
    '\n    Test passing ``--all-extras`` includes all applicable extras.\n    '
    meta_path = make_module(fname=fname, content=content)
    out = runner.invoke(cli, ['--output-file', '-', '--quiet', '--all-extras', '--find-links', fake_dists, '--no-annotate', '--no-emit-options', '--no-header', '--no-build-isolation', meta_path])
    assert out.exit_code == 0, out
    assert dedent('            small-fake-a==0.1\n            small-fake-b==0.2\n            small-fake-c==0.3\n            ') == out.stdout

@pytest.mark.parametrize(('fname', 'content'), METADATA_TEST_CASES[:1])
def test_all_extras_fail_with_extra(fake_dists, runner, make_module, fname, content):
    if False:
        i = 10
        return i + 15
    '\n    Test that passing ``--all-extras`` and ``--extra`` fails.\n    '
    meta_path = make_module(fname=fname, content=content)
    out = runner.invoke(cli, ['-n', '--all-extras', '--extra', 'dev', '--find-links', fake_dists, '--no-annotate', '--no-emit-options', '--no-header', '--no-build-isolation', meta_path])
    assert out.exit_code == 2
    exp = '--extra has no effect when used with --all-extras'
    assert exp in out.stderr

def _mock_resolver_cls(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    if False:
        i = 10
        return i + 15
    obj = MagicMock()
    obj.resolve = MagicMock(return_value=set())
    obj.resolve_hashes = MagicMock(return_value=dict())
    cls = MagicMock(return_value=obj)
    monkeypatch.setattr('piptools.scripts.compile.BacktrackingResolver', cls)
    monkeypatch.setattr('piptools.scripts.compile.LegacyResolver', cls)
    return cls

def _mock_build_project_metadata(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    if False:
        i = 10
        return i + 15
    func = MagicMock(return_value=ProjectMetadata(extras=('e',), requirements=(install_req_from_line('rdep0'), install_req_from_line("rdep1; extra=='e'")), build_requirements=(install_req_from_line('bdep0'),)))
    monkeypatch.setattr('piptools.scripts.compile.build_project_metadata', func)
    return func

@backtracking_resolver_only
@pytest.mark.network
def test_all_extras_and_all_build_deps(fake_dists_with_build_deps, runner, tmp_path, monkeypatch, current_resolver):
    if False:
        print('Hello World!')
    '\n    Test that trying to lock all dependencies gives the expected output.\n    '
    src_pkg_path = pathlib.Path(PACKAGES_PATH) / 'small_fake_with_build_deps'
    monkeypatch.setenv('PIP_FIND_LINKS', fake_dists_with_build_deps)
    with runner.isolated_filesystem(tmp_path) as tmp_pkg_path:
        shutil.copytree(src_pkg_path, tmp_pkg_path, dirs_exist_ok=True)
        out = runner.invoke(cli, ['--allow-unsafe', '--output-file', '-', '--quiet', '--no-emit-options', '--no-header', '--all-extras', '--all-build-deps'])
    assert out.exit_code == 0
    assert 'fake_transient_build_dep' not in out.stdout
    assert out.stdout == dedent('        fake-direct-extra-runtime-dep==0.2\n            # via small-fake-with-build-deps (setup.py)\n        fake-direct-runtime-dep==0.1\n            # via small-fake-with-build-deps (setup.py)\n        fake-dynamic-build-dep-for-all==0.2\n            # via\n            #   small-fake-with-build-deps (pyproject.toml::build-system.backend::editable)\n            #   small-fake-with-build-deps (pyproject.toml::build-system.backend::sdist)\n            #   small-fake-with-build-deps (pyproject.toml::build-system.backend::wheel)\n        fake-dynamic-build-dep-for-editable==0.5\n            # via small-fake-with-build-deps (pyproject.toml::build-system.backend::editable)\n        fake-dynamic-build-dep-for-sdist==0.3\n            # via small-fake-with-build-deps (pyproject.toml::build-system.backend::sdist)\n        fake-dynamic-build-dep-for-wheel==0.4\n            # via small-fake-with-build-deps (pyproject.toml::build-system.backend::wheel)\n        fake-static-build-dep==0.1\n            # via small-fake-with-build-deps (pyproject.toml::build-system.requires)\n        fake-transient-run-dep==0.3\n            # via fake-static-build-dep\n        wheel==0.41.1\n            # via\n            #   small-fake-with-build-deps (pyproject.toml::build-system.backend::wheel)\n            #   small-fake-with-build-deps (pyproject.toml::build-system.requires)\n\n        # The following packages are considered to be unsafe in a requirements file:\n        setuptools==68.1.2\n            # via small-fake-with-build-deps (pyproject.toml::build-system.requires)\n        ')

@backtracking_resolver_only
def test_all_build_deps(runner, tmp_path, monkeypatch):
    if False:
        while True:
            i = 10
    '\n    Test that ``--all-build-deps`` is equivalent to specifying every\n    ``--build-deps-for``.\n    '
    func = _mock_build_project_metadata(monkeypatch)
    _mock_resolver_cls(monkeypatch)
    src_file = tmp_path / 'pyproject.toml'
    src_file.touch()
    out = runner.invoke(cli, ['--all-build-deps', os.fspath(src_file)])
    assert out.exit_code == 0
    assert func.call_args.kwargs['build_targets'] == ('editable', 'sdist', 'wheel')

@backtracking_resolver_only
def test_only_build_deps(runner, tmp_path, monkeypatch):
    if False:
        return 10
    '\n    Test that ``--only-build-deps`` excludes dependencies other than build dependencies.\n    '
    _mock_build_project_metadata(monkeypatch)
    cls = _mock_resolver_cls(monkeypatch)
    src_file = tmp_path / 'pyproject.toml'
    src_file.touch()
    out = runner.invoke(cli, ['--all-build-deps', '--only-build-deps', os.fspath(src_file)])
    assert out.exit_code == 0
    assert [c.name for c in cls.call_args.kwargs['constraints']] == ['bdep0']

@backtracking_resolver_only
def test_all_build_deps_fail_with_build_target(runner):
    if False:
        i = 10
        return i + 15
    '\n    Test that passing ``--all-build-deps`` and ``--build-deps-for`` fails.\n    '
    out = runner.invoke(cli, ['--all-build-deps', '--build-deps-for', 'sdist'])
    exp = '--build-deps-for has no effect when used with --all-build-deps'
    assert out.exit_code == 2
    assert exp in out.stderr

@backtracking_resolver_only
def test_only_build_deps_fails_without_any_build_deps(runner):
    if False:
        return 10
    '\n    Test that passing ``--only-build-deps`` fails when it is not specified how build deps should\n    be gathered.\n    '
    out = runner.invoke(cli, ['--only-build-deps'])
    exp = '--only-build-deps requires either --build-deps-for or --all-build-deps'
    assert out.exit_code == 2
    assert exp in out.stderr

@backtracking_resolver_only
@pytest.mark.parametrize('option', ('--all-extras', '--extra=foo'))
def test_only_build_deps_fails_with_conflicting_options(runner, option):
    if False:
        return 10
    '\n    Test that passing ``--all-build-deps`` and conflicting option fails.\n    '
    out = runner.invoke(cli, ['--all-build-deps', '--only-build-deps', option])
    exp = '--only-build-deps cannot be used with any of --extra, --all-extras'
    assert out.exit_code == 2
    assert exp in out.stderr

@backtracking_resolver_only
@pytest.mark.parametrize('option', ('--all-build-deps', '--build-deps-for=wheel'))
def test_build_deps_fail_without_setup_file(runner, tmpdir, option):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that passing ``--build-deps-for`` or ``--all-build-deps`` fails when used with a\n    requirements file as opposed to a setup file.\n    '
    path = pathlib.Path(tmpdir) / 'requirements.in'
    path.write_text('\n')
    out = runner.invoke(cli, ['-n', option, os.fspath(path)])
    exp = '--build-deps-for and --all-build-deps can be used only with the setup.py, setup.cfg and pyproject.toml specs.'
    assert out.exit_code == 2
    assert exp in out.stderr

def test_extras_fail_with_requirements_in(runner, tmpdir):
    if False:
        print('Hello World!')
    '\n    Test that passing ``--extra`` with ``requirements.in`` input file fails.\n    '
    path = pathlib.Path(tmpdir) / 'requirements.in'
    path.write_text('\n')
    out = runner.invoke(cli, ['-n', '--extra', 'something', os.fspath(path)])
    assert out.exit_code == 2
    exp = '--extra has effect only with setup.py and PEP-517 input formats'
    assert exp in out.stderr

def test_cli_compile_strip_extras(runner, make_package, make_sdist, tmpdir):
    if False:
        i = 10
        return i + 15
    '\n    Assures that ``--strip-extras`` removes mention of extras from output.\n    '
    test_package_1 = make_package('test_package_1', version='0.1', extras_require={'more': 'test_package_2'})
    test_package_2 = make_package('test_package_2', version='0.1')
    dists_dir = tmpdir / 'dists'
    for pkg in (test_package_1, test_package_2):
        make_sdist(pkg, dists_dir)
    with open('requirements.in', 'w') as reqs_out:
        reqs_out.write('test_package_1[more]')
    out = runner.invoke(cli, ['--strip-extras', '--find-links', str(dists_dir)])
    assert out.exit_code == 0, out
    assert 'test-package-2==0.1' in out.stderr
    assert '[more]' not in out.stderr

@pytest.mark.parametrize(('package_specs', 'constraints', 'existing_reqs', 'expected_reqs'), (([{'name': 'test_package_1', 'version': '1.1', 'install_requires': ['test_package_2 ~= 1.1']}, {'name': 'test_package_2', 'version': '1.1', 'extras_require': {'more': 'test_package_3'}}], '\n            test_package_1 == 1.1\n            ', '\n            test_package_1 == 1.0\n            test_package_2 == 1.0\n            ', '\n            test-package-1==1.1\n            test-package-2==1.1\n            '), ([{'name': 'test_package_1', 'version': '1.1', 'install_requires': ['test_package_2[more] ~= 1.1']}, {'name': 'test_package_2', 'version': '1.1', 'extras_require': {'more': 'test_package_3'}}, {'name': 'test_package_3', 'version': '0.1'}], '\n            test_package_1 == 1.1\n            ', '\n            test_package_1 == 1.0\n            test_package_2 == 1.0\n            test_package_3 == 0.1\n            ', '\n            test-package-1==1.1\n            test-package-2==1.1\n            test-package-3==0.1\n            '), ([{'name': 'test_package_1', 'version': '1.1', 'install_requires': ['test_package_2[more] ~= 1.1']}, {'name': 'test_package_2', 'version': '1.1', 'extras_require': {'more': 'test_package_3'}}, {'name': 'test_package_3', 'version': '0.1'}], '\n            test_package_1 == 1.1\n            ', '\n            test_package_1 == 1.0\n            test_package_2[more] == 1.0\n            test_package_3 == 0.1\n            ', '\n            test-package-1==1.1\n            test-package-2==1.1\n            test-package-3==0.1\n            ')), ids=('no-extra', 'extra-stripped-from-existing', 'with-extra-in-existing'))
def test_resolver_drops_existing_conflicting_constraint(runner, make_package, make_sdist, tmpdir, package_specs, constraints, existing_reqs, expected_reqs) -> None:
    if False:
        print('Hello World!')
    '\n    Test that the resolver will find a solution even if some of the existing\n    (indirect) requirements are incompatible with the new constraints.\n\n    This must succeed even if the conflicting requirement includes some extra,\n    no matter whether the extra is mentioned in the existing requirements\n    or not (cf. `issue #1977 <https://github.com/jazzband/pip-tools/issues/1977>`_).\n    '
    expected_requirements = {line.strip() for line in expected_reqs.splitlines()}
    dists_dir = tmpdir / 'dists'
    packages = [make_package(**spec) for spec in package_specs]
    for pkg in packages:
        make_sdist(pkg, dists_dir)
    with open('requirements.txt', 'w') as existing_reqs_out:
        existing_reqs_out.write(dedent(existing_reqs))
    with open('requirements.in', 'w') as constraints_out:
        constraints_out.write(dedent(constraints))
    out = runner.invoke(cli, ['--strip-extras', '--find-links', str(dists_dir)])
    assert out.exit_code == 0, out
    with open('requirements.txt') as req_txt:
        req_txt_content = req_txt.read()
        assert expected_requirements.issubset(req_txt_content.splitlines())

def test_resolution_failure(runner):
    if False:
        i = 10
        return i + 15
    'Test resolution impossible for unknown package.'
    with open('requirements.in', 'w') as reqs_out:
        reqs_out.write('unknown-package')
    out = runner.invoke(cli)
    assert out.exit_code != 0, out

def test_resolver_reaches_max_rounds(runner):
    if False:
        while True:
            i = 10
    'Test resolver reched max rounds and raises error.'
    with open('requirements.in', 'w') as reqs_out:
        reqs_out.write('six')
    out = runner.invoke(cli, ['--max-rounds', 0])
    assert out.exit_code != 0, out

def test_preserve_via_requirements_constrained_dependencies_when_run_twice(pip_conf, runner):
    if False:
        while True:
            i = 10
    '\n    Test that 2 consecutive runs of pip-compile (first with a non-existing requirements.txt file,\n    second with an existing file) produce the same output.\n    '
    with open('constraints.txt', 'w') as constraints_in:
        constraints_in.write('small-fake-a==0.1')
    with open('requirements.in', 'w') as req_in:
        req_in.write('-c constraints.txt\nsmall_fake_with_deps')
    cli_arguments = ['--no-emit-options', '--no-header']
    first_out = runner.invoke(cli, cli_arguments)
    assert first_out.exit_code == 0, first_out
    with open('requirements.txt') as req_txt:
        first_output = req_txt.read()
    second_out = runner.invoke(cli, cli_arguments)
    assert second_out.exit_code == 0, second_out
    with open('requirements.txt') as req_txt:
        second_output = req_txt.read()
    expected_output = dedent('        small-fake-a==0.1\n            # via\n            #   -c constraints.txt\n            #   small-fake-with-deps\n        small-fake-with-deps==0.1\n            # via -r requirements.in\n        ')
    assert first_output == expected_output
    assert second_output == expected_output

def test_failure_of_legacy_resolver_prompts_for_backtracking(pip_conf, runner, tmpdir, make_package, make_wheel, current_resolver):
    if False:
        return 10
    'Test that pip-compile prompts to use the backtracking resolver'
    pkgs = [make_package('a', version='0.1', install_requires=['b==0.1']), make_package('a', version='0.2', install_requires=['b==0.2']), make_package('b', version='0.1'), make_package('b', version='0.2'), make_package('c', version='1', install_requires=['b==0.1', 'a'])]
    dists_dir = tmpdir / 'dists'
    for pkg in pkgs:
        make_wheel(pkg, dists_dir)
    with open('requirements.in', 'w') as req_in:
        req_in.writelines(['c'])
    out = runner.invoke(cli, ['--resolver', current_resolver, '--find-links', str(dists_dir)])
    if current_resolver == 'legacy':
        assert out.exit_code == 2, out
        assert 'Consider using backtracking resolver with' in out.stderr
    elif current_resolver == 'backtracking':
        assert out.exit_code == 0, out
    else:
        raise AssertionError('unreachable')

def test_print_deprecation_warning_if_using_legacy_resolver(runner, current_resolver):
    if False:
        while True:
            i = 10
    with open('requirements.in', 'w'):
        pass
    out = runner.invoke(cli)
    assert out.exit_code == 0, out
    expected_warning = 'WARNING: the legacy dependency resolver is deprecated'
    if current_resolver == 'legacy':
        assert expected_warning in out.stderr
    else:
        assert expected_warning not in out.stderr

@pytest.mark.parametrize('input_filenames', (pytest.param(('requirements.txt',), id='one file'), pytest.param(('requirements.txt', 'dev-requirements.in'), id='multiple files')))
def test_raise_error_when_input_and_output_filenames_are_matched(runner, tmp_path, input_filenames):
    if False:
        while True:
            i = 10
    req_in_paths = []
    for input_filename in input_filenames:
        req_in = tmp_path / input_filename
        req_in.touch()
        req_in_paths.append(req_in.as_posix())
    req_out = tmp_path / 'requirements.txt'
    req_out_path = req_out.as_posix()
    out = runner.invoke(cli, req_in_paths + ['--output-file', req_out_path])
    assert out.exit_code == 2
    expected_error = f'Error: input and output filenames must not be matched: {req_out_path}'
    assert expected_error in out.stderr.splitlines()

@pytest.mark.network
@backtracking_resolver_only
def test_pass_pip_cache_to_pip_args(tmpdir, runner, current_resolver):
    if False:
        for i in range(10):
            print('nop')
    cache_dir = tmpdir.mkdir('cache_dir')
    with open('requirements.in', 'w') as fp:
        fp.write('six==1.15.0')
    out = runner.invoke(cli, ['--cache-dir', str(cache_dir), '--resolver', current_resolver])
    assert out.exit_code == 0
    pip_current_version = get_pip_version_for_python_executable(sys.executable)
    pip_breaking_version = Version('23.3.dev0')
    if pip_current_version >= pip_breaking_version:
        pip_http_cache_dir = 'http-v2'
    else:
        pip_http_cache_dir = 'http'
    assert os.listdir(os.path.join(str(cache_dir), pip_http_cache_dir))

@backtracking_resolver_only
def test_compile_recursive_extras(runner, tmp_path, current_resolver):
    if False:
        while True:
            i = 10
    (tmp_path / 'pyproject.toml').write_text(dedent('\n            [project]\n            name = "foo"\n            version = "0.0.1"\n            dependencies = ["small-fake-a"]\n            [project.optional-dependencies]\n            footest = ["small-fake-b"]\n            dev = ["foo[footest]"]\n            '))
    out = runner.invoke(cli, ['--no-build-isolation', '--no-header', '--no-annotate', '--no-emit-options', '--extra', 'dev', '--find-links', os.fspath(MINIMAL_WHEELS_PATH), os.fspath(tmp_path / 'pyproject.toml'), '--output-file', '-'])
    expected = f'foo[footest] @ {tmp_path.as_uri()}\nsmall-fake-a==0.2\nsmall-fake-b==0.3\n'
    assert out.exit_code == 0
    assert expected == out.stdout

def test_config_option(pip_conf, runner, tmp_path, make_config_file):
    if False:
        while True:
            i = 10
    config_file = make_config_file('dry-run', True)
    req_in = tmp_path / 'requirements.in'
    req_in.touch()
    out = runner.invoke(cli, [req_in.as_posix(), '--config', config_file.as_posix()])
    assert out.exit_code == 0
    assert 'Dry-run, so nothing updated' in out.stderr

def test_default_config_option(pip_conf, runner, make_config_file, tmpdir_cwd):
    if False:
        for i in range(10):
            print('nop')
    make_config_file('dry-run', True)
    req_in = tmpdir_cwd / 'requirements.in'
    req_in.touch()
    out = runner.invoke(cli)
    assert out.exit_code == 0
    assert 'Dry-run, so nothing updated' in out.stderr

def test_no_config_option_overrides_config_with_defaults(pip_conf, runner, tmp_path, make_config_file):
    if False:
        i = 10
        return i + 15
    config_file = make_config_file('dry-run', True)
    req_in = tmp_path / 'requirements.in'
    req_in.touch()
    out = runner.invoke(cli, [req_in.as_posix(), '--no-config', '--config', config_file.as_posix()])
    assert out.exit_code == 0
    assert 'Dry-run, so nothing updated' not in out.stderr

def test_raise_error_on_unknown_config_option(pip_conf, runner, tmp_path, make_config_file):
    if False:
        print('Hello World!')
    config_file = make_config_file('unknown-option', True)
    req_in = tmp_path / 'requirements.in'
    req_in.touch()
    out = runner.invoke(cli, [req_in.as_posix(), '--config', config_file.as_posix()])
    assert out.exit_code == 2
    assert "No such config key 'unknown_option'" in out.stderr

def test_raise_error_on_invalid_config_option(pip_conf, runner, tmp_path, make_config_file):
    if False:
        for i in range(10):
            print('nop')
    config_file = make_config_file('dry-run', ['invalid', 'value'])
    req_in = tmp_path / 'requirements.in'
    req_in.touch()
    out = runner.invoke(cli, [req_in.as_posix(), '--config', config_file.as_posix()])
    assert out.exit_code == 2
    assert "Invalid value for config key 'dry_run': ['invalid', 'value']" in out.stderr

@pytest.mark.parametrize('option', ('-c', '--constraint'))
def test_constraint_option(pip_conf, runner, tmpdir_cwd, make_config_file, option):
    if False:
        i = 10
        return i + 15
    req_in = tmpdir_cwd / 'requirements.in'
    req_in.write_text('small-fake-a')
    constraints_txt = tmpdir_cwd / 'constraints.txt'
    constraints_txt.write_text('small-fake-a==0.1')
    out = runner.invoke(cli, [req_in.name, option, constraints_txt.name, '--output-file', '-', '--no-header', '--no-emit-options'])
    assert out.exit_code == 0
    assert out.stdout == dedent('        small-fake-a==0.1\n            # via\n            #   -c constraints.txt\n            #   -r requirements.in\n        ')

def test_allow_in_config_pip_sync_option(pip_conf, runner, tmp_path, make_config_file):
    if False:
        for i in range(10):
            print('nop')
    config_file = make_config_file('--ask', True)
    req_in = tmp_path / 'requirements.in'
    req_in.touch()
    out = runner.invoke(cli, [req_in.as_posix(), '--verbose', '--config', config_file.as_posix()])
    assert out.exit_code == 0
    assert 'Using pip-tools configuration defaults found' in out.stderr

def test_cli_boolean_flag_config_option_has_valid_context(pip_conf, runner, tmp_path, make_config_file):
    if False:
        print('Hello World!')
    config_file = make_config_file('no-annotate', True)
    req_in = tmp_path / 'requirements.in'
    req_in.write_text('small-fake-a==0.1')
    out = runner.invoke(cli, [req_in.as_posix(), '--config', config_file.as_posix(), '--no-emit-options', '--no-header', '--output-file', '-'])
    assert out.exit_code == 0
    assert out.stdout == 'small-fake-a==0.1\n'

def test_invalid_cli_boolean_flag_config_option_captured(pip_conf, runner, tmp_path, make_config_file):
    if False:
        i = 10
        return i + 15
    config_file = make_config_file('no-annnotate', True)
    req_in = tmp_path / 'requirements.in'
    req_in.touch()
    out = runner.invoke(cli, [req_in.as_posix(), '--config', config_file.as_posix()])
    assert out.exit_code == 2
    assert "No such config key 'no_annnotate'." in out.stderr
strip_extras_warning = 'WARNING: --strip-extras is becoming the default in version 8.0.0.'

def test_show_warning_on_default_strip_extras_option(runner, make_package, make_sdist, tmp_path):
    if False:
        i = 10
        return i + 15
    req_in = tmp_path / 'requirements.in'
    req_in.touch()
    out = runner.invoke(cli, req_in.as_posix())
    assert out.exit_code == 0
    assert strip_extras_warning in out.stderr

@pytest.mark.parametrize('option', ('--strip-extras', '--no-strip-extras'))
def test_do_not_show_warning_on_explicit_strip_extras_option(runner, make_package, make_sdist, tmp_path, option):
    if False:
        print('Hello World!')
    req_in = tmp_path / 'requirements.in'
    req_in.touch()
    out = runner.invoke(cli, [option, req_in.as_posix()])
    assert out.exit_code == 0
    assert strip_extras_warning not in out.stderr

def test_origin_of_extra_requirement_not_written_to_annotations(pip_conf, runner, make_package, make_wheel, tmp_path, tmpdir):
    if False:
        while True:
            i = 10
    req_in = tmp_path / 'requirements.in'
    package_with_extras = make_package('package_with_extras', version='0.1', extras_require={'extra1': ['small-fake-a==0.1'], 'extra2': ['small-fake-b==0.1']})
    dists_dir = tmpdir / 'dists'
    make_wheel(package_with_extras, dists_dir)
    with open(req_in, 'w') as req_out:
        req_out.write('package-with-extras[extra1,extra2]')
    out = runner.invoke(cli, ['--output-file', '-', '--quiet', '--no-header', '--find-links', str(dists_dir), '--no-emit-options', '--no-build-isolation', req_in.as_posix()])
    assert out.exit_code == 0, out
    assert dedent(f'        package-with-extras[extra1,extra2]==0.1\n            # via -r {req_in.as_posix()}\n        small-fake-a==0.1\n            # via package-with-extras\n        small-fake-b==0.1\n            # via package-with-extras\n        ') == out.stdout