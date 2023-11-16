import itertools
import os
import sys
import textwrap
from pathlib import Path
import pytest
from tests.lib import PipTestEnvironment, ResolverVariant, TestData, assert_all_changes, pyversion
from tests.lib.local_repos import local_checkout
from tests.lib.wheel import make_wheel

@pytest.mark.network
def test_no_upgrade_unless_requested(script: PipTestEnvironment) -> None:
    if False:
        print('Hello World!')
    '\n    No upgrade if not specifically requested.\n\n    '
    script.pip('install', 'INITools==0.1')
    result = script.pip('install', 'INITools')
    assert not result.files_created, 'pip install INITools upgraded when it should not have'

def test_invalid_upgrade_strategy_causes_error(script: PipTestEnvironment) -> None:
    if False:
        while True:
            i = 10
    '\n    It errors out when the upgrade-strategy is an invalid/unrecognised one\n\n    '
    result = script.pip_install_local('--upgrade', '--upgrade-strategy=bazinga', 'simple', expect_error=True)
    assert result.returncode
    assert 'invalid choice' in result.stderr

def test_only_if_needed_does_not_upgrade_deps_when_satisfied(script: PipTestEnvironment, resolver_variant: ResolverVariant) -> None:
    if False:
        print('Hello World!')
    "\n    It doesn't upgrade a dependency if it already satisfies the requirements.\n\n    "
    script.pip_install_local('simple==2.0')
    result = script.pip_install_local('--upgrade', '--upgrade-strategy=only-if-needed', 'require_simple')
    assert script.site_packages / 'require_simple-1.0.dist-info' not in result.files_deleted, 'should have installed require_simple==1.0'
    assert script.site_packages / 'simple-2.0.dist-info' not in result.files_deleted, 'should not have uninstalled simple==2.0'
    msg = 'Requirement already satisfied'
    if resolver_variant == 'legacy':
        msg = msg + ', skipping upgrade: simple'
    assert msg in result.stdout, 'did not print correct message for not-upgraded requirement'

def test_only_if_needed_does_upgrade_deps_when_no_longer_satisfied(script: PipTestEnvironment) -> None:
    if False:
        while True:
            i = 10
    '\n    It does upgrade a dependency if it no longer satisfies the requirements.\n\n    '
    script.pip_install_local('simple==1.0')
    result = script.pip_install_local('--upgrade', '--upgrade-strategy=only-if-needed', 'require_simple')
    assert script.site_packages / 'require_simple-1.0.dist-info' not in result.files_deleted, 'should have installed require_simple==1.0'
    expected = script.site_packages / 'simple-3.0.dist-info'
    result.did_create(expected, message='should have installed simple==3.0')
    expected = script.site_packages / 'simple-1.0.dist-info'
    assert expected in result.files_deleted, 'should have uninstalled simple==1.0'

def test_eager_does_upgrade_dependencies_when_currently_satisfied(script: PipTestEnvironment) -> None:
    if False:
        print('Hello World!')
    '\n    It does upgrade a dependency even if it already satisfies the requirements.\n\n    '
    script.pip_install_local('simple==2.0')
    result = script.pip_install_local('--upgrade', '--upgrade-strategy=eager', 'require_simple')
    assert script.site_packages / 'require_simple-1.0.dist-info' not in result.files_deleted, 'should have installed require_simple==1.0'
    assert script.site_packages / 'simple-2.0.dist-info' in result.files_deleted, 'should have uninstalled simple==2.0'

def test_eager_does_upgrade_dependencies_when_no_longer_satisfied(script: PipTestEnvironment) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    It does upgrade a dependency if it no longer satisfies the requirements.\n\n    '
    script.pip_install_local('simple==1.0')
    result = script.pip_install_local('--upgrade', '--upgrade-strategy=eager', 'require_simple')
    assert script.site_packages / 'require_simple-1.0.dist-info' not in result.files_deleted, 'should have installed require_simple==1.0'
    result.did_create(script.site_packages / 'simple-3.0.dist-info', message='should have installed simple==3.0')
    assert script.site_packages / 'simple-1.0.dist-info' in result.files_deleted, 'should have uninstalled simple==1.0'

@pytest.mark.network
def test_upgrade_to_specific_version(script: PipTestEnvironment) -> None:
    if False:
        while True:
            i = 10
    '\n    It does upgrade to specific version requested.\n\n    '
    script.pip('install', 'INITools==0.1')
    result = script.pip('install', 'INITools==0.2')
    assert result.files_created, 'pip install with specific version did not upgrade'
    assert script.site_packages / 'INITools-0.1.dist-info' in result.files_deleted
    result.did_create(script.site_packages / 'INITools-0.2.dist-info')

@pytest.mark.network
def test_upgrade_if_requested(script: PipTestEnvironment) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    And it does upgrade if requested.\n\n    '
    script.pip('install', 'INITools==0.1')
    result = script.pip('install', '--upgrade', 'INITools')
    assert result.files_created, 'pip install --upgrade did not upgrade'
    result.did_not_create(script.site_packages / 'INITools-0.1.dist-info')

def test_upgrade_with_newest_already_installed(script: PipTestEnvironment, data: TestData, resolver_variant: ResolverVariant) -> None:
    if False:
        return 10
    '\n    If the newest version of a package is already installed, the package should\n    not be reinstalled and the user should be informed.\n    '
    script.pip('install', '-f', data.find_links, '--no-index', 'simple')
    result = script.pip('install', '--upgrade', '-f', data.find_links, '--no-index', 'simple')
    assert not result.files_created, 'simple upgraded when it should not have'
    if resolver_variant == 'resolvelib':
        msg = 'Requirement already satisfied'
    else:
        msg = 'already up-to-date'
    assert msg in result.stdout, result.stdout

@pytest.mark.network
def test_upgrade_force_reinstall_newest(script: PipTestEnvironment) -> None:
    if False:
        while True:
            i = 10
    '\n    Force reinstallation of a package even if it is already at its newest\n    version if --force-reinstall is supplied.\n    '
    result = script.pip('install', 'INITools')
    result.did_create(script.site_packages / 'initools')
    result2 = script.pip('install', '--upgrade', '--force-reinstall', 'INITools')
    assert result2.files_updated, 'upgrade to INITools 0.3 failed'
    result3 = script.pip('uninstall', 'initools', '-y')
    assert_all_changes(result, result3, [script.venv / 'build', 'cache'])

@pytest.mark.network
def test_uninstall_before_upgrade(script: PipTestEnvironment) -> None:
    if False:
        print('Hello World!')
    '\n    Automatic uninstall-before-upgrade.\n\n    '
    result = script.pip('install', 'INITools==0.2')
    result.did_create(script.site_packages / 'initools')
    result2 = script.pip('install', 'INITools==0.3')
    assert result2.files_created, 'upgrade to INITools 0.3 failed'
    result3 = script.pip('uninstall', 'initools', '-y')
    assert_all_changes(result, result3, [script.venv / 'build', 'cache'])

@pytest.mark.network
def test_uninstall_before_upgrade_from_url(script: PipTestEnvironment) -> None:
    if False:
        while True:
            i = 10
    '\n    Automatic uninstall-before-upgrade from URL.\n\n    '
    result = script.pip('install', 'INITools==0.2')
    result.did_create(script.site_packages / 'initools')
    result2 = script.pip('install', 'https://files.pythonhosted.org/packages/source/I/INITools/INITools-0.3.tar.gz')
    assert result2.files_created, 'upgrade to INITools 0.3 failed'
    result3 = script.pip('uninstall', 'initools', '-y')
    assert_all_changes(result, result3, [script.venv / 'build', 'cache'])

@pytest.mark.network
def test_upgrade_to_same_version_from_url(script: PipTestEnvironment) -> None:
    if False:
        return 10
    '\n    When installing from a URL the same version that is already installed, no\n    need to uninstall and reinstall if --upgrade is not specified.\n\n    '
    result = script.pip('install', 'INITools==0.3')
    result.did_create(script.site_packages / 'initools')
    result2 = script.pip('install', 'https://files.pythonhosted.org/packages/source/I/INITools/INITools-0.3.tar.gz')
    assert script.site_packages / 'initools' not in result2.files_updated, 'INITools 0.3 reinstalled same version'
    result3 = script.pip('uninstall', 'initools', '-y')
    assert_all_changes(result, result3, [script.venv / 'build', 'cache'])

@pytest.mark.network
def test_upgrade_from_reqs_file(script: PipTestEnvironment) -> None:
    if False:
        return 10
    '\n    Upgrade from a requirements file.\n\n    '
    script.scratch_path.joinpath('test-req.txt').write_text(textwrap.dedent('        PyLogo<0.4\n        # and something else to test out:\n        INITools==0.3\n        '))
    install_result = script.pip('install', '-r', script.scratch_path / 'test-req.txt')
    script.scratch_path.joinpath('test-req.txt').write_text(textwrap.dedent('        PyLogo\n        # and something else to test out:\n        INITools\n        '))
    script.pip('install', '--upgrade', '-r', script.scratch_path / 'test-req.txt')
    uninstall_result = script.pip('uninstall', '-r', script.scratch_path / 'test-req.txt', '-y')
    assert_all_changes(install_result, uninstall_result, [script.venv / 'build', 'cache', script.scratch / 'test-req.txt'])

def test_uninstall_rollback(script: PipTestEnvironment, data: TestData) -> None:
    if False:
        return 10
    '\n    Test uninstall-rollback (using test package with a setup.py\n    crafted to fail on install).\n\n    '
    result = script.pip('install', '-f', data.find_links, '--no-index', 'broken==0.1')
    result.did_create(script.site_packages / 'broken.py')
    result2 = script.pip('install', '-f', data.find_links, '--no-index', 'broken===0.2broken', expect_error=True)
    assert result2.returncode == 1, str(result2)
    assert script.run('python', '-c', 'import broken; print(broken.VERSION)').stdout == '0.1\n'
    assert_all_changes(result.files_after, result2, [script.venv / 'build'])

@pytest.mark.network
def test_should_not_install_always_from_cache(script: PipTestEnvironment) -> None:
    if False:
        i = 10
        return i + 15
    '\n    If there is an old cached package, pip should download the newer version\n    Related to issue #175\n    '
    script.pip('install', 'INITools==0.2')
    script.pip('uninstall', '-y', 'INITools')
    result = script.pip('install', 'INITools==0.1')
    result.did_not_create(script.site_packages / 'INITools-0.2.dist-info')
    result.did_create(script.site_packages / 'INITools-0.1.dist-info')

@pytest.mark.network
def test_install_with_ignoreinstalled_requested(script: PipTestEnvironment) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Test old conflicting package is completely ignored\n    '
    script.pip('install', 'INITools==0.1')
    result = script.pip('install', '-I', 'INITools==0.3')
    assert result.files_created, 'pip install -I did not install'
    assert os.path.exists(script.site_packages_path / 'INITools-0.1.dist-info')
    assert os.path.exists(script.site_packages_path / 'INITools-0.3.dist-info')

@pytest.mark.network
def test_upgrade_vcs_req_with_no_dists_found(script: PipTestEnvironment, tmpdir: Path) -> None:
    if False:
        while True:
            i = 10
    'It can upgrade a VCS requirement that has no distributions otherwise.'
    req = '{checkout}#egg=pip-test-package'.format(checkout=local_checkout('git+https://github.com/pypa/pip-test-package.git', tmpdir))
    script.pip('install', req)
    result = script.pip('install', '-U', req)
    assert not result.returncode

@pytest.mark.network
def test_upgrade_vcs_req_with_dist_found(script: PipTestEnvironment) -> None:
    if False:
        for i in range(10):
            print('nop')
    'It can upgrade a VCS requirement that has distributions on the index.'
    req = '{url}#egg=pretend'.format(url='git+https://github.com/alex/pretend@e7f26ad7dbcb4a02a4995aade4743aad47656b27')
    script.pip('install', req, expect_stderr=True)
    result = script.pip('install', '-U', req, expect_stderr=True)
    assert 'pypi.org' not in result.stdout, result.stdout

@pytest.mark.parametrize('req1, req2', list(itertools.product(['foo.bar', 'foo_bar', 'foo-bar'], ['foo.bar', 'foo_bar', 'foo-bar'])))
def test_install_find_existing_package_canonicalize(script: PipTestEnvironment, req1: str, req2: str) -> None:
    if False:
        i = 10
        return i + 15
    'Ensure an already-installed dist is found no matter how the dist name\n    was normalized on installation. (pypa/pip#8645)\n    '
    req_container = script.scratch_path.joinpath('foo-bar')
    req_container.mkdir()
    req_path = make_wheel('foo_bar', '1.0').save_to_dir(req_container)
    script.pip('install', '--no-index', req_path)
    pkg_container = script.scratch_path.joinpath('pkg')
    pkg_container.mkdir()
    make_wheel('pkg', '1.0', metadata_updates={'Requires-Dist': req2}).save_to_dir(pkg_container)
    result = script.pip('install', '--no-index', '--find-links', pkg_container, 'pkg')
    satisfied_message = f'Requirement already satisfied: {req2}'
    assert satisfied_message in result.stdout, str(result)

@pytest.mark.network
@pytest.mark.skipif(sys.platform != 'win32', reason='Windows-only test')
def test_modifying_pip_presents_error(script: PipTestEnvironment) -> None:
    if False:
        print('Hello World!')
    result = script.pip('install', 'pip', '--force-reinstall', use_module=False, expect_error=True)
    assert 'python.exe' in result.stderr or 'python.EXE' in result.stderr, str(result)
    assert ' -m ' in result.stderr, str(result)