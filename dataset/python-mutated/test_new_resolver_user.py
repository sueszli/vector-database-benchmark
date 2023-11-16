import os
import textwrap
import pytest
from tests.lib import PipTestEnvironment, create_basic_wheel_for_package
from tests.lib.venv import VirtualEnvironment

@pytest.mark.usefixtures('enable_user_site')
def test_new_resolver_install_user(script: PipTestEnvironment) -> None:
    if False:
        i = 10
        return i + 15
    create_basic_wheel_for_package(script, 'base', '0.1.0')
    result = script.pip('install', '--no-cache-dir', '--no-index', '--find-links', script.scratch_path, '--user', 'base')
    result.did_create(script.user_site / 'base')

@pytest.mark.usefixtures('enable_user_site')
def test_new_resolver_install_user_satisfied_by_global_site(script: PipTestEnvironment) -> None:
    if False:
        return 10
    '\n    An install a matching version to user site should re-use a global site\n    installation if it satisfies.\n    '
    create_basic_wheel_for_package(script, 'base', '1.0.0')
    script.pip('install', '--no-cache-dir', '--no-index', '--find-links', script.scratch_path, 'base==1.0.0')
    result = script.pip('install', '--no-cache-dir', '--no-index', '--find-links', script.scratch_path, '--user', 'base==1.0.0')
    result.did_not_create(script.user_site / 'base')

@pytest.mark.usefixtures('enable_user_site')
def test_new_resolver_install_user_conflict_in_user_site(script: PipTestEnvironment) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Installing a different version in user site should uninstall an existing\n    different version in user site.\n    '
    create_basic_wheel_for_package(script, 'base', '1.0.0')
    create_basic_wheel_for_package(script, 'base', '2.0.0')
    script.pip('install', '--no-cache-dir', '--no-index', '--find-links', script.scratch_path, '--user', 'base==2.0.0')
    result = script.pip('install', '--no-cache-dir', '--no-index', '--find-links', script.scratch_path, '--user', 'base==1.0.0')
    base_1_dist_info = script.user_site / 'base-1.0.0.dist-info'
    base_2_dist_info = script.user_site / 'base-2.0.0.dist-info'
    result.did_create(base_1_dist_info)
    result.did_not_create(base_2_dist_info)

@pytest.fixture()
def patch_dist_in_site_packages(virtualenv: VirtualEnvironment) -> None:
    if False:
        print('Hello World!')
    virtualenv.sitecustomize = textwrap.dedent('\n        def dist_in_site_packages(dist):\n            return False\n\n        from pip._internal.metadata.base import BaseDistribution\n        BaseDistribution.in_site_packages = property(dist_in_site_packages)\n    ')

@pytest.mark.usefixtures('enable_user_site', 'patch_dist_in_site_packages')
def test_new_resolver_install_user_reinstall_global_site(script: PipTestEnvironment) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Specifying --force-reinstall makes a different version in user site,\n    ignoring the matching installation in global site.\n    '
    create_basic_wheel_for_package(script, 'base', '1.0.0')
    script.pip('install', '--no-cache-dir', '--no-index', '--find-links', script.scratch_path, 'base==1.0.0')
    result = script.pip('install', '--no-cache-dir', '--no-index', '--find-links', script.scratch_path, '--user', '--force-reinstall', 'base==1.0.0')
    result.did_create(script.user_site / 'base')
    site_packages_content = set(os.listdir(script.site_packages_path))
    assert 'base' in site_packages_content

@pytest.mark.usefixtures('enable_user_site', 'patch_dist_in_site_packages')
def test_new_resolver_install_user_conflict_in_global_site(script: PipTestEnvironment) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Installing a different version in user site should ignore an existing\n    different version in global site, and simply add to the user site.\n    '
    create_basic_wheel_for_package(script, 'base', '1.0.0')
    create_basic_wheel_for_package(script, 'base', '2.0.0')
    script.pip('install', '--no-cache-dir', '--no-index', '--find-links', script.scratch_path, 'base==1.0.0')
    result = script.pip('install', '--no-cache-dir', '--no-index', '--find-links', script.scratch_path, '--user', 'base==2.0.0')
    base_2_dist_info = script.user_site / 'base-2.0.0.dist-info'
    result.did_create(base_2_dist_info)
    site_packages_content = set(os.listdir(script.site_packages_path))
    assert 'base-1.0.0.dist-info' in site_packages_content

@pytest.mark.usefixtures('enable_user_site', 'patch_dist_in_site_packages')
def test_new_resolver_install_user_conflict_in_global_and_user_sites(script: PipTestEnvironment) -> None:
    if False:
        print('Hello World!')
    '\n    Installing a different version in user site should ignore an existing\n    different version in global site, but still upgrade the user site.\n    '
    create_basic_wheel_for_package(script, 'base', '1.0.0')
    create_basic_wheel_for_package(script, 'base', '2.0.0')
    script.pip('install', '--no-cache-dir', '--no-index', '--find-links', script.scratch_path, 'base==2.0.0')
    script.pip('install', '--no-cache-dir', '--no-index', '--find-links', script.scratch_path, '--user', '--force-reinstall', 'base==2.0.0')
    result = script.pip('install', '--no-cache-dir', '--no-index', '--find-links', script.scratch_path, '--user', 'base==1.0.0')
    base_1_dist_info = script.user_site / 'base-1.0.0.dist-info'
    base_2_dist_info = script.user_site / 'base-2.0.0.dist-info'
    result.did_create(base_1_dist_info)
    assert base_2_dist_info in result.files_deleted
    site_packages_content = set(os.listdir(script.site_packages_path))
    assert 'base-2.0.0.dist-info' in site_packages_content