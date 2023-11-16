"""
tests specific to "pip install --user"
"""
import os
import textwrap
from os.path import curdir, isdir, isfile
from pathlib import Path
import pytest
from tests.lib import PipTestEnvironment, TestData, create_basic_wheel_for_package, need_svn, pyversion
from tests.lib.local_repos import local_checkout
from tests.lib.venv import VirtualEnvironment

def _patch_dist_in_site_packages(virtualenv: VirtualEnvironment) -> None:
    if False:
        print('Hello World!')
    virtualenv.sitecustomize = textwrap.dedent('\n        def dist_in_site_packages(dist):\n            return False\n\n        from pip._internal.metadata.base import BaseDistribution\n        BaseDistribution.in_site_packages = property(dist_in_site_packages)\n    ')

@pytest.mark.usefixtures('enable_user_site')
class Tests_UserSite:

    @pytest.mark.network
    def test_reset_env_system_site_packages_usersite(self, script: PipTestEnvironment) -> None:
        if False:
            return 10
        '\n        Check user site works as expected.\n        '
        script.pip('install', '--user', 'INITools==0.2')
        result = script.run('python', '-c', "import pkg_resources; print(pkg_resources.get_distribution('initools').project_name)")
        project_name = result.stdout.strip()
        assert 'INITools' == project_name, project_name

    @pytest.mark.xfail
    @pytest.mark.network
    @need_svn
    def test_install_subversion_usersite_editable_with_distribute(self, script: PipTestEnvironment, tmpdir: Path) -> None:
        if False:
            print('Hello World!')
        "\n        Test installing current directory ('.') into usersite after installing\n        distribute\n        "
        result = script.pip('install', '--user', '-e', '{checkout}#egg=initools'.format(checkout=local_checkout('svn+http://svn.colorstudy.com/INITools', tmpdir)))
        result.assert_installed('INITools', use_user_site=True)

    def test_install_from_current_directory_into_usersite(self, script: PipTestEnvironment, data: TestData) -> None:
        if False:
            return 10
        "\n        Test installing current directory ('.') into usersite\n        "
        run_from = data.packages.joinpath('FSPkg')
        result = script.pip('install', '-vvv', '--user', curdir, cwd=run_from)
        fspkg_folder = script.user_site / 'fspkg'
        result.did_create(fspkg_folder)
        dist_info_folder = script.user_site / 'FSPkg-0.1.dev0.dist-info'
        result.did_create(dist_info_folder)

    def test_install_user_venv_nositepkgs_fails(self, virtualenv: VirtualEnvironment, script: PipTestEnvironment, data: TestData) -> None:
        if False:
            while True:
                i = 10
        '\n        user install in virtualenv (with no system packages) fails with message\n        '
        virtualenv.user_site_packages = False
        run_from = data.packages.joinpath('FSPkg')
        result = script.pip('install', '--user', curdir, cwd=run_from, expect_error=True)
        assert "Can not perform a '--user' install. User site-packages are not visible in this virtualenv." in result.stderr

    @pytest.mark.network
    def test_install_user_conflict_in_usersite(self, script: PipTestEnvironment) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test user install with conflict in usersite updates usersite.\n        '
        script.pip('install', '--user', 'INITools==0.3', '--no-binary=:all:')
        result2 = script.pip('install', '--user', 'INITools==0.1', '--no-binary=:all:')
        dist_info_folder = script.user_site / 'INITools-0.1.dist-info'
        initools_v3_file = script.base_path / script.user_site / 'initools' / 'configparser.py'
        result2.did_create(dist_info_folder)
        assert not isfile(initools_v3_file), initools_v3_file

    def test_install_user_conflict_in_globalsite(self, virtualenv: VirtualEnvironment, script: PipTestEnvironment) -> None:
        if False:
            return 10
        '\n        Test user install with conflict in global site ignores site and\n        installs to usersite\n        '
        create_basic_wheel_for_package(script, 'initools', '0.1')
        create_basic_wheel_for_package(script, 'initools', '0.2')
        _patch_dist_in_site_packages(virtualenv)
        script.pip('install', '--no-index', '--find-links', script.scratch_path, 'initools==0.2')
        result2 = script.pip('install', '--no-index', '--find-links', script.scratch_path, '--user', 'initools==0.1')
        dist_info_folder = script.user_site / 'initools-0.1.dist-info'
        initools_folder = script.user_site / 'initools'
        result2.did_create(dist_info_folder)
        result2.did_create(initools_folder)
        dist_info_folder = script.base_path / script.site_packages / 'initools-0.2.dist-info'
        initools_folder = script.base_path / script.site_packages / 'initools'
        assert isdir(dist_info_folder)
        assert isdir(initools_folder)

    def test_upgrade_user_conflict_in_globalsite(self, virtualenv: VirtualEnvironment, script: PipTestEnvironment) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test user install/upgrade with conflict in global site ignores site and\n        installs to usersite\n        '
        create_basic_wheel_for_package(script, 'initools', '0.2')
        create_basic_wheel_for_package(script, 'initools', '0.3.1')
        _patch_dist_in_site_packages(virtualenv)
        script.pip('install', '--no-index', '--find-links', script.scratch_path, 'initools==0.2')
        result2 = script.pip('install', '--no-index', '--find-links', script.scratch_path, '--user', '--upgrade', 'initools')
        dist_info_folder = script.user_site / 'initools-0.3.1.dist-info'
        initools_folder = script.user_site / 'initools'
        result2.did_create(dist_info_folder)
        result2.did_create(initools_folder)
        dist_info_folder = script.base_path / script.site_packages / 'initools-0.2.dist-info'
        initools_folder = script.base_path / script.site_packages / 'initools'
        assert isdir(dist_info_folder), result2.stdout
        assert isdir(initools_folder)

    def test_install_user_conflict_in_globalsite_and_usersite(self, virtualenv: VirtualEnvironment, script: PipTestEnvironment) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test user install with conflict in globalsite and usersite ignores\n        global site and updates usersite.\n        '
        initools_v3_file_name = os.path.join('initools', 'configparser.py')
        create_basic_wheel_for_package(script, 'initools', '0.1')
        create_basic_wheel_for_package(script, 'initools', '0.2')
        create_basic_wheel_for_package(script, 'initools', '0.3', extra_files={initools_v3_file_name: '# Hi!'})
        _patch_dist_in_site_packages(virtualenv)
        script.pip('install', '--no-index', '--find-links', script.scratch_path, 'initools==0.2')
        script.pip('install', '--no-index', '--find-links', script.scratch_path, '--user', 'initools==0.3')
        result3 = script.pip('install', '--no-index', '--find-links', script.scratch_path, '--user', 'initools==0.1')
        dist_info_folder = script.user_site / 'initools-0.1.dist-info'
        result3.did_create(dist_info_folder)
        initools_v3_file = script.base_path / script.user_site / initools_v3_file_name
        assert not isfile(initools_v3_file), initools_v3_file
        dist_info_folder = script.base_path / script.site_packages / 'initools-0.2.dist-info'
        initools_folder = script.base_path / script.site_packages / 'initools'
        assert isdir(dist_info_folder)
        assert isdir(initools_folder)

    def test_install_user_in_global_virtualenv_with_conflict_fails(self, script: PipTestEnvironment) -> None:
        if False:
            return 10
        '\n        Test user install in --system-site-packages virtualenv with conflict in\n        site fails.\n        '
        create_basic_wheel_for_package(script, 'pkg', '0.1')
        create_basic_wheel_for_package(script, 'pkg', '0.2')
        script.pip('install', '--no-cache-dir', '--no-index', '--find-links', script.scratch_path, 'pkg==0.2')
        result2 = script.pip('install', '--no-cache-dir', '--no-index', '--find-links', script.scratch_path, '--user', 'pkg==0.1', expect_error=True)
        resultp = script.run('python', '-c', "from pip._internal.metadata import get_default_environment; print(get_default_environment().get_distribution('pkg').location)")
        dist_location = resultp.stdout.strip()
        assert f'Will not install to the user site because it will lack sys.path precedence to pkg in {dist_location}' in result2.stderr