"""
tests specific to uninstalling --user installs
"""
from os.path import isdir, isfile, normcase
import pytest
from tests.functional.test_install_user import _patch_dist_in_site_packages
from tests.lib import PipTestEnvironment, TestData, assert_all_changes
from tests.lib.venv import VirtualEnvironment
from tests.lib.wheel import make_wheel

@pytest.mark.usefixtures('enable_user_site')
class Tests_UninstallUserSite:

    @pytest.mark.network
    def test_uninstall_from_usersite(self, script: PipTestEnvironment) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test uninstall from usersite\n        '
        result1 = script.pip('install', '--user', 'INITools==0.3')
        result2 = script.pip('uninstall', '-y', 'INITools')
        assert_all_changes(result1, result2, [script.venv / 'build', 'cache'])

    def test_uninstall_from_usersite_with_dist_in_global_site(self, virtualenv: VirtualEnvironment, script: PipTestEnvironment) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test uninstall from usersite (with same dist in global site)\n        '
        entry_points_txt = '[console_scripts]\nscript = pkg:func'
        make_wheel('pkg', '0.1', extra_metadata_files={'entry_points.txt': entry_points_txt}).save_to_dir(script.scratch_path)
        make_wheel('pkg', '0.1.1', extra_metadata_files={'entry_points.txt': entry_points_txt}).save_to_dir(script.scratch_path)
        _patch_dist_in_site_packages(virtualenv)
        script.pip('install', '--no-index', '--find-links', script.scratch_path, '--no-warn-script-location', 'pkg==0.1')
        result2 = script.pip('install', '--no-index', '--find-links', script.scratch_path, '--no-warn-script-location', '--user', 'pkg==0.1.1')
        result3 = script.pip('uninstall', '-vy', 'pkg')
        assert normcase(script.user_bin_path) in result3.stdout, str(result3)
        assert normcase(script.bin_path) not in result3.stdout, str(result3)
        assert_all_changes(result2, result3, [script.venv / 'build', 'cache'])
        dist_info_folder = script.base_path / script.site_packages / 'pkg-0.1.dist-info'
        assert isdir(dist_info_folder)

    def test_uninstall_editable_from_usersite(self, script: PipTestEnvironment, data: TestData) -> None:
        if False:
            return 10
        '\n        Test uninstall editable local user install\n        '
        assert script.user_site_path.exists()
        to_install = data.packages.joinpath('FSPkg')
        result1 = script.pip('install', '--user', '-e', to_install)
        egg_link = script.user_site / 'FSPkg.egg-link'
        result1.did_create(egg_link)
        result2 = script.pip('uninstall', '-y', 'FSPkg')
        assert not isfile(script.base_path / egg_link)
        assert_all_changes(result1, result2, [script.venv / 'build', 'cache', script.user_site / 'easy-install.pth'])