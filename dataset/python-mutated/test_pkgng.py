import textwrap
import pytest
import salt.modules.pkgng as pkgng
from salt.utils.odict import OrderedDict
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {pkgng: {}}

@pytest.fixture
def pkgs():
    if False:
        print('Hello World!')
    return [{'openvpn': '2.4.8_2'}, {'openvpn': '2.4.8_2', 'gettext-runtime': '0.20.1', 'p5-Mojolicious': '8.40'}]

def test_latest_version(pkgs):
    if False:
        return 10
    '\n    Test basic usage of pkgng.latest_version\n    '
    pkgs_mock = MagicMock(side_effect=pkgs)
    search_cmd = MagicMock(return_value='bash-5.1.4')
    with patch('salt.modules.pkgng.list_pkgs', pkgs_mock):
        with patch.dict(pkgng.__salt__, {'cmd.run': search_cmd}):
            result = pkgng.latest_version('bash')
            search_cmd.assert_called_with(['pkg', 'search', '-eqS', 'name', '-U', 'bash'], output_loglevel='trace', python_shell=False)
            assert result == '5.1.4'

def test_latest_version_origin(pkgs):
    if False:
        while True:
            i = 10
    '\n    Test pkgng.latest_version with a specific package origin\n    '
    pkgs_mock = MagicMock(side_effect=pkgs)
    search_cmd = MagicMock(return_value='bash-5.1.4_2')
    with patch('salt.modules.pkgng.list_pkgs', pkgs_mock):
        with patch.dict(pkgng.__salt__, {'cmd.run': search_cmd}):
            result = pkgng.latest_version('shells/bash')
            search_cmd.assert_called_with(['pkg', 'search', '-eqS', 'origin', '-U', 'shells/bash'], output_loglevel='trace', python_shell=False)
            assert result == '5.1.4_2'

def test_latest_version_outofdatedate(pkgs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test pkgng.latest_version with an out-of-date package\n    '
    pkgs_mock = MagicMock(side_effect=pkgs)
    search_cmd = MagicMock(return_value='openvpn-2.4.8_3')
    with patch('salt.modules.pkgng.list_pkgs', pkgs_mock):
        with patch.dict(pkgng.__salt__, {'cmd.run': search_cmd}):
            result = pkgng.latest_version('openvpn')
            search_cmd.assert_called_with(['pkg', 'search', '-eqS', 'name', '-U', 'openvpn'], output_loglevel='trace', python_shell=False)
            assert result == '2.4.8_3'

def test_latest_version_unavailable(pkgs):
    if False:
        print('Hello World!')
    '\n    Test pkgng.latest_version when the requested package is not available\n    '
    pkgs_mock = MagicMock(side_effect=pkgs)
    search_cmd = MagicMock(return_value='')
    with patch('salt.modules.pkgng.list_pkgs', pkgs_mock):
        with patch.dict(pkgng.__salt__, {'cmd.run': search_cmd}):
            result = pkgng.latest_version('does_not_exist')
            search_cmd.assert_called_with(['pkg', 'search', '-eqS', 'name', '-U', 'does_not_exist'], output_loglevel='trace', python_shell=False)

def test_latest_version_uptodate(pkgs):
    if False:
        while True:
            i = 10
    '\n    Test pkgng.latest_version with an up-to-date package\n    '
    pkgs_mock = MagicMock(side_effect=pkgs)
    search_cmd = MagicMock(return_value='openvpn-2.4.8_2')
    with patch('salt.modules.pkgng.list_pkgs', pkgs_mock):
        with patch.dict(pkgng.__salt__, {'cmd.run': search_cmd}):
            result = pkgng.latest_version('openvpn')
            search_cmd.assert_called_with(['pkg', 'search', '-eqS', 'name', '-U', 'openvpn'], output_loglevel='trace', python_shell=False)
            assert result == ''

def test_lock():
    if False:
        print('Hello World!')
    '\n    Test pkgng.lock\n    '
    lock_cmd = MagicMock(return_value={'stdout': 'pkga-1.0\npkgb-2.0\n', 'retcode': 0})
    with patch.dict(pkgng.__salt__, {'cmd.run_all': lock_cmd}):
        result = pkgng.lock('pkga')
        assert result
        lock_cmd.assert_called_with(['pkg', 'lock', '-y', '--quiet', '--show-locked', 'pkga'], output_loglevel='trace', python_shell=False)
        result = pkgng.lock('dummy')
        assert not result
        lock_cmd.assert_called_with(['pkg', 'lock', '-y', '--quiet', '--show-locked', 'dummy'], output_loglevel='trace', python_shell=False)

def test_unlock():
    if False:
        while True:
            i = 10
    '\n    Test pkgng.unlock\n    '
    unlock_cmd = MagicMock(return_value={'stdout': 'pkga-1.0\npkgb-2.0\n', 'retcode': 0})
    with patch.dict(pkgng.__salt__, {'cmd.run_all': unlock_cmd}):
        result = pkgng.unlock('pkga')
        assert not result
        unlock_cmd.assert_called_with(['pkg', 'unlock', '-y', '--quiet', '--show-locked', 'pkga'], output_loglevel='trace', python_shell=False)
        result = pkgng.unlock('dummy')
        assert result
        unlock_cmd.assert_called_with(['pkg', 'unlock', '-y', '--quiet', '--show-locked', 'dummy'], output_loglevel='trace', python_shell=False)

def test_locked():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test pkgng.unlock\n    '
    lock_cmd = MagicMock(return_value={'stdout': 'pkga-1.0\npkgb-2.0\n', 'retcode': 0})
    with patch.dict(pkgng.__salt__, {'cmd.run_all': lock_cmd}):
        result = pkgng.locked('pkga')
        assert result
        lock_cmd.assert_called_with(['pkg', 'lock', '-y', '--quiet', '--show-locked'], output_loglevel='trace', python_shell=False)
        result = pkgng.locked('dummy')
        assert not result
        lock_cmd.assert_called_with(['pkg', 'lock', '-y', '--quiet', '--show-locked'], output_loglevel='trace', python_shell=False)

def test_list_upgrades_present():
    if False:
        while True:
            i = 10
    '\n    Test pkgng.list_upgrades with upgrades available\n    '
    pkg_cmd = MagicMock(return_value=textwrap.dedent('\n        The following 6 package(s) will be affected (of 0 checked):\n\n        Installed packages to be UPGRADED:\n                pkga: 1.0 -> 1.1\n                pkgb: 2.0 -> 2.1 [FreeBSD]\n                pkgc: 3.0 -> 3.1 [FreeBSD] (dependency changed)\n                pkgd: 4.0 -> 4.1 (dependency changed)\n\n        New packages to be INSTALLED:\n                pkge: 1.0\n                pkgf: 2.0 [FreeBSD]\n                pkgg: 3.0 [FreeBSD] (dependency changed)\n                pkgh: 4.0 (dependency changed)\n\n        Installed packages to be REINSTALLED:\n                pkgi-1.0\n                pkgj-2.0 [FreeBSD]\n                pkgk-3.0 [FreeBSD] (direct dependency changed: pkga)\n                pkgl-4.0 (direct dependency changed: pkgb)\n\n        Installed packages to be DOWNGRADED:\n                pkgm: 1.1 -> 1.0\n                pkgn: 2.1 -> 2.0 [FreeBSD]\n                pkgo: 3.1 -> 3.0 [FreeBSD] (dependency changed)\n                pkgp: 4.1 -> 4.0 (dependency changed)\n\n        Installed packages to be REMOVED:\n                pkgq-1.0\n                pkgr-2.0 [FreeBSD]\n                pkgs-3.0 [FreeBSD] (direct dependency changed: pkga)\n                pkgt-4.0 (direct dependency changed: pkgb)\n\n        Number of packages to be upgraded: 2\n        Number of packages to be reinstalled: 2\n\n        The process will require 14 MiB more space.\n        22 MiB to be downloaded.\n        '))
    with patch.dict(pkgng.__salt__, {'cmd.run_stdout': pkg_cmd}):
        result = pkgng.list_upgrades(refresh=False)
        assert result == {'pkga': '1.1', 'pkgb': '2.1', 'pkgc': '3.1', 'pkgd': '4.1'}
        pkg_cmd.assert_called_with(['pkg', 'upgrade', '--dry-run', '--quiet', '--no-repo-update'], output_loglevel='trace', python_shell=False, ignore_retcode=True)

def test_list_upgrades_absent():
    if False:
        print('Hello World!')
    '\n    Test pkgng.list_upgrades with no upgrades available\n    '
    pkg_cmd = MagicMock(return_value='')
    with patch.dict(pkgng.__salt__, {'cmd.run_stdout': pkg_cmd}):
        result = pkgng.list_upgrades(refresh=False)
        assert result == {}
        pkg_cmd.assert_called_with(['pkg', 'upgrade', '--dry-run', '--quiet', '--no-repo-update'], output_loglevel='trace', python_shell=False, ignore_retcode=True)

def test_upgrade_without_fromrepo(pkgs):
    if False:
        i = 10
        return i + 15
    '\n    Test pkg upgrade to upgrade all available packages\n    '
    pkg_cmd = MagicMock(return_value={'retcode': 0})
    pkgs_mock = MagicMock(side_effect=pkgs)
    with patch.dict(pkgng.__salt__, {'cmd.run_all': pkg_cmd}):
        with patch('salt.modules.pkgng.list_pkgs', pkgs_mock):
            result = pkgng.upgrade()
            expected = {'gettext-runtime': {'new': '0.20.1', 'old': ''}, 'p5-Mojolicious': {'new': '8.40', 'old': ''}}
            assert result == expected
            pkg_cmd.assert_called_with(['pkg', 'upgrade', '-y'], output_loglevel='trace', python_shell=False)

def test_upgrade_with_fromrepo(pkgs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test pkg upgrade to upgrade all available packages\n    '
    pkg_cmd = MagicMock(return_value={'retcode': 0})
    pkgs_mock = MagicMock(side_effect=pkgs)
    with patch.dict(pkgng.__salt__, {'cmd.run_all': pkg_cmd}):
        with patch('salt.modules.pkgng.list_pkgs', pkgs_mock):
            result = pkgng.upgrade(fromrepo='FreeBSD')
            expected = {'gettext-runtime': {'new': '0.20.1', 'old': ''}, 'p5-Mojolicious': {'new': '8.40', 'old': ''}}
            assert result == expected
            pkg_cmd.assert_called_with(['pkg', 'upgrade', '-y', '--repository', 'FreeBSD'], output_loglevel='trace', python_shell=False)

def test_upgrade_with_fetchonly(pkgs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test pkg upgrade to fetch packages only\n    '
    pkg_cmd = MagicMock(return_value={'retcode': 0})
    pkgs_mock = MagicMock(side_effect=pkgs)
    with patch.dict(pkgng.__salt__, {'cmd.run_all': pkg_cmd}):
        with patch('salt.modules.pkgng.list_pkgs', pkgs_mock):
            result = pkgng.upgrade(fetchonly=True)
            expected = {'gettext-runtime': {'new': '0.20.1', 'old': ''}, 'p5-Mojolicious': {'new': '8.40', 'old': ''}}
            assert result == expected
            pkg_cmd.assert_called_with(['pkg', 'upgrade', '-Fy'], output_loglevel='trace', python_shell=False)

def test_upgrade_with_local(pkgs):
    if False:
        print('Hello World!')
    '\n    Test pkg upgrade to supress automatic update of the local copy of the\n    repository catalogue from remote\n    '
    pkg_cmd = MagicMock(return_value={'retcode': 0})
    pkgs_mock = MagicMock(side_effect=pkgs)
    with patch.dict(pkgng.__salt__, {'cmd.run_all': pkg_cmd}):
        with patch('salt.modules.pkgng.list_pkgs', pkgs_mock):
            result = pkgng.upgrade(local=True)
            expected = {'gettext-runtime': {'new': '0.20.1', 'old': ''}, 'p5-Mojolicious': {'new': '8.40', 'old': ''}}
            assert result == expected
            pkg_cmd.assert_called_with(['pkg', 'upgrade', '-Uy'], output_loglevel='trace', python_shell=False)

def test_stats_with_local():
    if False:
        i = 10
        return i + 15
    '\n    Test pkg.stats for local packages\n    '
    pkg_cmd = MagicMock(return_value='')
    with patch.dict(pkgng.__salt__, {'cmd.run': pkg_cmd}):
        result = pkgng.stats(local=True)
        assert result == []
        pkg_cmd.assert_called_with(['pkg', 'stats', '-l'], output_loglevel='trace', python_shell=False)

def test_stats_with_remote():
    if False:
        print('Hello World!')
    '\n    Test pkg.stats for remote packages\n    '
    pkg_cmd = MagicMock(return_value='')
    with patch.dict(pkgng.__salt__, {'cmd.run': pkg_cmd}):
        result = pkgng.stats(remote=True)
        assert result == []
        pkg_cmd.assert_called_with(['pkg', 'stats', '-r'], output_loglevel='trace', python_shell=False)

def test_stats_with_bytes_remote():
    if False:
        i = 10
        return i + 15
    '\n    Test pkg.stats to show disk space usage in bytes only for remote\n    '
    pkg_cmd = MagicMock(return_value='')
    with patch.dict(pkgng.__salt__, {'cmd.run': pkg_cmd}):
        result = pkgng.stats(remote=True, bytes=True)
        assert result == []
        pkg_cmd.assert_called_with(['pkg', 'stats', '-rb'], output_loglevel='trace', python_shell=False)

def test_stats_with_bytes_local():
    if False:
        return 10
    '\n    Test pkg.stats to show disk space usage in bytes only for local\n    '
    pkg_cmd = MagicMock(return_value='')
    with patch.dict(pkgng.__salt__, {'cmd.run': pkg_cmd}):
        result = pkgng.stats(local=True, bytes=True)
        assert result == []
        pkg_cmd.assert_called_with(['pkg', 'stats', '-lb'], output_loglevel='trace', python_shell=False)

def test_install_without_args(pkgs):
    if False:
        i = 10
        return i + 15
    '\n    Test pkg.install to install a package without arguments\n    '
    parsed_targets = (OrderedDict((('gettext-runtime', None), ('p5-Mojolicious', None))), 'repository')
    pkg_cmd = MagicMock(return_value={'retcode': 0})
    pkgs_mock = MagicMock(side_effect=pkgs)
    patches = {'cmd.run_all': pkg_cmd, 'pkg_resource.parse_targets': MagicMock(return_value=parsed_targets)}
    with patch.dict(pkgng.__salt__, patches):
        with patch('salt.modules.pkgng.list_pkgs', pkgs_mock):
            added = pkgng.install()
            expected = {'gettext-runtime': {'new': '0.20.1', 'old': ''}, 'p5-Mojolicious': {'new': '8.40', 'old': ''}}
            assert added == expected
            pkg_cmd.assert_called_with(['pkg', 'install', '-y', 'gettext-runtime', 'p5-Mojolicious'], output_loglevel='trace', python_shell=False, env={})

def test_install_with_local(pkgs):
    if False:
        while True:
            i = 10
    '\n    Test pkg.install to install a package with local=True argument\n    '
    parsed_targets = (OrderedDict((('gettext-runtime', None), ('p5-Mojolicious', None))), 'repository')
    pkg_cmd = MagicMock(return_value={'retcode': 0})
    pkgs_mock = MagicMock(side_effect=pkgs)
    patches = {'cmd.run_all': pkg_cmd, 'pkg_resource.parse_targets': MagicMock(return_value=parsed_targets)}
    with patch.dict(pkgng.__salt__, patches):
        with patch('salt.modules.pkgng.list_pkgs', pkgs_mock):
            added = pkgng.install(local=True)
            expected = {'gettext-runtime': {'new': '0.20.1', 'old': ''}, 'p5-Mojolicious': {'new': '8.40', 'old': ''}}
            assert added == expected
            pkg_cmd.assert_called_with(['pkg', 'install', '-yU', 'gettext-runtime', 'p5-Mojolicious'], output_loglevel='trace', python_shell=False, env={})

def test_install_with_fromrepo(pkgs):
    if False:
        print('Hello World!')
    '\n    Test pkg.install to install a package with fromrepo=FreeBSD argument\n    '
    parsed_targets = (OrderedDict((('gettext-runtime', None), ('p5-Mojolicious', None))), 'repository')
    pkg_cmd = MagicMock(return_value={'retcode': 0})
    pkgs_mock = MagicMock(side_effect=pkgs)
    patches = {'cmd.run_all': pkg_cmd, 'pkg_resource.parse_targets': MagicMock(return_value=parsed_targets)}
    with patch.dict(pkgng.__salt__, patches):
        with patch('salt.modules.pkgng.list_pkgs', pkgs_mock):
            added = pkgng.install(fromrepo='FreeBSD')
            expected = {'gettext-runtime': {'new': '0.20.1', 'old': ''}, 'p5-Mojolicious': {'new': '8.40', 'old': ''}}
            assert added == expected
            pkg_cmd.assert_called_with(['pkg', 'install', '-r', 'FreeBSD', '-y', 'gettext-runtime', 'p5-Mojolicious'], output_loglevel='trace', python_shell=False, env={})

def test_install_with_glob(pkgs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test pkg.install to install a package with glob=True argument\n    '
    parsed_targets = (OrderedDict((('gettext-runtime', None), ('p5-Mojolicious', None))), 'repository')
    pkg_cmd = MagicMock(return_value={'retcode': 0})
    pkgs_mock = MagicMock(side_effect=pkgs)
    patches = {'cmd.run_all': pkg_cmd, 'pkg_resource.parse_targets': MagicMock(return_value=parsed_targets)}
    with patch.dict(pkgng.__salt__, patches):
        with patch('salt.modules.pkgng.list_pkgs', pkgs_mock):
            added = pkgng.install(glob=True)
            expected = {'gettext-runtime': {'new': '0.20.1', 'old': ''}, 'p5-Mojolicious': {'new': '8.40', 'old': ''}}
            assert added == expected
            pkg_cmd.assert_called_with(['pkg', 'install', '-yg', 'gettext-runtime', 'p5-Mojolicious'], output_loglevel='trace', python_shell=False, env={})

def test_install_with_reinstall_requires(pkgs):
    if False:
        return 10
    '\n    Test pkg.install to install a package with reinstall_requires=True argument\n    '
    parsed_targets = (OrderedDict((('gettext-runtime', None), ('p5-Mojolicious', None))), 'repository')
    pkg_cmd = MagicMock(return_value={'retcode': 0})
    pkgs_mock = MagicMock(side_effect=pkgs)
    patches = {'cmd.run_all': pkg_cmd, 'pkg_resource.parse_targets': MagicMock(return_value=parsed_targets)}
    with patch.dict(pkgng.__salt__, patches):
        with patch('salt.modules.pkgng.list_pkgs', pkgs_mock):
            added = pkgng.install(reinstall_requires=True, force=True)
            expected = {'gettext-runtime': {'new': '0.20.1', 'old': ''}, 'p5-Mojolicious': {'new': '8.40', 'old': ''}}
            assert added == expected
            pkg_cmd.assert_called_with(['pkg', 'install', '-yfR', 'gettext-runtime', 'p5-Mojolicious'], output_loglevel='trace', python_shell=False, env={})

def test_install_with_regex(pkgs):
    if False:
        i = 10
        return i + 15
    '\n    Test pkg.install to install a package with regex=True argument\n    '
    parsed_targets = (OrderedDict((('gettext-runtime', None), ('p5-Mojolicious', None))), 'repository')
    pkg_cmd = MagicMock(return_value={'retcode': 0})
    pkgs_mock = MagicMock(side_effect=pkgs)
    patches = {'cmd.run_all': pkg_cmd, 'pkg_resource.parse_targets': MagicMock(return_value=parsed_targets)}
    with patch.dict(pkgng.__salt__, patches):
        with patch('salt.modules.pkgng.list_pkgs', pkgs_mock):
            added = pkgng.install(regex=True)
            expected = {'gettext-runtime': {'new': '0.20.1', 'old': ''}, 'p5-Mojolicious': {'new': '8.40', 'old': ''}}
            assert added == expected
            pkg_cmd.assert_called_with(['pkg', 'install', '-yx', 'gettext-runtime', 'p5-Mojolicious'], output_loglevel='trace', python_shell=False, env={})

def test_install_with_batch(pkgs):
    if False:
        return 10
    '\n    Test pkg.install to install a package with batch=True argument\n    '
    parsed_targets = (OrderedDict((('gettext-runtime', None), ('p5-Mojolicious', None))), 'repository')
    pkg_cmd = MagicMock(return_value={'retcode': 0})
    pkgs_mock = MagicMock(side_effect=pkgs)
    patches = {'cmd.run_all': pkg_cmd, 'pkg_resource.parse_targets': MagicMock(return_value=parsed_targets)}
    with patch.dict(pkgng.__salt__, patches):
        with patch('salt.modules.pkgng.list_pkgs', pkgs_mock):
            added = pkgng.install(batch=True)
            expected = {'gettext-runtime': {'new': '0.20.1', 'old': ''}, 'p5-Mojolicious': {'new': '8.40', 'old': ''}}
            assert added == expected
            pkg_cmd.assert_called_with(['pkg', 'install', '-y', 'gettext-runtime', 'p5-Mojolicious'], output_loglevel='trace', python_shell=False, env={'BATCH': 'true', 'ASSUME_ALWAYS_YES': 'YES'})

def test_install_with_pcre(pkgs):
    if False:
        while True:
            i = 10
    '\n    Test pkg.install to install a package with pcre=True argument\n    '
    parsed_targets = (OrderedDict((('gettext-runtime', None), ('p5-Mojolicious', None))), 'repository')
    pkg_cmd = MagicMock(return_value={'retcode': 0})
    pkgs_mock = MagicMock(side_effect=pkgs)
    patches = {'cmd.run_all': pkg_cmd, 'pkg_resource.parse_targets': MagicMock(return_value=parsed_targets)}
    with patch.dict(pkgng.__salt__, patches):
        with patch('salt.modules.pkgng.list_pkgs', pkgs_mock):
            added = pkgng.install(pcre=True)
            expected = {'gettext-runtime': {'new': '0.20.1', 'old': ''}, 'p5-Mojolicious': {'new': '8.40', 'old': ''}}
            assert added == expected
            pkg_cmd.assert_called_with(['pkg', 'install', '-yX', 'gettext-runtime', 'p5-Mojolicious'], output_loglevel='trace', python_shell=False, env={})

def test_install_with_orphan(pkgs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test pkg.install to install a package with orphan=True argument\n    '
    parsed_targets = (OrderedDict((('gettext-runtime', None), ('p5-Mojolicious', None))), 'repository')
    pkg_cmd = MagicMock(return_value={'retcode': 0})
    pkgs_mock = MagicMock(side_effect=pkgs)
    patches = {'cmd.run_all': pkg_cmd, 'pkg_resource.parse_targets': MagicMock(return_value=parsed_targets)}
    with patch.dict(pkgng.__salt__, patches):
        with patch('salt.modules.pkgng.list_pkgs', pkgs_mock):
            added = pkgng.install(orphan=True)
            expected = {'gettext-runtime': {'new': '0.20.1', 'old': ''}, 'p5-Mojolicious': {'new': '8.40', 'old': ''}}
            assert added == expected
            pkg_cmd.assert_called_with(['pkg', 'install', '-yA', 'gettext-runtime', 'p5-Mojolicious'], output_loglevel='trace', python_shell=False, env={})

def test_check_depends():
    if False:
        i = 10
        return i + 15
    '\n    Test pkgng.check to check and install missing dependencies\n    '
    pkg_cmd = MagicMock(return_value='')
    with patch.dict(pkgng.__salt__, {'cmd.run': pkg_cmd}):
        result = pkgng.check(depends=True)
        assert result == ''
        pkg_cmd.assert_called_with(['pkg', 'check', '-dy'], output_loglevel='trace', python_shell=False)

def test_check_checksum():
    if False:
        while True:
            i = 10
    '\n    Test pkgng.check for packages with invalid checksums\n    '
    pkg_cmd = MagicMock(return_value='')
    with patch.dict(pkgng.__salt__, {'cmd.run': pkg_cmd}):
        result = pkgng.check(checksum=True)
        assert result == ''
        pkg_cmd.assert_called_with(['pkg', 'check', '-s'], output_loglevel='trace', python_shell=False)

def test_check_recompute():
    if False:
        while True:
            i = 10
    '\n    Test pkgng.check to recalculate the checksums of installed packages\n    '
    pkg_cmd = MagicMock(return_value='')
    with patch.dict(pkgng.__salt__, {'cmd.run': pkg_cmd}):
        result = pkgng.check(recompute=True)
        assert result == ''
        pkg_cmd.assert_called_with(['pkg', 'check', '-r'], output_loglevel='trace', python_shell=False)

def test_check_checklibs():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test pkgng.check to regenerate the library dependency metadata\n    '
    pkg_cmd = MagicMock(return_value='')
    with patch.dict(pkgng.__salt__, {'cmd.run': pkg_cmd}):
        result = pkgng.check(checklibs=True)
        assert result == ''
        pkg_cmd.assert_called_with(['pkg', 'check', '-B'], output_loglevel='trace', python_shell=False)

def test_autoremove_with_dryrun():
    if False:
        i = 10
        return i + 15
    '\n    Test pkgng.autoremove with dryrun argument\n    '
    pkg_cmd = MagicMock(return_value='')
    with patch.dict(pkgng.__salt__, {'cmd.run': pkg_cmd}):
        result = pkgng.autoremove(dryrun=True)
        assert result == ''
        pkg_cmd.assert_called_with(['pkg', 'autoremove', '-n'], output_loglevel='trace', python_shell=False)

def test_autoremove():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test pkgng.autoremove\n    '
    pkg_cmd = MagicMock(return_value='')
    with patch.dict(pkgng.__salt__, {'cmd.run': pkg_cmd}):
        result = pkgng.autoremove()
        assert result == ''
        pkg_cmd.assert_called_with(['pkg', 'autoremove', '-y'], output_loglevel='trace', python_shell=False)

def test_audit():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test pkgng.audit\n    '
    pkg_cmd = MagicMock(return_value='')
    with patch.dict(pkgng.__salt__, {'cmd.run': pkg_cmd}):
        result = pkgng.audit()
        assert result == ''
        pkg_cmd.assert_called_with(['pkg', 'audit', '-F'], output_loglevel='trace', python_shell=False)

def test_version():
    if False:
        i = 10
        return i + 15
    '\n    Test pkgng.version\n    '
    version = '2.0.6'
    mock = MagicMock(return_value=version)
    with patch.dict(pkgng.__salt__, {'pkg_resource.version': mock}):
        assert pkgng.version(*['mutt']) == version
        assert not pkgng.version(*['mutt']) == '2.0.10'

def test_refresh_db_without_forced_flag():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test pkgng.refresh_db with force=False\n    '
    pkg_cmd = MagicMock(return_value=0)
    with patch('salt.utils.pkg.clear_rtag', MagicMock()):
        with patch.dict(pkgng.__salt__, {'cmd.retcode': pkg_cmd}):
            result = pkgng.refresh_db()
            assert result is True
            pkg_cmd.assert_called_with(['pkg', 'update'], python_shell=False)

def test_refresh_db_with_forced_flag():
    if False:
        while True:
            i = 10
    '\n    Test pkgng.refresh_db with force=True\n    '
    pkg_cmd = MagicMock(return_value=0)
    with patch('salt.utils.pkg.clear_rtag', MagicMock()):
        with patch.dict(pkgng.__salt__, {'cmd.retcode': pkg_cmd}):
            result = pkgng.refresh_db(force=True)
            assert result is True
            pkg_cmd.assert_called_with(['pkg', 'update', '-f'], python_shell=False)

def test_fetch_with_default_flag():
    if False:
        print('Hello World!')
    '\n    Test pkgng.fetch with default options\n    '
    targets = 'mutt'
    pkg_cmd = MagicMock(return_value=targets)
    patches = {'cmd.run': pkg_cmd, 'pkg_resource.stringify': MagicMock(), 'pkg_resource.parse_targets': MagicMock(return_value=targets)}
    with patch.dict(pkgng.__salt__, patches):
        pkgs = pkgng.fetch(targets)
        assert pkgs == targets
    pkg_cmd.assert_called_once_with(['pkg', 'fetch', '-y', '-g', 'mutt'], output_loglevel='trace', python_shell=False)

def test_fetch_with_dependency_flag():
    if False:
        i = 10
        return i + 15
    '\n    Test pkgng.fetch with enabled dependency flag\n    '
    targets = 'mutt'
    pkg_cmd = MagicMock(return_value=targets)
    patches = {'cmd.run': pkg_cmd, 'pkg_resource.stringify': MagicMock(), 'pkg_resource.parse_targets': MagicMock(return_value=targets)}
    with patch.dict(pkgng.__salt__, patches):
        pkgs = pkgng.fetch(targets, depends=True)
        assert pkgs == targets
    pkg_cmd.assert_called_once_with(['pkg', 'fetch', '-y', '-gd', 'mutt'], output_loglevel='trace', python_shell=False)

def test_fetch_with_regex_flag():
    if False:
        return 10
    '\n    Test pkgng.fetch with enabled regex flag\n    '
    targets = 'mutt'
    pkg_cmd = MagicMock(return_value=targets)
    patches = {'cmd.run': pkg_cmd, 'pkg_resource.stringify': MagicMock(), 'pkg_resource.parse_targets': MagicMock(return_value=targets)}
    with patch.dict(pkgng.__salt__, patches):
        pkgs = pkgng.fetch(targets, regex=True)
        assert pkgs == targets
    pkg_cmd.assert_called_once_with(['pkg', 'fetch', '-y', '-gx', 'mutt'], output_loglevel='trace', python_shell=False)

def test_fetch_with_fromrepo_flag():
    if False:
        return 10
    '\n    Test pkgng.fetch with enabled fromrepo flag\n    '
    targets = 'mutt'
    pkg_cmd = MagicMock(return_value=targets)
    patches = {'cmd.run': pkg_cmd, 'pkg_resource.stringify': MagicMock(), 'pkg_resource.parse_targets': MagicMock(return_value=targets)}
    with patch.dict(pkgng.__salt__, patches):
        pkgs = pkgng.fetch(targets, fromrepo='FreeBSD-poudriere')
        assert pkgs == targets
    pkg_cmd.assert_called_once_with(['pkg', 'fetch', '-y', '-r', 'FreeBSD-poudriere', '-g', 'mutt'], output_loglevel='trace', python_shell=False)

def test_fetch_with_localcache_flag():
    if False:
        i = 10
        return i + 15
    '\n    Test pkgng.fetch with enabled localcache flag\n    '
    targets = 'mutt'
    pkg_cmd = MagicMock(return_value=targets)
    patches = {'cmd.run': pkg_cmd, 'pkg_resource.stringify': MagicMock(), 'pkg_resource.parse_targets': MagicMock(return_value=targets)}
    with patch.dict(pkgng.__salt__, patches):
        pkgs = pkgng.fetch(targets, local=True)
        assert pkgs == targets
    pkg_cmd.assert_called_once_with(['pkg', 'fetch', '-y', '-gL', 'mutt'], output_loglevel='trace', python_shell=False)

def test_which_with_default_flags():
    if False:
        print('Hello World!')
    '\n    Test pkgng.which\n    '
    which_cmd = MagicMock(return_value={'stdout': '/usr/local/bin/mutt was installed by package mutt-2.0.6', 'retcode': 0})
    with patch.dict(pkgng.__salt__, {'cmd.run': which_cmd}):
        result = pkgng.which('/usr/local/bin/mutt')
        assert result
        which_cmd.assert_called_with(['pkg', 'which', '/usr/local/bin/mutt'], output_loglevel='trace', python_shell=False)

def test_which_with_origin_flag():
    if False:
        return 10
    '\n    Test pkgng.which with enabled origin flag\n    '
    which_cmd = MagicMock(return_value={'stdout': '/usr/local/bin/mutt was installed by package mail/mutt', 'retcode': 0})
    with patch.dict(pkgng.__salt__, {'cmd.run': which_cmd}):
        result = pkgng.which('/usr/local/bin/mutt', origin=True)
        assert result
        which_cmd.assert_called_with(['pkg', 'which', '-o', '/usr/local/bin/mutt'], output_loglevel='trace', python_shell=False)