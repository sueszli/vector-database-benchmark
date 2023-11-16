import configparser
import logging
import os
import re
import shutil
import tempfile
import time
import pytest
from saltfactories.utils.functional import Loaders
import salt.utils.path
import salt.utils.pkg
import salt.utils.platform
log = logging.getLogger(__name__)

@pytest.fixture
def ctx():
    if False:
        while True:
            i = 10
    return {}

@pytest.fixture
def preserve_rhel_yum_conf():
    if False:
        while True:
            i = 10
    cfg_file = '/etc/yum.conf'
    if not os.path.exists(cfg_file):
        pytest.skip('Only runs on RedHat.')
    tmp_dir = str(tempfile.gettempdir())
    tmp_file = os.path.join(tmp_dir, 'yum.conf')
    shutil.copy2(cfg_file, tmp_file)
    yield
    shutil.copy2(tmp_file, cfg_file)
    os.remove(tmp_file)

@pytest.fixture
def refresh_db(ctx, grains, modules):
    if False:
        while True:
            i = 10
    if 'refresh' not in ctx:
        modules.pkg.refresh_db()
        ctx['refresh'] = True
    if grains['os_family'] == 'Arch':
        for _ in range(12):
            if not os.path.isfile('/var/lib/pacman/db.lck'):
                break
            else:
                time.sleep(5)
        else:
            raise Exception('Package database locked after 60 seconds, bailing out')

@pytest.fixture(autouse=True)
def test_pkg(grains):
    if False:
        for i in range(10):
            print('nop')
    _pkg = 'figlet'
    if salt.utils.platform.is_windows():
        _pkg = 'putty'
    elif grains['os_family'] == 'RedHat':
        if grains['os'] == 'VMware Photon OS':
            _pkg = 'snoopy'
        else:
            _pkg = 'units'
    elif grains['os_family'] == 'Debian':
        _pkg = 'ifenslave'
    return _pkg

@pytest.mark.requires_salt_modules('pkg.list_pkgs')
@pytest.mark.slow_test
def test_list(modules, refresh_db):
    if False:
        for i in range(10):
            print('nop')
    '\n    verify that packages are installed\n    '
    ret = modules.pkg.list_pkgs()
    assert len(ret.keys()) != 0

@pytest.mark.requires_salt_modules('pkg.version_cmp')
@pytest.mark.slow_test
def test_version_cmp(grains, modules):
    if False:
        print('Hello World!')
    '\n    test package version comparison on supported platforms\n    '
    if grains['os_family'] == 'Debian':
        lt = ['0.2.4-0ubuntu1', '0.2.4.1-0ubuntu1']
        eq = ['0.2.4-0ubuntu1', '0.2.4-0ubuntu1']
        gt = ['0.2.4.1-0ubuntu1', '0.2.4-0ubuntu1']
    elif grains['os_family'] == 'Suse':
        lt = ['2.3.0-1', '2.3.1-15.1']
        eq = ['2.3.1-15.1', '2.3.1-15.1']
        gt = ['2.3.2-15.1', '2.3.1-15.1']
    else:
        lt = ['2.3.0', '2.3.1']
        eq = ['2.3.1', '2.3.1']
        gt = ['2.3.2', '2.3.1']
    assert modules.pkg.version_cmp(*lt) == -1
    assert modules.pkg.version_cmp(*eq) == 0
    assert modules.pkg.version_cmp(*gt) == 1

@pytest.mark.destructive_test
@pytest.mark.requires_salt_modules('pkg.mod_repo', 'pkg.del_repo', 'pkg.get_repo')
@pytest.mark.slow_test
@pytest.mark.requires_network
def test_mod_del_repo(grains, modules, refresh_db):
    if False:
        print('Hello World!')
    '\n    test modifying and deleting a software repository\n    '
    repo = None
    try:
        if grains['os'] == 'Ubuntu' and grains['osmajorrelease'] != 22:
            repo = 'ppa:otto-kesselgulasch/gimp-edge'
            uri = 'http://ppa.launchpad.net/otto-kesselgulasch/gimp-edge/ubuntu'
            ret = modules.pkg.mod_repo(repo, 'comps=main')
            assert ret != []
            ret = modules.pkg.get_repo(repo)
            assert isinstance(ret, dict) is True
            assert ret['uri'] == uri
        elif grains['os_family'] == 'RedHat':
            repo = 'saltstack'
            name = 'SaltStack repo for RHEL/CentOS {}'.format(grains['osmajorrelease'])
            baseurl = 'https://repo.saltproject.io/py3/redhat/{}/x86_64/latest/'.format(grains['osmajorrelease'])
            gpgkey = 'https://repo.saltproject.io/py3/redhat/{}/x86_64/latest/SALTSTACK-GPG-KEY.pub'.format(grains['osmajorrelease'])
            gpgcheck = 1
            enabled = 1
            ret = modules.pkg.mod_repo(repo, name=name, baseurl=baseurl, gpgkey=gpgkey, gpgcheck=gpgcheck, enabled=enabled)
            assert ret != {}
            repo_info = ret[next(iter(ret))]
            assert repo in repo_info
            assert repo_info[repo]['baseurl'] == baseurl
            ret = modules.pkg.get_repo(repo)
            assert ret['baseurl'] == baseurl
    finally:
        if repo is not None:
            modules.pkg.del_repo(repo)

@pytest.mark.slow_test
def test_mod_del_repo_multiline_values(modules, refresh_db):
    if False:
        for i in range(10):
            print('nop')
    '\n    test modifying and deleting a software repository defined with multiline values\n    '
    os_grain = modules.grains.item('os')['os']
    repo = None
    try:
        if os_grain in ['CentOS', 'RedHat', 'VMware Photon OS']:
            my_baseurl = 'http://my.fake.repo/foo/bar/\n http://my.fake.repo.alt/foo/bar/'
            expected_get_repo_baseurl = 'http://my.fake.repo/foo/bar/\nhttp://my.fake.repo.alt/foo/bar/'
            major_release = int(modules.grains.item('osmajorrelease')['osmajorrelease'])
            repo = 'fakerepo'
            name = 'Fake repo for RHEL/CentOS/SUSE'
            baseurl = my_baseurl
            gpgkey = 'https://my.fake.repo/foo/bar/MY-GPG-KEY.pub'
            failovermethod = 'priority'
            gpgcheck = 1
            enabled = 1
            ret = modules.pkg.mod_repo(repo, name=name, baseurl=baseurl, gpgkey=gpgkey, gpgcheck=gpgcheck, enabled=enabled, failovermethod=failovermethod)
            assert ret != {}
            repo_info = ret[next(iter(ret))]
            assert repo in repo_info
            assert repo_info[repo]['baseurl'] == my_baseurl
            ret = modules.pkg.get_repo(repo)
            assert ret['baseurl'] == expected_get_repo_baseurl
            modules.pkg.mod_repo(repo)
            ret = modules.pkg.get_repo(repo)
            assert ret['baseurl'] == expected_get_repo_baseurl
    finally:
        if repo is not None:
            modules.pkg.del_repo(repo)

@pytest.mark.requires_salt_modules('pkg.owner')
def test_owner(modules):
    if False:
        while True:
            i = 10
    '\n    test finding the package owning a file\n    '
    ret = modules.pkg.owner('/bin/ls')
    assert len(ret) != 0

@pytest.mark.skip_on_freebsd(reason='test for new package manager for FreeBSD')
@pytest.mark.requires_salt_modules('pkg.which')
def test_which(modules):
    if False:
        i = 10
        return i + 15
    '\n    test finding the package owning a file\n    '
    func = 'pkg.which'
    ret = modules.pkg.which('/usr/local/bin/salt-call')
    assert len(ret) != 0

@pytest.mark.destructive_test
@pytest.mark.requires_salt_modules('pkg.version', 'pkg.install', 'pkg.remove')
@pytest.mark.slow_test
@pytest.mark.requires_network
def test_install_remove(modules, test_pkg, refresh_db):
    if False:
        while True:
            i = 10
    '\n    successfully install and uninstall a package\n    '
    version = modules.pkg.version(test_pkg)

    def test_install():
        if False:
            for i in range(10):
                print('nop')
        install_ret = modules.pkg.install(test_pkg)
        assert test_pkg in install_ret

    def test_remove():
        if False:
            print('Hello World!')
        remove_ret = modules.pkg.remove(test_pkg)
        assert test_pkg in remove_ret
    if version and isinstance(version, dict):
        version = version[test_pkg]
    if version:
        test_remove()
        test_install()
    else:
        test_install()
        test_remove()

@pytest.mark.destructive_test
@pytest.mark.skip_on_photonos(reason='package hold/unhold unsupported on Photon OS')
@pytest.mark.requires_salt_modules('pkg.hold', 'pkg.unhold', 'pkg.install', 'pkg.version', 'pkg.remove', 'pkg.list_pkgs')
@pytest.mark.slow_test
@pytest.mark.requires_network
@pytest.mark.requires_salt_states('pkg.installed')
def test_hold_unhold(grains, modules, states, test_pkg, refresh_db):
    if False:
        for i in range(10):
            print('nop')
    '\n    test holding and unholding a package\n    '
    versionlock_pkg = None
    if grains['os_family'] == 'RedHat':
        pkgs = {p for p in modules.pkg.list_repo_pkgs() if '-versionlock' in p}
        if not pkgs:
            pytest.skip('No versionlock package found in repositories')
        for versionlock_pkg in pkgs:
            ret = states.pkg.installed(name=versionlock_pkg, refresh=False)
            try:
                assert ret.result is True
                break
            except AssertionError:
                pass
        else:
            pytest.fail(f'Could not install versionlock package from {pkgs}')
    modules.pkg.install(test_pkg)
    try:
        hold_ret = modules.pkg.hold(test_pkg)
        if versionlock_pkg and '-versionlock is not installed' in str(hold_ret):
            pytest.skip(f'{hold_ret}  `{versionlock_pkg}` is installed')
        assert test_pkg in hold_ret
        assert hold_ret[test_pkg]['result'] is True
        unhold_ret = modules.pkg.unhold(test_pkg)
        assert test_pkg in unhold_ret
        assert unhold_ret[test_pkg]['result'] is True
        modules.pkg.remove(test_pkg)
    except salt.exceptions.SaltInvocationError as err:
        if 'versionlock is not installed' in err.message:
            pytest.skip('Correct versionlock package is not installed')
    finally:
        if versionlock_pkg:
            ret = states.pkg.removed(name=versionlock_pkg)
            assert ret.result is True

@pytest.mark.destructive_test
@pytest.mark.requires_salt_modules('pkg.refresh_db')
@pytest.mark.slow_test
@pytest.mark.requires_network
def test_refresh_db(grains, tmp_path, minion_opts, refresh_db):
    if False:
        print('Hello World!')
    '\n    test refreshing the package database\n    '
    rtag = salt.utils.pkg.rtag(minion_opts)
    salt.utils.pkg.write_rtag(minion_opts)
    assert os.path.isfile(rtag) is True
    loader = Loaders(minion_opts)
    ret = loader.modules.pkg.refresh_db()
    if not isinstance(ret, dict):
        pytest.skip(f'Upstream repo did not return coherent results: {ret}')
    if grains['os_family'] == 'RedHat':
        assert ret in (True, None)
    elif grains['os_family'] == 'Suse':
        if not isinstance(ret, dict):
            pytest.skip('Upstream repo did not return coherent results. Skipping test.')
        assert ret != {}
        for (source, state) in ret.items():
            assert state in (True, False, None)
    assert os.path.isfile(rtag) is False

@pytest.mark.requires_salt_modules('pkg.info_installed')
@pytest.mark.slow_test
def test_pkg_info(grains, modules, test_pkg, refresh_db):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test returning useful information on Ubuntu systems.\n    '
    if grains['os_family'] == 'Debian':
        ret = modules.pkg.info_installed('bash', 'dpkg')
        keys = ret.keys()
        assert 'bash' in keys
        assert 'dpkg' in keys
    elif grains['os_family'] == 'RedHat':
        ret = modules.pkg.info_installed('rpm', 'bash')
        keys = ret.keys()
        assert 'rpm' in keys
        assert 'bash' in keys
    elif grains['os_family'] == 'Suse':
        ret = modules.pkg.info_installed('less', 'zypper')
        keys = ret.keys()
        assert 'less' in keys
        assert 'zypper' in keys
    else:
        ret = modules.pkg.info_installed(test_pkg)
        keys = ret.keys()
        assert test_pkg in keys

@pytest.mark.skipif(True, reason='Temporary Skip - Causes centos 8 test to fail')
@pytest.mark.destructive_test
@pytest.mark.requires_salt_modules('pkg.refresh_db', 'pkg.upgrade', 'pkg.install', 'pkg.list_repo_pkgs', 'pkg.list_upgrades')
@pytest.mark.slow_test
@pytest.mark.requires_network
def test_pkg_upgrade_has_pending_upgrades(grains, modules, test_pkg, refresh_db):
    if False:
        while True:
            i = 10
    '\n    Test running a system upgrade when there are packages that need upgrading\n    '
    if grains['os'] == 'Arch':
        pytest.skipTest("Arch moved to Python 3.8 and we're not ready for it yet")
    modules.pkg.upgrade()
    modules.pkg.refresh_db()
    if grains['os_family'] == 'Suse':
        packages = ('hwinfo', 'avrdude', 'diffoscope', 'vim')
        available = modules.pkg.list_repo_pkgs(packages)
        for package in packages:
            try:
                (new, old) = available[package][:2]
            except (KeyError, ValueError):
                continue
            else:
                target = package
                break
        else:
            pytest.fail('No suitable package found for this test')
        ret = modules.pkg.install(target, version=old)
        if not isinstance(ret, dict):
            if ret.startswith('ERROR'):
                pytest.skipTest(f'Could not install older {target} to complete test.')
        ret = modules.pkg.upgrade()
        if 'changes' in ret:
            assert target in ret['changes']
        else:
            assert target in ret
    else:
        ret = modules.pkg.list_upgrades()
        if ret == '' or ret == {}:
            pytest.skipTest('No updates available for this machine.  Skipping pkg.upgrade test.')
        else:
            args = []
            if grains['os_family'] == 'Debian':
                args = ['dist_upgrade=True']
            ret = modules.pkg.upgrade(args)
            assert ret != {}

@pytest.mark.destructive_test
@pytest.mark.skip_on_darwin(reason='The jenkins user is equivalent to root on mac, causing the test to be unrunnable')
@pytest.mark.requires_salt_modules('pkg.remove', 'pkg.latest_version')
@pytest.mark.slow_test
@pytest.mark.requires_salt_states('pkg.removed')
def test_pkg_latest_version(grains, modules, states, test_pkg, refresh_db):
    if False:
        return 10
    '\n    Check that pkg.latest_version returns the latest version of the uninstalled package.\n    The package is not installed. Only the package version is checked.\n    '
    states.pkg.removed(test_pkg)
    cmd_pkg = []
    if grains['os_family'] == 'RedHat':
        cmd_pkg = modules.cmd.run(f'yum list {test_pkg}')
    elif salt.utils.platform.is_windows():
        cmd_pkg = modules.pkg.list_available(test_pkg)
    elif grains['os_family'] == 'Debian':
        cmd_pkg = modules.cmd.run(f'apt list {test_pkg}')
    elif grains['os_family'] == 'Arch':
        cmd_pkg = modules.cmd.run(f'pacman -Si {test_pkg}')
    elif grains['os_family'] == 'FreeBSD':
        cmd_pkg = modules.cmd.run(f'pkg search -S name -qQ version -e {test_pkg}')
    elif grains['os_family'] == 'Suse':
        cmd_pkg = modules.cmd.run(f'zypper info {test_pkg}')
    elif grains['os_family'] == 'MacOS':
        brew_bin = salt.utils.path.which('brew')
        mac_user = modules.file.get_user(brew_bin)
        if mac_user == 'root':
            pytest.skip('brew cannot run as root, try a user in {}'.format(os.listdir('/Users/')))
        cmd_pkg = modules.cmd.run(f'brew info {test_pkg}', run_as=mac_user)
    else:
        pytest.skip('TODO: test not configured for {}'.format(grains['os_family']))
    pkg_latest = modules.pkg.latest_version(test_pkg)
    assert pkg_latest in cmd_pkg

@pytest.mark.destructive_test
@pytest.mark.requires_salt_modules('pkg.list_repos')
@pytest.mark.slow_test
def test_list_repos_duplicate_entries(preserve_rhel_yum_conf, grains, modules):
    if False:
        i = 10
        return i + 15
    '\n    test duplicate entries in /etc/yum.conf\n\n    This is a destructive test as it installs and then removes a package\n    '
    if grains['os_family'] != 'RedHat':
        pytest.skip('Only runs on RedHat.')
    if grains['os'] == 'Amazon':
        pytest.skip('Only runs on RedHat, Amazon /etc/yum.conf differs.')
    cfg_file = '/etc/yum.conf'
    with salt.utils.files.fpopen(cfg_file, 'w', mode=420) as fp_:
        fp_.write('[main]\n')
        fp_.write('gpgcheck=1\n')
        fp_.write('installonly_limit=3\n')
        fp_.write('clean_requirements_on_remove=True\n')
        fp_.write('best=True\n')
        fp_.write('skip_if_unavailable=False\n')
        fp_.write('http_caching=True\n')
        fp_.write('http_caching=True\n')
    ret = modules.pkg.list_repos(strict_config=False)
    assert ret != []
    assert isinstance(ret, dict) is True
    expected = "While reading from '/etc/yum.conf' [line  8]: option 'http_caching' in section 'main' already exists"
    with pytest.raises(configparser.DuplicateOptionError) as exc_info:
        result = modules.pkg.list_repos(strict_config=True)
    assert f'{exc_info.value}' == expected
    with pytest.raises(configparser.DuplicateOptionError) as exc_info:
        result = modules.pkg.list_repos()
    assert f'{exc_info.value}' == expected

@pytest.mark.destructive_test
@pytest.mark.slow_test
def test_pkg_install_port(grains, modules):
    if False:
        for i in range(10):
            print('nop')
    '\n    test install package with a port in the url\n    '
    pkgs = modules.pkg.list_pkgs()
    nano = pkgs.get('nano')
    if nano:
        modules.pkg.remove('nano')
    if grains['os_family'] == 'Debian':
        url = modules.cmd.run('apt download --print-uris nano').split()[-4]
        if url.startswith("'mirror+file"):
            url = 'http://ftp.debian.org/debian/pool/' + url.split('pool')[1].rstrip("'")
        try:
            ret = modules.pkg.install(sources=f'[{{"nano":{url}}}]')
            version = re.compile('\\d\\.\\d')
            assert version.search(url).group(0) in ret['nano']['new']
        finally:
            modules.pkg.remove('nano')
            if nano:
                modules.pkg.install(f'nano={nano}')