import io
import os
import pytest
import salt.modules.cmdmod as cmdmod
import salt.modules.restartcheck as restartcheck
import salt.modules.system as system
import salt.modules.systemd_service as service
import salt.utils.path
from tests.support.mock import ANY, MagicMock, create_autospec, patch
from tests.support.paths import SALT_CODE_DIR

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {restartcheck: {}}

def test_when_timestamp_file_does_not_exist_then_file_changed_nilrt_should_be_True():
    if False:
        for i in range(10):
            print('nop')
    expected_changed = True

    def timestamp_not_exists(filename):
        if False:
            i = 10
            return i + 15
        if filename.endswith('.timestamp'):
            return False
        return True
    with patch('os.path.exists', side_effect=timestamp_not_exists, autospec=True):
        actual_changed = restartcheck._file_changed_nilrt(full_filepath='fnord')
        assert actual_changed == expected_changed

def test_when_timestamp_file_exists_but_not_md5sum_file_then_file_changed_nilrt_should_be_True():
    if False:
        i = 10
        return i + 15
    expected_changed = True

    def timestamp_not_exists(filename):
        if False:
            return 10
        if filename.endswith('.md5sum'):
            return False
        return True
    with patch('os.path.exists', side_effect=timestamp_not_exists, autospec=True):
        actual_changed = restartcheck._file_changed_nilrt(full_filepath='fnord')
        assert actual_changed == expected_changed

def test_when_nisysapi_path_exists_and_nilrt_files_changed_then_sysapi_changed_nilrt_should_be_True():
    if False:
        return 10
    expected_change = True
    patch_os_path = patch('os.path.exists', return_value=True, autospec=True)
    patch_file_changed = patch('salt.modules.restartcheck._file_changed_nilrt', autospec=True, return_value=True)
    with patch_os_path, patch_file_changed:
        actual_change = restartcheck._sysapi_changed_nilrt()
    assert actual_change == expected_change

@pytest.mark.parametrize('cpuarch_grain', ['arm', 'x86_64'])
def test_when_nisysapi_conf_d_path_does_not_exist_then_sysapi_changed_should_be_False(cpuarch_grain):
    if False:
        return 10
    expected_change = False

    def conf_d_not_exists(filename):
        if False:
            while True:
                i = 10
        return 'nisysapi/conf.d' not in filename
    patch_os_path = patch('os.path.exists', side_effect=conf_d_not_exists, autospec=True)
    patch_file_changed = patch('salt.modules.restartcheck._file_changed_nilrt', autospec=True, return_value=False)
    patch_grain = patch.dict(restartcheck.__grains__, {'cpuarch': cpuarch_grain})
    with patch_os_path, patch_file_changed, patch_grain:
        actual_change = restartcheck._sysapi_changed_nilrt()
    assert actual_change == expected_change

@pytest.mark.parametrize('cpuarch_grain', ['arm', 'x86_64'])
def test_when_nisysapi_conf_d_path_does_exist_and_no_restart_check_file_exists_then_sysapi_changed_should_be_True(cpuarch_grain):
    if False:
        i = 10
        return i + 15
    expected_change = True

    def conf_d_not_exists(filename):
        if False:
            for i in range(10):
                print('nop')
        return not filename.endswith('/sysapi.conf.d.count')
    patch_os_path = patch('os.path.exists', side_effect=conf_d_not_exists, autospec=True)
    patch_file_changed = patch('salt.modules.restartcheck._file_changed_nilrt', autospec=True, return_value=False)
    patch_grain = patch.dict(restartcheck.__grains__, {'cpuarch': cpuarch_grain})
    with patch_os_path, patch_file_changed, patch_grain:
        actual_change = restartcheck._sysapi_changed_nilrt()
    assert actual_change == expected_change

@pytest.mark.parametrize('cpuarch_grain', ['arm', 'x86_64'])
def test_when_nisysapi_conf_d_path_does_exist_and_count_file_exists_and_count_is_different_than_files_in_conf_d_path_then_sysapi_changed_should_be_True(cpuarch_grain):
    if False:
        while True:
            i = 10
    expected_change = True
    fake_count = io.StringIO('42')
    patch_listdir = patch('os.listdir', autospec=True, return_value=['boop'])
    patch_os_path = patch('os.path.exists', return_value=True, autospec=True)
    patch_file_changed = patch('salt.modules.restartcheck._file_changed_nilrt', autospec=True, return_value=False)
    patch_grain = patch.dict(restartcheck.__grains__, {'cpuarch': cpuarch_grain})
    patch_fopen = patch('salt.utils.files.fopen', autospec=True, return_value=fake_count)
    with patch_os_path, patch_file_changed, patch_grain, patch_listdir, patch_fopen:
        actual_change = restartcheck._sysapi_changed_nilrt()
    assert actual_change == expected_change

@pytest.mark.parametrize('cpuarch_grain', ['arm', 'x86_64'])
def test_when_nisysapi_conf_d_path_does_exist_and_count_file_exists_and_count_is_same_as_files_in_conf_d_path_but_no_nilrt_files_changed_then_sysapi_changed_should_be_False(cpuarch_grain):
    if False:
        i = 10
        return i + 15
    expected_change = False
    fake_count = io.StringIO('42')
    patch_listdir = patch('os.listdir', autospec=True, return_value=['boop'] * 42)
    patch_os_path = patch('os.path.exists', return_value=True, autospec=True)
    patch_file_changed = patch('salt.modules.restartcheck._file_changed_nilrt', autospec=True, return_value=False)
    patch_grain = patch.dict(restartcheck.__grains__, {'cpuarch': cpuarch_grain})
    patch_fopen = patch('salt.utils.files.fopen', autospec=True, return_value=fake_count)
    with patch_os_path, patch_file_changed, patch_grain, patch_listdir, patch_fopen:
        actual_change = restartcheck._sysapi_changed_nilrt()
    assert actual_change == expected_change

@pytest.mark.parametrize('cpuarch_grain', ['arm', 'x86_64'])
def test_when_nisysapi_conf_d_path_does_exist_and_count_file_exists_and_count_is_same_as_files_in_conf_d_path_and_file_changed_nilrt_then_sysapi_changed_should_be_True(cpuarch_grain):
    if False:
        for i in range(10):
            print('nop')
    expected_change = True

    def fake_file_changed(filename):
        if False:
            return 10
        return filename != '/usr/local/natinst/share/nisysapi.ini'
    fake_count = io.StringIO('42')
    patch_listdir = patch('os.listdir', autospec=True, return_value=['boop'] * 42)
    patch_os_path = patch('os.path.exists', return_value=True, autospec=True)
    patch_file_changed = patch('salt.modules.restartcheck._file_changed_nilrt', autospec=True, side_effect=fake_file_changed)
    patch_grain = patch.dict(restartcheck.__grains__, {'cpuarch': cpuarch_grain})
    patch_fopen = patch('salt.utils.files.fopen', autospec=True, return_value=fake_count)
    with patch_os_path, patch_file_changed, patch_grain, patch_listdir, patch_fopen:
        actual_change = restartcheck._sysapi_changed_nilrt()
    assert actual_change == expected_change

def test_when_nilinuxrt_and_not_kernel_modules_changed_or_sysapi_files_changed_and_not_reboot_required_witnessed_then_no_reboot_should_be_required():
    if False:
        for i in range(10):
            print('nop')
    expected_result = 'No packages seem to need to be restarted.'
    restart_required = False
    current_kernel = 'fnord'
    patch_grains = patch.dict(restartcheck.__grains__, {'os_family': 'NILinuxRT'})
    patch_kernel_versions = patch('salt.modules.restartcheck._kernel_versions_nilrt', autospec=True, return_value=[current_kernel])
    patch_salt = patch.dict(restartcheck.__salt__, {'cmd.run': create_autospec(cmdmod.run, return_value=current_kernel), 'system.get_reboot_required_witnessed': create_autospec(system.get_reboot_required_witnessed, return_value=restart_required), 'service.get_running': create_autospec(service.get_running, return_value=[])})
    patch_kernel_mod_changed = patch('salt.modules.restartcheck._kernel_modules_changed_nilrt', autospec=True, return_value=False)
    patch_sysapi_changed = patch('salt.modules.restartcheck._sysapi_changed_nilrt', autospec=True, return_value=False)
    patch_del_files = patch('salt.modules.restartcheck._deleted_files', autospec=True, return_value=[])
    with patch_grains, patch_kernel_versions, patch_salt, patch_sysapi_changed, patch_kernel_mod_changed, patch_del_files:
        actual_result = restartcheck.restartcheck()
    assert actual_result == expected_result

def test_when_nilinuxrt_and_not_kernel_modules_changed_or_sysapi_files_changed_and_reboot_required_witnessed_then_reboot_should_be_required():
    if False:
        while True:
            i = 10
    expected_result = 'System restart required.\n\n'
    restart_required = True
    current_kernel = 'fnord'
    patch_grains = patch.dict(restartcheck.__grains__, {'os_family': 'NILinuxRT'})
    patch_kernel_versions = patch('salt.modules.restartcheck._kernel_versions_nilrt', autospec=True, return_value=[current_kernel])
    patch_salt = patch.dict(restartcheck.__salt__, {'cmd.run': create_autospec(cmdmod.run, return_value=current_kernel), 'system.get_reboot_required_witnessed': create_autospec(system.get_reboot_required_witnessed, return_value=restart_required), 'service.get_running': create_autospec(service.get_running, return_value=[])})
    patch_kernel_mod_changed = patch('salt.modules.restartcheck._kernel_modules_changed_nilrt', autospec=True, return_value=False)
    patch_sysapi_changed = patch('salt.modules.restartcheck._sysapi_changed_nilrt', autospec=True, return_value=False)
    patch_del_files = patch('salt.modules.restartcheck._deleted_files', autospec=True, return_value=[])
    with patch_grains, patch_kernel_versions, patch_salt, patch_sysapi_changed, patch_kernel_mod_changed, patch_del_files:
        actual_result = restartcheck.restartcheck()
    assert actual_result == expected_result

def test_kernel_versions_debian():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test kernel version debian\n    '
    mock = MagicMock(return_value='  Installed: 4.9.82-1+deb9u3')
    with patch.dict(restartcheck.__grains__, {'os': 'Debian'}):
        with patch.dict(restartcheck.__salt__, {'cmd.run': mock}):
            assert restartcheck._kernel_versions_debian() == ['4.9.82-1+deb9u3']

def test_kernel_versions_ubuntu():
    if False:
        return 10
    '\n    Test kernel version ubuntu\n    '
    mock = MagicMock(return_value='  Installed: 4.10.0-42.46')
    with patch.dict(restartcheck.__grains__, {'os': 'Ubuntu'}):
        with patch.dict(restartcheck.__salt__, {'cmd.run': mock}):
            assert restartcheck._kernel_versions_debian() == ['4.10.0-42.46', '4.10.0-42-generic #46', '4.10.0-42-lowlatency #46']

def test_kernel_versions_redhat():
    if False:
        return 10
    '\n    Test if it return a data structure of the current, in-memory rules\n    '
    mock = MagicMock(return_value='kernel-3.10.0-862.el7.x86_64                  Thu Apr 5 00:40:00 2018')
    with patch.dict(restartcheck.__salt__, {'cmd.run': mock}):
        assert restartcheck._kernel_versions_redhat() == ['3.10.0-862.el7.x86_64']

def test_valid_deleted_file_deleted():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test (deleted) file\n    '
    assert restartcheck._valid_deleted_file('/usr/lib/test (deleted)')

def test_valid_deleted_file_psth_inode():
    if False:
        while True:
            i = 10
    '\n    Test (path inode=1) file\n    '
    assert restartcheck._valid_deleted_file('/usr/lib/test (path inode=1)')

def test_valid_deleted_file_var_log():
    if False:
        return 10
    '\n    Test /var/log/\n    '
    assert not restartcheck._valid_deleted_file('/var/log/test')
    assert not restartcheck._valid_deleted_file('/var/log/test (deleted)')
    assert not restartcheck._valid_deleted_file('/var/log/test (path inode=1)')

def test_valid_deleted_file_var_local_log():
    if False:
        i = 10
        return i + 15
    '\n    Test /var/local/log/\n    '
    assert not restartcheck._valid_deleted_file('/var/local/log/test')
    assert not restartcheck._valid_deleted_file('/var/local/log/test (deleted)')
    assert not restartcheck._valid_deleted_file('/var/local/log/test (path inode=1)')

def test_valid_deleted_file_var_run():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test /var/run/\n    '
    assert not restartcheck._valid_deleted_file('/var/run/test')
    assert not restartcheck._valid_deleted_file('/var/run/test (deleted)')
    assert not restartcheck._valid_deleted_file('/var/run/test (path inode=1)')

def test_valid_deleted_file_var_local_run():
    if False:
        i = 10
        return i + 15
    '\n    Test /var/local/run/\n    '
    assert not restartcheck._valid_deleted_file('/var/local/run/test')
    assert not restartcheck._valid_deleted_file('/var/local/run/test (deleted)')
    assert not restartcheck._valid_deleted_file('/var/local/run/test (path inode=1)')

def test_valid_deleted_file_tmp():
    if False:
        print('Hello World!')
    '\n    Test /tmp/\n    '
    assert not restartcheck._valid_deleted_file('/tmp/test')
    assert not restartcheck._valid_deleted_file('/tmp/test (deleted)')
    assert not restartcheck._valid_deleted_file('/tmp/test (path inode=1)')

def test_valid_deleted_file_dev_shm():
    if False:
        return 10
    '\n    Test /dev/shm/\n    '
    assert not restartcheck._valid_deleted_file('/dev/shm/test')
    assert not restartcheck._valid_deleted_file('/dev/shm/test (deleted)')
    assert not restartcheck._valid_deleted_file('/dev/shm/test (path inode=1)')

def test_valid_deleted_file_run():
    if False:
        while True:
            i = 10
    '\n    Test /run/\n    '
    assert not restartcheck._valid_deleted_file('/run/test')
    assert not restartcheck._valid_deleted_file('/run/test (deleted)')
    assert not restartcheck._valid_deleted_file('/run/test (path inode=1)')

def test_valid_deleted_file_drm():
    if False:
        i = 10
        return i + 15
    '\n    Test /drm/\n    '
    assert not restartcheck._valid_deleted_file('/drm/test')
    assert not restartcheck._valid_deleted_file('/drm/test (deleted)')
    assert not restartcheck._valid_deleted_file('/drm/test (path inode=1)')

def test_valid_deleted_file_var_tmp():
    if False:
        print('Hello World!')
    '\n    Test /var/tmp/\n    '
    assert not restartcheck._valid_deleted_file('/var/tmp/test')
    assert not restartcheck._valid_deleted_file('/var/tmp/test (deleted)')
    assert not restartcheck._valid_deleted_file('/var/tmp/test (path inode=1)')

def test_valid_deleted_file_var_local_tmp():
    if False:
        while True:
            i = 10
    '\n    Test /var/local/tmp/\n    '
    assert not restartcheck._valid_deleted_file('/var/local/tmp/test')
    assert not restartcheck._valid_deleted_file('/var/local/tmp/test (deleted)')
    assert not restartcheck._valid_deleted_file('/var/local/tmp/test (path inode=1)')

def test_valid_deleted_file_dev_zero():
    if False:
        while True:
            i = 10
    '\n    Test /dev/zero/\n    '
    assert not restartcheck._valid_deleted_file('/dev/zero/test')
    assert not restartcheck._valid_deleted_file('/dev/zero/test (deleted)')
    assert not restartcheck._valid_deleted_file('/dev/zero/test (path inode=1)')

def test_valid_deleted_file_dev_pts():
    if False:
        return 10
    '\n    Test /dev/pts/\n    '
    assert not restartcheck._valid_deleted_file('/dev/pts/test')
    assert not restartcheck._valid_deleted_file('/dev/pts/test (deleted)')
    assert not restartcheck._valid_deleted_file('/dev/pts/test (path inode=1)')

def test_valid_deleted_file_usr_lib_locale():
    if False:
        print('Hello World!')
    '\n    Test /usr/lib/locale/\n    '
    assert not restartcheck._valid_deleted_file('/usr/lib/locale/test')
    assert not restartcheck._valid_deleted_file('/usr/lib/locale/test (deleted)')
    assert not restartcheck._valid_deleted_file('/usr/lib/locale/test (path inode=1)')

def test_valid_deleted_file_home():
    if False:
        while True:
            i = 10
    '\n    Test /home/\n    '
    assert not restartcheck._valid_deleted_file('/home/test')
    assert not restartcheck._valid_deleted_file('/home/test (deleted)')
    assert not restartcheck._valid_deleted_file('/home/test (path inode=1)')

def test_valid_deleted_file_icon_theme_cache():
    if False:
        while True:
            i = 10
    '\n    Test /test.icon-theme.cache\n    '
    assert not restartcheck._valid_deleted_file('/dev/test.icon-theme.cache')
    assert not restartcheck._valid_deleted_file('/dev/test.icon-theme.cache (deleted)')
    assert not restartcheck._valid_deleted_file('/dev/test.icon-theme.cache (path inode=1)')

def test_valid_deleted_file_var_cache_fontconfig():
    if False:
        i = 10
        return i + 15
    '\n    Test /var/cache/fontconfig/\n    '
    assert not restartcheck._valid_deleted_file('/var/cache/fontconfig/test')
    assert not restartcheck._valid_deleted_file('/var/cache/fontconfig/test (deleted)')
    assert not restartcheck._valid_deleted_file('/var/cache/fontconfig/test (path inode=1)')

def test_valid_deleted_file_var_lib_nagios3_spool():
    if False:
        while True:
            i = 10
    '\n    Test /var/lib/nagios3/spool/\n    '
    assert not restartcheck._valid_deleted_file('/var/lib/nagios3/spool/test')
    assert not restartcheck._valid_deleted_file('/var/lib/nagios3/spool/test (deleted)')
    assert not restartcheck._valid_deleted_file('/var/lib/nagios3/spool/test (path inode=1)')

def test_valid_deleted_file_var_lib_nagios3_spool_checkresults():
    if False:
        i = 10
        return i + 15
    '\n    Test /var/lib/nagios3/spool/checkresults/\n    '
    assert not restartcheck._valid_deleted_file('/var/lib/nagios3/spool/checkresults/test')
    assert not restartcheck._valid_deleted_file('/var/lib/nagios3/spool/checkresults/test (deleted)')
    assert not restartcheck._valid_deleted_file('/var/lib/nagios3/spool/checkresults/test (path inode=1)')

def test_valid_deleted_file_var_lib_postgresql():
    if False:
        i = 10
        return i + 15
    '\n    Test /var/lib/postgresql/\n    '
    assert not restartcheck._valid_deleted_file('/var/lib/postgresql/test')
    assert not restartcheck._valid_deleted_file('/var/lib/postgresql/test (deleted)')
    assert not restartcheck._valid_deleted_file('/var/lib/postgresql/test (path inode=1)')

def test_valid_deleted_file_var_lib_vdr():
    if False:
        i = 10
        return i + 15
    '\n    Test /var/lib/vdr/\n    '
    assert not restartcheck._valid_deleted_file('/var/lib/vdr/test')
    assert not restartcheck._valid_deleted_file('/var/lib/vdr/test (deleted)')
    assert not restartcheck._valid_deleted_file('/var/lib/vdr/test (path inode=1)')

def test_valid_deleted_file_aio():
    if False:
        print('Hello World!')
    '\n    Test /[aio]/\n    '
    assert not restartcheck._valid_deleted_file('/opt/test')
    assert not restartcheck._valid_deleted_file('/opt/test (deleted)')
    assert not restartcheck._valid_deleted_file('/opt/test (path inode=1)')
    assert not restartcheck._valid_deleted_file('/apt/test')
    assert not restartcheck._valid_deleted_file('/apt/test (deleted)')
    assert not restartcheck._valid_deleted_file('/apt/test (path inode=1)')
    assert not restartcheck._valid_deleted_file('/ipt/test')
    assert not restartcheck._valid_deleted_file('/ipt/test (deleted)')
    assert not restartcheck._valid_deleted_file('/ipt/test (path inode=1)')
    assert not restartcheck._valid_deleted_file('/aio/test')
    assert not restartcheck._valid_deleted_file('/aio/test (deleted)')
    assert not restartcheck._valid_deleted_file('/aio/test (path inode=1)')

def test_valid_deleted_file_sysv():
    if False:
        while True:
            i = 10
    '\n    Test /SYSV/\n    '
    assert not restartcheck._valid_deleted_file('/SYSV/test')
    assert not restartcheck._valid_deleted_file('/SYSV/test (deleted)')
    assert not restartcheck._valid_deleted_file('/SYSV/test (path inode=1)')

def test_valid_command():
    if False:
        return 10
    '\n    test for CVE-2020-28243\n    '
    create_file = os.path.join(SALT_CODE_DIR, 'created_file')
    patch_kernel = patch('salt.modules.restartcheck._kernel_versions_redhat', return_value=['3.10.0-1127.el7.x86_64'])
    services = {'NetworkManager': {'ExecMainPID': 123}, 'auditd': {'ExecMainPID': 456}, 'crond': {'ExecMainPID': 789}}
    patch_salt = patch.dict(restartcheck.__salt__, {'cmd.run': MagicMock(return_value='Linux localhost.localdomain 3.10.0-1127.el7.x86_64'), 'service.get_running': MagicMock(return_value=list(services.keys())), 'service.show': MagicMock(side_effect=list(services.values())), 'pkg.owner': MagicMock(return_value=''), 'service.available': MagicMock(return_value=True)})
    patch_deleted = patch('salt.modules.restartcheck._deleted_files', MagicMock(return_value=[(';touch {};'.format(create_file), 123, '/root/ (deleted)')]))
    patch_readlink = patch('os.readlink', return_value='/root/;touch {};'.format(create_file))
    check_error = True
    if salt.utils.path.which('repoquery'):
        check_error = False
    patch_grains = patch.dict(restartcheck.__grains__, {'os_family': 'RedHat'})
    with patch_kernel, patch_salt, patch_deleted, patch_readlink, patch_grains:
        if check_error:
            with pytest.raises(FileNotFoundError):
                restartcheck.restartcheck()
        else:
            ret = restartcheck.restartcheck()
            assert 'Found 1 processes using old versions of upgraded files' in ret
        assert not os.path.exists(create_file)

def test_valid_command_b():
    if False:
        i = 10
        return i + 15
    '\n    test for CVE-2020-28243\n    '
    create_file = os.path.join(SALT_CODE_DIR, 'created_file')
    patch_kernel = patch('salt.modules.restartcheck._kernel_versions_redhat', return_value=['3.10.0-1127.el7.x86_64'])
    services = {'NetworkManager': {'ExecMainPID': 123}, 'auditd': {'ExecMainPID': 456}, 'crond': {'ExecMainPID': 789}}
    patch_salt = patch.dict(restartcheck.__salt__, {'cmd.run': MagicMock(return_value='Linux localhost.localdomain 3.10.0-1127.el7.x86_64'), 'service.get_running': MagicMock(return_value=list(services.keys())), 'service.show': MagicMock(side_effect=list(services.values())), 'pkg.owner': MagicMock(return_value=''), 'service.available': MagicMock(return_value=True)})
    patch_deleted = patch('salt.modules.restartcheck._deleted_files', MagicMock(return_value=[('--admindir tmp dpkg', 123, '/root/ (deleted)')]))
    patch_readlink = patch('os.readlink', return_value='--admindir tmp dpkg')
    popen_mock = MagicMock()
    popen_mock.return_value.stdout.readline.side_effect = ['/usr/bin\n', '']
    patch_popen = patch('subprocess.Popen', popen_mock)
    patch_grains = patch.dict(restartcheck.__grains__, {'os_family': 'RedHat'})
    with patch_kernel, patch_salt, patch_deleted, patch_readlink, patch_grains, patch_popen:
        ret = restartcheck.restartcheck()
        assert 'Found 1 processes using old versions of upgraded files' in ret
        popen_mock.assert_called_with(['repoquery', '-l', '--admindir tmp dpkg'], stdout=ANY)