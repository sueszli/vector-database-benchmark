from __future__ import annotations
import os
import os.path
import stat
import pytest
from ansible.config.manager import find_ini_config_file
from ansible.module_utils.common.text.converters import to_text
real_exists = os.path.exists
real_isdir = os.path.isdir
working_dir = os.path.dirname(__file__)
cfg_in_cwd = os.path.join(working_dir, 'ansible.cfg')
cfg_dir = os.path.join(working_dir, 'data')
cfg_file = os.path.join(cfg_dir, 'ansible.cfg')
alt_cfg_file = os.path.join(cfg_dir, 'test.cfg')
cfg_in_homedir = os.path.expanduser('~/.ansible.cfg')

@pytest.fixture
def setup_env(request, monkeypatch):
    if False:
        print('Hello World!')
    cur_config = os.environ.get('ANSIBLE_CONFIG', None)
    cfg_path = request.param[0]
    if cfg_path is None and cur_config:
        monkeypatch.delenv('ANSIBLE_CONFIG')
    else:
        monkeypatch.setenv('ANSIBLE_CONFIG', request.param[0])
    yield

@pytest.fixture
def setup_existing_files(request, monkeypatch):
    if False:
        return 10

    def _os_path_exists(path):
        if False:
            while True:
                i = 10
        if to_text(path) in request.param[0]:
            return True
        else:
            return False

    def _os_access(path, access):
        if False:
            print('Hello World!')
        assert to_text(path) in request.param[0]
        return True
    monkeypatch.setattr('os.path.exists', _os_path_exists)
    monkeypatch.setattr('os.access', _os_access)
    monkeypatch.setattr('os.getcwd', lambda : os.path.dirname(cfg_dir))
    monkeypatch.setattr('os.path.isdir', lambda path: True if to_text(path) == cfg_dir else real_isdir(path))

class TestFindIniFile:

    @pytest.mark.parametrize('setup_env, expected', (([alt_cfg_file], alt_cfg_file), ([cfg_dir], cfg_file)), indirect=['setup_env'])
    @pytest.mark.parametrize('setup_existing_files', [[('/etc/ansible/ansible.cfg', cfg_in_homedir, cfg_in_cwd, alt_cfg_file, cfg_file)]], indirect=['setup_existing_files'])
    def test_env_has_cfg_file(self, setup_env, setup_existing_files, expected):
        if False:
            while True:
                i = 10
        'ANSIBLE_CONFIG is specified, use it'
        warnings = set()
        assert find_ini_config_file(warnings) == expected
        assert warnings == set()

    @pytest.mark.parametrize('setup_env', ([alt_cfg_file], [cfg_dir]), indirect=['setup_env'])
    @pytest.mark.parametrize('setup_existing_files', [[('/etc/ansible/ansible.cfg', cfg_in_homedir, cfg_in_cwd)]], indirect=['setup_existing_files'])
    def test_env_has_no_cfg_file(self, setup_env, setup_existing_files):
        if False:
            return 10
        'ANSIBLE_CONFIG is specified but the file does not exist'
        warnings = set()
        assert find_ini_config_file(warnings) == cfg_in_cwd
        assert warnings == set()

    @pytest.mark.parametrize('setup_env', [[None]], indirect=['setup_env'])
    @pytest.mark.parametrize('setup_existing_files', [[('/etc/ansible/ansible.cfg', cfg_in_homedir, cfg_in_cwd, cfg_file, alt_cfg_file)]], indirect=['setup_existing_files'])
    def test_ini_in_cwd(self, setup_env, setup_existing_files):
        if False:
            print('Hello World!')
        'ANSIBLE_CONFIG not specified.  Use the cwd cfg'
        warnings = set()
        assert find_ini_config_file(warnings) == cfg_in_cwd
        assert warnings == set()

    @pytest.mark.parametrize('setup_env', [[None]], indirect=['setup_env'])
    @pytest.mark.parametrize('setup_existing_files', [[('/etc/ansible/ansible.cfg', cfg_in_homedir, cfg_file, alt_cfg_file)]], indirect=['setup_existing_files'])
    def test_ini_in_homedir(self, setup_env, setup_existing_files):
        if False:
            return 10
        'First config found is in the homedir'
        warnings = set()
        assert find_ini_config_file(warnings) == cfg_in_homedir
        assert warnings == set()

    @pytest.mark.parametrize('setup_env', [[None]], indirect=['setup_env'])
    @pytest.mark.parametrize('setup_existing_files', [[('/etc/ansible/ansible.cfg', cfg_file, alt_cfg_file)]], indirect=['setup_existing_files'])
    def test_ini_in_systemdir(self, setup_env, setup_existing_files):
        if False:
            print('Hello World!')
        'First config found is the system config'
        warnings = set()
        assert find_ini_config_file(warnings) == '/etc/ansible/ansible.cfg'
        assert warnings == set()

    @pytest.mark.parametrize('setup_env', [[None]], indirect=['setup_env'])
    @pytest.mark.parametrize('setup_existing_files', [[('/etc/ansible/ansible.cfg', cfg_in_homedir, cfg_file, alt_cfg_file)]], indirect=['setup_existing_files'])
    def test_cwd_does_not_exist(self, setup_env, setup_existing_files, monkeypatch):
        if False:
            return 10
        "Smoketest current working directory doesn't exist"

        def _os_stat(path):
            if False:
                return 10
            raise OSError('%s does not exist' % path)
        monkeypatch.setattr('os.stat', _os_stat)
        warnings = set()
        assert find_ini_config_file(warnings) == cfg_in_homedir
        assert warnings == set()

    @pytest.mark.parametrize('setup_env', [[None]], indirect=['setup_env'])
    @pytest.mark.parametrize('setup_existing_files', [[list()]], indirect=['setup_existing_files'])
    def test_no_config(self, setup_env, setup_existing_files):
        if False:
            return 10
        'No config present, no config found'
        warnings = set()
        assert find_ini_config_file(warnings) is None
        assert warnings == set()

    @pytest.mark.parametrize('setup_env', [[None]], indirect=['setup_env'])
    @pytest.mark.parametrize('setup_existing_files', [[('/etc/ansible/ansible.cfg', cfg_in_homedir, cfg_file, alt_cfg_file)]], indirect=['setup_existing_files'])
    def test_no_cwd_cfg_no_warning_on_writable(self, setup_env, setup_existing_files, monkeypatch):
        if False:
            return 10
        'If the cwd is writable but there is no config file there, move on with no warning'
        real_stat = os.stat

        def _os_stat(path):
            if False:
                print('Hello World!')
            assert path == working_dir
            from posix import stat_result
            stat_info = list(real_stat(path))
            stat_info[stat.ST_MODE] |= stat.S_IWOTH
            return stat_result(stat_info)
        monkeypatch.setattr('os.stat', _os_stat)
        warnings = set()
        assert find_ini_config_file(warnings) == cfg_in_homedir
        assert len(warnings) == 0

    @pytest.mark.parametrize('setup_env', [[None]], indirect=['setup_env'])
    @pytest.mark.parametrize('setup_existing_files', [[('/etc/ansible/ansible.cfg', cfg_in_homedir, cfg_in_cwd, cfg_file, alt_cfg_file)]], indirect=['setup_existing_files'])
    def test_cwd_warning_on_writable(self, setup_env, setup_existing_files, monkeypatch):
        if False:
            return 10
        'If the cwd is writable, warn and skip it '
        real_stat = os.stat

        def _os_stat(path):
            if False:
                while True:
                    i = 10
            assert path == working_dir
            from posix import stat_result
            stat_info = list(real_stat(path))
            stat_info[stat.ST_MODE] |= stat.S_IWOTH
            return stat_result(stat_info)
        monkeypatch.setattr('os.stat', _os_stat)
        warnings = set()
        assert find_ini_config_file(warnings) == cfg_in_homedir
        assert len(warnings) == 1
        warning = warnings.pop()
        assert u'Ansible is being run in a world writable directory' in warning
        assert u'ignoring it as an ansible.cfg source' in warning

    @pytest.mark.parametrize('setup_env, expected', (([alt_cfg_file], alt_cfg_file), ([cfg_in_cwd], cfg_in_cwd)), indirect=['setup_env'])
    @pytest.mark.parametrize('setup_existing_files', [[('/etc/ansible/ansible.cfg', cfg_in_homedir, cfg_in_cwd, cfg_file, alt_cfg_file)]], indirect=['setup_existing_files'])
    def test_no_warning_on_writable_if_env_used(self, setup_env, setup_existing_files, monkeypatch, expected):
        if False:
            for i in range(10):
                print('nop')
        'If the cwd is writable but ANSIBLE_CONFIG was used, no warning should be issued'
        real_stat = os.stat

        def _os_stat(path):
            if False:
                return 10
            if path != working_dir:
                return real_stat(path)
            from posix import stat_result
            stat_info = list(real_stat(path))
            stat_info[stat.ST_MODE] |= stat.S_IWOTH
            return stat_result(stat_info)
        monkeypatch.setattr('os.stat', _os_stat)
        warnings = set()
        assert find_ini_config_file(warnings) == expected
        assert warnings == set()

    @pytest.mark.parametrize('setup_env', [[None]], indirect=['setup_env'])
    @pytest.mark.parametrize('setup_existing_files', [[('/etc/ansible/ansible.cfg', cfg_in_homedir, cfg_in_cwd, cfg_file, alt_cfg_file)]], indirect=['setup_existing_files'])
    def test_cwd_warning_on_writable_no_warning_set(self, setup_env, setup_existing_files, monkeypatch):
        if False:
            i = 10
            return i + 15
        'Smoketest that the function succeeds even though no warning set was passed in'
        real_stat = os.stat

        def _os_stat(path):
            if False:
                for i in range(10):
                    print('nop')
            assert path == working_dir
            from posix import stat_result
            stat_info = list(real_stat(path))
            stat_info[stat.ST_MODE] |= stat.S_IWOTH
            return stat_result(stat_info)
        monkeypatch.setattr('os.stat', _os_stat)
        assert find_ini_config_file() == cfg_in_homedir