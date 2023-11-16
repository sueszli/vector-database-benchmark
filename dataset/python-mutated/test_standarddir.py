"""Tests for qutebrowser.utils.standarddir."""
import os
import os.path
import sys
import json
import types
import textwrap
import logging
import subprocess
from qutebrowser.qt.core import QStandardPaths
import pytest
from qutebrowser.utils import standarddir, utils, version
APPNAME = 'qute_test'
pytestmark = pytest.mark.usefixtures('qapp')

@pytest.fixture
def fake_home_envvar(monkeypatch, tmp_path):
    if False:
        while True:
            i = 10
    'Fake a different HOME via environment variables.'
    for k in ['XDG_DATA_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME']:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv('HOME', str(tmp_path))

@pytest.fixture(autouse=True)
def clear_standarddir_cache_and_patch(qapp, monkeypatch):
    if False:
        i = 10
        return i + 15
    'Make sure the standarddir cache is cleared before/after each test.\n\n    Also, patch APPNAME to qute_test.\n    '
    assert qapp.applicationName() == APPNAME
    monkeypatch.setattr(standarddir, '_locations', {})
    monkeypatch.setattr(standarddir, 'APPNAME', APPNAME)
    yield
    monkeypatch.setattr(standarddir, '_locations', {})

@pytest.mark.parametrize('orgname, expected', [(None, ''), ('test', 'test')])
def test_unset_organization(qapp, orgname, expected):
    if False:
        for i in range(10):
            print('nop')
    'Test unset_organization.\n\n    Args:\n        orgname: The organizationName to set initially.\n        expected: The organizationName which is expected when reading back.\n    '
    qapp.setOrganizationName(orgname)
    assert qapp.organizationName() == expected
    with standarddir._unset_organization():
        assert qapp.organizationName() == ''
    assert qapp.organizationName() == expected

def test_unset_organization_no_qapp(monkeypatch):
    if False:
        i = 10
        return i + 15
    'Without a QApplication, _unset_organization should do nothing.'
    monkeypatch.setattr(standarddir.QApplication, 'instance', lambda : None)
    with standarddir._unset_organization():
        pass

@pytest.mark.fake_os('mac')
@pytest.mark.posix
def test_fake_mac_config(tmp_path, fake_home_envvar):
    if False:
        print('Hello World!')
    'Test standardir.config on a fake Mac.'
    expected = str(tmp_path) + '/.qute_test'
    standarddir._init_config(args=None)
    assert standarddir.config() == expected

@pytest.mark.parametrize('what', ['data', 'config', 'cache'])
@pytest.mark.not_mac
@pytest.mark.fake_os('windows')
def test_fake_windows(tmpdir, monkeypatch, what):
    if False:
        return 10
    'Make sure the config/data/cache dirs are correct on a fake Windows.'
    monkeypatch.setattr(standarddir.QStandardPaths, 'writableLocation', lambda typ: str(tmpdir / APPNAME))
    standarddir._init_config(args=None)
    standarddir._init_data(args=None)
    standarddir._init_cache(args=None)
    func = getattr(standarddir, what)
    assert func() == str(tmpdir / APPNAME / what)

@pytest.mark.posix
def test_fake_haiku(tmpdir, monkeypatch):
    if False:
        i = 10
        return i + 15
    'Test getting data dir on HaikuOS.'
    locations = {QStandardPaths.StandardLocation.AppDataLocation: '', QStandardPaths.StandardLocation.ConfigLocation: str(tmpdir / 'config' / APPNAME)}
    monkeypatch.setattr(standarddir.QStandardPaths, 'writableLocation', locations.get)
    monkeypatch.setattr(standarddir.sys, 'platform', 'haiku1')
    standarddir._init_data(args=None)
    assert standarddir.data() == str(tmpdir / 'config' / APPNAME / 'data')

class TestWritableLocation:
    """Tests for _writable_location."""

    def test_empty(self, monkeypatch):
        if False:
            for i in range(10):
                print('nop')
        'Test QStandardPaths returning an empty value.'
        monkeypatch.setattr('qutebrowser.utils.standarddir.QStandardPaths.writableLocation', lambda typ: '')
        with pytest.raises(standarddir.EmptyValueError):
            standarddir._writable_location(QStandardPaths.StandardLocation.AppDataLocation)

    def test_sep(self, monkeypatch):
        if False:
            print('Hello World!')
        'Make sure the right kind of separator is used.'
        monkeypatch.setattr(standarddir.os, 'sep', '\\')
        monkeypatch.setattr(standarddir.os.path, 'join', lambda *parts: '\\'.join(parts))
        loc = standarddir._writable_location(QStandardPaths.StandardLocation.AppDataLocation)
        assert '/' not in loc
        assert '\\' in loc

class TestStandardDir:

    @pytest.mark.parametrize('func, init_func, varname', [(standarddir.data, standarddir._init_data, 'XDG_DATA_HOME'), (standarddir.config, standarddir._init_config, 'XDG_CONFIG_HOME'), (lambda : standarddir.config(auto=True), standarddir._init_config, 'XDG_CONFIG_HOME'), (standarddir.cache, standarddir._init_cache, 'XDG_CACHE_HOME'), pytest.param(standarddir.runtime, standarddir._init_runtime, 'XDG_RUNTIME_DIR', marks=pytest.mark.not_flatpak)])
    @pytest.mark.linux
    def test_linux_explicit(self, monkeypatch, tmpdir, func, init_func, varname):
        if False:
            for i in range(10):
                print('nop')
        'Test dirs with XDG environment variables explicitly set.\n\n        Args:\n            func: The function to test.\n            init_func: The initialization function to call.\n            varname: The environment variable which should be set.\n        '
        monkeypatch.setenv(varname, str(tmpdir))
        if varname == 'XDG_RUNTIME_DIR':
            tmpdir.chmod(448)
        init_func(args=None)
        assert func() == str(tmpdir / APPNAME)

    @pytest.mark.parametrize('func, subdirs', [(standarddir.data, ['.local', 'share', APPNAME]), (standarddir.config, ['.config', APPNAME]), (lambda : standarddir.config(auto=True), ['.config', APPNAME]), (standarddir.cache, ['.cache', APPNAME]), (standarddir.download, ['Downloads'])])
    @pytest.mark.linux
    def test_linux_normal(self, fake_home_envvar, tmp_path, func, subdirs):
        if False:
            for i in range(10):
                print('nop')
        'Test dirs with XDG_*_HOME not set.'
        standarddir._init_dirs()
        assert func() == str(tmp_path.joinpath(*subdirs))

    @pytest.mark.linux
    @pytest.mark.parametrize('args_basedir', [True, False])
    def test_flatpak_runtimedir(self, fake_flatpak, monkeypatch, tmp_path, args_basedir):
        if False:
            print('Hello World!')
        runtime_path = tmp_path / 'runtime'
        runtime_path.mkdir()
        runtime_path.chmod(448)
        monkeypatch.setenv('XDG_RUNTIME_DIR', str(runtime_path))
        if args_basedir:
            init_args = types.SimpleNamespace(basedir=str(tmp_path))
            expected = tmp_path / 'runtime'
        else:
            init_args = None
            expected = runtime_path / 'app' / 'org.qutebrowser.qutebrowser'
        standarddir._init_runtime(args=init_args)
        assert standarddir.runtime() == str(expected)

    @pytest.mark.fake_os('windows')
    def test_runtimedir_empty_tempdir(self, monkeypatch, tmpdir):
        if False:
            while True:
                i = 10
        'With an empty tempdir on non-Linux, we should raise.'
        monkeypatch.setattr(standarddir.QStandardPaths, 'writableLocation', lambda typ: '')
        with pytest.raises(standarddir.EmptyValueError):
            standarddir._init_runtime(args=None)

    @pytest.mark.parametrize('func, elems, expected', [(standarddir.data, 2, [APPNAME, 'data']), (standarddir.config, 2, [APPNAME, 'config']), (lambda : standarddir.config(auto=True), 2, [APPNAME, 'config']), (standarddir.cache, 2, [APPNAME, 'cache']), (standarddir.download, 1, ['Downloads'])])
    @pytest.mark.windows
    def test_windows(self, func, elems, expected):
        if False:
            while True:
                i = 10
        standarddir._init_dirs()
        assert func().split(os.sep)[-elems:] == expected

    @pytest.mark.parametrize('func, elems, expected', [(standarddir.data, 2, ['Application Support', APPNAME]), (lambda : standarddir.config(auto=True), 1, [APPNAME]), (standarddir.config, 0, os.path.expanduser('~').split(os.sep) + ['.qute_test']), (standarddir.cache, 2, ['Caches', APPNAME]), (standarddir.download, 1, ['Downloads'])])
    @pytest.mark.mac
    def test_mac(self, func, elems, expected):
        if False:
            while True:
                i = 10
        standarddir._init_dirs()
        assert func().split(os.sep)[-elems:] == expected

class TestArguments:
    """Tests the --basedir argument."""

    @pytest.mark.parametrize('typ, args', [('config', []), ('config', [True]), ('data', []), ('cache', []), ('download', []), pytest.param('runtime', [], marks=pytest.mark.linux)])
    def test_basedir(self, tmpdir, typ, args):
        if False:
            i = 10
            return i + 15
        'Test --basedir.'
        expected = str(tmpdir / typ)
        init_args = types.SimpleNamespace(basedir=str(tmpdir))
        standarddir._init_dirs(init_args)
        func = getattr(standarddir, typ)
        assert func(*args) == expected

    def test_basedir_relative(self, tmpdir):
        if False:
            print('Hello World!')
        'Test --basedir with a relative path.'
        basedir = tmpdir / 'basedir'
        basedir.ensure(dir=True)
        with tmpdir.as_cwd():
            args = types.SimpleNamespace(basedir='basedir')
            standarddir._init_dirs(args)
            assert standarddir.config() == str(basedir / 'config')

    def test_config_py_arg(self, tmpdir):
        if False:
            while True:
                i = 10
        basedir = tmpdir / 'basedir'
        basedir.ensure(dir=True)
        with tmpdir.as_cwd():
            args = types.SimpleNamespace(basedir='foo', config_py='basedir/config.py')
            standarddir._init_dirs(args)
            assert standarddir.config_py() == str(basedir / 'config.py')

    def test_config_py_no_arg(self, tmpdir):
        if False:
            return 10
        basedir = tmpdir / 'basedir'
        basedir.ensure(dir=True)
        with tmpdir.as_cwd():
            args = types.SimpleNamespace(basedir='basedir')
            standarddir._init_dirs(args)
            assert standarddir.config_py() == str(basedir / 'config' / 'config.py')

class TestInitCacheDirTag:
    """Tests for _init_cachedir_tag."""

    def test_existent_cache_dir_tag(self, tmpdir, mocker, monkeypatch):
        if False:
            return 10
        'Test with an existent CACHEDIR.TAG.'
        monkeypatch.setattr(standarddir, 'cache', lambda : str(tmpdir))
        mocker.patch('builtins.open', side_effect=AssertionError)
        m = mocker.patch('qutebrowser.utils.standarddir.os')
        m.path.join.side_effect = os.path.join
        m.path.exists.return_value = True
        standarddir._init_cachedir_tag()
        assert not tmpdir.listdir()
        m.path.exists.assert_called_with(str(tmpdir / 'CACHEDIR.TAG'))

    def test_new_cache_dir_tag(self, tmpdir, mocker, monkeypatch):
        if False:
            for i in range(10):
                print('nop')
        'Test creating a new CACHEDIR.TAG.'
        monkeypatch.setattr(standarddir, 'cache', lambda : str(tmpdir))
        standarddir._init_cachedir_tag()
        assert tmpdir.listdir() == [tmpdir / 'CACHEDIR.TAG']
        data = (tmpdir / 'CACHEDIR.TAG').read_text('utf-8')
        assert data == textwrap.dedent('\n            Signature: 8a477f597d28d172789f06886806bc55\n            # This file is a cache directory tag created by qutebrowser.\n            # For information about cache directory tags, see:\n            #  https://bford.info/cachedir/\n        ').lstrip()

    def test_open_oserror(self, caplog, unwritable_tmp_path, monkeypatch):
        if False:
            return 10
        'Test creating a new CACHEDIR.TAG.'
        monkeypatch.setattr(standarddir, 'cache', lambda : str(unwritable_tmp_path))
        with caplog.at_level(logging.ERROR, 'init'):
            standarddir._init_cachedir_tag()
        assert caplog.messages == ['Failed to create CACHEDIR.TAG']

class TestCreatingDir:
    """Make sure inexistent directories are created properly."""
    DIR_TYPES = ['config', 'data', 'cache', 'download', 'runtime']

    @pytest.mark.parametrize('typ', DIR_TYPES)
    def test_basedir(self, tmpdir, typ):
        if False:
            print('Hello World!')
        'Test --basedir.'
        basedir = tmpdir / 'basedir'
        assert not basedir.exists()
        args = types.SimpleNamespace(basedir=str(basedir))
        standarddir._init_dirs(args)
        func = getattr(standarddir, typ)
        func()
        assert basedir.exists()
        if typ == 'download' or (typ == 'runtime' and (not utils.is_linux)):
            assert not (basedir / typ).exists()
        else:
            assert (basedir / typ).exists()
            if utils.is_posix:
                assert (basedir / typ).stat().mode & 511 == 448

    @pytest.mark.parametrize('typ', DIR_TYPES)
    def test_exists_race_condition(self, mocker, tmpdir, typ):
        if False:
            print('Hello World!')
        "Make sure there can't be a TOCTOU issue when creating the file.\n\n        See https://github.com/qutebrowser/qutebrowser/issues/942.\n        "
        (tmpdir / typ).ensure(dir=True)
        m = mocker.patch('qutebrowser.utils.standarddir.os')
        m.makedirs = os.makedirs
        m.sep = os.sep
        m.path.join = os.path.join
        m.expanduser = os.path.expanduser
        m.path.exists.return_value = False
        m.path.abspath = lambda x: x
        args = types.SimpleNamespace(basedir=str(tmpdir))
        standarddir._init_dirs(args)
        func = getattr(standarddir, typ)
        func()

class TestSystemData:
    """Test system data path."""

    @pytest.mark.linux
    @pytest.mark.parametrize('is_flatpak, expected', [(True, '/app/share/qute_test'), (False, '/usr/share/qute_test')])
    def test_system_datadir_exist_linux(self, monkeypatch, tmpdir, is_flatpak, expected):
        if False:
            while True:
                i = 10
        'Test that /usr/share/qute_test is used if path exists.'
        monkeypatch.setenv('XDG_DATA_HOME', str(tmpdir))
        monkeypatch.setattr(os.path, 'exists', lambda path: True)
        monkeypatch.setattr(version, 'is_flatpak', lambda : is_flatpak)
        standarddir._init_data(args=None)
        assert standarddir.data(system=True) == expected

    @pytest.mark.linux
    def test_system_datadir_not_exist_linux(self, monkeypatch, tmpdir, fake_args):
        if False:
            i = 10
            return i + 15
        "Test that system-wide path isn't used on linux if path not exist."
        fake_args.basedir = str(tmpdir)
        monkeypatch.setattr(os.path, 'exists', lambda path: False)
        standarddir._init_data(args=fake_args)
        assert standarddir.data(system=True) == standarddir.data()

    def test_system_datadir_unsupportedos(self, monkeypatch, tmpdir, fake_args):
        if False:
            while True:
                i = 10
        'Test that system-wide path is not used on non-Linux OS.'
        fake_args.basedir = str(tmpdir)
        monkeypatch.setattr(sys, 'platform', 'potato')
        standarddir._init_data(args=fake_args)
        assert standarddir.data(system=True) == standarddir.data()

@pytest.mark.parametrize('args_kind', ['basedir', 'normal', 'none'])
def test_init(tmp_path, args_kind, fake_home_envvar):
    if False:
        print('Hello World!')
    'Do some sanity checks for standarddir.init().\n\n    Things like _init_cachedir_tag() are tested in more detail in other tests.\n    '
    assert standarddir._locations == {}
    if args_kind == 'normal':
        args = types.SimpleNamespace(basedir=None)
    elif args_kind == 'basedir':
        args = types.SimpleNamespace(basedir=str(tmp_path))
    else:
        assert args_kind == 'none'
        args = None
    standarddir.init(args)
    assert standarddir._locations != {}

@pytest.mark.linux
def test_downloads_dir_not_created(monkeypatch, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Make sure ~/Downloads is not created.'
    download_dir = tmpdir / 'Downloads'
    monkeypatch.setenv('HOME', str(tmpdir))
    monkeypatch.delenv('XDG_CONFIG_HOME', raising=False)
    standarddir._init_dirs()
    assert standarddir.download() == str(download_dir)
    assert not download_dir.exists()

def test_no_qapplication(qapp, tmpdir, monkeypatch):
    if False:
        print('Hello World!')
    'Make sure directories with/without QApplication are equal.'
    sub_code = "\n        import sys\n        import json\n\n        sys.path = sys.argv[1:]  # make sure we have the same python path\n\n        from qutebrowser.qt.widgets import QApplication\n        from qutebrowser.utils import standarddir\n\n        assert QApplication.instance() is None\n\n        standarddir.APPNAME = 'qute_test'\n        standarddir._init_dirs()\n\n        locations = {k.name: v for k, v in standarddir._locations.items()}\n        print(json.dumps(locations))\n    "
    pyfile = tmpdir / 'sub.py'
    pyfile.write_text(textwrap.dedent(sub_code), encoding='ascii')
    for name in ['CONFIG', 'DATA', 'CACHE']:
        monkeypatch.delenv('XDG_{}_HOME'.format(name), raising=False)
    runtime_dir = tmpdir / 'runtime'
    runtime_dir.ensure(dir=True)
    runtime_dir.chmod(448)
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(runtime_dir))
    home_dir = tmpdir / 'home'
    home_dir.ensure(dir=True)
    monkeypatch.setenv('HOME', str(home_dir))
    proc = subprocess.run([sys.executable, str(pyfile)] + sys.path, text=True, check=True, stdout=subprocess.PIPE)
    sub_locations = json.loads(proc.stdout)
    standarddir._init_dirs()
    locations = {k.name: v for (k, v) in standarddir._locations.items()}
    assert sub_locations == locations