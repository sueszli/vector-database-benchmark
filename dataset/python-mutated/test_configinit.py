"""Tests for qutebrowser.config.configinit."""
import builtins
import logging
import unittest.mock
import pytest
from qutebrowser.config import config, configexc, configfiles, configinit, configdata, configtypes
from qutebrowser.utils import objreg, usertypes

@pytest.fixture
def init_patch(qapp, fake_save_manager, monkeypatch, config_tmpdir, data_tmpdir):
    if False:
        while True:
            i = 10
    monkeypatch.setattr(configfiles, 'state', None)
    monkeypatch.setattr(config, 'instance', None)
    monkeypatch.setattr(config, 'key_instance', None)
    monkeypatch.setattr(config, 'change_filters', [])
    monkeypatch.setattr(configinit, '_init_errors', None)
    monkeypatch.setattr(configtypes.FontBase, 'default_family', None)
    monkeypatch.setattr(configtypes.FontBase, 'default_size', None)
    yield
    try:
        objreg.delete('config-commands')
    except KeyError:
        pass

@pytest.fixture
def args(fake_args):
    if False:
        for i in range(10):
            print('nop')
    'Arguments needed for the config to init.'
    fake_args.temp_settings = []
    fake_args.config_py = None
    return fake_args

@pytest.fixture(autouse=True)
def configdata_init(monkeypatch):
    if False:
        print('Hello World!')
    "Make sure configdata is init'ed and no test re-init's it."
    if not configdata.DATA:
        configdata.init()
    monkeypatch.setattr(configdata, 'init', lambda : None)

class TestEarlyInit:

    def test_config_py_path(self, args, init_patch, config_py_arg):
        if False:
            print('Hello World!')
        config_py_arg.write('\n'.join(['config.load_autoconfig()', 'c.colors.hints.bg = "red"']))
        configinit.early_init(args)
        expected = 'colors.hints.bg = red'
        assert config.instance.dump_userconfig() == expected

    @pytest.mark.parametrize('config_py', [True, 'error', False])
    def test_config_py(self, init_patch, config_tmpdir, caplog, args, config_py):
        if False:
            while True:
                i = 10
        'Test loading with only a config.py.'
        config_py_file = config_tmpdir / 'config.py'
        if config_py:
            config_py_lines = ['c.colors.hints.bg = "red"', 'config.load_autoconfig(False)']
            if config_py == 'error':
                config_py_lines.append('c.foo = 42')
            config_py_file.write_text('\n'.join(config_py_lines), 'utf-8', ensure=True)
        with caplog.at_level(logging.ERROR):
            configinit.early_init(args)
        expected_errors = []
        if config_py == 'error':
            expected_errors.append("While setting 'foo': No option 'foo'")
        if configinit._init_errors is None:
            actual_errors = []
        else:
            actual_errors = [str(err) for err in configinit._init_errors.errors]
        assert actual_errors == expected_errors
        assert isinstance(config.instance, config.Config)
        assert isinstance(config.key_instance, config.KeyConfig)
        if config_py:
            expected = 'colors.hints.bg = red'
        else:
            expected = '<Default configuration>'
        assert config.instance.dump_userconfig() == expected

    @pytest.mark.parametrize('load_autoconfig', [True, False])
    @pytest.mark.parametrize('config_py', [True, 'error', False])
    @pytest.mark.parametrize('invalid_yaml', ['42', 'list', 'unknown', 'wrong-type', False])
    def test_autoconfig_yml(self, init_patch, config_tmpdir, caplog, args, load_autoconfig, config_py, invalid_yaml):
        if False:
            i = 10
            return i + 15
        'Test interaction between config.py and autoconfig.yml.'
        autoconfig_file = config_tmpdir / 'autoconfig.yml'
        config_py_file = config_tmpdir / 'config.py'
        yaml_lines = {'42': '42', 'list': '[1, 2]', 'unknown': ['settings:', '  colors.foobar:', '    global: magenta', 'config_version: 2'], 'wrong-type': ['settings:', '  tabs.position:', '    global: true', 'config_version: 2'], False: ['settings:', '  colors.hints.fg:', '    global: magenta', 'config_version: 2']}
        text = '\n'.join(yaml_lines[invalid_yaml])
        autoconfig_file.write_text(text, 'utf-8', ensure=True)
        if config_py:
            config_py_lines = ['c.colors.hints.bg = "red"']
            config_py_lines.append('config.load_autoconfig({})'.format(load_autoconfig))
            if config_py == 'error':
                config_py_lines.append('c.foo = 42')
            config_py_file.write_text('\n'.join(config_py_lines), 'utf-8', ensure=True)
        with caplog.at_level(logging.ERROR):
            configinit.early_init(args)
        expected_errors = []
        if load_autoconfig or not config_py:
            suffix = ' (autoconfig.yml)' if config_py else ''
            if invalid_yaml in ['42', 'list']:
                error = 'While loading data{}: Toplevel object is not a dict'.format(suffix)
                expected_errors.append(error)
            elif invalid_yaml == 'wrong-type':
                error = "Error{}: Invalid value 'True' - expected a value of type str but got bool.".format(suffix)
                expected_errors.append(error)
            elif invalid_yaml == 'unknown':
                error = 'While loading options{}: Unknown option colors.foobar'.format(suffix)
                expected_errors.append(error)
        if config_py == 'error':
            expected_errors.append("While setting 'foo': No option 'foo'")
        if configinit._init_errors is None:
            actual_errors = []
        else:
            actual_errors = [str(err) for err in configinit._init_errors.errors]
        assert actual_errors == expected_errors
        dump = config.instance.dump_userconfig()
        if config_py and load_autoconfig and (not invalid_yaml):
            expected = ['colors.hints.bg = red', 'colors.hints.fg = magenta']
        elif config_py:
            expected = ['colors.hints.bg = red']
        elif invalid_yaml:
            expected = ['<Default configuration>']
        else:
            expected = ['colors.hints.fg = magenta']
        assert dump == '\n'.join(expected)

    def test_autoconfig_warning(self, init_patch, args, config_tmpdir, caplog):
        if False:
            while True:
                i = 10
        'Test the warning shown for missing autoconfig loading.'
        config_py_file = config_tmpdir / 'config.py'
        config_py_file.ensure()
        with caplog.at_level(logging.ERROR):
            configinit.early_init(args)
        assert len(configinit._init_errors.errors) == 1
        error = configinit._init_errors.errors[0]
        assert str(error).startswith('autoconfig loading not specified')

    def test_autoconfig_warning_custom(self, init_patch, args, tmp_path, monkeypatch):
        if False:
            return 10
        'Make sure there is no autoconfig warning with --config-py.'
        config_py_path = tmp_path / 'config.py'
        config_py_path.touch()
        args.config_py = str(config_py_path)
        monkeypatch.setattr(configinit.standarddir, 'config_py', lambda : str(config_py_path))
        configinit.early_init(args)

    def test_custom_non_existing_file(self, init_patch, args, tmp_path, caplog, monkeypatch):
        if False:
            while True:
                i = 10
        "Make sure --config-py with a non-existent file doesn't fall back silently."
        config_py_path = tmp_path / 'config.py'
        assert not config_py_path.exists()
        args.config_py = str(config_py_path)
        monkeypatch.setattr(configinit.standarddir, 'config_py', lambda : str(config_py_path))
        with caplog.at_level(logging.ERROR):
            configinit.early_init(args)
        assert len(configinit._init_errors.errors) == 1
        error = configinit._init_errors.errors[0]
        assert isinstance(error.exception, FileNotFoundError)

    @pytest.mark.parametrize('byte', [b'\x00', b'\xda'])
    def test_state_init_errors(self, init_patch, args, data_tmpdir, byte):
        if False:
            return 10
        state_file = data_tmpdir / 'state'
        state_file.write_binary(byte)
        configinit.early_init(args)
        assert configinit._init_errors.errors

    def test_invalid_change_filter(self, init_patch, args):
        if False:
            i = 10
            return i + 15
        config.change_filter('foobar')
        with pytest.raises(configexc.NoOptionError):
            configinit.early_init(args)

    def test_temp_settings_valid(self, init_patch, args):
        if False:
            i = 10
            return i + 15
        args.temp_settings = [('colors.completion.fg', 'magenta')]
        configinit.early_init(args)
        assert config.instance.get_obj('colors.completion.fg') == 'magenta'

    def test_temp_settings_invalid(self, caplog, init_patch, message_mock, args):
        if False:
            for i in range(10):
                print('nop')
        'Invalid temp settings should show an error.'
        args.temp_settings = [('foo', 'bar')]
        with caplog.at_level(logging.ERROR):
            configinit.early_init(args)
        msg = message_mock.getmsg()
        assert msg.level == usertypes.MessageLevel.error
        assert msg.text == "set: NoOptionError - No option 'foo'"

class TestLateInit:

    @pytest.mark.parametrize('errors', [True, 'fatal', False])
    def test_late_init(self, init_patch, monkeypatch, fake_save_manager, args, mocker, errors):
        if False:
            for i in range(10):
                print('nop')
        configinit.early_init(args)
        if errors:
            err = configexc.ConfigErrorDesc('Error text', Exception('Exception'))
            errs = configexc.ConfigFileErrors('config.py', [err])
            if errors == 'fatal':
                errs.fatal = True
            monkeypatch.setattr(configinit, '_init_errors', errs)
        msgbox_mock = mocker.patch('qutebrowser.config.configinit.msgbox.msgbox', autospec=True)
        exit_mock = mocker.patch('qutebrowser.config.configinit.sys.exit', autospec=True)
        configinit.late_init(fake_save_manager)
        fake_save_manager.add_saveable.assert_any_call('state-config', unittest.mock.ANY)
        fake_save_manager.add_saveable.assert_any_call('yaml-config', unittest.mock.ANY, unittest.mock.ANY)
        if errors:
            assert len(msgbox_mock.call_args_list) == 1
            (_call_posargs, call_kwargs) = msgbox_mock.call_args_list[0]
            text = call_kwargs['text'].strip()
            assert text.startswith('Errors occurred while reading config.py:')
            assert '<b>Error text</b>: Exception' in text
            assert exit_mock.called == (errors == 'fatal')
        else:
            assert not msgbox_mock.called

    @pytest.mark.parametrize('settings, size, family', [([('fonts.default_family', 'Comic Sans MS')], 10, 'Comic Sans MS'), ([('fonts.default_family', 'Comic Sans MS'), ('fonts.default_size', '23pt')], 23, 'Comic Sans MS'), ([('fonts.default_family', 'Comic Sans MS'), ('fonts.keyhint', '12pt default_family')], 12, 'Comic Sans MS'), ([('fonts.default_family', 'Comic Sans MS'), ('fonts.default_size', '23pt'), ('fonts.keyhint', 'default_size default_family')], 23, 'Comic Sans MS')])
    @pytest.mark.parametrize('method', ['temp', 'auto', 'py'])
    def test_fonts_defaults_init(self, init_patch, args, config_tmpdir, fake_save_manager, method, settings, size, family):
        if False:
            print('Hello World!')
        'Ensure setting fonts.default_family at init works properly.\n\n        See https://github.com/qutebrowser/qutebrowser/issues/2973\n        and https://github.com/qutebrowser/qutebrowser/issues/5223\n        '
        if method == 'temp':
            args.temp_settings = settings
        elif method == 'auto':
            autoconfig_file = config_tmpdir / 'autoconfig.yml'
            lines = ['config_version: 2', 'settings:'] + ["  {}:\n    global:\n      '{}'".format(k, v) for (k, v) in settings]
            autoconfig_file.write_text('\n'.join(lines), 'utf-8', ensure=True)
        elif method == 'py':
            config_py_file = config_tmpdir / 'config.py'
            lines = ["c.{} = '{}'".format(k, v) for (k, v) in settings]
            lines.append('config.load_autoconfig(False)')
            config_py_file.write_text('\n'.join(lines), 'utf-8', ensure=True)
        configinit.early_init(args)
        configinit.late_init(fake_save_manager)
        expected = '{}pt "{}"'.format(size, family)
        assert config.instance.get('fonts.keyhint') == expected

    @pytest.fixture
    def run_configinit(self, init_patch, fake_save_manager, args):
        if False:
            for i in range(10):
                print('nop')
        'Run configinit.early_init() and .late_init().'
        configinit.early_init(args)
        configinit.late_init(fake_save_manager)

    def test_fonts_defaults_later(self, run_configinit):
        if False:
            i = 10
            return i + 15
        'Ensure setting fonts.default_family/size after init works properly.\n\n        See https://github.com/qutebrowser/qutebrowser/issues/2973\n        '
        changed_options = []
        config.instance.changed.connect(changed_options.append)
        config.instance.set_obj('fonts.default_family', 'Comic Sans MS')
        config.instance.set_obj('fonts.default_size', '23pt')
        assert 'fonts.keyhint' in changed_options
        assert config.instance.get('fonts.keyhint') == '23pt "Comic Sans MS"'
        assert 'fonts.web.family.standard' not in changed_options

    def test_setting_fonts_defaults_family(self, run_configinit):
        if False:
            i = 10
            return i + 15
        'Make sure setting fonts.default_family/size after a family works.\n\n        See https://github.com/qutebrowser/qutebrowser/issues/3130\n        '
        config.instance.set_str('fonts.web.family.standard', '')
        config.instance.set_str('fonts.default_family', 'Terminus')
        config.instance.set_str('fonts.default_size', '10pt')

    def test_default_size_hints(self, run_configinit):
        if False:
            while True:
                i = 10
        'Make sure default_size applies to the hints font.\n\n        See https://github.com/qutebrowser/qutebrowser/issues/5214\n        '
        config.instance.set_obj('fonts.default_family', 'SomeFamily')
        config.instance.set_obj('fonts.default_size', '23pt')
        assert config.instance.get('fonts.hints') == 'bold 23pt SomeFamily'

    def test_default_size_hints_changed(self, run_configinit):
        if False:
            for i in range(10):
                print('nop')
        config.instance.set_obj('fonts.hints', 'bold default_size SomeFamily')
        changed_options = []
        config.instance.changed.connect(changed_options.append)
        config.instance.set_obj('fonts.default_size', '23pt')
        assert config.instance.get('fonts.hints') == 'bold 23pt SomeFamily'
        assert 'fonts.hints' in changed_options

@pytest.mark.parametrize('arg, confval, used', [('webkit', 'webengine', usertypes.Backend.QtWebKit), (None, 'webkit', usertypes.Backend.QtWebKit)])
def test_get_backend(monkeypatch, args, config_stub, arg, confval, used):
    if False:
        i = 10
        return i + 15
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if False:
            return 10
        if name != 'qutebrowser.qt.webkit':
            return real_import(name, *args, **kwargs)
        raise ImportError
    args.backend = arg
    config_stub.val.backend = confval
    monkeypatch.setattr(builtins, '__import__', fake_import)
    assert configinit.get_backend(args) == used