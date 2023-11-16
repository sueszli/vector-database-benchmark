"""Tests for qutebrowser.config.configcommands."""
import logging
import functools
import unittest.mock
import pytest
from qutebrowser.qt.core import QUrl
from qutebrowser.config import configcommands
from qutebrowser.api import cmdutils
from qutebrowser.utils import usertypes, urlmatch, utils
from qutebrowser.keyinput import keyutils
from qutebrowser.misc import objects

def keyseq(s):
    if False:
        print('Hello World!')
    return keyutils.KeySequence.parse(s)

@pytest.fixture
def commands(config_stub, key_config_stub):
    if False:
        return 10
    return configcommands.ConfigCommands(config_stub, key_config_stub)

@pytest.fixture
def yaml_value(config_stub):
    if False:
        i = 10
        return i + 15
    'Fixture which provides a getter for a YAML value.'

    def getter(option):
        if False:
            print('Hello World!')
        return config_stub._yaml._values[option].get_for_url(fallback=False)
    return getter

class TestSet:
    """Tests for :set."""

    def test_set_no_args(self, commands, tabbed_browser_stubs):
        if False:
            for i in range(10):
                print('nop')
        "Run ':set'.\n\n        Should open qute://settings."
        commands.set(win_id=0)
        assert tabbed_browser_stubs[0].loaded_url == QUrl('qute://settings')

    @pytest.mark.parametrize('option', ['url.auto_search?', 'url.auto_search'])
    def test_get(self, config_stub, commands, message_mock, option):
        if False:
            print('Hello World!')
        "Run ':set url.auto_search?' / ':set url.auto_search'.\n\n        Should show the value.\n        "
        config_stub.val.url.auto_search = 'never'
        commands.set(win_id=0, option=option)
        msg = message_mock.getmsg(usertypes.MessageLevel.info)
        assert msg.text == 'url.auto_search = never'

    @pytest.mark.parametrize('temp', [True, False])
    @pytest.mark.parametrize('option, old_value, inp, new_value', [('url.auto_search', 'naive', 'dns', 'dns'), ('editor.command', ['gvim', '-f', '{file}', '-c', 'normal {line}G{column0}l'], '[emacs, "{}"]', ['emacs', '{}'])])
    def test_set_simple(self, monkeypatch, commands, config_stub, yaml_value, temp, option, old_value, inp, new_value):
        if False:
            while True:
                i = 10
        "Run ':set [-t] option value'.\n\n        Should set the setting accordingly.\n        "
        monkeypatch.setattr(objects, 'backend', usertypes.Backend.QtWebKit)
        assert config_stub.get(option) == old_value
        commands.set(0, option, inp, temp=temp)
        assert config_stub.get(option) == new_value
        assert yaml_value(option) == (usertypes.UNSET if temp else new_value)

    def test_set_with_pattern(self, monkeypatch, commands, config_stub):
        if False:
            for i in range(10):
                print('nop')
        monkeypatch.setattr(objects, 'backend', usertypes.Backend.QtWebKit)
        option = 'content.javascript.enabled'
        commands.set(0, option, 'false', pattern='*://example.com')
        pattern = urlmatch.UrlPattern('*://example.com')
        assert config_stub.get(option)
        assert not config_stub.get_obj_for_pattern(option, pattern=pattern)

    def test_set_invalid_pattern(self, monkeypatch, commands):
        if False:
            while True:
                i = 10
        monkeypatch.setattr(objects, 'backend', usertypes.Backend.QtWebKit)
        option = 'content.javascript.enabled'
        with pytest.raises(cmdutils.CommandError, match='Error while parsing http://: Pattern without host'):
            commands.set(0, option, 'false', pattern='http://')

    def test_set_no_pattern(self, monkeypatch, commands):
        if False:
            print('Hello World!')
        "Run ':set --pattern=*://* colors.statusbar.normal.bg #abcdef.\n\n        Should show an error as patterns are unsupported.\n        "
        with pytest.raises(cmdutils.CommandError, match='does not support URL patterns'):
            commands.set(0, 'colors.statusbar.normal.bg', '#abcdef', pattern='*://*')

    @pytest.mark.parametrize('temp', [True, False])
    def test_set_temp_override(self, commands, config_stub, yaml_value, temp):
        if False:
            print('Hello World!')
        'Invoking :set twice.\n\n        :set url.auto_search dns\n        :set -t url.auto_search never\n\n        Should set the setting accordingly.\n        '
        assert config_stub.val.url.auto_search == 'naive'
        commands.set(0, 'url.auto_search', 'dns')
        commands.set(0, 'url.auto_search', 'never', temp=True)
        assert config_stub.val.url.auto_search == 'never'
        assert yaml_value('url.auto_search') == 'dns'

    @pytest.mark.parametrize('pattern', [None, '*://example.com'])
    def test_set_print(self, config_stub, commands, message_mock, pattern):
        if False:
            i = 10
            return i + 15
        "Run ':set -p [-u *://example.com] content.javascript.enabled false'.\n\n        Should set show the value.\n        "
        assert config_stub.val.content.javascript.enabled
        commands.set(0, 'content.javascript.enabled', 'false', print_=True, pattern=pattern)
        value = config_stub.get_obj_for_pattern('content.javascript.enabled', pattern=None if pattern is None else urlmatch.UrlPattern(pattern))
        assert not value
        expected = 'content.javascript.enabled = false'
        if pattern is not None:
            expected += ' for {}'.format(pattern)
        msg = message_mock.getmsg(usertypes.MessageLevel.info)
        assert msg.text == expected

    def test_set_invalid_option(self, commands):
        if False:
            while True:
                i = 10
        "Run ':set foo bar'.\n\n        Should show an error.\n        "
        with pytest.raises(cmdutils.CommandError, match="No option 'foo'"):
            commands.set(0, 'foo', 'bar')

    def test_set_invalid_value(self, commands):
        if False:
            i = 10
            return i + 15
        "Run ':set auto_save.session blah'.\n\n        Should show an error.\n        "
        with pytest.raises(cmdutils.CommandError, match="Invalid value 'blah' - must be a boolean!"):
            commands.set(0, 'auto_save.session', 'blah')

    def test_set_wrong_backend(self, commands, monkeypatch):
        if False:
            return 10
        monkeypatch.setattr(objects, 'backend', usertypes.Backend.QtWebEngine)
        with pytest.raises(cmdutils.CommandError, match='The hints.find_implementation setting is not available with the QtWebEngine backend!'):
            commands.set(0, 'hints.find_implementation', 'javascript')

    def test_empty(self, commands):
        if False:
            return 10
        "Run ':set ?'.\n\n        Should show an error.\n        See https://github.com/qutebrowser/qutebrowser/issues/1109\n        "
        with pytest.raises(cmdutils.CommandError, match="No option '?'"):
            commands.set(win_id=0, option='?')

    def test_toggle(self, commands):
        if False:
            print('Hello World!')
        'Try toggling a value.\n\n        Should show an nicer error.\n        '
        with pytest.raises(cmdutils.CommandError, match='Toggling values was moved to the :config-cycle command'):
            commands.set(win_id=0, option='javascript.enabled!')

    def test_invalid(self, commands):
        if False:
            for i in range(10):
                print('nop')
        "Run ':set foo?'.\n\n        Should show an error.\n        "
        with pytest.raises(cmdutils.CommandError, match="No option 'foo'"):
            commands.set(win_id=0, option='foo?')

@pytest.mark.parametrize('include_hidden, url', [(True, 'qute://configdiff?include_hidden=true'), (False, 'qute://configdiff')])
def test_diff(commands, tabbed_browser_stubs, include_hidden, url):
    if False:
        while True:
            i = 10
    "Run ':config-diff'.\n\n    Should open qute://configdiff.\n    "
    commands.config_diff(win_id=0, include_hidden=include_hidden)
    assert tabbed_browser_stubs[0].loaded_url == QUrl(url)

class TestCycle:
    """Test :config-cycle."""

    @pytest.mark.parametrize('initial, expected', [('magenta', 'blue'), ('yellow', 'green'), ('red', 'green')])
    def test_cycling(self, commands, config_stub, yaml_value, initial, expected):
        if False:
            return 10
        "Run ':set' with multiple values."
        opt = 'colors.statusbar.normal.bg'
        config_stub.set_obj(opt, initial)
        commands.config_cycle(opt, 'green', 'magenta', 'blue', 'yellow')
        assert config_stub.get(opt) == expected
        assert yaml_value(opt) == expected

    def test_different_representation(self, commands, config_stub):
        if False:
            return 10
        'When using a different representation, cycling should work.\n\n        For example, we use [foo] which is represented as ["foo"].\n        '
        opt = 'qt.args'
        config_stub.set_obj(opt, ['foo'])
        commands.config_cycle(opt, '[foo]', '[bar]')
        assert config_stub.get(opt) == ['bar']
        commands.config_cycle(opt, '[foo]', '[bar]')
        assert config_stub.get(opt) == ['foo']

    def test_toggle(self, commands, config_stub, yaml_value):
        if False:
            return 10
        "Run ':config-cycle auto_save.session'.\n\n        Should toggle the value.\n        "
        assert not config_stub.val.auto_save.session
        commands.config_cycle('auto_save.session')
        assert config_stub.val.auto_save.session
        assert yaml_value('auto_save.session')

    @pytest.mark.parametrize('args', [['url.auto_search'], ['url.auto_search', 'foo']])
    def test_toggle_nonbool(self, commands, config_stub, args):
        if False:
            print('Hello World!')
        'Run :config-cycle without a bool and 0/1 value.\n\n        :config-cycle url.auto_search\n        :config-cycle url.auto_search foo\n\n        Should show an error.\n        '
        assert config_stub.val.url.auto_search == 'naive'
        with pytest.raises(cmdutils.CommandError, match='Need at least two values for non-boolean settings.'):
            commands.config_cycle(*args)
        assert config_stub.val.url.auto_search == 'naive'

    def test_set_toggle_print(self, commands, config_stub, message_mock):
        if False:
            while True:
                i = 10
        "Run ':config-cycle -p auto_save.session'.\n\n        Should toggle the value and show the new value.\n        "
        commands.config_cycle('auto_save.session', print_=True)
        msg = message_mock.getmsg(usertypes.MessageLevel.info)
        assert msg.text == 'auto_save.session = true'

class TestAdd:
    """Test :config-list-add and :config-dict-add."""

    @pytest.mark.parametrize('temp', [True, False])
    @pytest.mark.parametrize('value', ['test1', 'test2'])
    def test_list_add(self, commands, config_stub, yaml_value, temp, value):
        if False:
            for i in range(10):
                print('nop')
        name = 'content.blocking.whitelist'
        commands.config_list_add(name, value, temp=temp)
        assert str(config_stub.get(name)[-1]) == value
        if temp:
            assert yaml_value(name) == usertypes.UNSET
        else:
            assert yaml_value(name)[-1] == value

    def test_list_add_invalid_option(self, commands):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(cmdutils.CommandError, match="No option 'nonexistent'"):
            commands.config_list_add('nonexistent', 'value')

    def test_list_add_non_list(self, commands):
        if False:
            return 10
        with pytest.raises(cmdutils.CommandError, match=':config-list-add can only be used for lists'):
            commands.config_list_add('history_gap_interval', 'value')

    def test_list_add_invalid_value(self, commands):
        if False:
            print('Hello World!')
        with pytest.raises(cmdutils.CommandError, match="Invalid value ''"):
            commands.config_list_add('content.blocking.whitelist', '')

    @pytest.mark.parametrize('value', ['test1', 'test2'])
    @pytest.mark.parametrize('temp', [True, False])
    def test_dict_add(self, commands, config_stub, yaml_value, value, temp):
        if False:
            i = 10
            return i + 15
        name = 'aliases'
        key = 'missingkey'
        commands.config_dict_add(name, key, value, temp=temp)
        assert str(config_stub.get(name)[key]) == value
        if temp:
            assert yaml_value(name) == usertypes.UNSET
        else:
            assert yaml_value(name)[key] == value

    @pytest.mark.parametrize('replace', [True, False])
    def test_dict_add_replace(self, commands, config_stub, replace):
        if False:
            while True:
                i = 10
        name = 'aliases'
        key = 'w'
        value = 'anything'
        if replace:
            commands.config_dict_add(name, key, value, replace=True)
            assert str(config_stub.get(name)[key]) == value
        else:
            with pytest.raises(cmdutils.CommandError, match='w already exists in aliases - use --replace to overwrite!'):
                commands.config_dict_add(name, key, value, replace=False)

    def test_dict_add_invalid_option(self, commands):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(cmdutils.CommandError, match="No option 'nonexistent'"):
            commands.config_dict_add('nonexistent', 'key', 'value')

    def test_dict_add_non_dict(self, commands):
        if False:
            print('Hello World!')
        with pytest.raises(cmdutils.CommandError, match=':config-dict-add can only be used for dicts'):
            commands.config_dict_add('history_gap_interval', 'key', 'value')

    def test_dict_add_invalid_value(self, commands):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(cmdutils.CommandError, match="Invalid value ''"):
            commands.config_dict_add('aliases', 'missingkey', '')

    def test_dict_add_value_type(self, commands, config_stub):
        if False:
            i = 10
            return i + 15
        commands.config_dict_add('content.javascript.log_message.levels', 'example', "['error']")
        value = config_stub.val.content.javascript.log_message.levels['example']
        assert value == ['error']

class TestRemove:
    """Test :config-list-remove and :config-dict-remove."""

    @pytest.mark.parametrize('value', ['25%', '50%'])
    @pytest.mark.parametrize('temp', [True, False])
    def test_list_remove(self, commands, config_stub, yaml_value, value, temp):
        if False:
            print('Hello World!')
        name = 'zoom.levels'
        commands.config_list_remove(name, value, temp=temp)
        assert value not in config_stub.get(name)
        if temp:
            assert yaml_value(name) == usertypes.UNSET
        else:
            assert value not in yaml_value(name)

    def test_list_remove_invalid_option(self, commands):
        if False:
            print('Hello World!')
        with pytest.raises(cmdutils.CommandError, match="No option 'nonexistent'"):
            commands.config_list_remove('nonexistent', 'value')

    def test_list_remove_non_list(self, commands):
        if False:
            while True:
                i = 10
        with pytest.raises(cmdutils.CommandError, match=':config-list-remove can only be used for lists'):
            commands.config_list_remove('content.javascript.enabled', 'never')

    def test_list_remove_no_value(self, commands):
        if False:
            while True:
                i = 10
        with pytest.raises(cmdutils.CommandError, match='#133742 is not in colors.completion.fg!'):
            commands.config_list_remove('colors.completion.fg', '#133742')

    @pytest.mark.parametrize('key', ['w', 'q'])
    @pytest.mark.parametrize('temp', [True, False])
    def test_dict_remove(self, commands, config_stub, yaml_value, key, temp):
        if False:
            while True:
                i = 10
        name = 'aliases'
        commands.config_dict_remove(name, key, temp=temp)
        assert key not in config_stub.get(name)
        if temp:
            assert yaml_value(name) == usertypes.UNSET
        else:
            assert key not in yaml_value(name)

    def test_dict_remove_invalid_option(self, commands):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(cmdutils.CommandError, match="No option 'nonexistent'"):
            commands.config_dict_remove('nonexistent', 'key')

    def test_dict_remove_non_dict(self, commands):
        if False:
            while True:
                i = 10
        with pytest.raises(cmdutils.CommandError, match=':config-dict-remove can only be used for dicts'):
            commands.config_dict_remove('content.javascript.enabled', 'never')

    def test_dict_remove_no_value(self, commands):
        if False:
            return 10
        with pytest.raises(cmdutils.CommandError, match='never is not in aliases!'):
            commands.config_dict_remove('aliases', 'never')

class TestUnsetAndClear:
    """Test :config-unset and :config-clear."""

    @pytest.mark.parametrize('temp', [True, False])
    def test_unset(self, commands, config_stub, yaml_value, temp):
        if False:
            return 10
        name = 'tabs.show'
        config_stub.set_obj(name, 'never', save_yaml=True)
        commands.config_unset(name, temp=temp)
        assert config_stub.get(name) == 'always'
        assert yaml_value(name) == ('never' if temp else usertypes.UNSET)

    def test_unset_unknown_option(self, commands):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(cmdutils.CommandError, match="No option 'tabs'"):
            commands.config_unset('tabs')

    def test_unset_uncustomized(self, commands):
        if False:
            while True:
                i = 10
        with pytest.raises(cmdutils.CommandError, match='tabs.show is not customized'):
            commands.config_unset('tabs.show')

    @pytest.mark.parametrize('set_global', [True, False])
    def test_unset_pattern(self, commands, config_stub, set_global):
        if False:
            return 10
        name = 'content.javascript.enabled'
        pattern = urlmatch.UrlPattern('*://example.com')
        url = QUrl('https://example.com')
        if set_global:
            config_stub.set_obj(name, False)
            global_value = False
            local_value = True
        else:
            global_value = True
            local_value = False
        config_stub.set_obj(name, local_value, pattern=pattern)
        commands.config_unset(name, pattern=str(pattern))
        assert config_stub.get_obj(name, url=url) == global_value
        assert config_stub.get_obj(name, url=url, fallback=False) == usertypes.UNSET

    def test_unset_uncustomized_pattern(self, commands, config_stub):
        if False:
            print('Hello World!')
        name = 'content.javascript.enabled'
        pattern = 'example.com'
        config_stub.set_obj(name, False)
        with pytest.raises(cmdutils.CommandError, match=f'{name} is not customized for {pattern}'):
            commands.config_unset(name, pattern=pattern)

    @pytest.mark.parametrize('save', [True, False])
    def test_clear(self, commands, config_stub, yaml_value, save):
        if False:
            i = 10
            return i + 15
        name = 'tabs.show'
        config_stub.set_obj(name, 'never', save_yaml=True)
        commands.config_clear(save=save)
        assert config_stub.get(name) == 'always'
        assert yaml_value(name) == (usertypes.UNSET if save else 'never')

@pytest.mark.usefixtures('config_tmpdir', 'data_tmpdir', 'config_stub', 'key_config_stub')
class TestSource:
    """Test :config-source."""

    @pytest.mark.parametrize('location', ['default', 'absolute', 'relative'])
    @pytest.mark.parametrize('clear', [True, False])
    def test_config_source(self, tmp_path, commands, config_stub, config_tmpdir, location, clear):
        if False:
            while True:
                i = 10
        assert config_stub.val.content.javascript.enabled
        config_stub.val.search.ignore_case = 'always'
        if location == 'default':
            pyfile = config_tmpdir / 'config.py'
            arg = None
        elif location == 'absolute':
            pyfile = tmp_path / 'sourced.py'
            arg = str(pyfile)
        elif location == 'relative':
            pyfile = config_tmpdir / 'sourced.py'
            arg = 'sourced.py'
        else:
            raise utils.Unreachable(location)
        pyfile.write_text('\n'.join(['config.load_autoconfig(False)', 'c.content.javascript.enabled = False']), encoding='utf-8')
        commands.config_source(arg, clear=clear)
        assert not config_stub.val.content.javascript.enabled
        ignore_case = config_stub.val.search.ignore_case
        assert ignore_case == (usertypes.IgnoreCase.smart if clear else usertypes.IgnoreCase.always)

    def test_config_py_arg_source(self, commands, config_py_arg, config_stub):
        if False:
            while True:
                i = 10
        assert config_stub.val.content.javascript.enabled
        config_py_arg.write_text('\n'.join(['config.load_autoconfig(False)', 'c.content.javascript.enabled = False']), encoding='utf-8')
        commands.config_source()
        assert not config_stub.val.content.javascript.enabled

    def test_errors(self, commands, config_tmpdir):
        if False:
            for i in range(10):
                print('nop')
        pyfile = config_tmpdir / 'config.py'
        pyfile.write_text('\n'.join(['config.load_autoconfig(False)', 'c.foo = 42']), encoding='utf-8')
        with pytest.raises(cmdutils.CommandError) as excinfo:
            commands.config_source()
        expected = "Errors occurred while reading config.py:\n  While setting 'foo': No option 'foo'"
        assert str(excinfo.value) == expected

    def test_invalid_source(self, commands, config_tmpdir):
        if False:
            return 10
        pyfile = config_tmpdir / 'config.py'
        pyfile.write_text('\n'.join(['config.load_autoconfig(False)', '1/0']), encoding='utf-8')
        with pytest.raises(cmdutils.CommandError) as excinfo:
            commands.config_source()
        expected = 'Errors occurred while reading config.py:\n  Unhandled exception - ZeroDivisionError: division by zero'
        assert str(excinfo.value) == expected

    def test_invalid_mutable(self, commands, config_tmpdir):
        if False:
            return 10
        pyfile = config_tmpdir / 'config.py'
        src = 'c.url.searchengines["maps"] = "https://www.google.com/maps?q=%s"'
        pyfile.write_text(src, encoding='utf-8')
        with pytest.raises(cmdutils.CommandError) as excinfo:
            commands.config_source()
        err = 'Invalid value \'https://www.google.com/maps?q=%s\' - must contain "{}"'
        expected = f'Errors occurred while reading config.py:\n  While updating mutated values: {err}'
        assert str(excinfo.value) == expected

@pytest.mark.usefixtures('config_tmpdir', 'data_tmpdir', 'config_stub', 'key_config_stub', 'qapp')
class TestEdit:
    """Tests for :config-edit."""

    def test_no_source(self, commands, mocker):
        if False:
            return 10
        mock = mocker.patch('qutebrowser.config.configcommands.editor.ExternalEditor._start_editor', autospec=True)
        commands.config_edit(no_source=True)
        mock.assert_called_once_with(unittest.mock.ANY)

    @pytest.fixture
    def patch_editor(self, mocker):
        if False:
            i = 10
            return i + 15
        'Write a config.py file.'

        def do_patch(text):
            if False:
                for i in range(10):
                    print('nop')

            def _write_file(editor_self):
                if False:
                    while True:
                        i = 10
                with open(editor_self._filename, 'w', encoding='utf-8') as f:
                    f.write(text)
                editor_self.file_updated.emit(text)
            return mocker.patch('qutebrowser.config.configcommands.editor.ExternalEditor._start_editor', autospec=True, side_effect=_write_file)
        return do_patch

    def test_with_sourcing(self, commands, config_stub, patch_editor):
        if False:
            for i in range(10):
                print('nop')
        assert config_stub.val.content.javascript.enabled
        mock = patch_editor('\n'.join(['config.load_autoconfig(False)', 'c.content.javascript.enabled = False']))
        commands.config_edit()
        mock.assert_called_once_with(unittest.mock.ANY)
        assert not config_stub.val.content.javascript.enabled

    def test_config_py_with_sourcing(self, commands, config_stub, patch_editor, config_py_arg):
        if False:
            for i in range(10):
                print('nop')
        assert config_stub.val.content.javascript.enabled
        conf = ['config.load_autoconfig(False)', 'c.content.javascript.enabled = False']
        mock = patch_editor('\n'.join(conf))
        commands.config_edit()
        mock.assert_called_once_with(unittest.mock.ANY)
        assert not config_stub.val.content.javascript.enabled
        assert config_py_arg.read_text('utf-8').splitlines() == conf

    def test_error(self, commands, config_stub, patch_editor, message_mock, caplog):
        if False:
            while True:
                i = 10
        patch_editor('\n'.join(['config.load_autoconfig(False)', 'c.foo = 42']))
        with caplog.at_level(logging.ERROR):
            commands.config_edit()
        msg = message_mock.getmsg()
        expected = "Errors occurred while reading config.py:\n  While setting 'foo': No option 'foo'"
        assert msg.text == expected

class TestWritePy:
    """Tests for :config-write-py."""

    def test_custom(self, commands, config_stub, key_config_stub, tmp_path):
        if False:
            print('Hello World!')
        confpy = tmp_path / 'config.py'
        config_stub.val.content.javascript.enabled = True
        key_config_stub.bind(keyseq(',x'), 'message-info foo', mode='normal')
        commands.config_write_py(str(confpy))
        lines = confpy.read_text('utf-8').splitlines()
        assert 'c.content.javascript.enabled = True' in lines
        assert "config.bind(',x', 'message-info foo')" in lines

    def test_defaults(self, commands, tmp_path):
        if False:
            print('Hello World!')
        confpy = tmp_path / 'config.py'
        commands.config_write_py(str(confpy), defaults=True)
        lines = confpy.read_text('utf-8').splitlines()
        assert '# c.content.javascript.enabled = True' in lines
        assert "# config.bind('H', 'back')" in lines

    def test_default_location(self, commands, config_tmpdir):
        if False:
            i = 10
            return i + 15
        confpy = config_tmpdir / 'config.py'
        commands.config_write_py()
        lines = confpy.read_text('utf-8').splitlines()
        assert '# Autogenerated config.py' in lines

    def test_relative_path(self, commands, config_tmpdir):
        if False:
            return 10
        confpy = config_tmpdir / 'config2.py'
        commands.config_write_py('config2.py')
        lines = confpy.read_text('utf-8').splitlines()
        assert '# Autogenerated config.py' in lines

    @pytest.mark.posix
    def test_expanduser(self, commands, monkeypatch, tmp_path):
        if False:
            while True:
                i = 10
        'Make sure that using a path with ~/... works correctly.'
        home = tmp_path / 'home'
        home.mkdir()
        monkeypatch.setenv('HOME', str(home))
        commands.config_write_py('~/config.py')
        confpy = home / 'config.py'
        lines = confpy.read_text('utf-8').splitlines()
        assert '# Autogenerated config.py' in lines

    def test_existing_file(self, commands, tmp_path):
        if False:
            while True:
                i = 10
        confpy = tmp_path / 'config.py'
        confpy.touch()
        with pytest.raises(cmdutils.CommandError) as excinfo:
            commands.config_write_py(str(confpy))
        expected = ' already exists - use --force to overwrite!'
        assert str(excinfo.value).endswith(expected)

    def test_existing_file_force(self, commands, tmp_path):
        if False:
            print('Hello World!')
        confpy = tmp_path / 'config.py'
        confpy.touch()
        commands.config_write_py(str(confpy), force=True)
        lines = confpy.read_text('utf-8').splitlines()
        assert '# Autogenerated config.py' in lines

    def test_oserror(self, commands, tmp_path):
        if False:
            return 10
        'Test writing to a directory which does not exist.'
        with pytest.raises(cmdutils.CommandError):
            commands.config_write_py(str(tmp_path / 'foo' / 'config.py'))

    def test_config_py_arg(self, commands, config_py_arg):
        if False:
            print('Hello World!')
        config_py_arg.ensure()
        commands.config_write_py(str(config_py_arg), force=True)
        lines = config_py_arg.read_text('utf-8').splitlines()
        assert '# Autogenerated config.py' in lines

class TestBind:
    """Tests for :bind and :unbind."""

    @pytest.fixture
    def no_bindings(self):
        if False:
            for i in range(10):
                print('nop')
        'Get a dict with no bindings.'
        return {'normal': {}}

    @pytest.mark.parametrize('mode, url', [('normal', QUrl('qute://bindings')), ('passthrough', QUrl('qute://bindings#passthrough'))])
    def test_bind_no_args(self, commands, config_stub, no_bindings, tabbed_browser_stubs, mode, url):
        if False:
            print('Hello World!')
        "Run ':bind'.\n\n        Should open qute://bindings."
        config_stub.val.bindings.default = no_bindings
        config_stub.val.bindings.commands = no_bindings
        commands.bind(win_id=0, mode=mode)
        assert tabbed_browser_stubs[0].loaded_url == url

    @pytest.mark.parametrize('command', ['nop', 'nope'])
    def test_bind(self, commands, config_stub, no_bindings, key_config_stub, yaml_value, command):
        if False:
            return 10
        'Simple :bind test (and aliases).'
        config_stub.val.aliases = {'nope': 'nop'}
        config_stub.val.bindings.default = no_bindings
        config_stub.val.bindings.commands = no_bindings
        commands.bind(0, 'a', command)
        assert key_config_stub.get_command(keyseq('a'), 'normal') == command
        yaml_bindings = yaml_value('bindings.commands')['normal']
        assert yaml_bindings['a'] == command

    @pytest.mark.parametrize('key, mode, expected', [('a', 'normal', "a is bound to 'message-info a' in normal mode"), ('b', 'normal', "b is bound to 'mib' in normal mode"), ('c', 'normal', "c is bound to 'message-info c' in normal mode"), ('<Ctrl-X>', 'normal', "<Ctrl+x> is bound to 'message-info C-x' in normal mode"), ('x', 'normal', 'x is unbound in normal mode'), ('x', 'caret', "x is bound to 'nop' in caret mode")])
    def test_bind_print(self, commands, config_stub, message_mock, key, mode, expected):
        if False:
            i = 10
            return i + 15
        "Run ':bind key'.\n\n        Should print the binding.\n        "
        config_stub.val.aliases = {'mib': 'message-info b'}
        config_stub.val.bindings.default = {'normal': {'a': 'message-info a', 'b': 'mib', '<Ctrl+x>': 'message-info C-x'}, 'caret': {'x': 'nop'}}
        config_stub.val.bindings.commands = {'normal': {'c': 'message-info c'}}
        commands.bind(0, key, mode=mode)
        msg = message_mock.getmsg(usertypes.MessageLevel.info)
        assert msg.text == expected

    @pytest.mark.parametrize('command, args, kwargs, expected', [('bind', ['a', 'nop'], {'mode': 'wrongmode'}, 'Invalid mode wrongmode!'), ('bind', ['a'], {'mode': 'wrongmode'}, 'Invalid mode wrongmode!'), ('bind', ['a'], {'mode': 'wrongmode', 'default': True}, 'Invalid mode wrongmode!'), ('bind', ['foobar'], {'default': True}, "Can't find binding 'foobar' in normal mode"), ('bind', ['<blub>', 'nop'], {}, "Could not parse '<blub>': Got invalid key!"), ('unbind', ['foobar'], {}, "Can't find binding 'foobar' in normal mode"), ('unbind', ['x'], {'mode': 'wrongmode'}, 'Invalid mode wrongmode!'), ('unbind', ['<blub>'], {}, "Could not parse '<blub>': Got invalid key!")])
    def test_bind_invalid(self, commands, command, args, kwargs, expected):
        if False:
            for i in range(10):
                print('nop')
        'Run various wrong :bind/:unbind invocations.\n\n        Should show an error.\n        '
        if command == 'bind':
            func = functools.partial(commands.bind, 0)
        elif command == 'unbind':
            func = commands.unbind
        with pytest.raises(cmdutils.CommandError, match=expected):
            func(*args, **kwargs)

    @pytest.mark.parametrize('key', ['a', 'b', '<Ctrl-X>'])
    def test_bind_duplicate(self, commands, config_stub, key_config_stub, key):
        if False:
            print('Hello World!')
        "Run ':bind' with a key which already has been bound.'.\n\n        Also tests for https://github.com/qutebrowser/qutebrowser/issues/1544\n        "
        config_stub.val.bindings.default = {'normal': {'a': 'nop', '<Ctrl+x>': 'nop'}}
        config_stub.val.bindings.commands = {'normal': {'b': 'nop'}}
        commands.bind(0, key, 'message-info foo', mode='normal')
        command = key_config_stub.get_command(keyseq(key), 'normal')
        assert command == 'message-info foo'

    def test_bind_none(self, commands, config_stub):
        if False:
            for i in range(10):
                print('nop')
        config_stub.val.bindings.commands = None
        commands.bind(0, ',x', 'nop')

    def test_bind_default(self, commands, key_config_stub, config_stub):
        if False:
            i = 10
            return i + 15
        'Bind a key to its default.'
        default_cmd = 'message-info default'
        bound_cmd = 'message-info bound'
        config_stub.val.bindings.default = {'normal': {'a': default_cmd}}
        config_stub.val.bindings.commands = {'normal': {'a': bound_cmd}}
        command = key_config_stub.get_command(keyseq('a'), mode='normal')
        assert command == bound_cmd
        commands.bind(0, 'a', mode='normal', default=True)
        command = key_config_stub.get_command(keyseq('a'), mode='normal')
        assert command == default_cmd

    def test_unbind_none(self, commands, config_stub):
        if False:
            for i in range(10):
                print('nop')
        config_stub.val.bindings.commands = None
        commands.unbind('H')

    @pytest.mark.parametrize('key, normalized', [('a', 'a'), ('b', 'b'), ('c', 'c'), ('<Ctrl-X>', '<Ctrl+x>')])
    def test_unbind(self, commands, key_config_stub, config_stub, yaml_value, key, normalized):
        if False:
            return 10
        config_stub.val.bindings.default = {'normal': {'a': 'nop', '<ctrl+x>': 'nop'}, 'caret': {'a': 'nop', '<ctrl+x>': 'nop'}}
        config_stub.val.bindings.commands = {'normal': {'b': 'nop'}, 'caret': {'b': 'nop'}}
        if key == 'c':
            commands.bind(0, key, 'nop')
        commands.unbind(key)
        assert key_config_stub.get_command(keyseq(key), 'normal') is None
        yaml_bindings = yaml_value('bindings.commands')['normal']
        if key in 'bc':
            assert normalized not in yaml_bindings
        else:
            assert yaml_bindings[normalized] is None