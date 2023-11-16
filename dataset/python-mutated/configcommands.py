"""Commands related to the configuration."""
import os.path
import contextlib
from typing import TYPE_CHECKING, Iterator, List, Optional, Any, Tuple
from qutebrowser.qt.core import QUrl, QUrlQuery
from qutebrowser.api import cmdutils
from qutebrowser.completion.models import configmodel
from qutebrowser.utils import objreg, message, standarddir, urlmatch
from qutebrowser.config import configtypes, configexc, configfiles, configdata
from qutebrowser.misc import editor
from qutebrowser.keyinput import keyutils
if TYPE_CHECKING:
    from qutebrowser.config.config import Config, KeyConfig

class ConfigCommands:
    """qutebrowser commands related to the configuration."""

    def __init__(self, config: 'Config', keyconfig: 'KeyConfig') -> None:
        if False:
            print('Hello World!')
        self._config = config
        self._keyconfig = keyconfig

    @contextlib.contextmanager
    def _handle_config_error(self) -> Iterator[None]:
        if False:
            for i in range(10):
                print('nop')
        'Catch errors in set_command and raise CommandError.'
        try:
            yield
        except configexc.Error as e:
            raise cmdutils.CommandError(str(e))

    def _parse_pattern(self, pattern: Optional[str]) -> Optional[urlmatch.UrlPattern]:
        if False:
            return 10
        'Parse a pattern string argument to a pattern.'
        if pattern is None:
            return None
        try:
            return urlmatch.UrlPattern(pattern)
        except urlmatch.ParseError as e:
            raise cmdutils.CommandError('Error while parsing {}: {}'.format(pattern, str(e)))

    def _parse_key(self, key: str) -> keyutils.KeySequence:
        if False:
            for i in range(10):
                print('nop')
        'Parse a key argument.'
        try:
            return keyutils.KeySequence.parse(key)
        except keyutils.KeyParseError as e:
            raise cmdutils.CommandError(str(e))

    def _print_value(self, option: str, pattern: Optional[urlmatch.UrlPattern]) -> None:
        if False:
            while True:
                i = 10
        'Print the value of the given option.'
        with self._handle_config_error():
            value = self._config.get_str(option, pattern=pattern)
        text = '{} = {}'.format(option, value)
        if pattern is not None:
            text += ' for {}'.format(pattern)
        message.info(text)

    @cmdutils.register(instance='config-commands')
    @cmdutils.argument('option', completion=configmodel.option)
    @cmdutils.argument('value', completion=configmodel.value)
    @cmdutils.argument('win_id', value=cmdutils.Value.win_id)
    @cmdutils.argument('pattern', flag='u')
    def set(self, win_id: int, option: str=None, value: str=None, temp: bool=False, print_: bool=False, *, pattern: str=None) -> None:
        if False:
            return 10
        "Set an option.\n\n        If the option name ends with '?' or no value is provided, the\n        value of the option is shown instead.\n\n        Using :set without any arguments opens a page where settings can be\n        changed interactively.\n\n        Args:\n            option: The name of the option.\n            value: The value to set.\n            pattern: The link:configuring{outfilesuffix}#patterns[URL pattern] to use.\n            temp: Set value temporarily until qutebrowser is closed.\n            print_: Print the value after setting.\n        "
        if option is None:
            tabbed_browser = objreg.get('tabbed-browser', scope='window', window=win_id)
            tabbed_browser.load_url(QUrl('qute://settings'), newtab=False)
            return
        if option.endswith('!'):
            raise cmdutils.CommandError('Toggling values was moved to the :config-cycle command')
        parsed_pattern = self._parse_pattern(pattern)
        if option.endswith('?') and option != '?':
            self._print_value(option[:-1], pattern=parsed_pattern)
            return
        with self._handle_config_error():
            if value is None:
                self._print_value(option, pattern=parsed_pattern)
            else:
                self._config.set_str(option, value, pattern=parsed_pattern, save_yaml=not temp)
        if print_:
            self._print_value(option, pattern=parsed_pattern)

    @cmdutils.register(instance='config-commands', maxsplit=1, no_cmd_split=True, no_replace_variables=True)
    @cmdutils.argument('command', completion=configmodel.bind)
    @cmdutils.argument('win_id', value=cmdutils.Value.win_id)
    def bind(self, win_id: str, key: str=None, command: str=None, *, mode: str='normal', default: bool=False) -> None:
        if False:
            while True:
                i = 10
        'Bind a key to a command.\n\n        If no command is given, show the current binding for the given key.\n        Using :bind without any arguments opens a page showing all keybindings.\n\n        Args:\n            key: The keychain to bind. Examples of valid keychains are `gC`,\n                 `<Ctrl-X>` or `<Ctrl-C>a`.\n            command: The command to execute, with optional args.\n            mode: The mode to bind the key in (default: `normal`). See `:help\n                  bindings.commands` for the available modes.\n            default: If given, restore a default binding.\n        '
        if key is None:
            url = QUrl('qute://bindings')
            if mode != 'normal':
                url.setFragment(mode)
            tabbed_browser = objreg.get('tabbed-browser', scope='window', window=win_id)
            tabbed_browser.load_url(url, newtab=True)
            return
        seq = self._parse_key(key)
        if command is None:
            if default:
                with self._handle_config_error():
                    self._keyconfig.bind_default(seq, mode=mode, save_yaml=True)
                return
            with self._handle_config_error():
                cmd = self._keyconfig.get_command(seq, mode)
            if cmd is None:
                message.info('{} is unbound in {} mode'.format(seq, mode))
            else:
                message.info("{} is bound to '{}' in {} mode".format(seq, cmd, mode))
            return
        with self._handle_config_error():
            self._keyconfig.bind(seq, command, mode=mode, save_yaml=True)

    @cmdutils.register(instance='config-commands')
    def unbind(self, key: str, *, mode: str='normal') -> None:
        if False:
            i = 10
            return i + 15
        'Unbind a keychain.\n\n        Args:\n            key: The keychain to unbind. See the help for `:bind` for the\n                  correct syntax for keychains.\n            mode: The mode to unbind the key in (default: `normal`).\n                  See `:help bindings.commands` for the available modes.\n        '
        with self._handle_config_error():
            self._keyconfig.unbind(self._parse_key(key), mode=mode, save_yaml=True)

    @cmdutils.register(instance='config-commands', star_args_optional=True)
    @cmdutils.argument('option', completion=configmodel.option)
    @cmdutils.argument('values', completion=configmodel.value)
    @cmdutils.argument('pattern', flag='u')
    def config_cycle(self, option: str, *values: str, pattern: str=None, temp: bool=False, print_: bool=False) -> None:
        if False:
            print('Hello World!')
        'Cycle an option between multiple values.\n\n        Args:\n            option: The name of the option.\n            *values: The values to cycle through.\n            pattern: The link:configuring{outfilesuffix}#patterns[URL pattern] to use.\n            temp: Set value temporarily until qutebrowser is closed.\n            print_: Print the value after setting.\n        '
        parsed_pattern = self._parse_pattern(pattern)
        with self._handle_config_error():
            opt = self._config.get_opt(option)
            old_value = self._config.get_obj_for_pattern(option, pattern=parsed_pattern)
        if not values and isinstance(opt.typ, configtypes.Bool):
            values = ('true', 'false')
        if len(values) < 2:
            raise cmdutils.CommandError('Need at least two values for non-boolean settings.')
        with self._handle_config_error():
            cycle_values = [opt.typ.from_str(val) for val in values]
        try:
            idx = cycle_values.index(old_value)
            idx = (idx + 1) % len(cycle_values)
            value = cycle_values[idx]
        except ValueError:
            value = cycle_values[0]
        with self._handle_config_error():
            self._config.set_obj(option, value, pattern=parsed_pattern, save_yaml=not temp)
        if print_:
            self._print_value(option, pattern=parsed_pattern)

    @cmdutils.register(instance='config-commands')
    @cmdutils.argument('option', completion=configmodel.customized_option)
    @cmdutils.argument('pattern', flag='u')
    def config_unset(self, option: str, *, pattern: str=None, temp: bool=False) -> None:
        if False:
            return 10
        'Unset an option.\n\n        This sets an option back to its default and removes it from\n        autoconfig.yml.\n\n        Args:\n            option: The name of the option.\n            pattern: The link:configuring{outfilesuffix}#patterns[URL pattern] to use.\n            temp: Set value temporarily until qutebrowser is closed.\n        '
        parsed_pattern = self._parse_pattern(pattern)
        with self._handle_config_error():
            changed = self._config.unset(option, save_yaml=not temp, pattern=parsed_pattern)
        if not changed:
            text = f'{option} is not customized'
            if pattern is not None:
                text += f' for {pattern}'
            raise cmdutils.CommandError(text)

    @cmdutils.register(instance='config-commands')
    @cmdutils.argument('win_id', value=cmdutils.Value.win_id)
    def config_diff(self, win_id: int, include_hidden: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        'Show all customized options.\n\n        Args:\n            include_hidden: Also include internal qutebrowser settings.\n        '
        url = QUrl('qute://configdiff')
        if include_hidden:
            query = QUrlQuery()
            query.addQueryItem('include_hidden', 'true')
            url.setQuery(query)
        tabbed_browser = objreg.get('tabbed-browser', scope='window', window=win_id)
        tabbed_browser.load_url(url, newtab=False)

    @cmdutils.register(instance='config-commands')
    @cmdutils.argument('option', completion=configmodel.list_option)
    def config_list_add(self, option: str, value: str, temp: bool=False) -> None:
        if False:
            return 10
        'Append a value to a config option that is a list.\n\n        Args:\n            option: The name of the option.\n            value: The value to append to the end of the list.\n            temp: Add value temporarily until qutebrowser is closed.\n        '
        with self._handle_config_error():
            opt = self._config.get_opt(option)
        valid_list_types = (configtypes.List, configtypes.ListOrValue)
        if not isinstance(opt.typ, valid_list_types):
            raise cmdutils.CommandError(':config-list-add can only be used for lists')
        with self._handle_config_error():
            option_value = self._config.get_mutable_obj(option)
            option_value.append(opt.typ.valtype.from_str(value))
            self._config.update_mutables(save_yaml=not temp)

    @cmdutils.register(instance='config-commands')
    @cmdutils.argument('option', completion=configmodel.dict_option)
    def config_dict_add(self, option: str, key: str, value: str, temp: bool=False, replace: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Add a key/value pair to a dictionary option.\n\n        Args:\n            option: The name of the option.\n            key: The key to use.\n            value: The value to place in the dictionary.\n            temp: Add value temporarily until qutebrowser is closed.\n            replace: Replace existing values. By default, existing values are\n                     not overwritten.\n        '
        with self._handle_config_error():
            opt = self._config.get_opt(option)
        if not isinstance(opt.typ, configtypes.Dict):
            raise cmdutils.CommandError(':config-dict-add can only be used for dicts')
        with self._handle_config_error():
            option_value = self._config.get_mutable_obj(option)
            if key in option_value and (not replace):
                raise cmdutils.CommandError('{} already exists in {} - use --replace to overwrite!'.format(key, option))
            option_value[key] = opt.typ.valtype.from_str(value)
            self._config.update_mutables(save_yaml=not temp)

    @cmdutils.register(instance='config-commands')
    @cmdutils.argument('option', completion=configmodel.list_option)
    def config_list_remove(self, option: str, value: str, temp: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        'Remove a value from a list.\n\n        Args:\n            option: The name of the option.\n            value: The value to remove from the list.\n            temp: Remove value temporarily until qutebrowser is closed.\n        '
        with self._handle_config_error():
            opt = self._config.get_opt(option)
        valid_list_types = (configtypes.List, configtypes.ListOrValue)
        if not isinstance(opt.typ, valid_list_types):
            raise cmdutils.CommandError(':config-list-remove can only be used for lists')
        converted = opt.typ.valtype.from_str(value)
        with self._handle_config_error():
            option_value = self._config.get_mutable_obj(option)
            if converted not in option_value:
                raise cmdutils.CommandError(f'{value} is not in {option}!')
            option_value.remove(converted)
            self._config.update_mutables(save_yaml=not temp)

    @cmdutils.register(instance='config-commands')
    @cmdutils.argument('option', completion=configmodel.dict_option)
    def config_dict_remove(self, option: str, key: str, temp: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        'Remove a key from a dict.\n\n        Args:\n            option: The name of the option.\n            key: The key to remove from the dict.\n            temp: Remove value temporarily until qutebrowser is closed.\n        '
        with self._handle_config_error():
            opt = self._config.get_opt(option)
        if not isinstance(opt.typ, configtypes.Dict):
            raise cmdutils.CommandError(':config-dict-remove can only be used for dicts')
        with self._handle_config_error():
            option_value = self._config.get_mutable_obj(option)
            if key not in option_value:
                raise cmdutils.CommandError('{} is not in {}!'.format(key, option))
            del option_value[key]
            self._config.update_mutables(save_yaml=not temp)

    @cmdutils.register(instance='config-commands')
    def config_clear(self, save: bool=False) -> None:
        if False:
            while True:
                i = 10
        'Set all settings back to their default.\n\n        Args:\n            save: If given, all configuration in autoconfig.yml is also\n                  removed.\n        '
        self._config.clear(save_yaml=save)

    @cmdutils.register(instance='config-commands')
    def config_source(self, filename: str=None, clear: bool=False) -> None:
        if False:
            while True:
                i = 10
        'Read a config.py file.\n\n        Args:\n            filename: The file to load. If not given, loads the default\n                      config.py.\n            clear: Clear current settings first.\n        '
        if filename is None:
            filename = standarddir.config_py()
        else:
            filename = os.path.expanduser(filename)
            if not os.path.isabs(filename):
                filename = os.path.join(standarddir.config(), filename)
        if clear:
            self.config_clear()
        try:
            configfiles.read_config_py(filename)
        except configexc.ConfigFileErrors as e:
            raise cmdutils.CommandError(e)

    @cmdutils.register(instance='config-commands')
    def config_edit(self, no_source: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        "Open the config.py file in the editor.\n\n        Args:\n            no_source: Don't re-source the config file after editing.\n        "

        def on_file_updated() -> None:
            if False:
                i = 10
                return i + 15
            "Source the new config when editing finished.\n\n            This can't use cmdutils.CommandError as it's run async.\n            "
            try:
                configfiles.read_config_py(filename)
            except configexc.ConfigFileErrors as e:
                message.error(str(e))
        ed = editor.ExternalEditor(watch=True, parent=self._config)
        if not no_source:
            ed.file_updated.connect(on_file_updated)
        filename = standarddir.config_py()
        ed.edit_file(filename)

    @cmdutils.register(instance='config-commands')
    def config_write_py(self, filename: str=None, force: bool=False, defaults: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Write the current configuration to a config.py file.\n\n        Args:\n            filename: The file to write to, or None for the default config.py.\n            force: Force overwriting existing files.\n            defaults: Write the defaults instead of values configured via :set.\n        '
        if filename is None:
            filename = standarddir.config_py()
        else:
            filename = os.path.expanduser(filename)
            if not os.path.isabs(filename):
                filename = os.path.join(standarddir.config(), filename)
        if os.path.exists(filename) and (not force):
            raise cmdutils.CommandError('{} already exists - use --force to overwrite!'.format(filename))
        options: List[Tuple[Optional[urlmatch.UrlPattern], configdata.Option, Any]] = []
        if defaults:
            options = [(None, opt, opt.default) for (_name, opt) in sorted(configdata.DATA.items())]
            bindings = dict(configdata.DATA['bindings.default'].default)
            commented = True
        else:
            for values in self._config:
                for scoped in values:
                    options.append((scoped.pattern, values.opt, scoped.value))
            bindings = dict(self._config.get_mutable_obj('bindings.commands'))
            commented = False
        writer = configfiles.ConfigPyWriter(options, bindings, commented=commented)
        try:
            writer.write(filename)
        except OSError as e:
            raise cmdutils.CommandError(str(e))