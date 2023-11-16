"""IPython terminal interface using prompt_toolkit"""
import asyncio
import os
import sys
from warnings import warn
from typing import Union as UnionType, Optional
from IPython.core.async_helpers import get_asyncio_loop
from IPython.core.interactiveshell import InteractiveShell, InteractiveShellABC
from IPython.utils.py3compat import input
from IPython.utils.terminal import toggle_set_term_title, set_term_title, restore_term_title
from IPython.utils.process import abbrev_cwd
from traitlets import Bool, Unicode, Dict, Integer, List, observe, Instance, Type, default, Enum, Union, Any, validate, Float
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.enums import DEFAULT_BUFFER, EditingMode
from prompt_toolkit.filters import HasFocus, Condition, IsDone
from prompt_toolkit.formatted_text import PygmentsTokens
from prompt_toolkit.history import History
from prompt_toolkit.layout.processors import ConditionalProcessor, HighlightMatchingBracketProcessor
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import PromptSession, CompleteStyle, print_formatted_text
from prompt_toolkit.styles import DynamicStyle, merge_styles
from prompt_toolkit.styles.pygments import style_from_pygments_cls, style_from_pygments_dict
from prompt_toolkit import __version__ as ptk_version
from pygments.styles import get_style_by_name
from pygments.style import Style
from pygments.token import Token
from .debugger import TerminalPdb, Pdb
from .magics import TerminalMagics
from .pt_inputhooks import get_inputhook_name_and_func
from .prompts import Prompts, ClassicPrompts, RichPromptDisplayHook
from .ptutils import IPythonPTCompleter, IPythonPTLexer
from .shortcuts import KEY_BINDINGS, create_ipython_shortcuts, create_identifier, RuntimeBinding, add_binding
from .shortcuts.filters import KEYBINDING_FILTERS, filter_from_string
from .shortcuts.auto_suggest import NavigableAutoSuggestFromHistory, AppendAutoSuggestionInAnyLine
PTK3 = ptk_version.startswith('3.')

class _NoStyle(Style):
    pass
_style_overrides_light_bg = {Token.Prompt: '#ansibrightblue', Token.PromptNum: '#ansiblue bold', Token.OutPrompt: '#ansibrightred', Token.OutPromptNum: '#ansired bold'}
_style_overrides_linux = {Token.Prompt: '#ansibrightgreen', Token.PromptNum: '#ansigreen bold', Token.OutPrompt: '#ansibrightred', Token.OutPromptNum: '#ansired bold'}

def get_default_editor():
    if False:
        return 10
    try:
        return os.environ['EDITOR']
    except KeyError:
        pass
    except UnicodeError:
        warn('$EDITOR environment variable is not pure ASCII. Using platform default editor.')
    if os.name == 'posix':
        return 'vi'
    else:
        return 'notepad'
for _name in ('stdin', 'stdout', 'stderr'):
    _stream = getattr(sys, _name)
    try:
        if not _stream or not hasattr(_stream, 'isatty') or (not _stream.isatty()):
            _is_tty = False
            break
    except ValueError:
        _is_tty = False
        break
else:
    _is_tty = True
_use_simple_prompt = 'IPY_TEST_SIMPLE_PROMPT' in os.environ or not _is_tty

def black_reformat_handler(text_before_cursor):
    if False:
        while True:
            i = 10
    '\n    We do not need to protect against error,\n    this is taken care at a higher level where any reformat error is ignored.\n    Indeed we may call reformatting on incomplete code.\n    '
    import black
    formatted_text = black.format_str(text_before_cursor, mode=black.FileMode())
    if not text_before_cursor.endswith('\n') and formatted_text.endswith('\n'):
        formatted_text = formatted_text[:-1]
    return formatted_text

def yapf_reformat_handler(text_before_cursor):
    if False:
        return 10
    from yapf.yapflib import file_resources
    from yapf.yapflib import yapf_api
    style_config = file_resources.GetDefaultStyleForDir(os.getcwd())
    (formatted_text, was_formatted) = yapf_api.FormatCode(text_before_cursor, style_config=style_config)
    if was_formatted:
        if not text_before_cursor.endswith('\n') and formatted_text.endswith('\n'):
            formatted_text = formatted_text[:-1]
        return formatted_text
    else:
        return text_before_cursor

class PtkHistoryAdapter(History):
    """
    Prompt toolkit has it's own way of handling history, Where it assumes it can
    Push/pull from history.

    """

    def __init__(self, shell):
        if False:
            print('Hello World!')
        super().__init__()
        self.shell = shell
        self._refresh()

    def append_string(self, string):
        if False:
            for i in range(10):
                print('nop')
        self._loaded = False
        self._refresh()

    def _refresh(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._loaded:
            self._loaded_strings = list(self.load_history_strings())

    def load_history_strings(self):
        if False:
            i = 10
            return i + 15
        last_cell = ''
        res = []
        for (__, ___, cell) in self.shell.history_manager.get_tail(self.shell.history_load_length, include_latest=True):
            cell = cell.rstrip()
            if cell and cell != last_cell:
                res.append(cell)
                last_cell = cell
        yield from res[::-1]

    def store_string(self, string: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

class TerminalInteractiveShell(InteractiveShell):
    mime_renderers = Dict().tag(config=True)
    space_for_menu = Integer(6, help='Number of line at the bottom of the screen to reserve for the tab completion menu, search history, ...etc, the height of these menus will at most this value. Increase it is you prefer long and skinny menus, decrease for short and wide.').tag(config=True)
    pt_app: UnionType[PromptSession, None] = None
    auto_suggest: UnionType[AutoSuggestFromHistory, NavigableAutoSuggestFromHistory, None] = None
    debugger_history = None
    debugger_history_file = Unicode('~/.pdbhistory', help='File in which to store and read history').tag(config=True)
    simple_prompt = Bool(_use_simple_prompt, help='Use `raw_input` for the REPL, without completion and prompt colors.\n\n            Useful when controlling IPython as a subprocess, and piping STDIN/OUT/ERR. Known usage are:\n            IPython own testing machinery, and emacs inferior-shell integration through elpy.\n\n            This mode default to `True` if the `IPY_TEST_SIMPLE_PROMPT`\n            environment variable is set, or the current terminal is not a tty.').tag(config=True)

    @property
    def debugger_cls(self):
        if False:
            while True:
                i = 10
        return Pdb if self.simple_prompt else TerminalPdb
    confirm_exit = Bool(True, help="\n        Set to confirm when you try to exit IPython with an EOF (Control-D\n        in Unix, Control-Z/Enter in Windows). By typing 'exit' or 'quit',\n        you can force a direct exit without any confirmation.").tag(config=True)
    editing_mode = Unicode('emacs', help="Shortcut style to use at the prompt. 'vi' or 'emacs'.").tag(config=True)
    emacs_bindings_in_vi_insert_mode = Bool(True, help="Add shortcuts from 'emacs' insert mode to 'vi' insert mode.").tag(config=True)
    modal_cursor = Bool(True, help='\n       Cursor shape changes depending on vi mode: beam in vi insert mode,\n       block in nav mode, underscore in replace mode.').tag(config=True)
    ttimeoutlen = Float(0.01, help='The time in milliseconds that is waited for a key code\n       to complete.').tag(config=True)
    timeoutlen = Float(0.5, help='The time in milliseconds that is waited for a mapped key\n       sequence to complete.').tag(config=True)
    autoformatter = Unicode(None, help="Autoformatter to reformat Terminal code. Can be `'black'`, `'yapf'` or `None`", allow_none=True).tag(config=True)
    auto_match = Bool(False, help='\n        Automatically add/delete closing bracket or quote when opening bracket or quote is entered/deleted.\n        Brackets: (), [], {}\n        Quotes: \'\', ""\n        ').tag(config=True)
    mouse_support = Bool(False, help='Enable mouse support in the prompt\n(Note: prevents selecting text with the mouse)').tag(config=True)
    highlighting_style = Union([Unicode('legacy'), Type(klass=Style)], help='The name or class of a Pygments style to use for syntax\n        highlighting. To see available styles, run `pygmentize -L styles`.').tag(config=True)

    @validate('editing_mode')
    def _validate_editing_mode(self, proposal):
        if False:
            for i in range(10):
                print('nop')
        if proposal['value'].lower() == 'vim':
            proposal['value'] = 'vi'
        elif proposal['value'].lower() == 'default':
            proposal['value'] = 'emacs'
        if hasattr(EditingMode, proposal['value'].upper()):
            return proposal['value'].lower()
        return self.editing_mode

    @observe('editing_mode')
    def _editing_mode(self, change):
        if False:
            print('Hello World!')
        if self.pt_app:
            self.pt_app.editing_mode = getattr(EditingMode, change.new.upper())

    def _set_formatter(self, formatter):
        if False:
            while True:
                i = 10
        if formatter is None:
            self.reformat_handler = lambda x: x
        elif formatter == 'black':
            self.reformat_handler = black_reformat_handler
        elif formatter == 'yapf':
            self.reformat_handler = yapf_reformat_handler
        else:
            raise ValueError

    @observe('autoformatter')
    def _autoformatter_changed(self, change):
        if False:
            return 10
        formatter = change.new
        self._set_formatter(formatter)

    @observe('highlighting_style')
    @observe('colors')
    def _highlighting_style_changed(self, change):
        if False:
            while True:
                i = 10
        self.refresh_style()

    def refresh_style(self):
        if False:
            while True:
                i = 10
        self._style = self._make_style_from_name_or_cls(self.highlighting_style)
    highlighting_style_overrides = Dict(help='Override highlighting format for specific tokens').tag(config=True)
    true_color = Bool(False, help='Use 24bit colors instead of 256 colors in prompt highlighting.\n        If your terminal supports true color, the following command should\n        print ``TRUECOLOR`` in orange::\n\n            printf "\\x1b[38;2;255;100;0mTRUECOLOR\\x1b[0m\\n"\n        ').tag(config=True)
    editor = Unicode(get_default_editor(), help='Set the editor used by IPython (default to $EDITOR/vi/notepad).').tag(config=True)
    prompts_class = Type(Prompts, help='Class used to generate Prompt token for prompt_toolkit').tag(config=True)
    prompts = Instance(Prompts)

    @default('prompts')
    def _prompts_default(self):
        if False:
            return 10
        return self.prompts_class(self)

    @default('displayhook_class')
    def _displayhook_class_default(self):
        if False:
            print('Hello World!')
        return RichPromptDisplayHook
    term_title = Bool(True, help='Automatically set the terminal title').tag(config=True)
    term_title_format = Unicode('IPython: {cwd}', help='Customize the terminal title format.  This is a python format string. ' + 'Available substitutions are: {cwd}.').tag(config=True)
    display_completions = Enum(('column', 'multicolumn', 'readlinelike'), help="Options for displaying tab completions, 'column', 'multicolumn', and 'readlinelike'. These options are for `prompt_toolkit`, see `prompt_toolkit` documentation for more information.", default_value='multicolumn').tag(config=True)
    highlight_matching_brackets = Bool(True, help='Highlight matching brackets.').tag(config=True)
    extra_open_editor_shortcuts = Bool(False, help='Enable vi (v) or Emacs (C-X C-E) shortcuts to open an external editor. This is in addition to the F2 binding, which is always enabled.').tag(config=True)
    handle_return = Any(None, help='Provide an alternative handler to be called when the user presses Return. This is an advanced option intended for debugging, which may be changed or removed in later releases.').tag(config=True)
    enable_history_search = Bool(True, help='Allows to enable/disable the prompt toolkit history search').tag(config=True)
    autosuggestions_provider = Unicode('NavigableAutoSuggestFromHistory', help="Specifies from which source automatic suggestions are provided. Can be set to ``'NavigableAutoSuggestFromHistory'`` (:kbd:`up` and :kbd:`down` swap suggestions), ``'AutoSuggestFromHistory'``,  or ``None`` to disable automatic suggestions. Default is `'NavigableAutoSuggestFromHistory`'.", allow_none=True).tag(config=True)

    def _set_autosuggestions(self, provider):
        if False:
            print('Hello World!')
        if self.auto_suggest and isinstance(self.auto_suggest, NavigableAutoSuggestFromHistory):
            self.auto_suggest.disconnect()
        if provider is None:
            self.auto_suggest = None
        elif provider == 'AutoSuggestFromHistory':
            self.auto_suggest = AutoSuggestFromHistory()
        elif provider == 'NavigableAutoSuggestFromHistory':
            self.auto_suggest = NavigableAutoSuggestFromHistory()
        else:
            raise ValueError('No valid provider.')
        if self.pt_app:
            self.pt_app.auto_suggest = self.auto_suggest

    @observe('autosuggestions_provider')
    def _autosuggestions_provider_changed(self, change):
        if False:
            i = 10
            return i + 15
        provider = change.new
        self._set_autosuggestions(provider)
    shortcuts = List(trait=Dict(key_trait=Enum(['command', 'match_keys', 'match_filter', 'new_keys', 'new_filter', 'create']), per_key_traits={'command': Unicode(), 'match_keys': List(Unicode()), 'match_filter': Unicode(), 'new_keys': List(Unicode()), 'new_filter': Unicode(), 'create': Bool(False)}), help='Add, disable or modifying shortcuts.\n\n        Each entry on the list should be a dictionary with ``command`` key\n        identifying the target function executed by the shortcut and at least\n        one of the following:\n\n        - ``match_keys``: list of keys used to match an existing shortcut,\n        - ``match_filter``: shortcut filter used to match an existing shortcut,\n        - ``new_keys``: list of keys to set,\n        - ``new_filter``: a new shortcut filter to set\n\n        The filters have to be composed of pre-defined verbs and joined by one\n        of the following conjunctions: ``&`` (and), ``|`` (or), ``~`` (not).\n        The pre-defined verbs are:\n\n        {}\n\n\n        To disable a shortcut set ``new_keys`` to an empty list.\n        To add a shortcut add key ``create`` with value ``True``.\n\n        When modifying/disabling shortcuts, ``match_keys``/``match_filter`` can\n        be omitted if the provided specification uniquely identifies a shortcut\n        to be modified/disabled. When modifying a shortcut ``new_filter`` or\n        ``new_keys`` can be omitted which will result in reuse of the existing\n        filter/keys.\n\n        Only shortcuts defined in IPython (and not default prompt-toolkit\n        shortcuts) can be modified or disabled. The full list of shortcuts,\n        command identifiers and filters is available under\n        :ref:`terminal-shortcuts-list`.\n        '.format('\n        '.join([f'- `{k}`' for k in KEYBINDING_FILTERS]))).tag(config=True)

    @observe('shortcuts')
    def _shortcuts_changed(self, change):
        if False:
            while True:
                i = 10
        if self.pt_app:
            self.pt_app.key_bindings = self._merge_shortcuts(user_shortcuts=change.new)

    def _merge_shortcuts(self, user_shortcuts):
        if False:
            print('Hello World!')
        key_bindings = create_ipython_shortcuts(self)
        known_commands = {create_identifier(binding.command): binding.command for binding in KEY_BINDINGS}
        shortcuts_to_skip = []
        shortcuts_to_add = []
        for shortcut in user_shortcuts:
            command_id = shortcut['command']
            if command_id not in known_commands:
                allowed_commands = '\n - '.join(known_commands)
                raise ValueError(f'{command_id} is not a known shortcut command. Allowed commands are: \n - {allowed_commands}')
            old_keys = shortcut.get('match_keys', None)
            old_filter = filter_from_string(shortcut['match_filter']) if 'match_filter' in shortcut else None
            matching = [binding for binding in KEY_BINDINGS if (old_filter is None or binding.filter == old_filter) and (old_keys is None or [k for k in binding.keys] == old_keys) and (create_identifier(binding.command) == command_id)]
            new_keys = shortcut.get('new_keys', None)
            new_filter = shortcut.get('new_filter', None)
            command = known_commands[command_id]
            creating_new = shortcut.get('create', False)
            modifying_existing = not creating_new and (new_keys is not None or new_filter)
            if creating_new and new_keys == []:
                raise ValueError('Cannot add a shortcut without keys')
            if modifying_existing:
                specification = {key: shortcut[key] for key in ['command', 'filter'] if key in shortcut}
                if len(matching) == 0:
                    raise ValueError(f'No shortcuts matching {specification} found in {KEY_BINDINGS}')
                elif len(matching) > 1:
                    raise ValueError(f'Multiple shortcuts matching {specification} found, please add keys/filter to select one of: {matching}')
                matched = matching[0]
                old_filter = matched.filter
                old_keys = list(matched.keys)
                shortcuts_to_skip.append(RuntimeBinding(command, keys=old_keys, filter=old_filter))
            if new_keys != []:
                shortcuts_to_add.append(RuntimeBinding(command, keys=new_keys or old_keys, filter=filter_from_string(new_filter) if new_filter is not None else old_filter if old_filter is not None else filter_from_string('always')))
        key_bindings = create_ipython_shortcuts(self, skip=shortcuts_to_skip)
        for binding in shortcuts_to_add:
            add_binding(key_bindings, binding)
        return key_bindings
    prompt_includes_vi_mode = Bool(True, help='Display the current vi mode (when using vi editing mode).').tag(config=True)

    @observe('term_title')
    def init_term_title(self, change=None):
        if False:
            while True:
                i = 10
        if self.term_title and _is_tty:
            toggle_set_term_title(True)
            set_term_title(self.term_title_format.format(cwd=abbrev_cwd()))
        else:
            toggle_set_term_title(False)

    def restore_term_title(self):
        if False:
            for i in range(10):
                print('nop')
        if self.term_title and _is_tty:
            restore_term_title()

    def init_display_formatter(self):
        if False:
            for i in range(10):
                print('nop')
        super(TerminalInteractiveShell, self).init_display_formatter()
        self.display_formatter.active_types = ['text/plain']

    def init_prompt_toolkit_cli(self):
        if False:
            for i in range(10):
                print('nop')
        if self.simple_prompt:

            def prompt():
                if False:
                    i = 10
                    return i + 15
                prompt_text = ''.join((x[1] for x in self.prompts.in_prompt_tokens()))
                lines = [input(prompt_text)]
                prompt_continuation = ''.join((x[1] for x in self.prompts.continuation_prompt_tokens()))
                while self.check_complete('\n'.join(lines))[0] == 'incomplete':
                    lines.append(input(prompt_continuation))
                return '\n'.join(lines)
            self.prompt_for_code = prompt
            return
        key_bindings = self._merge_shortcuts(user_shortcuts=self.shortcuts)
        history = PtkHistoryAdapter(self)
        self._style = self._make_style_from_name_or_cls(self.highlighting_style)
        self.style = DynamicStyle(lambda : self._style)
        editing_mode = getattr(EditingMode, self.editing_mode.upper())
        self.pt_loop = asyncio.new_event_loop()
        self.pt_app = PromptSession(auto_suggest=self.auto_suggest, editing_mode=editing_mode, key_bindings=key_bindings, history=history, completer=IPythonPTCompleter(shell=self), enable_history_search=self.enable_history_search, style=self.style, include_default_pygments_style=False, mouse_support=self.mouse_support, enable_open_in_editor=self.extra_open_editor_shortcuts, color_depth=self.color_depth, tempfile_suffix='.py', **self._extra_prompt_options())
        if isinstance(self.auto_suggest, NavigableAutoSuggestFromHistory):
            self.auto_suggest.connect(self.pt_app)

    def _make_style_from_name_or_cls(self, name_or_cls):
        if False:
            while True:
                i = 10
        '\n        Small wrapper that make an IPython compatible style from a style name\n\n        We need that to add style for prompt ... etc.\n        '
        style_overrides = {}
        if name_or_cls == 'legacy':
            legacy = self.colors.lower()
            if legacy == 'linux':
                style_cls = get_style_by_name('monokai')
                style_overrides = _style_overrides_linux
            elif legacy == 'lightbg':
                style_overrides = _style_overrides_light_bg
                style_cls = get_style_by_name('pastie')
            elif legacy == 'neutral':
                style_cls = get_style_by_name('default')
                style_overrides.update({Token.Number: '#ansigreen', Token.Operator: 'noinherit', Token.String: '#ansiyellow', Token.Name.Function: '#ansiblue', Token.Name.Class: 'bold #ansiblue', Token.Name.Namespace: 'bold #ansiblue', Token.Name.Variable.Magic: '#ansiblue', Token.Prompt: '#ansigreen', Token.PromptNum: '#ansibrightgreen bold', Token.OutPrompt: '#ansired', Token.OutPromptNum: '#ansibrightred bold'})
                if os.name == 'nt':
                    style_overrides.update({Token.Prompt: '#ansidarkgreen', Token.PromptNum: '#ansigreen bold', Token.OutPrompt: '#ansidarkred', Token.OutPromptNum: '#ansired bold'})
            elif legacy == 'nocolor':
                style_cls = _NoStyle
                style_overrides = {}
            else:
                raise ValueError('Got unknown colors: ', legacy)
        else:
            if isinstance(name_or_cls, str):
                style_cls = get_style_by_name(name_or_cls)
            else:
                style_cls = name_or_cls
            style_overrides = {Token.Prompt: '#ansigreen', Token.PromptNum: '#ansibrightgreen bold', Token.OutPrompt: '#ansired', Token.OutPromptNum: '#ansibrightred bold'}
        style_overrides.update(self.highlighting_style_overrides)
        style = merge_styles([style_from_pygments_cls(style_cls), style_from_pygments_dict(style_overrides)])
        return style

    @property
    def pt_complete_style(self):
        if False:
            while True:
                i = 10
        return {'multicolumn': CompleteStyle.MULTI_COLUMN, 'column': CompleteStyle.COLUMN, 'readlinelike': CompleteStyle.READLINE_LIKE}[self.display_completions]

    @property
    def color_depth(self):
        if False:
            while True:
                i = 10
        return ColorDepth.TRUE_COLOR if self.true_color else None

    def _extra_prompt_options(self):
        if False:
            return 10
        '\n        Return the current layout option for the current Terminal InteractiveShell\n        '

        def get_message():
            if False:
                for i in range(10):
                    print('nop')
            return PygmentsTokens(self.prompts.in_prompt_tokens())
        if self.editing_mode == 'emacs':
            get_message = get_message()
        options = {'complete_in_thread': False, 'lexer': IPythonPTLexer(), 'reserve_space_for_menu': self.space_for_menu, 'message': get_message, 'prompt_continuation': lambda width, lineno, is_soft_wrap: PygmentsTokens(self.prompts.continuation_prompt_tokens(width)), 'multiline': True, 'complete_style': self.pt_complete_style, 'input_processors': [ConditionalProcessor(processor=HighlightMatchingBracketProcessor(chars='[](){}'), filter=HasFocus(DEFAULT_BUFFER) & ~IsDone() & Condition(lambda : self.highlight_matching_brackets)), ConditionalProcessor(processor=AppendAutoSuggestionInAnyLine(), filter=HasFocus(DEFAULT_BUFFER) & ~IsDone() & Condition(lambda : isinstance(self.auto_suggest, NavigableAutoSuggestFromHistory)))]}
        if not PTK3:
            options['inputhook'] = self.inputhook
        return options

    def prompt_for_code(self):
        if False:
            for i in range(10):
                print('nop')
        if self.rl_next_input:
            default = self.rl_next_input
            self.rl_next_input = None
        else:
            default = ''
        policy = asyncio.get_event_loop_policy()
        old_loop = get_asyncio_loop()
        if old_loop is not self.pt_loop:
            policy.set_event_loop(self.pt_loop)
        try:
            with patch_stdout(raw=True):
                text = self.pt_app.prompt(default=default, **self._extra_prompt_options())
        finally:
            if old_loop is not None and old_loop is not self.pt_loop:
                policy.set_event_loop(old_loop)
        return text

    def enable_win_unicode_console(self):
        if False:
            for i in range(10):
                print('nop')
        warn('`enable_win_unicode_console` is deprecated since IPython 7.10, does not do anything and will be removed in the future', DeprecationWarning, stacklevel=2)

    def init_io(self):
        if False:
            while True:
                i = 10
        if sys.platform not in {'win32', 'cli'}:
            return
        import colorama
        colorama.init()

    def init_magics(self):
        if False:
            for i in range(10):
                print('nop')
        super(TerminalInteractiveShell, self).init_magics()
        self.register_magics(TerminalMagics)

    def init_alias(self):
        if False:
            while True:
                i = 10
        super(TerminalInteractiveShell, self).init_alias()
        if os.name == 'posix':
            for cmd in ('clear', 'more', 'less', 'man'):
                self.alias_manager.soft_define_alias(cmd, cmd)

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super(TerminalInteractiveShell, self).__init__(*args, **kwargs)
        self._set_autosuggestions(self.autosuggestions_provider)
        self.init_prompt_toolkit_cli()
        self.init_term_title()
        self.keep_running = True
        self._set_formatter(self.autoformatter)

    def ask_exit(self):
        if False:
            while True:
                i = 10
        self.keep_running = False
    rl_next_input = None

    def interact(self):
        if False:
            for i in range(10):
                print('nop')
        self.keep_running = True
        while self.keep_running:
            print(self.separate_in, end='')
            try:
                code = self.prompt_for_code()
            except EOFError:
                if not self.confirm_exit or self.ask_yes_no('Do you really want to exit ([y]/n)?', 'y', 'n'):
                    self.ask_exit()
            else:
                if code:
                    self.run_cell(code, store_history=True)

    def mainloop(self):
        if False:
            i = 10
            return i + 15
        while True:
            try:
                self.interact()
                break
            except KeyboardInterrupt as e:
                print('\n%s escaped interact()\n' % type(e).__name__)
            finally:
                if hasattr(self, '_eventloop'):
                    self._eventloop.stop()
                self.restore_term_title()
        self._atexit_once()
    _inputhook = None

    def inputhook(self, context):
        if False:
            return 10
        if self._inputhook is not None:
            self._inputhook(context)
    active_eventloop: Optional[str] = None

    def enable_gui(self, gui: Optional[str]=None) -> None:
        if False:
            while True:
                i = 10
        if self.simple_prompt is True and gui is not None:
            print(f'Cannot install event loop hook for "{gui}" when running with `--simple-prompt`.')
            print('NOTE: Tk is supported natively; use Tk apps and Tk backends with `--simple-prompt`.')
            return
        if self._inputhook is None and gui is None:
            print('No event loop hook running.')
            return
        if self._inputhook is not None and gui is not None:
            (newev, newinhook) = get_inputhook_name_and_func(gui)
            if self._inputhook == newinhook:
                self.log.info(f'Shell is already running the {self.active_eventloop} eventloop. Doing nothing')
                return
            self.log.warning(f'Shell is already running a different gui event loop for {self.active_eventloop}. Call with no arguments to disable the current loop.')
            return
        if self._inputhook is not None and gui is None:
            self.active_eventloop = self._inputhook = None
        if gui and gui not in {'inline', 'webagg'}:
            (self.active_eventloop, self._inputhook) = get_inputhook_name_and_func(gui)
        else:
            self.active_eventloop = self._inputhook = None
        if PTK3:
            import asyncio
            from prompt_toolkit.eventloop import new_eventloop_with_inputhook
            if gui == 'asyncio':
                self.pt_loop = get_asyncio_loop()
                print('Installed asyncio event loop hook.')
            elif self._inputhook:
                self.pt_loop = new_eventloop_with_inputhook(self._inputhook)
                print(f'Installed {self.active_eventloop} event loop hook.')
            else:
                self.pt_loop = asyncio.new_event_loop()
                print('GUI event loop hook disabled.')
    system = InteractiveShell.system_raw

    def auto_rewrite_input(self, cmd):
        if False:
            i = 10
            return i + 15
        'Overridden from the parent class to use fancy rewriting prompt'
        if not self.show_rewritten_input:
            return
        tokens = self.prompts.rewrite_prompt_tokens()
        if self.pt_app:
            print_formatted_text(PygmentsTokens(tokens), end='', style=self.pt_app.app.style)
            print(cmd)
        else:
            prompt = ''.join((s for (t, s) in tokens))
            print(prompt, cmd, sep='')
    _prompts_before = None

    def switch_doctest_mode(self, mode):
        if False:
            for i in range(10):
                print('nop')
        'Switch prompts to classic for %doctest_mode'
        if mode:
            self._prompts_before = self.prompts
            self.prompts = ClassicPrompts(self)
        elif self._prompts_before:
            self.prompts = self._prompts_before
            self._prompts_before = None
InteractiveShellABC.register(TerminalInteractiveShell)
if __name__ == '__main__':
    TerminalInteractiveShell.instance().interact()