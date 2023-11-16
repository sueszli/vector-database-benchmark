"""Logger implementing the Command Line Interface.

A replacement for the standard Python `logging` API
designed for implementing a better CLI UX for the cluster launcher.

Supports color, bold text, italics, underlines, etc.
(depending on TTY features)
as well as indentation and other structured output.
"""
import inspect
import logging
import os
import sys
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
import click
import colorama
import ray
if sys.platform == 'win32':
    import msvcrt
else:
    import select

class _ColorfulMock:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.identity = lambda x: x
        self.colorful = self
        self.colormode = None
        self.NO_COLORS = None
        self.ANSI_8_COLORS = None

    def disable(self):
        if False:
            return 10
        pass

    @contextmanager
    def with_style(self, x):
        if False:
            for i in range(10):
                print('nop')

        class IdentityClass:

            def __getattr__(self, name):
                if False:
                    i = 10
                    return i + 15
                return lambda y: y
        yield IdentityClass()

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        if name == 'with_style':
            return self.with_style
        return self.identity
try:
    import colorful as _cf
    from colorful.core import ColorfulString
    _cf.use_8_ansi_colors()
except ModuleNotFoundError:

    class ColorfulString:
        pass
    _cf = _ColorfulMock()

class _ColorfulProxy:
    _proxy_allowlist = ['disable', 'reset', 'bold', 'italic', 'underlined', 'dimmed', 'dodgerBlue', 'limeGreen', 'red', 'orange', 'skyBlue', 'magenta', 'yellow']

    def __getattr__(self, name):
        if False:
            for i in range(10):
                print('nop')
        res = getattr(_cf, name)
        if callable(res) and name not in _ColorfulProxy._proxy_allowlist:
            raise ValueError("Usage of the colorful method '" + name + "' is forbidden by the proxy to keep a consistent color scheme. Check `cli_logger.py` for allowed methods")
        return res
cf = _ColorfulProxy()
colorama.init(strip=False)

def _patched_makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None):
    if False:
        for i in range(10):
            print('nop')
    "Monkey-patched version of logging.Logger.makeRecord\n    We have to patch default loggers so they use the proper frame for\n    line numbers and function names (otherwise everything shows up as\n    e.g. cli_logger:info() instead of as where it was called from).\n\n    In Python 3.8 we could just use stacklevel=2, but we have to support\n    Python 3.6 and 3.7 as well.\n\n    The solution is this Python magic superhack.\n\n    The default makeRecord will deliberately check that we don't override\n    any existing property on the LogRecord using `extra`,\n    so we remove that check.\n\n    This patched version is otherwise identical to the one in the standard\n    library.\n\n    TODO: Remove this magic superhack. Find a more responsible workaround.\n    "
    rv = logging._logRecordFactory(name, level, fn, lno, msg, args, exc_info, func, sinfo)
    if extra is not None:
        rv.__dict__.update(extra)
    return rv
logging.Logger.makeRecord = _patched_makeRecord

def _external_caller_info():
    if False:
        i = 10
        return i + 15
    'Get the info from the caller frame.\n\n    Used to override the logging function and line number with the correct\n    ones. See the comment on _patched_makeRecord for more info.\n    '
    frame = inspect.currentframe()
    caller = frame
    levels = 0
    while caller.f_code.co_filename == __file__:
        caller = caller.f_back
        levels += 1
    return {'lineno': caller.f_lineno, 'filename': os.path.basename(caller.f_code.co_filename)}

def _format_msg(msg: str, *args: Any, no_format: bool=None, _tags: Dict[str, Any]=None, _numbered: Tuple[str, int, int]=None, **kwargs: Any):
    if False:
        i = 10
        return i + 15
    'Formats a message for printing.\n\n    Renders `msg` using the built-in `str.format` and the passed-in\n    `*args` and `**kwargs`.\n\n    Args:\n        *args (Any): `.format` arguments for `msg`.\n        no_format (bool):\n            If `no_format` is `True`,\n            `.format` will not be called on the message.\n\n            Useful if the output is user-provided or may otherwise\n            contain an unexpected formatting string (e.g. "{}").\n        _tags (Dict[str, Any]):\n            key-value pairs to display at the end of\n            the message in square brackets.\n\n            If a tag is set to `True`, it is printed without the value,\n            the presence of the tag treated as a "flag".\n\n            E.g. `_format_msg("hello", _tags=dict(from=mom, signed=True))`\n                 `hello [from=Mom, signed]`\n        _numbered (Tuple[str, int, int]):\n            `(brackets, i, n)`\n\n            The `brackets` string is composed of two "bracket" characters,\n            `i` is the index, `n` is the total.\n\n            The string `{i}/{n}` surrounded by the "brackets" is\n            prepended to the message.\n\n            This is used to number steps in a procedure, with different\n            brackets specifying different major tasks.\n\n            E.g. `_format_msg("hello", _numbered=("[]", 0, 5))`\n                 `[0/5] hello`\n\n    Returns:\n        The formatted message.\n    '
    if isinstance(msg, str) or isinstance(msg, ColorfulString):
        tags_str = ''
        if _tags is not None:
            tags_list = []
            for (k, v) in _tags.items():
                if v is True:
                    tags_list += [k]
                    continue
                if v is False:
                    continue
                tags_list += [k + '=' + v]
            if tags_list:
                tags_str = cf.reset(cf.dimmed(' [{}]'.format(', '.join(tags_list))))
        numbering_str = ''
        if _numbered is not None:
            (chars, i, n) = _numbered
            numbering_str = cf.dimmed(chars[0] + str(i) + '/' + str(n) + chars[1]) + ' '
        if no_format:
            return numbering_str + msg + tags_str
        return numbering_str + msg.format(*args, **kwargs) + tags_str
    if kwargs:
        raise ValueError('We do not support printing kwargs yet.')
    res = [msg, *args]
    res = [str(x) for x in res]
    return ', '.join(res)

def _isatty():
    if False:
        return 10
    'More robust check for interactive terminal/tty.'
    try:
        return sys.__stdin__.isatty()
    except Exception:
        return False

class _CliLogger:
    """Singleton class for CLI logging.

    Without calling 'cli_logger.configure', the CLILogger will default
    to 'record' style logging.

    Attributes:
        color_mode (str):
            Can be "true", "false", or "auto".

            Enables or disables `colorful`.

            If `color_mode` is "auto", is set to `not stdout.isatty()`
        indent_level (int):
            The current indentation level.

            All messages will be indented by prepending `"  " * indent_level`
        vebosity (int):
            Output verbosity.

            Low verbosity will disable `verbose` and `very_verbose` messages.
    """
    color_mode: str
    indent_level: int
    interactive: bool
    VALID_LOG_STYLES = ('auto', 'record', 'pretty')
    _autodetected_cf_colormode: int

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.indent_level = 0
        self._verbosity = 0
        self._verbosity_overriden = False
        self._color_mode = 'auto'
        self._log_style = 'record'
        self.pretty = False
        self.interactive = False
        self._autodetected_cf_colormode = cf.colorful.colormode
        self.set_format()

    def set_format(self, format_tmpl=None):
        if False:
            for i in range(10):
                print('nop')
        if not format_tmpl:
            from ray.autoscaler._private.constants import LOGGER_FORMAT
            format_tmpl = LOGGER_FORMAT
        self._formatter = logging.Formatter(format_tmpl)

    def configure(self, log_style=None, color_mode=None, verbosity=None):
        if False:
            i = 10
            return i + 15
        'Configures the logger according to values.'
        if log_style is not None:
            self._set_log_style(log_style)
        if color_mode is not None:
            self._set_color_mode(color_mode)
        if verbosity is not None:
            self._set_verbosity(verbosity)
        self.detect_colors()

    @property
    def log_style(self):
        if False:
            i = 10
            return i + 15
        return self._log_style

    def _set_log_style(self, x):
        if False:
            i = 10
            return i + 15
        'Configures interactivity and formatting.'
        self._log_style = x.lower()
        self.interactive = _isatty()
        if self._log_style == 'auto':
            self.pretty = _isatty()
        elif self._log_style == 'record':
            self.pretty = False
            self._set_color_mode('false')
        elif self._log_style == 'pretty':
            self.pretty = True

    @property
    def color_mode(self):
        if False:
            return 10
        return self._color_mode

    def _set_color_mode(self, x):
        if False:
            i = 10
            return i + 15
        self._color_mode = x.lower()
        self.detect_colors()

    @property
    def verbosity(self):
        if False:
            return 10
        if self._verbosity_overriden:
            return self._verbosity
        elif not self.pretty:
            return 999
        return self._verbosity

    def _set_verbosity(self, x):
        if False:
            i = 10
            return i + 15
        self._verbosity = x
        self._verbosity_overriden = True

    def detect_colors(self):
        if False:
            for i in range(10):
                print('nop')
        'Update color output settings.\n\n        Parse the `color_mode` string and optionally disable or force-enable\n        color output\n        (8-color ANSI if no terminal detected to be safe) in colorful.\n        '
        if self.color_mode == 'true':
            if self._autodetected_cf_colormode != cf.NO_COLORS:
                cf.colormode = self._autodetected_cf_colormode
            else:
                cf.colormode = cf.ANSI_8_COLORS
            return
        if self.color_mode == 'false':
            cf.disable()
            return
        if self.color_mode == 'auto':
            return
        raise ValueError('Invalid log color setting: ' + self.color_mode)

    def newline(self):
        if False:
            return 10
        'Print a line feed.'
        self.print('')

    def _print(self, msg: str, _level_str: str='INFO', _linefeed: bool=True, end: str=None):
        if False:
            i = 10
            return i + 15
        'Proxy for printing messages.\n\n        Args:\n            msg: Message to print.\n            linefeed (bool):\n                If `linefeed` is `False` no linefeed is printed at the\n                end of the message.\n        '
        if self.pretty:
            rendered_message = '  ' * self.indent_level + msg
        else:
            if msg.strip() == '':
                return
            caller_info = _external_caller_info()
            record = logging.LogRecord(name='cli', level=0, pathname=caller_info['filename'], lineno=caller_info['lineno'], msg=msg, args={}, exc_info=None)
            record.levelname = _level_str
            rendered_message = self._formatter.format(record)
        if _level_str in ['WARNING', 'ERROR', 'PANIC']:
            stream = sys.stderr
        else:
            stream = sys.stdout
        if not _linefeed:
            stream.write(rendered_message)
            stream.flush()
            return
        kwargs = {'end': end}
        print(rendered_message, file=stream, **kwargs)

    def indented(self):
        if False:
            while True:
                i = 10
        'Context manager that starts an indented block of output.'
        cli_logger = self

        class IndentedContextManager:

            def __enter__(self):
                if False:
                    return 10
                cli_logger.indent_level += 1

            def __exit__(self, type, value, tb):
                if False:
                    print('Hello World!')
                cli_logger.indent_level -= 1
        return IndentedContextManager()

    def group(self, msg: str, *args: Any, **kwargs: Any):
        if False:
            while True:
                i = 10
        'Print a group title in a special color and start an indented block.\n\n        For arguments, see `_format_msg`.\n        '
        self.print(cf.dodgerBlue(msg), *args, **kwargs)
        return self.indented()

    def verbatim_error_ctx(self, msg: str, *args: Any, **kwargs: Any):
        if False:
            while True:
                i = 10
        'Context manager for printing multi-line error messages.\n\n        Displays a start sequence "!!! {optional message}"\n        and a matching end sequence "!!!".\n\n        The string "!!!" can be used as a "tombstone" for searching.\n\n        For arguments, see `_format_msg`.\n        '
        cli_logger = self

        class VerbatimErorContextManager:

            def __enter__(self):
                if False:
                    print('Hello World!')
                cli_logger.error(cf.bold('!!! ') + '{}', msg, *args, **kwargs)

            def __exit__(self, type, value, tb):
                if False:
                    while True:
                        i = 10
                cli_logger.error(cf.bold('!!!'))
        return VerbatimErorContextManager()

    def labeled_value(self, key: str, msg: str, *args: Any, **kwargs: Any):
        if False:
            for i in range(10):
                print('nop')
        'Displays a key-value pair with special formatting.\n\n        Args:\n            key: Label that is prepended to the message.\n\n        For other arguments, see `_format_msg`.\n        '
        self._print(cf.skyBlue(key) + ': ' + _format_msg(cf.bold(msg), *args, **kwargs))

    def verbose(self, msg: str, *args: Any, **kwargs: Any):
        if False:
            i = 10
            return i + 15
        'Prints a message if verbosity is not 0.\n\n        For arguments, see `_format_msg`.\n        '
        if self.verbosity > 0:
            self.print(msg, *args, _level_str='VINFO', **kwargs)

    def verbose_warning(self, msg, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Prints a formatted warning if verbosity is not 0.\n\n        For arguments, see `_format_msg`.\n        '
        if self.verbosity > 0:
            self._warning(msg, *args, _level_str='VWARN', **kwargs)

    def verbose_error(self, msg: str, *args: Any, **kwargs: Any):
        if False:
            return 10
        'Logs an error if verbosity is not 0.\n\n        For arguments, see `_format_msg`.\n        '
        if self.verbosity > 0:
            self._error(msg, *args, _level_str='VERR', **kwargs)

    def very_verbose(self, msg: str, *args: Any, **kwargs: Any):
        if False:
            i = 10
            return i + 15
        'Prints if verbosity is > 1.\n\n        For arguments, see `_format_msg`.\n        '
        if self.verbosity > 1:
            self.print(msg, *args, _level_str='VVINFO', **kwargs)

    def success(self, msg: str, *args: Any, **kwargs: Any):
        if False:
            return 10
        'Prints a formatted success message.\n\n        For arguments, see `_format_msg`.\n        '
        self.print(cf.limeGreen(msg), *args, _level_str='SUCC', **kwargs)

    def _warning(self, msg: str, *args: Any, _level_str: str=None, **kwargs: Any):
        if False:
            print('Hello World!')
        'Prints a formatted warning message.\n\n        For arguments, see `_format_msg`.\n        '
        if _level_str is None:
            raise ValueError('Log level not set.')
        self.print(cf.orange(msg), *args, _level_str=_level_str, **kwargs)

    def warning(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self._warning(*args, _level_str='WARN', **kwargs)

    def _error(self, msg: str, *args: Any, _level_str: str=None, **kwargs: Any):
        if False:
            for i in range(10):
                print('nop')
        'Prints a formatted error message.\n\n        For arguments, see `_format_msg`.\n        '
        if _level_str is None:
            raise ValueError('Log level not set.')
        self.print(cf.red(msg), *args, _level_str=_level_str, **kwargs)

    def error(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self._error(*args, _level_str='ERR', **kwargs)

    def panic(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self._error(*args, _level_str='PANIC', **kwargs)

    def print(self, msg: str, *args: Any, _level_str: str='INFO', end: str=None, **kwargs: Any):
        if False:
            i = 10
            return i + 15
        'Prints a message.\n\n        For arguments, see `_format_msg`.\n        '
        self._print(_format_msg(msg, *args, **kwargs), _level_str=_level_str, end=end)

    def info(self, msg: str, no_format=True, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.print(msg, *args, no_format=no_format, **kwargs)

    def abort(self, msg: Optional[str]=None, *args: Any, exc: Any=None, **kwargs: Any):
        if False:
            return 10
        'Prints an error and aborts execution.\n\n        Print an error and throw an exception to terminate the program\n        (the exception will not print a message).\n        '
        if msg is not None:
            self._error(msg, *args, _level_str='PANIC', **kwargs)
        if exc is not None:
            raise exc
        exc_cls = click.ClickException
        if self.pretty:
            exc_cls = SilentClickException
        if msg is None:
            msg = 'Exiting due to cli_logger.abort()'
        raise exc_cls(msg)

    def doassert(self, val: bool, msg: str, *args: Any, **kwargs: Any):
        if False:
            return 10
        'Handle assertion without throwing a scary exception.\n\n        Args:\n            val: Value to check.\n\n        For other arguments, see `_format_msg`.\n        '
        if not val:
            exc = None
            if not self.pretty:
                exc = AssertionError()
            self.abort(msg, *args, exc=exc, **kwargs)

    def render_list(self, xs: List[str], separator: str=cf.reset(', ')):
        if False:
            while True:
                i = 10
        'Render a list of bolded values using a non-bolded separator.'
        return separator.join([str(cf.bold(x)) for x in xs])

    def confirm(self, yes: bool, msg: str, *args: Any, _abort: bool=False, _default: bool=False, _timeout_s: Optional[float]=None, **kwargs: Any):
        if False:
            i = 10
            return i + 15
        'Display a confirmation dialog.\n\n        Valid answers are "y/yes/true/1" and "n/no/false/0".\n\n        Args:\n            yes: If `yes` is `True` the dialog will default to "yes"\n                        and continue without waiting for user input.\n            _abort (bool):\n                If `_abort` is `True`,\n                "no" means aborting the program.\n            _default (bool):\n                The default action to take if the user just presses enter\n                with no input.\n            _timeout_s (float):\n                If user has no input within _timeout_s seconds, the default\n                action is taken. None means no timeout.\n        '
        should_abort = _abort
        default = _default
        if not self.interactive and (not yes):
            self.error('This command requires user confirmation. When running non-interactively, supply --yes to skip.')
            raise ValueError('Non-interactive confirm without --yes.')
        if default:
            yn_str = 'Y/n'
        else:
            yn_str = 'y/N'
        confirm_str = cf.underlined('Confirm [' + yn_str + ']:') + ' '
        rendered_message = _format_msg(msg, *args, **kwargs)
        if rendered_message and (not msg.endswith('\n')):
            rendered_message += ' '
        msg_len = len(rendered_message.split('\n')[-1])
        complete_str = rendered_message + confirm_str
        if yes:
            self._print(complete_str + 'y ' + cf.dimmed('[automatic, due to --yes]'))
            return True
        self._print(complete_str, _linefeed=False)
        res = None
        yes_answers = ['y', 'yes', 'true', '1']
        no_answers = ['n', 'no', 'false', '0']
        try:
            while True:
                if _timeout_s is None:
                    ans = sys.stdin.readline()
                elif sys.platform == 'win32':
                    start_time = time.time()
                    ans = ''
                    while True:
                        if time.time() - start_time >= _timeout_s:
                            self.newline()
                            ans = '\n'
                            break
                        elif msvcrt.kbhit():
                            ch = msvcrt.getwch()
                            if ch in ('\n', '\r'):
                                self.newline()
                                ans = ans + '\n'
                                break
                            elif ch == '\x08':
                                if ans:
                                    ans = ans[:-1]
                                    print('\x08 \x08', end='', flush=True)
                            else:
                                ans = ans + ch
                                print(ch, end='', flush=True)
                        else:
                            time.sleep(0.1)
                else:
                    (ready, _, _) = select.select([sys.stdin], [], [], _timeout_s)
                    if not ready:
                        self.newline()
                        ans = '\n'
                    else:
                        ans = sys.stdin.readline()
                ans = ans.lower()
                if ans == '\n':
                    res = default
                    break
                ans = ans.strip()
                if ans in yes_answers:
                    res = True
                    break
                if ans in no_answers:
                    res = False
                    break
                indent = ' ' * msg_len
                self.error('{}Invalid answer: {}. Expected {} or {}', indent, cf.bold(ans.strip()), self.render_list(yes_answers, '/'), self.render_list(no_answers, '/'))
                self._print(indent + confirm_str, _linefeed=False)
        except KeyboardInterrupt:
            self.newline()
            res = default
        if not res and should_abort:
            self._print('Exiting...')
            raise SilentClickException('Exiting due to the response to confirm(should_abort=True).')
        return res

    def prompt(self, msg: str, *args, **kwargs):
        if False:
            print('Hello World!')
        'Prompt the user for some text input.\n\n        Args:\n            msg: The mesage to display to the user before the prompt.\n\n        Returns:\n            The string entered by the user.\n        '
        complete_str = cf.underlined(msg)
        rendered_message = _format_msg(complete_str, *args, **kwargs)
        if rendered_message and (not msg.endswith('\n')):
            rendered_message += ' '
        self._print(rendered_message, linefeed=False)
        res = ''
        try:
            ans = sys.stdin.readline()
            ans = ans.lower()
            res = ans.strip()
        except KeyboardInterrupt:
            self.newline()
        return res

    def flush(self):
        if False:
            print('Hello World!')
        sys.stdout.flush()
        sys.stderr.flush()

class SilentClickException(click.ClickException):
    """`ClickException` that does not print a message.

    Some of our tooling relies on catching ClickException in particular.

    However the default prints a message, which is undesirable since we expect
    our code to log errors manually using `cli_logger.error()` to allow for
    colors and other formatting.
    """

    def __init__(self, message: str):
        if False:
            for i in range(10):
                print('nop')
        super(SilentClickException, self).__init__(message)

    def show(self, file=None):
        if False:
            while True:
                i = 10
        pass
cli_logger = _CliLogger()
CLICK_LOGGING_OPTIONS = [click.option('--log-style', required=False, type=click.Choice(cli_logger.VALID_LOG_STYLES, case_sensitive=False), default='auto', help="If 'pretty', outputs with formatting and color. If 'record', outputs record-style without formatting. 'auto' defaults to 'pretty', and disables pretty logging if stdin is *not* a TTY."), click.option('--log-color', required=False, type=click.Choice(['auto', 'false', 'true'], case_sensitive=False), default='auto', help='Use color logging. Auto enables color logging if stdout is a TTY.'), click.option('-v', '--verbose', default=None, count=True)]

def add_click_logging_options(f: Callable) -> Callable:
    if False:
        for i in range(10):
            print('nop')
    for option in reversed(CLICK_LOGGING_OPTIONS):
        f = option(f)

    @wraps(f)
    def wrapper(*args, log_style=None, log_color=None, verbose=None, **kwargs):
        if False:
            i = 10
            return i + 15
        cli_logger.configure(log_style, log_color, verbose)
        return f(*args, **kwargs)
    return wrapper