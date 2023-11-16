from __future__ import annotations
import atexit
import contextlib
import enum
import logging
import os
import warnings
from tempfile import mktemp
from typing import TYPE_CHECKING
from rich.box import ROUNDED
from rich.console import Console
from rich.progress import Progress, ProgressColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.theme import Theme
from pdm.exceptions import PDMWarning
if TYPE_CHECKING:
    from typing import Any, Iterator, Sequence
    from pdm._types import RichProtocol, Spinner, SpinnerT
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())
unearth_logger = logging.getLogger('unearth')
unearth_logger.setLevel(logging.DEBUG)
DEFAULT_THEME = {'primary': 'cyan', 'success': 'green', 'warning': 'yellow', 'error': 'red', 'info': 'blue', 'req': 'bold green'}
_console = Console(highlight=False, theme=Theme(DEFAULT_THEME))
_err_console = Console(stderr=True, theme=Theme(DEFAULT_THEME))

def is_interactive(console: Console | None=None) -> bool:
    if False:
        while True:
            i = 10
    'Check if the terminal is run under interactive mode'
    if console is None:
        console = _console
    return console.is_interactive

def is_legacy_windows(console: Console | None=None) -> bool:
    if False:
        print('Hello World!')
    'Legacy Windows renderer may have problem rendering emojis'
    if console is None:
        console = _console
    return console.legacy_windows

def style(text: str, *args: str, style: str | None=None, **kwargs: Any) -> str:
    if False:
        for i in range(10):
            print('nop')
    'return text with ansi codes using rich console\n\n    :param text: message with rich markup, defaults to "".\n    :param style: rich style to apply to whole string\n    :return: string containing ansi codes\n    '
    with _console.capture() as capture:
        _console.print(text, *args, end='', style=style, **kwargs)
    return capture.get()

def confirm(*args: str, **kwargs: Any) -> bool:
    if False:
        print('Hello World!')
    kwargs.setdefault('default', False)
    return Confirm.ask(*args, **kwargs)

def ask(*args: str, prompt_type: type[str] | type[int] | None=None, **kwargs: Any) -> str:
    if False:
        for i in range(10):
            print('nop')
    "prompt user and return response\n\n    :prompt_type: which rich prompt to use, defaults to str.\n    :raises ValueError: unsupported prompt type\n    :return: str of user's selection\n    "
    if not prompt_type or prompt_type is str:
        return Prompt.ask(*args, **kwargs)
    elif prompt_type is int:
        return str(IntPrompt.ask(*args, **kwargs))
    else:
        raise ValueError(f'unsupported {prompt_type}')

class Verbosity(enum.IntEnum):
    QUIET = -1
    NORMAL = 0
    DETAIL = 1
    DEBUG = 2
LOG_LEVELS = {Verbosity.NORMAL: logging.WARN, Verbosity.DETAIL: logging.INFO, Verbosity.DEBUG: logging.DEBUG}

class Emoji:
    if is_legacy_windows():
        SUCC = 'v'
        FAIL = 'x'
        LOCK = ' '
        CONGRAT = ' '
        POPPER = ' '
        ELLIPSIS = '...'
        ARROW_SEPARATOR = '>'
    else:
        SUCC = ':heavy_check_mark:'
        FAIL = ':heavy_multiplication_x:'
        LOCK = ':lock:'
        POPPER = ':party_popper:'
        ELLIPSIS = '…'
        ARROW_SEPARATOR = '➤'
if is_legacy_windows():
    SPINNER = 'line'
else:
    SPINNER = 'dots'

class DummySpinner:
    """A dummy spinner class implementing needed interfaces.
    But only display text onto screen.
    """

    def __init__(self, text: str) -> None:
        if False:
            return 10
        self.text = text

    def _show(self) -> None:
        if False:
            i = 10
            return i + 15
        _err_console.print(f'[primary]STATUS:[/] {self.text}')

    def update(self, text: str) -> None:
        if False:
            while True:
                i = 10
        self.text = text
        self._show()

    def __enter__(self: SpinnerT) -> SpinnerT:
        if False:
            for i in range(10):
                print('nop')
        self._show()
        return self

    def __exit__(self, *args: Any) -> None:
        if False:
            i = 10
            return i + 15
        pass

class SilentSpinner(DummySpinner):

    def _show(self) -> None:
        if False:
            print('Hello World!')
        pass

class UI:
    """Terminal UI object"""

    def __init__(self, verbosity: Verbosity=Verbosity.NORMAL) -> None:
        if False:
            i = 10
            return i + 15
        self.verbosity = verbosity

    def set_verbosity(self, verbosity: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.verbosity = Verbosity(verbosity)
        if self.verbosity == Verbosity.QUIET:
            warnings.simplefilter('ignore', PDMWarning, append=True)
            warnings.simplefilter('ignore', FutureWarning, append=True)

    def set_theme(self, theme: Theme) -> None:
        if False:
            return 10
        'set theme for rich console\n\n        :param theme: dict of theme\n        '
        _console.push_theme(theme)
        _err_console.push_theme(theme)

    def echo(self, message: str | RichProtocol='', err: bool=False, verbosity: Verbosity=Verbosity.QUIET, **kwargs: Any) -> None:
        if False:
            return 10
        'print message using rich console\n\n        :param message: message with rich markup, defaults to "".\n        :param err: if true print to stderr, defaults to False.\n        :param verbosity: verbosity level, defaults to NORMAL.\n        '
        if self.verbosity >= verbosity:
            console = _err_console if err else _console
            if not console.is_interactive:
                kwargs.setdefault('crop', False)
                kwargs.setdefault('overflow', 'ignore')
            console.print(message, **kwargs)

    def display_columns(self, rows: Sequence[Sequence[str]], header: list[str] | None=None) -> None:
        if False:
            while True:
                i = 10
        'Print rows in aligned columns.\n\n        :param rows: a rows of data to be displayed.\n        :param header: a list of header strings.\n        '
        if header:
            table = Table(box=ROUNDED)
            for title in header:
                if title[0] == '^':
                    (title, justify) = (title[1:], 'center')
                elif title[0] == '>':
                    (title, justify) = (title[1:], 'right')
                else:
                    (title, justify) = (title, 'left')
                table.add_column(title, justify=justify)
        else:
            table = Table.grid(padding=(0, 1))
            for _ in rows[0]:
                table.add_column()
        for row in rows:
            table.add_row(*row)
        _console.print(table)

    @contextlib.contextmanager
    def logging(self, type_: str='install') -> Iterator[logging.Logger]:
        if False:
            print('Hello World!')
        'A context manager that opens a file for logging when verbosity is NORMAL or\n        print to the stdout otherwise.\n        '
        file_name: str | None = None
        if self.verbosity >= Verbosity.DETAIL:
            handler: logging.Handler = logging.StreamHandler()
            handler.setLevel(LOG_LEVELS[self.verbosity])
        else:
            file_name = mktemp('.log', f'pdm-{type_}-')
            handler = logging.FileHandler(file_name, encoding='utf-8')
            handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(name)s: %(message)s'))
        logger.addHandler(handler)
        unearth_logger.addHandler(handler)

        def cleanup() -> None:
            if False:
                print('Hello World!')
            if not file_name:
                return
            with contextlib.suppress(OSError):
                os.unlink(file_name)
        try:
            yield logger
        except Exception:
            if self.verbosity < Verbosity.DETAIL:
                logger.exception('Error occurs')
                self.echo(f'See [warning]{file_name}[/] for detailed debug log.', style='error', err=True)
            raise
        else:
            atexit.register(cleanup)
        finally:
            logger.removeHandler(handler)
            unearth_logger.removeHandler(handler)
            handler.close()

    def open_spinner(self, title: str) -> Spinner:
        if False:
            print('Hello World!')
        'Open a spinner as a context manager.'
        if self.verbosity >= Verbosity.DETAIL or not is_interactive():
            return DummySpinner(title)
        else:
            return _err_console.status(title, spinner=SPINNER, spinner_style='primary')

    def make_progress(self, *columns: str | ProgressColumn, **kwargs: Any) -> Progress:
        if False:
            i = 10
            return i + 15
        'create a progress instance for indented spinners'
        return Progress(*columns, console=_console, disable=self.verbosity >= Verbosity.DETAIL, **kwargs)