"""Qt wrapper selection.

Contains selection logic and globals for Qt wrapper selection.

All other files in this package are intended to be simple wrappers around Qt imports.
Depending on what is set in this module, they import from PyQt5 or PyQt6.

The import wrappers are intended to be as thin as possible. They will not unify
API-level differences between Qt 5 and Qt 6. This is best handled by the calling code,
which has a better picture of what changed between APIs and how to best handle it.

What they *will* do is handle simple 1:1 renames of classes, or moves between
modules (where they aim to always expose the Qt 6 API). See e.g. webenginecore.py.
"""
import os
import sys
import enum
import html
import argparse
import warnings
import importlib
import dataclasses
from typing import Optional, Dict
from qutebrowser.utils import log
_WRAPPER_OVERRIDE = None
WRAPPERS = ['PyQt6', 'PyQt5']

class Error(Exception):
    """Base class for all exceptions in this module."""

class Unavailable(Error, ImportError):
    """Raised when a module is unavailable with the given wrapper."""

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        super().__init__(f'Unavailable with {INFO.wrapper}')

class NoWrapperAvailableError(Error, ImportError):
    """Raised when no Qt wrapper is available."""

    def __init__(self, info: 'SelectionInfo') -> None:
        if False:
            while True:
                i = 10
        super().__init__(f'No Qt wrapper was importable.\n\n{info}')

class UnknownWrapper(Error):
    """Raised when an Qt module is imported but the wrapper values are unknown.

    Should never happen (unless a new wrapper is added).
    """

class SelectionReason(enum.Enum):
    """Reasons for selecting a Qt wrapper."""
    cli = '--qt-wrapper'
    env = 'QUTE_QT_WRAPPER'
    auto = 'autoselect'
    default = 'default'
    fake = 'fake'
    override = 'override'
    unknown = 'unknown'

@dataclasses.dataclass
class SelectionInfo:
    """Information about outcomes of importing Qt wrappers."""
    wrapper: Optional[str] = None
    outcomes: Dict[str, str] = dataclasses.field(default_factory=dict)
    reason: SelectionReason = SelectionReason.unknown

    def set_module_error(self, name: str, error: Exception) -> None:
        if False:
            print('Hello World!')
        'Set the outcome for a module import.'
        self.outcomes[name] = f'{type(error).__name__}: {error}'

    def use_wrapper(self, wrapper: str) -> None:
        if False:
            print('Hello World!')
        'Set the wrapper to use.'
        self.wrapper = wrapper
        self.outcomes[wrapper] = 'success'

    def __str__(self) -> str:
        if False:
            return 10
        if not self.outcomes:
            return f'Qt wrapper: {self.wrapper} (via {self.reason.value})'
        lines = ['Qt wrapper info:']
        for wrapper in WRAPPERS:
            outcome = self.outcomes.get(wrapper, 'not imported')
            lines.append(f'  {wrapper}: {outcome}')
        lines.append(f'  -> selected: {self.wrapper} (via {self.reason.value})')
        return '\n'.join(lines)

    def to_html(self) -> str:
        if False:
            i = 10
            return i + 15
        return html.escape(str(self)).replace('\n', '<br>')

def _autoselect_wrapper() -> SelectionInfo:
    if False:
        for i in range(10):
            print('nop')
    'Autoselect a Qt wrapper.\n\n    This goes through all wrappers defined in WRAPPER.\n    The first one which can be imported is returned.\n    '
    info = SelectionInfo(reason=SelectionReason.auto)
    for wrapper in WRAPPERS:
        try:
            importlib.import_module(wrapper)
        except ModuleNotFoundError as e:
            info.set_module_error(wrapper, e)
            continue
        except ImportError as e:
            info.set_module_error(wrapper, e)
            break
        info.use_wrapper(wrapper)
        return info
    return info

def _select_wrapper(args: Optional[argparse.Namespace]) -> SelectionInfo:
    if False:
        for i in range(10):
            print('nop')
    'Select a Qt wrapper.\n\n    - If --qt-wrapper is given, use that.\n    - Otherwise, if the QUTE_QT_WRAPPER environment variable is set, use that.\n    - Otherwise, try the wrappers in WRAPPER in order (PyQt6 -> PyQt5)\n    '
    for name in WRAPPERS:
        if name in sys.modules:
            warnings.warn(f'{name} already imported', stacklevel=1)
    if args is not None and args.qt_wrapper is not None:
        assert args.qt_wrapper in WRAPPERS, args.qt_wrapper
        return SelectionInfo(wrapper=args.qt_wrapper, reason=SelectionReason.cli)
    env_var = 'QUTE_QT_WRAPPER'
    env_wrapper = os.environ.get(env_var)
    if env_wrapper:
        if env_wrapper == 'auto':
            return _autoselect_wrapper()
        elif env_wrapper not in WRAPPERS:
            raise Error(f"Unknown wrapper {env_wrapper} set via {env_var}, allowed: {', '.join(WRAPPERS)}")
        return SelectionInfo(wrapper=env_wrapper, reason=SelectionReason.env)
    if _WRAPPER_OVERRIDE is not None:
        assert _WRAPPER_OVERRIDE in WRAPPERS
        return SelectionInfo(wrapper=_WRAPPER_OVERRIDE, reason=SelectionReason.override)
    return _autoselect_wrapper()
INFO: SelectionInfo
USE_PYQT5: bool
USE_PYQT6: bool
USE_PYSIDE6: bool
IS_QT5: bool
IS_QT6: bool
IS_PYQT: bool
IS_PYSIDE: bool
_initialized = False

def _set_globals(info: SelectionInfo) -> None:
    if False:
        while True:
            i = 10
    'Set all global variables in this module based on the given SelectionInfo.\n\n    Those are split into multiple global variables because that way we can teach mypy\n    about them via --always-true and --always-false, see tox.ini.\n    '
    global INFO, USE_PYQT5, USE_PYQT6, USE_PYSIDE6, IS_QT5, IS_QT6, IS_PYQT, IS_PYSIDE, _initialized
    assert info.wrapper is not None, info
    assert not _initialized
    _initialized = True
    INFO = info
    USE_PYQT5 = info.wrapper == 'PyQt5'
    USE_PYQT6 = info.wrapper == 'PyQt6'
    USE_PYSIDE6 = info.wrapper == 'PySide6'
    assert USE_PYQT5 + USE_PYQT6 + USE_PYSIDE6 == 1
    IS_QT5 = USE_PYQT5
    IS_QT6 = USE_PYQT6 or USE_PYSIDE6
    IS_PYQT = USE_PYQT5 or USE_PYQT6
    IS_PYSIDE = USE_PYSIDE6
    assert IS_QT5 ^ IS_QT6
    assert IS_PYQT ^ IS_PYSIDE

def init_implicit() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Initialize Qt wrapper globals implicitly at Qt import time.\n\n    This gets called when any qutebrowser.qt module is imported, and implicitly\n    initializes the Qt wrapper globals.\n\n    After this is called, no explicit initialization via machinery.init() is possible\n    anymore - thus, this should never be called before init() when running qutebrowser\n    as an application (and any further calls will be a no-op).\n\n    However, this ensures that any qutebrowser module can be imported without\n    having to worry about machinery.init().  This is useful for e.g. tests or\n    manual interactive usage of the qutebrowser code.\n    '
    if _initialized:
        return
    info = _select_wrapper(args=None)
    if info.wrapper is None:
        raise NoWrapperAvailableError(info)
    _set_globals(info)

def init(args: argparse.Namespace) -> SelectionInfo:
    if False:
        return 10
    "Initialize Qt wrapper globals during qutebrowser application start.\n\n    This gets called from earlyinit.py, i.e. after we have an argument parser,\n    but before any kinds of Qt usage. This allows `args` to be passed, which is\n    used to select the Qt wrapper (if --qt-wrapper is given).\n\n    If any qutebrowser.qt module is imported before this, init_implicit() will be called\n    instead, which means this can't be called anymore.\n    "
    if _initialized:
        raise Error('init() already called before application init')
    info = _select_wrapper(args)
    if info.wrapper is not None:
        _set_globals(info)
        log.init.debug(str(info))
    return info