"""Bridge to provide readline-like shortcuts for QLineEdits."""
import os
from typing import Iterable, Optional, MutableMapping, Any, Callable
from qutebrowser.qt.widgets import QApplication, QLineEdit
from qutebrowser.api import cmdutils

class _ReadlineBridge:
    """Bridge which provides readline-like commands for the current QLineEdit.

    Attributes:
        _deleted: Mapping from widgets to their last deleted text.
    """

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._deleted: MutableMapping[QLineEdit, str] = {}

    def _widget(self) -> Optional[QLineEdit]:
        if False:
            print('Hello World!')
        'Get the currently active QLineEdit.'
        qapp = QApplication.instance()
        assert isinstance(qapp, QApplication), qapp
        w = qapp.focusWidget()
        if isinstance(w, QLineEdit):
            return w
        else:
            return None

    def _dispatch(self, name: str, *, mark: bool=None, delete: bool=False) -> None:
        if False:
            while True:
                i = 10
        widget = self._widget()
        if widget is None:
            return
        method = getattr(widget, name)
        if mark is None:
            method()
        else:
            method(mark)
        if delete:
            self._deleted[widget] = widget.selectedText()
            widget.del_()

    def backward_char(self) -> None:
        if False:
            while True:
                i = 10
        self._dispatch('cursorBackward', mark=False)

    def forward_char(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._dispatch('cursorForward', mark=False)

    def backward_word(self) -> None:
        if False:
            i = 10
            return i + 15
        self._dispatch('cursorWordBackward', mark=False)

    def forward_word(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._dispatch('cursorWordForward', mark=False)

    def beginning_of_line(self) -> None:
        if False:
            i = 10
            return i + 15
        self._dispatch('home', mark=False)

    def end_of_line(self) -> None:
        if False:
            i = 10
            return i + 15
        self._dispatch('end', mark=False)

    def unix_line_discard(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._dispatch('home', mark=True, delete=True)

    def kill_line(self) -> None:
        if False:
            while True:
                i = 10
        self._dispatch('end', mark=True, delete=True)

    def rubout(self, delim: Iterable[str]) -> None:
        if False:
            return 10
        "Delete backwards using the characters in delim as boundaries.\n\n        With delim=[' '], this acts like unix-word-rubout.\n        With delim=[' ', '/'], this acts like unix-filename-rubout.\n        With delim=[os.sep], this serves as a more useful filename-rubout.\n        "
        widget = self._widget()
        if widget is None:
            return
        cursor_position = widget.cursorPosition()
        text = widget.text()
        target_position = cursor_position
        is_boundary = True
        while is_boundary and target_position > 0:
            is_boundary = text[target_position - 1] in delim
            target_position -= 1
        is_boundary = False
        while not is_boundary and target_position > 0:
            is_boundary = text[target_position - 1] in delim
            target_position -= 1
        if not is_boundary:
            assert target_position == 0, (text, delim)
            target_position -= 1
        moveby = cursor_position - target_position - 1
        widget.cursorBackward(True, moveby)
        self._deleted[widget] = widget.selectedText()
        widget.del_()

    def backward_kill_word(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._dispatch('cursorWordBackward', mark=True, delete=True)

    def kill_word(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._dispatch('cursorWordForward', mark=True, delete=True)

    def yank(self) -> None:
        if False:
            while True:
                i = 10
        'Paste previously deleted text.'
        widget = self._widget()
        if widget is None or widget not in self._deleted:
            return
        widget.insert(self._deleted[widget])

    def delete_char(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._dispatch('del_')

    def backward_delete_char(self) -> None:
        if False:
            print('Hello World!')
        self._dispatch('backspace')
bridge = _ReadlineBridge()

def _register(**kwargs: Any) -> Callable[..., Any]:
    if False:
        print('Hello World!')
    return cmdutils.register(modes=[cmdutils.KeyMode.command, cmdutils.KeyMode.prompt], **kwargs)

@_register()
def rl_backward_char() -> None:
    if False:
        print('Hello World!')
    "Move back a character.\n\n    This acts like readline's backward-char.\n    "
    bridge.backward_char()

@_register()
def rl_forward_char() -> None:
    if False:
        i = 10
        return i + 15
    "Move forward a character.\n\n    This acts like readline's forward-char.\n    "
    bridge.forward_char()

@_register()
def rl_backward_word() -> None:
    if False:
        print('Hello World!')
    "Move back to the start of the current or previous word.\n\n    This acts like readline's backward-word.\n    "
    bridge.backward_word()

@_register()
def rl_forward_word() -> None:
    if False:
        return 10
    "Move forward to the end of the next word.\n\n    This acts like readline's forward-word.\n    "
    bridge.forward_word()

@_register()
def rl_beginning_of_line() -> None:
    if False:
        print('Hello World!')
    "Move to the start of the line.\n\n    This acts like readline's beginning-of-line.\n    "
    bridge.beginning_of_line()

@_register()
def rl_end_of_line() -> None:
    if False:
        return 10
    "Move to the end of the line.\n\n    This acts like readline's end-of-line.\n    "
    bridge.end_of_line()

@_register()
def rl_unix_line_discard() -> None:
    if False:
        return 10
    "Remove chars backward from the cursor to the beginning of the line.\n\n    This acts like readline's unix-line-discard.\n    "
    bridge.unix_line_discard()

@_register()
def rl_kill_line() -> None:
    if False:
        i = 10
        return i + 15
    "Remove chars from the cursor to the end of the line.\n\n    This acts like readline's kill-line.\n    "
    bridge.kill_line()

@_register(deprecated="Use :rl-rubout ' ' instead.")
def rl_unix_word_rubout() -> None:
    if False:
        while True:
            i = 10
    "Remove chars from the cursor to the beginning of the word.\n\n    This acts like readline's unix-word-rubout. Whitespace is used as a\n    word delimiter.\n    "
    bridge.rubout([' '])

@_register(deprecated='Use :rl-filename-rubout or :rl-rubout " /" instead (see their `:help` for details).')
def rl_unix_filename_rubout() -> None:
    if False:
        print('Hello World!')
    "Remove chars from the cursor to the previous path separator.\n\n    This acts like readline's unix-filename-rubout.\n    "
    bridge.rubout([' ', '/'])

@_register()
def rl_rubout(delim: str) -> None:
    if False:
        while True:
            i = 10
    'Delete backwards using the given characters as boundaries.\n\n    With " ", this acts like readline\'s `unix-word-rubout`.\n\n    With " /", this acts like readline\'s `unix-filename-rubout`, but consider\n    using `:rl-filename-rubout` instead: It uses the OS path separator (i.e. `\\`\n    on Windows) and ignores spaces.\n\n    Args:\n        delim: A string of characters (or a single character) until which text\n               will be deleted.\n    '
    bridge.rubout(list(delim))

@_register()
def rl_filename_rubout() -> None:
    if False:
        return 10
    'Delete backwards using the OS path separator as boundary.\n\n    For behavior that matches readline\'s `unix-filename-rubout` exactly, use\n    `:rl-rubout "/ "` instead. This command uses the OS path separator (i.e.\n    `\\` on Windows) and ignores spaces.\n    '
    bridge.rubout(os.sep)

@_register()
def rl_backward_kill_word() -> None:
    if False:
        return 10
    "Remove chars from the cursor to the beginning of the word.\n\n    This acts like readline's backward-kill-word. Any non-alphanumeric\n    character is considered a word delimiter.\n    "
    bridge.backward_kill_word()

@_register()
def rl_kill_word() -> None:
    if False:
        return 10
    "Remove chars from the cursor to the end of the current word.\n\n    This acts like readline's kill-word.\n    "
    bridge.kill_word()

@_register()
def rl_yank() -> None:
    if False:
        for i in range(10):
            print('nop')
    "Paste the most recently deleted text.\n\n    This acts like readline's yank.\n    "
    bridge.yank()

@_register()
def rl_delete_char() -> None:
    if False:
        i = 10
        return i + 15
    "Delete the character after the cursor.\n\n    This acts like readline's delete-char.\n    "
    bridge.delete_char()

@_register()
def rl_backward_delete_char() -> None:
    if False:
        return 10
    "Delete the character before the cursor.\n\n    This acts like readline's backward-delete-char.\n    "
    bridge.backward_delete_char()