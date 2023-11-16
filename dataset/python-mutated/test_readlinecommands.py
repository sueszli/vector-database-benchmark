import os
import re
import inspect
from qutebrowser.qt.widgets import QLineEdit, QApplication
import pytest
from qutebrowser.components import readlinecommands
fixme = pytest.mark.xfail(reason='readline compatibility - see #678')

class LineEdit(QLineEdit):
    """QLineEdit with some methods to make testing easier."""

    def _get_index(self, haystack, needle):
        if False:
            i = 10
            return i + 15
        "Get the index of a char (needle) in a string (haystack).\n\n        Return:\n            The position where needle was found, or None if it wasn't found.\n        "
        try:
            return haystack.index(needle)
        except ValueError:
            return None

    def set_aug_text(self, text):
        if False:
            for i in range(10):
                print('nop')
        'Set a text with </> markers for selected text and | as cursor.'
        real_text = re.sub('[<>|]', '', text)
        self.setText(real_text)
        cursor_pos = self._get_index(text, '|')
        sel_start_pos = self._get_index(text, '<')
        sel_end_pos = self._get_index(text, '>')
        if sel_start_pos is not None and sel_end_pos is None:
            raise ValueError('< given without >!')
        if sel_start_pos is None and sel_end_pos is not None:
            raise ValueError('> given without <!')
        if cursor_pos is not None:
            if sel_start_pos is not None or sel_end_pos is not None:
                raise ValueError("Can't mix | and </>!")
            self.setCursorPosition(cursor_pos)
        elif sel_start_pos is not None:
            if sel_start_pos > sel_end_pos:
                raise ValueError('< given after >!')
            sel_len = sel_end_pos - sel_start_pos - 1
            self.setSelection(sel_start_pos, sel_len)

    def aug_text(self):
        if False:
            while True:
                i = 10
        'Get a text with </> markers for selected text and | as cursor.'
        text = self.text()
        chars = list(text)
        cur_pos = self.cursorPosition()
        assert cur_pos >= 0
        chars.insert(cur_pos, '|')
        if self.hasSelectedText():
            selected_text = self.selectedText()
            sel_start = self.selectionStart()
            sel_end = sel_start + len(selected_text)
            assert sel_start > 0
            assert sel_end > 0
            assert sel_end > sel_start
            assert cur_pos == sel_end
            assert text[sel_start:sel_end] == selected_text
            chars.insert(sel_start, '<')
            chars.insert(sel_end + 1, '>')
        return ''.join(chars)

def _validate_deletion(lineedit, method, args, text, deleted, rest):
    if False:
        for i in range(10):
            print('nop')
    "Run and validate a text deletion method on the ReadLine bridge.\n\n    Args:\n        lineedit: The LineEdit instance.\n        method: Reference to the method on the bridge to test.\n        args: Arguments to pass to the method.\n        text: The starting 'augmented' text (see LineEdit.set_aug_text)\n        deleted: The text that should be deleted when the method is invoked.\n        rest: The augmented text that should remain after method is invoked.\n    "
    lineedit.set_aug_text(text)
    method(*args)
    assert readlinecommands.bridge._deleted[lineedit] == deleted
    assert lineedit.aug_text() == rest
    lineedit.clear()
    readlinecommands.rl_yank()
    assert lineedit.aug_text() == deleted + '|'

@pytest.fixture
def lineedit(qtbot, monkeypatch):
    if False:
        return 10
    'Fixture providing a LineEdit.'
    le = LineEdit()
    qtbot.add_widget(le)
    monkeypatch.setattr(QApplication.instance(), 'focusWidget', lambda : le)
    return le

def test_none(qtbot):
    if False:
        return 10
    'Call each rl_* method with a None focusWidget.'
    assert QApplication.instance().focusWidget() is None
    for (name, method) in inspect.getmembers(readlinecommands, inspect.isfunction):
        if name == 'rl_rubout':
            method(delim=' ')
        elif name.startswith('rl_'):
            method()

@pytest.mark.parametrize('text, expected', [('f<oo>bar', 'fo|obar'), ('|foobar', '|foobar')])
def test_rl_backward_char(text, expected, lineedit):
    if False:
        print('Hello World!')
    'Test rl_backward_char.'
    lineedit.set_aug_text(text)
    readlinecommands.rl_backward_char()
    assert lineedit.aug_text() == expected

@pytest.mark.parametrize('text, expected', [('f<oo>bar', 'foob|ar'), ('foobar|', 'foobar|')])
def test_rl_forward_char(text, expected, lineedit):
    if False:
        for i in range(10):
            print('nop')
    'Test rl_forward_char.'
    lineedit.set_aug_text(text)
    readlinecommands.rl_forward_char()
    assert lineedit.aug_text() == expected

@pytest.mark.parametrize('text, expected', [('one <tw>o', 'one |two'), ('<one >two', '|one two'), ('|one two', '|one two')])
def test_rl_backward_word(text, expected, lineedit):
    if False:
        for i in range(10):
            print('nop')
    'Test rl_backward_word.'
    lineedit.set_aug_text(text)
    readlinecommands.rl_backward_word()
    assert lineedit.aug_text() == expected

@pytest.mark.parametrize('text, expected', [pytest.param('<o>ne two', 'one| two', marks=fixme), ('<o>ne two', 'one |two'), pytest.param('<one> two', 'one two|', marks=fixme), ('<one> two', 'one |two'), ('one t<wo>', 'one two|')])
def test_rl_forward_word(text, expected, lineedit):
    if False:
        return 10
    'Test rl_forward_word.'
    lineedit.set_aug_text(text)
    readlinecommands.rl_forward_word()
    assert lineedit.aug_text() == expected

def test_rl_beginning_of_line(lineedit):
    if False:
        i = 10
        return i + 15
    'Test rl_beginning_of_line.'
    lineedit.set_aug_text('f<oo>bar')
    readlinecommands.rl_beginning_of_line()
    assert lineedit.aug_text() == '|foobar'

def test_rl_end_of_line(lineedit):
    if False:
        i = 10
        return i + 15
    'Test rl_end_of_line.'
    lineedit.set_aug_text('f<oo>bar')
    readlinecommands.rl_end_of_line()
    assert lineedit.aug_text() == 'foobar|'

@pytest.mark.parametrize('text, expected', [('foo|bar', 'foo|ar'), ('foobar|', 'foobar|'), ('|foobar', '|oobar'), ('f<oo>bar', 'f|bar')])
def test_rl_delete_char(text, expected, lineedit):
    if False:
        return 10
    'Test rl_delete_char.'
    lineedit.set_aug_text(text)
    readlinecommands.rl_delete_char()
    assert lineedit.aug_text() == expected

@pytest.mark.parametrize('text, expected', [('foo|bar', 'fo|bar'), ('foobar|', 'fooba|'), ('|foobar', '|foobar'), ('f<oo>bar', 'f|bar')])
def test_rl_backward_delete_char(text, expected, lineedit):
    if False:
        return 10
    'Test rl_backward_delete_char.'
    lineedit.set_aug_text(text)
    readlinecommands.rl_backward_delete_char()
    assert lineedit.aug_text() == expected

@pytest.mark.parametrize('text, deleted, rest', [('delete this| test', 'delete this', '| test'), pytest.param('delete <this> test', 'delete this', '| test', marks=fixme), ('delete <this> test', 'delete ', '|this test'), pytest.param('f<oo>bar', 'foo', '|bar', marks=fixme), ('f<oo>bar', 'f', '|oobar')])
def test_rl_unix_line_discard(lineedit, text, deleted, rest):
    if False:
        while True:
            i = 10
    'Delete from the cursor to the beginning of the line and yank back.'
    _validate_deletion(lineedit, readlinecommands.rl_unix_line_discard, [], text, deleted, rest)

@pytest.mark.parametrize('text, deleted, rest', [('test |delete this', 'delete this', 'test |'), pytest.param('<test >delete this', 'test delete this', 'test |', marks=fixme), ('<test >delete this', 'test delete this', '|')])
def test_rl_kill_line(lineedit, text, deleted, rest):
    if False:
        i = 10
        return i + 15
    'Delete from the cursor to the end of line and yank back.'
    _validate_deletion(lineedit, readlinecommands.rl_kill_line, [], text, deleted, rest)

@pytest.mark.parametrize('text, deleted, rest', [('test delete|foobar', 'delete', 'test |foobar'), ('test delete |foobar', 'delete ', 'test |foobar'), ('open -t github.com/foo/bar  |', 'github.com/foo/bar  ', 'open -t |'), ('open -t |github.com/foo/bar', '-t ', 'open |github.com/foo/bar'), pytest.param('test del<ete>foobar', 'delete', 'test |foobar', marks=fixme), ('test del<ete >foobar', 'del', 'test |ete foobar')])
@pytest.mark.parametrize('method, args', [(readlinecommands.rl_unix_word_rubout, []), (readlinecommands.rl_rubout, [' '])])
def test_rl_unix_word_rubout(lineedit, text, deleted, rest, method, args):
    if False:
        for i in range(10):
            print('nop')
    'Delete to word beginning and see if it comes back with yank.'
    _validate_deletion(lineedit, method, args, text, deleted, rest)

@pytest.mark.parametrize('text, deleted, rest', [('test delete|foobar', 'delete', 'test |foobar'), ('test delete |foobar', 'delete ', 'test |foobar'), ('open -t github.com/foo/bar  |', 'bar  ', 'open -t github.com/foo/|'), ('open -t |github.com/foo/bar', '-t ', 'open |github.com/foo/bar'), ('open foo/bar.baz|', 'bar.baz', 'open foo/|')])
@pytest.mark.parametrize('method, args', [(readlinecommands.rl_unix_filename_rubout, []), (readlinecommands.rl_rubout, [' /'])])
def test_rl_unix_filename_rubout(lineedit, text, deleted, rest, method, args):
    if False:
        return 10
    'Delete filename segment and see if it comes back with yank.'
    _validate_deletion(lineedit, method, args, text, deleted, rest)

@pytest.mark.parametrize('os_sep, text, deleted, rest', [('/', 'path|', 'path', '|'), ('/', '/path|', 'path', '/|'), ('/', '/path/sub|', 'sub', '/path/|'), ('/', '/path/trailing/|', 'trailing/', '/path/|'), ('/', '/test/path with spaces|', 'path with spaces', '/test/|'), ('/', '/test/path\\backslashes\\eww|', 'path\\backslashes\\eww', '/test/|'), ('\\', 'path|', 'path', '|'), ('\\', 'C:\\path|', 'path', 'C:\\|'), ('\\', 'C:\\path\\sub|', 'sub', 'C:\\path\\|'), ('\\', 'C:\\test\\path with spaces|', 'path with spaces', 'C:\\test\\|'), ('\\', 'C:\\path\\trailing\\|', 'trailing\\', 'C:\\path\\|')])
def test_filename_rubout(os_sep, monkeypatch, lineedit, text, deleted, rest):
    if False:
        i = 10
        return i + 15
    'Delete filename segment and see if it comes back with yank.'
    monkeypatch.setattr(os, 'sep', os_sep)
    _validate_deletion(lineedit, readlinecommands.rl_filename_rubout, [], text, deleted, rest)

@pytest.mark.parametrize('text, deleted, rest', [pytest.param('test foobar| delete', ' delete', 'test foobar|', marks=fixme), ('test foobar| delete', ' ', 'test foobar|delete'), pytest.param('test foo|delete bar', 'delete', 'test foo| bar', marks=fixme), ('test foo|delete bar', 'delete ', 'test foo|bar'), pytest.param('test foo<bar> delete', ' delete', 'test foobar|', marks=fixme), ('test foo<bar>delete', 'bardelete', 'test foo|')])
def test_rl_kill_word(lineedit, text, deleted, rest):
    if False:
        while True:
            i = 10
    'Delete to word end and see if it comes back with yank.'
    _validate_deletion(lineedit, readlinecommands.rl_kill_word, [], text, deleted, rest)

@pytest.mark.parametrize('text, deleted, rest', [('test delete|foobar', 'delete', 'test |foobar'), ('test delete |foobar', 'delete ', 'test |foobar'), ('open -t github.com/foo/bar  |', 'bar  ', 'open -t github.com/foo/|'), ('open -t |github.com/foo/bar', 't ', 'open -|github.com/foo/bar'), pytest.param('test del<ete>foobar', 'delete', 'test |foobar', marks=fixme), ('test del<ete >foobar', 'del', 'test |ete foobar'), ('open foo/bar.baz|', 'baz', 'open foo/bar.|')])
def test_rl_backward_kill_word(lineedit, text, deleted, rest):
    if False:
        while True:
            i = 10
    'Delete to word beginning and see if it comes back with yank.'
    _validate_deletion(lineedit, readlinecommands.rl_backward_kill_word, [], text, deleted, rest)

def test_rl_yank_no_text(lineedit):
    if False:
        for i in range(10):
            print('nop')
    'Test yank without having deleted anything.'
    lineedit.clear()
    readlinecommands.rl_yank()
    assert lineedit.aug_text() == '|'