"""Tests for qutebrowser.misc.msgbox."""
import pytest
from qutebrowser.qt.core import Qt
from qutebrowser.qt.widgets import QMessageBox, QWidget
from qutebrowser.misc import msgbox
from qutebrowser.utils import utils

@pytest.fixture(autouse=True)
def patch_args(fake_args):
    if False:
        for i in range(10):
            print('nop')
    fake_args.no_err_windows = False

def test_attributes(qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Test basic QMessageBox attributes.'
    title = 'title'
    text = 'text'
    parent = QWidget()
    qtbot.add_widget(parent)
    icon = QMessageBox.Icon.Critical
    buttons = QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
    box = msgbox.msgbox(parent=parent, title=title, text=text, icon=icon, buttons=buttons)
    qtbot.add_widget(box)
    if not utils.is_mac:
        assert box.windowTitle() == title
    assert box.icon() == icon
    assert box.standardButtons() == buttons
    assert box.text() == text
    assert box.parent() is parent

@pytest.mark.parametrize('plain_text, expected', [(True, Qt.TextFormat.PlainText), (False, Qt.TextFormat.RichText), (None, Qt.TextFormat.AutoText)])
def test_plain_text(qtbot, plain_text, expected):
    if False:
        for i in range(10):
            print('nop')
    box = msgbox.msgbox(parent=None, title='foo', text='foo', icon=QMessageBox.Icon.Information, plain_text=plain_text)
    qtbot.add_widget(box)
    assert box.textFormat() == expected

def test_finished_signal(qtbot):
    if False:
        print('Hello World!')
    'Make sure we can pass a slot to be called when the dialog finished.'
    signal_triggered = False

    def on_finished():
        if False:
            print('Hello World!')
        nonlocal signal_triggered
        signal_triggered = True
    box = msgbox.msgbox(parent=None, title='foo', text='foo', icon=QMessageBox.Icon.Information, on_finished=on_finished)
    qtbot.add_widget(box)
    with qtbot.wait_signal(box.finished):
        box.accept()
    assert signal_triggered

def test_information(qtbot):
    if False:
        print('Hello World!')
    box = msgbox.information(parent=None, title='foo', text='bar')
    qtbot.add_widget(box)
    if not utils.is_mac:
        assert box.windowTitle() == 'foo'
    assert box.text() == 'bar'
    assert box.icon() == QMessageBox.Icon.Information

def test_no_err_windows(fake_args, caplog):
    if False:
        for i in range(10):
            print('nop')
    fake_args.no_err_windows = True
    box = msgbox.information(parent=None, title='foo', text='bar')
    box.exec()
    assert caplog.messages == ['foo\n\nbar']