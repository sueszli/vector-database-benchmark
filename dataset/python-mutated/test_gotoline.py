"""Tests for gotoline.py"""
from qtpy.QtWidgets import QDialogButtonBox, QLineEdit
from spyder.plugins.editor.widgets.gotoline import GoToLineDialog

def test_gotolinedialog_has_cancel_button(codeeditor, qtbot, tmpdir):
    if False:
        print('Hello World!')
    '\n    Test that GoToLineDialog has a Cancel button.\n\n    Test that a GoToLineDialog has a button in a dialog button box and that\n    this button cancels the dialog window.\n    '
    editor = codeeditor
    dialog = GoToLineDialog(editor)
    qtbot.addWidget(dialog)
    ok_button = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok)
    cancel_button = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Cancel)
    assert not ok_button.isEnabled()
    with qtbot.waitSignal(dialog.rejected):
        cancel_button.click()

def test_gotolinedialog_enter_plus(codeeditor, qtbot):
    if False:
        while True:
            i = 10
    '\n    Regression test for spyder-ide/spyder#12693\n    '
    editor = codeeditor
    dialog = GoToLineDialog(editor)
    qtbot.addWidget(dialog)
    ok_button = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok)
    cancel_button = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Cancel)
    lineedit = dialog.findChild(QLineEdit)
    lineedit.setText('+')
    lineedit.setText('+')
    assert lineedit.text() == ''
    assert not ok_button.isEnabled()

def test_gotolinedialog_check_valid(codeeditor, qtbot):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check ok button enabled if valid text entered\n    '
    editor = codeeditor
    dialog = GoToLineDialog(editor)
    qtbot.addWidget(dialog)
    ok_button = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok)
    lineedit = dialog.findChild(QLineEdit)
    lineedit.setText('1')
    assert lineedit.text() == '1'
    assert ok_button.isEnabled()
    with qtbot.waitSignal(dialog.accepted):
        ok_button.click()
    assert dialog.get_line_number() == 1