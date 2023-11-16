"""
Tests for helperwidgets.py
"""
import pytest
from qtpy.QtWidgets import QMessageBox
from spyder.widgets.helperwidgets import MessageCheckBox

@pytest.fixture
def messagecheckbox(qtbot):
    if False:
        while True:
            i = 10
    'Set up MessageCheckBox.'
    widget = MessageCheckBox()
    qtbot.addWidget(widget)
    return widget

def test_messagecheckbox(messagecheckbox, qtbot):
    if False:
        print('Hello World!')
    'Run Message Checkbox.'
    box = messagecheckbox
    box.setWindowTitle('Spyder updates')
    box.setText('Testing checkbox')
    box.set_checkbox_text('Check for updates on startup?')
    box.setStandardButtons(QMessageBox.Ok)
    box.setDefaultButton(QMessageBox.Ok)
    box.setIcon(QMessageBox.Information)
    box.show()
    assert box
if __name__ == '__main__':
    pytest.main()