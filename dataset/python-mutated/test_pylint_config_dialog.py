"""Tests for plugin config dialog."""
from unittest.mock import MagicMock
from qtpy.QtCore import Signal, QObject
from qtpy.QtWidgets import QApplication, QMainWindow
import pytest
if QApplication.instance() is None:
    app = QApplication([])
from spyder.plugins.pylint.plugin import Pylint
from spyder.plugins.preferences.tests.conftest import config_dialog

class MainWindowMock(QMainWindow):
    sig_editor_focus_changed = Signal(str)

    def __init__(self, parent):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.editor = MagicMock()
        self.editor.sig_editor_focus_changed = self.sig_editor_focus_changed
        self.projects = MagicMock()

@pytest.mark.parametrize('config_dialog', [[MainWindowMock, [], [Pylint]]], indirect=True)
def test_config_dialog(config_dialog):
    if False:
        return 10
    configpage = config_dialog.get_page()
    configpage.save_to_conf()
    assert configpage