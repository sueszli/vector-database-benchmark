"""Tests for plugin config dialog."""
from unittest.mock import MagicMock
import pytest
from qtpy.QtWidgets import QMainWindow
from spyder.plugins.ipythonconsole.plugin import IPythonConsole
from spyder.plugins.preferences.tests.conftest import config_dialog

class MainWindowMock(QMainWindow):
    register_shortcut = MagicMock()
    editor = MagicMock()

    def __getattr__(self, attr):
        if False:
            while True:
                i = 10
        return MagicMock()

@pytest.mark.parametrize('config_dialog', [[MainWindowMock, [], [IPythonConsole]]], indirect=True)
def test_config_dialog(config_dialog):
    if False:
        i = 10
        return i + 15
    configpage = config_dialog.get_page()
    assert configpage
    configpage.save_to_conf()