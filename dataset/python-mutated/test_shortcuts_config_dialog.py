"""Tests for plugin config dialog."""
import pytest
from spyder.plugins.shortcuts.plugin import Shortcuts
from spyder.plugins.preferences.tests.conftest import config_dialog

@pytest.mark.parametrize('config_dialog', [[None, [], [Shortcuts]]], indirect=True)
def test_config_dialog(config_dialog):
    if False:
        return 10
    configpage = config_dialog.get_page()
    configpage.save_to_conf()
    assert configpage