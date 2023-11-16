"""Tests for plugin config dialog."""
import pytest
from spyder.plugins.explorer.plugin import Explorer
from spyder.plugins.preferences.tests.conftest import config_dialog

@pytest.mark.parametrize('config_dialog', [[None, [], [Explorer]]], indirect=True)
def test_config_dialog(config_dialog):
    if False:
        i = 10
        return i + 15
    configpage = config_dialog.get_page()
    assert configpage
    configpage.save_to_conf()