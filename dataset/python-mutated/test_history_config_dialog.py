"""Tests for plugin config dialog."""
import pytest
from spyder.plugins.history.plugin import HistoryLog
from spyder.plugins.preferences.tests.conftest import config_dialog

@pytest.mark.parametrize('config_dialog', [[None, [], [HistoryLog]]], indirect=True)
def test_config_dialog(config_dialog):
    if False:
        while True:
            i = 10
    configpage = config_dialog.get_page()
    assert configpage
    configpage.save_to_conf()