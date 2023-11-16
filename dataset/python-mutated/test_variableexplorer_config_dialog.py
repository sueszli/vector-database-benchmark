"""Tests for plugin config dialog."""
import pytest
from spyder.plugins.variableexplorer.plugin import VariableExplorer
from spyder.plugins.preferences.tests.conftest import config_dialog

@pytest.mark.parametrize('config_dialog', [[None, [], [VariableExplorer]]], indirect=True)
def test_config_dialog(config_dialog):
    if False:
        while True:
            i = 10
    configpage = config_dialog.get_page()
    configpage.save_to_conf()
    assert configpage