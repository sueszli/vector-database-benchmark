"""Tests for plugin config dialog."""
import pytest
from spyder.plugins.workingdirectory.plugin import WorkingDirectory
from spyder.plugins.preferences.tests.conftest import config_dialog

@pytest.mark.parametrize('config_dialog', [[None, [], [WorkingDirectory]]], indirect=True)
def test_config_dialog(qtbot, config_dialog):
    if False:
        for i in range(10):
            print('nop')
    configpage = config_dialog.get_page()
    configpage.save_to_conf()
    assert configpage