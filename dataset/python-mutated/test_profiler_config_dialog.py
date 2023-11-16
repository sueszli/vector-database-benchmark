"""Tests for plugin config dialog."""
import pytest
from spyder.plugins.profiler.plugin import Profiler
from spyder.plugins.preferences.tests.conftest import config_dialog

@pytest.mark.parametrize('config_dialog', [[None, [], [Profiler]]], indirect=True)
def test_config_dialog(config_dialog):
    if False:
        i = 10
        return i + 15
    configpage = config_dialog.get_page()
    configpage.save_to_conf()
    assert configpage