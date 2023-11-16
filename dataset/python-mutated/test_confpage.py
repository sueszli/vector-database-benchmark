import sys
import time
import pytest
from spyder.api.plugins import Plugins
from spyder.api.plugin_registration.registry import PLUGIN_REGISTRY
from spyder.plugins.maininterpreter.plugin import MainInterpreter
from spyder.plugins.preferences.tests.conftest import MainWindowMock
from spyder.utils.conda import get_list_conda_envs
from spyder.utils.pyenv import get_list_pyenv_envs
t0 = time.time()
conda_envs = get_list_conda_envs()
pyenv_envs = get_list_pyenv_envs()
GET_ENVS_TIME = time.time() - t0

@pytest.mark.skipif(len(conda_envs) == 0 and len(pyenv_envs) == 0 or sys.platform == 'darwin', reason='Makes no sense if conda and pyenv are not installed, fails on mac')
def test_load_time(qtbot):
    if False:
        print('Hello World!')
    main = MainWindowMock(None)
    preferences = main.get_plugin(Plugins.Preferences)
    PLUGIN_REGISTRY.register_plugin(main, MainInterpreter)
    t0 = time.time()
    preferences.open_dialog()
    load_time = time.time() - t0
    container = preferences.get_container()
    dlg = container.dialog
    widget = dlg.get_page()
    assert widget.cus_exec_combo.combobox.count() > 0
    assert load_time < GET_ENVS_TIME
    assert load_time < 0.5