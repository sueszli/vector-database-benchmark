"""Tests for workingdirectory plugin."""
import os
import os.path as osp
import sys
from unittest.mock import Mock
import pytest
from qtpy.QtWidgets import QMainWindow
from qtpy.QtCore import Qt
from spyder.app.cli_options import get_options
from spyder.config.base import get_home_dir
from spyder.config.manager import CONF
from spyder.plugins.workingdirectory.plugin import WorkingDirectory
NEW_DIR = 'new_workingdir'

class MainWindow(QMainWindow):

    def __init__(self):
        if False:
            print('Hello World!')
        sys_argv = [sys.argv[0]]
        self._cli_options = get_options(sys_argv)[0]
        super().__init__()

    def get_plugin(self, plugin, error=True):
        if False:
            print('Hello World!')
        return Mock()

@pytest.fixture
def setup_workingdirectory(qtbot, request, tmpdir):
    if False:
        while True:
            i = 10
    'Setup working directory plugin.'
    CONF.reset_to_defaults()
    use_startup_wdir = request.node.get_closest_marker('use_startup_wdir')
    use_cli_wdir = request.node.get_closest_marker('use_cli_wdir')
    CONF.set('workingdir', 'startup/use_project_or_home_directory', True)
    CONF.set('workingdir', 'startup/use_fixed_directory', False)
    main_window = MainWindow()
    if use_startup_wdir:
        new_wdir = tmpdir.mkdir(NEW_DIR + '_startup')
        CONF.set('workingdir', 'startup/use_project_or_home_directory', False)
        CONF.set('workingdir', 'startup/use_fixed_directory', True)
        CONF.set('workingdir', 'startup/fixed_directory', str(new_wdir))
    elif use_cli_wdir:
        new_wdir = tmpdir.mkdir(NEW_DIR + '_cli')
        main_window._cli_options.working_directory = str(new_wdir)
    workingdirectory = WorkingDirectory(main_window, configuration=CONF)
    workingdirectory.on_initialize()
    workingdirectory.close = lambda : True
    return workingdirectory

def test_basic_initialization(setup_workingdirectory):
    if False:
        return 10
    'Test Working Directory plugin initialization.'
    workingdirectory = setup_workingdirectory
    assert workingdirectory is not None

def test_get_workingdir(setup_workingdirectory):
    if False:
        return 10
    'Test the method that defines the working directory at home.'
    workingdirectory = setup_workingdirectory
    act_wdir = workingdirectory.get_workdir()
    assert act_wdir == get_home_dir()

@pytest.mark.use_startup_wdir
def test_get_workingdir_startup(setup_workingdirectory):
    if False:
        while True:
            i = 10
    '\n    Test the method that sets the working directory according to the one\n    selected in preferences.\n    '
    workingdirectory = setup_workingdirectory
    cwd = workingdirectory.get_workdir()
    folders = osp.split(cwd)
    assert folders[-1] == NEW_DIR + '_startup'
    CONF.reset_to_defaults()

@pytest.mark.use_cli_wdir
def test_get_workingdir_cli(setup_workingdirectory):
    if False:
        while True:
            i = 10
    '\n    Test that the plugin sets the working directory passed by users on the\n    command line with the --workdir option.\n    '
    workingdirectory = setup_workingdirectory
    cwd = workingdirectory.get_container().history[-1]
    folders = osp.split(cwd)
    assert folders[-1] == NEW_DIR + '_cli'
    CONF.reset_to_defaults()

def test_file_goto(qtbot, setup_workingdirectory):
    if False:
        print('Hello World!')
    '\n    Test that putting a file in the workingdirectory emits a edit_goto signal.\n    '
    container = setup_workingdirectory.get_container()
    signal_res = {}

    def test_slot(filename, line, word):
        if False:
            while True:
                i = 10
        signal_res['filename'] = filename
        signal_res['line'] = line
    container.edit_goto.connect(test_slot)
    pathedit = container.pathedit
    wd = setup_workingdirectory.get_workdir()
    filename = osp.join(wd, 'myfile_workingdirectory_test.py')
    with open(filename, 'w') as f:
        f.write('\n' * 5)
    with qtbot.waitSignal(container.edit_goto):
        pathedit.add_text(filename + ':1')
        qtbot.keyClick(pathedit, Qt.Key_Return)
    assert signal_res['filename'] in filename
    assert signal_res['line'] == 1
    os.remove(filename)
if __name__ == '__main__':
    pytest.main()