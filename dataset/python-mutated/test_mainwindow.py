"""
Tests for the main window.
"""
import gc
import os
import os.path as osp
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
from textwrap import dedent
import time
from unittest.mock import Mock
import uuid
from flaky import flaky
import ipykernel
from IPython.core import release as ipy_release
from matplotlib.testing.compare import compare_images
import nbconvert
import numpy as np
from numpy.testing import assert_array_equal
from packaging.version import parse
import pylint
import pytest
from qtpy import PYQT_VERSION, PYQT5
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QImage, QTextCursor
from qtpy.QtWidgets import QAction, QApplication, QInputDialog, QWidget
from qtpy.QtWebEngineWidgets import WEBENGINE
from spyder import __trouble_url__
from spyder.api.utils import get_class_values
from spyder.api.widgets.auxiliary_widgets import SpyderWindowWidget
from spyder.api.plugins import Plugins
from spyder.app.tests.conftest import COMPILE_AND_EVAL_TIMEOUT, COMPLETION_TIMEOUT, EVAL_TIMEOUT, generate_run_parameters, find_desired_tab_in_window, LOCATION, open_file_in_editor, preferences_dialog_helper, read_asset_file, reset_run_code, SHELL_TIMEOUT, start_new_kernel
from spyder.config.base import get_home_dir, get_conf_path, get_module_path, running_in_ci
from spyder.config.manager import CONF
from spyder.dependencies import DEPENDENCIES
from spyder.plugins.debugger.api import DebuggerWidgetActions
from spyder.plugins.externalterminal.api import ExtTerminalShConfiguration
from spyder.plugins.help.widgets import ObjectComboBox
from spyder.plugins.help.tests.test_plugin import check_text
from spyder.plugins.ipythonconsole.utils.kernel_handler import KernelHandler
from spyder.plugins.ipythonconsole.api import IPythonConsolePyConfiguration
from spyder.plugins.mainmenu.api import ApplicationMenus
from spyder.plugins.layout.layouts import DefaultLayouts
from spyder.plugins.toolbar.api import ApplicationToolbars
from spyder.plugins.run.api import RunExecutionParameters, ExtendedRunExecutionParameters, WorkingDirOpts, WorkingDirSource, RunContext
from spyder.py3compat import qbytearray_to_str, to_text_string
from spyder.utils.environ import set_user_env
from spyder.utils.misc import remove_backslashes, rename_file
from spyder.utils.clipboard_helper import CLIPBOARD_HELPER
from spyder.utils.programs import find_program
from spyder.widgets.dock import DockTitleBar

@pytest.mark.order(1)
@pytest.mark.single_instance
@pytest.mark.known_leak
@pytest.mark.skipif(not running_in_ci(), reason="It's not meant to be run outside of CIs")
def test_single_instance_and_edit_magic(main_window, qtbot, tmpdir):
    if False:
        i = 10
        return i + 15
    'Test single instance mode and %edit magic.'
    editorstack = main_window.editor.get_current_editorstack()
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    spy_dir = osp.dirname(get_module_path('spyder'))
    lock_code = "import sys\nsys.path.append(r'{spy_dir_str}')\nfrom spyder.utils.external import lockfile\nlock_file = r'{lock_file}'\nlock = lockfile.FilesystemLock(lock_file)\nlock_created = lock.lock()\nprint(lock_created)".format(spy_dir_str=spy_dir, lock_file=get_conf_path('spyder.lock'))
    with qtbot.waitSignal(shell.executed, timeout=2000):
        shell.execute(lock_code)
    qtbot.wait(1000)
    assert not shell.get_value('lock_created')
    n_editors = editorstack.get_stack_count()
    p = tmpdir.mkdir('foo').join('bar.py')
    p.write(lock_code)
    with qtbot.waitSignal(shell.executed):
        shell.execute('%edit {}'.format(to_text_string(p)))
    qtbot.wait(3000)
    assert editorstack.get_stack_count() == n_editors + 1
    assert editorstack.get_current_editor().toPlainText() == lock_code
    main_window.editor.close_file()

@pytest.mark.use_introspection
@pytest.mark.skipif(os.name == 'nt', reason='Fails on Windows')
def test_leaks(main_window, qtbot):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test leaks in mainwindow when closing a file or a console.\n\n    Many other ways of leaking exist but are not covered here.\n    '

    def ns_fun(main_window, qtbot):
        if False:
            while True:
                i = 10
        shell = main_window.ipyconsole.get_current_shellwidget()
        qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
        KernelHandler.wait_all_shutdown_threads()
        gc.collect()
        objects = gc.get_objects()
        n_code_editor_init = 0
        for o in objects:
            if type(o).__name__ == 'CodeEditor':
                n_code_editor_init += 1
        n_shell_init = 0
        for o in objects:
            if type(o).__name__ == 'ShellWidget':
                n_shell_init += 1
        main_window.editor.new()
        main_window.ipyconsole.create_new_client()
        code_editor = main_window.editor.get_focus_widget()
        code_editor.set_text('aaa')
        shell = main_window.ipyconsole.get_current_shellwidget()
        qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
        with qtbot.waitSignal(shell.executed):
            shell.execute('%debug print()')
        main_window.editor.close_all_files()
        main_window.ipyconsole.restart()
        KernelHandler.wait_all_shutdown_threads()
        return (n_shell_init, n_code_editor_init)
    (n_shell_init, n_code_editor_init) = ns_fun(main_window, qtbot)
    qtbot.wait(1000)
    gc.collect()
    objects = gc.get_objects()
    n_code_editor = 0
    for o in objects:
        if type(o).__name__ == 'CodeEditor':
            n_code_editor += 1
    n_shell = 0
    for o in objects:
        if type(o).__name__ == 'ShellWidget':
            n_shell += 1
    assert n_shell <= n_shell_init
    assert n_code_editor <= n_code_editor_init

def test_lock_action(main_window, qtbot):
    if False:
        return 10
    'Test the lock interface action.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    action = main_window.layouts.lock_interface_action
    plugins = main_window.widgetlist
    assert main_window.layouts._interface_locked
    for plugin in plugins:
        title_bar = plugin.dockwidget.titleBarWidget()
        assert not isinstance(title_bar, DockTitleBar)
        assert isinstance(title_bar, QWidget)
    action.trigger()
    for plugin in plugins:
        title_bar = plugin.dockwidget.titleBarWidget()
        assert isinstance(title_bar, DockTitleBar)
    assert not main_window.layouts._interface_locked
    action.trigger()
    assert main_window.layouts._interface_locked

@pytest.mark.order(1)
@pytest.mark.skipif(sys.platform.startswith('linux') and (not running_in_ci()), reason='Fails on Linux when run locally')
@pytest.mark.skipif(sys.platform == 'darwin' and running_in_ci(), reason='Fails on MacOS when run in CI')
def test_default_plugin_actions(main_window, qtbot):
    if False:
        print('Hello World!')
    'Test the effect of dock, undock, close and toggle view actions.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    file_explorer = main_window.explorer
    main_widget = file_explorer.get_widget()
    main_widget.undock_action.triggered.emit(True)
    qtbot.wait(500)
    main_widget.windowwidget.move(200, 200)
    assert not file_explorer.dockwidget.isVisible()
    assert main_widget.undock_action is not None
    assert isinstance(main_widget.windowwidget, SpyderWindowWidget)
    assert main_widget.windowwidget.centralWidget() == main_widget
    main_widget.dock_action.triggered.emit(True)
    qtbot.wait(500)
    assert file_explorer.dockwidget.isVisible()
    assert main_widget.windowwidget is None
    geometry = file_explorer.get_conf('window_geometry')
    assert geometry != ''
    file_explorer.set_conf('undocked_on_window_close', True)
    main_window.restore_undocked_plugins()
    assert main_widget.windowwidget is not None
    assert geometry == qbytearray_to_str(main_widget.windowwidget.saveGeometry())
    main_widget.windowwidget.close()
    main_widget.close_action.triggered.emit(True)
    qtbot.wait(500)
    assert not file_explorer.dockwidget.isVisible()
    assert not file_explorer.toggle_view_action.isChecked()
    file_explorer.toggle_view_action.setChecked(True)
    assert file_explorer.dockwidget.isVisible()

@flaky(max_runs=3)
@pytest.mark.parametrize('main_window', [{'spy_config': ('main', 'opengl', 'software')}], indirect=True)
def test_opengl_implementation(main_window, qtbot):
    if False:
        while True:
            i = 10
    '\n    Test that we are setting the selected OpenGL implementation\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    assert main_window._test_setting_opengl('software')
    CONF.set('main', 'opengl', 'automatic')

@flaky(max_runs=3)
@pytest.mark.skipif(np.__version__ < '1.14.0', reason='This only happens in Numpy 1.14+')
@pytest.mark.parametrize('main_window', [{'spy_config': ('variable_explorer', 'minmax', True)}], indirect=True)
def test_filter_numpy_warning(main_window, qtbot):
    if False:
        while True:
            i = 10
    "\n    Test that we filter a warning shown when an array contains nan\n    values and the Variable Explorer option 'Show arrays min/man'\n    is on.\n\n    For spyder-ide/spyder#7063.\n    "
    shell = main_window.ipyconsole.get_current_shellwidget()
    control = shell._control
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed):
        shell.execute('import numpy as np; A=np.full(16, np.nan)')
    qtbot.wait(1000)
    assert 'warning' not in control.toPlainText()
    assert 'Warning' not in control.toPlainText()
    CONF.set('variable_explorer', 'minmax', False)

@flaky(max_runs=3)
@pytest.mark.skipif(not sys.platform == 'darwin', reason='Fails on other than macOS')
@pytest.mark.known_leak
def test_get_help_combo(main_window, qtbot):
    if False:
        print('Hello World!')
    '\n    Test that Help can display docstrings for names typed in its combobox.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    help_plugin = main_window.help
    webview = help_plugin.get_widget().rich_text.webview._webview
    if WEBENGINE:
        webpage = webview.page()
    else:
        webpage = webview.page().mainFrame()
    with qtbot.waitSignal(shell.executed):
        shell.execute('import numpy as np')
    object_combo = help_plugin.get_widget().object_combo
    object_combo.setFocus()
    qtbot.keyClicks(object_combo, 'numpy', delay=100)
    qtbot.waitUntil(lambda : check_text(webpage, 'NumPy'), timeout=6000)
    qtbot.keyClicks(object_combo, '.arange', delay=100)
    qtbot.waitUntil(lambda : check_text(webpage, 'arange'), timeout=6000)
    object_combo.set_current_text('')
    qtbot.keyClicks(object_combo, 'np', delay=100)
    qtbot.waitUntil(lambda : check_text(webpage, 'NumPy'), timeout=6000)
    qtbot.keyClicks(object_combo, '.arange', delay=100)
    qtbot.waitUntil(lambda : check_text(webpage, 'arange'), timeout=6000)

@pytest.mark.known_leak
def test_get_help_ipython_console_dot_notation(main_window, qtbot, tmpdir):
    if False:
        i = 10
        return i + 15
    '\n    Test that Help works when called from the IPython console\n    with dot calls i.e np.sin\n\n    See spyder-ide/spyder#11821\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    control = shell._control
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    test_file = osp.join(LOCATION, 'script_unicode.py')
    main_window.editor.load(test_file)
    code_editor = main_window.editor.get_focus_widget()
    run_parameters = generate_run_parameters(main_window, test_file)
    CONF.set('run', 'last_used_parameters', run_parameters)
    qtbot.keyClick(code_editor, Qt.Key_F5)
    qtbot.wait(500)
    help_plugin = main_window.help
    webview = help_plugin.get_widget().rich_text.webview._webview
    webpage = webview.page() if WEBENGINE else webview.page().mainFrame()
    qtbot.keyClicks(control, u'np.linalg.norm')
    control.inspect_current_object()
    qtbot.waitUntil(lambda : check_text(webpage, 'Matrix or vector norm.'), timeout=6000)

@flaky(max_runs=3)
@pytest.mark.order(after='test_debug_unsaved_function')
@pytest.mark.skipif(sys.platform == 'darwin', reason='Too flaky on Mac')
def test_get_help_ipython_console_special_characters(main_window, qtbot, tmpdir):
    if False:
        print('Hello World!')
    '\n    Test that Help works when called from the IPython console\n    for unusual characters.\n\n    See spyder-ide/spyder#7699\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    control = shell._control
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    test_file = osp.join(LOCATION, 'script_unicode.py')
    main_window.editor.load(test_file)
    code_editor = main_window.editor.get_focus_widget()
    run_parameters = generate_run_parameters(main_window, test_file)
    CONF.set('run', 'last_used_parameters', run_parameters)
    qtbot.keyClick(code_editor, Qt.Key_F5)
    qtbot.wait(500)
    help_plugin = main_window.help
    webview = help_plugin.get_widget().rich_text.webview._webview
    webpage = webview.page() if WEBENGINE else webview.page().mainFrame()

    def check_control(control, value):
        if False:
            for i in range(10):
                print('nop')
        return value in control.toPlainText()
    qtbot.keyClicks(control, u'aa\t')
    qtbot.waitUntil(lambda : check_control(control, u'aaʹbb'), timeout=SHELL_TIMEOUT)
    control.inspect_current_object()
    qtbot.waitUntil(lambda : check_text(webpage, 'This function docstring.'), timeout=6000)

@flaky(max_runs=3)
@pytest.mark.skipif(os.name == 'nt' and running_in_ci(), reason='Times out on Windows')
def test_get_help_ipython_console(main_window, qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Test that Help works when called from the IPython console.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    control = shell._control
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    help_plugin = main_window.help
    webview = help_plugin.get_widget().rich_text.webview._webview
    webpage = webview.page() if WEBENGINE else webview.page().mainFrame()
    qtbot.keyClicks(control, 'get_ipython')
    control.inspect_current_object()
    qtbot.waitUntil(lambda : check_text(webpage, 'SpyderShell'), timeout=6000)

@flaky(max_runs=3)
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Does not work on Mac and Windows!')
@pytest.mark.use_introspection
@pytest.mark.order(after='test_debug_unsaved_function')
@pytest.mark.parametrize('object_info', [('range', 'range'), ('import numpy as np', 'An array object of arbitrary homogeneous items')])
def test_get_help_editor(main_window, qtbot, object_info):
    if False:
        while True:
            i = 10
    'Test that Help works when called from the Editor.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    help_plugin = main_window.help
    webview = help_plugin.get_widget().rich_text.webview._webview
    webpage = webview.page() if WEBENGINE else webview.page().mainFrame()
    main_window.editor.new(fname='test.py', text='')
    code_editor = main_window.editor.get_focus_widget()
    editorstack = main_window.editor.get_current_editorstack()
    qtbot.waitUntil(lambda : code_editor.completions_available, timeout=COMPLETION_TIMEOUT)
    (object_name, expected_text) = object_info
    code_editor.set_text(object_name)
    code_editor.move_cursor(len(object_name))
    with qtbot.waitSignal(code_editor.completions_response_signal, timeout=COMPLETION_TIMEOUT):
        code_editor.document_did_change()
    with qtbot.waitSignal(code_editor.sig_display_object_info, timeout=30000):
        editorstack.inspect_current_object()
    qtbot.waitUntil(lambda : check_text(webpage, expected_text), timeout=30000)
    assert check_text(webpage, expected_text)

def test_window_title(main_window, tmpdir, qtbot):
    if False:
        while True:
            i = 10
    'Test window title with non-ascii characters.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    projects = main_window.projects
    path = to_text_string(tmpdir.mkdir(u'測試'))
    projects.open_project(path=path)
    main_window._cli_options.window_title = u'اختبار'
    main_window.set_window_title()
    title = main_window.base_title
    assert u'Spyder' in title
    assert u'Python' in title
    assert u'اختبار' in title
    assert u'測試' in title
    projects.close_project()

@flaky(max_runs=3)
@pytest.mark.parametrize('debugcell', [True, False])
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Fails sometimes on Windows and Mac')
def test_move_to_first_breakpoint(main_window, qtbot, debugcell):
    if False:
        print('Hello World!')
    "Test that we move to the first breakpoint if there's one present."
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    control = shell._control
    debug_button = main_window.debug_button
    main_window.debugger.clear_all_breakpoints()
    test_file = osp.join(LOCATION, 'script.py')
    main_window.editor.load(test_file)
    code_editor = main_window.editor.get_focus_widget()
    code_editor.breakpoints_manager.toogle_breakpoint(line_number=10)
    qtbot.wait(500)
    cursor = code_editor.textCursor()
    cursor.setPosition(0)
    code_editor.setTextCursor(cursor)
    if debugcell:
        for _ in range(2):
            with qtbot.waitSignal(shell.executed):
                qtbot.mouseClick(main_window.run_cell_and_advance_button, Qt.LeftButton)
        debug_cell_action = main_window.run.get_action('run cell in debugger')
        with qtbot.waitSignal(shell.executed):
            debug_cell_action.trigger()
        assert shell.kernel_handler.kernel_comm.is_open()
        assert shell.is_waiting_pdb_input()
        with qtbot.waitSignal(shell.executed):
            shell.pdb_execute('!b')
        assert 'script.py:10' in shell._control.toPlainText()
        with qtbot.waitSignal(shell.executed):
            shell.pdb_execute('!c')
    else:
        with qtbot.waitSignal(shell.executed):
            qtbot.mouseClick(debug_button, Qt.LeftButton)
    shell.clear_console()
    qtbot.wait(500)
    with qtbot.waitSignal(shell.executed):
        shell.pdb_execute('!list')
    assert '1--> 10 arr = np.array(li)' in control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        shell.pdb_execute('!exit')
    code_editor.breakpoints_manager.toogle_breakpoint(line_number=2)
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(debug_button, Qt.LeftButton)
    assert '2---> 2 a = 10' in control.toPlainText()
    assert shell.is_waiting_pdb_input()
    main_window.debugger.clear_all_breakpoints()
    main_window.editor.close_file()

@flaky(max_runs=3)
@pytest.mark.order(after='test_debug_unsaved_function')
@pytest.mark.skipif(os.name == 'nt', reason='Fails on windows!')
def test_runconfig_workdir(main_window, qtbot, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Test runconfig workdir options.'
    CONF.set('run', 'parameters', {})
    test_file = osp.join(LOCATION, 'script.py')
    main_window.editor.load(test_file)
    code_editor = main_window.editor.get_focus_widget()
    ipyconsole = main_window.ipyconsole
    ipy_conf = IPythonConsolePyConfiguration(current=True, post_mortem=False, python_args_enabled=False, python_args='', clear_namespace=False, console_namespace=False)
    wdir_opts = WorkingDirOpts(source=WorkingDirSource.CurrentDirectory, path=None)
    exec_conf = RunExecutionParameters(working_dir=wdir_opts, executor_params=ipy_conf)
    exec_uuid = str(uuid.uuid4())
    ext_exec_conf = ExtendedRunExecutionParameters(uuid=exec_uuid, name='TestConf', params=exec_conf)
    ipy_dict = {ipyconsole.NAME: {('py', RunContext.File): {'params': {exec_uuid: ext_exec_conf}}}}
    CONF.set('run', 'parameters', ipy_dict)
    run_parameters = generate_run_parameters(main_window, test_file, exec_uuid)
    CONF.set('run', 'last_used_parameters', run_parameters)
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    qtbot.keyClick(code_editor, Qt.Key_F5)
    qtbot.wait(500)
    with qtbot.waitSignal(shell.executed):
        shell.execute('import os; current_dir = os.getcwd()')
    assert shell.get_value('current_dir') == get_home_dir()
    temp_dir = str(tmpdir.mkdir('test_dir'))
    wdir_opts = WorkingDirOpts(source=WorkingDirSource.CustomDirectory, path=temp_dir)
    exec_conf = RunExecutionParameters(working_dir=wdir_opts, executor_params=ipy_conf)
    ext_exec_conf['params'] = exec_conf
    ipy_dict = {ipyconsole.NAME: {('py', RunContext.File): {'params': {exec_uuid: ext_exec_conf}}}}
    CONF.set('run', 'parameters', ipy_dict)
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    qtbot.keyClick(code_editor, Qt.Key_F5)
    qtbot.wait(500)
    with qtbot.waitSignal(shell.executed):
        shell.execute('import os; current_dir = os.getcwd()')
    assert shell.get_value('current_dir') == temp_dir
    main_window.editor.close_file()
    CONF.set('run', 'parameters', {})

@pytest.mark.order(1)
@pytest.mark.no_new_console
@flaky(max_runs=3)
@pytest.mark.skipif(sys.platform == 'darwin', reason='Hangs sometimes on Mac')
def test_dedicated_consoles(main_window, qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Test running code in dedicated consoles.'
    test_file = osp.join(LOCATION, 'script.py')
    main_window.editor.load(test_file)
    code_editor = main_window.editor.get_focus_widget()
    ipyconsole = main_window.ipyconsole
    ipy_conf = IPythonConsolePyConfiguration(current=False, post_mortem=False, python_args_enabled=False, python_args='', clear_namespace=False, console_namespace=False)
    wdir_opts = WorkingDirOpts(source=WorkingDirSource.ConfigurationDirectory, path=None)
    exec_conf = RunExecutionParameters(working_dir=wdir_opts, executor_params=ipy_conf)
    exec_uuid = str(uuid.uuid4())
    ext_exec_conf = ExtendedRunExecutionParameters(uuid=exec_uuid, name='TestConf', params=exec_conf)
    ipy_dict = {ipyconsole.NAME: {('py', RunContext.File): {'params': {exec_uuid: ext_exec_conf}}}}
    CONF.set('run', 'parameters', ipy_dict)
    run_parameters = generate_run_parameters(main_window, test_file, exec_uuid)
    CONF.set('run', 'last_used_parameters', run_parameters)
    qtbot.keyClick(code_editor, Qt.Key_F5)
    qtbot.wait(500)
    shell = main_window.ipyconsole.get_current_shellwidget()
    control = shell._control
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    nsb = main_window.variableexplorer.current_widget()
    assert len(main_window.ipyconsole.get_clients()) == 2
    assert main_window.ipyconsole.get_widget().filenames == ['', test_file]
    assert main_window.ipyconsole.get_widget().tabwidget.tabText(1) == 'script.py/A'
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() == 4)
    assert nsb.editor.source_model.rowCount() == 4
    text = control.toPlainText()
    assert 'runfile' in text and (not ('Python' in text or 'IPython' in text))
    with qtbot.waitSignal(shell.executed):
        shell.execute('zz = -1')
    qtbot.keyClick(code_editor, Qt.Key_F5)
    qtbot.waitUntil(lambda : shell.is_defined('zz'))
    assert shell.is_defined('zz')
    assert 'runfile' in control.toPlainText()
    ipy_conf['clear_namespace'] = True
    CONF.set('run', 'parameters', ipy_dict)
    qtbot.wait(500)
    qtbot.keyClick(code_editor, Qt.Key_F5)
    qtbot.waitUntil(lambda : not shell.is_defined('zz'))
    assert not shell.is_defined('zz')
    assert 'runfile' in control.toPlainText()
    main_window.editor.close_file()
    CONF.set('run', 'configurations', {})
    CONF.set('run', 'last_used_parameters', {})

@flaky(max_runs=3)
@pytest.mark.order(after='test_dedicated_consoles')
def test_shell_execution(main_window, qtbot, tmpdir):
    if False:
        i = 10
        return i + 15
    'Test that bash/batch files can be executed.'
    ext = 'sh'
    script = 'bash_example.sh'
    interpreter = 'bash'
    opts = ''
    if sys.platform == 'darwin':
        interpreter = 'zsh'
    elif os.name == 'nt':
        interpreter = find_program('cmd.exe')
        script = 'batch_example.bat'
        ext = 'bat'
        opts = '/K'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    test_file = osp.join(LOCATION, script)
    main_window.editor.load(test_file)
    code_editor = main_window.editor.get_focus_widget()
    external_terminal = main_window.external_terminal
    temp_dir = str(tmpdir.mkdir('test_dir'))
    ext_conf = ExtTerminalShConfiguration(interpreter=interpreter, interpreter_opts_enabled=False, interpreter_opts=opts, script_opts_enabled=True, script_opts=temp_dir, close_after_exec=True)
    wdir_opts = WorkingDirOpts(source=WorkingDirSource.ConfigurationDirectory, path=None)
    exec_conf = RunExecutionParameters(working_dir=wdir_opts, executor_params=ext_conf)
    exec_uuid = str(uuid.uuid4())
    ext_exec_conf = ExtendedRunExecutionParameters(uuid=exec_uuid, name='TestConf', params=exec_conf)
    ipy_dict = {external_terminal.NAME: {(ext, RunContext.File): {'params': {exec_uuid: ext_exec_conf}}}}
    CONF.set('run', 'parameters', ipy_dict)
    run_parameters = generate_run_parameters(main_window, test_file, exec_uuid, external_terminal.NAME)
    CONF.set('run', 'last_used_parameters', run_parameters)
    qtbot.keyClick(code_editor, Qt.Key_F5)
    qtbot.wait(1000)
    qtbot.waitUntil(lambda : osp.exists(osp.join(temp_dir, 'output_file.txt')), timeout=EVAL_TIMEOUT)
    qtbot.wait(1000)
    with open(osp.join(temp_dir, 'output_file.txt'), 'r') as f:
        lines = f.read()
    assert lines.lower().strip().replace('"', '') == f'this is a temporary file created by {sys.platform}'

@flaky(max_runs=3)
@pytest.mark.skipif(sys.platform.startswith('linux'), reason='Fails frequently on Linux')
@pytest.mark.order(after='test_debug_unsaved_function')
def test_connection_to_external_kernel(main_window, qtbot):
    if False:
        i = 10
        return i + 15
    'Test that only Spyder kernels are connected to the Variable Explorer.'
    (km, kc) = start_new_kernel()
    main_window.ipyconsole.create_client_for_kernel(kc.connection_file)
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed):
        shell.execute('a = 10')
    main_window.variableexplorer.change_visibility(True)
    nsb = main_window.variableexplorer.current_widget()
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() == 0)
    assert nsb.editor.source_model.rowCount() == 0
    python_shell = shell
    (spykm, spykc) = start_new_kernel(spykernel=True)
    main_window.ipyconsole.create_client_for_kernel(spykc.connection_file)
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed):
        shell.execute('a = 10')
    main_window.variableexplorer.change_visibility(True)
    nsb = main_window.variableexplorer.current_widget()
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() == 1)
    assert nsb.editor.source_model.rowCount() == 1
    main_window.editor.new()
    code_editor = main_window.editor.get_focus_widget()
    code_editor.set_text('print(2 + 1)')
    file_path = main_window.editor.get_current_filename()
    run_parameters = generate_run_parameters(main_window, file_path)
    CONF.set('run', 'last_used_parameters', run_parameters)
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(main_window.run_button, Qt.LeftButton)
    assert 'runfile' in shell._control.toPlainText()
    assert '3' in shell._control.toPlainText()
    if os.name != 'nt':
        with qtbot.waitSignal(shell.executed):
            shell.execute('%matplotlib qt5')
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug print()')
    with qtbot.waitSignal(shell.executed):
        shell.execute('1 + 1')
    with qtbot.waitSignal(shell.executed):
        shell.execute('q')
    shell.execute('quit()')
    python_shell.execute('quit()')
    qtbot.waitUntil(lambda : not km.is_alive())
    assert not km.is_alive()
    qtbot.waitUntil(lambda : not spykm.is_alive())
    assert not spykm.is_alive()
    spykc.stop_channels()
    kc.stop_channels()

@pytest.mark.order(1)
@flaky(max_runs=3)
@pytest.mark.skipif(os.name == 'nt', reason='It times out sometimes on Windows')
def test_change_types_in_varexp(main_window, qtbot):
    if False:
        i = 10
        return i + 15
    "Test that variable types can't be changed in the Variable Explorer."
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed):
        shell.execute('a = 10')
    main_window.variableexplorer.change_visibility(True)
    nsb = main_window.variableexplorer.current_widget()
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() > 0, timeout=EVAL_TIMEOUT)
    nsb.editor.setFocus()
    nsb.editor.edit_item()
    qtbot.keyClicks(QApplication.focusWidget(), "'s'")
    qtbot.keyClick(QApplication.focusWidget(), Qt.Key_Enter)
    qtbot.wait(1000)
    assert shell.get_value('a') == 10

@flaky(max_runs=3)
@pytest.mark.parametrize('test_directory', [u'non_ascii_ñ_í_ç', u'test_dir'])
@pytest.mark.skipif(sys.platform == 'darwin', reason='It fails on macOS')
def test_change_cwd_ipython_console(main_window, qtbot, tmpdir, test_directory):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test synchronization with working directory and File Explorer when\n    changing cwd in the IPython console.\n    '
    wdir = main_window.workingdirectory
    treewidget = main_window.explorer.get_widget().treewidget
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    temp_dir = str(tmpdir.mkdir(test_directory))
    with qtbot.waitSignal(shell.executed):
        shell.execute(u'%cd {}'.format(temp_dir))
    qtbot.waitUntil(lambda : osp.normpath(wdir.get_container().history[-1]) == osp.normpath(temp_dir), timeout=SHELL_TIMEOUT)
    assert osp.normpath(wdir.get_container().history[-1]) == osp.normpath(temp_dir)
    qtbot.waitUntil(lambda : osp.normpath(treewidget.get_current_folder()) == osp.normpath(temp_dir), timeout=SHELL_TIMEOUT)
    assert osp.normpath(treewidget.get_current_folder()) == osp.normpath(temp_dir)

@flaky(max_runs=3)
@pytest.mark.parametrize('test_directory', [u'non_ascii_ñ_í_ç', u'test_dir'])
@pytest.mark.skipif(sys.platform == 'darwin', reason='It fails on macOS')
def test_change_cwd_explorer(main_window, qtbot, tmpdir, test_directory):
    if False:
        while True:
            i = 10
    '\n    Test synchronization with working directory and IPython console when\n    changing directories in the File Explorer.\n    '
    wdir = main_window.workingdirectory
    explorer = main_window.explorer
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    temp_dir = to_text_string(tmpdir.mkdir(test_directory))
    explorer.chdir(temp_dir)
    qtbot.waitUntil(lambda : osp.normpath(temp_dir) == osp.normpath(shell.get_cwd()))
    assert osp.normpath(wdir.get_container().history[-1]) == osp.normpath(temp_dir)
    assert osp.normpath(temp_dir) == osp.normpath(shell.get_cwd())

@flaky(max_runs=3)
@pytest.mark.skipif(os.name == 'nt' or sys.platform == 'darwin' or parse(ipy_release.version) == parse('7.11.0'), reason='Hard to test on Windows and macOS and fails for IPython 7.11.0')
@pytest.mark.order(after='test_debug_unsaved_function')
def test_run_cython_code(main_window, qtbot):
    if False:
        print('Hello World!')
    'Test all the different ways we have to run Cython code'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    code_editor = main_window.editor.get_focus_widget()
    file_path = osp.join(LOCATION, 'pyx_script.pyx')
    main_window.editor.load(file_path)
    run_parameters = generate_run_parameters(main_window, file_path)
    CONF.set('run', 'last_used_parameters', run_parameters)
    qtbot.keyClick(code_editor, Qt.Key_F5)
    nsb = main_window.variableexplorer.current_widget()
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() == 1, timeout=COMPILE_AND_EVAL_TIMEOUT)
    shell = main_window.ipyconsole.get_current_shellwidget()
    assert shell.get_value('a') == 3628800
    reset_run_code(qtbot, shell, code_editor, nsb)
    main_window.editor.close_file()
    file_path = osp.join(LOCATION, 'pyx_lib_import.py')
    main_window.editor.load(file_path)
    run_parameters = generate_run_parameters(main_window, file_path)
    CONF.set('run', 'last_used_parameters', run_parameters)
    qtbot.keyClick(code_editor, Qt.Key_F5)
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() == 1, timeout=COMPILE_AND_EVAL_TIMEOUT)
    assert shell.get_value('b') == 3628800
    main_window.editor.close_file()

@flaky(max_runs=5)
def test_project_path(main_window, tmpdir, qtbot):
    if False:
        return 10
    'Test project path added to spyder_pythonpath and IPython Console.'
    projects = main_window.projects
    path = str(tmpdir.mkdir('project_path'))
    assert path not in projects.get_conf('spyder_pythonpath', section='pythonpath_manager')
    projects.open_project(path=path)
    assert path in projects.get_conf('spyder_pythonpath', section='pythonpath_manager')
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed):
        shell.execute("import sys; import os; sys_path = sys.path; os_path = os.environ.get('PYTHONPATH', [])")
    assert path in shell.get_value('sys_path')
    assert path in shell.get_value('os_path')
    projects.close_project()
    assert path not in projects.get_conf('spyder_pythonpath', section='pythonpath_manager')
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed):
        shell.execute("import sys; import os; sys_path = sys.path; os_path = os.environ.get('PYTHONPATH', [])")
    assert path not in shell.get_value('sys_path')
    assert path not in shell.get_value('os_path')

@flaky(max_runs=3)
@pytest.mark.skipif(os.name == 'nt', reason='It fails on Windows.')
def test_open_notebooks_from_project_explorer(main_window, qtbot, tmpdir):
    if False:
        while True:
            i = 10
    'Test that notebooks are open from the Project explorer.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    projects = main_window.projects
    projects.toggle_view_action.setChecked(True)
    editorstack = main_window.editor.get_current_editorstack()
    project_dir = to_text_string(tmpdir.mkdir('test'))
    nb = osp.join(LOCATION, 'notebook.ipynb')
    shutil.copy(nb, osp.join(project_dir, 'notebook.ipynb'))
    with qtbot.waitSignal(projects.sig_project_loaded):
        projects.create_project(project_dir)
    idx = projects.get_widget().treewidget.get_index(osp.join(project_dir, 'notebook.ipynb'))
    projects.get_widget().treewidget.setCurrentIndex(idx)
    qtbot.keyClick(projects.get_widget().treewidget, Qt.Key_Enter)
    assert 'notebook.ipynb' in editorstack.get_current_filename()
    projects.get_widget().treewidget.convert_notebook(osp.join(project_dir, 'notebook.ipynb'))
    assert 'untitled' in editorstack.get_current_filename()
    file_text = editorstack.get_current_editor().toPlainText()
    if nbconvert.__version__ >= '5.4.0':
        expected_text = '#!/usr/bin/env python\n# coding: utf-8\n\n# In[1]:\n\n\n1 + 1\n\n\n# In[ ]:\n\n\n\n\n'
    else:
        expected_text = '\n# coding: utf-8\n\n# In[1]:\n\n\n1 + 1\n\n\n'
    assert file_text == expected_text
    projects.close_project()

@flaky(max_runs=3)
def test_runfile_from_project_explorer(main_window, qtbot, tmpdir):
    if False:
        print('Hello World!')
    'Test that file are run from the Project explorer.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    projects = main_window.projects
    projects.toggle_view_action.setChecked(True)
    editorstack = main_window.editor.get_current_editorstack()
    project_dir = to_text_string(tmpdir.mkdir('test'))
    test_file = osp.join(LOCATION, 'script.py')
    shutil.copy(test_file, osp.join(project_dir, 'script.py'))
    with qtbot.waitSignal(projects.sig_project_loaded):
        projects.create_project(project_dir)
    idx = projects.get_widget().treewidget.get_index(osp.join(project_dir, 'script.py'))
    projects.get_widget().treewidget.setCurrentIndex(idx)
    qtbot.keyClick(projects.get_widget().treewidget, Qt.Key_Enter)
    assert 'script.py' in editorstack.get_current_filename()
    projects.get_widget().treewidget.run([osp.join(project_dir, 'script.py')])
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    nsb = main_window.variableexplorer.current_widget()
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() == 4, timeout=EVAL_TIMEOUT)
    assert shell.get_value('a') == 10
    assert shell.get_value('s') == 'Z:\\escape\\test\\string\n'
    assert shell.get_value('li') == [1, 2, 3]
    assert_array_equal(shell.get_value('arr'), np.array([1, 2, 3]))
    projects.close_project()

@flaky(max_runs=3)
@pytest.mark.skipif(os.name == 'nt', reason='It times out sometimes on Windows')
def test_set_new_breakpoints(main_window, qtbot):
    if False:
        return 10
    'Test that new breakpoints are set in the IPython console.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    control = shell._control
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    main_window.debugger.clear_all_breakpoints()
    test_file = osp.join(LOCATION, 'script.py')
    main_window.editor.load(test_file)
    debug_button = main_window.debug_button
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(debug_button, Qt.LeftButton)
    code_editor = main_window.editor.get_focus_widget()
    code_editor.breakpoints_manager.toogle_breakpoint(line_number=6)
    with qtbot.waitSignal(shell.executed):
        shell.pdb_execute('!b')
    assert '1   breakpoint   keep yes   at {}:6'.format(test_file) in control.toPlainText()
    main_window.debugger.clear_all_breakpoints()
    main_window.editor.close_file()

@flaky(max_runs=3)
@pytest.mark.order(after='test_debug_unsaved_function')
def test_run_code(main_window, qtbot, tmpdir):
    if False:
        i = 10
        return i + 15
    'Test all the different ways we have to run code'
    p = tmpdir.mkdir(u"runtest's folder èáïü Øαôå 字分误").join(u"runtest's file èáïü Øαôå 字分误.py")
    filepath = to_text_string(p)
    shutil.copyfile(osp.join(LOCATION, 'script.py'), filepath)
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    main_window.editor.load(filepath)
    editor = main_window.editor
    code_editor = editor.get_focus_widget()
    code_editor.setFocus()
    qtbot.keyClick(code_editor, Qt.Key_Home, modifier=Qt.ControlModifier)
    nsb = main_window.variableexplorer.current_widget()
    run_parameters = generate_run_parameters(main_window, filepath)
    CONF.set('run', 'last_used_parameters', run_parameters)
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(code_editor, Qt.Key_F5)
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() == 4, timeout=EVAL_TIMEOUT)
    assert shell.get_value('a') == 10
    assert shell.get_value('s') == 'Z:\\escape\\test\\string\n'
    assert shell.get_value('li') == [1, 2, 3]
    assert_array_equal(shell.get_value('arr'), np.array([1, 2, 3]))
    reset_run_code(qtbot, shell, code_editor, nsb)
    for _ in range(code_editor.blockCount()):
        with qtbot.waitSignal(shell.executed):
            qtbot.mouseClick(main_window.run_selection_button, Qt.LeftButton)
            qtbot.wait(200)
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() == 4, timeout=EVAL_TIMEOUT)
    assert shell.get_value('a') == 10
    assert shell.get_value('s') == 'Z:\\escape\\test\\string\n'
    assert shell.get_value('li') == [1, 2, 3]
    assert_array_equal(shell.get_value('arr'), np.array([1, 2, 3]))
    reset_run_code(qtbot, shell, code_editor, nsb)
    editor.go_to_line(10)
    qtbot.keyClick(code_editor, Qt.Key_Right)
    run_to_line_action = main_window.run.get_action('run selection up to line')
    with qtbot.waitSignal(shell.executed):
        run_to_line_action.trigger()
    qtbot.wait(500)
    assert shell.get_value('a') == 10
    assert shell.get_value('li') == [1, 2, 3]
    assert 'arr' not in nsb.editor.source_model._data.keys()
    assert 's' not in nsb.editor.source_model._data.keys()
    reset_run_code(qtbot, shell, code_editor, nsb)
    shell.execute('a = 100')
    editor.go_to_line(6)
    qtbot.keyClick(code_editor, Qt.Key_Right)
    run_from_line_action = main_window.run.get_action('run selection from line')
    with qtbot.waitSignal(shell.executed):
        run_from_line_action.trigger()
    qtbot.wait(500)
    assert shell.get_value('s') == 'Z:\\escape\\test\\string\n'
    assert shell.get_value('li') == [1, 2, 3]
    assert_array_equal(shell.get_value('arr'), np.array([1, 2, 3]))
    assert shell.get_value('a') == 100
    reset_run_code(qtbot, shell, code_editor, nsb)
    qtbot.keyClicks(code_editor, 'a = 10')
    qtbot.keyClick(code_editor, Qt.Key_Return)
    qtbot.keyClick(code_editor, Qt.Key_Up)
    for _ in range(5):
        with qtbot.waitSignal(shell.executed):
            qtbot.mouseClick(main_window.run_cell_and_advance_button, Qt.LeftButton)
            qtbot.wait(500)
    assert 'runcell' in shell._control.toPlainText()
    assert 'Error:' not in shell._control.toPlainText()
    control_text = shell._control.toPlainText()
    shell.setFocus()
    qtbot.keyClick(shell._control, Qt.Key_Up)
    qtbot.wait(500)
    qtbot.keyClick(shell._control, Qt.Key_Enter, modifier=Qt.ShiftModifier)
    qtbot.wait(500)
    code_editor.setFocus()
    assert control_text != shell._control.toPlainText()
    control_text = shell._control.toPlainText()[len(control_text):]
    assert 'runcell' in control_text
    assert 'Error' not in control_text
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() == 4, timeout=EVAL_TIMEOUT)
    assert ']: 10\n' in shell._control.toPlainText()
    assert shell.get_value('a') == 10
    assert shell.get_value('s') == 'Z:\\escape\\test\\string\n'
    assert shell.get_value('li') == [1, 2, 3]
    assert_array_equal(shell.get_value('arr'), np.array([1, 2, 3]))
    reset_run_code(qtbot, shell, code_editor, nsb)
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(main_window.run_cell_button, Qt.LeftButton)
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() == 1, timeout=EVAL_TIMEOUT)
    assert shell.get_value('a') == 10
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(main_window.run_cell_button, Qt.LeftButton)
    assert nsb.editor.source_model.rowCount() == 1
    reset_run_code(qtbot, shell, code_editor, nsb)
    debug_cell_action = main_window.run.get_action('run cell in debugger')
    with qtbot.waitSignal(shell.executed):
        debug_cell_action.trigger()
    qtbot.keyClicks(shell._control, '!c')
    qtbot.keyClick(shell._control, Qt.Key_Enter)
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() == 1, timeout=EVAL_TIMEOUT)
    reset_run_code(qtbot, shell, code_editor, nsb)
    for _ in range(3):
        with qtbot.waitSignal(shell.executed):
            qtbot.mouseClick(main_window.run_cell_and_advance_button, Qt.LeftButton)
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() == 2, timeout=EVAL_TIMEOUT)
    with qtbot.waitSignal(shell.executed):
        shell.execute('%reset -f')
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() == 0, timeout=EVAL_TIMEOUT)
    re_run_action = main_window.run.get_action('re-run cell')
    with qtbot.waitSignal(shell.executed):
        re_run_action.trigger()
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() == 1, timeout=EVAL_TIMEOUT)
    assert shell.get_value('li') == [1, 2, 3]
    shell.clear()
    with qtbot.waitSignal(shell.executed):
        shell.execute('%reset -f')
    with qtbot.waitSignal(shell.executed):
        shell.execute('%runcell -i 0')
    assert shell.get_value('a') == 10
    assert 'error' not in shell._control.toPlainText().lower()
    main_window.editor.close_file()

@flaky(max_runs=3)
@pytest.mark.skipif(sys.platform == 'darwin', reason='It fails on macOS')
@pytest.mark.parametrize('main_window', [{'spy_config': ('run', 'run_cell_copy', True)}], indirect=True)
@pytest.mark.order(after='test_debug_unsaved_function')
def test_run_cell_copy(main_window, qtbot, tmpdir):
    if False:
        i = 10
        return i + 15
    'Test all the different ways we have to run code'
    p = tmpdir.mkdir(u"runtest's folder èáïü Øαôå 字分误").join(u"runtest's file èáïü Øαôå 字分误.py")
    filepath = to_text_string(p)
    shutil.copyfile(osp.join(LOCATION, 'script.py'), filepath)
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    main_window.editor.load(filepath)
    code_editor = main_window.editor.get_focus_widget()
    code_editor.setFocus()
    qtbot.keyClick(code_editor, Qt.Key_Home, modifier=Qt.ControlModifier)
    nsb = main_window.variableexplorer.current_widget()
    for _ in range(4):
        with qtbot.waitSignal(shell.executed):
            qtbot.mouseClick(main_window.run_cell_and_advance_button, Qt.LeftButton)
    assert 'runcell' not in shell._control.toPlainText()
    assert 'a = 10' in shell._control.toPlainText()
    assert 'Error:' not in shell._control.toPlainText()
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() == 4, timeout=EVAL_TIMEOUT)
    assert ']: 10\n' in shell._control.toPlainText()
    assert shell.get_value('a') == 10
    assert shell.get_value('s') == 'Z:\\escape\\test\\string\n'
    assert shell.get_value('li') == [1, 2, 3]
    assert_array_equal(shell.get_value('arr'), np.array([1, 2, 3]))
    main_window.editor.close_file()
    CONF.set('run', 'run_cell_copy', False)

@flaky(max_runs=3)
@pytest.mark.skipif(running_in_ci(), reason='Fails on CIs')
def test_open_files_in_new_editor_window(main_window, qtbot):
    if False:
        while True:
            i = 10
    '\n    This tests that opening files in a new editor window\n    is working as expected.\n\n    Test for spyder-ide/spyder#4085.\n    '
    QTimer.singleShot(2000, lambda : open_file_in_editor(main_window, 'script.py', directory=LOCATION))
    main_window.editor.create_new_window()
    main_window.editor.load()
    editorstack = main_window.editor.get_current_editorstack()
    assert editorstack.get_stack_count() == 2

@flaky(max_runs=3)
def test_close_when_file_is_changed(main_window, qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Test closing spyder when there is a file with modifications open.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    test_file = osp.join(LOCATION, 'script.py')
    main_window.editor.load(test_file)
    editorstack = main_window.editor.get_current_editorstack()
    editor = editorstack.get_current_editor()
    editor.document().setModified(True)
    qtbot.wait(3000)

@flaky(max_runs=3)
def test_maximize_minimize_plugins(main_window, qtbot):
    if False:
        return 10
    'Test that the maximize button is working as expected.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)

    def get_random_plugin():
        if False:
            while True:
                i = 10
        'Get a random dockable plugin and give it focus'
        plugins = main_window.get_dockable_plugins()
        for (plugin_name, plugin) in plugins:
            if plugin_name in [Plugins.Editor, Plugins.IPythonConsole]:
                plugins.remove((plugin_name, plugin))
        plugin = random.choice(plugins)[1]
        if not plugin.get_widget().toggle_view_action.isChecked():
            plugin.toggle_view(True)
            plugin._hide_after_test = True
        plugin.get_widget().get_focus_widget().setFocus()
        return plugin
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    max_action = main_window.layouts.maximize_action
    toolbar = main_window.get_plugin(Plugins.Toolbar)
    main_toolbar = toolbar.get_application_toolbar(ApplicationToolbars.Main)
    max_button = main_toolbar.widgetForAction(max_action)
    plugin_1 = get_random_plugin()
    qtbot.mouseClick(max_button, Qt.LeftButton)
    test_file = osp.join(LOCATION, 'script.py')
    main_window.editor.load(test_file)
    assert not plugin_1.get_widget().get_maximized_state()
    assert QApplication.focusWidget() is main_window.editor.get_focus_widget()
    assert not max_action.isChecked()
    if hasattr(plugin_1, '_hide_after_test'):
        plugin_1.toggle_view(False)
    qtbot.mouseClick(max_button, Qt.LeftButton)
    assert main_window.editor._ismaximized
    qtbot.mouseClick(max_button, Qt.LeftButton)
    assert not main_window.editor._ismaximized
    qtbot.mouseClick(max_button, Qt.LeftButton)
    assert main_window.editor._ismaximized
    ipyconsole = main_window.get_plugin(Plugins.IPythonConsole)
    ipyconsole.create_window()
    assert main_window.editor._ismaximized
    ipyconsole.close_window()
    assert not main_window.editor._ismaximized
    plugin_2 = get_random_plugin()
    qtbot.mouseClick(max_button, Qt.LeftButton)
    debug_button = main_window.debug_button
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(debug_button, Qt.LeftButton)
    qtbot.waitUntil(lambda : 'IPdb' in shell._control.toPlainText())
    assert not plugin_2.get_widget().get_maximized_state()
    assert not max_action.isChecked()
    if hasattr(plugin_2, '_hide_after_test'):
        plugin_2.toggle_view(False)
    debugger = main_window.debugger
    debug_next_action = debugger.get_action(DebuggerWidgetActions.Next)
    debug_next_button = debugger.get_widget()._main_toolbar.widgetForAction(debug_next_action)
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(debug_next_button, Qt.LeftButton)
    assert not main_window.editor._ismaximized
    assert not max_action.isChecked()
    debugger.get_widget().get_focus_widget().setFocus()
    qtbot.mouseClick(max_button, Qt.LeftButton)
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(debug_next_button, Qt.LeftButton)
    assert not debugger.get_widget().get_maximized_state()
    assert not max_action.isChecked()
    with qtbot.waitSignal(shell.executed):
        shell.stop_debugging()
    plugin_3 = get_random_plugin()
    qtbot.mouseClick(max_button, Qt.LeftButton)
    run_parameters = generate_run_parameters(main_window, test_file)
    CONF.set('run', 'last_used_parameters', run_parameters)
    qtbot.mouseClick(main_window.run_button, Qt.LeftButton)
    assert not plugin_3.get_widget().get_maximized_state()
    assert not max_action.isChecked()
    if hasattr(plugin_3, '_hide_after_test'):
        plugin_3.toggle_view(False)
    plugin_4 = get_random_plugin()
    qtbot.mouseClick(max_button, Qt.LeftButton)
    qtbot.mouseClick(main_window.run_cell_button, Qt.LeftButton)
    assert not plugin_4.get_widget().get_maximized_state()
    assert not max_action.isChecked()
    if hasattr(plugin_4, '_hide_after_test'):
        plugin_4.toggle_view(False)
    plugin_5 = get_random_plugin()
    qtbot.mouseClick(max_button, Qt.LeftButton)
    qtbot.mouseClick(main_window.run_selection_button, Qt.LeftButton)
    assert not plugin_5.get_widget().get_maximized_state()
    assert not max_action.isChecked()
    if hasattr(plugin_5, '_hide_after_test'):
        plugin_5.toggle_view(False)

@flaky(max_runs=3)
@pytest.mark.skipif(os.name == 'nt' or (running_in_ci() and (PYQT5 and PYQT_VERSION >= '5.9')), reason='It times out on Windows and segfaults in our CIs with PyQt >= 5.9')
def test_issue_4066(main_window, qtbot):
    if False:
        print('Hello World!')
    '\n    Test for a segfault when these steps are followed:\n\n    1. Open an object present in the Variable Explorer (e.g. a list).\n    2. Delete that object in its corresponding console while its\n       editor is still open.\n    3. Closing that editor by pressing its *Ok* button.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed):
        shell.execute('myobj = [1, 2, 3]')
    nsb = main_window.variableexplorer.current_widget()
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() > 0, timeout=EVAL_TIMEOUT)
    nsb.editor.setFocus()
    nsb.editor.edit_item()
    obj_editor_id = list(nsb.editor.delegate._editors.keys())[0]
    obj_editor = nsb.editor.delegate._editors[obj_editor_id]['editor']
    main_window.ipyconsole.get_widget().get_focus_widget().setFocus()
    with qtbot.waitSignal(shell.executed):
        shell.execute('del myobj')
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() == 0, timeout=EVAL_TIMEOUT)
    ok_widget = obj_editor.btn_close
    qtbot.mouseClick(ok_widget, Qt.LeftButton)
    qtbot.wait(3000)

@flaky(max_runs=3)
@pytest.mark.skipif(os.name == 'nt', reason='It times out sometimes on Windows')
def test_varexp_edit_inline(main_window, qtbot):
    if False:
        while True:
            i = 10
    "\n    Test for errors when editing inline values in the Variable Explorer\n    and then moving to another plugin.\n\n    Note: Errors for this test don't appear related to it but instead they\n    are shown down the road. That's because they are generated by an\n    async C++ RuntimeError.\n    "
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed):
        shell.execute('a = 10')
    main_window.variableexplorer.change_visibility(True)
    nsb = main_window.variableexplorer.current_widget()
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() > 0, timeout=EVAL_TIMEOUT)
    nsb.editor.setFocus()
    nsb.editor.edit_item()
    main_window.ipyconsole.get_widget().get_focus_widget().setFocus()
    qtbot.wait(3000)

@flaky(max_runs=3)
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='It times out sometimes on Windows and macOS')
def test_c_and_n_pdb_commands(main_window, qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Test that c and n Pdb commands update the Variable Explorer.'
    nsb = main_window.variableexplorer.current_widget()
    shell = main_window.ipyconsole.get_current_shellwidget()
    control = shell._control
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    main_window.debugger.clear_all_breakpoints()
    test_file = osp.join(LOCATION, 'script.py')
    main_window.editor.load(test_file)
    debug_button = main_window.debug_button
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(debug_button, Qt.LeftButton)
    code_editor = main_window.editor.get_focus_widget()
    code_editor.breakpoints_manager.toogle_breakpoint(line_number=6)
    qtbot.wait(500)
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClicks(control, '!c')
        qtbot.keyClick(control, Qt.Key_Enter)
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() == 1)
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClicks(control, '!n')
        qtbot.keyClick(control, Qt.Key_Enter)
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() == 2)
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClicks(control, '!n')
        qtbot.keyClick(control, Qt.Key_Enter)
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClicks(control, '!n')
        qtbot.keyClick(control, Qt.Key_Enter)
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() == 3)
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClicks(control, '!n')
        qtbot.keyClick(control, Qt.Key_Enter)
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClicks(control, '!n')
        qtbot.keyClick(control, Qt.Key_Enter)
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClicks(control, '!n')
        qtbot.keyClick(control, Qt.Key_Enter)
    shell.clear_console()
    assert 'In [2]:' in control.toPlainText()
    main_window.debugger.clear_all_breakpoints()
    main_window.editor.close_file()

@flaky(max_runs=3)
@pytest.mark.skipif(os.name == 'nt', reason='It times out sometimes on Windows')
def test_stop_dbg(main_window, qtbot):
    if False:
        while True:
            i = 10
    'Test that we correctly stop a debugging session.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    main_window.debugger.clear_all_breakpoints()
    test_file = osp.join(LOCATION, 'script.py')
    main_window.editor.load(test_file)
    debug_button = main_window.debug_button
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(debug_button, Qt.LeftButton)
    with qtbot.waitSignal(shell.executed):
        shell.pdb_execute('!n')
    with qtbot.waitSignal(shell.executed):
        shell.stop_debugging()
    assert shell._control.toPlainText().count('IPdb') == 2
    main_window.debugger.clear_all_breakpoints()
    main_window.editor.close_file()

@flaky(max_runs=3)
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='It only works on Linux')
def test_change_cwd_dbg(main_window, qtbot):
    if False:
        print('Hello World!')
    '\n    Test that using the Working directory toolbar is working while debugging.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    test_file = osp.join(LOCATION, 'script.py')
    main_window.editor.load(test_file)
    control = main_window.ipyconsole.get_widget().get_focus_widget()
    control.setFocus()
    debug_button = main_window.debug_button
    qtbot.mouseClick(debug_button, Qt.LeftButton)
    qtbot.waitUntil(lambda : 'IPdb' in control.toPlainText())
    main_window.workingdirectory.chdir(tempfile.gettempdir())
    qtbot.wait(1000)
    print(repr(control.toPlainText()))
    shell.clear_console()
    qtbot.waitUntil(lambda : 'IPdb [2]:' in control.toPlainText())
    qtbot.keyClicks(control, 'import os; os.getcwd()')
    qtbot.keyClick(control, Qt.Key_Enter)
    qtbot.waitUntil(lambda : tempfile.gettempdir() in control.toPlainText())
    assert tempfile.gettempdir() in control.toPlainText()

@flaky(max_runs=3)
@pytest.mark.skipif(os.name == 'nt', reason='Times out sometimes')
def test_varexp_magic_dbg(main_window, qtbot):
    if False:
        print('Hello World!')
    'Test that %varexp is working while debugging.'
    nsb = main_window.variableexplorer.current_widget()
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    test_file = osp.join(LOCATION, 'script.py')
    main_window.editor.load(test_file)
    control = main_window.ipyconsole.get_widget().get_focus_widget()
    control.setFocus()
    debug_button = main_window.debug_button
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(debug_button, Qt.LeftButton)
    for _ in range(3):
        with qtbot.waitSignal(shell.executed):
            qtbot.keyClicks(control, '!n')
            qtbot.keyClick(control, Qt.Key_Enter)
    nsb.editor.plot('li', 'plot')
    qtbot.wait(1000)
    assert shell._control.toHtml().count('img src') == 1

@flaky(max_runs=3)
@pytest.mark.parametrize('main_window', [{'spy_config': ('ipython_console', 'pylab/inline/figure_format', 'svg')}, {'spy_config': ('ipython_console', 'pylab/inline/figure_format', 'png')}], indirect=True)
def test_plots_plugin(main_window, qtbot, tmpdir, mocker):
    if False:
        print('Hello World!')
    '\n    Test that plots generated in the IPython console are properly displayed\n    in the plots plugin.\n    '
    assert CONF.get('plots', 'mute_inline_plotting') is False
    shell = main_window.ipyconsole.get_current_shellwidget()
    figbrowser = main_window.plots.current_widget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed):
        shell.execute("import matplotlib.pyplot as plt\nfig = plt.plot([1, 2, 3, 4], '.')\n")
    if CONF.get('ipython_console', 'pylab/inline/figure_format') == 'png':
        assert figbrowser.figviewer.figcanvas.fmt == 'image/png'
    else:
        assert figbrowser.figviewer.figcanvas.fmt == 'image/svg+xml'
    html = shell._control.toHtml()
    img_name = re.search('<img src="(.+?)" /></p>', html).group(1)
    ipython_figname = osp.join(to_text_string(tmpdir), 'ipython_img.png')
    ipython_qimg = shell._get_image(img_name)
    ipython_qimg.save(ipython_figname)
    plots_figname = osp.join(to_text_string(tmpdir), 'plots_img.png')
    mocker.patch('spyder.plugins.plots.widgets.figurebrowser.getsavefilename', return_value=(plots_figname, '.png'))
    figbrowser.save_figure()
    assert compare_images(ipython_figname, plots_figname, 0.1) is None

def test_plots_scroll(main_window, qtbot):
    if False:
        while True:
            i = 10
    'Test plots plugin scrolling'
    CONF.set('plots', 'mute_inline_plotting', True)
    shell = main_window.ipyconsole.get_current_shellwidget()
    figbrowser = main_window.plots.current_widget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed, timeout=SHELL_TIMEOUT):
        shell.execute("import matplotlib.pyplot as plt\nfig = plt.plot([1, 2, 3, 4], '.')\n")
    sb = figbrowser.thumbnails_sb
    assert len(sb._thumbnails) == 1
    assert sb._thumbnails[-1] == sb.current_thumbnail
    with qtbot.waitSignal(shell.executed, timeout=SHELL_TIMEOUT):
        shell.execute("for i in range(4):\n    plt.figure()\n    plt.plot([1, 2, 3, 4], '.')")
    assert len(sb._thumbnails) == 5
    assert sb._thumbnails[-1] == sb.current_thumbnail
    with qtbot.waitSignal(shell.executed, timeout=SHELL_TIMEOUT):
        shell.execute("for i in range(20):\n    plt.figure()\n    plt.plot([1, 2, 3, 4], '.')")
    scrollbar = sb.scrollarea.verticalScrollBar()
    assert len(sb._thumbnails) == 25
    assert sb._thumbnails[-1] == sb.current_thumbnail
    assert scrollbar.value() == scrollbar.maximum()
    with qtbot.waitSignal(shell.executed, timeout=SHELL_TIMEOUT):
        shell.execute("import time\nfor i in range(20):\n    plt.figure()\n    plt.plot([1, 2, 3, 4], '.')\n    plt.show()\n    time.sleep(.1)")
        qtbot.waitUntil(lambda : sb._first_thumbnail_shown, timeout=SHELL_TIMEOUT)
        sb.set_current_index(5)
        scrollbar.setValue(scrollbar.minimum())
    assert len(sb._thumbnails) == 45
    assert sb._thumbnails[-1] != sb.current_thumbnail
    assert scrollbar.value() != scrollbar.maximum()
    with qtbot.waitSignal(shell.executed, timeout=SHELL_TIMEOUT):
        shell.execute("fig = plt.plot([1, 2, 3, 4], '.')\n")
    assert len(sb._thumbnails) == 46
    assert sb._thumbnails[-1] == sb.current_thumbnail
    assert scrollbar.value() == scrollbar.maximum()
    CONF.set('plots', 'mute_inline_plotting', False)

@flaky(max_runs=3)
@pytest.mark.skipif(parse(ipy_release.version) >= parse('7.23.0') and parse(ipykernel.__version__) <= parse('5.5.3'), reason='Fails due to a bug in the %matplotlib magic')
@pytest.mark.skipif(sys.platform.startswith('linux'), reason='Timeouts a lot on Linux')
def test_tight_layout_option_for_inline_plot(main_window, qtbot, tmpdir):
    if False:
        while True:
            i = 10
    "\n    Test that the option to set bbox_inches to 'tight' or 'None' is\n    working when plotting inline in the IPython console. By default, figures\n    are plotted inline with bbox_inches='tight'.\n    "
    tmpdir = to_text_string(tmpdir)
    assert CONF.get('ipython_console', 'pylab/inline/bbox_inches') is True
    fig_dpi = float(CONF.get('ipython_console', 'pylab/inline/resolution'))
    fig_width = float(CONF.get('ipython_console', 'pylab/inline/width'))
    fig_height = float(CONF.get('ipython_console', 'pylab/inline/height'))
    widget = main_window.ipyconsole.get_widget()
    shell = main_window.ipyconsole.get_current_shellwidget()
    client = main_window.ipyconsole.get_current_client()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    control = main_window.ipyconsole.get_widget().get_focus_widget()
    control.setFocus()
    savefig_figname = osp.join(tmpdir, 'savefig_bbox_inches_tight.png').replace('\\', '/')
    with qtbot.waitSignal(shell.executed):
        shell.execute("import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nfig.set_size_inches(%f, %f)\nax.set_position([0.25, 0.25, 0.5, 0.5])\nax.set_xticks(range(10))\nax.xaxis.set_ticklabels([])\nax.set_yticks(range(10))\nax.yaxis.set_ticklabels([])\nax.tick_params(axis='both', length=0)\nfor loc in ax.spines:\n    ax.spines[loc].set_color('#000000')\n    ax.spines[loc].set_linewidth(2)\nax.axis([0, 9, 0, 9])\nax.plot(range(10), color='#000000', lw=2)\nfig.savefig('%s',\n            bbox_inches='tight',\n            dpi=%f)" % (fig_width, fig_height, savefig_figname, fig_dpi))
    html = shell._control.toHtml()
    img_name = re.search('<img src="(.+?)" /></p>', html).group(1)
    qimg = shell._get_image(img_name)
    assert isinstance(qimg, QImage)
    inline_figname = osp.join(tmpdir, 'inline_bbox_inches_tight.png')
    qimg.save(inline_figname)
    assert compare_images(savefig_figname, inline_figname, 0.1) is None
    CONF.set('ipython_console', 'pylab/inline/bbox_inches', False)
    with qtbot.waitSignal(client.sig_execution_state_changed, timeout=SHELL_TIMEOUT):
        widget.restart_kernel(client, False)
    qtbot.waitUntil(lambda : 'In [1]:' in control.toPlainText(), timeout=SHELL_TIMEOUT * 2)
    savefig_figname = osp.join(tmpdir, 'savefig_bbox_inches_None.png').replace('\\', '/')
    with qtbot.waitSignal(shell.executed):
        shell.execute("import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nfig.set_size_inches(%f, %f)\nax.set_position([0.25, 0.25, 0.5, 0.5])\nax.set_xticks(range(10))\nax.xaxis.set_ticklabels([])\nax.set_yticks(range(10))\nax.yaxis.set_ticklabels([])\nax.tick_params(axis='both', length=0)\nfor loc in ax.spines:\n    ax.spines[loc].set_color('#000000')\n    ax.spines[loc].set_linewidth(2)\nax.axis([0, 9, 0, 9])\nax.plot(range(10), color='#000000', lw=2)\nfig.savefig('%s',\n            bbox_inches=None,\n            dpi=%f)" % (fig_width, fig_height, savefig_figname, fig_dpi))
    html = shell._control.toHtml()
    img_name = re.search('<img src="(.+?)" /></p>', html).group(1)
    qimg = shell._get_image(img_name)
    assert isinstance(qimg, QImage)
    inline_figname = osp.join(tmpdir, 'inline_bbox_inches_None.png')
    qimg.save(inline_figname)
    assert compare_images(savefig_figname, inline_figname, 0.1) is None

def test_plot_from_collectioneditor(main_window, qtbot):
    if False:
        i = 10
        return i + 15
    '\n    Create a variable with value `[[1, 2, 3], [4, 5, 6]]`, use the variable\n    explorer to open a collection editor and plot the first sublist. Check\n    that a plot is displayed in the Plots pane.\n    '
    CONF.set('plots', 'mute_inline_plotting', True)
    shell = main_window.ipyconsole.get_current_shellwidget()
    figbrowser = main_window.plots.current_widget()
    nsb = main_window.variableexplorer.current_widget()
    assert len(figbrowser.thumbnails_sb._thumbnails) == 0
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed):
        shell.execute('nested_list = [[1, 2, 3], [4, 5, 6]]')
    main_window.variableexplorer.change_visibility(True)
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() > 0, timeout=EVAL_TIMEOUT)
    nsb.editor.setFocus()
    nsb.editor.edit_item()
    from spyder.widgets.collectionseditor import CollectionsEditor
    for child in nsb.editor.children():
        for grandchild in child.children():
            if isinstance(grandchild, CollectionsEditor):
                collections_editor = grandchild
    collections_editor.widget.editor.plot(0, 'plot')
    assert len(figbrowser.thumbnails_sb._thumbnails) == 1

@flaky(max_runs=3)
@pytest.mark.use_introspection
@pytest.mark.order(after='test_debug_unsaved_function')
def test_switcher(main_window, qtbot, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Test the use of shorten paths when necessary in the switcher.'
    switcher = main_window.switcher
    switcher_widget = switcher._switcher
    file_a = tmpdir.join('test_file_a.py')
    file_a.write('\ndef example_def():\n    pass\n\ndef example_def_2():\n    pass\n')
    main_window.editor.load(str(file_a))
    switcher.open_switcher()
    switcher_paths = [switcher_widget.model.item(item_idx).get_description() for item_idx in range(switcher_widget.model.rowCount())]
    assert osp.dirname(str(file_a)) in switcher_paths or len(str(file_a)) > 75
    switcher.on_close()
    dir_b = tmpdir
    for _ in range(3):
        dir_b = dir_b.mkdir(str(uuid.uuid4()))
    file_b = dir_b.join('test_file_b.py')
    file_b.write('bar\n')
    main_window.editor.load(str(file_b))
    switcher.open_switcher()
    file_b_text = switcher_widget.model.item(switcher_widget.model.rowCount() - 1).get_description()
    assert '...' in file_b_text
    switcher.on_close()
    search_texts = ['test_file_a', 'file_b', 'foo_spam']
    expected_paths = [file_a, file_b, None]
    for (search_text, expected_path) in zip(search_texts, expected_paths):
        switcher.open_switcher()
        qtbot.keyClicks(switcher_widget.edit, search_text)
        qtbot.wait(500)
        assert switcher_widget.count() == bool(expected_path)
        switcher.on_close()
    main_window.editor.set_current_filename(str(file_a))
    code_editor = main_window.editor.get_focus_widget()
    qtbot.waitUntil(lambda : code_editor.completions_available, timeout=COMPLETION_TIMEOUT)
    with qtbot.waitSignal(code_editor.completions_response_signal, timeout=COMPLETION_TIMEOUT):
        code_editor.request_symbols()
    qtbot.wait(9000)
    switcher.open_switcher()
    qtbot.keyClicks(switcher_widget.edit, '@')
    qtbot.wait(500)
    assert switcher_widget.count() == 2
    switcher.on_close()

@flaky(max_runs=3)
def test_editorstack_open_switcher_dlg(main_window, tmpdir, qtbot):
    if False:
        i = 10
        return i + 15
    '\n    Test that the file switcher is working as expected when called from the\n    editorstack.\n\n    Regression test for spyder-ide/spyder#10684\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    file = tmpdir.join('test_file_open_switcher_dlg.py')
    file.write('a test file for test_editorstack_open_switcher_dlg')
    main_window.editor.load(str(file))
    editorstack = main_window.editor.get_current_editorstack()
    editorstack.switcher_plugin.open_switcher()
    assert editorstack.switcher_plugin
    assert editorstack.switcher_plugin.is_visible()
    assert editorstack.switcher_plugin.count() == len(main_window.editor.get_filenames())

@flaky(max_runs=3)
@pytest.mark.use_introspection
@pytest.mark.order(after='test_debug_unsaved_function')
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='It times out too much on Windows and macOS')
def test_editorstack_open_symbolfinder_dlg(main_window, qtbot, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that the symbol finder is working as expected when called from the\n    editorstack.\n\n    Regression test for spyder-ide/spyder#10684\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    file = tmpdir.join('test_file.py')
    file.write('\ndef example_def():\n    pass\n\ndef example_def_2():\n    pass\n')
    main_window.editor.load(str(file))
    code_editor = main_window.editor.get_focus_widget()
    qtbot.waitUntil(lambda : code_editor.completions_available, timeout=COMPLETION_TIMEOUT)
    with qtbot.waitSignal(code_editor.completions_response_signal, timeout=COMPLETION_TIMEOUT):
        code_editor.request_symbols()
    qtbot.wait(5000)
    editorstack = main_window.editor.get_current_editorstack()
    editorstack.switcher_plugin.open_symbolfinder()
    qtbot.wait(500)
    assert editorstack.switcher_plugin
    assert editorstack.switcher_plugin.is_visible()
    assert editorstack.switcher_plugin.count() == 2

@flaky(max_runs=3)
@pytest.mark.skipif(running_in_ci(), reason="Can't run on CI")
def test_switcher_projects_integration(main_window, pytestconfig, qtbot, tmp_path):
    if False:
        print('Hello World!')
    'Test integration between the Switcher and Projects plugins.'
    capmanager = pytestconfig.pluginmanager.getplugin('capturemanager')
    capmanager.suspend_global_capture(in_=True)
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    switcher = main_window.switcher
    switcher_widget = switcher._switcher
    projects = main_window.projects
    projects.toggle_view_action.setChecked(True)
    editorstack = main_window.editor.get_current_editorstack()
    project_dir = tmp_path / 'test-projects-switcher'
    project_dir.mkdir()
    n_files_project = 3
    for i in range(n_files_project):
        fpath = project_dir / f'test_file{i}.py'
        fpath.touch()
    binary_file = Path(LOCATION).parents[1] / 'images' / 'windows_app_icon.ico'
    binary_file_copy = project_dir / 'windows.ico'
    shutil.copyfile(binary_file, binary_file_copy)
    with qtbot.waitSignal(projects.sig_project_loaded):
        projects.create_project(str(project_dir))
    qtbot.waitUntil(lambda : projects.get_widget()._default_switcher_paths != [], timeout=1000)
    switcher.open_switcher()
    n_files_open = editorstack.get_stack_count()
    assert switcher.count() == n_files_open + n_files_project
    switcher.on_close()
    switcher.open_switcher()
    sections = []
    for row in range(switcher.count()):
        item = switcher_widget.model.item(row)
        if item._section_visible:
            sections.append(item.get_section())
    assert len(sections) == 2
    switcher.on_close()
    switcher.open_switcher()
    switcher.set_search_text('0')
    qtbot.wait(500)
    assert switcher.count() == 1
    switcher.on_close()
    switcher.open_switcher()
    switcher.set_search_text('foo')
    qtbot.wait(500)
    assert switcher.count() == 0
    switcher.on_close()
    switcher.open_switcher()
    switcher.set_search_text('windows')
    qtbot.wait(500)
    assert switcher.count() == 0
    switcher.on_close()
    n_files_project -= 1
    os.remove(str(project_dir / 'test_file1.py'))
    qtbot.wait(500)
    switcher.open_switcher()
    assert switcher.count() == n_files_open + n_files_project
    switcher.on_close()
    idx = projects.get_widget().treewidget.get_index(str(project_dir / 'test_file0.py'))
    projects.get_widget().treewidget.setCurrentIndex(idx)
    qtbot.keyClick(projects.get_widget().treewidget, Qt.Key_Enter)
    switcher.open_switcher()
    n_files_open = editorstack.get_stack_count()
    assert switcher.count() == n_files_open + n_files_project - 1
    switcher.on_close()
    fzf = projects.get_widget()._fzf
    projects.get_widget()._fzf = None
    projects.get_widget()._default_switcher_paths = []
    switcher.open_switcher()
    switcher.set_search_text('0')
    qtbot.wait(500)
    assert switcher.count() == 1
    switcher.on_close()
    projects.get_widget()._fzf = fzf
    capmanager.resume_global_capture()

@flaky(max_runs=3)
@pytest.mark.skipif(sys.platform == 'darwin', reason='Times out sometimes on macOS')
def test_run_static_code_analysis(main_window, qtbot):
    if False:
        while True:
            i = 10
    'This tests that the Pylint plugin is working as expected.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    from spyder.plugins.pylint.main_widget import PylintWidgetActions
    pylint_plugin = main_window.get_plugin(Plugins.Pylint)
    test_file = osp.join(LOCATION, 'script_pylint.py')
    main_window.editor.load(test_file)
    pylint_plugin.get_action(PylintWidgetActions.RunCodeAnalysis).trigger()
    qtbot.wait(3000)
    treewidget = pylint_plugin.get_widget().get_focus_widget()
    qtbot.waitUntil(lambda : treewidget.results is not None, timeout=SHELL_TIMEOUT)
    result_content = treewidget.results
    assert result_content['C:']
    pylint_version = parse(pylint.__version__)
    if pylint_version < parse('2.5.0'):
        number_of_conventions = 5
    else:
        number_of_conventions = 3
    assert len(result_content['C:']) == number_of_conventions
    main_window.editor.close_file()

@flaky(max_runs=3)
@pytest.mark.close_main_window
@pytest.mark.skipif(sys.platform.startswith('linux') and running_in_ci(), reason='It stalls the CI sometimes on Linux')
def test_troubleshooting_menu_item_and_url(main_window, qtbot, monkeypatch):
    if False:
        print('Hello World!')
    'Test that the troubleshooting menu item calls the valid URL.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    application_plugin = main_window.application
    MockQDesktopServices = Mock()
    mockQDesktopServices_instance = MockQDesktopServices()
    attr_to_patch = 'spyder.utils.qthelpers.QDesktopServices'
    monkeypatch.setattr(attr_to_patch, MockQDesktopServices)
    application_plugin.trouble_action.trigger()
    assert MockQDesktopServices.openUrl.call_count == 1
    mockQDesktopServices_instance.openUrl.called_once_with(__trouble_url__)

@flaky(max_runs=3)
@pytest.mark.skipif(os.name == 'nt', reason='It fails on Windows')
@pytest.mark.skipif(sys.platform == 'darwin' and running_in_ci(), reason='It stalls the CI sometimes on MacOS')
@pytest.mark.close_main_window
def test_help_opens_when_show_tutorial_full(main_window, qtbot):
    if False:
        return 10
    "\n    Test fix for spyder-ide/spyder#6317.\n\n    'Show tutorial' opens the help plugin if closed.\n    "
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    HELP_STR = 'Help'
    help_pane_menuitem = None
    for action in main_window.layouts.plugins_menu.get_actions():
        if action.text() == HELP_STR:
            help_pane_menuitem = action
            break
    main_window.help.toggle_view_action.setChecked(False)
    qtbot.wait(500)
    (help_tabbar, help_index) = find_desired_tab_in_window(HELP_STR, main_window)
    assert help_tabbar is None and help_index is None
    assert not isinstance(main_window.focusWidget(), ObjectComboBox)
    assert not help_pane_menuitem.isChecked()
    main_window.help.show_tutorial()
    qtbot.wait(500)
    (help_tabbar, help_index) = find_desired_tab_in_window(HELP_STR, main_window)
    assert None not in (help_tabbar, help_index)
    assert help_index == help_tabbar.currentIndex()
    assert help_pane_menuitem.isChecked()
    help_tabbar.setCurrentIndex((help_tabbar.currentIndex() + 1) % help_tabbar.count())
    qtbot.wait(500)
    (help_tabbar, help_index) = find_desired_tab_in_window(HELP_STR, main_window)
    assert None not in (help_tabbar, help_index)
    assert help_index != help_tabbar.currentIndex()
    assert help_pane_menuitem.isChecked()
    main_window.help.show_tutorial()
    qtbot.wait(500)
    (help_tabbar, help_index) = find_desired_tab_in_window(HELP_STR, main_window)
    assert None not in (help_tabbar, help_index)
    assert help_index == help_tabbar.currentIndex()
    assert help_pane_menuitem.isChecked()
    qtbot.wait(500)
    main_window.help.show_tutorial()
    (help_tabbar, help_index) = find_desired_tab_in_window(HELP_STR, main_window)
    qtbot.wait(500)
    assert None not in (help_tabbar, help_index)
    assert help_index == help_tabbar.currentIndex()
    assert help_pane_menuitem.isChecked()

@flaky(max_runs=3)
@pytest.mark.close_main_window
def test_report_issue(main_window, qtbot):
    if False:
        return 10
    'Test that the report error dialog opens correctly.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    main_window.console.report_issue()
    qtbot.waitUntil(lambda : main_window.console.get_widget()._report_dlg is not None)
    assert main_window.console.get_widget()._report_dlg.isVisible()
    assert main_window.console.get_widget()._report_dlg.close()

@flaky(max_runs=3)
@pytest.mark.skipif(not os.name == 'nt', reason='It segfaults on Linux and Mac')
def test_custom_layouts(main_window, qtbot):
    if False:
        while True:
            i = 10
    'Test that layout are showing the expected widgets visible.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    mw = main_window
    mw.first_spyder_run = False
    prefix = 'window' + '/'
    settings = mw.layouts.load_window_settings(prefix=prefix, default=True)
    for layout_idx in get_class_values(DefaultLayouts):
        with qtbot.waitSignal(mw.sig_layout_setup_ready, timeout=5000):
            layout = mw.layouts.setup_default_layouts(layout_idx, settings=settings)
            qtbot.wait(500)
            for area in layout._areas:
                if area['visible']:
                    for plugin_id in area['plugin_ids']:
                        if plugin_id not in area['hidden_plugin_ids']:
                            plugin = mw.get_plugin(plugin_id)
                            print(plugin)
                            try:
                                assert plugin.get_widget().isVisible()
                            except AttributeError:
                                assert plugin.isVisible()

@flaky(max_runs=3)
@pytest.mark.skipif(not running_in_ci() or sys.platform.startswith('linux'), reason='Only runs in CIs and fails on Linux sometimes')
def test_programmatic_custom_layouts(main_window, qtbot):
    if False:
        return 10
    '\n    Test that a custom layout gets registered and it is recognized.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    mw = main_window
    mw.first_spyder_run = False
    layout_id = 'testing layout'
    mw.get_plugin('spyder_boilerplate')
    layout = mw.layouts.get_layout(layout_id)
    with qtbot.waitSignal(mw.sig_layout_setup_ready, timeout=5000):
        mw.layouts.quick_layout_switch(layout_id)
        qtbot.wait(500)
        for area in layout._areas:
            if area['visible']:
                for plugin_id in area['plugin_ids']:
                    if plugin_id not in area['hidden_plugin_ids']:
                        plugin = mw.get_plugin(plugin_id)
                        print(plugin)
                        try:
                            assert plugin.get_widget().isVisible()
                        except AttributeError:
                            assert plugin.isVisible()

@flaky(max_runs=3)
def test_save_on_runfile(main_window, qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Test that layout are showing the expected widgets visible.'
    test_file = osp.join(LOCATION, 'script.py')
    test_file_copy = test_file[:-3] + '_copy.py'
    shutil.copyfile(test_file, test_file_copy)
    main_window.editor.load(test_file_copy)
    code_editor = main_window.editor.get_focus_widget()
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    qtbot.keyClicks(code_editor, 'test_var = 123', delay=100)
    filename = code_editor.filename
    with qtbot.waitSignal(shell.sig_prompt_ready):
        shell.execute('%runfile {}'.format(repr(remove_backslashes(filename))))
    assert shell.get_value('test_var') == 123
    main_window.editor.close_file()
    os.remove(test_file_copy)

@pytest.mark.skipif(sys.platform == 'darwin', reason='Fails on macOS')
@pytest.mark.skipif(sys.platform.startswith('linux'), reason='Fails on Linux sometimes')
def test_pylint_follows_file(qtbot, tmpdir, main_window):
    if False:
        i = 10
        return i + 15
    'Test that file editor focus change updates pylint combobox filename.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    pylint_plugin = main_window.get_plugin(Plugins.Pylint)
    pylint_plugin.dockwidget.show()
    pylint_plugin.dockwidget.raise_()
    basedir = tmpdir.mkdir('foo')
    for idx in range(2):
        fh = basedir.join('{}.py'.format(idx))
        fname = str(fh)
        fh.write('print("Hello world!")')
        main_window.open_file(fh)
        qtbot.wait(200)
        assert fname == pylint_plugin.get_filename()
    main_window.editor.editorsplitter.split(orientation=Qt.Vertical)
    qtbot.wait(500)
    for idx in range(4):
        fh = basedir.join('{}.py'.format(idx))
        fh.write('print("Hello world!")')
        fname = str(fh)
        main_window.open_file(fh)
        qtbot.wait(200)
        assert fname == pylint_plugin.get_filename()
    for editorstack in reversed(main_window.editor.editorstacks):
        editorstack.close_split()
        break
    qtbot.wait(1000)

@flaky(max_runs=3)
def test_report_comms_error(qtbot, main_window):
    if False:
        for i in range(10):
            print('nop')
    'Test if a comms error is correctly displayed.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed):
        shell.execute('def foo(): import foo')
    with qtbot.waitSignal(shell.executed):
        shell.execute("get_ipython().kernel.frontend_comm.register_call_handler('foo', foo)")
    try:
        shell.call_kernel(blocking=True).foo()
        assert False
    except ModuleNotFoundError as e:
        assert 'foo' in str(e)

@flaky(max_runs=3)
def test_break_while_running(main_window, qtbot, tmpdir):
    if False:
        return 10
    'Test that we can set breakpoints while running.'
    code = 'import time\nfor i in range(100):\n    print(i)\n    time.sleep(0.1)\n'
    p = tmpdir.join('loop_script.py')
    p.write(code)
    test_file = to_text_string(p)
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    debug_button = main_window.debug_button
    main_window.editor.load(test_file)
    code_editor = main_window.editor.get_focus_widget()
    main_window.debugger.clear_all_breakpoints()
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(debug_button, Qt.LeftButton)
        qtbot.wait(1000)
    qtbot.keyClicks(shell._control, '!c')
    qtbot.keyClick(shell._control, Qt.Key_Enter)
    qtbot.wait(500)
    with qtbot.waitSignal(shell.executed):
        code_editor.breakpoints_manager.toogle_breakpoint(line_number=3)
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClicks(shell._control, '!q')
        qtbot.keyClick(shell._control, Qt.Key_Enter)
    main_window.debugger.clear_all_breakpoints()

@flaky(max_runs=5)
def test_preferences_run_section_exists(main_window, qtbot):
    if False:
        return 10
    '\n    Test for spyder-ide/spyder#13524 regression.\n    Ensure the Run section exists.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    (dlg, index, page) = preferences_dialog_helper(qtbot, main_window, 'run')
    assert page
    dlg.ok_btn.animateClick()
    preferences = main_window.preferences
    container = preferences.get_container()
    qtbot.waitUntil(lambda : container.dialog is None, timeout=5000)

def test_preferences_checkboxes_not_checked_regression(main_window, qtbot):
    if False:
        return 10
    '\n    Test for spyder-ide/spyder/#10139 regression.\n\n    Enabling codestyle/docstyle on the completion section of preferences,\n    was not updating correctly.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    CONF.set('completions', ('provider_configuration', 'lsp', 'values', 'pydocstyle'), False)
    CONF.set('completions', ('provider_configuration', 'lsp', 'values', 'pycodestyle'), False)
    (dlg, index, page) = preferences_dialog_helper(qtbot, main_window, 'completions')
    tnames = [page.tabs.tabText(i).lower() for i in range(page.tabs.count())]
    tabs = [(page.tabs.widget(i).layout().itemAt(0).widget(), i) for i in range(page.tabs.count())]
    tabs = dict(zip(tnames, tabs))
    tab_widgets = {'code style and formatting': 'code_style_check', 'docstring style': 'docstring_style_check'}
    for tabname in tab_widgets:
        (tab, idx) = tabs[tabname]
        check_name = tab_widgets[tabname]
        check = getattr(tab, check_name)
        page.tabs.setCurrentIndex(idx)
        check.checkbox.animateClick()
        qtbot.wait(500)
    dlg.ok_btn.animateClick()
    preferences = main_window.preferences
    container = preferences.get_container()
    qtbot.waitUntil(lambda : container.dialog is None, timeout=5000)
    count = 0
    mainmenu = main_window.get_plugin(Plugins.MainMenu)
    source_menu_actions = mainmenu.get_application_menu(ApplicationMenus.Source).get_actions()
    for menu_item in source_menu_actions:
        if menu_item and isinstance(menu_item, QAction):
            print(menu_item.text(), menu_item.isChecked())
            if 'code style' in menu_item.text():
                assert menu_item.isChecked()
                count += 1
            elif 'docstring style' in menu_item.text():
                assert menu_item.isChecked()
                count += 1
    assert count == 2
    CONF.set('completions', ('provider_configuration', 'lsp', 'values', 'pydocstyle'), False)
    CONF.set('completions', ('provider_configuration', 'lsp', 'values', 'pycodestyle'), False)

@pytest.mark.skipif(sys.platform.startswith('linux'), reason='Makes other tests hang on Linux')
def test_preferences_change_font_regression(main_window, qtbot):
    if False:
        print('Hello World!')
    '\n    Test for spyder-ide/spyder#10284 regression.\n\n    Changing font resulted in error.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    (dlg, index, page) = preferences_dialog_helper(qtbot, main_window, 'appearance')
    fontbox = page.plain_text_font.fontbox
    current_family = fontbox.currentFont().family()
    fontbox.setFocus()
    idx = fontbox.currentIndex()
    fontbox.setCurrentIndex(idx + 1)
    dlg.apply_btn.animateClick()
    qtbot.wait(1000)
    new_family = fontbox.currentFont().family()
    assert new_family != current_family
    ipyconsole = main_window.ipyconsole
    assert ipyconsole.get_current_shellwidget().font.family() == new_family
    preferences = main_window.preferences
    container = preferences.get_container()
    dlg.ok_btn.animateClick()
    qtbot.waitUntil(lambda : container.dialog is None, timeout=5000)

@pytest.mark.skipif(running_in_ci(), reason='Fails on CIs')
@pytest.mark.parametrize('main_window', [{'spy_config': ('run', 'run_cell_copy', True)}], indirect=True)
def test_preferences_empty_shortcut_regression(main_window, qtbot):
    if False:
        print('Hello World!')
    '\n    Test for spyder-ide/spyder/#12992 regression.\n\n    Overwriting shortcuts results in a shortcuts conflict.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    base_run_cell_advance = CONF.get_shortcut('editor', 'run cell and advance')
    base_run_selection = CONF.get_shortcut('_', 'run selection')
    assert base_run_cell_advance == 'Shift+Return'
    assert base_run_selection == 'F9'
    CONF.set_shortcut('editor', 'run cell and advance', '')
    CONF.set_shortcut('_', 'run selection', base_run_cell_advance)
    with qtbot.waitSignal(main_window.shortcuts.sig_shortcuts_updated):
        main_window.shortcuts.apply_shortcuts()
    main_window.editor.new()
    code_editor = main_window.editor.get_focus_widget()
    code_editor.set_text(u'print(0)\n#%%\nprint(ññ)')
    fname = main_window.editor.get_current_filename()
    run_parameters = generate_run_parameters(main_window, fname)
    CONF.set('run', 'last_used_parameters', run_parameters)
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(code_editor, Qt.Key_Return, modifier=Qt.ShiftModifier)
    qtbot.waitUntil(lambda : u'print(0)' in shell._control.toPlainText())
    assert u'ññ' not in shell._control.toPlainText()
    CONF.set_shortcut('_', 'run selection', 'F9')
    CONF.set_shortcut('editor', 'run cell and advance', 'Shift+Return')
    with qtbot.waitSignal(main_window.shortcuts.sig_shortcuts_updated):
        main_window.shortcuts.apply_shortcuts()
    code_editor.setFocus()
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(code_editor, Qt.Key_Return, modifier=Qt.ShiftModifier)
    qtbot.waitUntil(lambda : u'ññ' in shell._control.toPlainText(), timeout=EVAL_TIMEOUT)
    assert u'ññ' in shell._control.toPlainText()

def test_preferences_shortcut_reset_regression(main_window, qtbot):
    if False:
        return 10
    '\n    Test for spyder-ide/spyder/#11132 regression.\n\n    Resetting shortcut resulted in error.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    (dlg, index, page) = preferences_dialog_helper(qtbot, main_window, 'shortcuts')
    page.reset_to_default(force=True)
    dlg.ok_btn.animateClick()
    qtbot.waitUntil(lambda : main_window.preferences.get_container().dialog is None, timeout=EVAL_TIMEOUT)

@pytest.mark.order(1)
@flaky(max_runs=3)
def test_preferences_change_interpreter(qtbot, main_window):
    if False:
        while True:
            i = 10
    'Test that on main interpreter change signal is emitted.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    lsp = main_window.completions.get_provider('lsp')
    config = lsp.generate_python_config()
    jedi = config['configurations']['pylsp']['plugins']['jedi']
    assert jedi['environment'] is sys.executable
    assert jedi['extra_paths'] == []
    (dlg, index, page) = preferences_dialog_helper(qtbot, main_window, 'main_interpreter')
    page.cus_exec_radio.radiobutton.setChecked(True)
    page.cus_exec_combo.combobox.setCurrentText(sys.executable)
    mi_container = main_window.main_interpreter.get_container()
    with qtbot.waitSignal(mi_container.sig_interpreter_changed, timeout=5000, raising=True):
        dlg.ok_btn.animateClick()
    config = lsp.generate_python_config()
    jedi = config['configurations']['pylsp']['plugins']['jedi']
    assert jedi['environment'] == sys.executable
    assert jedi['extra_paths'] == []

@pytest.mark.skipif(sys.platform.startswith('linux'), reason='Segfaults on Linux')
def test_preferences_last_page_is_loaded(qtbot, main_window):
    if False:
        for i in range(10):
            print('nop')
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    (dlg, index, page) = preferences_dialog_helper(qtbot, main_window, 'main_interpreter')
    preferences = main_window.preferences
    container = preferences.get_container()
    qtbot.waitUntil(lambda : container.dialog is not None, timeout=5000)
    dlg.ok_btn.animateClick()
    qtbot.waitUntil(lambda : container.dialog is None, timeout=5000)
    main_window.show_preferences()
    qtbot.waitUntil(lambda : container.dialog is not None, timeout=5000)
    dlg = container.dialog
    assert dlg.get_current_index() == index
    dlg.ok_btn.animateClick()
    qtbot.waitUntil(lambda : container.dialog is None, timeout=5000)

@flaky(max_runs=3)
@pytest.mark.use_introspection
@pytest.mark.order(after='test_debug_unsaved_function')
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='It times out too much on Windows and macOS')
def test_go_to_definition(main_window, qtbot, capsys):
    if False:
        return 10
    'Test that go-to-definition works as expected.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    code_no_def = dedent('\n    from qtpy.QtCore import Qt\n    Qt.FramelessWindowHint')
    main_window.editor.new(text=code_no_def)
    code_editor = main_window.editor.get_focus_widget()
    qtbot.waitUntil(lambda : code_editor.completions_available, timeout=COMPLETION_TIMEOUT)
    code_editor.move_cursor(-1)
    with qtbot.waitSignal(code_editor.completions_response_signal, timeout=COMPLETION_TIMEOUT):
        code_editor.go_to_definition_from_cursor()
    sys_stream = capsys.readouterr()
    assert sys_stream.err == u''
    code_def = 'import qtpy.QtCore'
    main_window.editor.new(text=code_def)
    code_editor = main_window.editor.get_focus_widget()
    qtbot.waitUntil(lambda : code_editor.completions_available, timeout=COMPLETION_TIMEOUT)
    code_editor.move_cursor(-1)
    with qtbot.waitSignal(code_editor.completions_response_signal, timeout=COMPLETION_TIMEOUT):
        code_editor.go_to_definition_from_cursor()

    def _get_filenames():
        if False:
            print('Hello World!')
        return [osp.basename(f) for f in main_window.editor.get_filenames()]
    qtbot.waitUntil(lambda : 'QtCore.py' in _get_filenames())
    assert 'QtCore.py' in _get_filenames()

@flaky(max_runs=3)
@pytest.mark.skipif(sys.platform == 'darwin', reason='It times out on macOS')
def test_debug_unsaved_file(main_window, qtbot):
    if False:
        i = 10
        return i + 15
    'Test that we can debug an unsaved file.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    control = shell._control
    debug_button = main_window.debug_button
    main_window.debugger.clear_all_breakpoints()
    main_window.editor.new()
    code_editor = main_window.editor.get_focus_widget()
    code_editor.set_text('print(0)\nprint(1)\nprint(2)')
    code_editor.breakpoints_manager.toogle_breakpoint(line_number=2)
    qtbot.wait(500)
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(debug_button, Qt.LeftButton)
    assert '1---> 2 print(1)' in control.toPlainText()
    assert shell.is_waiting_pdb_input()

@flaky(max_runs=3)
@pytest.mark.parametrize('debug', [True, False])
@pytest.mark.known_leak
def test_runcell(main_window, qtbot, tmpdir, debug):
    if False:
        return 10
    'Test the runcell command.'
    code = u'result = 10; fname = __file__'
    p = tmpdir.join('cell-test.py')
    p.write(code)
    main_window.editor.load(to_text_string(p))
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    if debug:
        function = 'debugcell'
    else:
        function = 'runcell'
    with qtbot.waitSignal(shell.executed):
        shell.execute('%{} -i 0 {}'.format(function, repr(to_text_string(p))))
    if debug:
        shell.pdb_execute('!c')
    qtbot.wait(1000)
    assert shell.get_value('result') == 10
    assert 'cell-test.py' in shell.get_value('fname')
    try:
        shell.get_value('__file__')
        assert False
    except KeyError:
        pass

@flaky(max_runs=3)
def test_runcell_leading_indent(main_window, qtbot, tmpdir):
    if False:
        i = 10
        return i + 15
    'Test the runcell command with leading indent.'
    code = "def a():\n    return\nif __name__ == '__main__':\n# %%\n    print(1233 + 1)\n"
    p = tmpdir.join('cell-test.py')
    p.write(code)
    main_window.editor.load(to_text_string(p))
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed):
        shell.execute('%runcell -i 1 {}'.format(repr(to_text_string(p))))
    assert '1234' in shell._control.toPlainText()
    assert 'This is not valid Python code' not in shell._control.toPlainText()

@flaky(max_runs=3)
@pytest.mark.order(after='test_debug_unsaved_function')
def test_varexp_rename(main_window, qtbot, tmpdir):
    if False:
        return 10
    '\n    Test renaming a variable.\n    Regression test for spyder-ide/spyder#10735\n    '
    p = tmpdir.mkdir(u'varexp_rename').join(u'script.py')
    filepath = to_text_string(p)
    shutil.copyfile(osp.join(LOCATION, 'script.py'), filepath)
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    main_window.editor.load(filepath)
    code_editor = main_window.editor.get_focus_widget()
    code_editor.setFocus()
    qtbot.keyClick(code_editor, Qt.Key_Home, modifier=Qt.ControlModifier)
    nsb = main_window.variableexplorer.current_widget()
    run_parameters = generate_run_parameters(main_window, filepath)
    CONF.set('run', 'last_used_parameters', run_parameters)
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(main_window.run_button, Qt.LeftButton)
    qtbot.waitUntil(lambda : nsb.editor.model.rowCount() == 4, timeout=EVAL_TIMEOUT)
    nsb.editor.setCurrentIndex(nsb.editor.model.index(1, 0))
    nsb.editor.rename_item(new_name='arr2')

    def data(cm, i, j):
        if False:
            i = 10
            return i + 15
        return cm.data(cm.index(i, j))
    qtbot.waitUntil(lambda : data(nsb.editor.model, 1, 0) == 'arr2', timeout=EVAL_TIMEOUT)
    assert data(nsb.editor.model, 0, 0) == 'a'
    assert data(nsb.editor.model, 1, 0) == 'arr2'
    assert data(nsb.editor.model, 2, 0) == 'li'
    assert data(nsb.editor.model, 3, 0) == 's'
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(main_window.run_button, Qt.LeftButton)
    qtbot.waitUntil(lambda : nsb.editor.model.rowCount() == 5, timeout=EVAL_TIMEOUT)
    assert data(nsb.editor.model, 0, 0) == 'a'
    assert data(nsb.editor.model, 1, 0) == 'arr'
    assert data(nsb.editor.model, 2, 0) == 'arr2'
    assert data(nsb.editor.model, 3, 0) == 'li'
    assert data(nsb.editor.model, 4, 0) == 's'

@flaky(max_runs=3)
@pytest.mark.order(after='test_debug_unsaved_function')
def test_varexp_remove(main_window, qtbot, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test removing a variable.\n    Regression test for spyder-ide/spyder#10709\n    '
    p = tmpdir.mkdir(u'varexp_remove').join(u'script.py')
    filepath = to_text_string(p)
    shutil.copyfile(osp.join(LOCATION, 'script.py'), filepath)
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    main_window.editor.load(filepath)
    code_editor = main_window.editor.get_focus_widget()
    code_editor.setFocus()
    qtbot.keyClick(code_editor, Qt.Key_Home, modifier=Qt.ControlModifier)
    nsb = main_window.variableexplorer.current_widget()
    run_parameters = generate_run_parameters(main_window, filepath)
    CONF.set('run', 'last_used_parameters', run_parameters)
    with qtbot.waitSignal(shell.executed, timeout=SHELL_TIMEOUT):
        qtbot.mouseClick(main_window.run_button, Qt.LeftButton)
    qtbot.waitUntil(lambda : nsb.editor.model.rowCount() == 4, timeout=EVAL_TIMEOUT)
    nsb.editor.setCurrentIndex(nsb.editor.model.index(1, 0))
    nsb.editor.remove_item(force=True)
    qtbot.waitUntil(lambda : nsb.editor.model.rowCount() == 3, timeout=EVAL_TIMEOUT)

    def data(cm, i, j):
        if False:
            print('Hello World!')
        assert cm.rowCount() == 3
        return cm.data(cm.index(i, j))
    assert data(nsb.editor.model, 0, 0) == 'a'
    assert data(nsb.editor.model, 1, 0) == 'li'
    assert data(nsb.editor.model, 2, 0) == 's'

@flaky(max_runs=3)
def test_varexp_refresh(main_window, qtbot):
    if False:
        i = 10
        return i + 15
    '\n    Test refreshing the variable explorer while the kernel is executing.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    control = main_window.ipyconsole.get_widget().get_focus_widget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    shell.execute("import time\nfor i in range(10):\n    print('i = {}'.format(i))\n    time.sleep(.1)\n")
    qtbot.waitUntil(lambda : 'i = 0' in control.toPlainText())
    qtbot.wait(300)
    nsb = main_window.variableexplorer.current_widget()
    assert len(nsb.editor.source_model._data) == 0
    nsb.refresh_table()
    qtbot.waitUntil(lambda : len(nsb.editor.source_model._data) == 1)
    assert 0 < int(nsb.editor.source_model._data['i']['view']) < 9

@flaky(max_runs=3)
@pytest.mark.no_new_console
@pytest.mark.skipif(sys.platform == 'darwin' or os.name == 'nt', reason='Fails on macOS and Windows')
@pytest.mark.parametrize('main_window', [{'spy_config': ('run', 'run_cell_copy', False)}], indirect=True)
@pytest.mark.order(after='test_debug_unsaved_function')
def test_runcell_edge_cases(main_window, qtbot, tmpdir):
    if False:
        i = 10
        return i + 15
    '\n    Test if runcell works with an unnamed cell at the top of the file\n    and with an empty cell.\n    '
    code = 'if True:\n    a = 1\n#%%'
    p = tmpdir.join('test.py')
    p.write(code)
    main_window.editor.load(to_text_string(p))
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    fname = main_window.editor.get_current_filename()
    run_parameters = generate_run_parameters(main_window, fname)
    CONF.set('run', 'last_used_parameters', run_parameters)
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(main_window.run_cell_and_advance_button, Qt.LeftButton)
    qtbot.waitUntil(lambda : '%runcell -i 0' in shell._control.toPlainText(), timeout=SHELL_TIMEOUT)
    assert '%runcell -i 0' in shell._control.toPlainText()
    assert 'cell is empty' not in shell._control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(main_window.run_cell_and_advance_button, Qt.LeftButton)
    qtbot.waitUntil(lambda : '%runcell -i 1' in shell._control.toPlainText(), timeout=SHELL_TIMEOUT)
    assert '%runcell -i 1' in shell._control.toPlainText()
    assert 'Error' not in shell._control.toPlainText()
    assert 'cell is empty' in shell._control.toPlainText()

@flaky(max_runs=3)
@pytest.mark.skipif(sys.platform == 'darwin' or os.name == 'nt', reason='Fails on Mac and Windows')
@pytest.mark.order(after='test_debug_unsaved_function')
def test_runcell_pdb(main_window, qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Test the runcell command in pdb.'
    code = "if 'abba' in dir():\n    print('abba {}'.format(abba))\nelse:\n    def foo():\n        abba = 27\n    foo()\n"
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    debug_button = main_window.debug_button
    main_window.debugger.clear_all_breakpoints()
    main_window.editor.new()
    code_editor = main_window.editor.get_focus_widget()
    code_editor.set_text(code)
    fname = main_window.editor.get_current_filename()
    run_parameters = generate_run_parameters(main_window, fname)
    CONF.set('run', 'last_used_parameters', run_parameters)
    with qtbot.waitSignal(shell.executed, timeout=SHELL_TIMEOUT):
        qtbot.mouseClick(debug_button, Qt.LeftButton)
    for key in ['!n', '!n', '!s', '!n', '!n']:
        with qtbot.waitSignal(shell.executed, timeout=SHELL_TIMEOUT):
            qtbot.keyClicks(shell._control, key)
            qtbot.keyClick(shell._control, Qt.Key_Enter)
    assert shell.get_value('abba') == 27
    code_editor.setFocus()
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(main_window.run_cell_and_advance_button, Qt.LeftButton)
    assert 'runcell' in shell._control.toPlainText()
    assert 'abba 27' in shell._control.toPlainText()

@flaky(max_runs=3)
@pytest.mark.parametrize('debug', [False, True])
@pytest.mark.skipif(os.name == 'nt', reason='Timeouts on Windows')
@pytest.mark.order(after='test_debug_unsaved_function')
def test_runcell_cache(main_window, qtbot, debug):
    if False:
        i = 10
        return i + 15
    'Test the runcell command cache.'
    code = "import time\ntime.sleep(.5)\n# %%\nprint('Done')\n"
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    main_window.editor.new()
    code_editor = main_window.editor.get_focus_widget()
    code_editor.set_text(code)
    fname = main_window.editor.get_current_filename()
    run_parameters = generate_run_parameters(main_window, fname)
    CONF.set('run', 'last_used_parameters', run_parameters)
    if debug:
        with qtbot.waitSignal(shell.executed):
            shell.execute('%debug print()')
    code_editor.setFocus()
    code_editor.move_cursor(0)
    for _ in range(2):
        with qtbot.waitSignal(shell.executed):
            qtbot.mouseClick(main_window.run_cell_and_advance_button, Qt.LeftButton)
    qtbot.waitUntil(lambda : 'Done' in shell._control.toPlainText())

@flaky(max_runs=3)
def test_path_manager_updates_clients(qtbot, main_window, tmpdir):
    if False:
        i = 10
        return i + 15
    'Check that on path manager updates, consoles correctly update.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    python_path_manager = main_window.get_plugin(Plugins.PythonpathManager)
    python_path_manager.show_path_manager()
    dlg = python_path_manager.path_manager_dialog
    test_folder = 'foo-spam-bar-123'
    folder = str(tmpdir.mkdir(test_folder))
    dlg.add_path(folder)
    qtbot.waitUntil(lambda : dlg.button_ok.isEnabled(), timeout=EVAL_TIMEOUT)
    with qtbot.waitSignal(dlg.sig_path_changed, timeout=EVAL_TIMEOUT):
        dlg.button_ok.animateClick()
    cmd = 'import sys;print(sys.path)'
    shells = [c.shellwidget for c in main_window.ipyconsole.get_clients() if c is not None]
    assert len(shells) >= 1
    for shell in shells:
        control = shell._control
        control.setFocus()
        qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
        with qtbot.waitSignal(shell.executed, timeout=SHELL_TIMEOUT):
            shell.execute(cmd)
        qtbot.waitUntil(lambda : test_folder in control.toPlainText(), timeout=SHELL_TIMEOUT)

@flaky(max_runs=3)
@pytest.mark.skipif(os.name == 'nt' or sys.platform == 'darwin', reason='It times out on macOS and Windows')
def test_pdb_key_leak(main_window, qtbot, tmpdir):
    if False:
        while True:
            i = 10
    "\n    Check that pdb notify spyder doesn't call\n    QApplication.processEvents(). If it does there might be keystoke leakage.\n    see #10834\n    "
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    control = shell._control
    code1 = 'def a():\n    1/0'
    code2 = 'from tmp import a\na()'
    folder = tmpdir.join('tmp_folder')
    test_file = folder.join('tmp.py')
    test_file.write(code1, ensure=True)
    test_file2 = folder.join('tmp2.py')
    test_file2.write(code2)
    with qtbot.waitSignal(shell.executed):
        shell.execute('%runfile ' + repr(str(test_file2).replace('\\', '/')) + ' --wdir ' + repr(str(folder).replace('\\', '/')))
    assert '1/0' in control.toPlainText()
    super_processEvents = QApplication.processEvents

    def processEvents():
        if False:
            while True:
                i = 10
        processEvents.called = True
        return super_processEvents()
    processEvents.called = False
    try:
        QApplication.processEvents = processEvents
        with qtbot.waitSignal(shell.executed):
            shell.execute('%debug')
        with qtbot.waitSignal(shell.executed):
            qtbot.keyClicks(control, '!u')
            qtbot.keyClick(control, Qt.Key_Enter)
        qtbot.waitUntil(lambda : osp.normpath(str(test_file)) in [osp.normpath(p) for p in main_window.editor.get_filenames()])
        qtbot.waitUntil(lambda : str(test_file2) in [osp.normpath(p) for p in main_window.editor.get_filenames()])
        assert not processEvents.called
    finally:
        QApplication.processEvents = super_processEvents

@flaky(max_runs=3)
@pytest.mark.skipif(sys.platform == 'darwin', reason='It times out on macOS')
@pytest.mark.parametrize('where', [True, False])
def test_pdb_step(main_window, qtbot, tmpdir, where):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check that pdb notify Spyder only moves when a new line is reached.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    control = shell._control
    code1 = 'def a():\n    1/0'
    code2 = 'from tmp import a\na()'
    folder = tmpdir.join('tmp_folder')
    test_file = folder.join('tmp.py')
    test_file.write(code1, ensure=True)
    test_file2 = folder.join('tmp2.py')
    test_file2.write(code2)
    with qtbot.waitSignal(shell.executed):
        shell.execute('%runfile ' + repr(str(test_file2).replace('\\', '/')) + ' --wdir ' + repr(str(folder).replace('\\', '/')))
    qtbot.wait(1000)
    assert '1/0' in control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug')
    qtbot.waitUntil(lambda : osp.samefile(main_window.editor.get_current_editor().filename, str(test_file)))
    main_window.editor.new()
    qtbot.wait(100)
    assert main_window.editor.get_current_editor().filename != str(test_file)
    current_filename = main_window.editor.get_current_editor().filename
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClicks(control, '!a')
        qtbot.keyClick(control, Qt.Key_Enter)
    qtbot.wait(1000)
    assert current_filename == main_window.editor.get_current_editor().filename
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClicks(control, '!u')
        qtbot.keyClick(control, Qt.Key_Enter)
    qtbot.waitUntil(lambda : osp.samefile(main_window.editor.get_current_editor().filename, str(test_file2)))
    editor_stack = main_window.editor.get_current_editorstack()
    index = editor_stack.has_filename(str(test_file))
    assert index is not None
    editor_stack.set_stack_index(index)
    assert osp.samefile(main_window.editor.get_current_editor().filename, str(test_file))
    if where:
        with qtbot.waitSignal(shell.executed):
            qtbot.keyClicks(control, '!w')
            qtbot.keyClick(control, Qt.Key_Enter)
        qtbot.wait(1000)
        assert osp.samefile(main_window.editor.get_current_editor().filename, str(test_file2))
    else:
        with qtbot.waitSignal(shell.executed):
            qtbot.keyClicks(control, '!a')
            qtbot.keyClick(control, Qt.Key_Enter)
        qtbot.wait(1000)
        assert osp.samefile(main_window.editor.get_current_editor().filename, str(test_file))

@flaky(max_runs=3)
@pytest.mark.skipif(sys.platform == 'darwin' or os.name == 'nt', reason='Fails sometimes on macOS and Windows')
@pytest.mark.order(after='test_debug_unsaved_function')
def test_runcell_after_restart(main_window, qtbot):
    if False:
        return 10
    'Test runcell after a kernel restart.'
    code = "print('test_runcell_after_restart')"
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    main_window.editor.new()
    code_editor = main_window.editor.get_focus_widget()
    code_editor.set_text(code)
    fname = main_window.editor.get_current_filename()
    run_parameters = generate_run_parameters(main_window, fname)
    CONF.set('run', 'last_used_parameters', run_parameters)
    widget = main_window.ipyconsole.get_widget()
    with qtbot.waitSignal(shell.sig_prompt_ready, timeout=10000):
        widget.restart_kernel(shell.ipyclient, False)
    code_editor.setFocus()
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(main_window.run_cell_and_advance_button, Qt.LeftButton)
    qtbot.waitUntil(lambda : 'test_runcell_after_restart' in shell._control.toPlainText())
    assert 'error' not in shell._control.toPlainText().lower()

@flaky(max_runs=3)
@pytest.mark.skipif(not os.name == 'nt', reason='Sometimes fails on Linux and hangs on Mac')
@pytest.mark.parametrize('ipython', [True, False])
@pytest.mark.parametrize('test_cell_magic', [True, False])
def test_ipython_magic(main_window, qtbot, tmpdir, ipython, test_cell_magic):
    if False:
        while True:
            i = 10
    'Test the runcell command with cell magic.'
    write_file = tmpdir.mkdir('foo').join('bar.txt')
    assert not osp.exists(to_text_string(write_file))
    if test_cell_magic:
        code = '\n\n%%writefile ' + to_text_string(write_file) + '\ntest\n'
    else:
        code = '\n\n%debug print()'
    if ipython:
        fn = 'cell-test.ipy'
    else:
        fn = 'cell-test.py'
    p = tmpdir.join(fn)
    p.write(code)
    main_window.editor.load(to_text_string(p))
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed):
        shell.execute('%runcell -i 0 {}'.format(repr(to_text_string(p))))
    control = main_window.ipyconsole.get_widget().get_focus_widget()
    error_text = 'save this file with the .ipy extension'
    try:
        if ipython:
            if test_cell_magic:
                qtbot.waitUntil(lambda : 'Writing' in control.toPlainText())
                assert osp.exists(to_text_string(write_file))
            else:
                qtbot.waitSignal(shell.executed)
            assert error_text not in control.toPlainText()
        else:
            qtbot.waitUntil(lambda : error_text in control.toPlainText())
    finally:
        if osp.exists(to_text_string(write_file)):
            os.remove(to_text_string(write_file))

@flaky(max_runs=3)
def test_running_namespace(main_window, qtbot, tmpdir):
    if False:
        i = 10
        return i + 15
    '\n    Test that the running namespace is correctly sent when debugging in a\n    new namespace.\n    '
    code = "def test(a):\n    print('a:',a)\na = 10\ntest(5)"
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    debug_button = main_window.debug_button
    main_window.debugger.clear_all_breakpoints()
    main_window.editor.new()
    code_editor = main_window.editor.get_focus_widget()
    code_editor.set_text(code)
    code_editor.breakpoints_manager.toogle_breakpoint(line_number=2)
    with qtbot.waitSignal(shell.executed):
        shell.execute('b = 10')
    nsb = main_window.variableexplorer.current_widget()
    qtbot.waitUntil(lambda : 'b' in nsb.editor.source_model._data)
    assert nsb.editor.source_model._data['b']['view'] == '10'
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(debug_button, Qt.LeftButton)
    qtbot.waitUntil(lambda : 'a' in nsb.editor.source_model._data and nsb.editor.source_model._data['a']['view'] == '5', timeout=3000)
    assert 'b' not in nsb.editor.source_model._data
    assert nsb.editor.source_model._data['a']['view'] == '5'
    qtbot.waitUntil(shell.is_waiting_pdb_input)
    with qtbot.waitSignal(shell.executed):
        shell.pdb_execute('!c')
    qtbot.waitUntil(lambda : 'b' in nsb.editor.source_model._data)
    assert nsb.editor.source_model._data['a']['view'] == '10'
    assert nsb.editor.source_model._data['b']['view'] == '10'

@flaky(max_runs=3)
def test_running_namespace_refresh(main_window, qtbot, tmpdir):
    if False:
        while True:
            i = 10
    '\n    Test that the running namespace can be accessed recursively\n    '
    code_i = 'import time\nfor i in range(10):\n    time.sleep(.1)\n'
    code_j = 'import time\nfor j in range(10):\n    time.sleep(.1)\n'
    file1 = tmpdir.join('file1.py')
    file1.write(code_i)
    file2 = tmpdir.join('file2.py')
    file2.write(code_j)
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    main_window.debugger.clear_all_breakpoints()
    shell.execute('%runfile ' + repr(str(file2)))
    nsb = main_window.variableexplorer.current_widget()
    assert len(nsb.editor.source_model._data) == 0
    qtbot.wait(500)
    nsb.refresh_table()
    qtbot.waitUntil(lambda : len(nsb.editor.source_model._data) == 1)
    assert 0 < int(nsb.editor.source_model._data['j']['view']) <= 9
    qtbot.waitSignal(shell.executed)
    with qtbot.waitSignal(shell.executed):
        shell.execute('del j')
    qtbot.waitUntil(lambda : len(nsb.editor.source_model._data) == 0)
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debugfile ' + repr(str(file1)))
    shell.execute('c')
    qtbot.wait(500)
    nsb.refresh_table()
    qtbot.waitUntil(lambda : len(nsb.editor.source_model._data) == 1)
    assert 0 < int(nsb.editor.source_model._data['i']['view']) <= 9

@flaky(max_runs=3)
def test_debug_namespace(main_window, qtbot, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that the running namespace is correctly sent when debugging\n\n    Regression test for spyder-ide/spyder-kernels#394.\n    '
    code1 = 'file1_global_ns = True\ndef f(file1_local_ns = True):\n    return\n'
    code2 = 'from file1 import f\nfile2_global_ns = True\nf()\n'
    file1 = tmpdir.join('file1.py')
    file1.write(code1)
    file2 = tmpdir.join('file2.py')
    file2.write(code2)
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    main_window.debugger.clear_all_breakpoints()
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debugfile ' + repr(str(file2)) + ' --wdir ' + repr(str(tmpdir)))
    nsb = main_window.variableexplorer.current_widget()
    assert len(nsb.editor.source_model._data) == 0
    with qtbot.waitSignal(shell.executed):
        shell.execute('n')
    with qtbot.waitSignal(shell.executed):
        shell.execute('n')
    qtbot.waitUntil(lambda : len(nsb.editor.source_model._data) == 1)
    assert 'file2_global_ns' in nsb.editor.source_model._data
    with qtbot.waitSignal(shell.executed):
        shell.execute('s')
    qtbot.waitUntil(lambda : len(nsb.editor.source_model._data) == 2)
    assert 'file2_global_ns' not in nsb.editor.source_model._data
    assert 'file1_global_ns' in nsb.editor.source_model._data
    assert 'file1_local_ns' in nsb.editor.source_model._data

@flaky(max_runs=3)
def test_post_mortem(main_window, qtbot, tmpdir):
    if False:
        i = 10
        return i + 15
    'Test post mortem works'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    control = main_window.ipyconsole.get_widget().get_focus_widget()
    test_file = tmpdir.join('test.py')
    test_file.write('raise RuntimeError\n')
    with qtbot.waitSignal(shell.executed):
        shell.execute('%runfile ' + repr(str(test_file)) + ' --post-mortem')
    assert 'IPdb [' in control.toPlainText()

@flaky(max_runs=3)
@pytest.mark.order(after='test_debug_unsaved_function')
def test_run_unsaved_file_multiprocessing(main_window, qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Test that we can run an unsaved file with multiprocessing.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    main_window.editor.new()
    code_editor = main_window.editor.get_focus_widget()
    if sys.platform == 'darwin':
        text = 'import multiprocessing\nmultiprocessing.set_start_method("fork")\nimport traceback\nif __name__ == "__main__":\n    p = multiprocessing.Process(target=traceback.print_exc)\n    p.start()\n    p.join()\n'
    else:
        text = 'import multiprocessing\nimport traceback\nif __name__ == "__main__":\n    p = multiprocessing.Process(target=traceback.print_exc)\n    p.start()\n    p.join()\n'
    code_editor.set_text(text)
    fname = main_window.editor.get_current_filename()
    run_parameters = generate_run_parameters(main_window, fname)
    CONF.set('run', 'last_used_parameters', run_parameters)
    qtbot.mouseClick(main_window.run_button, Qt.LeftButton)
    if os.name == 'nt':
        qtbot.waitUntil(lambda : 'Warning: multiprocessing' in shell._control.toPlainText(), timeout=SHELL_TIMEOUT)
    else:
        qtbot.waitUntil(lambda : 'None' in shell._control.toPlainText(), timeout=SHELL_TIMEOUT)

@flaky(max_runs=3)
def test_varexp_cleared_after_kernel_restart(main_window, qtbot):
    if False:
        print('Hello World!')
    '\n    Test that the variable explorer is cleared after a kernel restart.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed):
        shell.execute('a = 10')
    nsb = main_window.variableexplorer.current_widget()
    qtbot.waitUntil(lambda : 'a' in nsb.editor.source_model._data, timeout=3000)
    widget = main_window.ipyconsole.get_widget()
    with qtbot.waitSignal(shell.sig_prompt_ready, timeout=10000):
        widget.restart_kernel(shell.ipyclient, False)
    qtbot.waitUntil(lambda : 'a' not in nsb.editor.source_model._data, timeout=3000)

@flaky(max_runs=3)
def test_varexp_cleared_after_reset(main_window, qtbot):
    if False:
        i = 10
        return i + 15
    '\n    Test that the variable explorer is cleared after triggering a\n    reset in the IPython console and variable explorer panes.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed):
        shell.execute('a = 10')
    nsb = main_window.variableexplorer.current_widget()
    qtbot.waitUntil(lambda : 'a' in nsb.editor.source_model._data, timeout=3000)
    nsb.reset_namespace()
    qtbot.waitUntil(lambda : 'a' not in nsb.editor.source_model._data, timeout=3000)
    with qtbot.waitSignal(shell.executed):
        shell.execute('a = 10')
    nsb = main_window.variableexplorer.current_widget()
    qtbot.waitUntil(lambda : 'a' in nsb.editor.source_model._data, timeout=3000)
    shell.ipyclient.reset_namespace()
    qtbot.waitUntil(lambda : 'a' not in nsb.editor.source_model._data, timeout=3000)

@flaky(max_runs=3)
def test_immediate_debug(main_window, qtbot):
    if False:
        return 10
    '\n    Check if we can enter debugging immediately\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed, timeout=SHELL_TIMEOUT):
        shell.execute('%debug print()')

@flaky(max_runs=3)
def test_local_namespace(main_window, qtbot, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that the local namespace is not reset.\n\n    This can happen if `frame.f_locals` is called on the current frame, as this\n    has the side effect of discarding the pdb locals.\n    '
    code = "\ndef hello():\n    test = 1\n    print('test ==', test)\nhello()\n"
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    debug_button = main_window.debug_button
    main_window.debugger.clear_all_breakpoints()
    main_window.editor.new()
    code_editor = main_window.editor.get_focus_widget()
    code_editor.set_text(code)
    code_editor.breakpoints_manager.toogle_breakpoint(line_number=4)
    nsb = main_window.variableexplorer.current_widget()
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(debug_button, Qt.LeftButton)
    qtbot.waitUntil(lambda : 'test' in nsb.editor.source_model._data and nsb.editor.source_model._data['test']['view'] == '1', timeout=3000)
    with qtbot.waitSignal(shell.executed):
        shell.execute('test = 1 + 1')
    with qtbot.waitSignal(shell.executed):
        shell.execute("print('test =', test)")
    qtbot.waitUntil(lambda : 'test = 2' in shell._control.toPlainText(), timeout=SHELL_TIMEOUT)
    assert 'test = 2' in shell._control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        shell.execute('test = 1 + 1 + 1')
    with qtbot.waitSignal(shell.executed):
        shell.pdb_execute('!next')
    qtbot.waitUntil(lambda : 'test == 3' in shell._control.toPlainText(), timeout=SHELL_TIMEOUT)
    assert 'test == 3' in shell._control.toPlainText()
    assert 'test' in nsb.editor.source_model._data and nsb.editor.source_model._data['test']['view'] == '3'

@flaky(max_runs=3)
@pytest.mark.use_introspection
@pytest.mark.order(after='test_debug_unsaved_function')
@pytest.mark.preload_project
@pytest.mark.skipif(os.name == 'nt', reason='Times out on Windows')
@pytest.mark.skipif(sys.platform.startswith('linux') and running_in_ci(), reason='Too flaky with Linux on CI')
@pytest.mark.known_leak
@pytest.mark.close_main_window
def test_ordering_lsp_requests_at_startup(main_window, qtbot):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the ordering of requests we send to the LSP at startup when a\n    project was left open during the previous session.\n\n    This is a regression test for spyder-ide/spyder#13351.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    code_editor = main_window.editor.get_current_editor()
    qtbot.waitSignal(code_editor.completions_response_signal, timeout=30000)
    lsp = main_window.completions.get_provider('lsp')
    python_client = lsp.clients['python']
    qtbot.wait(5000)
    expected_requests = ['initialize', 'initialized', 'workspace/didChangeConfiguration', 'workspace/didChangeWorkspaceFolders', 'textDocument/didOpen']
    skip_intermediate = {'initialized': {'workspace/didChangeConfiguration'}}
    lsp_requests = python_client['instance']._requests
    start_idx = lsp_requests.index((0, 'initialize'))
    request_order = []
    expected_iter = iter(expected_requests)
    current_expected = next(expected_iter)
    for i in range(start_idx, len(lsp_requests)):
        if current_expected is None:
            break
        (_, req_type) = lsp_requests[i]
        if req_type == current_expected:
            request_order.append(req_type)
            current_expected = next(expected_iter, None)
        else:
            skip_set = skip_intermediate.get(current_expected, set({}))
            if req_type in skip_set:
                continue
            else:
                assert req_type == current_expected
    assert request_order == expected_requests

@flaky(max_runs=3)
@pytest.mark.parametrize('main_window', [{'spy_config': ('tours', 'show_tour_message', True)}], indirect=True)
def test_tour_message(main_window, qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Test that the tour message displays and sends users to the tour.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    tours = main_window.get_plugin(Plugins.Tours)
    tour_dialog = tours.get_container()._tour_dialog
    animated_tour = tours.get_container()._tour_widget
    qtbot.waitSignal(main_window.sig_setup_finished, timeout=30000)
    assert tours.get_conf('show_tour_message')
    tours.show_tour_message(force=True)
    qtbot.waitUntil(lambda : bool(tour_dialog), timeout=5000)
    qtbot.waitUntil(lambda : tour_dialog.isVisible(), timeout=2000)
    qtbot.mouseClick(tour_dialog.dismiss_button, Qt.LeftButton)
    qtbot.waitUntil(lambda : not tour_dialog.isVisible(), timeout=2000)
    assert not tours.get_conf('show_tour_message')
    tours.show_tour_message()
    qtbot.wait(2000)
    assert not tour_dialog.isVisible()
    tours.show_tour_message(force=True)
    qtbot.waitUntil(lambda : tour_dialog.isVisible(), timeout=5000)
    qtbot.mouseClick(tour_dialog.launch_tour_button, Qt.LeftButton)
    qtbot.waitUntil(lambda : animated_tour.is_running, timeout=9000)
    assert not tour_dialog.isVisible()
    assert not tours.get_conf('show_tour_message')
    animated_tour.close_tour()
    qtbot.waitUntil(lambda : not animated_tour.is_running, timeout=9000)

@flaky(max_runs=3)
@pytest.mark.use_introspection
@pytest.mark.order(after='test_debug_unsaved_function')
@pytest.mark.preload_complex_project
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Only works on Linux')
@pytest.mark.known_leak
def test_update_outline(main_window, qtbot, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that files in the Outline pane are updated at startup and\n    after switching projects.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)

    def editors_filled(treewidget):
        if False:
            print('Hello World!')
        editors_py = [editor for editor in treewidget.editor_ids.keys() if editor.get_language() == 'Python']
        return all([treewidget.editor_items[editor.get_id()].node.childCount() == 2 for editor in editors_py])

    def editors_with_info(treewidget):
        if False:
            print('Hello World!')
        editors_py = [editor for editor in treewidget.editor_ids.keys() if editor.get_language() == 'Python']
        return all([editor.info is not None for editor in editors_py])
    outline_explorer = main_window.outlineexplorer
    outline_explorer.toggle_view_action.setChecked(True)
    treewidget = outline_explorer.get_widget().treewidget
    qtbot.waitUntil(lambda : editors_filled(treewidget), timeout=25000)
    editorstack_1 = main_window.editor.get_current_editorstack()
    editorstack_1.sig_split_vertically.emit()
    qtbot.wait(1000)
    qtbot.waitUntil(lambda : editors_with_info(treewidget), timeout=25000)
    editorstack_2 = main_window.editor.get_current_editorstack()
    editorstack_2.set_stack_index(2)
    editor_1 = editorstack_2.get_current_editor()
    assert osp.splitext(editor_1.filename)[1] == '.txt'
    assert editor_1.is_cloned
    editor_tree = treewidget.current_editor
    tree = treewidget.editor_tree_cache[editor_tree.get_id()]
    assert len(tree) == 0
    assert not outline_explorer.get_widget()._spinner.isSpinning()
    idx = random.choice(range(3, editorstack_2.tabs.count()))
    editorstack_2.set_stack_index(idx)
    qtbot.wait(500)
    root_1 = treewidget.editor_items[treewidget.current_editor.get_id()]
    assert root_1.node.childCount() == 2
    editorstack_1.set_stack_index(idx)
    editor_1 = editorstack_1.get_current_editor()
    editor_2 = editorstack_2.get_current_editor()
    assert editor_2.is_cloned
    assert editor_2.classfuncdropdown.class_cb.count() == 2
    assert editor_2.classfuncdropdown.method_cb.count() == 4
    assert editor_1.classfuncdropdown._data == editor_2.classfuncdropdown._data

    def get_cb_list(cb):
        if False:
            i = 10
            return i + 15
        return [cb.itemText(i) for i in range(cb.count())]
    assert get_cb_list(editor_1.classfuncdropdown.class_cb) == get_cb_list(editor_2.classfuncdropdown.class_cb)
    assert get_cb_list(editor_1.classfuncdropdown.method_cb) == get_cb_list(editor_2.classfuncdropdown.method_cb)
    with qtbot.waitSignal(editor_2.oe_proxy.sig_outline_explorer_data_changed, timeout=5000):
        editor_2.set_text('def baz(x):\n    return x')
    assert editor_2.is_cloned
    assert editor_2.classfuncdropdown.class_cb.count() == 1
    assert editor_2.classfuncdropdown.method_cb.count() == 2
    assert editor_1.classfuncdropdown._data == editor_2.classfuncdropdown._data
    assert get_cb_list(editor_1.classfuncdropdown.class_cb) == get_cb_list(editor_2.classfuncdropdown.class_cb)
    assert get_cb_list(editor_1.classfuncdropdown.method_cb) == get_cb_list(editor_2.classfuncdropdown.method_cb)
    outline_explorer.toggle_view_action.setChecked(False)
    editorstack_2.set_stack_index(0)
    editor_2 = editorstack_2.get_current_editor()
    with qtbot.waitSignal(editor_2.oe_proxy.sig_outline_explorer_data_changed, timeout=5000):
        editor_2.selectAll()
        editor_2.cut()
        editorstack_2.save()
    len(treewidget.editor_tree_cache[treewidget.current_editor.get_id()]) == 4
    prev_filenames = ['prev_file_1.py', 'prev_file_2.py']
    prev_paths = []
    for fname in prev_filenames:
        file = tmpdir.join(fname)
        file.write(read_asset_file('script_outline_1.py'))
        prev_paths.append(str(file))
    CONF.set('editor', 'filenames', prev_paths)
    main_window.projects.close_project()
    outline_explorer.toggle_view_action.setChecked(True)
    qtbot.waitUntil(lambda : editors_filled(treewidget), timeout=25000)
    editorwindow = main_window.editor.create_new_window()
    treewidget_on_window = editorwindow.editorwidget.outlineexplorer.treewidget
    qtbot.waitUntil(lambda : editors_with_info(treewidget_on_window), timeout=25000)
    main_window.activateWindow()
    editorstack_2.set_stack_index(1)
    editor_3 = editorstack_2.get_current_editor()
    with qtbot.waitSignal(editor_3.oe_proxy.sig_outline_explorer_data_changed, timeout=5000):
        editor_3.set_text('def baz(x):\n    return x')
    editorwindow.activateWindow()
    editorstack_on_window = editorwindow.editorwidget.editorstacks[0]
    editorstack_on_window.set_stack_index(1)
    qtbot.wait(500)
    root_2 = treewidget_on_window.editor_items[treewidget_on_window.current_editor.get_id()]
    qtbot.wait(500)
    assert root_2.node.childCount() == 1
    CONF.set('editor', 'filenames', [])

@flaky(max_runs=3)
@pytest.mark.use_introspection
@pytest.mark.order(3)
@pytest.mark.preload_namespace_project
@pytest.mark.known_leak
@pytest.mark.skipif(sys.platform == 'darwin', reason="Doesn't work on Mac")
def test_no_update_outline(main_window, qtbot, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the Outline is not updated in different scenarios.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    outline_explorer = main_window.outlineexplorer
    treewidget = outline_explorer.get_widget().treewidget
    editor_stack = main_window.editor.get_current_editorstack()
    outline_explorer.toggle_view_action.setChecked(False)

    def trees_update_state(treewidget):
        if False:
            for i in range(10):
                print('nop')
        proxy_editors = treewidget.editor_ids.keys()
        return [pe.is_tree_updated for pe in proxy_editors]

    def write_code(code, treewidget):
        if False:
            for i in range(10):
                print('nop')
        proxy_editors = treewidget.editor_ids.keys()
        for (i, pe) in enumerate(proxy_editors):
            code_editor = pe._editor
            with qtbot.waitSignal(pe.sig_outline_explorer_data_changed, timeout=5000):
                editor_stack.tabs.setCurrentIndex(i)
                qtbot.mouseClick(editor_stack.tabs.currentWidget(), Qt.LeftButton)
                code_editor.set_text(code.format(i=i))
                qtbot.wait(300)

    def check_symbols_number(number, treewidget):
        if False:
            print('Hello World!')
        proxy_editors = treewidget.editor_ids.keys()
        assert all([len(treewidget.editor_tree_cache[pe.get_id()]) == number for pe in proxy_editors])

    def editors_with_info(treewidget):
        if False:
            while True:
                i = 10
        editors = treewidget.editor_ids.keys()
        return all([editor.info is not None for editor in editors])

    def move_across_tabs(editorstack):
        if False:
            while True:
                i = 10
        for i in range(editorstack.tabs.count()):
            editorstack.tabs.setCurrentIndex(i)
            qtbot.mouseClick(editorstack.tabs.currentWidget(), Qt.LeftButton)
            qtbot.wait(300)
    qtbot.waitUntil(lambda : not treewidget.starting.get('python', True), timeout=10000)
    assert not any(trees_update_state(treewidget))
    write_code('def foo{i}(x):\n    return x', treewidget)
    assert not any(trees_update_state(treewidget))
    outline_explorer.toggle_view_action.setChecked(True)
    qtbot.waitUntil(lambda : all(trees_update_state(treewidget)))
    check_symbols_number(1, treewidget)
    outline_explorer.create_window()
    main_window.activateWindow()
    write_code('def bar{i}(y):\n    return y\n\ndef baz{i}(z):\n    return z', treewidget)
    check_symbols_number(2, treewidget)
    outline_explorer.get_widget().windowwidget.showMinimized()
    write_code('def func{i}(x):\n    return x', treewidget)
    assert not any(trees_update_state(treewidget))
    outline_explorer.get_widget().windowwidget.showNormal()
    qtbot.waitUntil(lambda : all(trees_update_state(treewidget)))
    check_symbols_number(1, treewidget)
    outline_explorer.toggle_view_action.setChecked(False)
    assert outline_explorer.get_widget().windowwidget is None
    write_code('def blah{i}(x):\n    return x', treewidget)
    editor_stack.save_all()
    assert not any(trees_update_state(treewidget))
    editorwindow = main_window.editor.create_new_window()
    editorwidget = editorwindow.editorwidget
    treewidget_on_window = editorwidget.outlineexplorer.treewidget
    qtbot.waitUntil(lambda : editors_with_info(treewidget_on_window), timeout=5000)
    editorwindow.showMinimized()
    main_window.activateWindow()
    write_code('def bar{i}(y):\n    return y\n\ndef baz{i}(z):\n    return z', treewidget)
    assert not any(trees_update_state(treewidget_on_window))
    editorwindow.showNormal()
    editorwindow.activateWindow()
    editorstack_on_window = editorwidget.editorstacks[0]
    move_across_tabs(editorstack_on_window)
    qtbot.waitUntil(lambda : all(trees_update_state(treewidget_on_window)))
    check_symbols_number(2, treewidget_on_window)
    splitter_on_window = editorwidget.splitter
    split_sizes = splitter_on_window.sizes()
    splitter_on_window.moveSplitter(editorwidget.size().width(), 0)
    write_code('def blah{i}(x):\n    return x', treewidget_on_window)
    assert not any(trees_update_state(treewidget_on_window))
    splitter_on_window.moveSplitter(split_sizes[0], 1)
    move_across_tabs(editorstack_on_window)
    qtbot.waitUntil(lambda : all(trees_update_state(treewidget_on_window)))
    check_symbols_number(1, treewidget_on_window)
    outline_explorer.toggle_view_action.setChecked(True)
    main_window.showMinimized()
    editorwindow.activateWindow()
    write_code('def bar{i}(y):\n    return y\n\ndef baz{i}(z):\n    return z', treewidget_on_window)
    qtbot.waitUntil(lambda : editors_with_info(treewidget_on_window), timeout=5000)
    assert not any(trees_update_state(treewidget))
    main_window.showNormal()
    main_window.showMaximized()
    qtbot.waitUntil(lambda : all(trees_update_state(treewidget)))
    check_symbols_number(2, treewidget)
    outline_explorer.toggle_view_action.setChecked(False)
    editorwindow.close()
    qtbot.wait(1000)
    outline_explorer.toggle_view_action.setChecked(True)
    main_window.projects.close_project()

@flaky(max_runs=3)
def test_prevent_closing(main_window, qtbot):
    if False:
        return 10
    '\n    Check we can bypass prevent closing.\n    '
    code = 'print(1 + 6)\nprint(1 + 6)\n'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    debug_button = main_window.debug_button
    main_window.debugger.clear_all_breakpoints()
    main_window.editor.new()
    code_editor = main_window.editor.get_focus_widget()
    code_editor.set_text(code)
    code_editor.breakpoints_manager.toogle_breakpoint(line_number=1)
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(debug_button, Qt.LeftButton)
    CONF.set('debugger', 'pdb_prevent_closing', False)
    assert main_window.editor.get_current_editorstack().close_file()
    CONF.set('debugger', 'pdb_prevent_closing', True)
    assert shell.is_debugging()

@flaky(max_runs=3)
def test_continue_first_line(main_window, qtbot):
    if False:
        return 10
    '\n    Check we can bypass prevent closing.\n    '
    CONF.set('debugger', 'pdb_stop_first_line', False)
    code = "print('a =', 1 + 6)\nprint('b =', 1 + 8)\n"
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    debug_button = main_window.debug_button
    main_window.debugger.clear_all_breakpoints()
    main_window.editor.new()
    code_editor = main_window.editor.get_focus_widget()
    code_editor.set_text(code)
    qtbot.wait(1000)
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(debug_button, Qt.LeftButton)
    qtbot.waitUntil(lambda : not shell.is_debugging())
    CONF.set('debugger', 'pdb_stop_first_line', True)
    qtbot.waitUntil(lambda : 'a = 7' in shell._control.toPlainText())
    assert 'b = 9' in shell._control.toPlainText()

@flaky(max_runs=3)
@pytest.mark.use_introspection
@pytest.mark.order(after='test_debug_unsaved_function')
@pytest.mark.skipif(os.name == 'nt', reason='Fails on Windows')
def test_outline_no_init(main_window, qtbot):
    if False:
        print('Hello World!')
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    spy_dir = osp.dirname(get_module_path('spyder'))
    main_window.editor.load(osp.join(spy_dir, 'tools', 'rm_whitespace.py'))
    outline_explorer = main_window.outlineexplorer
    outline_explorer.toggle_view_action.setChecked(True)
    qtbot.wait(5000)
    treewidget = outline_explorer.get_widget().treewidget
    editor_id = list(treewidget.editor_ids.values())[1]
    assert len(treewidget.editor_tree_cache[editor_id]) > 0

@flaky(max_runs=3)
@pytest.mark.skipif(sys.platform.startswith('linux'), reason='Flaky on Linux')
def test_pdb_ipykernel(main_window, qtbot):
    if False:
        i = 10
        return i + 15
    'Check if pdb works without spyder kernel.'
    (km, kc) = start_new_kernel()
    main_window.ipyconsole.create_client_for_kernel(kc.connection_file)
    ipyconsole = main_window.ipyconsole
    shell = ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    control = ipyconsole.get_widget().get_focus_widget()
    shell.execute('%debug print()')
    qtbot.waitUntil(lambda : 'IPdb [1]:' in control.toPlainText())
    qtbot.keyClicks(control, "print('Two: ' + str(1+1))")
    qtbot.keyClick(control, Qt.Key_Enter)
    qtbot.waitUntil(lambda : 'IPdb [2]:' in control.toPlainText())
    assert 'Two: 2' in control.toPlainText()
    with qtbot.waitSignal(shell.sig_pdb_step):
        main_window.debugger.get_widget().debug_command('step')
    with qtbot.waitSignal(shell.executed):
        shell.stop_debugging()
    shell.execute('quit()')
    qtbot.waitUntil(lambda : not km.is_alive())
    assert not km.is_alive()
    kc.stop_channels()

@flaky(max_runs=3)
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Flaky on Mac and Windows')
def test_print_comms(main_window, qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Test warning printed when comms print.'
    code = 'class Test:\n    @property\n    def shape(self):\n        print((10,))'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    control = main_window.ipyconsole.get_widget().get_focus_widget()
    nsb = main_window.variableexplorer.current_widget()
    with qtbot.waitSignal(shell.executed):
        shell.execute(code)
    assert nsb.editor.source_model.rowCount() == 0
    with qtbot.waitSignal(shell.executed):
        shell.execute('a = Test()')
    qtbot.waitUntil(lambda : nsb.editor.source_model.rowCount() == 1, timeout=EVAL_TIMEOUT)
    assert 'Output from spyder call' in control.toPlainText()

@flaky(max_runs=3)
@pytest.mark.skipif(os.name == 'nt', reason='UTF8 on Windows')
def test_goto_find(main_window, qtbot, tmpdir):
    if False:
        while True:
            i = 10
    'Test find goes to the right place.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    code = 'we Weee wee\nWe\n🚫 wee'
    match_positions = [(0, 2), (3, 7), (8, 11), (12, 14), (18, 21)]
    subdir = tmpdir.mkdir('find-sub')
    p = subdir.join('find-test.py')
    p.write(code)
    main_window.editor.load(to_text_string(p))
    code_editor = main_window.editor.get_focus_widget()
    main_window.explorer.chdir(str(subdir))
    main_window.findinfiles.switch_to_plugin()
    findinfiles = main_window.findinfiles.get_widget()
    findinfiles.set_search_text('we+')
    findinfiles.search_regexp_action.setChecked(True)
    findinfiles.case_action.setChecked(False)
    with qtbot.waitSignal(findinfiles.sig_finished, timeout=SHELL_TIMEOUT):
        findinfiles.find()
    results = findinfiles.result_browser.data
    assert len(results) == 5
    assert len(findinfiles.result_browser.files) == 1
    file_item = list(findinfiles.result_browser.files.values())[0]
    assert file_item.childCount() == 5
    for i in range(5):
        item = file_item.child(i)
        findinfiles.result_browser.setCurrentItem(item)
        findinfiles.result_browser.activated(item)
        cursor = code_editor.textCursor()
        position = (cursor.selectionStart(), cursor.selectionEnd())
        assert position == match_positions[i]

@flaky(max_runs=3)
@pytest.mark.skipif(os.name == 'nt', reason='test fails on windows.')
def test_copy_paste(main_window, qtbot, tmpdir):
    if False:
        print('Hello World!')
    'Test copy paste.'
    code = 'if True:\n    class a():\n        def b():\n            print()\n        def c():\n            print()\n'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    main_window.editor.new()
    code_editor = main_window.editor.get_focus_widget()
    code_editor.set_text(code)
    cursor = code_editor.textCursor()
    cursor.setPosition(69)
    cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
    code_editor.setTextCursor(cursor)
    qtbot.keyClick(code_editor, 'c', modifier=Qt.ControlModifier)
    assert QApplication.clipboard().text() == 'def c():\n            print()\n'
    assert CLIPBOARD_HELPER.metadata_indent == 8
    qtbot.keyClick(shell._control, 'v', modifier=Qt.ControlModifier)
    expected = 'In [1]: def c():\n   ...:     print()'
    assert expected in shell._control.toPlainText()
    qtbot.keyClick(code_editor, Qt.Key_Backspace)
    qtbot.keyClick(code_editor, Qt.Key_Backspace)
    qtbot.keyClick(code_editor, Qt.Key_Backspace)
    assert QApplication.clipboard().text() == 'def c():\n            print()\n'
    assert CLIPBOARD_HELPER.metadata_indent == 8
    qtbot.keyClick(code_editor, 'v', modifier=Qt.ControlModifier)
    assert '\ndef c():\n    print()' in code_editor.toPlainText()
    qtbot.keyClick(code_editor, 'z', modifier=Qt.ControlModifier)
    qtbot.keyClick(code_editor, Qt.Key_Tab)
    qtbot.keyClick(code_editor, 'v', modifier=Qt.ControlModifier)
    expected = '\n            def c():\n                print()\n'
    assert expected in code_editor.toPlainText()

@pytest.mark.skipif(not running_in_ci(), reason='Only works in CIs')
def test_add_external_plugins_to_dependencies(main_window, qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Test that we register external plugins in the main window.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    external_names = []
    for dep in DEPENDENCIES:
        name = getattr(dep, 'package_name', None)
        if name:
            external_names.append(name)
    assert 'spyder-boilerplate' in external_names

def test_locals_globals_var_debug(main_window, qtbot, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Test that the debugger can handle variables named globals and locals.'
    ipyconsole = main_window.ipyconsole
    shell = ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    code = 'globals = 10\ndef fun():\n    locals = 15\n    return\nfun()'
    p = tmpdir.join('test_gl.py')
    p.write(code)
    main_window.editor.load(to_text_string(p))
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debugfile ' + repr(str(p)))
    with qtbot.waitSignal(shell.executed):
        shell.execute('b 4')
    with qtbot.waitSignal(shell.executed):
        shell.execute('c')
    with qtbot.waitSignal(shell.executed):
        shell.execute('globals')
    assert 'Out  [3]: 10' in shell._control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        shell.execute('locals')
    assert 'Out  [4]: 15' in shell._control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        shell.execute('q')
    assert 'error' not in shell._control.toPlainText().lower()

@flaky(max_runs=3)
@pytest.mark.order(after='test_debug_unsaved_function')
def test_print_multiprocessing(main_window, qtbot, tmpdir):
    if False:
        i = 10
        return i + 15
    'Test print commands from multiprocessing.'
    code = '\nimport multiprocessing\nimport sys\ndef test_func():\n    print("Test stdout")\n    print("Test stderr", file=sys.stderr)\n\nif __name__ == "__main__":\n    p = multiprocessing.Process(target=test_func)\n    p.start()\n    p.join()\n'
    p = tmpdir.join('print-test.py')
    p.write(code)
    main_window.editor.load(to_text_string(p))
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    control = main_window.ipyconsole.get_widget().get_focus_widget()
    fname = main_window.editor.get_current_filename()
    run_parameters = generate_run_parameters(main_window, fname)
    CONF.set('run', 'last_used_parameters', run_parameters)
    main_window.editor.update_run_focus_file()
    qtbot.wait(2000)
    with qtbot.waitSignal(shell.executed, timeout=SHELL_TIMEOUT):
        qtbot.mouseClick(main_window.run_button, Qt.LeftButton)
    qtbot.wait(1000)
    assert 'Test stdout' in control.toPlainText()
    assert 'Test stderr' in control.toPlainText()

@flaky(max_runs=3)
@pytest.mark.skipif(os.name == 'nt', reason="ctypes.string_at(0) doesn't segfaults on Windows")
@pytest.mark.order(after='test_debug_unsaved_function')
def test_print_faulthandler(main_window, qtbot, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Test printing segfault info from kernel crashes.'
    code = '\ndef crash_func():\n    import ctypes; ctypes.string_at(0)\ncrash_func()\n'
    p = tmpdir.join('print-test.py')
    p.write(code)
    main_window.editor.load(to_text_string(p))
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    control = main_window.ipyconsole.get_widget().get_focus_widget()
    fname = main_window.editor.get_current_filename()
    run_parameters = generate_run_parameters(main_window, fname)
    CONF.set('run', 'last_used_parameters', run_parameters)
    main_window.editor.update_run_focus_file()
    qtbot.wait(2000)
    qtbot.mouseClick(main_window.run_button, Qt.LeftButton)
    qtbot.waitUntil(lambda : 'Segmentation fault' in control.toPlainText(), timeout=SHELL_TIMEOUT)
    assert 'Segmentation fault' in control.toPlainText()
    assert 'in crash_func' in control.toPlainText()

@flaky(max_runs=3)
@pytest.mark.skipif(os.name == 'nt', reason='Tour messes up focus on Windows')
def test_focus_for_plugins_with_raise_and_focus(main_window, qtbot):
    if False:
        print('Hello World!')
    '\n    Check that we give focus to the focus widget declared by plugins that use\n    the RAISE_AND_FOCUS class constant.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    control = main_window.ipyconsole.get_widget().get_focus_widget()
    console = main_window.get_plugin(Plugins.Console)
    console.toggle_view_action.setChecked(True)
    main_window.ipyconsole.dockwidget.raise_()
    focus_widget = QApplication.focusWidget()
    assert focus_widget is control
    console.dockwidget.raise_()
    focus_widget = QApplication.focusWidget()
    assert focus_widget is console.get_widget().get_focus_widget()
    find = main_window.get_plugin(Plugins.Find)
    find.toggle_view_action.setChecked(True)
    focus_widget = QApplication.focusWidget()
    assert focus_widget is find.get_widget().get_focus_widget()

@flaky(max_runs=3)
@pytest.mark.order(1)
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Hangs sometimes on Windows and Mac')
def test_rename_files_in_editor_after_folder_rename(main_window, mocker, tmpdir, qtbot):
    if False:
        print('Hello World!')
    '\n    Check that we rename files in the editor after the directory that\n    contains them was renamed in Files.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    old_path = 'test_rename_old'
    new_path = 'test_rename_new'
    fname = 'foo.py'
    mocker.patch.object(QInputDialog, 'getText', return_value=(new_path, True))
    file = tmpdir.mkdir(old_path).join(fname)
    file.write("print('Hello world!')")
    editor = main_window.get_plugin(Plugins.Editor)
    editor.load(str(file))
    explorer = main_window.get_plugin(Plugins.Explorer)
    explorer.chdir(str(tmpdir))
    explorer.switch_to_plugin()
    explorer.get_widget().get_focus_widget().setFocus()
    treewidget = explorer.get_widget().treewidget
    idx = treewidget.get_index(old_path)
    treewidget.setCurrentIndex(idx)
    treewidget.rename()
    codeeditor = editor.get_current_editor()
    assert codeeditor.filename == osp.join(str(tmpdir), new_path, fname)

@flaky(max_runs=3)
def test_history_from_ipyconsole(main_window, qtbot):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check that we register commands introduced in the IPython console in\n    the History pane.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    code = '5 + 3'
    with qtbot.waitSignal(shell.executed):
        shell.execute(code)
    history = main_window.get_plugin(Plugins.History)
    history.switch_to_plugin()
    history_editor = history.get_widget().editors[0]
    text = history_editor.toPlainText()
    assert text.splitlines()[-1] == code

def test_debug_unsaved_function(main_window, qtbot):
    if False:
        print('Hello World!')
    '\n    Test that a breakpoint in an unsaved file is reached.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    shell = main_window.ipyconsole.get_current_shellwidget()
    control = shell._control
    main_window.debugger.clear_all_breakpoints()
    main_window.editor.new()
    code_editor = main_window.editor.get_focus_widget()
    code_editor.set_text('def foo():\n    print(1)')
    fname = main_window.editor.get_current_filename()
    run_parameters = generate_run_parameters(main_window, fname)
    CONF.set('run', 'last_used_parameters', run_parameters)
    main_window.editor.update_run_focus_file()
    qtbot.wait(2000)
    code_editor.breakpoints_manager.toogle_breakpoint(line_number=2)
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(main_window.run_button, Qt.LeftButton)
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug foo()')
    with qtbot.waitSignal(shell.executed):
        shell.execute('continue')
    assert '1---> 2     print(1)' in control.toPlainText()

@flaky(max_runs=5)
@pytest.mark.close_main_window
@pytest.mark.order(after='test_debug_unsaved_function')
def test_out_runfile_runcell(main_window, qtbot):
    if False:
        i = 10
        return i + 15
    '\n    Test that runcell and runfile return values if last statment\n    is expression.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    control = main_window.ipyconsole.get_widget().get_focus_widget()
    codes = {'a = 1 + 1; a': (2, True), 'a = 1 + 3; a;': (4, False), 'a = 1 + 5\na': (6, True), 'a = 1 + 7\na;': (8, False)}
    for code in codes:
        (num, shown) = codes[code]
        main_window.editor.new()
        code_editor = main_window.editor.get_focus_widget()
        code_editor.set_text(code)
        fname = main_window.editor.get_current_filename()
        run_parameters = generate_run_parameters(main_window, fname)
        CONF.set('run', 'last_used_parameters', run_parameters)
        with qtbot.waitSignal(shell.executed):
            qtbot.mouseClick(main_window.run_cell_button, Qt.LeftButton)
        if shown:
            assert ']: ' + str(num) in control.toPlainText()
        else:
            assert not ']: ' + str(num) in control.toPlainText()

@flaky(max_runs=3)
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Does not work on Mac and Windows')
@pytest.mark.parametrize('thread', [False, True])
@pytest.mark.order(after='test_debug_unsaved_function')
def test_print_frames(main_window, qtbot, tmpdir, thread):
    if False:
        return 10
    'Test that frames are displayed as expected.'
    if thread:
        code = 'import threading\ndef deadlock():\n    lock = threading.Lock()\n    lock.acquire()\n    lock.acquire()\nt = threading.Thread(target=deadlock)\nt.start()\nt.join()\n'
        expected_number_threads = 2
    else:
        code = 'import threading\nlock = threading.Lock()\nlock.acquire()\nlock.acquire()'
        expected_number_threads = 1
    p = tmpdir.join('print-test.py')
    p.write(code)
    main_window.editor.load(to_text_string(p))
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    debugger = main_window.debugger.get_widget()
    frames_browser = debugger.current_widget().results_browser
    run_parameters = generate_run_parameters(main_window, str(p))
    CONF.set('run', 'last_used_parameters', run_parameters)
    qtbot.mouseClick(main_window.run_button, Qt.LeftButton)
    qtbot.wait(1000)
    control = main_window.ipyconsole.get_widget().get_focus_widget()
    assert ']:' not in control.toPlainText().split()[-1]
    debugger.capture_frames()
    qtbot.wait(1000)
    qtbot.waitUntil(lambda : len(frames_browser.data) > 0, timeout=10000)
    if len(frames_browser.frames) != expected_number_threads:
        import pprint
        pprint.pprint(frames_browser.frames)
    assert len(frames_browser.frames) == expected_number_threads

@flaky(max_runs=3)
def test_debugger_plugin(main_window, qtbot):
    if False:
        print('Hello World!')
    'Test debugger plugin.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    debugger = main_window.debugger.get_widget()
    frames_browser = debugger.current_widget().results_browser
    enter_debug_action = debugger.get_action(DebuggerWidgetActions.EnterDebug)
    assert not enter_debug_action.isEnabled()
    with qtbot.waitSignal(shell.executed):
        shell.execute('1/0')
    assert len(frames_browser.frames) == 1
    assert list(frames_browser.frames.keys())[0] == 'ZeroDivisionError'
    assert enter_debug_action.isEnabled()
    with qtbot.waitSignal(shell.executed):
        debugger.enter_debug()
    assert len(frames_browser.frames) == 1
    assert list(frames_browser.frames.keys())[0] == 'pdb'
    assert not enter_debug_action.isEnabled()
    with qtbot.waitSignal(shell.executed):
        shell.execute('a = 1')
    assert len(frames_browser.frames) == 1
    assert list(frames_browser.frames.keys())[0] == 'pdb'
    assert not enter_debug_action.isEnabled()
    with qtbot.waitSignal(shell.executed):
        shell.execute('w')
    assert len(frames_browser.frames) == 1
    assert list(frames_browser.frames.keys())[0] == 'pdb'
    assert not enter_debug_action.isEnabled()
    with qtbot.waitSignal(shell.executed):
        shell.execute('q')
    assert frames_browser.frames is None
    assert not enter_debug_action.isEnabled()
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug print()')
    assert len(frames_browser.frames) == 1
    assert list(frames_browser.frames.keys())[0] == 'pdb'
    assert not enter_debug_action.isEnabled()
    widget = main_window.ipyconsole.get_widget()
    with qtbot.waitSignal(shell.sig_prompt_ready, timeout=10000):
        widget.restart_kernel(shell.ipyclient, False)
    assert frames_browser.frames is None
    assert not enter_debug_action.isEnabled()
    if os.name == 'nt':
        return
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug print()')
    assert len(frames_browser.frames) == 1
    assert list(frames_browser.frames.keys())[0] == 'pdb'
    assert not enter_debug_action.isEnabled()
    with qtbot.waitSignal(shell.sig_prompt_ready, timeout=10000):
        shell.execute('import ctypes; ctypes.string_at(0)')
    assert frames_browser.frames is None
    assert not enter_debug_action.isEnabled()

@flaky(max_runs=3)
def test_enter_debugger(main_window, qtbot):
    if False:
        print('Hello World!')
    '\n    Test that we can enter the debugger while code is running in the kernel.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    debugger = main_window.debugger.get_widget()
    enter_debug_action = debugger.get_action(DebuggerWidgetActions.EnterDebug)
    inspect_action = debugger.get_action(DebuggerWidgetActions.Inspect)
    with qtbot.waitSignal(shell.executed):
        shell.execute('import time')
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug for i in range(100): time.sleep(.1)')
    assert not enter_debug_action.isEnabled()
    assert not inspect_action.isEnabled()
    shell.execute('c')
    qtbot.wait(200)
    assert enter_debug_action.isEnabled()
    assert inspect_action.isEnabled()
    with qtbot.waitSignal(shell.executed):
        debugger.enter_debug()
    assert not enter_debug_action.isEnabled()
    assert not inspect_action.isEnabled()
    assert shell.is_debugging()
    assert 0 < shell.get_value('i') < 99
    with qtbot.waitSignal(shell.executed):
        shell.execute('q')
    assert not shell.is_debugging()
    if os.name == 'nt':
        return
    assert not enter_debug_action.isEnabled()
    assert not inspect_action.isEnabled()
    shell.execute('for i in range(100): time.sleep(.1)')
    qtbot.wait(200)
    assert enter_debug_action.isEnabled()
    assert inspect_action.isEnabled()
    with qtbot.waitSignal(shell.executed):
        debugger.enter_debug()
    assert shell.is_debugging()
    assert not enter_debug_action.isEnabled()
    assert not inspect_action.isEnabled()
    assert 0 < shell.get_value('i') < 99
    shell.execute('c')
    qtbot.wait(200)
    with qtbot.waitSignal(shell.executed):
        debugger.enter_debug()
    assert not enter_debug_action.isEnabled()
    assert not inspect_action.isEnabled()
    assert 0 < shell.get_value('i') < 99
    with qtbot.waitSignal(shell.executed):
        shell.execute('q')

@flaky(max_runs=3)
def test_recursive_debug(main_window, qtbot):
    if False:
        i = 10
        return i + 15
    'Test recurside debug.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    debugger = main_window.debugger.get_widget()
    frames_browser = debugger.current_widget().results_browser
    with qtbot.waitSignal(shell.executed):
        shell.execute('def a():\n    return\ndef b():\n    return')
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug a()')
    with qtbot.waitSignal(shell.executed):
        shell.execute('s')
    assert frames_browser.frames['pdb'][2].name == 'a'
    with qtbot.waitSignal(shell.executed):
        shell.execute('debug b()')
    with qtbot.waitSignal(shell.executed):
        shell.execute('s')
    assert frames_browser.frames['pdb'][2].name == 'b'
    with qtbot.waitSignal(shell.executed):
        shell.execute('q')
    assert frames_browser.frames['pdb'][2].name == 'a'
    with qtbot.waitSignal(shell.executed):
        shell.execute('q')

@flaky(max_runs=3)
@pytest.mark.skipif(os.name == 'nt', reason='SIGINT is not processed correctly on CI for Windows')
def test_interrupt(main_window, qtbot):
    if False:
        while True:
            i = 10
    'Test interrupt.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    debugger = main_window.debugger.get_widget()
    frames_browser = debugger.current_widget().results_browser
    with qtbot.waitSignal(shell.executed):
        shell.execute('import time')
    shell.execute('for i in range(100): time.sleep(.1)')
    qtbot.wait(200)
    with qtbot.waitSignal(shell.executed):
        shell.call_kernel(interrupt=True).raise_interrupt_signal()
    assert 0 < shell.get_value('i') < 99
    assert list(frames_browser.frames.keys())[0] == 'KeyboardInterrupt'
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug for i in range(100): time.sleep(.1)')
    shell.execute('c')
    qtbot.wait(200)
    with qtbot.waitSignal(shell.executed):
        shell.call_kernel(interrupt=True).raise_interrupt_signal()
    assert 'Program interrupted' in shell._control.toPlainText()
    assert 0 < shell.get_value('i') < 99
    with qtbot.waitSignal(shell.executed):
        shell.execute('q')
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug time.sleep(20)')
    shell.execute('c')
    qtbot.wait(100)
    shell.call_kernel(interrupt=True).request_pdb_stop()
    qtbot.wait(100)
    t0 = time.time()
    with qtbot.waitSignal(shell.executed):
        shell.interrupt_kernel()
    assert time.time() - t0 < 10
    with qtbot.waitSignal(shell.executed):
        shell.execute('q')
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug time.sleep(20)')
    shell.execute('c')
    qtbot.wait(100)
    shell.call_kernel(interrupt=True).request_pdb_stop()
    qtbot.wait(100)
    t0 = time.time()
    with qtbot.waitSignal(shell.executed):
        shell.call_kernel(interrupt=True).raise_interrupt_signal()
    assert time.time() - t0 < 10
    with qtbot.waitSignal(shell.executed):
        shell.execute('q')

def test_visible_plugins(main_window, qtbot):
    if False:
        while True:
            i = 10
    '\n    Test that saving and restoring visible plugins works as expected.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    main_window.layouts.quick_layout_switch(DefaultLayouts.SpyderLayout)
    selected = [Plugins.Plots, Plugins.History]
    for plugin_name in selected:
        main_window.get_plugin(plugin_name).dockwidget.raise_()
    main_window.layouts.save_visible_plugins()
    for plugin_name in [Plugins.VariableExplorer, Plugins.IPythonConsole]:
        main_window.get_plugin(plugin_name).dockwidget.raise_()
    for plugin_name in selected:
        assert not main_window.get_plugin(plugin_name).get_widget().is_visible
    main_window.layouts.restore_visible_plugins()
    visible_plugins = []
    for (plugin_name, plugin) in main_window.get_dockable_plugins():
        if plugin_name != Plugins.Editor and plugin.get_widget().is_visible:
            visible_plugins.append(plugin_name)
    assert set(selected) == set(visible_plugins)

def test_cwd_is_synced_when_switching_consoles(main_window, qtbot, tmpdir):
    if False:
        return 10
    '\n    Test that the current working directory is synced between the IPython\n    console and other plugins when switching consoles.\n    '
    ipyconsole = main_window.ipyconsole
    workdir = main_window.workingdirectory
    files = main_window.get_plugin(Plugins.Explorer)
    shell = ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    for i in range(2):
        sync_dir = tmpdir.mkdir(f'test_sync_{i}')
        ipyconsole.create_new_client()
        shell = ipyconsole.get_current_shellwidget()
        qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
        with qtbot.waitSignal(shell.executed):
            shell.execute(f'cd {str(sync_dir)}')
    for i in range(3):
        ipyconsole.get_widget().tabwidget.setCurrentIndex(i)
        shell_cwd = ipyconsole.get_current_shellwidget().get_cwd()
        assert shell_cwd == workdir.get_workdir() == files.get_current_folder()

@flaky(max_runs=5)
def test_console_initial_cwd_is_synced(main_window, qtbot, tmpdir):
    if False:
        i = 10
        return i + 15
    '\n    Test that the initial current working directory for new consoles is synced\n    with other plugins.\n    '
    ipyconsole = main_window.ipyconsole
    workdir = main_window.workingdirectory
    files = main_window.get_plugin(Plugins.Explorer)
    shell = ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    files.get_widget().treewidget.open_interpreter([str(tmpdir)])
    shell = ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    qtbot.waitUntil(lambda : shell.get_cwd() == str(tmpdir))
    assert shell.get_cwd() == str(tmpdir) == workdir.get_workdir() == files.get_current_folder()
    ipyconsole.create_new_client()
    shell = ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    qtbot.waitUntil(lambda : shell.get_cwd() == str(tmpdir))
    assert shell.get_cwd() == str(tmpdir) == workdir.get_workdir() == files.get_current_folder()
    ipyconsole.set_conf('console/use_cwd', False, section='workingdir')
    ipyconsole.set_conf('console/use_fixed_directory', True, section='workingdir')
    fixed_dir = str(tmpdir.mkdir('fixed_dir'))
    ipyconsole.set_conf('console/fixed_directory', fixed_dir, section='workingdir')
    ipyconsole.create_new_client()
    shell = ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    qtbot.waitUntil(lambda : shell.get_cwd() == fixed_dir)
    assert shell.get_cwd() == fixed_dir == workdir.get_workdir() == files.get_current_folder()
    project_path = str(tmpdir.mkdir('test_project'))
    main_window.projects.open_project(path=project_path)
    qtbot.wait(500)
    shell = ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    qtbot.waitUntil(lambda : shell.get_cwd() == project_path)
    assert shell.get_cwd() == project_path == workdir.get_workdir() == files.get_current_folder()
    main_window.projects.close_project()
    qtbot.wait(500)
    shell = ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    qtbot.waitUntil(lambda : shell.get_cwd() == get_home_dir())
    assert shell.get_cwd() == get_home_dir() == workdir.get_workdir() == files.get_current_folder()

def test_debug_selection(main_window, qtbot):
    if False:
        print('Hello World!')
    'test debug selection.'
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    shell = main_window.ipyconsole.get_current_shellwidget()
    control = shell._control
    debug_widget = main_window.debugger.get_widget()
    debug_selection_action = main_window.run.get_action('run selection in debugger')
    continue_action = debug_widget.get_action(DebuggerWidgetActions.Continue)
    main_window.editor.new()
    code_editor = main_window.editor.get_focus_widget()
    code = 'print(1 + 2)\nprint(2 + 4)'
    code_editor.set_text(code)
    with qtbot.waitSignal(shell.executed):
        debug_selection_action.trigger()
    assert shell.is_debugging()
    assert 'print(1 + 2)' in control.toPlainText()
    assert '%%debug' in control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        continue_action.trigger()
    assert not shell.is_debugging()
    with qtbot.waitSignal(shell.executed):
        shell.execute('%clear')
    assert 'print(1 + 2)' not in control.toPlainText()
    cursor = code_editor.textCursor()
    cursor.movePosition(QTextCursor.End, QTextCursor.MoveAnchor)
    cursor.movePosition(QTextCursor.Start, QTextCursor.KeepAnchor)
    code_editor.setTextCursor(cursor)
    with qtbot.waitSignal(shell.executed):
        debug_selection_action.trigger()
    assert shell.is_debugging()
    with qtbot.waitSignal(shell.executed):
        continue_action.trigger()
    assert not shell.is_debugging()
    assert 'print(1 + 2)' in control.toPlainText()
    assert 'print(2 + 4)' in control.toPlainText()
    assert '%%debug' in control.toPlainText()

@flaky(max_runs=3)
@pytest.mark.use_introspection
@pytest.mark.order(after='test_debug_unsaved_function')
@pytest.mark.preload_namespace_project
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Only works on Linux')
@pytest.mark.known_leak
def test_outline_namespace_package(main_window, qtbot, tmpdir):
    if False:
        i = 10
        return i + 15
    '\n    Test that we show symbols in the Outline pane for projects that have\n    namespace packages, i.e. with no __init__.py file in its root directory.\n\n    This is a regression test for issue spyder-ide/spyder#16406.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    outline_explorer = main_window.outlineexplorer
    outline_explorer.toggle_view_action.setChecked(True)
    treewidget = outline_explorer.get_widget().treewidget
    editors_py = [editor for editor in treewidget.editor_ids.keys() if editor.get_language() == 'Python']

    def editors_filled():
        if False:
            print('Hello World!')
        return all([len(treewidget.editor_tree_cache[editor.get_id()]) == 4 for editor in editors_py])
    qtbot.waitUntil(editors_filled, timeout=25000)
    assert editors_filled()
    CONF.set('editor', 'filenames', [])

@pytest.mark.skipif(sys.platform == 'darwin', reason='Only works on Windows and Linux')
@pytest.mark.order(before='test_tour_message')
def test_switch_to_plugin(main_window, qtbot):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that switching between the two most important plugins, the Editor and\n    the IPython console, is working as expected.\n\n    This is a regression test for issue spyder-ide/spyder#19374.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    qtbot.keyClick(main_window, Qt.Key_I, modifier=Qt.ControlModifier | Qt.ShiftModifier)
    control = main_window.ipyconsole.get_widget().get_focus_widget()
    assert QApplication.focusWidget() is control
    qtbot.keyClick(main_window, Qt.Key_E, modifier=Qt.ControlModifier | Qt.ShiftModifier)
    code_editor = main_window.editor.get_current_editor()
    assert QApplication.focusWidget() is code_editor

@flaky(max_runs=5)
def test_PYTHONPATH_in_consoles(main_window, qtbot, tmp_path, restore_user_env):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that PYTHONPATH is passed to IPython consoles under different\n    scenarios.\n    '
    ipyconsole = main_window.ipyconsole
    shell = ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    new_dir = tmp_path / 'new_dir'
    new_dir.mkdir()
    set_user_env({'PYTHONPATH': str(new_dir)})
    ppm = main_window.get_plugin(Plugins.PythonpathManager)
    ppm.show_path_manager()
    qtbot.wait(500)
    ppm.path_manager_dialog.close()
    with qtbot.waitSignal(shell.executed, timeout=2000):
        shell.execute('import sys; sys_path = sys.path')
    assert str(new_dir) in shell.get_value('sys_path')
    ipyconsole.create_new_client()
    shell = ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed, timeout=2000):
        shell.execute('import sys; sys_path = sys.path')
    assert str(new_dir) in shell.get_value('sys_path')

@flaky(max_runs=10)
@pytest.mark.skipif(sys.platform == 'darwin', reason='Fails on Mac')
def test_clickable_ipython_tracebacks(main_window, qtbot, tmp_path):
    if False:
        print('Hello World!')
    '\n    Test that file names in IPython console tracebacks are clickable.\n\n    This is a regression test for issue spyder-ide/spyder#20407.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    control = shell._control
    qtbot.waitUntil(lambda : shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    test_file_orig = osp.join(LOCATION, 'script.py')
    test_file = str(tmp_path / 'script.py')
    shutil.copyfile(test_file_orig, test_file)
    main_window.editor.load(test_file)
    code_editor = main_window.editor.get_focus_widget()
    text = code_editor.toPlainText()
    assert text.splitlines(keepends=True)[-1].endswith('\n')
    cursor = code_editor.textCursor()
    cursor.movePosition(QTextCursor.End, QTextCursor.MoveAnchor)
    code_editor.setTextCursor(cursor)
    qtbot.keyClicks(code_editor, '1/0')
    run_parameters = generate_run_parameters(main_window, test_file)
    CONF.set('run', 'last_used_parameters', run_parameters)
    qtbot.mouseClick(main_window.run_button, Qt.LeftButton)
    qtbot.wait(500)
    control.setFocus()
    find_widget = main_window.ipyconsole.get_widget().find_widget
    find_widget.show()
    find_widget.search_text.lineEdit().setText('  File')
    find_widget.find_previous()
    cursor_point = control.cursorRect(control.textCursor()).center()
    qtbot.mouseMove(control, cursor_point)
    qtbot.wait(500)
    assert QApplication.overrideCursor().shape() == Qt.PointingHandCursor
    qtbot.mouseClick(control.viewport(), Qt.LeftButton, pos=cursor_point, delay=300)
    assert QApplication.focusWidget() is code_editor
    cursor = code_editor.textCursor()
    assert cursor.blockNumber() == code_editor.blockCount() - 1

def test_recursive_debug_exception(main_window, qtbot):
    if False:
        print('Hello World!')
    '\n    Test that an exception in a recursive debug does not break the debugger.\n    '
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    shell = main_window.ipyconsole.get_current_shellwidget()
    control = shell._control
    main_window.editor.new()
    code_editor = main_window.editor.get_focus_widget()
    code = 'print("res", 1 + 2)\nprint("res", 2 + 4)'
    code_editor.set_text(code)
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debugfile ' + remove_backslashes(str(main_window.editor.get_current_filename())))
    assert shell.is_debugging()
    assert '----> 1 print("res", 1 + 2)' in control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        shell.execute('debug 1/0')
    assert 'Entering recursive debugger' in control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        shell.execute('c')
    assert 'ZeroDivisionError' in control.toPlainText()
    assert 'Leaving recursive debugger' in control.toPlainText()
    assert 'IPdb [2]:' in control.toPlainText()
    assert shell.is_debugging()
    with qtbot.waitSignal(shell.executed):
        shell.execute('n')
    assert 'res 3' in control.toPlainText()
    assert shell.is_debugging()
    with qtbot.waitSignal(shell.executed):
        shell.execute('q')
    assert not shell.is_debugging()

@flaky(max_runs=3)
def test_runfile_namespace(main_window, qtbot, tmpdir):
    if False:
        print('Hello World!')
    'Test that namespaces behave correctly when using runfile.'
    baba_file = tmpdir.join('baba.py')
    baba_file.write('baba = 1')
    baba_path = to_text_string(baba_file)
    code = '\n'.join(['def fun():', '    %runfile {}'.format(repr(baba_path)), '    print("test_locals", "baba" in locals(), "baba" in globals())', 'fun()', 'def fun():', '    ns = {}', '    %runfile {} --namespace ns'.format(repr(baba_path)), '    print("test_locals_namespace", "baba" in ns, "baba" in locals(), "baba" in globals())', 'fun()', 'ns = {}', '%runfile {} --namespace ns'.format(repr(baba_path)), 'print("test_globals_namespace", "baba" in ns, "baba" in globals())', '%runfile {}'.format(repr(baba_path)), 'print("test_globals", "baba" in globals())'])
    p = tmpdir.join('test.ipy')
    p.write(code)
    test_file = to_text_string(p)
    shell = main_window.ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.sig_prompt_ready):
        shell.execute('%runfile {}'.format(repr(test_file)))
    control = shell._control
    assert 'test_locals True False' in control.toPlainText()
    assert 'test_locals_namespace True False False' in control.toPlainText()
    assert 'test_globals_namespace True False' in control.toPlainText()
    assert 'test_globals True' in control.toPlainText()

@pytest.mark.skipif(os.name == 'nt', reason='No quotes on Windows file paths')
def test_quotes_rename_ipy(main_window, qtbot, tmpdir):
    if False:
        print('Hello World!')
    '\n    Test that we can run files with quotes in name, renamed files,\n    and ipy files.\n    '
    path = 'a\'b"c\\.py'
    file = tmpdir.join(path)
    file.write('print(23 + 780)')
    path = to_text_string(file)
    main_window.editor.load(path)
    shell = main_window.ipyconsole.get_current_shellwidget()
    control = shell._control
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(main_window.run_button, Qt.LeftButton)
    assert '803' in control.toPlainText()
    assert 'error' not in control.toPlainText()
    code_editor = main_window.editor.get_focus_widget()
    code_editor.set_text('print(22 + 780)')
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(main_window.run_cell_button, Qt.LeftButton)
    assert '802' in control.toPlainText()
    assert 'error' not in control.toPlainText()
    rename_file(path, path[:-2] + 'ipy')
    explorer = main_window.get_plugin(Plugins.Explorer)
    explorer.sig_file_renamed.emit(path, path[:-2] + 'ipy')
    code_editor.set_text('print(21 + 780)')
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(main_window.run_button, Qt.LeftButton)
    assert '801' in control.toPlainText()
    assert 'error' not in control.toPlainText()
    assert '\\.ipy' in control.toPlainText()
    main_window.editor.new()
    assert 'untitled' in main_window.editor.get_current_filename()
    code_editor = main_window.editor.get_focus_widget()
    code_editor.set_text('print(20 + 780)')
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(main_window.run_cell_button, Qt.LeftButton)
    assert '800' in control.toPlainText()
    assert 'error' not in control.toPlainText()
    assert 'untitled' in control.toPlainText()
    code_editor.set_text('print(19 + 780)')
    with tempfile.TemporaryDirectory() as td:
        editorstack = main_window.editor.get_current_editorstack()
        editorstack.select_savename = lambda fn: os.path.join(td, 'fn.ipy')
        main_window.editor.save()
        with qtbot.waitSignal(shell.executed):
            qtbot.mouseClick(main_window.run_cell_button, Qt.LeftButton)
        assert '799' in control.toPlainText()
        assert 'error' not in control.toPlainText()
        assert 'fn.ipy' in control.toPlainText()
        main_window.editor.close_file()
if __name__ == '__main__':
    pytest.main()