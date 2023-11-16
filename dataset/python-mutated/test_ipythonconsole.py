"""
Tests for the IPython console plugin.
"""
import os
import os.path as osp
import shutil
import sys
from textwrap import dedent
from ipykernel._version import __version__ as ipykernel_version
import IPython
from IPython.core import release as ipy_release
from IPython.core.application import get_ipython_dir
from flaky import flaky
import numpy as np
from packaging.version import parse
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWebEngineWidgets import WEBENGINE
from spyder_kernels import __version__ as spyder_kernels_version
import sympy
from spyder.config.base import running_in_ci, running_in_ci_with_conda
from spyder.config.gui import get_color_scheme
from spyder.config.utils import is_anaconda
from spyder.py3compat import to_text_string
from spyder.plugins.help.tests.test_plugin import check_text
from spyder.plugins.ipythonconsole.tests.conftest import get_conda_test_env, get_console_background_color, get_console_font_color, NEW_DIR, SHELL_TIMEOUT, TEMP_DIRECTORY
from spyder.plugins.ipythonconsole.utils.kernel_handler import KernelConnectionState
from spyder.plugins.ipythonconsole.widgets import ShellWidget
from spyder.utils.conda import get_list_conda_envs

@flaky(max_runs=3)
@pytest.mark.external_interpreter
def test_banners(ipyconsole, qtbot):
    if False:
        print('Hello World!')
    'Test that console banners are generated correctly.'
    shell = ipyconsole.get_current_shellwidget()
    control = shell._control
    text = control.toPlainText().splitlines()
    if 'Update LANGUAGE_CODES' in text[0]:
        text = text[1:]
        while not text[0].strip():
            text = text[1:]
    py_ver = sys.version.splitlines()[0].strip()
    assert py_ver in text[0]
    assert 'license' in text[1]
    assert '' == text[2]
    assert ipy_release.version in text[3]
    short_banner = shell.short_banner()
    py_ver = sys.version.split(' ')[0]
    expected = 'Python %s -- IPython %s' % (py_ver, ipy_release.version)
    assert expected == short_banner

@flaky(max_runs=3)
@pytest.mark.parametrize('function, signature, documentation', [('np.arange', ['start', 'stop'], ['Return evenly spaced values within a given interval.<br>', 'open interval ...']), ('np.vectorize', ['pyfunc', 'otype', 'signature'], ['Returns an object that acts like pyfunc, but takes arrays as<br>input.<br>', 'Define a vectorized function which takes a nested sequence ...']), ('np.abs', ['x', '/', 'out'], ['Calculate the absolute value']), ('np.where', ['condition', '/'], ['Return elements chosen from `x`']), ('np.array', ['object', 'dtype=None'], ['Create an array.<br><br>', 'Parameters']), ('np.linalg.norm', ['x', 'ord=None'], ['Matrix or vector norm']), ('range', ['stop'], ['range(stop) -> range object']), ('dict', ['mapping'], ['dict() -> new empty dictionary']), ('foo', ['x', 'y'], ['My function'])])
@pytest.mark.skipif(running_in_ci() and (not os.name == 'nt'), reason='Times out on macOS and fails on Linux')
@pytest.mark.skipif(parse(np.__version__) < parse('1.25.0'), reason='Documentation for np.vectorize is different')
def test_get_calltips(ipyconsole, qtbot, function, signature, documentation):
    if False:
        i = 10
        return i + 15
    'Test that calltips show the documentation.'
    shell = ipyconsole.get_current_shellwidget()
    control = shell._control
    with qtbot.waitSignal(shell.executed):
        shell.execute('import numpy as np')
    if function == 'foo':
        with qtbot.waitSignal(shell.executed):
            code = dedent('\n            def foo(x, y):\n                """\n                My function\n                """\n                return x + y\n            ')
            shell.execute(code)
    with qtbot.waitSignal(shell.kernel_client.shell_channel.message_received):
        qtbot.keyClicks(control, function + '(')
    qtbot.waitUntil(lambda : control.calltip_widget.isVisible())
    assert control.calltip_widget.isVisible()
    control.calltip_widget.hide()
    for element in signature:
        assert element in control.calltip_widget.text()
    for element in documentation:
        assert element in control.calltip_widget.text()

@flaky(max_runs=3)
@pytest.mark.auto_backend
def test_auto_backend(ipyconsole, qtbot):
    if False:
        print('Hello World!')
    'Test that the automatic backend was set correctly.'
    shell = ipyconsole.get_current_shellwidget()
    with qtbot.waitSignal(shell.executed):
        shell.execute('get_ipython().kernel.eventloop')
    control = ipyconsole.get_widget().get_focus_widget()
    assert 'NOTE' not in control.toPlainText()
    assert 'Error' not in control.toPlainText()
    assert 'loop_qt5' in control.toPlainText() or 'loop_qt' in control.toPlainText()

@flaky(max_runs=3)
@pytest.mark.tk_backend
@pytest.mark.skipif(os.name == 'nt' and parse(ipykernel_version) == parse('6.21.0'), reason='Fails on Windows with IPykernel 6.21.0')
def test_tk_backend(ipyconsole, qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Test that the Tkinter backend was set correctly.'
    shell = ipyconsole.get_current_shellwidget()
    with qtbot.waitSignal(shell.executed):
        shell.execute('get_ipython().kernel.eventloop')
    control = ipyconsole.get_widget().get_focus_widget()
    assert 'loop_tk' in control.toPlainText()

@flaky(max_runs=3)
@pytest.mark.pylab_client
def test_pylab_client(ipyconsole, qtbot):
    if False:
        while True:
            i = 10
    'Test that the Pylab console is working correctly.'
    shell = ipyconsole.get_current_shellwidget()
    with qtbot.waitSignal(shell.executed):
        shell.execute('e')
    control = ipyconsole.get_widget().get_focus_widget()
    assert 'Error' not in control.toPlainText()
    shell.reset_namespace()
    qtbot.wait(1000)
    with qtbot.waitSignal(shell.executed):
        shell.execute('e')
    control = ipyconsole.get_widget().get_focus_widget()
    assert 'Error' not in control.toPlainText()

@flaky(max_runs=3)
@pytest.mark.sympy_client
@pytest.mark.xfail(parse('1.0') < parse(sympy.__version__) < parse('1.2'), reason='A bug with sympy 1.1.1 and IPython-Qtconsole')
def test_sympy_client(ipyconsole, qtbot):
    if False:
        return 10
    'Test that the SymPy console is working correctly.'
    shell = ipyconsole.get_current_shellwidget()
    with qtbot.waitSignal(shell.executed):
        shell.execute('x')
    control = ipyconsole.get_widget().get_focus_widget()
    assert 'NameError' not in control.toPlainText()
    shell.reset_namespace()
    qtbot.wait(1000)
    with qtbot.waitSignal(shell.executed):
        shell.execute('x')
    control = ipyconsole.get_widget().get_focus_widget()
    assert 'NameError' not in control.toPlainText()

@flaky(max_runs=3)
@pytest.mark.cython_client
@pytest.mark.skipif(not sys.platform.startswith('linux') or parse(ipy_release.version) == parse('7.11.0'), reason='It only works reliably on Linux and fails for IPython 7.11.0')
def test_cython_client(ipyconsole, qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Test that the Cython console is working correctly.'
    shell = ipyconsole.get_current_shellwidget()
    with qtbot.waitSignal(shell.executed, timeout=SHELL_TIMEOUT):
        shell.execute('%%cython\ncdef int ctest(int x, int y):\n    return x + y')
    control = ipyconsole.get_widget().get_focus_widget()
    assert 'Error' not in control.toPlainText()
    shell.reset_namespace()
    qtbot.wait(1000)
    with qtbot.waitSignal(shell.executed, timeout=SHELL_TIMEOUT):
        shell.execute('%%cython\ncdef int ctest(int x, int y):\n    return x + y')
    control = ipyconsole.get_widget().get_focus_widget()
    assert 'Error' not in control.toPlainText()

@flaky(max_runs=3)
@pytest.mark.order(1)
@pytest.mark.environment_client
@pytest.mark.skipif(not is_anaconda(), reason='Only works with Anaconda')
@pytest.mark.skipif(not running_in_ci(), reason='Only works on CIs')
@pytest.mark.skipif(not os.name == 'nt', reason='Works reliably on Windows')
def test_environment_client(ipyconsole, qtbot):
    if False:
        while True:
            i = 10
    '\n    Test that when creating a console for a specific conda environment, the\n    environment is activated before a kernel is created for it.\n    '
    shell = ipyconsole.get_current_shellwidget()
    client = ipyconsole.get_current_client()
    client.get_name() == 'spytest-ž 1/A'
    with qtbot.waitSignal(shell.executed):
        shell.execute("import os; conda_prefix = os.environ.get('CONDA_PREFIX')")
    expected_output = get_conda_test_env()[0].replace('\\', '/')
    output = shell.get_value('conda_prefix').replace('\\', '/')
    assert expected_output == output

@flaky(max_runs=3)
def test_tab_rename_for_slaves(ipyconsole, qtbot):
    if False:
        i = 10
        return i + 15
    'Test slave clients are renamed correctly.'
    cf = ipyconsole.get_current_client().connection_file
    ipyconsole.create_client_for_kernel(cf)
    qtbot.waitUntil(lambda : len(ipyconsole.get_clients()) == 2)
    ipyconsole.get_widget().rename_tabs_after_change('foo')
    assert 'foo' in ipyconsole.get_clients()[0].get_name()
    assert 'foo' in ipyconsole.get_clients()[1].get_name()

@flaky(max_runs=3)
def test_no_repeated_tabs_name(ipyconsole, qtbot):
    if False:
        while True:
            i = 10
    "Test that tabs can't have repeated given names."
    ipyconsole.get_widget().rename_tabs_after_change('foo')
    ipyconsole.create_new_client()
    ipyconsole.get_widget().rename_tabs_after_change('foo')
    client_name = ipyconsole.get_current_client().get_name()
    assert '2' in client_name

@flaky(max_runs=3)
@pytest.mark.skipif(running_in_ci() and sys.platform == 'darwin', reason='Hangs sometimes on macOS')
@pytest.mark.skipif(os.name == 'nt' and running_in_ci_with_conda(), reason='It hangs on Windows CI using conda')
def test_tabs_preserve_name_after_move(ipyconsole, qtbot):
    if False:
        print('Hello World!')
    'Test that tabs preserve their names after they are moved.'
    ipyconsole.create_new_client()
    ipyconsole.get_widget().tabwidget.tabBar().moveTab(0, 1)
    client_name = ipyconsole.get_clients()[0].get_name()
    assert '2' in client_name

@flaky(max_runs=3)
def test_conf_env_vars(ipyconsole, qtbot):
    if False:
        while True:
            i = 10
    'Test that kernels have env vars set by our kernel spec.'
    shell = ipyconsole.get_current_shellwidget()
    with qtbot.waitSignal(shell.executed):
        shell.execute("import os; a = os.environ.get('SPY_TESTING')")
    assert shell.get_value('a') == 'True'

@flaky(max_runs=3)
def test_console_import_namespace(ipyconsole, qtbot):
    if False:
        while True:
            i = 10
    "Test an import of the form 'from foo import *'."
    shell = ipyconsole.get_current_shellwidget()
    with qtbot.waitSignal(shell.executed):
        shell.execute('from numpy import *')
    assert shell.get_value('e') == 2.718281828459045

@flaky(max_runs=3)
def test_console_disambiguation(ipyconsole, qtbot):
    if False:
        return 10
    'Test the disambiguation of dedicated consoles.'
    dir_b = osp.join(TEMP_DIRECTORY, 'a', 'b')
    filename_b = osp.join(dir_b, 'c.py')
    if not osp.isdir(dir_b):
        os.makedirs(dir_b)
    if not osp.isfile(filename_b):
        file_c = open(filename_b, 'w+')
        file_c.close()
    dir_d = osp.join(TEMP_DIRECTORY, 'a', 'd')
    filename_d = osp.join(dir_d, 'c.py')
    if not osp.isdir(dir_d):
        os.makedirs(dir_d)
    if not osp.isfile(filename_d):
        file_e = open(filename_d, 'w+')
        file_e.close()
    ipyconsole.create_client_for_file(filename_b)
    client = ipyconsole.get_current_client()
    assert client.get_name() == 'c.py/A'
    ipyconsole.create_client_for_file(filename_d)
    client = ipyconsole.get_current_client()
    assert client.get_name() == 'c.py - d/A'
    ipyconsole.get_widget().tabwidget.setCurrentIndex(1)
    client = ipyconsole.get_current_client()
    assert client.get_name() == 'c.py - b/A'

@flaky(max_runs=3)
def test_console_coloring(ipyconsole, qtbot):
    if False:
        i = 10
        return i + 15
    'Test that console gets the same coloring present in the Editor.'
    config_options = ipyconsole.get_widget().config_options()
    syntax_style = config_options.JupyterWidget.syntax_style
    style_sheet = config_options.JupyterWidget.style_sheet
    console_font_color = get_console_font_color(syntax_style)
    console_background_color = get_console_background_color(style_sheet)
    selected_color_scheme = ipyconsole.get_conf('selected', section='appearance')
    color_scheme = get_color_scheme(selected_color_scheme)
    editor_background_color = color_scheme['background']
    editor_font_color = color_scheme['normal'][0]
    console_background_color = console_background_color.replace("'", '')
    editor_background_color = editor_background_color.replace("'", '')
    console_font_color = console_font_color.replace("'", '')
    editor_font_color = editor_font_color.replace("'", '')
    assert console_background_color.strip() == editor_background_color.strip()
    assert console_font_color.strip() == editor_font_color.strip()

@flaky(max_runs=3)
def test_set_cwd(ipyconsole, qtbot, tmpdir):
    if False:
        return 10
    'Test kernel when changing cwd.'
    shell = ipyconsole.get_current_shellwidget()
    savetemp = shell.get_cwd()
    tempdir = to_text_string(tmpdir.mkdir("queen's"))
    shell.set_cwd(tempdir)
    with qtbot.waitSignal(shell.executed):
        shell.execute('import os; cwd = os.getcwd()')
    assert shell.get_value('cwd') == tempdir
    shell.set_cwd(savetemp)

@flaky(max_runs=3)
def test_get_cwd(ipyconsole, qtbot, tmpdir):
    if False:
        print('Hello World!')
    'Test current working directory.'
    shell = ipyconsole.get_current_shellwidget()
    savetemp = shell.get_cwd()
    tempdir = to_text_string(tmpdir.mkdir("queen's"))
    assert shell.get_cwd() != tempdir
    if os.name == 'nt':
        tempdir = tempdir.replace(u'\\', u'\\\\')
    with qtbot.waitSignal(shell.executed):
        shell.execute(u"import os; os.chdir(u'''{}''')".format(tempdir))
    if os.name == 'nt':
        tempdir = tempdir.replace(u'\\\\', u'\\')
    assert shell.get_cwd() == tempdir
    shell.set_cwd(savetemp)

@flaky(max_runs=3)
def test_request_env(ipyconsole, qtbot):
    if False:
        while True:
            i = 10
    'Test that getting env vars from the kernel is working as expected.'
    shell = ipyconsole.get_current_shellwidget()
    with qtbot.waitSignal(shell.executed):
        shell.execute("import os; os.environ['FOO'] = 'bar'")
    with qtbot.waitSignal(shell.sig_show_env) as blocker:
        shell.request_env()
    env_contents = blocker.args[0]
    assert env_contents['FOO'] == 'bar'

@flaky(max_runs=3)
@pytest.mark.skipif(os.name == 'nt', reason='Fails due to differences in path handling')
def test_request_syspath(ipyconsole, qtbot, tmpdir):
    if False:
        return 10
    '\n    Test that getting sys.path contents from the kernel is working as\n    expected.\n    '
    shell = ipyconsole.get_current_shellwidget()
    with qtbot.waitSignal(shell.executed):
        tmp_dir = to_text_string(tmpdir)
        shell.execute("import sys; sys.path.append('%s')" % tmp_dir)
    with qtbot.waitSignal(shell.sig_show_syspath) as blocker:
        shell.request_syspath()
    syspath_contents = blocker.args[0]
    assert tmp_dir in syspath_contents

@flaky(max_runs=10)
@pytest.mark.skipif(os.name == 'nt', reason="It doesn't work on Windows")
def test_save_history_dbg(ipyconsole, qtbot):
    if False:
        return 10
    'Test that browsing command history is working while debugging.'
    shell = ipyconsole.get_current_shellwidget()
    control = ipyconsole.get_widget().get_focus_widget()
    control.setFocus()
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug print()')
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClicks(control, 'aa = 10')
        qtbot.keyClick(control, Qt.Key_Enter)
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClicks(control, '!u')
        qtbot.keyClick(control, Qt.Key_Enter)
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(control, Qt.Key_Enter)
    shell.reset(clear=True)
    qtbot.waitUntil(lambda : shell.is_waiting_pdb_input())
    assert shell.is_waiting_pdb_input()
    qtbot.keyClick(control, Qt.Key_Up)
    assert 'aa = 10' in control.toPlainText()
    ipyconsole.create_new_client()
    shell = ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    control = ipyconsole.get_widget().get_focus_widget()
    control.setFocus()
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug print()')
    qtbot.keyClick(control, Qt.Key_Up)
    assert 'aa = 10' in control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(control, Qt.Key_Enter)
    shell._pdb_history.append('if True:\n    print(1)')
    shell._pdb_history.append('print(2)')
    shell._pdb_history.append('if True:\n    print(10)')
    shell._pdb_history_index = len(shell._pdb_history)
    qtbot.keyClick(control, Qt.Key_Up)
    assert '...:     print(10)' in control.toPlainText()
    shell._control.set_cursor_position(shell._control.get_position('eof') - 25)
    qtbot.keyClick(control, Qt.Key_Up)
    assert '...:     print(1)' in control.toPlainText()

@flaky(max_runs=3)
@pytest.mark.skipif(IPython.version_info < (7, 17), reason='insert is not the same in pre 7.17 ipython')
def test_dbg_input(ipyconsole, qtbot):
    if False:
        i = 10
        return i + 15
    "Test that spyder doesn't send pdb commands to unrelated input calls."
    shell = ipyconsole.get_current_shellwidget()
    control = ipyconsole.get_widget().get_focus_widget()
    control.setFocus()
    with qtbot.waitSignal(shell.executed):
        shell.execute("%debug print('Hello', input('name'))")
    shell.pdb_execute('!n')
    qtbot.wait(100)
    qtbot.waitUntil(lambda : control.toPlainText().split()[-1] == 'name')
    shell.pdb_execute('!n')
    shell.pdb_execute('aa = 10')
    qtbot.wait(500)
    assert control.toPlainText().split()[-1] == 'name'
    shell.kernel_client.input('test')
    qtbot.waitUntil(lambda : 'Hello test' in control.toPlainText())

@flaky(max_runs=3)
def test_unicode_vars(ipyconsole, qtbot):
    if False:
        print('Hello World!')
    '\n    Test that the Variable Explorer Works with unicode variables.\n    '
    shell = ipyconsole.get_current_shellwidget()
    with qtbot.waitSignal(shell.executed):
        shell.execute('д = 10')
    assert shell.get_value('д') == 10
    shell.set_value('д', 20)
    qtbot.waitUntil(lambda : shell.get_value('д') == 20)
    assert shell.get_value('д') == 20

@flaky(max_runs=10)
@pytest.mark.no_xvfb
@pytest.mark.skipif(running_in_ci() and os.name == 'nt', reason='Times out on Windows')
def test_values_dbg(ipyconsole, qtbot):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that getting, setting, copying and removing values is working while\n    debugging.\n    '
    shell = ipyconsole.get_current_shellwidget()
    control = ipyconsole.get_widget().get_focus_widget()
    control.setFocus()
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug print()')
    with qtbot.waitSignal(shell.executed):
        shell.execute('aa = 10')
    assert 'aa = 10' in control.toPlainText()
    assert shell.get_value('aa') == 10
    shell.set_value('aa', 20)
    qtbot.waitUntil(lambda : shell.get_value('aa') == 20)
    assert shell.get_value('aa') == 20
    shell.copy_value('aa', 'bb')
    qtbot.waitUntil(lambda : shell.get_value('bb') == 20)
    assert shell.get_value('bb') == 20
    shell.remove_value('aa')

    def is_defined(val):
        if False:
            return 10
        try:
            shell.get_value(val)
            return True
        except KeyError:
            return False
    qtbot.waitUntil(lambda : not is_defined('aa'))
    with qtbot.waitSignal(shell.executed):
        shell.execute('aa')
    assert "*** NameError: name 'aa' is not defined" in control.toPlainText()

@flaky(max_runs=3)
def test_execute_events_dbg(ipyconsole, qtbot):
    if False:
        print('Hello World!')
    'Test execute events while debugging'
    shell = ipyconsole.get_current_shellwidget()
    control = ipyconsole.get_widget().get_focus_widget()
    control.setFocus()
    with qtbot.waitSignal(shell.executed):
        shell.execute('import matplotlib.pyplot as plt')
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug print()')
    ipyconsole.set_conf('pdb_execute_events', True, section='debugger')
    shell.set_kernel_configuration('pdb', {'pdb_execute_events': True})
    qtbot.keyClicks(control, 'plt.plot(range(10))')
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(control, Qt.Key_Enter)
    assert shell._control.toHtml().count('img src') == 1
    ipyconsole.set_conf('pdb_execute_events', False, section='debugger')
    shell.set_kernel_configuration('pdb', {'pdb_execute_events': False})
    qtbot.keyClicks(control, 'plt.plot(range(10))')
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(control, Qt.Key_Enter)
    assert shell._control.toHtml().count('img src') == 1
    qtbot.keyClicks(control, 'plt.show()')
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(control, Qt.Key_Enter)
    assert shell._control.toHtml().count('img src') == 2

@flaky(max_runs=3)
def test_run_doctest(ipyconsole, qtbot):
    if False:
        return 10
    '\n    Test that doctests can be run without problems\n    '
    shell = ipyconsole.get_current_shellwidget()
    code = dedent('\n    def add(x, y):\n        """\n        >>> add(1, 2)\n        3\n        >>> add(5.1, 2.2)\n        7.3\n        """\n        return x + y\n    ')
    with qtbot.waitSignal(shell.executed):
        shell.execute(code)
    with qtbot.waitSignal(shell.executed):
        shell.execute('import doctest')
    with qtbot.waitSignal(shell.executed):
        shell.execute('doctest.testmod()')
    assert 'TestResults(failed=0, attempted=2)' in shell._control.toPlainText()

@flaky(max_runs=3)
@pytest.mark.skipif(not os.name == 'nt' and running_in_ci(), reason='Fails on Linux/Mac and CIs')
def test_mpl_backend_change(ipyconsole, qtbot):
    if False:
        i = 10
        return i + 15
    '\n    Test that Matplotlib backend is changed correctly when\n    using the %matplotlib magic\n    '
    shell = ipyconsole.get_current_shellwidget()
    with qtbot.waitSignal(shell.executed):
        shell.execute('import matplotlib.pyplot as plt')
    with qtbot.waitSignal(shell.executed):
        shell.execute('plt.plot(range(10))')
    with qtbot.waitSignal(shell.executed):
        shell.execute('%matplotlib tk')
    with qtbot.waitSignal(shell.executed):
        shell.execute('plt.plot(range(10))')
    assert shell._control.toHtml().count('img src') == 1

@flaky(max_runs=10)
@pytest.mark.skipif(os.name == 'nt', reason="It doesn't work on Windows")
def test_clear_and_reset_magics_dbg(ipyconsole, qtbot):
    if False:
        print('Hello World!')
    '\n    Test that clear and reset magics are working while debugging\n    '
    shell = ipyconsole.get_current_shellwidget()
    control = ipyconsole.get_widget().get_focus_widget()
    control.setFocus()
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug print()')
    shell.clear_console()
    qtbot.waitUntil(lambda : '\nIPdb [2]: ' == control.toPlainText())
    qtbot.keyClicks(control, 'bb = 10')
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(control, Qt.Key_Enter)
    assert shell.get_value('bb') == 10
    shell.reset_namespace()
    qtbot.wait(1000)
    qtbot.keyClicks(control, 'bb')
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(control, Qt.Key_Enter)
    assert "*** NameError: name 'bb' is not defined" in control.toPlainText()

@flaky(max_runs=3)
def test_restart_kernel(ipyconsole, mocker, qtbot):
    if False:
        i = 10
        return i + 15
    '\n    Test that kernel is restarted correctly\n    '
    mocker.patch.object(ShellWidget, 'send_spyder_kernel_configuration')
    ipyconsole.create_new_client()
    shell = ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed):
        shell.execute('a = 10')
    with qtbot.waitSignal(shell.executed):
        shell.execute('import sys; sys.__stderr__.write("HEL"+"LO")')
    qtbot.waitUntil(lambda : 'HELLO' in shell._control.toPlainText(), timeout=SHELL_TIMEOUT)
    shell._prompt_html = None
    ipyconsole.restart_kernel()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    assert 'Restarting kernel...' in shell._control.toPlainText()
    assert 'HELLO' not in shell._control.toPlainText()
    assert not shell.is_defined('a')
    qtbot.waitUntil(lambda : ShellWidget.send_spyder_kernel_configuration.call_count == 2)

@flaky(max_runs=3)
def test_load_kernel_file_from_id(ipyconsole, qtbot):
    if False:
        print('Hello World!')
    '\n    Test that a new client is created using its id\n    '
    client = ipyconsole.get_current_client()
    connection_file = osp.basename(client.connection_file)
    id_ = connection_file.split('kernel-')[-1].split('.json')[0]
    ipyconsole.create_client_for_kernel(id_)
    qtbot.waitUntil(lambda : len(ipyconsole.get_clients()) == 2)
    new_client = ipyconsole.get_clients()[1]
    assert new_client.id_ == dict(int_id='1', str_id='B')

@flaky(max_runs=3)
def test_load_kernel_file_from_location(ipyconsole, qtbot, tmpdir):
    if False:
        print('Hello World!')
    '\n    Test that a new client is created using a connection file\n    placed in a different location from jupyter_runtime_dir\n    '
    client = ipyconsole.get_current_client()
    fname = osp.basename(client.connection_file)
    connection_file = to_text_string(tmpdir.join(fname))
    shutil.copy2(client.connection_file, connection_file)
    ipyconsole.create_client_for_kernel(connection_file)
    qtbot.waitUntil(lambda : len(ipyconsole.get_clients()) == 2)
    assert len(ipyconsole.get_clients()) == 2

@flaky(max_runs=3)
def test_load_kernel_file(ipyconsole, qtbot, tmpdir):
    if False:
        print('Hello World!')
    '\n    Test that a new client is created using the connection file\n    of an existing client\n    '
    shell = ipyconsole.get_current_shellwidget()
    client = ipyconsole.get_current_client()
    ipyconsole.create_client_for_kernel(client.connection_file)
    qtbot.waitUntil(lambda : len(ipyconsole.get_clients()) == 2)
    new_client = ipyconsole.get_clients()[1]
    new_shell = new_client.shellwidget
    qtbot.waitUntil(lambda : new_shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(new_shell.executed):
        new_shell.execute('a = 10')
    assert new_client.id_ == dict(int_id='1', str_id='B')
    assert shell.get_value('a') == new_shell.get_value('a')

@flaky(max_runs=3)
def test_sys_argv_clear(ipyconsole, qtbot):
    if False:
        return 10
    'Test that sys.argv is cleared up correctly'
    shell = ipyconsole.get_current_shellwidget()
    with qtbot.waitSignal(shell.executed):
        shell.execute('import sys; A = sys.argv')
    argv = shell.get_value('A')
    assert argv == ['']

@flaky(max_runs=5)
@pytest.mark.skipif(os.name == 'nt', reason='Fails sometimes on Windows')
def test_set_elapsed_time(ipyconsole, qtbot):
    if False:
        while True:
            i = 10
    'Test that the IPython console elapsed timer is set correctly.'
    client = ipyconsole.get_current_client()
    main_widget = ipyconsole.get_widget()
    main_widget.set_show_elapsed_time_current_client(True)
    client.t0 -= 120
    with qtbot.waitSignal(client.timer.timeout, timeout=5000):
        client.timer.timeout.connect(client.show_time)
        client.timer.start(1000)
    assert '00:02:00' in main_widget.time_label.text() or '00:02:01' in main_widget.time_label.text()
    with qtbot.waitSignal(client.timer.timeout, timeout=5000):
        pass
    assert '00:02:01' in main_widget.time_label.text() or '00:02:02' in main_widget.time_label.text()
    client.t0 += 2000
    with qtbot.waitSignal(client.timer.timeout, timeout=5000):
        pass
    assert '00:00:00' in main_widget.time_label.text()
    client.timer.timeout.disconnect(client.show_time)

@flaky(max_runs=3)
@pytest.mark.skipif(sys.platform == 'darwin', reason='Fails sometimes on macOS')
def test_kernel_crash(ipyconsole, qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Test that we show an error message when a kernel crash occurs.'
    ipy_kernel_cfg = osp.join(get_ipython_dir(), 'profile_default', 'ipython_config.py')
    try:
        with open(ipy_kernel_cfg, 'w') as f:
            f.write('c.InteractiveShellApp.extra_extensions = 1')
        ipyconsole.get_widget().close_cached_kernel()
        ipyconsole.create_new_client()
        error_client = ipyconsole.get_clients()[-1]
        qtbot.waitUntil(lambda : bool(error_client.error_text), timeout=6000)
        assert error_client.error_text
        webview = error_client.infowidget
        if WEBENGINE:
            webpage = webview.page()
        else:
            webpage = webview.page().mainFrame()
        qtbot.waitUntil(lambda : check_text(webpage, 'Bad config encountered'), timeout=6000)
        qtbot.waitUntil(lambda : bool(ipyconsole.get_widget()._cached_kernel_properties[-1]._init_stderr))
        ipyconsole.create_new_client()
        error_client = ipyconsole.get_clients()[-1]
        qtbot.waitUntil(lambda : bool(error_client.error_text), timeout=6000)
    finally:
        os.remove(ipy_kernel_cfg)

@flaky(max_runs=3)
@pytest.mark.use_startup_wdir
def test_startup_working_directory(ipyconsole, qtbot):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that the fixed startup working directory option works as expected.\n    '
    shell = ipyconsole.get_current_shellwidget()
    with qtbot.waitSignal(shell.executed):
        shell.execute('import os; cwd = os.getcwd()')
    current_wdir = shell.get_value('cwd')
    folders = osp.split(current_wdir)
    assert folders[-1] == NEW_DIR

@flaky(max_runs=3)
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Only works on Linux')
@pytest.mark.skipif(parse('8.7.0') < parse(ipy_release.version) < parse('8.11.0'), reason='Fails for IPython 8.8.0, 8.9.0 and 8.10.0')
def test_console_complete(ipyconsole, qtbot, tmpdir):
    if False:
        while True:
            i = 10
    'Test code completions in the console.'
    shell = ipyconsole.get_current_shellwidget()
    control = ipyconsole.get_widget().get_focus_widget()
    control.setFocus()

    def check_value(name, value):
        if False:
            return 10
        try:
            return shell.get_value(name) == value
        except KeyError:
            return False
    with qtbot.waitSignal(shell.executed):
        shell.execute('cbs = 1')
    qtbot.waitUntil(lambda : check_value('cbs', 1))
    qtbot.wait(500)
    qtbot.keyClicks(control, 'cb')
    qtbot.keyClick(control, Qt.Key_Tab)
    qtbot.waitUntil(lambda : control.toPlainText().split()[-1] == 'cbs', timeout=6000)
    with qtbot.waitSignal(shell.executed):
        shell.execute('cbba = 1')
    qtbot.waitUntil(lambda : check_value('cbba', 1))
    qtbot.keyClicks(control, 'cb')
    qtbot.keyClick(control, Qt.Key_Tab)
    qtbot.waitUntil(shell._completion_widget.isVisible)
    assert control.toPlainText().split()[-1] == 'cb'
    qtbot.keyClick(shell._completion_widget, Qt.Key_Enter)
    qtbot.waitUntil(lambda : control.toPlainText().split()[-1] == 'cbba')
    with qtbot.waitSignal(shell.executed):
        shell.execute('import pandas as pd')
    qtbot.keyClicks(control, 'test = pd.conc')
    qtbot.keyClick(control, Qt.Key_Tab)
    qtbot.wait(500)
    completed_text = control.toPlainText().splitlines()[-1].split(':')[-1]
    assert completed_text.strip() == 'test = pd.concat'
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug print()')
    qtbot.keyClicks(control, 'ab')
    qtbot.keyClick(control, Qt.Key_Tab)
    qtbot.waitUntil(lambda : control.toPlainText().split()[-1] == 'abs')
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(control, Qt.Key_Enter)
    qtbot.keyClicks(control, 'print(ab')
    qtbot.keyClick(control, Qt.Key_Tab)
    qtbot.waitUntil(lambda : control.toPlainText().split()[-1] == 'print(abs')
    qtbot.keyClicks(control, ')')
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(control, Qt.Key_Enter)
    qtbot.keyClicks(control, 'baab = 10')
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(control, Qt.Key_Enter)
    qtbot.wait(100)
    qtbot.waitUntil(lambda : check_value('baab', 10))
    qtbot.keyClicks(control, 'baa')
    qtbot.keyClick(control, Qt.Key_Tab)
    qtbot.waitUntil(lambda : control.toPlainText().split()[-1] == 'baab')
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(control, Qt.Key_Enter)
    qtbot.keyClicks(control, 'abba = 10')
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(control, Qt.Key_Enter)
    qtbot.wait(100)
    qtbot.waitUntil(lambda : check_value('abba', 10))
    qtbot.keyClicks(control, 'ab')
    qtbot.keyClick(control, Qt.Key_Tab)
    qtbot.waitUntil(shell._completion_widget.isVisible)
    assert control.toPlainText().split()[-1] == 'ab'
    qtbot.keyClick(shell._completion_widget, Qt.Key_Enter)
    qtbot.waitUntil(lambda : control.toPlainText().split()[-1] == 'abba')
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(control, Qt.Key_Enter)
    qtbot.keyClicks(control, 'class A(): baba = 1')
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(control, Qt.Key_Enter)
    qtbot.wait(100)
    qtbot.waitUntil(lambda : shell.is_defined('A'))
    qtbot.keyClicks(control, 'a = A()')
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(control, Qt.Key_Enter)
    qtbot.wait(100)
    qtbot.waitUntil(lambda : shell.is_defined('a'))
    qtbot.keyClicks(control, 'a.ba')
    qtbot.keyClick(control, Qt.Key_Tab)
    qtbot.waitUntil(lambda : control.toPlainText().split()[-1] == 'a.baba')
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(control, Qt.Key_Enter)
    qtbot.keyClicks(control, '!longl')
    qtbot.keyClick(control, Qt.Key_Tab)
    qtbot.waitUntil(lambda : control.toPlainText().split()[-1] == '!longlist')
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(control, Qt.Key_Enter)
    test_file = tmpdir.join('test.py')
    test_file.write('stuff\n')
    qtbot.keyClicks(control, '!b ' + str(test_file) + ':1')
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(control, Qt.Key_Enter)
    qtbot.keyClicks(control, '!ignore ')
    qtbot.keyClick(control, Qt.Key_Tab)
    qtbot.waitUntil(lambda : control.toPlainText().split()[-1] == '1')

@flaky(max_runs=10)
def test_pdb_multiline(ipyconsole, qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Test entering a multiline statment into pdb'
    shell = ipyconsole.get_current_shellwidget()
    control = ipyconsole.get_widget().get_focus_widget()
    control.setFocus()
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug print()')
    assert '\nIPdb [' in control.toPlainText()
    qtbot.keyClicks(control, 'if True:')
    qtbot.keyClick(control, Qt.Key_Enter)
    qtbot.wait(500)
    qtbot.keyClicks(control, 'bb = 10')
    qtbot.keyClick(control, Qt.Key_Enter)
    qtbot.wait(500)
    qtbot.keyClick(control, Qt.Key_Enter)
    qtbot.wait(500)
    assert shell.get_value('bb') == 10
    assert 'if True:\n     ...:     bb = 10\n' in control.toPlainText()

@flaky(max_runs=3)
@pytest.mark.parametrize('show_lib', [True, False])
def test_pdb_ignore_lib(ipyconsole, qtbot, show_lib):
    if False:
        while True:
            i = 10
    'Test that pdb can avoid closed files.'
    shell = ipyconsole.get_current_shellwidget()
    control = ipyconsole.get_widget().get_focus_widget()
    control.setFocus()
    qtbot.wait(1000)
    ipyconsole.set_conf('pdb_ignore_lib', not show_lib, section='debugger')
    qtbot.wait(1000)
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug print()')
    with qtbot.waitSignal(shell.executed):
        shell.execute('"value = " + str(get_ipython().pdb_session.pdb_ignore_lib)')
    assert 'value = ' + str(not show_lib) in control.toPlainText()
    qtbot.keyClicks(control, '!s')
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(control, Qt.Key_Enter)
    qtbot.wait(500)
    qtbot.keyClicks(control, '!q')
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(control, Qt.Key_Enter)
    if show_lib:
        assert 'iostream.py' in control.toPlainText()
    else:
        assert 'iostream.py' not in control.toPlainText()
    ipyconsole.set_conf('pdb_ignore_lib', True, section='debugger')

@flaky(max_runs=3)
@pytest.mark.skipif(sys.platform == 'darwin', reason='Times out on macOS')
def test_calltip(ipyconsole, qtbot):
    if False:
        i = 10
        return i + 15
    '\n    Test Calltip.\n\n    See spyder-ide/spyder#10842\n    '
    shell = ipyconsole.get_current_shellwidget()
    control = ipyconsole.get_widget().get_focus_widget()
    control.setFocus()
    with qtbot.waitSignal(shell.executed):
        shell.execute('a = {"a": 1}')
    qtbot.keyClicks(control, 'a.keys(', delay=100)
    qtbot.wait(1000)
    assert control.calltip_widget.isVisible()

@flaky(max_runs=3)
@pytest.mark.order(1)
@pytest.mark.test_environment_interpreter
@pytest.mark.skipif(not is_anaconda(), reason='Only works with Anaconda')
@pytest.mark.skipif(not running_in_ci(), reason='Only works on CIs')
@pytest.mark.skipif(not os.name == 'nt', reason='Works reliably on Windows')
def test_conda_env_activation(ipyconsole, qtbot):
    if False:
        print('Hello World!')
    '\n    Test that the conda environment associated with an external interpreter\n    is activated before a kernel is created for it.\n    '
    shell = ipyconsole.get_current_shellwidget()
    with qtbot.waitSignal(shell.executed):
        shell.execute("import os; conda_prefix = os.environ.get('CONDA_PREFIX')")
    expected_output = get_conda_test_env()[0].replace('\\', '/')
    output = shell.get_value('conda_prefix').replace('\\', '/')
    assert expected_output == output

@flaky(max_runs=3)
@pytest.mark.skipif(os.name == 'nt', reason='no SIGTERM on Windows')
def test_kernel_kill(ipyconsole, qtbot):
    if False:
        while True:
            i = 10
    '\n    Test that the kernel correctly restarts after a kill.\n    '
    shell = ipyconsole.get_current_shellwidget()
    qtbot.wait(3000)
    crash_string = 'import os, signal; os.kill(os.getpid(), signal.SIGTERM)'
    old_open_comms = list(shell.kernel_handler.kernel_comm._comms.keys())
    assert len(old_open_comms) == 1
    with qtbot.waitSignal(shell.sig_prompt_ready, timeout=30000):
        shell.execute(crash_string)
    assert crash_string in shell._control.toPlainText()
    assert 'Restarting kernel...' in shell._control.toPlainText()
    new_open_comms = list(shell.kernel_handler.kernel_comm._comms.keys())
    assert len(new_open_comms) == 1
    assert old_open_comms[0] != new_open_comms[0]
    qtbot.waitUntil(lambda : shell.kernel_handler.kernel_comm._comms[new_open_comms[0]]['status'] == 'ready')
    assert shell.kernel_handler.kernel_comm._comms[new_open_comms[0]]['status'] == 'ready'

@flaky(max_runs=3)
@pytest.mark.parametrize('spyder_pythonpath', [True, False])
def test_wrong_std_module(ipyconsole, qtbot, tmpdir, spyder_pythonpath):
    if False:
        for i in range(10):
            print('nop')
    "\n    Test that a file with the same name of a standard library module in\n    the current working directory doesn't break the console.\n    "
    if spyder_pythonpath:
        wrong_random_mod = tmpdir.join('random.py')
        wrong_random_mod.write('')
        wrong_random_mod = str(wrong_random_mod)
        ipyconsole.set_conf('spyder_pythonpath', [str(tmpdir)], section='pythonpath_manager')
    else:
        wrong_random_mod = osp.join(os.getcwd(), 'random.py')
        with open(wrong_random_mod, 'w') as f:
            f.write('')
    ipyconsole.create_new_client()
    shell = ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    if spyder_pythonpath:
        check_sys_path = "import sys; path_added = r'{}' in sys.path".format(str(tmpdir))
        with qtbot.waitSignal(shell.sig_prompt_ready, timeout=30000):
            shell.execute(check_sys_path)
        assert shell.get_value('path_added')
    os.remove(wrong_random_mod)
    ipyconsole.set_conf('spyder_pythonpath', [], section='pythonpath_manager')

@flaky(max_runs=3)
@pytest.mark.known_leak
@pytest.mark.skipif(os.name == 'nt', reason='no SIGTERM on Windows')
def test_kernel_restart_after_manual_restart_and_crash(ipyconsole, qtbot):
    if False:
        while True:
            i = 10
    '\n    Test that the kernel restarts correctly after being restarted\n    manually and then it crashes.\n\n    This is a regresion for spyder-ide/spyder#12972.\n    '
    shell = ipyconsole.get_current_shellwidget()
    shell._prompt_html = None
    ipyconsole.restart_kernel()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    qtbot.wait(3000)
    crash_string = 'import os, signal; os.kill(os.getpid(), signal.SIGTERM)'
    with qtbot.waitSignal(shell.sig_prompt_ready, timeout=30000):
        shell.execute(crash_string)
    assert crash_string in shell._control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        shell.execute('a = 10')
    assert shell.is_defined('a')
    open_comms = list(shell.kernel_handler.kernel_comm._comms.keys())
    qtbot.waitUntil(lambda : shell.kernel_handler.kernel_comm._comms[open_comms[0]]['status'] == 'ready')

@flaky(max_runs=3)
def test_stderr_poll(ipyconsole, qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Test if the content of stderr is printed to the console.'
    shell = ipyconsole.get_current_shellwidget()
    with qtbot.waitSignal(shell.executed):
        shell.execute('import sys; print("test_" + "test", file=sys.__stderr__)')
    qtbot.waitUntil(lambda : 'test_test' in ipyconsole.get_widget().get_focus_widget().toPlainText())
    assert 'test_test' in ipyconsole.get_widget().get_focus_widget().toPlainText()
    with qtbot.waitSignal(shell.executed):
        shell.execute('import sys; print("test_" + "test", file=sys.__stderr__)')
    qtbot.waitUntil(lambda : ipyconsole.get_widget().get_focus_widget().toPlainText().count('test_test') == 2)
    assert ipyconsole.get_widget().get_focus_widget().toPlainText().count('test_test') == 2

@flaky(max_runs=3)
def test_stdout_poll(ipyconsole, qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Test if the content of stdout is printed to the console.'
    shell = ipyconsole.get_current_shellwidget()
    with qtbot.waitSignal(shell.executed):
        shell.execute('import sys; print("test_test", file=sys.__stdout__)')
    qtbot.waitUntil(lambda : 'test_test' in ipyconsole.get_widget().get_focus_widget().toPlainText(), timeout=5000)

@flaky(max_runs=10)
def test_startup_code_pdb(ipyconsole, qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Test that startup code for pdb works.'
    shell = ipyconsole.get_current_shellwidget()
    control = ipyconsole.get_widget().get_focus_widget()
    control.setFocus()
    ipyconsole.set_conf('startup/pdb_run_lines', 'abba = 12; print("Hello")')
    shell.execute('%debug print()')
    qtbot.waitUntil(lambda : 'Hello' in control.toPlainText())
    assert shell.get_value('abba') == 12
    ipyconsole.set_conf('startup/pdb_run_lines', '')

@flaky(max_runs=3)
@pytest.mark.parametrize('backend', ['inline', 'qt', 'tk', 'osx'])
@pytest.mark.skipif(sys.platform == 'darwin', reason='Hangs frequently on Mac')
def test_pdb_eventloop(ipyconsole, qtbot, backend):
    if False:
        while True:
            i = 10
    'Check if setting an event loop while debugging works.'
    if backend == 'osx' and sys.platform != 'darwin':
        return
    if backend == 'qt' and (not os.name == 'nt') and running_in_ci():
        return
    shell = ipyconsole.get_current_shellwidget()
    control = ipyconsole.get_widget().get_focus_widget()
    with qtbot.waitSignal(shell.executed):
        shell.execute('%matplotlib ' + backend)
    qtbot.wait(1000)
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug print()')
    with qtbot.waitSignal(shell.executed):
        shell.execute("print('Two: ' + str(1+1))")
    assert 'Two: 2' in control.toPlainText()

@flaky(max_runs=3)
def test_recursive_pdb(ipyconsole, qtbot):
    if False:
        return 10
    'Check commands and code are separted.'
    shell = ipyconsole.get_current_shellwidget()
    control = ipyconsole.get_widget().get_focus_widget()
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug print()')
    with qtbot.waitSignal(shell.executed):
        shell.pdb_execute('abab = 10')
    with qtbot.waitSignal(shell.executed):
        shell.pdb_execute('%debug print()')
    assert '(IPdb [1]):' in control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        shell.pdb_execute('!debug print()')
    assert '((IPdb [1])):' in control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        shell.pdb_execute('!debug print()')
    assert '(((IPdb [1]))):' in control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        shell.pdb_execute('!quit')
    with qtbot.waitSignal(shell.executed):
        shell.pdb_execute('!quit')
    assert control.toPlainText().split()[-2:] == ['(IPdb', '[2]):']
    qtbot.keyClicks(control, 'aba')
    qtbot.keyClick(control, Qt.Key_Tab)
    qtbot.waitUntil(lambda : control.toPlainText().split()[-1] == 'abab', timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed):
        shell.pdb_execute('!quit')
    assert control.toPlainText().split()[-2:] == ['IPdb', '[3]:']
    qtbot.keyClicks(control, 'aba')
    qtbot.keyClick(control, Qt.Key_Tab)
    qtbot.waitUntil(lambda : control.toPlainText().split()[-1] == 'abab', timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed):
        shell.pdb_execute('!quit')
    with qtbot.waitSignal(shell.executed):
        shell.execute('1 + 1')
    assert control.toPlainText().split()[-2:] == ['In', '[3]:']

@flaky(max_runs=3)
@pytest.mark.skipif(os.name == 'nt', reason="Doesn't work on windows")
def test_stop_pdb(ipyconsole, qtbot):
    if False:
        return 10
    'Test if we can stop pdb'
    shell = ipyconsole.get_current_shellwidget()
    control = ipyconsole.get_widget().get_focus_widget()
    stop_button = ipyconsole.get_widget().stop_button
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug print()')
    shell.execute('import time; time.sleep(10)')
    qtbot.wait(500)
    with qtbot.waitSignal(shell.executed, timeout=10000):
        qtbot.mouseClick(stop_button, Qt.LeftButton)
    assert 'KeyboardInterrupt' in control.toPlainText()
    assert 'IPdb [2]:' in control.toPlainText()
    assert 'In [2]:' not in control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        qtbot.mouseClick(stop_button, Qt.LeftButton)
    assert 'In [2]:' in control.toPlainText()

@flaky(max_runs=3)
def test_code_cache(ipyconsole, qtbot):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that code sent to execute is properly cached\n    and that the cache is emptied on interrupt.\n    '
    shell = ipyconsole.get_current_shellwidget()
    control = ipyconsole.get_widget().get_focus_widget()
    control.setFocus()

    def check_value(name, value):
        if False:
            i = 10
            return i + 15
        try:
            return shell.get_value(name) == value
        except KeyError:
            return False
    shell.execute('import time; time.sleep(.5)')
    with qtbot.waitSignal(shell.executed):
        shell.execute('var = 142')
    qtbot.wait(500)
    qtbot.waitUntil(lambda : check_value('var', 142))
    assert shell.get_value('var') == 142
    shell.execute('import time; time.sleep(.5)')
    shell.execute('var = 1000')
    qtbot.wait(100)
    shell.interrupt_kernel()
    qtbot.wait(1000)
    assert shell.get_value('var') == 142
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug print()')
    assert 'IPdb [' in shell._control.toPlainText()
    shell.execute('time.sleep(.5)')
    shell.execute('var = 318')
    qtbot.wait(500)
    qtbot.waitUntil(lambda : check_value('var', 318))
    assert shell.get_value('var') == 318
    shell.execute('import time; time.sleep(.5)')
    shell.execute('var = 1000')
    qtbot.wait(100)
    shell.interrupt_kernel()
    qtbot.wait(1000)
    assert shell.get_value('var') == 318

@flaky(max_runs=3)
def test_pdb_code_and_cmd_separation(ipyconsole, qtbot):
    if False:
        while True:
            i = 10
    'Check commands and code are separted.'
    shell = ipyconsole.get_current_shellwidget()
    control = ipyconsole.get_widget().get_focus_widget()
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug print()')
    assert 'Error' not in control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        shell.execute('e')
    assert "name 'e' is not defined" in control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        shell.execute('!n')
    assert '--Return--' in control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        shell.execute('a')
    assert "*** NameError: name 'a' is not defined" not in control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        shell.execute('abba')
    assert "name 'abba' is not defined" in control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        shell.execute('!abba')
    assert "Unknown command 'abba'" in control.toPlainText()

@flaky(max_runs=3)
def test_breakpoint_builtin(ipyconsole, qtbot, tmpdir):
    if False:
        while True:
            i = 10
    'Check that the breakpoint builtin is working.'
    shell = ipyconsole.get_current_shellwidget()
    control = ipyconsole.get_widget().get_focus_widget()
    code = dedent("\n    print('foo')\n    breakpoint()\n    ")
    file = tmpdir.join('test_breakpoint.py')
    file.write(code)
    with qtbot.waitSignal(shell.executed):
        shell.execute(f'%runfile {repr(str(file))}')
    qtbot.wait(5000)
    assert 'foo' in control.toPlainText()
    assert 'IPdb [1]:' in control.toPlainText()

def test_pdb_out(ipyconsole, qtbot):
    if False:
        while True:
            i = 10
    'Test that browsing command history is working while debugging.'
    shell = ipyconsole.get_current_shellwidget()
    control = ipyconsole.get_widget().get_focus_widget()
    control.setFocus()
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug print()')
    with qtbot.waitSignal(shell.executed):
        shell.pdb_execute('a = 12 + 1; a')
    assert '[1]: 13' in control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        shell.pdb_execute('a = 14 + 1; a;')
    assert '[2]: 15' not in control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        shell.pdb_execute('a = 16 + 1\na')
    assert '[3]: 17' in control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        shell.pdb_execute('a = 18 + 1\na;')
    assert '[4]: 19' not in control.toPlainText()
    assert 'IPdb [4]:' in control.toPlainText()

@flaky(max_runs=3)
@pytest.mark.auto_backend
@pytest.mark.skipif(running_in_ci() and (not os.name == 'nt'), reason='Times out on Linux and macOS')
@pytest.mark.skipif(parse(spyder_kernels_version) < parse('3.0.0.dev0'), reason='Not reliable with Spyder-kernels 2')
def test_shutdown_kernel(ipyconsole, qtbot):
    if False:
        return 10
    '\n    Check that the kernel is shutdown after creating plots with the\n    automatic backend.\n\n    This is a regression test for issue spyder-ide/spyder#17011\n    '
    shell = ipyconsole.get_current_shellwidget()
    qtbot.wait(1000)
    with qtbot.waitSignal(shell.executed):
        shell.execute('import matplotlib.pyplot as plt; plt.plot(range(10))')
    qtbot.wait(1000)
    with qtbot.waitSignal(shell.executed):
        shell.execute('import os; pid = os.getpid()')
    qtbot.wait(1000)
    kernel_pid = shell.get_value('pid')
    ipyconsole.get_widget().close_client()
    qtbot.wait(5000)
    shell = ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed):
        shell.execute(f'import psutil; kernel_exists = psutil.pid_exists({kernel_pid})')
    assert not shell.get_value('kernel_exists')

def test_pdb_comprehension_namespace(ipyconsole, qtbot, tmpdir):
    if False:
        i = 10
        return i + 15
    'Check that the debugger handles the namespace of a comprehension.'
    shell = ipyconsole.get_current_shellwidget()
    control = ipyconsole.get_widget().get_focus_widget()
    code = 'locals = 1\nx = [locals + i for i in range(2)]'
    file = tmpdir.join('test_breakpoint.py')
    file.write(code)
    with qtbot.waitSignal(shell.executed):
        shell.execute(f'%debugfile {repr(str(file))}')
    for i in range(4):
        with qtbot.waitSignal(shell.executed):
            shell.pdb_execute('s')
    assert 'Error' not in control.toPlainText()
    with qtbot.waitSignal(shell.executed):
        shell.pdb_execute("print('test', locals + i + 10)")
    assert 'Error' not in control.toPlainText()
    assert 'test 11' in control.toPlainText()
    settings = {'check_all': False, 'exclude_callables_and_modules': True, 'exclude_capitalized': False, 'exclude_private': True, 'exclude_unsupported': False, 'exclude_uppercase': True, 'excluded_names': [], 'minmax': False, 'show_callable_attributes': True, 'show_special_attributes': False, 'filter_on': True}
    shell.set_kernel_configuration('namespace_view_settings', settings)
    namespace = shell.call_kernel(blocking=True).get_namespace_view()
    for key in namespace:
        assert '_spyderpdb' not in key

@flaky(max_runs=3)
@pytest.mark.auto_backend
def test_restart_intertactive_backend(ipyconsole, qtbot):
    if False:
        i = 10
        return i + 15
    '\n    Test that we ask for a restart after switching to a different interactive\n    backend in preferences.\n    '
    main_widget = ipyconsole.get_widget()
    qtbot.wait(1000)
    main_widget.change_possible_restart_and_mpl_conf('pylab/backend', 'tk')
    assert bool(os.environ.get('BACKEND_REQUIRE_RESTART'))

@flaky(max_runs=3)
@pytest.mark.no_web_widgets
def test_no_infowidget(ipyconsole):
    if False:
        return 10
    "Test that we don't create the infowidget if requested by the user."
    client = ipyconsole.get_widget().get_current_client()
    assert client.infowidget is None

@flaky(max_runs=3)
def test_cwd_console_options(ipyconsole, qtbot, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that the working directory options for new consoles work as expected.\n    '

    def get_cwd_of_new_client():
        if False:
            return 10
        ipyconsole.create_new_client()
        shell = ipyconsole.get_current_shellwidget()
        qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
        with qtbot.waitSignal(shell.executed):
            shell.execute('import os; cwd = os.getcwd()')
        return shell.get_value('cwd')
    ipyconsole.set_conf('console/use_project_or_home_directory', True, section='workingdir')
    project_dir = str(tmpdir.mkdir('ipyconsole_project_test'))
    ipyconsole.get_widget().update_active_project_path(project_dir)
    assert get_cwd_of_new_client() == project_dir
    ipyconsole.set_conf('console/use_project_or_home_directory', False, section='workingdir')
    ipyconsole.set_conf('console/use_cwd', True, section='workingdir')
    cwd_dir = str(tmpdir.mkdir('ipyconsole_cwd_test'))
    ipyconsole.get_widget().save_working_directory(cwd_dir)
    assert get_cwd_of_new_client() == cwd_dir
    ipyconsole.set_conf('console/use_cwd', False, section='workingdir')
    ipyconsole.set_conf('console/use_fixed_directory', True, section='workingdir')
    fixed_dir = str(tmpdir.mkdir('ipyconsole_fixed_test'))
    ipyconsole.set_conf('console/fixed_directory', fixed_dir, section='workingdir')
    assert get_cwd_of_new_client() == fixed_dir

def test_startup_run_lines_project_directory(ipyconsole, qtbot, tmpdir):
    if False:
        i = 10
        return i + 15
    "\n    Test 'startup/run_lines' config works with code from an active project.\n    "
    project = tmpdir.mkdir('ipyconsole_project_test')
    project_dir = str(project)
    project_script = project.join('project_script.py')
    project_script.write('from numpy import pi')
    ipyconsole.set_conf('spyder_pythonpath', [project_dir], section='pythonpath_manager')
    ipyconsole.set_conf('startup/run_lines', 'from project_script import *', section='ipython_console')
    ipyconsole.set_conf('console/use_project_or_home_directory', True, section='workingdir')
    ipyconsole.get_widget().update_active_project_path(project_dir)
    ipyconsole.restart()
    shell = ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell.spyder_kernel_ready and shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    assert shell.get_value('pi')
    ipyconsole.set_conf('spyder_pythonpath', [], section='pythonpath_manager')
    ipyconsole.set_conf('startup/run_lines', '', section='ipython_console')

def test_varexp_magic_dbg_locals(ipyconsole, qtbot):
    if False:
        print('Hello World!')
    'Test that %varexp is working while debugging locals.'
    shell = ipyconsole.get_current_shellwidget()
    with qtbot.waitSignal(shell.executed):
        shell.execute('def f():\n    li = [1, 2]\n    return li')
    with qtbot.waitSignal(shell.executed):
        shell.execute('%debug f()')
    for _ in range(4):
        with qtbot.waitSignal(shell.executed):
            shell.execute('!s')
    with qtbot.waitSignal(shell.executed):
        shell.execute('%varexp --plot li')
    qtbot.wait(1000)
    assert shell._control.toHtml().count('img src') == 1

@pytest.mark.skipif(os.name == 'nt', reason='Fails on windows')
def test_old_kernel_version(ipyconsole, qtbot):
    if False:
        while True:
            i = 10
    '\n    Check that an error is shown when an version of spyder-kernels is used.\n    '
    w = ipyconsole.get_widget()
    kernel_handler = w._cached_kernel_properties[-1]
    kernel_handler.kernel_client.sig_spyder_kernel_info.disconnect()
    qtbot.waitUntil(lambda : kernel_handler._comm_ready_received, timeout=SHELL_TIMEOUT)
    kernel_handler.check_spyder_kernel_info(('1.0.0', ''))
    w.create_new_client()
    client = w.get_current_client()
    control = client.get_control()
    qtbot.waitUntil(lambda : '1.0.0' in control.toPlainText(), timeout=SHELL_TIMEOUT)
    assert 'pip install spyder' in control.toPlainText()

def test_run_script(ipyconsole, qtbot, tmp_path):
    if False:
        while True:
            i = 10
    '\n    Test running multiple scripts at the same time.\n\n    This is a regression test for issue spyder-ide/spyder#15405\n    '
    dir_a = tmp_path / 'a'
    dir_a.mkdir()
    filename_a = dir_a / 'a.py'
    filename_a.write_text('a = 1')
    dir_b = tmp_path / 'b'
    dir_b.mkdir()
    filename_b = dir_a / 'b.py'
    filename_b.write_text('b = 1')
    filenames = [str(filename_a), str(filename_b)]
    for filename in filenames:
        ipyconsole.run_script(filename=filename, wdir=osp.dirname(filename), current_client=False, clear_variables=True)
    for filename in filenames:
        basename = osp.basename(filename)
        client_name = f'{basename}/A'
        variable_name = basename.split('.')[0]
        client = ipyconsole.get_client_for_file(filename)
        assert client.get_name() == client_name
        sw = client.shellwidget
        qtbot.waitUntil(lambda : sw._prompt_html is not None, timeout=SHELL_TIMEOUT)
        control = client.get_control()
        qtbot.waitUntil(lambda : 'In [2]:' in control.toPlainText(), timeout=SHELL_TIMEOUT)
        assert sw.get_value(variable_name) == 1

@pytest.mark.skipif(not is_anaconda(), reason='Only works with Anaconda')
def test_show_spyder_kernels_error_on_restart(ipyconsole, qtbot):
    if False:
        return 10
    'Test that we show Spyder-kernels error message on restarts.'
    shell = ipyconsole.get_current_shellwidget()
    qtbot.waitUntil(lambda : shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    ipyconsole.set_conf('default', False, section='main_interpreter')
    pyexec = get_list_conda_envs()['conda: base'][0]
    ipyconsole.set_conf('executable', pyexec, section='main_interpreter')
    ipyconsole.restart_kernel()
    info_page = ipyconsole.get_current_client().infowidget.page()
    qtbot.waitUntil(lambda : check_text(info_page, 'The Python environment or installation'), timeout=6000)
    qtbot.wait(500)
    main_widget = ipyconsole.get_widget()
    assert not main_widget.restart_action.isEnabled()
    assert not main_widget.reset_action.isEnabled()
    assert not main_widget.env_action.isEnabled()
    assert not main_widget.syspath_action.isEnabled()
    assert not main_widget.show_time_action.isEnabled()
if __name__ == '__main__':
    pytest.main()