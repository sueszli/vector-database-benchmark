import importlib
import importlib.util
import inspect
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from PIL import Image
import pytest
import matplotlib as mpl
from matplotlib import _c_internal_utils
from matplotlib.backend_tools import ToolToggleBase
from matplotlib.testing import subprocess_run_helper as _run_helper

class _WaitForStringPopen(subprocess.Popen):
    """
    A Popen that passes flags that allow triggering KeyboardInterrupt.
    """

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        if sys.platform == 'win32':
            kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
        super().__init__(*args, **kwargs, env={**os.environ, 'MPLBACKEND': 'Agg', 'SOURCE_DATE_EPOCH': '0'}, stdout=subprocess.PIPE, universal_newlines=True)

    def wait_for(self, terminator):
        if False:
            i = 10
            return i + 15
        'Read until the terminator is reached.'
        buf = ''
        while True:
            c = self.stdout.read(1)
            if not c:
                raise RuntimeError(f'Subprocess died before emitting expected {terminator!r}')
            buf += c
            if buf.endswith(terminator):
                return

def _get_testable_interactive_backends():
    if False:
        while True:
            i = 10
    envs = []
    for (deps, env) in [*[([qt_api], {'MPLBACKEND': 'qtagg', 'QT_API': qt_api}) for qt_api in ['PyQt6', 'PySide6', 'PyQt5', 'PySide2']], *[([qt_api, 'cairocffi'], {'MPLBACKEND': 'qtcairo', 'QT_API': qt_api}) for qt_api in ['PyQt6', 'PySide6', 'PyQt5', 'PySide2']], *[(['cairo', 'gi'], {'MPLBACKEND': f'gtk{version}{renderer}'}) for version in [3, 4] for renderer in ['agg', 'cairo']], (['tkinter'], {'MPLBACKEND': 'tkagg'}), (['wx'], {'MPLBACKEND': 'wx'}), (['wx'], {'MPLBACKEND': 'wxagg'}), (['matplotlib.backends._macosx'], {'MPLBACKEND': 'macosx'})]:
        reason = None
        missing = [dep for dep in deps if not importlib.util.find_spec(dep)]
        if sys.platform == 'linux' and (not _c_internal_utils.display_is_valid()):
            reason = '$DISPLAY and $WAYLAND_DISPLAY are unset'
        elif missing:
            reason = '{} cannot be imported'.format(', '.join(missing))
        elif env['MPLBACKEND'] == 'macosx' and os.environ.get('TF_BUILD'):
            reason = 'macosx backend fails on Azure'
        elif env['MPLBACKEND'].startswith('gtk'):
            import gi
            version = env['MPLBACKEND'][3]
            repo = gi.Repository.get_default()
            if f'{version}.0' not in repo.enumerate_versions('Gtk'):
                reason = 'no usable GTK bindings'
        marks = []
        if reason:
            marks.append(pytest.mark.skip(reason=f'Skipping {env} because {reason}'))
        elif env['MPLBACKEND'].startswith('wx') and sys.platform == 'darwin':
            marks.append(pytest.mark.xfail(reason='github #16849'))
        elif env['MPLBACKEND'] == 'tkagg' and ('TF_BUILD' in os.environ or 'GITHUB_ACTION' in os.environ) and (sys.platform == 'darwin') and (sys.version_info[:2] < (3, 11)):
            marks.append(pytest.mark.xfail(reason='Tk version mismatch on Azure macOS CI'))
        envs.append(pytest.param({**env, 'BACKEND_DEPS': ','.join(deps)}, marks=marks, id=str(env)))
    return envs

def is_ci_environment():
    if False:
        return 10
    ci_environment_variables = ['CI', 'CONTINUOUS_INTEGRATION', 'TRAVIS', 'CIRCLECI', 'JENKINS', 'GITLAB_CI', 'GITHUB_ACTIONS', 'TEAMCITY_VERSION']
    for env_var in ci_environment_variables:
        if os.getenv(env_var):
            return True
    return False
_test_timeout = 120 if is_ci_environment() else 20

def _test_toolbar_button_la_mode_icon(fig):
    if False:
        print('Hello World!')
    with tempfile.TemporaryDirectory() as tempdir:
        img = Image.new('LA', (26, 26))
        tmp_img_path = os.path.join(tempdir, 'test_la_icon.png')
        img.save(tmp_img_path)

        class CustomTool(ToolToggleBase):
            image = tmp_img_path
            description = ''
        toolmanager = fig.canvas.manager.toolmanager
        toolbar = fig.canvas.manager.toolbar
        toolmanager.add_tool('test', CustomTool)
        toolbar.add_tool('test', 'group')

def _test_interactive_impl():
    if False:
        i = 10
        return i + 15
    import importlib.util
    import io
    import json
    import sys
    import pytest
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from matplotlib.backend_bases import KeyEvent
    mpl.rcParams.update({'webagg.open_in_browser': False, 'webagg.port_retries': 1})
    mpl.rcParams.update(json.loads(sys.argv[1]))
    backend = plt.rcParams['backend'].lower()
    if backend.endswith('agg') and (not backend.startswith(('gtk', 'web'))):
        plt.figure()
        if backend != 'tkagg':
            with pytest.raises(ImportError):
                mpl.use('tkagg', force=True)

        def check_alt_backend(alt_backend):
            if False:
                return 10
            mpl.use(alt_backend, force=True)
            fig = plt.figure()
            assert type(fig.canvas).__module__ == f'matplotlib.backends.backend_{alt_backend}'
            plt.close('all')
        if importlib.util.find_spec('cairocffi'):
            check_alt_backend(backend[:-3] + 'cairo')
        check_alt_backend('svg')
    mpl.use(backend, force=True)
    (fig, ax) = plt.subplots()
    assert type(fig.canvas).__module__ == f'matplotlib.backends.backend_{backend}'
    assert fig.canvas.manager.get_window_title() == 'Figure 1'
    if mpl.rcParams['toolbar'] == 'toolmanager':
        _test_toolbar_button_la_mode_icon(fig)
    ax.plot([0, 1], [2, 3])
    if fig.canvas.toolbar:
        fig.canvas.toolbar.draw_rubberband(None, 1.0, 1, 2.0, 2)
    timer = fig.canvas.new_timer(1.0)
    timer.add_callback(KeyEvent('key_press_event', fig.canvas, 'q')._process)
    fig.canvas.mpl_connect('draw_event', lambda event: timer.start())
    fig.canvas.mpl_connect('close_event', print)
    result = io.BytesIO()
    fig.savefig(result, format='png')
    plt.show()
    plt.pause(0.5)
    result_after = io.BytesIO()
    fig.savefig(result_after, format='png')
    if not backend.startswith('qt5') and sys.platform == 'darwin':
        assert result.getvalue() == result_after.getvalue()

@pytest.mark.parametrize('env', _get_testable_interactive_backends())
@pytest.mark.parametrize('toolbar', ['toolbar2', 'toolmanager'])
@pytest.mark.flaky(reruns=3)
def test_interactive_backend(env, toolbar):
    if False:
        while True:
            i = 10
    if env['MPLBACKEND'] == 'macosx':
        if toolbar == 'toolmanager':
            pytest.skip('toolmanager is not implemented for macosx.')
    if env['MPLBACKEND'] == 'wx':
        pytest.skip('wx backend is deprecated; tests failed on appveyor')
    try:
        proc = _run_helper(_test_interactive_impl, json.dumps({'toolbar': toolbar}), timeout=_test_timeout, extra_env=env)
    except subprocess.CalledProcessError as err:
        pytest.fail('Subprocess failed to test intended behavior\n' + str(err.stderr))
    assert proc.stdout.count('CloseEvent') == 1

def _test_thread_impl():
    if False:
        return 10
    from concurrent.futures import ThreadPoolExecutor
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    mpl.rcParams.update({'webagg.open_in_browser': False, 'webagg.port_retries': 1})
    (fig, ax) = plt.subplots()
    plt.pause(0.5)
    future = ThreadPoolExecutor().submit(ax.plot, [1, 3, 6])
    future.result()
    fig.canvas.mpl_connect('close_event', print)
    future = ThreadPoolExecutor().submit(fig.canvas.draw)
    plt.pause(0.5)
    future.result()
    plt.close()
    if plt.rcParams['backend'].startswith('WX'):
        fig.canvas.flush_events()
_thread_safe_backends = _get_testable_interactive_backends()
for param in _thread_safe_backends:
    backend = param.values[0]['MPLBACKEND']
    if 'cairo' in backend:
        param.marks.append(pytest.mark.xfail(raises=subprocess.CalledProcessError))
    elif backend == 'wx':
        param.marks.append(pytest.mark.xfail(raises=subprocess.CalledProcessError))
    elif backend == 'macosx':
        from packaging.version import parse
        mac_ver = platform.mac_ver()[0]
        if mac_ver and parse(mac_ver) < parse('10.16'):
            param.marks.append(pytest.mark.xfail(raises=subprocess.TimeoutExpired, strict=True))
    elif param.values[0].get('QT_API') == 'PySide2':
        param.marks.append(pytest.mark.xfail(raises=subprocess.CalledProcessError))
    elif backend == 'tkagg' and platform.python_implementation() != 'CPython':
        param.marks.append(pytest.mark.xfail(reason='PyPy does not support Tkinter threading: https://foss.heptapod.net/pypy/pypy/-/issues/1929', strict=True))
    elif backend == 'tkagg' and ('TF_BUILD' in os.environ or 'GITHUB_ACTION' in os.environ) and (sys.platform == 'darwin') and (sys.version_info[:2] < (3, 11)):
        param.marks.append(pytest.mark.xfail('Tk version mismatch on Azure macOS CI'))

@pytest.mark.parametrize('env', _thread_safe_backends)
@pytest.mark.flaky(reruns=3)
def test_interactive_thread_safety(env):
    if False:
        return 10
    proc = _run_helper(_test_thread_impl, timeout=_test_timeout, extra_env=env)
    assert proc.stdout.count('CloseEvent') == 1

def _impl_test_lazy_auto_backend_selection():
    if False:
        return 10
    import matplotlib
    import matplotlib.pyplot as plt
    bk = matplotlib.rcParams._get('backend')
    assert not isinstance(bk, str)
    assert plt._backend_mod is None
    plt.plot(5)
    assert plt._backend_mod is not None
    bk = matplotlib.rcParams._get('backend')
    assert isinstance(bk, str)

def test_lazy_auto_backend_selection():
    if False:
        while True:
            i = 10
    _run_helper(_impl_test_lazy_auto_backend_selection, timeout=_test_timeout)

def _implqt5agg():
    if False:
        i = 10
        return i + 15
    import matplotlib.backends.backend_qt5agg
    import sys
    assert 'PyQt6' not in sys.modules
    assert 'pyside6' not in sys.modules
    assert 'PyQt5' in sys.modules or 'pyside2' in sys.modules

def _implcairo():
    if False:
        while True:
            i = 10
    import matplotlib.backends.backend_qt5cairo
    import sys
    assert 'PyQt6' not in sys.modules
    assert 'pyside6' not in sys.modules
    assert 'PyQt5' in sys.modules or 'pyside2' in sys.modules

def _implcore():
    if False:
        print('Hello World!')
    import matplotlib.backends.backend_qt5
    import sys
    assert 'PyQt6' not in sys.modules
    assert 'pyside6' not in sys.modules
    assert 'PyQt5' in sys.modules or 'pyside2' in sys.modules

def test_qt5backends_uses_qt5():
    if False:
        return 10
    qt5_bindings = [dep for dep in ['PyQt5', 'pyside2'] if importlib.util.find_spec(dep) is not None]
    qt6_bindings = [dep for dep in ['PyQt6', 'pyside6'] if importlib.util.find_spec(dep) is not None]
    if len(qt5_bindings) == 0 or len(qt6_bindings) == 0:
        pytest.skip('need both QT6 and QT5 bindings')
    _run_helper(_implqt5agg, timeout=_test_timeout)
    if importlib.util.find_spec('pycairo') is not None:
        _run_helper(_implcairo, timeout=_test_timeout)
    _run_helper(_implcore, timeout=_test_timeout)

def _impl_missing():
    if False:
        for i in range(10):
            print('nop')
    import sys
    sys.modules['PyQt6'] = None
    sys.modules['PyQt5'] = None
    sys.modules['PySide2'] = None
    sys.modules['PySide6'] = None
    import matplotlib.pyplot as plt
    with pytest.raises(ImportError, match='Failed to import any of the following Qt'):
        plt.switch_backend('qtagg')
    with pytest.raises(ImportError, match='^(?:(?!(PySide6|PyQt6)).)*$'):
        plt.switch_backend('qt5agg')

def test_qt_missing():
    if False:
        while True:
            i = 10
    _run_helper(_impl_missing, timeout=_test_timeout)

def _impl_test_cross_Qt_imports():
    if False:
        return 10
    import sys
    import importlib
    import pytest
    (_, host_binding, mpl_binding) = sys.argv
    importlib.import_module(f'{mpl_binding}.QtCore')
    mpl_binding_qwidgets = importlib.import_module(f'{mpl_binding}.QtWidgets')
    import matplotlib.backends.backend_qt
    host_qwidgets = importlib.import_module(f'{host_binding}.QtWidgets')
    host_app = host_qwidgets.QApplication(['mpl testing'])
    with pytest.warns(UserWarning, match='Mixing Qt major'):
        matplotlib.backends.backend_qt._create_qApp()

def test_cross_Qt_imports():
    if False:
        print('Hello World!')
    qt5_bindings = [dep for dep in ['PyQt5', 'PySide2'] if importlib.util.find_spec(dep) is not None]
    qt6_bindings = [dep for dep in ['PyQt6', 'PySide6'] if importlib.util.find_spec(dep) is not None]
    if len(qt5_bindings) == 0 or len(qt6_bindings) == 0:
        pytest.skip('need both QT6 and QT5 bindings')
    for qt5 in qt5_bindings:
        for qt6 in qt6_bindings:
            for pair in ([qt5, qt6], [qt6, qt5]):
                try:
                    _run_helper(_impl_test_cross_Qt_imports, *pair, timeout=_test_timeout)
                except subprocess.CalledProcessError as ex:
                    if ex.returncode == -signal.SIGSEGV:
                        continue
                    elif ex.returncode == -signal.SIGABRT:
                        continue
                    raise

@pytest.mark.skipif('TF_BUILD' in os.environ, reason='this test fails an azure for unknown reasons')
@pytest.mark.skipif(sys.platform == 'win32', reason='Cannot send SIGINT on Windows.')
def test_webagg():
    if False:
        for i in range(10):
            print('nop')
    pytest.importorskip('tornado')
    proc = subprocess.Popen([sys.executable, '-c', inspect.getsource(_test_interactive_impl) + '\n_test_interactive_impl()', '{}'], env={**os.environ, 'MPLBACKEND': 'webagg', 'SOURCE_DATE_EPOCH': '0'})
    url = f"http://{mpl.rcParams['webagg.address']}:{mpl.rcParams['webagg.port']}"
    timeout = time.perf_counter() + _test_timeout
    try:
        while True:
            try:
                retcode = proc.poll()
                assert retcode is None
                conn = urllib.request.urlopen(url)
                break
            except urllib.error.URLError:
                if time.perf_counter() > timeout:
                    pytest.fail('Failed to connect to the webagg server.')
                else:
                    continue
        conn.close()
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=_test_timeout) == 0
    finally:
        if proc.poll() is None:
            proc.kill()

def _lazy_headless():
    if False:
        return 10
    import os
    import sys
    (backend, deps) = sys.argv[1:]
    deps = deps.split(',')
    os.environ.pop('DISPLAY', None)
    os.environ.pop('WAYLAND_DISPLAY', None)
    for dep in deps:
        assert dep not in sys.modules
    import matplotlib.pyplot as plt
    assert plt.get_backend() == 'agg'
    for dep in deps:
        assert dep not in sys.modules
    for dep in deps:
        importlib.import_module(dep)
        assert dep in sys.modules
    try:
        plt.switch_backend(backend)
    except ImportError:
        pass
    else:
        sys.exit(1)

@pytest.mark.skipif(sys.platform != 'linux', reason='this a linux-only test')
@pytest.mark.parametrize('env', _get_testable_interactive_backends())
def test_lazy_linux_headless(env):
    if False:
        while True:
            i = 10
    proc = _run_helper(_lazy_headless, env.pop('MPLBACKEND'), env.pop('BACKEND_DEPS'), timeout=_test_timeout, extra_env={**env, 'DISPLAY': '', 'WAYLAND_DISPLAY': ''})

def _test_number_of_draws_script():
    if False:
        while True:
            i = 10
    import matplotlib.pyplot as plt
    (fig, ax) = plt.subplots()
    (ln,) = ax.plot([0, 1], [1, 2], animated=True)
    plt.show(block=False)
    plt.pause(0.3)
    fig.canvas.mpl_connect('draw_event', print)
    bg = fig.canvas.copy_from_bbox(fig.bbox)
    ax.draw_artist(ln)
    fig.canvas.blit(fig.bbox)
    for j in range(10):
        fig.canvas.restore_region(bg)
        (ln,) = ax.plot([0, 1], [1, 2])
        ax.draw_artist(ln)
        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()
    plt.pause(0.1)
_blit_backends = _get_testable_interactive_backends()
for param in _blit_backends:
    backend = param.values[0]['MPLBACKEND']
    if backend == 'gtk3cairo':
        param.marks.append(pytest.mark.skip('gtk3cairo does not support blitting'))
    elif backend == 'gtk4cairo':
        param.marks.append(pytest.mark.skip('gtk4cairo does not support blitting'))
    elif backend == 'wx':
        param.marks.append(pytest.mark.skip('wx does not support blitting'))
    elif backend == 'tkagg' and ('TF_BUILD' in os.environ or 'GITHUB_ACTION' in os.environ) and (sys.platform == 'darwin') and (sys.version_info[:2] < (3, 11)):
        param.marks.append(pytest.mark.xfail('Tk version mismatch on Azure macOS CI'))

@pytest.mark.parametrize('env', _blit_backends)
@pytest.mark.flaky(reruns=4)
def test_blitting_events(env):
    if False:
        while True:
            i = 10
    proc = _run_helper(_test_number_of_draws_script, timeout=_test_timeout, extra_env=env)
    ndraws = proc.stdout.count('DrawEvent')
    assert 0 < ndraws < 5

def _test_figure_leak():
    if False:
        return 10
    import gc
    import sys
    import psutil
    from matplotlib import pyplot as plt
    t = float(sys.argv[1])
    p = psutil.Process()
    for _ in range(2):
        fig = plt.figure()
        if t:
            plt.pause(t)
        plt.close(fig)
    mem = p.memory_info().rss
    gc.collect()
    for _ in range(5):
        fig = plt.figure()
        if t:
            plt.pause(t)
        plt.close(fig)
        gc.collect()
    growth = p.memory_info().rss - mem
    print(growth)

@pytest.mark.skipif(sys.platform == 'win32', reason='appveyor tests fail; gh-22988 suggests reworking')
@pytest.mark.parametrize('env', _get_testable_interactive_backends())
@pytest.mark.parametrize('time_mem', [(0.0, 2000000), (0.1, 30000000)])
def test_figure_leak_20490(env, time_mem, request):
    if False:
        print('Hello World!')
    pytest.importorskip('psutil', reason='psutil needed to run this test')
    (pause_time, acceptable_memory_leakage) = time_mem
    if env['MPLBACKEND'] == 'wx':
        pytest.skip('wx backend is deprecated; tests failed on appveyor')
    if env['MPLBACKEND'] == 'macosx':
        request.node.add_marker(pytest.mark.xfail(reason='macosx backend is leaky'))
    if env['MPLBACKEND'] == 'tkagg' and sys.platform == 'darwin':
        acceptable_memory_leakage += 11000000
    result = _run_helper(_test_figure_leak, str(pause_time), timeout=_test_timeout, extra_env=env)
    growth = int(result.stdout)
    assert growth <= acceptable_memory_leakage

def _impl_test_interactive_timers():
    if False:
        print('Hello World!')
    import os
    from unittest.mock import Mock
    import matplotlib.pyplot as plt
    pause_time = 2 if os.getenv('CI') else 0.5
    fig = plt.figure()
    plt.pause(pause_time)
    timer = fig.canvas.new_timer(0.1)
    mock = Mock()
    timer.add_callback(mock)
    timer.start()
    plt.pause(pause_time)
    timer.stop()
    assert mock.call_count > 1
    mock.call_count = 0
    timer.single_shot = True
    timer.start()
    plt.pause(pause_time)
    assert mock.call_count == 1
    timer.start()
    plt.pause(pause_time)
    assert mock.call_count == 2
    plt.close('all')

@pytest.mark.parametrize('env', _get_testable_interactive_backends())
def test_interactive_timers(env):
    if False:
        for i in range(10):
            print('nop')
    if env['MPLBACKEND'] == 'gtk3cairo' and os.getenv('CI'):
        pytest.skip('gtk3cairo timers do not work in remote CI')
    if env['MPLBACKEND'] == 'wx':
        pytest.skip('wx backend is deprecated; tests failed on appveyor')
    _run_helper(_impl_test_interactive_timers, timeout=_test_timeout, extra_env=env)

def _test_sigint_impl(backend, target_name, kwargs):
    if False:
        i = 10
        return i + 15
    import sys
    import matplotlib.pyplot as plt
    import os
    import threading
    plt.switch_backend(backend)

    def interrupter():
        if False:
            for i in range(10):
                print('nop')
        if sys.platform == 'win32':
            import win32api
            win32api.GenerateConsoleCtrlEvent(0, 0)
        else:
            import signal
            os.kill(os.getpid(), signal.SIGINT)
    target = getattr(plt, target_name)
    timer = threading.Timer(1, interrupter)
    fig = plt.figure()
    fig.canvas.mpl_connect('draw_event', lambda *args: print('DRAW', flush=True))
    fig.canvas.mpl_connect('draw_event', lambda *args: timer.start())
    try:
        target(**kwargs)
    except KeyboardInterrupt:
        print('SUCCESS', flush=True)

@pytest.mark.parametrize('env', _get_testable_interactive_backends())
@pytest.mark.parametrize('target, kwargs', [('show', {'block': True}), ('pause', {'interval': 10})])
def test_sigint(env, target, kwargs):
    if False:
        return 10
    backend = env.get('MPLBACKEND')
    if not backend.startswith(('qt', 'macosx')):
        pytest.skip('SIGINT currently only tested on qt and macosx')
    proc = _WaitForStringPopen([sys.executable, '-c', inspect.getsource(_test_sigint_impl) + f'\n_test_sigint_impl({backend!r}, {target!r}, {kwargs!r})'])
    try:
        proc.wait_for('DRAW')
        (stdout, _) = proc.communicate(timeout=_test_timeout)
    except Exception:
        proc.kill()
        (stdout, _) = proc.communicate()
        raise
    assert 'SUCCESS' in stdout

def _test_other_signal_before_sigint_impl(backend, target_name, kwargs):
    if False:
        return 10
    import signal
    import matplotlib.pyplot as plt
    plt.switch_backend(backend)
    target = getattr(plt, target_name)
    fig = plt.figure()
    fig.canvas.mpl_connect('draw_event', lambda *args: print('DRAW', flush=True))
    timer = fig.canvas.new_timer(interval=1)
    timer.single_shot = True
    timer.add_callback(print, 'SIGUSR1', flush=True)

    def custom_signal_handler(signum, frame):
        if False:
            print('Hello World!')
        timer.start()
    signal.signal(signal.SIGUSR1, custom_signal_handler)
    try:
        target(**kwargs)
    except KeyboardInterrupt:
        print('SUCCESS', flush=True)

@pytest.mark.skipif(sys.platform == 'win32', reason='No other signal available to send on Windows')
@pytest.mark.parametrize('env', _get_testable_interactive_backends())
@pytest.mark.parametrize('target, kwargs', [('show', {'block': True}), ('pause', {'interval': 10})])
def test_other_signal_before_sigint(env, target, kwargs, request):
    if False:
        while True:
            i = 10
    backend = env.get('MPLBACKEND')
    if not backend.startswith(('qt', 'macosx')):
        pytest.skip('SIGINT currently only tested on qt and macosx')
    if backend == 'macosx':
        request.node.add_marker(pytest.mark.xfail(reason='macosx backend is buggy'))
    proc = _WaitForStringPopen([sys.executable, '-c', inspect.getsource(_test_other_signal_before_sigint_impl) + f'\n_test_other_signal_before_sigint_impl({backend!r}, {target!r}, {kwargs!r})'])
    try:
        proc.wait_for('DRAW')
        os.kill(proc.pid, signal.SIGUSR1)
        proc.wait_for('SIGUSR1')
        os.kill(proc.pid, signal.SIGINT)
        (stdout, _) = proc.communicate(timeout=_test_timeout)
    except Exception:
        proc.kill()
        (stdout, _) = proc.communicate()
        raise
    print(stdout)
    assert 'SUCCESS' in stdout