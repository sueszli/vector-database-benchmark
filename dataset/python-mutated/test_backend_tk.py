import functools
import importlib
import os
import platform
import subprocess
import sys
import pytest
from matplotlib import _c_internal_utils
from matplotlib.testing import subprocess_run_helper
_test_timeout = 60

def _isolated_tk_test(success_count, func=None):
    if False:
        while True:
            i = 10
    '\n    A decorator to run *func* in a subprocess and assert that it prints\n    "success" *success_count* times and nothing on stderr.\n\n    TkAgg tests seem to have interactions between tests, so isolate each test\n    in a subprocess. See GH#18261\n    '
    if func is None:
        return functools.partial(_isolated_tk_test, success_count)
    if 'MPL_TEST_ESCAPE_HATCH' in os.environ:
        return func

    @pytest.mark.skipif(not importlib.util.find_spec('tkinter'), reason='missing tkinter')
    @pytest.mark.skipif(sys.platform == 'linux' and (not _c_internal_utils.display_is_valid()), reason='$DISPLAY and $WAYLAND_DISPLAY are unset')
    @pytest.mark.xfail(('TF_BUILD' in os.environ or 'GITHUB_ACTION' in os.environ) and sys.platform == 'darwin' and (sys.version_info[:2] < (3, 11)), reason='Tk version mismatch on Azure macOS CI')
    @functools.wraps(func)
    def test_func():
        if False:
            for i in range(10):
                print('nop')
        pytest.importorskip('tkinter')
        try:
            proc = subprocess_run_helper(func, timeout=_test_timeout, extra_env=dict(MPLBACKEND='TkAgg', MPL_TEST_ESCAPE_HATCH='1'))
        except subprocess.TimeoutExpired:
            pytest.fail('Subprocess timed out')
        except subprocess.CalledProcessError as e:
            pytest.fail('Subprocess failed to test intended behavior\n' + str(e.stderr))
        else:
            ignored_lines = ['OpenGL', 'CFMessagePort: bootstrap_register', '/usr/include/servers/bootstrap_defs.h']
            assert not [line for line in proc.stderr.splitlines() if all((msg not in line for msg in ignored_lines))]
            assert proc.stdout.count('success') == success_count
    return test_func

@_isolated_tk_test(success_count=6)
def test_blit():
    if False:
        while True:
            i = 10
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.backends.backend_tkagg
    from matplotlib.backends import _backend_tk, _tkagg
    (fig, ax) = plt.subplots()
    photoimage = fig.canvas._tkphoto
    data = np.ones((4, 4, 4))
    (height, width) = data.shape[:2]
    dataptr = (height, width, data.ctypes.data)
    bad_boxes = ((-1, 2, 0, 2), (2, 0, 0, 2), (1, 6, 0, 2), (0, 2, -1, 2), (0, 2, 2, 0), (0, 2, 1, 6))
    for bad_box in bad_boxes:
        try:
            _tkagg.blit(photoimage.tk.interpaddr(), str(photoimage), dataptr, 0, (0, 1, 2, 3), bad_box)
        except ValueError:
            print('success')
    plt.close(fig)
    _backend_tk.blit(photoimage, data, (0, 1, 2, 3))

@_isolated_tk_test(success_count=1)
def test_figuremanager_preserves_host_mainloop():
    if False:
        for i in range(10):
            print('nop')
    import tkinter
    import matplotlib.pyplot as plt
    success = []

    def do_plot():
        if False:
            print('Hello World!')
        plt.figure()
        plt.plot([1, 2], [3, 5])
        plt.close()
        root.after(0, legitimate_quit)

    def legitimate_quit():
        if False:
            for i in range(10):
                print('nop')
        root.quit()
        success.append(True)
    root = tkinter.Tk()
    root.after(0, do_plot)
    root.mainloop()
    if success:
        print('success')

@pytest.mark.skipif(platform.python_implementation() != 'CPython', reason='PyPy does not support Tkinter threading: https://foss.heptapod.net/pypy/pypy/-/issues/1929')
@pytest.mark.flaky(reruns=3)
@_isolated_tk_test(success_count=1)
def test_figuremanager_cleans_own_mainloop():
    if False:
        return 10
    import tkinter
    import time
    import matplotlib.pyplot as plt
    import threading
    from matplotlib.cbook import _get_running_interactive_framework
    root = tkinter.Tk()
    plt.plot([1, 2, 3], [1, 2, 5])

    def target():
        if False:
            while True:
                i = 10
        while not 'tk' == _get_running_interactive_framework():
            time.sleep(0.01)
        plt.close()
        if show_finished_event.wait():
            print('success')
    show_finished_event = threading.Event()
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    plt.show(block=True)
    show_finished_event.set()
    thread.join()

@pytest.mark.flaky(reruns=3)
@_isolated_tk_test(success_count=0)
def test_never_update():
    if False:
        for i in range(10):
            print('nop')
    import tkinter
    del tkinter.Misc.update
    del tkinter.Misc.update_idletasks
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.show(block=False)
    plt.draw()
    fig.canvas.toolbar.configure_subplots()
    fig.canvas.get_tk_widget().after(100, plt.close, fig)
    plt.show(block=True)

@_isolated_tk_test(success_count=2)
def test_missing_back_button():
    if False:
        while True:
            i = 10
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

    class Toolbar(NavigationToolbar2Tk):
        toolitems = [t for t in NavigationToolbar2Tk.toolitems if t[0] in ('Home', 'Pan', 'Zoom')]
    fig = plt.figure()
    print('success')
    Toolbar(fig.canvas, fig.canvas.manager.window)
    print('success')

@_isolated_tk_test(success_count=1)
def test_canvas_focus():
    if False:
        while True:
            i = 10
    import tkinter as tk
    import matplotlib.pyplot as plt
    success = []

    def check_focus():
        if False:
            print('Hello World!')
        tkcanvas = fig.canvas.get_tk_widget()
        if not tkcanvas.winfo_viewable():
            tkcanvas.wait_visibility()
        if tkcanvas.focus_lastfor() == tkcanvas:
            success.append(True)
        plt.close()
        root.destroy()
    root = tk.Tk()
    fig = plt.figure()
    plt.plot([1, 2, 3])
    root.after(0, plt.show)
    root.after(100, check_focus)
    root.mainloop()
    if success:
        print('success')

@_isolated_tk_test(success_count=2)
def test_embedding():
    if False:
        return 10
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.backend_bases import key_press_handler
    from matplotlib.figure import Figure
    root = tk.Tk()

    def test_figure(master):
        if False:
            return 10
        fig = Figure()
        ax = fig.add_subplot()
        ax.plot([1, 2, 3])
        canvas = FigureCanvasTkAgg(fig, master=master)
        canvas.draw()
        canvas.mpl_connect('key_press_event', key_press_handler)
        canvas.get_tk_widget().pack(expand=True, fill='both')
        toolbar = NavigationToolbar2Tk(canvas, master, pack_toolbar=False)
        toolbar.pack(expand=True, fill='x')
        canvas.get_tk_widget().forget()
        toolbar.forget()
    test_figure(root)
    print('success')
    root.tk_setPalette(background='sky blue', selectColor='midnight blue', foreground='white')
    test_figure(root)
    print('success')