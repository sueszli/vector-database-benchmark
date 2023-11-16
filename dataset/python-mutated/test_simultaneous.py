import sys
from time import sleep
import pytest
import numpy as np
from numpy.testing import assert_allclose
from vispy.app import use_app, Canvas, Timer
from vispy.testing import requires_application, SkipTest, run_tests_if_main, IS_TRAVIS_CI
from vispy.util.ptime import time
from vispy.gloo import gl
from vispy.gloo.util import _screenshot
_win_size = (200, 50)

def _update_process_check(canvas, val, draw=True):
    if False:
        while True:
            i = 10
    'Update, process, and check result'
    if draw:
        canvas.update()
        canvas.app.process_events()
        canvas.app.process_events()
        sleep(0.03)
    canvas._backend._vispy_set_current()
    print('           check %s' % val)
    ss = _screenshot()
    try:
        assert_allclose(ss.shape[:2], _win_size[::-1])
    except Exception:
        print('!!!!!!!!!! FAIL  bad size %s' % list(ss.shape[:2]))
        raise
    goal = val * np.ones(ss.shape)
    try:
        assert_allclose(ss[:, :, :3], goal[:, :, :3], atol=1)
    except Exception:
        print('!!!!!!!!!! FAIL  %s' % np.unique(ss))
        raise

@pytest.mark.xfail(IS_TRAVIS_CI and 'darwin' in sys.platform, reason='Travis OSX causes segmentation fault on this test for an unknown reason.')
@requires_application()
def test_multiple_canvases():
    if False:
        return 10
    'Testing multiple canvases'
    n_check = 3
    app = use_app()
    with Canvas(app=app, size=_win_size, title='same_0') as c0:
        with Canvas(app=app, size=_win_size, title='same_1') as c1:
            ct = [0, 0]

            @c0.events.draw.connect
            def draw0(event):
                if False:
                    print('Hello World!')
                ct[0] += 1
                c0.update()

            @c1.events.draw.connect
            def draw1(event):
                if False:
                    i = 10
                    return i + 15
                ct[1] += 1
                c1.update()
            c0.show()
            c1.show()
            c0.update()
            c1.update()
            timeout = time() + 2.0
            while (ct[0] < n_check or ct[1] < n_check) and time() < timeout:
                app.process_events()
            print((ct, n_check))
            assert n_check <= ct[0] <= n_check + 20
            assert n_check <= ct[1] <= n_check + 20
            global timer_ran
            timer_ran = False

            def on_timer(_):
                if False:
                    print('Hello World!')
                global timer_ran
                timer_ran = True
            t = Timer(0.1, app=app, connect=on_timer, iterations=1, start=True)
            app.process_events()
            sleep(0.5)
            app.process_events()
            app.process_events()
            assert timer_ran
    if app.backend_name.lower() == 'wx':
        raise SkipTest('wx fails test #2')
    kwargs = dict(app=app, autoswap=False, size=_win_size, show=True)
    with Canvas(title='0', **kwargs) as c0:
        with Canvas(title='1', **kwargs) as c1:
            bgcolors = [None] * 2

            @c0.events.draw.connect
            def draw00(event):
                if False:
                    while True:
                        i = 10
                print('  {0:7}: {1}'.format('0', bgcolors[0]))
                if bgcolors[0] is not None:
                    gl.glViewport(0, 0, *list(_win_size))
                    gl.glClearColor(*bgcolors[0])
                    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                    gl.glFinish()

            @c1.events.draw.connect
            def draw11(event):
                if False:
                    while True:
                        i = 10
                print('  {0:7}: {1}'.format('1', bgcolors[1]))
                if bgcolors[1] is not None:
                    gl.glViewport(0, 0, *list(_win_size))
                    gl.glClearColor(*bgcolors[1])
                    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                    gl.glFinish()
            for (ci, canvas) in enumerate((c0, c1)):
                print('draw %s' % canvas.title)
                bgcolors[ci] = [0.5, 0.5, 0.5, 1.0]
                _update_process_check(canvas, 127)
            for (ci, canvas) in enumerate((c0, c1)):
                print('test')
                _update_process_check(canvas, 127, draw=False)
                bgcolors[ci] = [1.0, 1.0, 1.0, 1.0]
                _update_process_check(canvas, 255)
                bgcolors[ci] = [0.25, 0.25, 0.25, 0.25]
                _update_process_check(canvas, 64)
run_tests_if_main()