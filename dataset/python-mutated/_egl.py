"""vispy headless backend for egl."""
from __future__ import division
import atexit
from time import sleep
from ..base import BaseApplicationBackend, BaseCanvasBackend, BaseTimerBackend
from ...util.ptime import time
from ... import config
try:
    import os
    x11_dpy = os.getenv('DISPLAY')
    if x11_dpy is not None:
        os.unsetenv('DISPLAY')
    from ...ext import egl
    _EGL_DISPLAY = egl.eglGetDisplay()
    if x11_dpy is not None:
        os.environ['DISPLAY'] = x11_dpy
    egl.eglInitialize(_EGL_DISPLAY)
    version = [egl.eglQueryString(_EGL_DISPLAY, x) for x in [egl.EGL_VERSION, egl.EGL_VENDOR, egl.EGL_CLIENT_APIS]]
    version = [v.decode('utf-8') for v in version]
    version = version[0] + ' ' + version[1] + ': ' + version[2].strip()
    atexit.register(egl.eglTerminate, _EGL_DISPLAY)
except Exception as exp:
    (available, testable, why_not, which) = (False, False, str(exp), None)
else:
    (available, testable, why_not) = (True, False, '')
    which = 'EGL ' + str(version)
_VP_EGL_ALL_WINDOWS = []

def _get_egl_windows():
    if False:
        while True:
            i = 10
    wins = list()
    for win in _VP_EGL_ALL_WINDOWS:
        if isinstance(win, CanvasBackend):
            wins.append(win)
    return wins
capability = dict(title=True, size=True, position=True, show=True, vsync=False, resizable=True, decorate=False, fullscreen=False, context=False, multi_window=True, scroll=False, parent=False, always_on_top=False)

class ApplicationBackend(BaseApplicationBackend):

    def __init__(self):
        if False:
            while True:
                i = 10
        BaseApplicationBackend.__init__(self)
        self._timers = list()

    def _add_timer(self, timer):
        if False:
            print('Hello World!')
        if timer not in self._timers:
            self._timers.append(timer)

    def _vispy_get_backend_name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'egl'

    def _vispy_process_events(self):
        if False:
            i = 10
            return i + 15
        for timer in self._timers:
            timer._tick()
        wins = _get_egl_windows()
        for win in wins:
            if win._needs_draw:
                win._needs_draw = False
                win._on_draw()

    def _vispy_run(self):
        if False:
            for i in range(10):
                print('nop')
        wins = _get_egl_windows()
        while all((w._surface is not None for w in wins)):
            self._vispy_process_events()
        self._vispy_quit()

    def _vispy_quit(self):
        if False:
            return 10
        wins = _get_egl_windows()
        for win in wins:
            win._vispy_close()
        for timer in self._timers:
            timer._vispy_stop()
        self._timers = []

    def _vispy_get_native_app(self):
        if False:
            return 10
        return egl

class CanvasBackend(BaseCanvasBackend):
    """EGL backend for Canvas abstract class."""

    def __init__(self, vispy_canvas, **kwargs):
        if False:
            while True:
                i = 10
        BaseCanvasBackend.__init__(self, vispy_canvas)
        p = self._process_backend_kwargs(kwargs)
        self._initialized = False
        p.context.shared.add_ref('egl', self)
        if p.context.shared.ref is self:
            attribs = [egl.EGL_RED_SIZE, 8, egl.EGL_BLUE_SIZE, 8, egl.EGL_GREEN_SIZE, 8, egl.EGL_ALPHA_SIZE, 8, egl.EGL_COLOR_BUFFER_TYPE, egl.EGL_RGB_BUFFER, egl.EGL_SURFACE_TYPE, egl.EGL_PBUFFER_BIT]
            api = None
            if 'es' in config['gl_backend']:
                attribs.extend([egl.EGL_RENDERABLE_TYPE, egl.EGL_OPENGL_ES2_BIT])
                api = egl.EGL_OPENGL_ES_API
            else:
                attribs.extend([egl.EGL_RENDERABLE_TYPE, egl.EGL_OPENGL_BIT])
                api = egl.EGL_OPENGL_API
            self._native_config = egl.eglChooseConfig(_EGL_DISPLAY, attribs)[0]
            egl.eglBindAPI(api)
            self._native_context = egl.eglCreateContext(_EGL_DISPLAY, self._native_config, None)
        else:
            self._native_config = p.context.shared.ref._native_config
            self._native_context = p.context.shared.ref._native_context
        self._surface = None
        self._vispy_set_size(*p.size)
        _VP_EGL_ALL_WINDOWS.append(self)
        self._initialized = True
        self._vispy_canvas.set_current()
        self._vispy_canvas.events.initialize()

    def _destroy_surface(self):
        if False:
            i = 10
            return i + 15
        if self._surface is not None:
            egl.eglDestroySurface(_EGL_DISPLAY, self._surface)
            self._surface = None

    def _vispy_set_size(self, w, h):
        if False:
            print('Hello World!')
        if self._surface is not None:
            self._destroy_surface()
        attrib_list = (egl.EGL_WIDTH, w, egl.EGL_HEIGHT, h)
        self._surface = egl.eglCreatePbufferSurface(_EGL_DISPLAY, self._native_config, attrib_list)
        if self._surface == egl.EGL_NO_SURFACE:
            raise RuntimeError('Could not create rendering surface')
        self._size = (w, h)
        self._vispy_update()

    def _vispy_warmup(self):
        if False:
            return 10
        etime = time() + 0.25
        while time() < etime:
            sleep(0.01)
            self._vispy_canvas.set_current()
            self._vispy_canvas.app.process_events()

    def _vispy_set_current(self):
        if False:
            while True:
                i = 10
        if self._surface is None:
            return
        egl.eglMakeCurrent(_EGL_DISPLAY, self._surface, self._surface, self._native_context)

    def _vispy_swap_buffers(self):
        if False:
            return 10
        if self._surface is None:
            return
        egl.eglSwapBuffers(_EGL_DISPLAY, self._surface)

    def _vispy_set_title(self, title):
        if False:
            print('Hello World!')
        pass

    def _vispy_set_position(self, x, y):
        if False:
            while True:
                i = 10
        pass

    def _vispy_set_visible(self, visible):
        if False:
            for i in range(10):
                print('nop')
        pass

    def _vispy_update(self):
        if False:
            i = 10
            return i + 15
        self._needs_draw = True

    def _vispy_close(self):
        if False:
            print('Hello World!')
        self._destroy_surface()

    def _vispy_get_size(self):
        if False:
            i = 10
            return i + 15
        if self._surface is None:
            return
        return self._size

    def _vispy_get_position(self):
        if False:
            return 10
        return (0, 0)

    def _on_draw(self, _id=None):
        if False:
            while True:
                i = 10
        if self._vispy_canvas is None or self._surface is None:
            return
        self._vispy_canvas.set_current()
        self._vispy_canvas.events.draw(region=None)

class TimerBackend(BaseTimerBackend):

    def __init__(self, vispy_timer):
        if False:
            print('Hello World!')
        BaseTimerBackend.__init__(self, vispy_timer)
        vispy_timer._app._backend._add_timer(self)
        self._vispy_stop()

    def _vispy_start(self, interval):
        if False:
            for i in range(10):
                print('nop')
        self._interval = interval
        self._next_time = time() + self._interval

    def _vispy_stop(self):
        if False:
            for i in range(10):
                print('nop')
        self._next_time = float('inf')

    def _tick(self):
        if False:
            print('Hello World!')
        if time() >= self._next_time:
            self._vispy_timer._timeout()
            self._next_time = time() + self._interval