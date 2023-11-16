"""OSMesa backend for offscreen rendering on Linux/Unix."""
from __future__ import division
from ...util.ptime import time
from ..base import BaseApplicationBackend, BaseCanvasBackend, BaseTimerBackend
from ...gloo import gl
from time import sleep
try:
    from ...ext import osmesa
except Exception as exp:
    (available, testable, why_not, which) = (False, False, str(exp), None)
else:
    (available, testable, why_not, which) = (True, True, None, 'OSMesa')
capability = dict(title=True, size=True, position=False, show=True, vsync=False, resizable=False, decorate=True, fullscreen=False, context=True, multi_window=True, scroll=False, parent=False, always_on_top=False)
_VP_OSMESA_ALL_WINDOWS = []

def _get_osmesa_windows():
    if False:
        return 10
    return [win for win in _VP_OSMESA_ALL_WINDOWS if isinstance(win, CanvasBackend)]

class ApplicationBackend(BaseApplicationBackend):

    def __init__(self):
        if False:
            print('Hello World!')
        BaseApplicationBackend.__init__(self)
        self._timers = list()

    def _add_timer(self, timer):
        if False:
            print('Hello World!')
        if timer not in self._timers:
            self._timers.append(timer)

    def _vispy_get_backend_name(self):
        if False:
            print('Hello World!')
        return 'osmesa'

    def _vispy_process_events(self):
        if False:
            i = 10
            return i + 15
        for timer in self._timers:
            timer._tick()
        wins = _get_osmesa_windows()
        for win in wins:
            if win._needs_draw:
                win._needs_draw = False
                win._on_draw()

    def _vispy_run(self):
        if False:
            while True:
                i = 10
        wins = _get_osmesa_windows()
        while not all((w.closed for w in wins)):
            self._vispy_process_events()
        self._vispy_quit()

    def _vispy_quit(self):
        if False:
            while True:
                i = 10
        wins = _get_osmesa_windows()
        for win in wins:
            win._vispy_close()
        for timer in self._timers:
            timer._vispy_stop()
        self._timers = []

    def _vispy_get_native_app(self):
        if False:
            i = 10
            return i + 15
        return osmesa

class OSMesaContext(object):
    """
    A wrapper around an OSMesa context that destroy the context when
    garbage collected
    """

    def __init__(self):
        if False:
            return 10
        self.context = osmesa.OSMesaCreateContext()

    def make_current(self, pixels, width, height):
        if False:
            i = 10
            return i + 15
        return osmesa.OSMesaMakeCurrent(self.context, pixels, width, height)

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        osmesa.OSMesaDestroyContext(self.context)

class CanvasBackend(BaseCanvasBackend):
    """OSMesa backend for Canvas"""

    def __init__(self, vispy_canvas, **kwargs):
        if False:
            return 10
        BaseCanvasBackend.__init__(self, vispy_canvas)
        p = self._process_backend_kwargs(kwargs)
        p.context.shared.add_ref('osmesa', self)
        if p.context.shared.ref is self:
            self._native_context = OSMesaContext()
        else:
            self._native_context = p.context.shared.ref._native_context
        self._closed = False
        self._pixels = None
        self._vispy_set_size(*p.size)
        _VP_OSMESA_ALL_WINDOWS.append(self)
        self._vispy_canvas.set_current()
        self._vispy_canvas.events.initialize()

    def _vispy_set_current(self):
        if False:
            return 10
        if self._native_context is None:
            raise RuntimeError('Native context is None')
        if self._pixels is None:
            raise RuntimeError('Pixel buffer has already been deleted')
        ok = self._native_context.make_current(self._pixels, self._size[0], self._size[1])
        if not ok:
            raise RuntimeError('Failed attaching OSMesa rendering buffer')

    def _vispy_swap_buffers(self):
        if False:
            while True:
                i = 10
        if self._pixels is None:
            raise RuntimeError('No pixel buffer')
        gl.glFinish()

    def _vispy_set_title(self, title):
        if False:
            i = 10
            return i + 15
        pass

    def _vispy_set_size(self, w, h):
        if False:
            print('Hello World!')
        self._pixels = osmesa.allocate_pixels_buffer(w, h)
        self._size = (w, h)
        self._vispy_canvas.events.resize(size=(w, h))
        self._vispy_set_current()
        self._vispy_update()

    def _vispy_set_position(self, x, y):
        if False:
            while True:
                i = 10
        pass

    def _vispy_set_visible(self, visible):
        if False:
            print('Hello World!')
        if visible:
            self._vispy_set_current()
            self._vispy_update()

    def _vispy_set_fullscreen(self, fullscreen):
        if False:
            while True:
                i = 10
        pass

    def _vispy_update(self):
        if False:
            i = 10
            return i + 15
        self._needs_draw = True

    def _vispy_close(self):
        if False:
            return 10
        if self.closed:
            return
        self._closed = True
        return

    def _vispy_warmup(self):
        if False:
            while True:
                i = 10
        etime = time() + 0.1
        while time() < etime:
            sleep(0.01)
            self._vispy_canvas.set_current()
            self._vispy_canvas.app.process_events()

    def _vispy_get_size(self):
        if False:
            while True:
                i = 10
        if self._pixels is None:
            return
        return self._size

    @property
    def closed(self):
        if False:
            return 10
        return self._closed

    def _vispy_get_position(self):
        if False:
            for i in range(10):
                print('nop')
        return (0, 0)

    def _vispy_get_fullscreen(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def _on_draw(self):
        if False:
            i = 10
            return i + 15
        if self._vispy_canvas is None or self._pixels is None:
            raise RuntimeError('draw with no canvas or pixels attached')
            return
        self._vispy_set_current()
        self._vispy_canvas.events.draw(region=None)

class TimerBackend(BaseTimerBackend):

    def __init__(self, vispy_timer):
        if False:
            return 10
        BaseTimerBackend.__init__(self, vispy_timer)
        vispy_timer._app._backend._add_timer(self)
        self._vispy_stop()

    def _vispy_start(self, interval):
        if False:
            print('Hello World!')
        self._interval = interval
        self._next_time = time() + self._interval

    def _vispy_stop(self):
        if False:
            return 10
        self._next_time = float('inf')

    def _tick(self):
        if False:
            while True:
                i = 10
        if time() > self._next_time:
            self._vispy_timer._timeout()
            self._next_time = time() + self._interval