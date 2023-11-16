import asyncio
from ..base import BaseApplicationBackend, BaseCanvasBackend, BaseTimerBackend
from ...app import Timer
from ...util import keys
from ._offscreen_util import OffscreenContext, FrameBufferHelper
try:
    from jupyter_rfb import RemoteFrameBuffer
except Exception:
    RemoteFrameBuffer = object
    _msg = 'The jupyter_rfb backend relies on a the jupyter_rfb library: ``pip install jupyter_rfb``'
    (available, testable, why_not, which) = (False, False, _msg, None)
else:
    (available, testable, why_not) = (True, False, None)
    which = 'jupyter_rfb'
capability = dict(title=False, size=True, position=False, show=False, vsync=False, resizable=True, decorate=False, fullscreen=False, context=False, multi_window=True, scroll=True, parent=False, always_on_top=False)

class ApplicationBackend(BaseApplicationBackend):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()

    def _vispy_get_backend_name(self):
        if False:
            print('Hello World!')
        return 'jupyter_rfb'

    def _vispy_process_events(self):
        if False:
            while True:
                i = 10
        raise RuntimeError('Cannot process events while asyncio event-loop is running.')

    def _vispy_run(self):
        if False:
            i = 10
            return i + 15
        pass

    def _vispy_quit(self):
        if False:
            return 10
        pass

    def _vispy_get_native_app(self):
        if False:
            while True:
                i = 10
        return asyncio

class CanvasBackend(BaseCanvasBackend, RemoteFrameBuffer):
    _double_click_supported = True

    def __init__(self, vispy_canvas, **kwargs):
        if False:
            while True:
                i = 10
        BaseCanvasBackend.__init__(self, vispy_canvas)
        RemoteFrameBuffer.__init__(self)
        self._context = OffscreenContext()
        self._helper = FrameBufferHelper()
        self._loop = asyncio.get_event_loop()
        self._logical_size = (1, 1)
        self._physical_size = (1, 1)
        self._lifecycle = 0
        self._vispy_set_size(*kwargs['size'])
        self.resizable = kwargs['resizable']
        self._vispy_update()

    def handle_event(self, ev):
        if False:
            i = 10
            return i + 15
        type = ev['event_type']
        if type == 'resize':
            (w, h, r) = (ev['width'], ev['height'], ev['pixel_ratio'])
            self._logical_size = (w, h)
            self._physical_size = (int(w * r), int(h * r))
            self._helper.set_physical_size(*self._physical_size)
            self._loop.call_soon(self._emit_resize_event)
            self._vispy_update()
        elif type == 'pointer_down':
            self._vispy_mouse_press(native=ev, pos=(ev['x'], ev['y']), button=ev['button'], modifiers=self._modifiers(ev))
        elif type == 'pointer_up':
            self._vispy_mouse_release(native=ev, pos=(ev['x'], ev['y']), button=ev['button'], modifiers=self._modifiers(ev))
        elif type == 'pointer_move':
            self._vispy_mouse_move(native=ev, pos=(ev['x'], ev['y']), button=ev['button'], modifiers=self._modifiers(ev))
        elif type == 'double_click':
            self._vispy_mouse_double_click(native=ev, pos=(ev['x'], ev['y']), button=ev['button'], modifiers=self._modifiers(ev))
        elif type == 'wheel':
            self._vispy_canvas.events.mouse_wheel(native=ev, pos=(ev['x'], ev['y']), delta=(ev['dx'] / 100, -ev['dy'] / 100), modifiers=self._modifiers(ev))
        elif type == 'key_down':
            self._vispy_canvas.events.key_press(native=ev, key=keys.Key(ev['key']), modifiers=self._modifiers(ev), text=ev['key'])
        elif type == 'key_up':
            self._vispy_canvas.events.key_release(native=ev, key=keys.Key(ev['key']), modifiers=self._modifiers(ev), text=ev['key'])
        elif type == 'close':
            self._lifecycle = 2
            self._context.close()
            _stop_timers(self._vispy_canvas)
        else:
            pass

    def _modifiers(self, ev):
        if False:
            i = 10
            return i + 15
        return tuple((getattr(keys, m.upper()) for m in ev['modifiers']))

    def _emit_resize_event(self):
        if False:
            for i in range(10):
                print('nop')
        self._vispy_canvas.events.resize(size=self._logical_size, physical_size=self._physical_size)

    def get_frame(self):
        if False:
            while True:
                i = 10
        if self._physical_size[0] <= 1 or self._physical_size[1] <= 1:
            return None
        if not self._lifecycle:
            self._lifecycle = 1
            self._vispy_canvas.set_current()
            self._vispy_canvas.events.initialize()
            self._emit_resize_event()
        self._vispy_canvas.set_current()
        with self._helper:
            self._vispy_canvas.events.draw(region=None)
            array = self._helper.get_frame()
        self._vispy_canvas.context.flush_commands()
        return array

    def _vispy_warmup(self):
        if False:
            print('Hello World!')
        self._vispy_canvas.set_current()

    def _vispy_set_current(self):
        if False:
            i = 10
            return i + 15
        self._context.make_current()

    def _vispy_swap_buffers(self):
        if False:
            while True:
                i = 10
        pass

    def _vispy_set_title(self, title):
        if False:
            i = 10
            return i + 15
        pass

    def _vispy_set_size(self, w, h):
        if False:
            i = 10
            return i + 15
        self.css_width = f'{w}px'
        self.css_height = f'{h}px'

    def _vispy_set_position(self, x, y):
        if False:
            print('Hello World!')
        pass

    def _vispy_set_visible(self, visible):
        if False:
            i = 10
            return i + 15
        if not visible:
            raise NotImplementedError('Cannot hide the RFB widget')

    def _vispy_set_fullscreen(self, fullscreen):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def _vispy_update(self):
        if False:
            print('Hello World!')
        self.request_draw()

    def _vispy_close(self):
        if False:
            i = 10
            return i + 15
        self.close()

    def _vispy_get_size(self):
        if False:
            i = 10
            return i + 15
        return self._logical_size

    def _vispy_get_physical_size(self):
        if False:
            for i in range(10):
                print('nop')
        return self._physical_size

    def _vispy_get_position(self):
        if False:
            return 10
        return (0, 0)

    def _vispy_get_fullscreen(self):
        if False:
            while True:
                i = 10
        return False

class TimerBackend(BaseTimerBackend):

    def __init__(self, vispy_timer):
        if False:
            print('Hello World!')
        super().__init__(vispy_timer)
        self._loop = asyncio.get_event_loop()
        self._task = None

    async def _timer_coro(self, interval):
        while True:
            await asyncio.sleep(interval)
            self._vispy_timeout()

    def _vispy_start(self, interval):
        if False:
            for i in range(10):
                print('nop')
        if self._task is not None:
            self._task.cancel()
        self._task = asyncio.create_task(self._timer_coro(interval))

    def _vispy_stop(self):
        if False:
            print('Hello World!')
        self._task.cancel()
        self._task = None

    def _vispy_timeout(self):
        if False:
            return 10
        self._loop.call_soon(self._vispy_timer._timeout)

def _stop_timers(canvas):
    if False:
        print('Hello World!')
    'Stop all timers associated with a canvas.'
    for attr in dir(canvas):
        try:
            attr_obj = getattr(canvas, attr)
        except NotImplementedError:
            continue
        else:
            if isinstance(attr_obj, Timer):
                attr_obj.stop()