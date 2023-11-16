"""vispy backend for glfw."""
from __future__ import division
import atexit
from time import sleep
import gc
import os
from ..base import BaseApplicationBackend, BaseCanvasBackend, BaseTimerBackend
from ...util import keys, logger
from ...util.ptime import time
from ... import config
USE_EGL = config['gl_backend'].lower().startswith('es')
glfw = None
try:
    import glfw
except ImportError:
    why_not = 'Could not import glfw, you may need to `pip install glfw` first.'
    (available, testable, why_not, which) = (False, False, why_not, None)
except Exception as err:
    why_not = 'Error importing glfw: ' + str(err)
    (available, testable, why_not, which) = (False, False, why_not, None)
else:
    if USE_EGL:
        (available, testable, why_not) = (False, False, 'EGL not supported')
        which = 'glfw ' + str(glfw.__version__)
    else:
        (available, testable, why_not) = (True, True, None)
        which = 'glfw ' + str(glfw.__version__)
if glfw:
    KEYMAP = {glfw.KEY_LEFT_SHIFT: keys.SHIFT, glfw.KEY_RIGHT_SHIFT: keys.SHIFT, glfw.KEY_LEFT_CONTROL: keys.CONTROL, glfw.KEY_RIGHT_CONTROL: keys.CONTROL, glfw.KEY_LEFT_ALT: keys.ALT, glfw.KEY_RIGHT_ALT: keys.ALT, glfw.KEY_LEFT_SUPER: keys.META, glfw.KEY_RIGHT_SUPER: keys.META, glfw.KEY_LEFT: keys.LEFT, glfw.KEY_UP: keys.UP, glfw.KEY_RIGHT: keys.RIGHT, glfw.KEY_DOWN: keys.DOWN, glfw.KEY_PAGE_UP: keys.PAGEUP, glfw.KEY_PAGE_DOWN: keys.PAGEDOWN, glfw.KEY_INSERT: keys.INSERT, glfw.KEY_DELETE: keys.DELETE, glfw.KEY_HOME: keys.HOME, glfw.KEY_END: keys.END, glfw.KEY_ESCAPE: keys.ESCAPE, glfw.KEY_BACKSPACE: keys.BACKSPACE, glfw.KEY_F1: keys.F1, glfw.KEY_F2: keys.F2, glfw.KEY_F3: keys.F3, glfw.KEY_F4: keys.F4, glfw.KEY_F5: keys.F5, glfw.KEY_F6: keys.F6, glfw.KEY_F7: keys.F7, glfw.KEY_F8: keys.F8, glfw.KEY_F9: keys.F9, glfw.KEY_F10: keys.F10, glfw.KEY_F11: keys.F11, glfw.KEY_F12: keys.F12, glfw.KEY_SPACE: keys.SPACE, glfw.KEY_ENTER: keys.ENTER, '\r': keys.ENTER, glfw.KEY_TAB: keys.TAB}
    BUTTONMAP = {glfw.MOUSE_BUTTON_LEFT: 1, glfw.MOUSE_BUTTON_RIGHT: 2, glfw.MOUSE_BUTTON_MIDDLE: 3}
MOD_KEYS = [keys.SHIFT, keys.ALT, keys.CONTROL, keys.META]
_GLFW_INITIALIZED = False
_VP_GLFW_ALL_WINDOWS = []

def _get_glfw_windows():
    if False:
        i = 10
        return i + 15
    wins = list()
    for win in _VP_GLFW_ALL_WINDOWS:
        if isinstance(win, CanvasBackend):
            wins.append(win)
    return wins
capability = dict(title=True, size=True, position=True, show=True, vsync=True, resizable=True, decorate=True, fullscreen=True, context=True, multi_window=True, scroll=True, parent=False, always_on_top=True)

def _set_config(c):
    if False:
        return 10
    'Set gl configuration for GLFW.'
    glfw.window_hint(glfw.RED_BITS, c['red_size'])
    glfw.window_hint(glfw.GREEN_BITS, c['green_size'])
    glfw.window_hint(glfw.BLUE_BITS, c['blue_size'])
    glfw.window_hint(glfw.ALPHA_BITS, c['alpha_size'])
    glfw.window_hint(glfw.ACCUM_RED_BITS, 0)
    glfw.window_hint(glfw.ACCUM_GREEN_BITS, 0)
    glfw.window_hint(glfw.ACCUM_BLUE_BITS, 0)
    glfw.window_hint(glfw.ACCUM_ALPHA_BITS, 0)
    glfw.window_hint(glfw.DEPTH_BITS, c['depth_size'])
    glfw.window_hint(glfw.STENCIL_BITS, c['stencil_size'])
    glfw.window_hint(glfw.SAMPLES, c['samples'])
    glfw.window_hint(glfw.STEREO, c['stereo'])
    if not c['double_buffer']:
        raise RuntimeError('GLFW must double buffer, consider using a different backend, or using double buffering')
_glfw_errors = []

def _error_callback(num, descr):
    if False:
        return 10
    _glfw_errors.append('Error %s: %s' % (num, descr))

class ApplicationBackend(BaseApplicationBackend):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
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
        return 'Glfw'

    def _vispy_process_events(self):
        if False:
            while True:
                i = 10
        glfw.poll_events()
        for timer in self._timers:
            timer._tick()
        wins = _get_glfw_windows()
        for win in wins:
            if win._needs_draw:
                win._needs_draw = False
                win._on_draw()

    def _vispy_run(self):
        if False:
            for i in range(10):
                print('nop')
        wins = _get_glfw_windows()
        while any((w._id is not None and (not glfw.window_should_close(w._id)) for w in wins)):
            self._vispy_process_events()
        self._vispy_quit()

    def _vispy_quit(self):
        if False:
            print('Hello World!')
        wins = _get_glfw_windows()
        for win in wins:
            if win._vispy_canvas is not None:
                win._vispy_canvas.close()
        for timer in self._timers:
            timer._vispy_stop()
        self._timers = []

    def _vispy_get_native_app(self):
        if False:
            i = 10
            return i + 15
        global _GLFW_INITIALIZED
        if not _GLFW_INITIALIZED:
            cwd = os.getcwd()
            glfw.set_error_callback(_error_callback)
            try:
                if not glfw.init():
                    raise OSError('Could not init glfw:\n%r' % _glfw_errors)
            finally:
                os.chdir(cwd)
            glfw.set_error_callback(None)
            atexit.register(glfw.terminate)
            _GLFW_INITIALIZED = True
        return glfw

class CanvasBackend(BaseCanvasBackend):
    """Glfw backend for Canvas abstract class."""

    def __init__(self, vispy_canvas, **kwargs):
        if False:
            i = 10
            return i + 15
        BaseCanvasBackend.__init__(self, vispy_canvas)
        p = self._process_backend_kwargs(kwargs)
        self._initialized = False
        _set_config(p.context.config)
        p.context.shared.add_ref('glfw', self)
        if p.context.shared.ref is self:
            share = None
        else:
            share = p.context.shared.ref._id
        glfw.window_hint(glfw.REFRESH_RATE, 0)
        glfw.window_hint(glfw.RESIZABLE, int(p.resizable))
        glfw.window_hint(glfw.DECORATED, int(p.decorate))
        glfw.window_hint(glfw.VISIBLE, 0)
        glfw.window_hint(glfw.FLOATING, int(p.always_on_top))
        if p.fullscreen is not False:
            self._fullscreen = True
            if p.fullscreen is True:
                monitor = glfw.get_primary_monitor()
            else:
                monitor = glfw.get_monitors()
                if p.fullscreen >= len(monitor):
                    raise ValueError('fullscreen must be <= %s' % len(monitor))
                monitor = monitor[p.fullscreen]
            use_size = glfw.get_video_mode(monitor)[0][:2]
            if use_size != tuple(p.size):
                logger.debug('Requested size %s, will be ignored to use fullscreen mode %s' % (p.size, use_size))
            size = use_size
        else:
            self._fullscreen = False
            monitor = None
            size = p.size
        self._id = glfw.create_window(width=size[0], height=size[1], title=p.title, monitor=monitor, share=share)
        if not self._id:
            raise RuntimeError('Could not create window')
        glfw.make_context_current(self._id)
        glfw.swap_interval(1 if p.vsync else 0)
        _VP_GLFW_ALL_WINDOWS.append(self)
        self._mod = list()
        glfw.set_window_refresh_callback(self._id, self._on_draw)
        glfw.set_window_size_callback(self._id, self._on_resize)
        glfw.set_key_callback(self._id, self._on_key_press)
        glfw.set_char_callback(self._id, self._on_key_char)
        glfw.set_mouse_button_callback(self._id, self._on_mouse_button)
        glfw.set_scroll_callback(self._id, self._on_mouse_scroll)
        glfw.set_cursor_pos_callback(self._id, self._on_mouse_motion)
        glfw.set_window_close_callback(self._id, self._on_close)
        self._vispy_canvas_ = None
        self._needs_draw = False
        self._vispy_canvas.set_current()
        if p.position is not None:
            self._vispy_set_position(*p.position)
        if p.show:
            glfw.show_window(self._id)
        self._initialized = True
        self._next_key_events = []
        self._next_key_text = {}
        self._vispy_canvas.set_current()
        self._vispy_canvas.events.initialize()
        self._on_resize(self._id, size[0], size[1])

    def _vispy_warmup(self):
        if False:
            i = 10
            return i + 15
        etime = time() + 0.25
        while time() < etime:
            sleep(0.01)
            self._vispy_canvas.set_current()
            self._vispy_canvas.app.process_events()

    def _vispy_set_current(self):
        if False:
            print('Hello World!')
        if self._id is None:
            return
        glfw.make_context_current(self._id)

    def _vispy_swap_buffers(self):
        if False:
            print('Hello World!')
        if self._id is None:
            return
        glfw.swap_buffers(self._id)

    def _vispy_set_title(self, title):
        if False:
            for i in range(10):
                print('nop')
        if self._id is None:
            return
        glfw.set_window_title(self._id, title)

    def _vispy_set_size(self, w, h):
        if False:
            i = 10
            return i + 15
        if self._id is None:
            return
        glfw.set_window_size(self._id, w, h)

    def _vispy_set_position(self, x, y):
        if False:
            i = 10
            return i + 15
        if self._id is None:
            return
        glfw.set_window_pos(self._id, x, y)

    def _vispy_set_visible(self, visible):
        if False:
            while True:
                i = 10
        if self._id is None:
            return
        if visible:
            glfw.show_window(self._id)
            self._vispy_update()
        else:
            glfw.hide_window(self._id)

    def _vispy_set_fullscreen(self, fullscreen):
        if False:
            i = 10
            return i + 15
        logger.warn('Cannot change fullscreen mode for GLFW backend')

    def _vispy_update(self):
        if False:
            while True:
                i = 10
        if self._vispy_canvas is None or self._id is None:
            return
        self._needs_draw = True

    def _vispy_close(self):
        if False:
            for i in range(10):
                print('nop')
        if self._id is not None:
            self._vispy_canvas = None
            self._vispy_set_visible(False)
            (self._id, id_) = (None, self._id)
            glfw.destroy_window(id_)
            gc.collect()

    def _vispy_get_size(self):
        if False:
            i = 10
            return i + 15
        if self._id is None:
            return
        (w, h) = glfw.get_window_size(self._id)
        return (w, h)

    def _vispy_get_physical_size(self):
        if False:
            return 10
        if self._id is None:
            return
        (w, h) = glfw.get_framebuffer_size(self._id)
        return (w, h)

    def _vispy_get_position(self):
        if False:
            i = 10
            return i + 15
        if self._id is None:
            return
        (x, y) = glfw.get_window_pos(self._id)
        return (x, y)

    def _vispy_get_fullscreen(self):
        if False:
            while True:
                i = 10
        return self._fullscreen

    def _on_resize(self, _id, w, h):
        if False:
            print('Hello World!')
        if self._vispy_canvas is None:
            return
        self._vispy_canvas.events.resize(size=(w, h), physical_size=self._vispy_get_physical_size())

    def _on_close(self, _id):
        if False:
            return 10
        if self._vispy_canvas is None:
            return
        self._vispy_canvas.close()

    def _on_draw(self, _id=None):
        if False:
            return 10
        if self._vispy_canvas is None or self._id is None:
            return
        self._vispy_canvas.set_current()
        self._vispy_canvas.events.draw(region=None)

    def _on_mouse_button(self, _id, button, action, mod):
        if False:
            print('Hello World!')
        if self._vispy_canvas is None and self._id is not None:
            return
        pos = glfw.get_cursor_pos(self._id)
        if button < 3:
            button = BUTTONMAP.get(button, 0)
            if action == glfw.PRESS:
                fun = self._vispy_mouse_press
            elif action == glfw.RELEASE:
                fun = self._vispy_mouse_release
            else:
                return
            fun(pos=pos, button=button, modifiers=self._mod)

    def _on_mouse_scroll(self, _id, x_off, y_off):
        if False:
            for i in range(10):
                print('nop')
        if self._vispy_canvas is None and self._id is not None:
            return
        pos = glfw.get_cursor_pos(self._id)
        delta = (float(x_off), float(y_off))
        self._vispy_canvas.events.mouse_wheel(pos=pos, delta=delta, modifiers=self._mod)

    def _on_mouse_motion(self, _id, x, y):
        if False:
            return 10
        if self._vispy_canvas is None:
            return
        self._vispy_mouse_move(pos=(x, y), modifiers=self._mod)

    def _on_key_press(self, _id, key, scancode, action, mod):
        if False:
            i = 10
            return i + 15
        if self._vispy_canvas is None:
            return
        (key, text) = self._process_key(key)
        if action == glfw.PRESS:
            fun = self._vispy_canvas.events.key_press
            down = True
        elif action == glfw.RELEASE:
            fun = self._vispy_canvas.events.key_release
            down = False
        else:
            return
        self._process_mod(key, down=down)
        if text != '' and action == glfw.PRESS:
            self._next_key_events.append((fun, key, self._mod))
        else:
            if key in self._next_key_text:
                text = self._next_key_text[key]
                del self._next_key_text[key]
            fun(key=key, text=text, modifiers=self._mod)

    def _on_key_char(self, _id, text):
        if False:
            print('Hello World!')
        if len(self._next_key_events) == 0:
            return
        (fun, key, mod) = self._next_key_events.pop(0)
        fun(key=key, text=chr(text), modifiers=mod)
        self._next_key_text[key] = text

    def _process_key(self, key):
        if False:
            print('Hello World!')
        if 32 <= key <= 127:
            return (keys.Key(chr(key)), chr(key))
        elif key in KEYMAP:
            return (KEYMAP[key], '')
        else:
            return (None, '')

    def _process_mod(self, key, down):
        if False:
            return 10
        'Process (possible) keyboard modifiers\n\n        GLFW provides "mod" with many callbacks, but not (critically) the\n        scroll callback, so we keep track on our own here.\n        '
        if key in MOD_KEYS:
            if down:
                if key not in self._mod:
                    self._mod.append(key)
            elif key in self._mod:
                self._mod.pop(self._mod.index(key))
        return self._mod

class TimerBackend(BaseTimerBackend):

    def __init__(self, vispy_timer):
        if False:
            return 10
        BaseTimerBackend.__init__(self, vispy_timer)
        vispy_timer._app._backend._add_timer(self)
        self._vispy_stop()

    def _vispy_start(self, interval):
        if False:
            return 10
        self._interval = interval
        self._next_time = time() + self._interval

    def _vispy_stop(self):
        if False:
            i = 10
            return i + 15
        self._next_time = float('inf')

    def _tick(self):
        if False:
            i = 10
            return i + 15
        if time() >= self._next_time:
            self._vispy_timer._timeout()
            self._next_time = time() + self._interval