"""vispy backend for Tkinter."""
from __future__ import division
from time import sleep
import warnings
from ..base import BaseApplicationBackend, BaseCanvasBackend, BaseTimerBackend
from ...util import keys
from ...util.ptime import time
from ...gloo import gl
try:
    import sys
    (_tk_on_linux, _tk_on_darwin, _tk_on_windows) = map(sys.platform.startswith, ('linux', 'darwin', 'win'))
    _tk_pyopengltk_imported = False
    import tkinter as tk
    import pyopengltk
except (ModuleNotFoundError, ImportError):
    (available, testable, why_not, which) = (False, False, 'Could not import Tkinter or pyopengltk, module(s) not found.', None)
else:
    which_pyopengltk = getattr(pyopengltk, '__version__', '???')
    which = f'Tkinter {tk.TkVersion} (with pyopengltk {which_pyopengltk})'
    if hasattr(pyopengltk, 'OpenGLFrame'):
        _tk_pyopengltk_imported = True
        (available, testable, why_not) = (True, True, None)
    else:
        (available, testable, why_not) = (False, False, f'pyopengltk {which_pyopengltk} is not supported on this platform ({sys.platform})!')
if _tk_pyopengltk_imported:
    OpenGLFrame = pyopengltk.OpenGLFrame
else:

    class OpenGLFrame(object):
        pass
KEYMAP = {65505: keys.SHIFT, 65506: keys.SHIFT, 65507: keys.CONTROL, 65508: keys.CONTROL, 65513: keys.ALT, 65514: keys.ALT, 65371: keys.META, 65372: keys.META, 65361: keys.LEFT, 65362: keys.UP, 65363: keys.RIGHT, 65364: keys.DOWN, 65365: keys.PAGEUP, 65366: keys.PAGEDOWN, 65379: keys.INSERT, 65535: keys.DELETE, 65360: keys.HOME, 65367: keys.END, 65307: keys.ESCAPE, 65288: keys.BACKSPACE, 32: keys.SPACE, 65293: keys.ENTER, 65289: keys.TAB, 65470: keys.F1, 65471: keys.F2, 65472: keys.F3, 65473: keys.F4, 65474: keys.F5, 65475: keys.F6, 65476: keys.F7, 65477: keys.F8, 65478: keys.F9, 65479: keys.F10, 65480: keys.F11, 65481: keys.F12}
KEY_STATE_MAP = {1: keys.SHIFT, 4: keys.CONTROL, 128: keys.ALT, 131072: keys.ALT}
MOUSE_BUTTON_MAP = {1: 1, 2: 3, 3: 2}
capability = dict(title=True, size=True, position=True, show=True, vsync=False, resizable=True, decorate=True, fullscreen=True, context=False, multi_window=True, scroll=True, parent=True, always_on_top=True)

def _set_config(c):
    if False:
        for i in range(10):
            print('nop')
    'Set gl configuration for template.\n    Currently not used for Tkinter backend.\n    '
    return []

class _TkInstanceManager:
    _tk_inst = None
    _tk_inst_owned = False
    _canvasses = []

    @classmethod
    def get_tk_instance(cls):
        if False:
            while True:
                i = 10
        'Return the Tk instance.\n\n        Returns\n        -------\n        tk.Tk\n            The tk.Tk instance.\n        '
        if cls._tk_inst is None:
            if tk._default_root:
                cls._tk_inst = tk._default_root
                cls._tk_inst_owned = False
            else:
                cls._tk_inst = tk.Tk()
                cls._tk_inst.withdraw()
                cls._tk_inst_owned = True
        return cls._tk_inst

    @classmethod
    def get_canvasses(cls):
        if False:
            while True:
                i = 10
        'Return a list of CanvasBackends.\n\n        Returns\n        -------\n        list\n            A list with CanvasBackends.\n        '
        return cls._canvasses

    @classmethod
    def new_toplevel(cls, canvas, *args, **kwargs):
        if False:
            return 10
        'Create and return a new withdrawn Toplevel.\n        Create a tk.Toplevel with the given args and kwargs,\n        minimize it and add it to the list before returning\n\n        Parameters\n        ----------\n        canvas : CanvasBackend\n            The CanvasBackend instance that wants a new Toplevel.\n        *args\n            Variable length argument list.\n        **kwargs\n            Arbitrary keyword arguments.\n\n        Returns\n        -------\n        tk.Toplevel\n            Return the created tk.Toplevel\n        '
        tl = tk.Toplevel(cls._tk_inst, *args, **kwargs)
        tl.withdraw()
        cls._canvasses.append(canvas)
        return tl

    @classmethod
    def del_toplevel(cls, canvas=None):
        if False:
            while True:
                i = 10
        '\n        Destroy the given Toplevel, and if it was the last one,\n        also destroy the Tk instance if we created it.\n\n        Parameters\n        ----------\n        canvas : CanvasBackend\n            The CanvasBackend to destroy, defaults to None.\n        '
        if canvas:
            try:
                canvas.destroy()
                if canvas.top:
                    canvas.top.destroy()
                cls._canvasses.remove(canvas)
            except Exception:
                pass
        if cls._tk_inst and (not cls._canvasses) and cls._tk_inst_owned:
            cls._tk_inst.quit()
            cls._tk_inst.destroy()
            cls._tk_inst = None

class ApplicationBackend(BaseApplicationBackend):

    def _vispy_get_backend_name(self):
        if False:
            while True:
                i = 10
        '\n        Returns\n        -------\n        str\n            The name of the backend.\n        '
        return tk.__name__

    def _vispy_process_events(self):
        if False:
            i = 10
            return i + 15
        'Process events related to the spawned Tk application window.\n        First, update the Tk instance, then call `_delayed_update`\n        on every created Toplevel (to force a redraw), and process some Tkinter\n        GUI events by calling the Tk.mainloop and immediately exiting.\n        '
        app = self._vispy_get_native_app()
        app.update_idletasks()
        for c in _TkInstanceManager.get_canvasses():
            c._delayed_update()
        app.after(0, lambda : app.quit())
        app.mainloop()

    def _vispy_run(self):
        if False:
            for i in range(10):
                print('nop')
        'Start the Tk.mainloop. This will block until all Tk windows are destroyed.'
        self._vispy_get_native_app().mainloop()

    def _vispy_quit(self):
        if False:
            return 10
        'Destroy each created Toplevel by calling _vispy_close on it.\n        If there are no Toplevels left, also destroy the Tk instance.\n        '
        for c in _TkInstanceManager.get_canvasses():
            c._vispy_close()
        _TkInstanceManager.del_toplevel()

    def _vispy_get_native_app(self):
        if False:
            while True:
                i = 10
        'Get or create the Tk instance.\n\n        Returns\n        -------\n        tk.Tk\n            The tk.Tk instance.\n        '
        return _TkInstanceManager.get_tk_instance()

class CanvasBackend(BaseCanvasBackend, OpenGLFrame):
    """Tkinter backend for Canvas abstract class.
    Uses pyopengltk.OpenGLFrame as the internal tk.Frame instance that
    is able to receive OpenGL draw commands and display the results,
    while also being placeable in another Toplevel window.
    """

    def __init__(self, vispy_canvas, **kwargs):
        if False:
            i = 10
            return i + 15
        BaseCanvasBackend.__init__(self, vispy_canvas)
        p = self._process_backend_kwargs(kwargs)
        self._double_click_supported = True
        p.context.shared.add_ref('tk', self)
        if p.context.shared.ref is self:
            self._native_context = None
        else:
            self._native_context = p.context.shared.ref._native_context
        kwargs.pop('parent')
        kwargs.pop('title')
        kwargs.pop('size')
        kwargs.pop('position')
        kwargs.pop('show')
        kwargs.pop('vsync')
        kwargs.pop('resizable')
        kwargs.pop('decorate')
        kwargs.pop('always_on_top')
        kwargs.pop('fullscreen')
        kwargs.pop('context')
        if p.parent is None:
            self.top = _TkInstanceManager.new_toplevel(self)
            if p.title:
                self._vispy_set_title(p.title)
            if p.size:
                self._vispy_set_size(p.size[0], p.size[1])
            if p.position:
                self._vispy_set_position(p.position[0], p.position[1])
            self.top.update_idletasks()
            if not p.resizable:
                self.top.resizable(False, False)
            if not p.decorate:
                self.top.overrideredirect(True)
            if p.always_on_top:
                self.top.wm_attributes('-topmost', 'True')
            self._fullscreen = bool(p.fullscreen)
            self.top.protocol('WM_DELETE_WINDOW', self._vispy_close)
            parent = self.top
        else:
            self.top = None
            parent = p.parent
            self._fullscreen = False
        self._init = False
        self.is_destroyed = False
        self._dynamic_keymap = {}
        OpenGLFrame.__init__(self, parent, **kwargs)
        if self.top:
            self.top.configure(bg='black')
            self.pack(fill=tk.BOTH, expand=True)
            self.top.bind('<Any-KeyPress>', self._on_key_down)
            self.top.bind('<Any-KeyRelease>', self._on_key_up)
        else:
            self.bind('<Any-KeyPress>', self._on_key_down)
            self.bind('<Any-KeyRelease>', self._on_key_up)
        self.bind('<Enter>', self._on_mouse_enter)
        self.bind('<Leave>', self._on_mouse_leave)
        self.bind('<Motion>', self._on_mouse_move)
        self.bind('<Any-Button>', self._on_mouse_button_press)
        self.bind('<Double-Any-Button>', self._on_mouse_double_button_press)
        self.bind('<Any-ButtonRelease>', self._on_mouse_button_release)
        self.bind('<Configure>', self._on_configure, add='+')
        self._vispy_set_visible(p.show)
        self.focus_force()

    def initgl(self):
        if False:
            return 10
        'Overridden from OpenGLFrame\n        Gets called on init or when the frame is remapped into its container.\n        '
        if not hasattr(self, '_native_context') or self._native_context is None:
            self._native_context = vars(self).get('_CanvasBackend__context', None)
        self.update_idletasks()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)

    def redraw(self, *args):
        if False:
            while True:
                i = 10
        'Overridden from OpenGLFrame\n        Gets called when the OpenGLFrame redraws itself.\n        It will set the current buffer, call self.redraw() and\n        swap buffers afterwards.\n        '
        if self._vispy_canvas is None:
            return
        if not self._init:
            self._initialize()
        self._vispy_canvas.set_current()
        self._vispy_canvas.events.draw(region=None)

    def _delayed_update(self):
        if False:
            i = 10
            return i + 15
        '\n        Expose a new frame to the canvas. This will call self.redraw() internally.\n\n        The self.animate sets the refresh rate in milliseconds. Using this is not\n        necessary because VisPy will use the TimerBackend to periodically call\n        self._vispy_update, resulting in the exact same behaviour.\n        So we set it to `0` to tell OpenGLFrame not to redraw itself on its own.\n        '
        if self.is_destroyed:
            return
        self.animate = 0
        self.tkExpose(None)

    def _on_configure(self, e):
        if False:
            print('Hello World!')
        'Called when the frame get configured or resized.'
        if self._vispy_canvas is None or not self._init:
            return
        size_tup = e if isinstance(e, tuple) else (e.width, e.height)
        self._vispy_canvas.events.resize(size=size_tup)

    def _initialize(self):
        if False:
            return 10
        'Initialise the Canvas for drawing.'
        self.initgl()
        if self._vispy_canvas is None:
            return
        self._init = True
        self._vispy_canvas.set_current()
        self._vispy_canvas.events.initialize()
        self.update_idletasks()
        self._on_configure(self._vispy_get_size())

    def _vispy_warmup(self):
        if False:
            for i in range(10):
                print('nop')
        "Provided for VisPy tests, so they can 'warm the canvas up'.\n        Mostly taken from the wxWidgets backend.\n        "
        tk_inst = _TkInstanceManager.get_tk_instance()
        etime = time() + 0.3
        while time() < etime:
            sleep(0.01)
            self._vispy_canvas.set_current()
            self._vispy_canvas.app.process_events()
            tk_inst.after(0, lambda : tk_inst.quit())
            tk_inst.mainloop()

    def _parse_state(self, e):
        if False:
            for i in range(10):
                print('nop')
        "Helper to parse event.state into modifier keys.\n\n        Parameters\n        ----------\n        e : tk.Event\n            The passed in Event.\n\n        Returns\n        -------\n        list\n            A list of modifier keys that are active (from vispy's keys)\n        "
        return [key for (mask, key) in KEY_STATE_MAP.items() if e.state & mask]

    def _parse_keys(self, e):
        if False:
            print('Hello World!')
        'Helper to parse key states into Vispy keys.\n\n        Parameters\n        ----------\n        e : tk.Event\n            The passed in Event.\n\n        Returns\n        -------\n        tuple\n            A tuple (key.Key(), chr(key)), which has the vispy key object and\n            the character representation if available.\n        '
        if e.keysym_num in KEYMAP:
            return (KEYMAP[e.keysym_num], '')
        if e.char:
            self._dynamic_keymap[e.keycode] = e.char
            return (keys.Key(e.char), e.char)
        if e.keycode in self._dynamic_keymap:
            char = self._dynamic_keymap[e.keycode]
            return (keys.Key(char), char)
        warnings.warn('The key you typed is not supported by the tkinter backend.Please map your functionality to a different key')
        return (None, None)

    def _on_mouse_enter(self, e):
        if False:
            i = 10
            return i + 15
        'Event callback when the mouse enters the canvas.\n\n        Parameters\n        ----------\n        e : tk.Event\n            The passed in Event.\n        '
        if self._vispy_canvas is None:
            return
        if _tk_on_linux:
            self.bind_all('<Button-4>', self._on_mouse_wheel)
            self.bind_all('<Button-5>', self._on_mouse_wheel)
        else:
            self.bind_all('<MouseWheel>', self._on_mouse_wheel)
        self._vispy_mouse_move(pos=(e.x, e.y), modifiers=self._parse_state(e))

    def _on_mouse_leave(self, e):
        if False:
            i = 10
            return i + 15
        'Event callback when the mouse leaves the canvas.\n\n        Parameters\n        ----------\n        e : tk.Event\n            The passed in Event.\n        '
        if self._vispy_canvas is None:
            return
        if _tk_on_linux:
            self.unbind_all('<Button-4>')
            self.unbind_all('<Button-5>')
        else:
            self.unbind_all('<MouseWheel>')

    def _on_mouse_move(self, e):
        if False:
            for i in range(10):
                print('nop')
        'Event callback when the mouse is moved within the canvas.\n\n        Parameters\n        ----------\n        e : tk.Event\n            The passed in Event.\n        '
        if self._vispy_canvas is None:
            return
        self._vispy_mouse_move(pos=(e.x, e.y), modifiers=self._parse_state(e))

    def _on_mouse_wheel(self, e):
        if False:
            for i in range(10):
                print('nop')
        'Event callback when the mouse wheel changes within the canvas.\n\n        Parameters\n        ----------\n        e : tk.Event\n            The passed in Event.\n        '
        if self._vispy_canvas is None:
            return
        if _tk_on_linux:
            e.delta = {4: 120, 5: -120}.get(e.num, 0)
        self._vispy_canvas.events.mouse_wheel(delta=(0.0, float(e.delta / 120)), pos=(e.x, e.y), modifiers=self._parse_state(e))

    def _on_mouse_button_press(self, e):
        if False:
            for i in range(10):
                print('nop')
        'Event callback when a mouse button is pressed within the canvas.\n\n        Parameters\n        ----------\n        e : tk.Event\n            The passed in Event.\n        '
        if self._vispy_canvas is None:
            return
        if _tk_on_linux and e.num in (4, 5):
            return
        self._vispy_mouse_press(pos=(e.x, e.y), button=MOUSE_BUTTON_MAP.get(e.num, e.num), modifiers=self._parse_state(e))

    def _vispy_detect_double_click(self, e):
        if False:
            print('Hello World!')
        'Override base class function\n        since double click handling is native in Tk.\n        '
        pass

    def _on_mouse_double_button_press(self, e):
        if False:
            i = 10
            return i + 15
        'Event callback when a mouse button is double clicked within the canvas.\n\n        Parameters\n        ----------\n        e : tk.Event\n            The passed in Event.\n        '
        if self._vispy_canvas is None:
            return
        if _tk_on_linux and e.num in (4, 5):
            return
        self._vispy_mouse_double_click(pos=(e.x, e.y), button=MOUSE_BUTTON_MAP.get(e.num, e.num), modifiers=self._parse_state(e))

    def _on_mouse_button_release(self, e):
        if False:
            return 10
        'Event callback when a mouse button is released within the canvas.\n\n        Parameters\n        ----------\n        e : tk.Event\n            The passed in Event.\n        '
        if self._vispy_canvas is None:
            return
        if _tk_on_linux and e.num in (4, 5):
            return
        self._vispy_mouse_release(pos=(e.x, e.y), button=MOUSE_BUTTON_MAP.get(e.num, e.num), modifiers=self._parse_state(e))

    def _on_key_down(self, e):
        if False:
            print('Hello World!')
        "Event callback when a key is pressed within the canvas or window.\n\n        Ignore keys.ESCAPE if this is an embedded canvas,\n        as this would make it unresponsive (because it won't close the entire window),\n        while still being updateable.\n\n        Parameters\n        ----------\n        e : tk.Event\n            The passed in Event.\n        "
        if self._vispy_canvas is None:
            return
        (key, text) = self._parse_keys(e)
        if not self.top and key == keys.ESCAPE:
            return
        self._vispy_canvas.events.key_press(key=key, text=text, modifiers=self._parse_state(e))

    def _on_key_up(self, e):
        if False:
            i = 10
            return i + 15
        "Event callback when a key is released within the canvas or window.\n\n        Ignore keys.ESCAPE if this is an embedded canvas,\n        as this would make it unresponsive (because it won't close the entire window),\n        while still being updateable.\n\n        Parameters\n        ----------\n        e : tk.Event\n            The passed in Event.\n        "
        if self._vispy_canvas is None:
            return
        (key, text) = self._parse_keys(e)
        if not self.top and key == keys.ESCAPE:
            return
        self._vispy_canvas.events.key_release(key=key, text=text, modifiers=self._parse_state(e))

    def _vispy_set_current(self):
        if False:
            print('Hello World!')
        'Make this the current context.'
        if not self.is_destroyed:
            self.tkMakeCurrent()

    def _vispy_swap_buffers(self):
        if False:
            for i in range(10):
                print('nop')
        'Swap front and back buffer. This is done internally inside OpenGLFrame.'
        self._vispy_canvas.set_current()

    def _vispy_set_title(self, title):
        if False:
            i = 10
            return i + 15
        'Set the window title. Has no effect for widgets.'
        if self.top:
            self.top.title(title)

    def _vispy_set_size(self, w, h):
        if False:
            while True:
                i = 10
        'Set size of the window. Has no effect for widgets.'
        if self.top:
            self.top.geometry(f'{w}x{h}')

    def _vispy_set_position(self, x, y):
        if False:
            while True:
                i = 10
        'Set location of the window. Has no effect for widgets.'
        if self.top:
            self.top.geometry(f'+{x}+{y}')

    def _vispy_set_visible(self, visible):
        if False:
            print('Hello World!')
        'Show or hide the window. Has no effect for widgets.'
        if self.top:
            if visible:
                self.top.wm_deiconify()
                self.top.lift()
                self.top.attributes('-fullscreen', self._fullscreen)
            else:
                self.top.withdraw()

    def _vispy_set_fullscreen(self, fullscreen):
        if False:
            print('Hello World!')
        'Set the current fullscreen state.\n\n        Has no effect for widgets. If you want it to become fullscreen,\n        while embedded in another Toplevel window, you should make that\n        window fullscreen instead.\n        '
        self._fullscreen = bool(fullscreen)
        if self.top:
            self._vispy_set_visible(True)

    def _vispy_update(self):
        if False:
            print('Hello World!')
        'Invoke a redraw\n\n        Delay this by letting Tk call it later, even a delay of 0 will do.\n        Doing this, prevents EventEmitter loops that are caused\n        by wanting to draw too fast.\n        '
        self.after(0, self._delayed_update)

    def _vispy_close(self):
        if False:
            while True:
                i = 10
        'Force the window to close, destroying the canvas in the process.\n\n        When this was the last VisPy window, also quit the Tk instance.\n        This will not interfere if there is already another user window,\n        unrelated top VisPy open.\n        '
        if self.top and (not self.is_destroyed):
            self.is_destroyed = True
            self._vispy_canvas.close()
            _TkInstanceManager.del_toplevel(self)

    def destroy(self):
        if False:
            i = 10
            return i + 15
        'Callback when the window gets closed.\n        Destroy the VisPy canvas by calling close on it.\n        '
        self._vispy_canvas.close()

    def _vispy_get_size(self):
        if False:
            return 10
        'Return the actual size of the frame.'
        if self.top:
            self.top.update_idletasks()
        return (self.winfo_width(), self.winfo_height())

    def _vispy_get_position(self):
        if False:
            i = 10
            return i + 15
        'Return the widget or window position.'
        return (self.winfo_x(), self.winfo_y())

    def _vispy_get_fullscreen(self):
        if False:
            print('Hello World!')
        "Return the last set full screen state, regardless if it's actually in that state.\n\n        When using the canvas as a widget, it will not go into fullscreen.\n        See _vispy_set_fullscreen\n        "
        return self._fullscreen

class TimerBackend(BaseTimerBackend):

    def __init__(self, vispy_timer):
        if False:
            print('Hello World!')
        BaseTimerBackend.__init__(self, vispy_timer)
        self._tk = _TkInstanceManager.get_tk_instance()
        if self._tk is None:
            raise Exception('TimerBackend: No toplevel?')
        self._id = None
        self.last_interval = 1

    def _vispy_start(self, interval):
        if False:
            for i in range(10):
                print('nop')
        'Start the timer.\n        Use Tk.after to schedule timer events.\n        '
        self._vispy_stop()
        self.last_interval = max(0, int(round(interval * 1000)))
        self._id = self._tk.after(self.last_interval, self._vispy_timeout)

    def _vispy_stop(self):
        if False:
            return 10
        'Stop the timer.\n        Unschedule the previous callback if it exists.\n        '
        if self._id is not None:
            self._tk.after_cancel(self._id)
            self._id = None

    def _vispy_timeout(self):
        if False:
            return 10
        'Callback when the timer finishes.\n        Also reschedules the next callback.\n        '
        self._vispy_timer._timeout()
        self._id = self._tk.after(self.last_interval, self._vispy_timeout)