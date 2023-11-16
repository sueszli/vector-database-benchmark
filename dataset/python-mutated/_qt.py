"""
Base code for the Qt backends. Note that this is *not* (anymore) a
backend by itself! One has to explicitly use either PySide, PyQt4 or
PySide2, PyQt5 or PyQt6. Note that the automatic backend selection prefers
a GUI toolkit that is already imported.

The _pyside, _pyqt4, _pyside2, _pyqt5 and _pyside6 modules will
import * from this module, and also keep a ref to the module object.
Note that if two of the backends are used, this module is actually
reloaded. This is a sorts of poor mans "subclassing" to get a working
version for both backends using the same code.

Note that it is strongly discouraged to use the
PySide/PyQt4/PySide2/PyQt5/PySide6 backends simultaneously. It is
known to cause unpredictable behavior and segfaults.
"""
from __future__ import division
from time import sleep, time
import math
import os
import sys
import atexit
import ctypes
from packaging.version import Version
from ...util import logger
from ..base import BaseApplicationBackend, BaseCanvasBackend, BaseTimerBackend
from ...util import keys
from ... import config
from . import qt_lib
USE_EGL = config['gl_backend'].lower().startswith('es')
IS_LINUX = IS_OSX = IS_WIN = IS_RPI = False
if sys.platform.startswith('linux'):
    if os.uname()[4].startswith('arm'):
        IS_RPI = True
    else:
        IS_LINUX = True
elif sys.platform.startswith('darwin'):
    IS_OSX = True
elif sys.platform.startswith('win'):
    IS_WIN = True

def _check_imports(lib):
    if False:
        i = 10
        return i + 15
    libs = ['PyQt4', 'PyQt5', 'PyQt6', 'PySide', 'PySide2', 'PySide6']
    libs.remove(lib)
    for lib2 in libs:
        lib2 += '.QtCore'
        if lib2 in sys.modules:
            raise RuntimeError('Refusing to import %s because %s is already imported.' % (lib, lib2))

def _get_event_xy(ev):
    if False:
        print('Hello World!')
    if hasattr(ev, 'pos'):
        (posx, posy) = (ev.pos().x(), ev.pos().y())
    else:
        (posx, posy) = (ev.position().x(), ev.position().y())
    return (posx, posy)
QGLWidget = object
QT5_NEW_API = False
PYSIDE6_API = False
PYQT6_API = False
if qt_lib == 'pyqt4':
    _check_imports('PyQt4')
    if not USE_EGL:
        from PyQt4.QtOpenGL import QGLWidget, QGLFormat
    from PyQt4 import QtGui, QtCore, QtTest
    (QWidget, QApplication) = (QtGui.QWidget, QtGui.QApplication)
elif qt_lib == 'pyqt5':
    _check_imports('PyQt5')
    if not USE_EGL:
        from PyQt5.QtCore import QT_VERSION_STR
        if Version(QT_VERSION_STR) >= Version('5.4.0'):
            from PyQt5.QtWidgets import QOpenGLWidget as QGLWidget
            from PyQt5.QtGui import QSurfaceFormat as QGLFormat
            QT5_NEW_API = True
        else:
            from PyQt5.QtOpenGL import QGLWidget, QGLFormat
    from PyQt5 import QtGui, QtCore, QtWidgets, QtTest
    (QWidget, QApplication) = (QtWidgets.QWidget, QtWidgets.QApplication)
elif qt_lib == 'pyqt6':
    _check_imports('PyQt6')
    if not USE_EGL:
        from PyQt6.QtCore import QT_VERSION_STR
        if Version(QT_VERSION_STR) >= Version('6.0.0'):
            from PyQt6.QtOpenGLWidgets import QOpenGLWidget as QGLWidget
            from PyQt6.QtGui import QSurfaceFormat as QGLFormat
            PYQT6_API = True
        else:
            from PyQt6.QtOpenGL import QGLWidget, QGLFormat
    from PyQt6 import QtGui, QtCore, QtWidgets, QtTest
    (QWidget, QApplication) = (QtWidgets.QWidget, QtWidgets.QApplication)
elif qt_lib == 'pyside6':
    _check_imports('PySide6')
    if not USE_EGL:
        from PySide6.QtCore import __version__ as QT_VERSION_STR
        if Version(QT_VERSION_STR) >= Version('6.0.0'):
            from PySide6.QtOpenGLWidgets import QOpenGLWidget as QGLWidget
            from PySide6.QtGui import QSurfaceFormat as QGLFormat
            PYSIDE6_API = True
        else:
            from PySide6.QtOpenGL import QGLWidget, QGLFormat
    from PySide6 import QtGui, QtCore, QtWidgets, QtTest
    (QWidget, QApplication) = (QtWidgets.QWidget, QtWidgets.QApplication)
elif qt_lib == 'pyside2':
    _check_imports('PySide2')
    if not USE_EGL:
        from PySide2.QtCore import __version__ as QT_VERSION_STR
        if Version(QT_VERSION_STR) >= Version('5.4.0'):
            from PySide2.QtWidgets import QOpenGLWidget as QGLWidget
            from PySide2.QtGui import QSurfaceFormat as QGLFormat
            QT5_NEW_API = True
        else:
            from PySide2.QtOpenGL import QGLWidget, QGLFormat
    from PySide2 import QtGui, QtCore, QtWidgets, QtTest
    (QWidget, QApplication) = (QtWidgets.QWidget, QtWidgets.QApplication)
elif qt_lib == 'pyside':
    _check_imports('PySide')
    if not USE_EGL:
        from PySide.QtOpenGL import QGLWidget, QGLFormat
    from PySide import QtGui, QtCore, QtTest
    (QWidget, QApplication) = (QtGui.QWidget, QtGui.QApplication)
elif qt_lib:
    raise RuntimeError('Invalid value for qt_lib %r.' % qt_lib)
else:
    raise RuntimeError('Module backends._qt should not be imported directly.')
qt_keys = QtCore.Qt.Key if qt_lib == 'pyqt6' else QtCore.Qt
KEYMAP = {qt_keys.Key_Shift: keys.SHIFT, qt_keys.Key_Control: keys.CONTROL, qt_keys.Key_Alt: keys.ALT, qt_keys.Key_AltGr: keys.ALT, qt_keys.Key_Meta: keys.META, qt_keys.Key_Left: keys.LEFT, qt_keys.Key_Up: keys.UP, qt_keys.Key_Right: keys.RIGHT, qt_keys.Key_Down: keys.DOWN, qt_keys.Key_PageUp: keys.PAGEUP, qt_keys.Key_PageDown: keys.PAGEDOWN, qt_keys.Key_Insert: keys.INSERT, qt_keys.Key_Delete: keys.DELETE, qt_keys.Key_Home: keys.HOME, qt_keys.Key_End: keys.END, qt_keys.Key_Escape: keys.ESCAPE, qt_keys.Key_Backspace: keys.BACKSPACE, qt_keys.Key_F1: keys.F1, qt_keys.Key_F2: keys.F2, qt_keys.Key_F3: keys.F3, qt_keys.Key_F4: keys.F4, qt_keys.Key_F5: keys.F5, qt_keys.Key_F6: keys.F6, qt_keys.Key_F7: keys.F7, qt_keys.Key_F8: keys.F8, qt_keys.Key_F9: keys.F9, qt_keys.Key_F10: keys.F10, qt_keys.Key_F11: keys.F11, qt_keys.Key_F12: keys.F12, qt_keys.Key_Space: keys.SPACE, qt_keys.Key_Enter: keys.ENTER, qt_keys.Key_Return: keys.ENTER, qt_keys.Key_Tab: keys.TAB}
if PYQT6_API or PYSIDE6_API:
    BUTTONMAP = {QtCore.Qt.MouseButton.NoButton: 0, QtCore.Qt.MouseButton.LeftButton: 1, QtCore.Qt.MouseButton.RightButton: 2, QtCore.Qt.MouseButton.MiddleButton: 3, QtCore.Qt.MouseButton.BackButton: 4, QtCore.Qt.MouseButton.ForwardButton: 5}
else:
    BUTTONMAP = {0: 0, 1: 1, 2: 2, 4: 3, 8: 4, 16: 5}

def message_handler(*args):
    if False:
        i = 10
        return i + 15
    if qt_lib in ('pyqt4', 'pyside'):
        (msg_type, msg) = args
    elif qt_lib in ('pyqt5', 'pyqt6', 'pyside2', 'pyside6'):
        (msg_type, context, msg) = args
    elif qt_lib:
        raise RuntimeError('Invalid value for qt_lib %r.' % qt_lib)
    else:
        raise RuntimeError('Module backends._qt ', 'should not be imported directly.')
    BLACKLIST = ['QCocoaView handleTabletEvent: This tablet device is unknown', 'QSocketNotifier: Multiple socket notifiers for same']
    for item in BLACKLIST:
        if msg.startswith(item):
            return
    msg = msg.decode() if not isinstance(msg, str) else msg
    logger.warning(msg)

def use_shared_contexts():
    if False:
        for i in range(10):
            print('nop')
    'Enable context sharing for PyQt5 5.4+ API applications.\n\n    This is disabled by default for PyQt5 5.4+ due to occasional segmentation\n    faults and other issues when contexts are shared.\n\n    '
    forced_env_var = os.getenv('VISPY_PYQT5_SHARE_CONTEXT', 'false').lower() == 'true'
    return not (QT5_NEW_API or PYSIDE6_API or PYQT6_API) or forced_env_var
try:
    QtCore.qInstallMsgHandler(message_handler)
except AttributeError:
    QtCore.qInstallMessageHandler(message_handler)
capability = dict(title=True, size=True, position=True, show=True, vsync=True, resizable=True, decorate=True, fullscreen=True, context=use_shared_contexts(), multi_window=True, scroll=True, parent=True, always_on_top=True)

def _set_config(c):
    if False:
        for i in range(10):
            print('nop')
    'Set the OpenGL configuration'
    glformat = QGLFormat()
    glformat.setRedBufferSize(c['red_size'])
    glformat.setGreenBufferSize(c['green_size'])
    glformat.setBlueBufferSize(c['blue_size'])
    glformat.setAlphaBufferSize(c['alpha_size'])
    if QT5_NEW_API:
        glformat.setSwapBehavior(glformat.DoubleBuffer if c['double_buffer'] else glformat.SingleBuffer)
    elif PYQT6_API or PYSIDE6_API:
        glformat.setSwapBehavior(glformat.SwapBehavior.DoubleBuffer if c['double_buffer'] else glformat.SwapBehavior.SingleBuffer)
    else:
        glformat.setAccum(False)
        glformat.setRgba(True)
        glformat.setDoubleBuffer(True if c['double_buffer'] else False)
        glformat.setDepth(True if c['depth_size'] else False)
        glformat.setStencil(True if c['stencil_size'] else False)
        glformat.setSampleBuffers(True if c['samples'] else False)
    glformat.setDepthBufferSize(c['depth_size'] if c['depth_size'] else 0)
    glformat.setStencilBufferSize(c['stencil_size'] if c['stencil_size'] else 0)
    glformat.setSamples(c['samples'] if c['samples'] else 0)
    glformat.setStereo(c['stereo'])
    return glformat

class ApplicationBackend(BaseApplicationBackend):

    def __init__(self):
        if False:
            return 10
        BaseApplicationBackend.__init__(self)
        if (QT5_NEW_API or PYSIDE6_API) and use_shared_contexts():
            QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts, True)
        elif PYQT6_API and use_shared_contexts():
            QApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)

    def _vispy_get_backend_name(self):
        if False:
            i = 10
            return i + 15
        name = QtCore.__name__.split('.')[0]
        return name

    def _vispy_process_events(self):
        if False:
            return 10
        app = self._vispy_get_native_app()
        app.sendPostedEvents()
        app.processEvents()

    def _vispy_run(self):
        if False:
            print('Hello World!')
        app = self._vispy_get_native_app()
        if hasattr(app, '_in_event_loop') and app._in_event_loop:
            pass
        else:
            exec_func = app.exec if hasattr(app, 'exec') else app.exec_
            return exec_func()

    def _vispy_quit(self):
        if False:
            i = 10
            return i + 15
        return self._vispy_get_native_app().quit()

    def _vispy_get_native_app(self):
        if False:
            return 10
        app = QApplication.instance()
        if app is None:
            app = QApplication([''])
        QtGui._qApp = app
        return app

    def _vispy_sleep(self, duration_sec):
        if False:
            while True:
                i = 10
        QtTest.QTest.qWait(duration_sec * 1000)

def _get_qpoint_pos(pos):
    if False:
        while True:
            i = 10
    'Return the coordinates of a QPointF object.'
    return (pos.x(), pos.y())

class QtBaseCanvasBackend(BaseCanvasBackend):
    """Base functionality of Qt backend. No OpenGL Stuff."""

    def __init__(self, vispy_canvas, **kwargs):
        if False:
            return 10
        BaseCanvasBackend.__init__(self, vispy_canvas)
        p = self._process_backend_kwargs(kwargs)
        self._initialized = False
        self._init_specific(p, kwargs)
        assert self._initialized
        self.setMouseTracking(True)
        self._vispy_set_title(p.title)
        self._vispy_set_size(*p.size)
        if p.fullscreen is not False:
            if p.fullscreen is not True:
                logger.warning('Cannot specify monitor number for Qt fullscreen, using default')
            self._fullscreen = True
        else:
            self._fullscreen = False
        if hasattr(self, 'devicePixelRatioF'):
            ratio = self.devicePixelRatioF()
        else:
            ratio = 1
        self._physical_size = (p.size[0] * ratio, p.size[1] * ratio)
        if not p.resizable:
            self.setFixedSize(self.size())
        if p.position is not None:
            self._vispy_set_position(*p.position)
        if p.show:
            self._vispy_set_visible(True)
        self._double_click_supported = True
        try:
            self.window().windowHandle().screenChanged.connect(self.screen_changed)
        except AttributeError:
            pass
        self._native_gesture_scale_values = []
        self._native_gesture_rotation_values = []

    def screen_changed(self, new_screen):
        if False:
            i = 10
            return i + 15
        'Window moved from one display to another, resize canvas.\n\n        If display resolutions are the same this is essentially a no-op except for the redraw.\n        If the display resolutions differ (HiDPI versus regular displays) the canvas needs to\n        be redrawn to reset the physical size based on the current `devicePixelRatioF()` and\n        redrawn with that new size.\n\n        '
        self.resizeGL(*self._vispy_get_size())

    def _vispy_warmup(self):
        if False:
            for i in range(10):
                print('nop')
        etime = time() + 0.25
        while time() < etime:
            sleep(0.01)
            self._vispy_canvas.set_current()
            self._vispy_canvas.app.process_events()

    def _vispy_set_title(self, title):
        if False:
            print('Hello World!')
        if self._vispy_canvas is None:
            return
        self.setWindowTitle(title)

    def _vispy_set_size(self, w, h):
        if False:
            for i in range(10):
                print('nop')
        self.resize(w, h)

    def _vispy_set_physical_size(self, w, h):
        if False:
            return 10
        self._physical_size = (w, h)

    def _vispy_get_physical_size(self):
        if False:
            return 10
        if self._vispy_canvas is None:
            return
        return self._physical_size

    def _vispy_set_position(self, x, y):
        if False:
            while True:
                i = 10
        self.move(x, y)

    def _vispy_set_visible(self, visible):
        if False:
            for i in range(10):
                print('nop')
        if visible:
            if self._fullscreen:
                self.showFullScreen()
            else:
                self.showNormal()
        else:
            self.hide()

    def _vispy_set_fullscreen(self, fullscreen):
        if False:
            i = 10
            return i + 15
        self._fullscreen = bool(fullscreen)
        self._vispy_set_visible(True)

    def _vispy_get_fullscreen(self):
        if False:
            i = 10
            return i + 15
        return self._fullscreen

    def _vispy_update(self):
        if False:
            print('Hello World!')
        if self._vispy_canvas is None:
            return
        self.update()

    def _vispy_get_position(self):
        if False:
            while True:
                i = 10
        g = self.geometry()
        return (g.x(), g.y())

    def _vispy_get_size(self):
        if False:
            while True:
                i = 10
        g = self.geometry()
        return (g.width(), g.height())

    def sizeHint(self):
        if False:
            return 10
        return self.size()

    def mousePressEvent(self, ev):
        if False:
            for i in range(10):
                print('nop')
        if self._vispy_canvas is None:
            return
        self._vispy_mouse_press(native=ev, pos=_get_event_xy(ev), button=BUTTONMAP.get(ev.button(), 0), modifiers=self._modifiers(ev))

    def mouseReleaseEvent(self, ev):
        if False:
            while True:
                i = 10
        if self._vispy_canvas is None:
            return
        self._vispy_mouse_release(native=ev, pos=_get_event_xy(ev), button=BUTTONMAP[ev.button()], modifiers=self._modifiers(ev))

    def mouseDoubleClickEvent(self, ev):
        if False:
            print('Hello World!')
        if self._vispy_canvas is None:
            return
        self._vispy_mouse_double_click(native=ev, pos=_get_event_xy(ev), button=BUTTONMAP.get(ev.button(), 0), modifiers=self._modifiers(ev))

    def mouseMoveEvent(self, ev):
        if False:
            while True:
                i = 10
        if self._vispy_canvas is None:
            return
        self._vispy_mouse_move(native=ev, pos=_get_event_xy(ev), modifiers=self._modifiers(ev))

    def wheelEvent(self, ev):
        if False:
            print('Hello World!')
        if self._vispy_canvas is None:
            return
        (deltax, deltay) = (0.0, 0.0)
        if hasattr(ev, 'orientation'):
            if ev.orientation == QtCore.Qt.Horizontal:
                deltax = ev.delta() / 120.0
            else:
                deltay = ev.delta() / 120.0
        else:
            delta = ev.angleDelta()
            (deltax, deltay) = (delta.x() / 120.0, delta.y() / 120.0)
        self._vispy_canvas.events.mouse_wheel(native=ev, delta=(deltax, deltay), pos=_get_event_xy(ev), modifiers=self._modifiers(ev))

    def keyPressEvent(self, ev):
        if False:
            while True:
                i = 10
        self._keyEvent(self._vispy_canvas.events.key_press, ev)

    def keyReleaseEvent(self, ev):
        if False:
            return 10
        self._keyEvent(self._vispy_canvas.events.key_release, ev)

    def _handle_native_gesture_event(self, ev):
        if False:
            i = 10
            return i + 15
        if self._vispy_canvas is None:
            return
        t = ev.gestureType()
        try:
            pos = self.mapFromGlobal(ev.globalPosition().toPoint())
        except AttributeError:
            pos = self.mapFromGlobal(ev.globalPos())
        pos = (pos.x(), pos.y())
        if t == QtCore.Qt.NativeGestureType.BeginNativeGesture:
            self._vispy_canvas.events.touch(type='gesture_begin', pos=_get_event_xy(ev))
        elif t == QtCore.Qt.NativeGestureType.EndNativeGesture:
            self._native_touch_total_rotation = []
            self._native_touch_total_scale = []
            self._vispy_canvas.events.touch(type='gesture_end', pos=_get_event_xy(ev))
        elif t == QtCore.Qt.NativeGestureType.RotateNativeGesture:
            angle = ev.value()
            last_angle = self._native_gesture_rotation_values[-1] if self._native_gesture_rotation_values else None
            self._native_gesture_rotation_values.append(angle)
            total_rotation_angle = math.fsum(self._native_gesture_rotation_values)
            self._vispy_canvas.events.touch(type='gesture_rotate', pos=pos, rotation=angle, last_rotation=last_angle, total_rotation_angle=total_rotation_angle)
        elif t == QtCore.Qt.NativeGestureType.ZoomNativeGesture:
            scale = ev.value()
            last_scale = self._native_gesture_scale_values[-1] if self._native_gesture_scale_values else None
            self._native_gesture_scale_values.append(scale)
            total_scale_factor = math.fsum(self._native_gesture_scale_values)
            self._vispy_canvas.events.touch(type='gesture_zoom', pos=pos, last_scale=last_scale, scale=scale, total_scale_factor=total_scale_factor)

    def event(self, ev):
        if False:
            print('Hello World!')
        out = super(QtBaseCanvasBackend, self).event(ev)
        if (QT5_NEW_API or PYSIDE6_API or PYQT6_API) and isinstance(ev, QtGui.QNativeGestureEvent):
            self._handle_native_gesture_event(ev)
        return out

    def _keyEvent(self, func, ev):
        if False:
            print('Hello World!')
        key = int(ev.key())
        if key in KEYMAP:
            key = KEYMAP[key]
        elif 32 <= key <= 127:
            key = keys.Key(chr(key))
        else:
            key = None
        mod = self._modifiers(ev)
        func(native=ev, key=key, text=str(ev.text()), modifiers=mod)

    def _modifiers(self, event):
        if False:
            print('Hello World!')
        mod = ()
        qtmod = event.modifiers()
        qt_keyboard_modifiers = QtCore.Qt.KeyboardModifier if PYQT6_API else QtCore.Qt
        for (q, v) in ([qt_keyboard_modifiers.ShiftModifier, keys.SHIFT], [qt_keyboard_modifiers.ControlModifier, keys.CONTROL], [qt_keyboard_modifiers.AltModifier, keys.ALT], [qt_keyboard_modifiers.MetaModifier, keys.META]):
            if qtmod & q:
                mod += (v,)
        return mod
_EGL_DISPLAY = None
egl = None

class CanvasBackendEgl(QtBaseCanvasBackend, QWidget):

    def _init_specific(self, p, kwargs):
        if False:
            while True:
                i = 10
        global _EGL_DISPLAY
        global egl
        if egl is None:
            from ...ext import egl as _egl
            egl = _egl
            if IS_LINUX and (not IS_RPI):
                os.environ['EGL_SOFTWARE'] = 'true'
            _EGL_DISPLAY = egl.eglGetDisplay()
            CanvasBackendEgl._EGL_VERSION = egl.eglInitialize(_EGL_DISPLAY)
            atexit.register(egl.eglTerminate, _EGL_DISPLAY)
        p.context.shared.add_ref('qt-egl', self)
        if p.context.shared.ref is self:
            self._native_config = c = egl.eglChooseConfig(_EGL_DISPLAY)[0]
            self._native_context = egl.eglCreateContext(_EGL_DISPLAY, c, None)
        else:
            self._native_config = p.context.shared.ref._native_config
            self._native_context = p.context.shared.ref._native_context
        qt_window_types = QtCore.Qt.WindowType if PYQT6_API else QtCore.Qt
        if p.always_on_top or not p.decorate:
            hint = 0
            hint |= 0 if p.decorate else qt_window_types.FramelessWindowHint
            hint |= qt_window_types.WindowStaysOnTopHint if p.always_on_top else 0
        else:
            hint = qt_window_types.Widget
        QWidget.__init__(self, p.parent, hint)
        qt_window_attributes = QtCore.Qt.WidgetAttribute if PYQT6_API else QtCore.Qt
        if 0:
            self.setAutoFillBackground(False)
            self.setAttribute(qt_window_attributes.WA_NoSystemBackground, True)
            self.setAttribute(qt_window_attributes.WA_OpaquePaintEvent, True)
        elif IS_WIN:
            self.setAttribute(qt_window_attributes.WA_PaintOnScreen, True)
            self.setAutoFillBackground(False)
        w = self.get_window_id()
        self._surface = egl.eglCreateWindowSurface(_EGL_DISPLAY, c, w)
        self.initializeGL()
        self._initialized = True

    def get_window_id(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the window id of a PySide Widget. Might also work for PyQt4.'
        winid = self.winId()
        if IS_RPI:
            nw = (ctypes.c_int * 3)(winid, self.width(), self.height())
            return ctypes.pointer(nw)
        elif IS_LINUX:
            return int(winid)
        ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
        ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        name = ctypes.pythonapi.PyCapsule_GetName(winid)
        handle = ctypes.pythonapi.PyCapsule_GetPointer(winid, name)
        return handle

    def _vispy_close(self):
        if False:
            i = 10
            return i + 15
        if self._surface is not None:
            egl.eglDestroySurface(_EGL_DISPLAY, self._surface)
            self._surface = None
        self.close()

    def _vispy_set_current(self):
        if False:
            for i in range(10):
                print('nop')
        egl.eglMakeCurrent(_EGL_DISPLAY, self._surface, self._surface, self._native_context)

    def _vispy_swap_buffers(self):
        if False:
            while True:
                i = 10
        egl.eglSwapBuffers(_EGL_DISPLAY, self._surface)

    def initializeGL(self):
        if False:
            while True:
                i = 10
        self._vispy_canvas.set_current()
        self._vispy_canvas.events.initialize()

    def resizeEvent(self, event):
        if False:
            i = 10
            return i + 15
        (w, h) = (event.size().width(), event.size().height())
        self._vispy_canvas.events.resize(size=(w, h))

    def paintEvent(self, event):
        if False:
            while True:
                i = 10
        self._vispy_canvas.events.draw(region=None)
        if IS_LINUX or IS_RPI:
            from ... import gloo
            import numpy as np
            if not hasattr(self, '_gl_buffer'):
                self._gl_buffer = np.ones(3000 * 3000 * 4, np.uint8) * 255
            im = gloo.read_pixels()
            sze = im.shape[0] * im.shape[1]
            self._gl_buffer[0:0 + sze * 4:4] = im[:, :, 2].ravel()
            self._gl_buffer[1:0 + sze * 4:4] = im[:, :, 1].ravel()
            self._gl_buffer[2:2 + sze * 4:4] = im[:, :, 0].ravel()
            img = QtGui.QImage(self._gl_buffer, im.shape[1], im.shape[0], QtGui.QImage.Format_RGB32)
            painter = QtGui.QPainter()
            painter.begin(self)
            rect = QtCore.QRect(0, 0, self.width(), self.height())
            painter.drawImage(rect, img)
            painter.end()

    def paintEngine(self):
        if False:
            while True:
                i = 10
        if IS_LINUX and (not IS_RPI):
            return QWidget.paintEngine(self)
        else:
            return None

class CanvasBackendDesktop(QtBaseCanvasBackend, QGLWidget):

    def _init_specific(self, p, kwargs):
        if False:
            print('Hello World!')
        glformat = _set_config(p.context.config)
        glformat.setSwapInterval(1 if p.vsync else 0)
        widget = kwargs.pop('shareWidget', None) or self
        p.context.shared.add_ref('qt', widget)
        if p.context.shared.ref is widget:
            if widget is self:
                widget = None
        else:
            widget = p.context.shared.ref
            if 'shareWidget' in kwargs:
                raise RuntimeError('Cannot use vispy to share context and use built-in shareWidget.')
        qt_window_types = QtCore.Qt.WindowType if PYQT6_API else QtCore.Qt
        if p.always_on_top or not p.decorate:
            hint = 0
            hint |= 0 if p.decorate else qt_window_types.FramelessWindowHint
            hint |= qt_window_types.WindowStaysOnTopHint if p.always_on_top else 0
        else:
            hint = qt_window_types.Widget
        if QT5_NEW_API or PYSIDE6_API or PYQT6_API:
            QGLWidget.__init__(self, p.parent, hint)
            self._secondary_context = QtGui.QOpenGLContext()
            self._secondary_context.setShareContext(self.context())
            self._secondary_context.setFormat(glformat)
            self._secondary_context.create()
            self._surface = QtGui.QOffscreenSurface()
            self._surface.setFormat(glformat)
            self._surface.create()
            self._secondary_context.makeCurrent(self._surface)
        else:
            QGLWidget.__init__(self, p.parent, widget, hint)
            self._secondary_context = None
            self._surface = None
        self.setFormat(glformat)
        self._initialized = True
        if not QT5_NEW_API and (not PYSIDE6_API) and (not PYQT6_API) and (not self.isValid()):
            raise RuntimeError('context could not be created')
        if not QT5_NEW_API and (not PYSIDE6_API) and (not PYQT6_API):
            self.setAutoBufferSwap(False)
        qt_focus_policies = QtCore.Qt.FocusPolicy if PYQT6_API else QtCore.Qt
        self.setFocusPolicy(qt_focus_policies.WheelFocus)

    def _vispy_close(self):
        if False:
            return 10
        self.close()
        self.doneCurrent()
        if not QT5_NEW_API and (not PYSIDE6_API) and (not PYQT6_API):
            self.context().reset()
        if self._vispy_canvas is not None:
            self._vispy_canvas.app.process_events()
            self._vispy_canvas.app.process_events()

    def _vispy_set_current(self):
        if False:
            i = 10
            return i + 15
        if self._vispy_canvas is None:
            return
        if self.isValid():
            self.makeCurrent()

    def _vispy_swap_buffers(self):
        if False:
            return 10
        if self._vispy_canvas is None:
            return
        if QT5_NEW_API or PYSIDE6_API or PYQT6_API:
            ctx = self.context()
            ctx.swapBuffers(ctx.surface())
        else:
            self.swapBuffers()

    def _vispy_get_fb_bind_location(self):
        if False:
            print('Hello World!')
        if QT5_NEW_API or PYSIDE6_API or PYQT6_API:
            return self.defaultFramebufferObject()
        else:
            return QtBaseCanvasBackend._vispy_get_fb_bind_location(self)

    def initializeGL(self):
        if False:
            for i in range(10):
                print('nop')
        if self._vispy_canvas is None:
            return
        self._vispy_canvas.events.initialize()

    def resizeGL(self, w, h):
        if False:
            i = 10
            return i + 15
        if self._vispy_canvas is None:
            return
        if hasattr(self, 'devicePixelRatioF'):
            ratio = self.devicePixelRatioF()
            w = int(w * ratio)
            h = int(h * ratio)
        self._vispy_set_physical_size(w, h)
        self._vispy_canvas.events.resize(size=(self.width(), self.height()), physical_size=(w, h))

    def paintGL(self):
        if False:
            return 10
        if self._vispy_canvas is None:
            return
        self._vispy_canvas.set_current()
        self._vispy_canvas.events.draw(region=None)
        if QT5_NEW_API or PYSIDE6_API or PYQT6_API:
            context = self._vispy_canvas.context
            context.set_color_mask(False, False, False, True)
            context.clear(color=True, depth=False, stencil=False)
            context.set_color_mask(True, True, True, True)
            context.flush()
if USE_EGL:
    CanvasBackend = CanvasBackendEgl
else:
    CanvasBackend = CanvasBackendDesktop

class TimerBackend(BaseTimerBackend, QtCore.QTimer):

    def __init__(self, vispy_timer):
        if False:
            for i in range(10):
                print('nop')
        app = ApplicationBackend()
        app._vispy_get_native_app()
        BaseTimerBackend.__init__(self, vispy_timer)
        QtCore.QTimer.__init__(self)
        self.timeout.connect(self._vispy_timeout)

    def _vispy_start(self, interval):
        if False:
            while True:
                i = 10
        self.start(int(interval * 1000))

    def _vispy_stop(self):
        if False:
            i = 10
            return i + 15
        self.stop()

    def _vispy_timeout(self):
        if False:
            print('Hello World!')
        self._vispy_timer._timeout()