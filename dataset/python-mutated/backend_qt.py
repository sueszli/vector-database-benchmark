import functools
import os
import sys
import traceback
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import _Backend, FigureCanvasBase, FigureManagerBase, NavigationToolbar2, TimerBase, cursors, ToolContainerBase, MouseButton, CloseEvent, KeyEvent, LocationEvent, MouseEvent, ResizeEvent, _allow_interrupt
import matplotlib.backends.qt_editor.figureoptions as figureoptions
from . import qt_compat
from .qt_compat import QtCore, QtGui, QtWidgets, __version__, QT_API, _to_int, _isdeleted
SPECIAL_KEYS = {_to_int(getattr(QtCore.Qt.Key, k)): v for (k, v) in [('Key_Escape', 'escape'), ('Key_Tab', 'tab'), ('Key_Backspace', 'backspace'), ('Key_Return', 'enter'), ('Key_Enter', 'enter'), ('Key_Insert', 'insert'), ('Key_Delete', 'delete'), ('Key_Pause', 'pause'), ('Key_SysReq', 'sysreq'), ('Key_Clear', 'clear'), ('Key_Home', 'home'), ('Key_End', 'end'), ('Key_Left', 'left'), ('Key_Up', 'up'), ('Key_Right', 'right'), ('Key_Down', 'down'), ('Key_PageUp', 'pageup'), ('Key_PageDown', 'pagedown'), ('Key_Shift', 'shift'), ('Key_Control', 'control' if sys.platform != 'darwin' else 'cmd'), ('Key_Meta', 'meta' if sys.platform != 'darwin' else 'control'), ('Key_Alt', 'alt'), ('Key_CapsLock', 'caps_lock'), ('Key_F1', 'f1'), ('Key_F2', 'f2'), ('Key_F3', 'f3'), ('Key_F4', 'f4'), ('Key_F5', 'f5'), ('Key_F6', 'f6'), ('Key_F7', 'f7'), ('Key_F8', 'f8'), ('Key_F9', 'f9'), ('Key_F10', 'f10'), ('Key_F10', 'f11'), ('Key_F12', 'f12'), ('Key_Super_L', 'super'), ('Key_Super_R', 'super')]}
_MODIFIER_KEYS = [(_to_int(getattr(QtCore.Qt.KeyboardModifier, mod)), _to_int(getattr(QtCore.Qt.Key, key))) for (mod, key) in [('ControlModifier', 'Key_Control'), ('AltModifier', 'Key_Alt'), ('ShiftModifier', 'Key_Shift'), ('MetaModifier', 'Key_Meta')]]
cursord = {k: getattr(QtCore.Qt.CursorShape, v) for (k, v) in [(cursors.MOVE, 'SizeAllCursor'), (cursors.HAND, 'PointingHandCursor'), (cursors.POINTER, 'ArrowCursor'), (cursors.SELECT_REGION, 'CrossCursor'), (cursors.WAIT, 'WaitCursor'), (cursors.RESIZE_HORIZONTAL, 'SizeHorCursor'), (cursors.RESIZE_VERTICAL, 'SizeVerCursor')]}

@functools.lru_cache(1)
def _create_qApp():
    if False:
        i = 10
        return i + 15
    app = QtWidgets.QApplication.instance()
    if app is None:
        if not mpl._c_internal_utils.display_is_valid():
            raise RuntimeError('Invalid DISPLAY variable')
        if QT_API in {'PyQt6', 'PySide6'}:
            other_bindings = ('PyQt5', 'PySide2')
            qt_version = 6
        elif QT_API in {'PyQt5', 'PySide2'}:
            other_bindings = ('PyQt6', 'PySide6')
            qt_version = 5
        else:
            raise RuntimeError('Should never be here')
        for binding in other_bindings:
            mod = sys.modules.get(f'{binding}.QtWidgets')
            if mod is not None and mod.QApplication.instance() is not None:
                other_core = sys.modules.get(f'{binding}.QtCore')
                _api.warn_external(f'Matplotlib is using {QT_API} which wraps {QtCore.qVersion()} however an instantiated QApplication from {binding} which wraps {other_core.qVersion()} exists.  Mixing Qt major versions may not work as expected.')
                break
        if qt_version == 5:
            try:
                QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
            except AttributeError:
                pass
        try:
            QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        except AttributeError:
            pass
        app = QtWidgets.QApplication(['matplotlib'])
        if sys.platform == 'darwin':
            image = str(cbook._get_data_path('images/matplotlib.svg'))
            icon = QtGui.QIcon(image)
            app.setWindowIcon(icon)
        app.setQuitOnLastWindowClosed(True)
        cbook._setup_new_guiapp()
        if qt_version == 5:
            app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    return app

def _allow_interrupt_qt(qapp_or_eventloop):
    if False:
        i = 10
        return i + 15
    'A context manager that allows terminating a plot by sending a SIGINT.'

    def prepare_notifier(rsock):
        if False:
            return 10
        sn = QtCore.QSocketNotifier(rsock.fileno(), QtCore.QSocketNotifier.Type.Read)

        @sn.activated.connect
        def _may_clear_sock():
            if False:
                for i in range(10):
                    print('nop')
            try:
                rsock.recv(1)
            except BlockingIOError:
                pass
        return sn

    def handle_sigint():
        if False:
            return 10
        if hasattr(qapp_or_eventloop, 'closeAllWindows'):
            qapp_or_eventloop.closeAllWindows()
        qapp_or_eventloop.quit()
    return _allow_interrupt(prepare_notifier, handle_sigint)

class TimerQT(TimerBase):
    """Subclass of `.TimerBase` using QTimer events."""

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._on_timer)
        super().__init__(*args, **kwargs)

    def __del__(self):
        if False:
            return 10
        if not _isdeleted(self._timer):
            self._timer_stop()

    def _timer_set_single_shot(self):
        if False:
            while True:
                i = 10
        self._timer.setSingleShot(self._single)

    def _timer_set_interval(self):
        if False:
            return 10
        self._timer.setInterval(self._interval)

    def _timer_start(self):
        if False:
            print('Hello World!')
        self._timer.start()

    def _timer_stop(self):
        if False:
            return 10
        self._timer.stop()

class FigureCanvasQT(FigureCanvasBase, QtWidgets.QWidget):
    required_interactive_framework = 'qt'
    _timer_cls = TimerQT
    manager_class = _api.classproperty(lambda cls: FigureManagerQT)
    buttond = {getattr(QtCore.Qt.MouseButton, k): v for (k, v) in [('LeftButton', MouseButton.LEFT), ('RightButton', MouseButton.RIGHT), ('MiddleButton', MouseButton.MIDDLE), ('XButton1', MouseButton.BACK), ('XButton2', MouseButton.FORWARD)]}

    def __init__(self, figure=None):
        if False:
            return 10
        _create_qApp()
        super().__init__(figure=figure)
        self._draw_pending = False
        self._is_drawing = False
        self._draw_rect_callback = lambda painter: None
        self._in_resize_event = False
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setMouseTracking(True)
        self.resize(*self.get_width_height())
        palette = QtGui.QPalette(QtGui.QColor('white'))
        self.setPalette(palette)

    def _update_pixel_ratio(self):
        if False:
            i = 10
            return i + 15
        if self._set_device_pixel_ratio(self.devicePixelRatioF() or 1):
            event = QtGui.QResizeEvent(self.size(), self.size())
            self.resizeEvent(event)

    def _update_screen(self, screen):
        if False:
            print('Hello World!')
        self._update_pixel_ratio()
        if screen is not None:
            screen.physicalDotsPerInchChanged.connect(self._update_pixel_ratio)
            screen.logicalDotsPerInchChanged.connect(self._update_pixel_ratio)

    def showEvent(self, event):
        if False:
            print('Hello World!')
        window = self.window().windowHandle()
        window.screenChanged.connect(self._update_screen)
        self._update_screen(window.screen())

    def set_cursor(self, cursor):
        if False:
            i = 10
            return i + 15
        self.setCursor(_api.check_getitem(cursord, cursor=cursor))

    def mouseEventCoords(self, pos=None):
        if False:
            print('Hello World!')
        '\n        Calculate mouse coordinates in physical pixels.\n\n        Qt uses logical pixels, but the figure is scaled to physical\n        pixels for rendering.  Transform to physical pixels so that\n        all of the down-stream transforms work as expected.\n\n        Also, the origin is different and needs to be corrected.\n        '
        if pos is None:
            pos = self.mapFromGlobal(QtGui.QCursor.pos())
        elif hasattr(pos, 'position'):
            pos = pos.position()
        elif hasattr(pos, 'pos'):
            pos = pos.pos()
        x = pos.x()
        y = self.figure.bbox.height / self.device_pixel_ratio - pos.y()
        return (x * self.device_pixel_ratio, y * self.device_pixel_ratio)

    def enterEvent(self, event):
        if False:
            while True:
                i = 10
        mods = QtWidgets.QApplication.instance().queryKeyboardModifiers()
        if self.figure is None:
            return
        LocationEvent('figure_enter_event', self, *self.mouseEventCoords(event), modifiers=self._mpl_modifiers(mods), guiEvent=event)._process()

    def leaveEvent(self, event):
        if False:
            i = 10
            return i + 15
        QtWidgets.QApplication.restoreOverrideCursor()
        if self.figure is None:
            return
        LocationEvent('figure_leave_event', self, *self.mouseEventCoords(), modifiers=self._mpl_modifiers(), guiEvent=event)._process()

    def mousePressEvent(self, event):
        if False:
            i = 10
            return i + 15
        button = self.buttond.get(event.button())
        if button is not None and self.figure is not None:
            MouseEvent('button_press_event', self, *self.mouseEventCoords(event), button, modifiers=self._mpl_modifiers(), guiEvent=event)._process()

    def mouseDoubleClickEvent(self, event):
        if False:
            while True:
                i = 10
        button = self.buttond.get(event.button())
        if button is not None and self.figure is not None:
            MouseEvent('button_press_event', self, *self.mouseEventCoords(event), button, dblclick=True, modifiers=self._mpl_modifiers(), guiEvent=event)._process()

    def mouseMoveEvent(self, event):
        if False:
            print('Hello World!')
        if self.figure is None:
            return
        MouseEvent('motion_notify_event', self, *self.mouseEventCoords(event), modifiers=self._mpl_modifiers(), guiEvent=event)._process()

    def mouseReleaseEvent(self, event):
        if False:
            print('Hello World!')
        button = self.buttond.get(event.button())
        if button is not None and self.figure is not None:
            MouseEvent('button_release_event', self, *self.mouseEventCoords(event), button, modifiers=self._mpl_modifiers(), guiEvent=event)._process()

    def wheelEvent(self, event):
        if False:
            print('Hello World!')
        if event.pixelDelta().isNull() or QtWidgets.QApplication.instance().platformName() == 'xcb':
            steps = event.angleDelta().y() / 120
        else:
            steps = event.pixelDelta().y()
        if steps and self.figure is not None:
            MouseEvent('scroll_event', self, *self.mouseEventCoords(event), step=steps, modifiers=self._mpl_modifiers(), guiEvent=event)._process()

    def keyPressEvent(self, event):
        if False:
            while True:
                i = 10
        key = self._get_key(event)
        if key is not None and self.figure is not None:
            KeyEvent('key_press_event', self, key, *self.mouseEventCoords(), guiEvent=event)._process()

    def keyReleaseEvent(self, event):
        if False:
            while True:
                i = 10
        key = self._get_key(event)
        if key is not None and self.figure is not None:
            KeyEvent('key_release_event', self, key, *self.mouseEventCoords(), guiEvent=event)._process()

    def resizeEvent(self, event):
        if False:
            i = 10
            return i + 15
        if self._in_resize_event:
            return
        if self.figure is None:
            return
        self._in_resize_event = True
        try:
            w = event.size().width() * self.device_pixel_ratio
            h = event.size().height() * self.device_pixel_ratio
            dpival = self.figure.dpi
            winch = w / dpival
            hinch = h / dpival
            self.figure.set_size_inches(winch, hinch, forward=False)
            QtWidgets.QWidget.resizeEvent(self, event)
            ResizeEvent('resize_event', self)._process()
            self.draw_idle()
        finally:
            self._in_resize_event = False

    def sizeHint(self):
        if False:
            for i in range(10):
                print('nop')
        (w, h) = self.get_width_height()
        return QtCore.QSize(w, h)

    def minumumSizeHint(self):
        if False:
            for i in range(10):
                print('nop')
        return QtCore.QSize(10, 10)

    @staticmethod
    def _mpl_modifiers(modifiers=None, *, exclude=None):
        if False:
            i = 10
            return i + 15
        if modifiers is None:
            modifiers = QtWidgets.QApplication.instance().keyboardModifiers()
        modifiers = _to_int(modifiers)
        return [SPECIAL_KEYS[key].replace('control', 'ctrl') for (mask, key) in _MODIFIER_KEYS if exclude != key and modifiers & mask]

    def _get_key(self, event):
        if False:
            for i in range(10):
                print('nop')
        event_key = event.key()
        mods = self._mpl_modifiers(exclude=event_key)
        try:
            key = SPECIAL_KEYS[event_key]
        except KeyError:
            if event_key > sys.maxunicode:
                return None
            key = chr(event_key)
            if 'shift' in mods:
                mods.remove('shift')
            else:
                key = key.lower()
        return '+'.join(mods + [key])

    def flush_events(self):
        if False:
            return 10
        QtWidgets.QApplication.instance().processEvents()

    def start_event_loop(self, timeout=0):
        if False:
            return 10
        if hasattr(self, '_event_loop') and self._event_loop.isRunning():
            raise RuntimeError('Event loop already running')
        self._event_loop = event_loop = QtCore.QEventLoop()
        if timeout > 0:
            _ = QtCore.QTimer.singleShot(int(timeout * 1000), event_loop.quit)
        with _allow_interrupt_qt(event_loop):
            qt_compat._exec(event_loop)

    def stop_event_loop(self, event=None):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, '_event_loop'):
            self._event_loop.quit()

    def draw(self):
        if False:
            while True:
                i = 10
        'Render the figure, and queue a request for a Qt draw.'
        if self._is_drawing:
            return
        with cbook._setattr_cm(self, _is_drawing=True):
            super().draw()
        self.update()

    def draw_idle(self):
        if False:
            i = 10
            return i + 15
        'Queue redraw of the Agg buffer and request Qt paintEvent.'
        if not (getattr(self, '_draw_pending', False) or getattr(self, '_is_drawing', False)):
            self._draw_pending = True
            QtCore.QTimer.singleShot(0, self._draw_idle)

    def blit(self, bbox=None):
        if False:
            while True:
                i = 10
        if bbox is None and self.figure:
            bbox = self.figure.bbox
        (l, b, w, h) = [int(pt / self.device_pixel_ratio) for pt in bbox.bounds]
        t = b + h
        self.repaint(l, self.rect().height() - t, w, h)

    def _draw_idle(self):
        if False:
            for i in range(10):
                print('nop')
        with self._idle_draw_cntx():
            if not self._draw_pending:
                return
            self._draw_pending = False
            if self.height() < 0 or self.width() < 0:
                return
            try:
                self.draw()
            except Exception:
                traceback.print_exc()

    def drawRectangle(self, rect):
        if False:
            while True:
                i = 10
        if rect is not None:
            (x0, y0, w, h) = [int(pt / self.device_pixel_ratio) for pt in rect]
            x1 = x0 + w
            y1 = y0 + h

            def _draw_rect_callback(painter):
                if False:
                    i = 10
                    return i + 15
                pen = QtGui.QPen(QtGui.QColor('black'), 1 / self.device_pixel_ratio)
                pen.setDashPattern([3, 3])
                for (color, offset) in [(QtGui.QColor('black'), 0), (QtGui.QColor('white'), 3)]:
                    pen.setDashOffset(offset)
                    pen.setColor(color)
                    painter.setPen(pen)
                    painter.drawLine(x0, y0, x0, y1)
                    painter.drawLine(x0, y0, x1, y0)
                    painter.drawLine(x0, y1, x1, y1)
                    painter.drawLine(x1, y0, x1, y1)
        else:

            def _draw_rect_callback(painter):
                if False:
                    return 10
                return
        self._draw_rect_callback = _draw_rect_callback
        self.update()

class MainWindow(QtWidgets.QMainWindow):
    closing = QtCore.Signal()

    def closeEvent(self, event):
        if False:
            while True:
                i = 10
        self.closing.emit()
        super().closeEvent(event)

class FigureManagerQT(FigureManagerBase):
    """
    Attributes
    ----------
    canvas : `FigureCanvas`
        The FigureCanvas instance
    num : int or str
        The Figure number
    toolbar : qt.QToolBar
        The qt.QToolBar
    window : qt.QMainWindow
        The qt.QMainWindow
    """

    def __init__(self, canvas, num):
        if False:
            for i in range(10):
                print('nop')
        self.window = MainWindow()
        super().__init__(canvas, num)
        self.window.closing.connect(self._widgetclosed)
        if sys.platform != 'darwin':
            image = str(cbook._get_data_path('images/matplotlib.svg'))
            icon = QtGui.QIcon(image)
            self.window.setWindowIcon(icon)
        self.window._destroying = False
        if self.toolbar:
            self.window.addToolBar(self.toolbar)
            tbs_height = self.toolbar.sizeHint().height()
        else:
            tbs_height = 0
        cs = canvas.sizeHint()
        cs_height = cs.height()
        height = cs_height + tbs_height
        self.window.resize(cs.width(), height)
        self.window.setCentralWidget(self.canvas)
        if mpl.is_interactive():
            self.window.show()
            self.canvas.draw_idle()
        self.canvas.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.canvas.setFocus()
        self.window.raise_()

    def full_screen_toggle(self):
        if False:
            print('Hello World!')
        if self.window.isFullScreen():
            self.window.showNormal()
        else:
            self.window.showFullScreen()

    def _widgetclosed(self):
        if False:
            return 10
        CloseEvent('close_event', self.canvas)._process()
        if self.window._destroying:
            return
        self.window._destroying = True
        try:
            Gcf.destroy(self)
        except AttributeError:
            pass

    def resize(self, width, height):
        if False:
            print('Hello World!')
        width = int(width / self.canvas.device_pixel_ratio)
        height = int(height / self.canvas.device_pixel_ratio)
        extra_width = self.window.width() - self.canvas.width()
        extra_height = self.window.height() - self.canvas.height()
        self.canvas.resize(width, height)
        self.window.resize(width + extra_width, height + extra_height)

    @classmethod
    def start_main_loop(cls):
        if False:
            print('Hello World!')
        qapp = QtWidgets.QApplication.instance()
        if qapp:
            with _allow_interrupt_qt(qapp):
                qt_compat._exec(qapp)

    def show(self):
        if False:
            i = 10
            return i + 15
        self.window._destroying = False
        self.window.show()
        if mpl.rcParams['figure.raise_window']:
            self.window.activateWindow()
            self.window.raise_()

    def destroy(self, *args):
        if False:
            i = 10
            return i + 15
        if QtWidgets.QApplication.instance() is None:
            return
        if self.window._destroying:
            return
        self.window._destroying = True
        if self.toolbar:
            self.toolbar.destroy()
        self.window.close()

    def get_window_title(self):
        if False:
            for i in range(10):
                print('nop')
        return self.window.windowTitle()

    def set_window_title(self, title):
        if False:
            for i in range(10):
                print('nop')
        self.window.setWindowTitle(title)

class NavigationToolbar2QT(NavigationToolbar2, QtWidgets.QToolBar):
    _message = QtCore.Signal(str)
    message = _api.deprecate_privatize_attribute('3.8')
    toolitems = [*NavigationToolbar2.toolitems]
    toolitems.insert([name for (name, *_) in toolitems].index('Subplots') + 1, ('Customize', 'Edit axis, curve and image parameters', 'qt4_editor_options', 'edit_parameters'))

    def __init__(self, canvas, parent=None, coordinates=True):
        if False:
            for i in range(10):
                print('nop')
        'coordinates: should we show the coordinates on the right?'
        QtWidgets.QToolBar.__init__(self, parent)
        self.setAllowedAreas(QtCore.Qt.ToolBarArea(_to_int(QtCore.Qt.ToolBarArea.TopToolBarArea) | _to_int(QtCore.Qt.ToolBarArea.BottomToolBarArea)))
        self.coordinates = coordinates
        self._actions = {}
        self._subplot_dialog = None
        for (text, tooltip_text, image_file, callback) in self.toolitems:
            if text is None:
                self.addSeparator()
            else:
                a = self.addAction(self._icon(image_file + '.png'), text, getattr(self, callback))
                self._actions[callback] = a
                if callback in ['zoom', 'pan']:
                    a.setCheckable(True)
                if tooltip_text is not None:
                    a.setToolTip(tooltip_text)
        if self.coordinates:
            self.locLabel = QtWidgets.QLabel('', self)
            self.locLabel.setAlignment(QtCore.Qt.AlignmentFlag(_to_int(QtCore.Qt.AlignmentFlag.AlignRight) | _to_int(QtCore.Qt.AlignmentFlag.AlignVCenter)))
            self.locLabel.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Ignored))
            labelAction = self.addWidget(self.locLabel)
            labelAction.setVisible(True)
        NavigationToolbar2.__init__(self, canvas)

    def _icon(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a `.QIcon` from an image file *name*, including the extension\n        and relative to Matplotlib\'s "images" data directory.\n        '
        path_regular = cbook._get_data_path('images', name)
        path_large = path_regular.with_name(path_regular.name.replace('.png', '_large.png'))
        filename = str(path_large if path_large.exists() else path_regular)
        pm = QtGui.QPixmap(filename)
        pm.setDevicePixelRatio(self.devicePixelRatioF() or 1)
        if self.palette().color(self.backgroundRole()).value() < 128:
            icon_color = self.palette().color(self.foregroundRole())
            mask = pm.createMaskFromColor(QtGui.QColor('black'), QtCore.Qt.MaskMode.MaskOutColor)
            pm.fill(icon_color)
            pm.setMask(mask)
        return QtGui.QIcon(pm)

    def edit_parameters(self):
        if False:
            while True:
                i = 10
        axes = self.canvas.figure.get_axes()
        if not axes:
            QtWidgets.QMessageBox.warning(self.canvas.parent(), 'Error', 'There are no axes to edit.')
            return
        elif len(axes) == 1:
            (ax,) = axes
        else:
            titles = [ax.get_label() or ax.get_title() or ax.get_title('left') or ax.get_title('right') or ' - '.join(filter(None, [ax.get_xlabel(), ax.get_ylabel()])) or f'<anonymous {type(ax).__name__}>' for ax in axes]
            duplicate_titles = [title for title in titles if titles.count(title) > 1]
            for (i, ax) in enumerate(axes):
                if titles[i] in duplicate_titles:
                    titles[i] += f' (id: {id(ax):#x})'
            (item, ok) = QtWidgets.QInputDialog.getItem(self.canvas.parent(), 'Customize', 'Select axes:', titles, 0, False)
            if not ok:
                return
            ax = axes[titles.index(item)]
        figureoptions.figure_edit(ax, self)

    def _update_buttons_checked(self):
        if False:
            for i in range(10):
                print('nop')
        if 'pan' in self._actions:
            self._actions['pan'].setChecked(self.mode.name == 'PAN')
        if 'zoom' in self._actions:
            self._actions['zoom'].setChecked(self.mode.name == 'ZOOM')

    def pan(self, *args):
        if False:
            print('Hello World!')
        super().pan(*args)
        self._update_buttons_checked()

    def zoom(self, *args):
        if False:
            return 10
        super().zoom(*args)
        self._update_buttons_checked()

    def set_message(self, s):
        if False:
            i = 10
            return i + 15
        self._message.emit(s)
        if self.coordinates:
            self.locLabel.setText(s)

    def draw_rubberband(self, event, x0, y0, x1, y1):
        if False:
            print('Hello World!')
        height = self.canvas.figure.bbox.height
        y1 = height - y1
        y0 = height - y0
        rect = [int(val) for val in (x0, y0, x1 - x0, y1 - y0)]
        self.canvas.drawRectangle(rect)

    def remove_rubberband(self):
        if False:
            return 10
        self.canvas.drawRectangle(None)

    def configure_subplots(self):
        if False:
            while True:
                i = 10
        if self._subplot_dialog is None:
            self._subplot_dialog = SubplotToolQt(self.canvas.figure, self.canvas.parent())
            self.canvas.mpl_connect('close_event', lambda e: self._subplot_dialog.reject())
        self._subplot_dialog.update_from_current_subplotpars()
        self._subplot_dialog.setModal(True)
        self._subplot_dialog.show()
        return self._subplot_dialog

    def save_figure(self, *args):
        if False:
            return 10
        filetypes = self.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = sorted(filetypes.items())
        default_filetype = self.canvas.get_default_filetype()
        startpath = os.path.expanduser(mpl.rcParams['savefig.directory'])
        start = os.path.join(startpath, self.canvas.get_default_filename())
        filters = []
        selectedFilter = None
        for (name, exts) in sorted_filetypes:
            exts_list = ' '.join(['*.%s' % ext for ext in exts])
            filter = f'{name} ({exts_list})'
            if default_filetype in exts:
                selectedFilter = filter
            filters.append(filter)
        filters = ';;'.join(filters)
        (fname, filter) = QtWidgets.QFileDialog.getSaveFileName(self.canvas.parent(), 'Choose a filename to save to', start, filters, selectedFilter)
        if fname:
            if startpath != '':
                mpl.rcParams['savefig.directory'] = os.path.dirname(fname)
            try:
                self.canvas.figure.savefig(fname)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'Error saving file', str(e), QtWidgets.QMessageBox.StandardButton.Ok, QtWidgets.QMessageBox.StandardButton.NoButton)

    def set_history_buttons(self):
        if False:
            while True:
                i = 10
        can_backward = self._nav_stack._pos > 0
        can_forward = self._nav_stack._pos < len(self._nav_stack) - 1
        if 'back' in self._actions:
            self._actions['back'].setEnabled(can_backward)
        if 'forward' in self._actions:
            self._actions['forward'].setEnabled(can_forward)

class SubplotToolQt(QtWidgets.QDialog):

    def __init__(self, targetfig, parent):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.setWindowIcon(QtGui.QIcon(str(cbook._get_data_path('images/matplotlib.png'))))
        self.setObjectName('SubplotTool')
        self._spinboxes = {}
        main_layout = QtWidgets.QHBoxLayout()
        self.setLayout(main_layout)
        for (group, spinboxes, buttons) in [('Borders', ['top', 'bottom', 'left', 'right'], [('Export values', self._export_values)]), ('Spacings', ['hspace', 'wspace'], [('Tight layout', self._tight_layout), ('Reset', self._reset), ('Close', self.close)])]:
            layout = QtWidgets.QVBoxLayout()
            main_layout.addLayout(layout)
            box = QtWidgets.QGroupBox(group)
            layout.addWidget(box)
            inner = QtWidgets.QFormLayout(box)
            for name in spinboxes:
                self._spinboxes[name] = spinbox = QtWidgets.QDoubleSpinBox()
                spinbox.setRange(0, 1)
                spinbox.setDecimals(3)
                spinbox.setSingleStep(0.005)
                spinbox.setKeyboardTracking(False)
                spinbox.valueChanged.connect(self._on_value_changed)
                inner.addRow(name, spinbox)
            layout.addStretch(1)
            for (name, method) in buttons:
                button = QtWidgets.QPushButton(name)
                button.setAutoDefault(False)
                button.clicked.connect(method)
                layout.addWidget(button)
                if name == 'Close':
                    button.setFocus()
        self._figure = targetfig
        self._defaults = {}
        self._export_values_dialog = None
        self.update_from_current_subplotpars()

    def update_from_current_subplotpars(self):
        if False:
            i = 10
            return i + 15
        self._defaults = {spinbox: getattr(self._figure.subplotpars, name) for (name, spinbox) in self._spinboxes.items()}
        self._reset()

    def _export_values(self):
        if False:
            while True:
                i = 10
        self._export_values_dialog = QtWidgets.QDialog()
        layout = QtWidgets.QVBoxLayout()
        self._export_values_dialog.setLayout(layout)
        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        layout.addWidget(text)
        text.setPlainText(',\n'.join((f'{attr}={spinbox.value():.3}' for (attr, spinbox) in self._spinboxes.items())))
        size = text.maximumSize()
        size.setHeight(QtGui.QFontMetrics(text.document().defaultFont()).size(0, text.toPlainText()).height() + 20)
        text.setMaximumSize(size)
        self._export_values_dialog.show()

    def _on_value_changed(self):
        if False:
            print('Hello World!')
        spinboxes = self._spinboxes
        for (lower, higher) in [('bottom', 'top'), ('left', 'right')]:
            spinboxes[higher].setMinimum(spinboxes[lower].value() + 0.001)
            spinboxes[lower].setMaximum(spinboxes[higher].value() - 0.001)
        self._figure.subplots_adjust(**{attr: spinbox.value() for (attr, spinbox) in spinboxes.items()})
        self._figure.canvas.draw_idle()

    def _tight_layout(self):
        if False:
            print('Hello World!')
        self._figure.tight_layout()
        for (attr, spinbox) in self._spinboxes.items():
            spinbox.blockSignals(True)
            spinbox.setValue(getattr(self._figure.subplotpars, attr))
            spinbox.blockSignals(False)
        self._figure.canvas.draw_idle()

    def _reset(self):
        if False:
            i = 10
            return i + 15
        for (spinbox, value) in self._defaults.items():
            spinbox.setRange(0, 1)
            spinbox.blockSignals(True)
            spinbox.setValue(value)
            spinbox.blockSignals(False)
        self._on_value_changed()

class ToolbarQt(ToolContainerBase, QtWidgets.QToolBar):

    def __init__(self, toolmanager, parent=None):
        if False:
            for i in range(10):
                print('nop')
        ToolContainerBase.__init__(self, toolmanager)
        QtWidgets.QToolBar.__init__(self, parent)
        self.setAllowedAreas(QtCore.Qt.ToolBarArea(_to_int(QtCore.Qt.ToolBarArea.TopToolBarArea) | _to_int(QtCore.Qt.ToolBarArea.BottomToolBarArea)))
        message_label = QtWidgets.QLabel('')
        message_label.setAlignment(QtCore.Qt.AlignmentFlag(_to_int(QtCore.Qt.AlignmentFlag.AlignRight) | _to_int(QtCore.Qt.AlignmentFlag.AlignVCenter)))
        message_label.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Ignored))
        self._message_action = self.addWidget(message_label)
        self._toolitems = {}
        self._groups = {}

    def add_toolitem(self, name, group, position, image_file, description, toggle):
        if False:
            while True:
                i = 10
        button = QtWidgets.QToolButton(self)
        if image_file:
            button.setIcon(NavigationToolbar2QT._icon(self, image_file))
        button.setText(name)
        if description:
            button.setToolTip(description)

        def handler():
            if False:
                i = 10
                return i + 15
            self.trigger_tool(name)
        if toggle:
            button.setCheckable(True)
            button.toggled.connect(handler)
        else:
            button.clicked.connect(handler)
        self._toolitems.setdefault(name, [])
        self._add_to_group(group, name, button, position)
        self._toolitems[name].append((button, handler))

    def _add_to_group(self, group, name, button, position):
        if False:
            print('Hello World!')
        gr = self._groups.get(group, [])
        if not gr:
            sep = self.insertSeparator(self._message_action)
            gr.append(sep)
        before = gr[position]
        widget = self.insertWidget(before, button)
        gr.insert(position, widget)
        self._groups[group] = gr

    def toggle_toolitem(self, name, toggled):
        if False:
            while True:
                i = 10
        if name not in self._toolitems:
            return
        for (button, handler) in self._toolitems[name]:
            button.toggled.disconnect(handler)
            button.setChecked(toggled)
            button.toggled.connect(handler)

    def remove_toolitem(self, name):
        if False:
            print('Hello World!')
        for (button, handler) in self._toolitems[name]:
            button.setParent(None)
        del self._toolitems[name]

    def set_message(self, s):
        if False:
            return 10
        self.widgetForAction(self._message_action).setText(s)

@backend_tools._register_tool_class(FigureCanvasQT)
class ConfigureSubplotsQt(backend_tools.ConfigureSubplotsBase):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self._subplot_dialog = None

    def trigger(self, *args):
        if False:
            return 10
        NavigationToolbar2QT.configure_subplots(self)

@backend_tools._register_tool_class(FigureCanvasQT)
class SaveFigureQt(backend_tools.SaveFigureBase):

    def trigger(self, *args):
        if False:
            for i in range(10):
                print('nop')
        NavigationToolbar2QT.save_figure(self._make_classic_style_pseudo_toolbar())

@backend_tools._register_tool_class(FigureCanvasQT)
class RubberbandQt(backend_tools.RubberbandBase):

    def draw_rubberband(self, x0, y0, x1, y1):
        if False:
            for i in range(10):
                print('nop')
        NavigationToolbar2QT.draw_rubberband(self._make_classic_style_pseudo_toolbar(), None, x0, y0, x1, y1)

    def remove_rubberband(self):
        if False:
            i = 10
            return i + 15
        NavigationToolbar2QT.remove_rubberband(self._make_classic_style_pseudo_toolbar())

@backend_tools._register_tool_class(FigureCanvasQT)
class HelpQt(backend_tools.ToolHelpBase):

    def trigger(self, *args):
        if False:
            while True:
                i = 10
        QtWidgets.QMessageBox.information(None, 'Help', self._get_help_html())

@backend_tools._register_tool_class(FigureCanvasQT)
class ToolCopyToClipboardQT(backend_tools.ToolCopyToClipboardBase):

    def trigger(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pixmap = self.canvas.grab()
        QtWidgets.QApplication.instance().clipboard().setPixmap(pixmap)
FigureManagerQT._toolbar2_class = NavigationToolbar2QT
FigureManagerQT._toolmanager_toolbar_class = ToolbarQt

@_Backend.export
class _BackendQT(_Backend):
    backend_version = __version__
    FigureCanvas = FigureCanvasQT
    FigureManager = FigureManagerQT
    mainloop = FigureManagerQT.start_main_loop