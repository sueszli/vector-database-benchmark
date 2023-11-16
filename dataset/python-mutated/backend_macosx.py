import os
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib._pylab_helpers import Gcf
from . import _macosx
from .backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import _Backend, FigureCanvasBase, FigureManagerBase, NavigationToolbar2, ResizeEvent, TimerBase, _allow_interrupt

class TimerMac(_macosx.Timer, TimerBase):
    """Subclass of `.TimerBase` using CFRunLoop timer events."""

def _allow_interrupt_macos():
    if False:
        for i in range(10):
            print('nop')
    'A context manager that allows terminating a plot by sending a SIGINT.'
    return _allow_interrupt(lambda rsock: _macosx.wake_on_fd_write(rsock.fileno()), _macosx.stop)

class FigureCanvasMac(FigureCanvasAgg, _macosx.FigureCanvas, FigureCanvasBase):
    required_interactive_framework = 'macosx'
    _timer_cls = TimerMac
    manager_class = _api.classproperty(lambda cls: FigureManagerMac)

    def __init__(self, figure):
        if False:
            while True:
                i = 10
        super().__init__(figure=figure)
        self._draw_pending = False
        self._is_drawing = False
        self._timers = set()

    def draw(self):
        if False:
            print('Hello World!')
        'Render the figure and update the macosx canvas.'
        if self._is_drawing:
            return
        with cbook._setattr_cm(self, _is_drawing=True):
            super().draw()
        self.update()

    def draw_idle(self):
        if False:
            for i in range(10):
                print('nop')
        if not (getattr(self, '_draw_pending', False) or getattr(self, '_is_drawing', False)):
            self._draw_pending = True
            self._single_shot_timer(self._draw_idle)

    def _single_shot_timer(self, callback):
        if False:
            return 10
        'Add a single shot timer with the given callback'

        def callback_func(callback, timer):
            if False:
                print('Hello World!')
            callback()
            self._timers.remove(timer)
            timer.stop()
        timer = self.new_timer(interval=0)
        timer.single_shot = True
        timer.add_callback(callback_func, callback, timer)
        self._timers.add(timer)
        timer.start()

    def _draw_idle(self):
        if False:
            i = 10
            return i + 15
        '\n        Draw method for singleshot timer\n\n        This draw method can be added to a singleshot timer, which can\n        accumulate draws while the eventloop is spinning. This method will\n        then only draw the first time and short-circuit the others.\n        '
        with self._idle_draw_cntx():
            if not self._draw_pending:
                return
            self._draw_pending = False
            self.draw()

    def blit(self, bbox=None):
        if False:
            for i in range(10):
                print('nop')
        super().blit(bbox)
        self.update()

    def resize(self, width, height):
        if False:
            return 10
        scale = self.figure.dpi / self.device_pixel_ratio
        width /= scale
        height /= scale
        self.figure.set_size_inches(width, height, forward=False)
        ResizeEvent('resize_event', self)._process()
        self.draw_idle()

    def start_event_loop(self, timeout=0):
        if False:
            print('Hello World!')
        with _allow_interrupt_macos():
            self._start_event_loop(timeout=timeout)

class NavigationToolbar2Mac(_macosx.NavigationToolbar2, NavigationToolbar2):

    def __init__(self, canvas):
        if False:
            i = 10
            return i + 15
        data_path = cbook._get_data_path('images')
        (_, tooltips, image_names, _) = zip(*NavigationToolbar2.toolitems)
        _macosx.NavigationToolbar2.__init__(self, canvas, tuple((str(data_path / image_name) + '.pdf' for image_name in image_names if image_name is not None)), tuple((tooltip for tooltip in tooltips if tooltip is not None)))
        NavigationToolbar2.__init__(self, canvas)

    def draw_rubberband(self, event, x0, y0, x1, y1):
        if False:
            return 10
        self.canvas.set_rubberband(int(x0), int(y0), int(x1), int(y1))

    def remove_rubberband(self):
        if False:
            for i in range(10):
                print('nop')
        self.canvas.remove_rubberband()

    def save_figure(self, *args):
        if False:
            i = 10
            return i + 15
        directory = os.path.expanduser(mpl.rcParams['savefig.directory'])
        filename = _macosx.choose_save_file('Save the figure', directory, self.canvas.get_default_filename())
        if filename is None:
            return
        if mpl.rcParams['savefig.directory']:
            mpl.rcParams['savefig.directory'] = os.path.dirname(filename)
        self.canvas.figure.savefig(filename)

class FigureManagerMac(_macosx.FigureManager, FigureManagerBase):
    _toolbar2_class = NavigationToolbar2Mac

    def __init__(self, canvas, num):
        if False:
            while True:
                i = 10
        self._shown = False
        _macosx.FigureManager.__init__(self, canvas)
        icon_path = str(cbook._get_data_path('images/matplotlib.pdf'))
        _macosx.FigureManager.set_icon(icon_path)
        FigureManagerBase.__init__(self, canvas, num)
        self._set_window_mode(mpl.rcParams['macosx.window_mode'])
        if self.toolbar is not None:
            self.toolbar.update()
        if mpl.is_interactive():
            self.show()
            self.canvas.draw_idle()

    def _close_button_pressed(self):
        if False:
            while True:
                i = 10
        Gcf.destroy(self)
        self.canvas.flush_events()

    def destroy(self):
        if False:
            i = 10
            return i + 15
        while self.canvas._timers:
            timer = self.canvas._timers.pop()
            timer.stop()
        super().destroy()

    @classmethod
    def start_main_loop(cls):
        if False:
            print('Hello World!')
        with _allow_interrupt_macos():
            _macosx.show()

    def show(self):
        if False:
            print('Hello World!')
        if not self._shown:
            self._show()
            self._shown = True
        if mpl.rcParams['figure.raise_window']:
            self._raise()

@_Backend.export
class _BackendMac(_Backend):
    FigureCanvas = FigureCanvasMac
    FigureManager = FigureManagerMac
    mainloop = FigureManagerMac.start_main_loop