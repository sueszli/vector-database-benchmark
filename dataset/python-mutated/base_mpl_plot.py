from ..vis_base import VisBase
import sys, logging
from abc import abstractmethod
from .. import utils

class BaseMplPlot(VisBase):

    def __init__(self, cell: VisBase.widgets.Box=None, title: str=None, show_legend: bool=None, is_3d: bool=False, stream_name: str=None, console_debug: bool=False, **vis_args):
        if False:
            for i in range(10):
                print('nop')
        super(BaseMplPlot, self).__init__(VisBase.widgets.Output(), cell, title, show_legend, stream_name=stream_name, console_debug=console_debug, **vis_args)
        self._fig_init_done = False
        self.show_legend = show_legend
        self.is_3d = is_3d
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
        self.figure = None
        self._ax_main = None
        self.animation = None
        self.anim_interval = None

    def init_fig(self, anim_interval: float=1.0):
        if False:
            while True:
                i = 10
        import matplotlib.pyplot as plt
        '(for derived class) Initializes matplotlib figure'
        if self._fig_init_done:
            return False
        self.figure = plt.figure(figsize=(8, 3))
        self.anim_interval = anim_interval
        import matplotlib.pyplot as plt
        plt.set_cmap('Dark2')
        plt.rcParams['image.cmap'] = 'Dark2'
        self._fig_init_done = True
        return True

    def get_main_axis(self):
        if False:
            while True:
                i = 10
        if not self._ax_main:
            self._ax_main = self.figure.add_subplot(111, projection=None if not self.is_3d else '3d')
            self._ax_main.grid(self.is_show_grid())
            self._ax_main.spines['right'].set_color((0.8, 0.8, 0.8))
            self._ax_main.spines['top'].set_color((0.8, 0.8, 0.8))
            if self.title is not None:
                title = self._ax_main.set_title(self.title)
                title.set_weight('bold')
        return self._ax_main

    def is_show_grid(self):
        if False:
            print('Hello World!')
        return True

    def _on_update(self, frame):
        if False:
            print('Hello World!')
        try:
            self._update_stream_plots()
        except Exception as ex:
            self.last_ex = ex
            logging.exception('Exception in matplotlib update loop')

    def show(self, blocking=False):
        if False:
            i = 10
            return i + 15
        if not self.is_shown and self.anim_interval:
            from matplotlib.animation import FuncAnimation
            self.animation = FuncAnimation(self.figure, self._on_update, interval=self.anim_interval * 1000.0)
        super(BaseMplPlot, self).show(blocking)

    def _post_update_stream_plot(self, stream_vis):
        if False:
            i = 10
            return i + 15
        import matplotlib.pyplot as plt
        utils.debug_log('Plot updated', stream_vis.stream.stream_name, verbosity=5)
        if self.layout_dirty:
            self.figure.tight_layout()
            self.layout_dirty = False
        if self._use_hbox and VisBase.get_ipython():
            self.widget.clear_output(wait=True)
            with self.widget:
                plt.show(self.figure)

    def _post_add_subscription(self, stream_vis, **stream_vis_args):
        if False:
            for i in range(10):
                print('nop')
        import matplotlib.pyplot as plt
        self.init_fig()
        self.init_stream_plot(stream_vis, **stream_vis_args)
        if self.show_legend:
            self.figure.legend(loc='lower right')
        plt.subplots_adjust(hspace=0.6)

    def _show_widget_native(self, blocking: bool):
        if False:
            return 10
        import matplotlib.pyplot as plt
        return plt.show(block=blocking)

    def _show_widget_notebook(self):
        if False:
            return 10
        return None

    def _can_update_stream_plots(self):
        if False:
            print('Hello World!')
        return False

    @abstractmethod
    def init_stream_plot(self, stream_vis, **stream_vis_args):
        if False:
            while True:
                i = 10
        '(for derived class) Create new plot info for this stream'
        pass

    def _save_widget(self, filepath: str) -> None:
        if False:
            return 10
        self._update_stream_plots()
        self.figure.savefig(filepath)