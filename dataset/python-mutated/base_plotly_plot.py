from ..vis_base import VisBase
import time
from abc import abstractmethod
from .. import utils

class BasePlotlyPlot(VisBase):

    def __init__(self, cell: VisBase.widgets.Box=None, title=None, show_legend: bool=None, is_3d: bool=False, stream_name: str=None, console_debug: bool=False, **vis_args):
        if False:
            return 10
        import plotly.graph_objs as go
        super(BasePlotlyPlot, self).__init__(go.FigureWidget(), cell, title, show_legend, stream_name=stream_name, console_debug=console_debug, **vis_args)
        self.is_3d = is_3d
        self.widget.layout.title = title
        self.widget.layout.showlegend = show_legend if show_legend is not None else True

    def _add_trace(self, stream_vis):
        if False:
            for i in range(10):
                print('nop')
        stream_vis.trace_index = len(self.widget.data)
        trace = self._create_trace(stream_vis)
        if stream_vis.opacity is not None:
            trace.opacity = stream_vis.opacity
        self.widget.add_trace(trace)

    def _add_trace_with_history(self, stream_vis):
        if False:
            return 10
        if stream_vis.history_len > len(stream_vis.trace_history):
            self._add_trace(stream_vis)
            stream_vis.trace_history.append(len(self.widget.data) - 1)
            stream_vis.cur_history_index = len(stream_vis.trace_history) - 1
        else:
            stream_vis.cur_history_index = (stream_vis.cur_history_index + 1) % stream_vis.history_len
            stream_vis.trace_index = stream_vis.trace_history[stream_vis.cur_history_index]
            self.clear_plot(stream_vis, False)
            self.widget.data[stream_vis.trace_index].opacity = stream_vis.opacity or 1
        cur_history_len = len(stream_vis.trace_history)
        if stream_vis.dim_history and cur_history_len > 1:
            max_opacity = stream_vis.opacity or 1
            (min_alpha, max_alpha, dimmed_len) = (max_opacity * 0.05, max_opacity * 0.8, cur_history_len - 1)
            alphas = list(utils.frange(max_alpha, min_alpha, steps=dimmed_len))
            for (i, thi) in enumerate(range(stream_vis.cur_history_index + 1, stream_vis.cur_history_index + cur_history_len)):
                trace_index = stream_vis.trace_history[thi % cur_history_len]
                self.widget.data[trace_index].opacity = alphas[i]

    @staticmethod
    def get_pallet_color(i: int):
        if False:
            for i in range(10):
                print('nop')
        import plotly
        return plotly.colors.DEFAULT_PLOTLY_COLORS[i % len(plotly.colors.DEFAULT_PLOTLY_COLORS)]

    @staticmethod
    def _get_axis_common_props(title: str, axis_range: tuple):
        if False:
            return 10
        props = {'showline': True, 'showgrid': True, 'showticklabels': True, 'ticks': 'inside'}
        if title:
            props['title'] = title
        if axis_range:
            props['range'] = list(axis_range)
        return props

    def _can_update_stream_plots(self):
        if False:
            i = 10
            return i + 15
        return time.time() - self.q_last_processed > 0.5

    def _post_add_subscription(self, stream_vis, **stream_vis_args):
        if False:
            i = 10
            return i + 15
        (stream_vis.trace_history, stream_vis.cur_history_index) = ([], None)
        self._add_trace_with_history(stream_vis)
        self._setup_layout(stream_vis)
        if not self.widget.layout.title:
            self.widget.layout.title = stream_vis.title
        if stream_vis.history_len > 1:
            self.widget.layout.showlegend = False

    def _show_widget_native(self, blocking: bool):
        if False:
            return 10
        pass

    def _show_widget_notebook(self):
        if False:
            for i in range(10):
                print('nop')
        return None

    def _post_update_stream_plot(self, stream_vis):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def clear_plot(self, stream_vis, clear_history):
        if False:
            return 10
        '(for derived class) Clears the data in specified plot before new data is redrawn'
        pass

    @abstractmethod
    def _show_stream_items(self, stream_vis, stream_items):
        if False:
            i = 10
            return i + 15
        'Paint the given stream_items in to visualizer. If visualizer is dirty then return False else True.\n        '
        pass

    @abstractmethod
    def _setup_layout(self, stream_vis):
        if False:
            return 10
        pass

    @abstractmethod
    def _create_trace(self, stream_vis):
        if False:
            return 10
        pass