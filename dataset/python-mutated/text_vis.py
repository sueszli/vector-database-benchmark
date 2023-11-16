from . import utils
from .vis_base import VisBase

class TextVis(VisBase):

    def __init__(self, cell: VisBase.widgets.Box=None, title: str=None, show_legend: bool=None, stream_name: str=None, console_debug: bool=False, **vis_args):
        if False:
            print('Hello World!')
        import pandas as pd
        super(TextVis, self).__init__(VisBase.widgets.HTML(), cell, title, show_legend, stream_name=stream_name, console_debug=console_debug, **vis_args)
        self.df = pd.DataFrame([])
        self.SeriesClass = pd.Series

    def _get_column_prefix(self, stream_vis, i):
        if False:
            while True:
                i = 10
        return '[S.{}]:{}'.format(stream_vis.index, i)

    def _get_title(self, stream_vis):
        if False:
            for i in range(10):
                print('nop')
        title = stream_vis.title or 'Stream ' + str(len(self._stream_vises))
        return title

    def _append(self, stream_vis, vals):
        if False:
            print('Hello World!')
        if vals is None:
            self.df = self.df.append(self.SeriesClass({self._get_column_prefix(stream_vis, 0): None}), sort=False, ignore_index=True)
            return
        for val in vals:
            if val is None or utils.is_scalar(val):
                self.df = self.df.append(self.SeriesClass({self._get_column_prefix(stream_vis, 0): val}), sort=False, ignore_index=True)
            elif utils.is_array_like(val):
                val_dict = {}
                for (i, val_i) in enumerate(val):
                    val_dict[self._get_column_prefix(stream_vis, i)] = val_i
                self.df = self.df.append(self.SeriesClass(val_dict), sort=False, ignore_index=True)
            else:
                self.df = self.df.append(self.SeriesClass(val.__dict__), sort=False, ignore_index=True)

    def _post_add_subscription(self, stream_vis, **stream_vis_args):
        if False:
            return 10
        only_summary = stream_vis_args.get('only_summary', False)
        stream_vis.text = self._get_title(stream_vis)
        stream_vis.only_summary = only_summary

    def clear_plot(self, stream_vis, clear_history):
        if False:
            for i in range(10):
                print('nop')
        self.df = self.df.iloc[0:0]

    def _show_stream_items(self, stream_vis, stream_items):
        if False:
            i = 10
            return i + 15
        'Paint the given stream_items in to visualizer. If visualizer is dirty then return False else True.\n        '
        for stream_item in stream_items:
            if stream_item.ended:
                self.df = self.df.append(self.SeriesClass({'Ended': True}), sort=False, ignore_index=True)
            else:
                vals = self._extract_vals((stream_item,))
                self._append(stream_vis, vals)
        return False

    def _post_update_stream_plot(self, stream_vis):
        if False:
            print('Hello World!')
        if VisBase.get_ipython():
            if not stream_vis.only_summary:
                self.widget.value = self.df.to_html(classes=['output_html', 'rendered_html'])
            else:
                self.widget.value = self.df.describe().to_html(classes=['output_html', 'rendered_html'])
        else:
            last_recs = self.df.iloc[[-1]].to_dict('records')
            if len(last_recs) == 1:
                print(last_recs[0])
            else:
                print(last_recs)

    def _show_widget_native(self, blocking: bool):
        if False:
            print('Hello World!')
        return None

    def _show_widget_notebook(self):
        if False:
            return 10
        return self.widget