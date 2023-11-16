"""Module visualizes PCollection data.

For internal use only; no backwards-compatibility guarantees.
Only works with Python 3.5+.
"""
import base64
import datetime
import html
import logging
from datetime import timedelta
from typing import Optional
from dateutil import tz
import apache_beam as beam
from apache_beam.runners.interactive import interactive_environment as ie
from apache_beam.runners.interactive.utils import elements_to_df
from apache_beam.transforms.window import GlobalWindow
from apache_beam.transforms.window import IntervalWindow
try:
    from IPython import get_ipython
    from IPython.display import HTML
    from IPython.display import Javascript
    from IPython.display import display
    from IPython.display import display_javascript
    from facets_overview.generic_feature_statistics_generator import GenericFeatureStatisticsGenerator
    from timeloop import Timeloop
    if get_ipython():
        _pcoll_visualization_ready = True
    else:
        _pcoll_visualization_ready = False
except ImportError:
    _pcoll_visualization_ready = False
_LOGGER = logging.getLogger(__name__)
_CSS = '\n            <style>\n            .p-Widget.jp-OutputPrompt.jp-OutputArea-prompt:empty {{\n              padding: 0;\n              border: 0;\n            }}\n            .p-Widget.jp-RenderedJavaScript.jp-mod-trusted.jp-OutputArea-output:empty {{\n              padding: 0;\n              border: 0;\n            }}\n            </style>'
_DIVE_SCRIPT_TEMPLATE = '\n            try {{\n              document\n                .getElementById("{display_id}")\n                .contentDocument\n                .getElementById("{display_id}")\n                .data = {jsonstr};\n            }} catch (e) {{\n              // NOOP when the user has cleared the output from the notebook.\n            }}'
_DIVE_HTML_TEMPLATE = _CSS + '\n            <iframe id={display_id} style="border:none" width="100%" height="600px"\n              srcdoc=\'\n                <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>\n                <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html">\n                <facets-dive sprite-image-width="{sprite_size}" sprite-image-height="{sprite_size}" id="{display_id}" height="600"></facets-dive>\n                <script>\n                  document.getElementById("{display_id}").data = {jsonstr};\n                </script>\n              \'>\n            </iframe>'
_OVERVIEW_SCRIPT_TEMPLATE = '\n              try {{\n                document\n                  .getElementById("{display_id}")\n                  .contentDocument\n                  .getElementById("{display_id}")\n                  .protoInput = "{protostr}";\n              }} catch (e) {{\n                // NOOP when the user has cleared the output from the notebook.\n              }}'
_OVERVIEW_HTML_TEMPLATE = _CSS + '\n            <iframe id={display_id} style="border:none" width="100%" height="600px"\n              srcdoc=\'\n                <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>\n                <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html">\n                <facets-overview id="{display_id}"></facets-overview>\n                <script>\n                  document.getElementById("{display_id}").protoInput = "{protostr}";\n                </script>\n              \'>\n            </iframe>'
_DATATABLE_INITIALIZATION_CONFIG = '\n            bAutoWidth: false,\n            columns: {columns},\n            destroy: true,\n            responsive: true,\n            columnDefs: [\n              {{\n                targets: "_all",\n                className: "dt-left"\n              }},\n              {{\n                "targets": 0,\n                "width": "10px",\n                "title": ""\n              }}\n            ]'
_DATAFRAME_SCRIPT_TEMPLATE = '\n            var dt;\n            if ($.fn.dataTable.isDataTable("#{table_id}")) {{\n              dt = $("#{table_id}").dataTable();\n            }} else if ($("#{table_id}_wrapper").length == 0) {{\n              dt = $("#{table_id}").dataTable({{\n                ' + _DATATABLE_INITIALIZATION_CONFIG + "\n              }});\n            }} else {{\n              return;\n            }}\n            dt.api()\n              .clear()\n              .rows.add({data_as_rows})\n              .draw('full-hold');"
_DATAFRAME_PAGINATION_TEMPLATE = _CSS + '\n            <link rel="stylesheet" href="https://cdn.datatables.net/1.10.20/css/jquery.dataTables.min.css">\n            <table id="{table_id}" class="display" style="display:block"></table>\n            <script>\n              {script_in_jquery_with_datatable}\n            </script>'
_NO_DATA_TEMPLATE = _CSS + '\n            <div id="no_data_{id}">No data to display.</div>'
_NO_DATA_REMOVAL_SCRIPT = '\n            $("#no_data_{id}").remove();'

def visualize(stream, dynamic_plotting_interval=None, include_window_info=False, display_facets=False, element_type=None):
    if False:
        for i in range(10):
            print('nop')
    "Visualizes the data of a given PCollection. Optionally enables dynamic\n  plotting with interval in seconds if the PCollection is being produced by a\n  running pipeline or the pipeline is streaming indefinitely. The function\n  always returns immediately and is asynchronous when dynamic plotting is on.\n\n  If dynamic plotting enabled, the visualization is updated continuously until\n  the pipeline producing the PCollection is in an end state. The visualization\n  would be anchored to the notebook cell output area. The function\n  asynchronously returns a handle to the visualization job immediately. The user\n  could manually do::\n\n    # In one notebook cell, enable dynamic plotting every 1 second:\n    handle = visualize(pcoll, dynamic_plotting_interval=1)\n    # Visualization anchored to the cell's output area.\n    # In a different cell:\n    handle.stop()\n    # Will stop the dynamic plotting of the above visualization manually.\n    # Otherwise, dynamic plotting ends when pipeline is not running anymore.\n\n  If dynamic_plotting is not enabled (by default), None is returned.\n\n  If include_window_info is True, the data will include window information,\n  which consists of the event timestamps, windows, and pane info.\n\n  If display_facets is True, the facets widgets will be rendered. Otherwise, the\n  facets widgets will not be rendered.\n\n  The function is experimental. For internal use only; no\n  backwards-compatibility guarantees.\n  "
    if not _pcoll_visualization_ready:
        return None
    pv = PCollectionVisualization(stream, include_window_info=include_window_info, display_facets=display_facets, element_type=element_type)
    if ie.current_env().is_in_notebook:
        pv.display()
    else:
        pv.display_plain_text()
        return None
    if dynamic_plotting_interval:
        logging.getLogger('timeloop').disabled = True
        tl = Timeloop()

        def dynamic_plotting(stream, pv, tl, include_window_info, display_facets):
            if False:
                for i in range(10):
                    print('nop')

            @tl.job(interval=timedelta(seconds=dynamic_plotting_interval))
            def continuous_update_display():
                if False:
                    print('Hello World!')
                updated_pv = PCollectionVisualization(stream, include_window_info=include_window_info, display_facets=display_facets, element_type=element_type)
                updated_pv.display(updating_pv=pv)
                if stream.is_done():
                    try:
                        tl.stop()
                    except RuntimeError:
                        pass
            tl.start()
            return tl
        return dynamic_plotting(stream, pv, tl, include_window_info, display_facets)
    return None

def visualize_computed_pcoll(pcoll_name: str, pcoll: beam.pvalue.PCollection, max_n: int, max_duration_secs: float, dynamic_plotting_interval: Optional[int]=None, include_window_info: bool=False, display_facets: bool=False) -> None:
    if False:
        print('Hello World!')
    'A simple visualize alternative.\n\n  When the pcoll_name and pcoll pair identifies a watched and computed\n  PCollection in the current interactive environment without ambiguity, an\n  ElementStream can be built directly from cache. Returns immediately, the\n  visualization is asynchronous, but guaranteed to end in the near future.\n\n  Args:\n    pcoll_name: the variable name of the PCollection.\n    pcoll: the PCollection to be visualized.\n    max_n: the maximum number of elements to visualize.\n    max_duration_secs: max duration of elements to read in seconds.\n    dynamic_plotting_interval: the interval in seconds between visualization\n      updates if provided; otherwise, no dynamic plotting.\n    include_window_info: whether to include windowing info in the elements.\n    display_facets: whether to display the facets widgets.\n  '
    pipeline = ie.current_env().user_pipeline(pcoll.pipeline)
    rm = ie.current_env().get_recording_manager(pipeline, create_if_absent=True)
    stream = rm.read(pcoll_name, pcoll, max_n=max_n, max_duration_secs=max_duration_secs)
    if stream:
        visualize(stream, dynamic_plotting_interval=dynamic_plotting_interval, include_window_info=include_window_info, display_facets=display_facets, element_type=pcoll.element_type)

class PCollectionVisualization(object):
    """A visualization of a PCollection.

  The class relies on creating a PipelineInstrument w/o actual instrument to
  access current interactive environment for materialized PCollection data at
  the moment of self instantiation through cache.
  """

    def __init__(self, stream, include_window_info=False, display_facets=False, element_type=None):
        if False:
            return 10
        assert _pcoll_visualization_ready, 'Dependencies for PCollection visualization are not available. Please use `pip install apache-beam[interactive]` to install necessary dependencies and make sure that you are executing code in an interactive environment such as a Jupyter notebook.'
        self._stream = stream
        self._pcoll_var = stream.var
        if not self._pcoll_var:
            self._pcoll_var = 'Value'
        obfuscated_id = stream.display_id(id(self))
        self._dive_display_id = 'facets_dive_{}'.format(obfuscated_id)
        self._overview_display_id = 'facets_overview_{}'.format(obfuscated_id)
        self._df_display_id = 'df_{}'.format(obfuscated_id)
        self._include_window_info = include_window_info
        self._display_facets = display_facets
        self._is_datatable_empty = True
        self._element_type = element_type

    def display_plain_text(self):
        if False:
            while True:
                i = 10
        "Displays a head sample of the normalized PCollection data.\n\n    This function is used when the ipython kernel is not connected to a\n    notebook frontend such as when running ipython in terminal or in unit tests.\n    It's a visualization in terminal-like UI, not a function to retrieve data\n    for programmatically usages.\n    "
        if _pcoll_visualization_ready:
            data = self._to_dataframe()
            data_sample = data.head(25)
            display(data_sample)

    def display(self, updating_pv=None):
        if False:
            while True:
                i = 10
        'Displays the visualization through IPython.\n\n    Args:\n      updating_pv: A PCollectionVisualization object. When provided, the\n        display_id of each visualization part will inherit from the initial\n        display of updating_pv and only update that visualization web element\n        instead of creating new ones.\n\n    The visualization has 3 parts: facets-dive, facets-overview and paginated\n    data table. Each part is assigned an auto-generated unique display id\n    (the uniqueness is guaranteed throughout the lifespan of the PCollection\n    variable).\n    '
        data = self._to_dataframe()
        data.columns = [self._pcoll_var + '.' + str(column) if isinstance(column, int) else column for column in data.columns]
        data = data.applymap(lambda x: str(x) if isinstance(x, dict) else x)
        if updating_pv:
            if data.empty:
                _LOGGER.debug('Skip a visualization update due to empty data.')
            else:
                self._display_dataframe(data.copy(deep=True), updating_pv)
                if self._display_facets:
                    self._display_dive(data.copy(deep=True), updating_pv)
                    self._display_overview(data.copy(deep=True), updating_pv)
        else:
            self._display_dataframe(data.copy(deep=True))
            if self._display_facets:
                self._display_dive(data.copy(deep=True))
                self._display_overview(data.copy(deep=True))

    def _display_dive(self, data, update=None):
        if False:
            while True:
                i = 10
        sprite_size = 32 if len(data.index) > 50000 else 64
        format_window_info_in_dataframe(data)
        jsonstr = data.to_json(orient='records', default_handler=str)
        if update:
            script = _DIVE_SCRIPT_TEMPLATE.format(display_id=update._dive_display_id, jsonstr=jsonstr)
            display_javascript(Javascript(script))
        else:
            html_str = _DIVE_HTML_TEMPLATE.format(display_id=self._dive_display_id, jsonstr=html.escape(jsonstr), sprite_size=sprite_size)
            display(HTML(html_str))

    def _display_overview(self, data, update=None):
        if False:
            print('Hello World!')
        if not data.empty and self._include_window_info and all((column in data.columns for column in ('event_time', 'windows', 'pane_info'))):
            data = data.drop(['event_time', 'windows', 'pane_info'], axis=1)
        data.columns = data.columns.astype(str)
        gfsg = GenericFeatureStatisticsGenerator()
        proto = gfsg.ProtoFromDataFrames([{'name': 'data', 'table': data}])
        protostr = base64.b64encode(proto.SerializeToString()).decode('utf-8')
        if update:
            script = _OVERVIEW_SCRIPT_TEMPLATE.format(display_id=update._overview_display_id, protostr=protostr)
            display_javascript(Javascript(script))
        else:
            html_str = _OVERVIEW_HTML_TEMPLATE.format(display_id=self._overview_display_id, protostr=protostr)
            display(HTML(html_str))

    def _display_dataframe(self, data, update=None):
        if False:
            i = 10
            return i + 15
        table_id = 'table_{}'.format(update._df_display_id if update else self._df_display_id)
        columns = [{'title': ''}] + [{'title': str(column)} for column in data.columns]
        format_window_info_in_dataframe(data)
        rows = data.applymap(lambda x: str(x)).to_dict('split')['data']
        rows = [{k + 1: v for (k, v) in enumerate(row)} for row in rows]
        for (k, row) in enumerate(rows):
            row[0] = k
        script = _DATAFRAME_SCRIPT_TEMPLATE.format(table_id=table_id, columns=columns, data_as_rows=rows)
        script_in_jquery_with_datatable = ie._JQUERY_WITH_DATATABLE_TEMPLATE.format(customized_script=script)
        if update and (not update._is_datatable_empty):
            display_javascript(Javascript(script_in_jquery_with_datatable))
        else:
            if data.empty:
                html_str = _NO_DATA_TEMPLATE.format(id=table_id)
            else:
                html_str = _DATAFRAME_PAGINATION_TEMPLATE.format(table_id=table_id, script_in_jquery_with_datatable=script_in_jquery_with_datatable)
            if update:
                if not data.empty:
                    display(Javascript(ie._JQUERY_WITH_DATATABLE_TEMPLATE.format(customized_script=_NO_DATA_REMOVAL_SCRIPT.format(id=table_id))))
                    display(HTML(html_str), display_id=update._df_display_id)
                    update._is_datatable_empty = False
            else:
                display(HTML(html_str), display_id=self._df_display_id)
                if not data.empty:
                    self._is_datatable_empty = False

    def _to_dataframe(self):
        if False:
            print('Hello World!')
        results = list(self._stream.read(tail=False))
        return elements_to_df(results, self._include_window_info, element_type=self._element_type)

def format_window_info_in_dataframe(data):
    if False:
        return 10
    if 'event_time' in data.columns:
        data['event_time'] = data['event_time'].apply(event_time_formatter)
    if 'windows' in data.columns:
        data['windows'] = data['windows'].apply(windows_formatter)
    if 'pane_info' in data.columns:
        data['pane_info'] = data['pane_info'].apply(pane_info_formatter)

def event_time_formatter(event_time_us):
    if False:
        return 10
    options = ie.current_env().options
    to_tz = options.display_timezone
    try:
        return datetime.datetime.utcfromtimestamp(event_time_us / 1000000).replace(tzinfo=tz.tzutc()).astimezone(to_tz).strftime(options.display_timestamp_format)
    except ValueError:
        if event_time_us < 0:
            return 'Min Timestamp'
        return 'Max Timestamp'

def windows_formatter(windows):
    if False:
        i = 10
        return i + 15
    result = []
    for w in windows:
        if isinstance(w, GlobalWindow):
            result.append(str(w))
        elif isinstance(w, IntervalWindow):
            duration = w.end.micros - w.start.micros
            duration_secs = duration // 1000000
            (hours, remainder) = divmod(duration_secs, 3600)
            (minutes, seconds) = divmod(remainder, 60)
            micros = (duration - duration_secs * 1000000) % 1000000
            duration = ''
            if hours:
                duration += '{}h '.format(hours)
            if minutes or (hours and seconds):
                duration += '{}m '.format(minutes)
            if seconds:
                if micros:
                    duration += '{}.{:06}s'.format(seconds, micros)
                else:
                    duration += '{}s'.format(seconds)
            start = event_time_formatter(w.start.micros)
            result.append('{} ({})'.format(start, duration))
    return ','.join(result)

def pane_info_formatter(pane_info):
    if False:
        return 10
    from apache_beam.utils.windowed_value import PaneInfo
    from apache_beam.utils.windowed_value import PaneInfoTiming
    assert isinstance(pane_info, PaneInfo)
    result = 'Pane {}'.format(pane_info.index)
    timing_info = '{}{}'.format('Final ' if pane_info.is_last else '', PaneInfoTiming.to_string(pane_info.timing).lower().capitalize() if pane_info.timing in (PaneInfoTiming.EARLY, PaneInfoTiming.LATE) else '')
    if timing_info:
        result += ': ' + timing_info
    return result