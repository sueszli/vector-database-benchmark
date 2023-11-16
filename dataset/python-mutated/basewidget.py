import ipywidgets as widgets
from traitlets import List, Unicode, Dict, observe, Integer
from .basedatatypes import BaseFigure, BasePlotlyType
from .callbacks import BoxSelector, LassoSelector, InputDeviceState, Points
from .serializers import custom_serializers
from .version import __frontend_version__

@widgets.register
class BaseFigureWidget(BaseFigure, widgets.DOMWidget):
    """
    Base class for FigureWidget. The FigureWidget class is code-generated as a
    subclass
    """
    _view_name = Unicode('FigureView').tag(sync=True)
    _view_module = Unicode('jupyterlab-plotly').tag(sync=True)
    _view_module_version = Unicode(__frontend_version__).tag(sync=True)
    _model_name = Unicode('FigureModel').tag(sync=True)
    _model_module = Unicode('jupyterlab-plotly').tag(sync=True)
    _model_module_version = Unicode(__frontend_version__).tag(sync=True)
    _layout = Dict().tag(sync=True, **custom_serializers)
    _data = List().tag(sync=True, **custom_serializers)
    _config = Dict().tag(sync=True, **custom_serializers)
    _py2js_addTraces = Dict(allow_none=True).tag(sync=True, **custom_serializers)
    _py2js_restyle = Dict(allow_none=True).tag(sync=True, **custom_serializers)
    _py2js_relayout = Dict(allow_none=True).tag(sync=True, **custom_serializers)
    _py2js_update = Dict(allow_none=True).tag(sync=True, **custom_serializers)
    _py2js_animate = Dict(allow_none=True).tag(sync=True, **custom_serializers)
    _py2js_deleteTraces = Dict(allow_none=True).tag(sync=True, **custom_serializers)
    _py2js_moveTraces = Dict(allow_none=True).tag(sync=True, **custom_serializers)
    _py2js_removeLayoutProps = Dict(allow_none=True).tag(sync=True, **custom_serializers)
    _py2js_removeTraceProps = Dict(allow_none=True).tag(sync=True, **custom_serializers)
    _js2py_traceDeltas = Dict(allow_none=True).tag(sync=True, **custom_serializers)
    _js2py_layoutDelta = Dict(allow_none=True).tag(sync=True, **custom_serializers)
    _js2py_restyle = Dict(allow_none=True).tag(sync=True, **custom_serializers)
    _js2py_relayout = Dict(allow_none=True).tag(sync=True, **custom_serializers)
    _js2py_update = Dict(allow_none=True).tag(sync=True, **custom_serializers)
    _js2py_pointsCallback = Dict(allow_none=True).tag(sync=True, **custom_serializers)
    _last_layout_edit_id = Integer(0).tag(sync=True)
    _last_trace_edit_id = Integer(0).tag(sync=True)
    _set_trace_uid = True
    _allow_disable_validation = False

    def __init__(self, data=None, layout=None, frames=None, skip_invalid=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(BaseFigureWidget, self).__init__(data=data, layout_plotly=layout, frames=frames, skip_invalid=skip_invalid, **kwargs)
        if self._frame_objs:
            BaseFigureWidget._display_frames_error()
        self._last_layout_edit_id = 0
        self._layout_edit_in_process = False
        self._waiting_edit_callbacks = []
        self._last_trace_edit_id = 0
        self._trace_edit_in_process = False
        self._view_count = 0

    def _send_relayout_msg(self, layout_data, source_view_id=None):
        if False:
            i = 10
            return i + 15
        "\n        Send Plotly.relayout message to the frontend\n\n        Parameters\n        ----------\n        layout_data : dict\n            Plotly.relayout layout data\n        source_view_id : str\n            UID of view that triggered this relayout operation\n            (e.g. By the user clicking 'zoom' in the toolbar). None if the\n            operation was not triggered by a frontend view\n        "
        layout_edit_id = self._last_layout_edit_id + 1
        self._last_layout_edit_id = layout_edit_id
        self._layout_edit_in_process = True
        msg_data = {'relayout_data': layout_data, 'layout_edit_id': layout_edit_id, 'source_view_id': source_view_id}
        self._py2js_relayout = msg_data
        self._py2js_relayout = None

    def _send_restyle_msg(self, restyle_data, trace_indexes=None, source_view_id=None):
        if False:
            while True:
                i = 10
        '\n        Send Plotly.restyle message to the frontend\n\n        Parameters\n        ----------\n        restyle_data : dict\n            Plotly.restyle restyle data\n        trace_indexes : list[int]\n            List of trace indexes that the restyle operation\n            applies to\n        source_view_id : str\n            UID of view that triggered this restyle operation\n            (e.g. By the user clicking the legend to hide a trace).\n            None if the operation was not triggered by a frontend view\n        '
        trace_indexes = self._normalize_trace_indexes(trace_indexes)
        layout_edit_id = self._last_layout_edit_id + 1
        self._last_layout_edit_id = layout_edit_id
        self._layout_edit_in_process = True
        trace_edit_id = self._last_trace_edit_id + 1
        self._last_trace_edit_id = trace_edit_id
        self._trace_edit_in_process = True
        restyle_msg = {'restyle_data': restyle_data, 'restyle_traces': trace_indexes, 'trace_edit_id': trace_edit_id, 'layout_edit_id': layout_edit_id, 'source_view_id': source_view_id}
        self._py2js_restyle = restyle_msg
        self._py2js_restyle = None

    def _send_addTraces_msg(self, new_traces_data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Send Plotly.addTraces message to the frontend\n\n        Parameters\n        ----------\n        new_traces_data : list[dict]\n            List of trace data for new traces as accepted by Plotly.addTraces\n        '
        layout_edit_id = self._last_layout_edit_id + 1
        self._last_layout_edit_id = layout_edit_id
        self._layout_edit_in_process = True
        trace_edit_id = self._last_trace_edit_id + 1
        self._last_trace_edit_id = trace_edit_id
        self._trace_edit_in_process = True
        add_traces_msg = {'trace_data': new_traces_data, 'trace_edit_id': trace_edit_id, 'layout_edit_id': layout_edit_id}
        self._py2js_addTraces = add_traces_msg
        self._py2js_addTraces = None

    def _send_moveTraces_msg(self, current_inds, new_inds):
        if False:
            i = 10
            return i + 15
        '\n        Send Plotly.moveTraces message to the frontend\n\n        Parameters\n        ----------\n        current_inds : list[int]\n            List of current trace indexes\n        new_inds : list[int]\n            List of new trace indexes\n        '
        move_msg = {'current_trace_inds': current_inds, 'new_trace_inds': new_inds}
        self._py2js_moveTraces = move_msg
        self._py2js_moveTraces = None

    def _send_update_msg(self, restyle_data, relayout_data, trace_indexes=None, source_view_id=None):
        if False:
            print('Hello World!')
        '\n        Send Plotly.update message to the frontend\n\n        Parameters\n        ----------\n        restyle_data : dict\n            Plotly.update restyle data\n        relayout_data : dict\n            Plotly.update relayout data\n        trace_indexes : list[int]\n            List of trace indexes that the update operation applies to\n        source_view_id : str\n            UID of view that triggered this update operation\n            (e.g. By the user clicking a button).\n            None if the operation was not triggered by a frontend view\n        '
        trace_indexes = self._normalize_trace_indexes(trace_indexes)
        trace_edit_id = self._last_trace_edit_id + 1
        self._last_trace_edit_id = trace_edit_id
        self._trace_edit_in_process = True
        layout_edit_id = self._last_layout_edit_id + 1
        self._last_layout_edit_id = layout_edit_id
        self._layout_edit_in_process = True
        update_msg = {'style_data': restyle_data, 'layout_data': relayout_data, 'style_traces': trace_indexes, 'trace_edit_id': trace_edit_id, 'layout_edit_id': layout_edit_id, 'source_view_id': source_view_id}
        self._py2js_update = update_msg
        self._py2js_update = None

    def _send_animate_msg(self, styles_data, relayout_data, trace_indexes, animation_opts):
        if False:
            for i in range(10):
                print('nop')
        '\n        Send Plotly.update message to the frontend\n\n        Note: there is no source_view_id parameter because animations\n        triggered by the fontend are not currently supported\n\n        Parameters\n        ----------\n        styles_data : list[dict]\n            Plotly.animate styles data\n        relayout_data : dict\n            Plotly.animate relayout data\n        trace_indexes : list[int]\n            List of trace indexes that the animate operation applies to\n        '
        trace_indexes = self._normalize_trace_indexes(trace_indexes)
        trace_edit_id = self._last_trace_edit_id + 1
        self._last_trace_edit_id = trace_edit_id
        self._trace_edit_in_process = True
        layout_edit_id = self._last_layout_edit_id + 1
        self._last_layout_edit_id = layout_edit_id
        self._layout_edit_in_process = True
        animate_msg = {'style_data': styles_data, 'layout_data': relayout_data, 'style_traces': trace_indexes, 'animation_opts': animation_opts, 'trace_edit_id': trace_edit_id, 'layout_edit_id': layout_edit_id, 'source_view_id': None}
        self._py2js_animate = animate_msg
        self._py2js_animate = None

    def _send_deleteTraces_msg(self, delete_inds):
        if False:
            i = 10
            return i + 15
        '\n        Send Plotly.deleteTraces message to the frontend\n\n        Parameters\n        ----------\n        delete_inds : list[int]\n            List of trace indexes of traces to delete\n        '
        trace_edit_id = self._last_trace_edit_id + 1
        self._last_trace_edit_id = trace_edit_id
        self._trace_edit_in_process = True
        layout_edit_id = self._last_layout_edit_id + 1
        self._last_layout_edit_id = layout_edit_id
        self._layout_edit_in_process = True
        delete_msg = {'delete_inds': delete_inds, 'layout_edit_id': layout_edit_id, 'trace_edit_id': trace_edit_id}
        self._py2js_deleteTraces = delete_msg
        self._py2js_deleteTraces = None

    @observe('_js2py_traceDeltas')
    def _handler_js2py_traceDeltas(self, change):
        if False:
            return 10
        '\n        Process trace deltas message from the frontend\n        '
        msg_data = change['new']
        if not msg_data:
            self._js2py_traceDeltas = None
            return
        trace_deltas = msg_data['trace_deltas']
        trace_edit_id = msg_data['trace_edit_id']
        if trace_edit_id == self._last_trace_edit_id:
            for delta in trace_deltas:
                trace_uid = delta['uid']
                trace_uids = [trace.uid for trace in self.data]
                trace_index = trace_uids.index(trace_uid)
                uid_trace = self.data[trace_index]
                delta_transform = BaseFigureWidget._transform_data(uid_trace._prop_defaults, delta)
                remove_props = self._remove_overlapping_props(uid_trace._props, uid_trace._prop_defaults)
                if remove_props:
                    remove_trace_props_msg = {'remove_trace': trace_index, 'remove_props': remove_props}
                    self._py2js_removeTraceProps = remove_trace_props_msg
                    self._py2js_removeTraceProps = None
                self._dispatch_trace_change_callbacks(delta_transform, [trace_index])
            self._trace_edit_in_process = False
            if not self._layout_edit_in_process:
                while self._waiting_edit_callbacks:
                    self._waiting_edit_callbacks.pop()()
        self._js2py_traceDeltas = None

    @observe('_js2py_layoutDelta')
    def _handler_js2py_layoutDelta(self, change):
        if False:
            for i in range(10):
                print('nop')
        '\n        Process layout delta message from the frontend\n        '
        msg_data = change['new']
        if not msg_data:
            self._js2py_layoutDelta = None
            return
        layout_delta = msg_data['layout_delta']
        layout_edit_id = msg_data['layout_edit_id']
        if layout_edit_id == self._last_layout_edit_id:
            delta_transform = BaseFigureWidget._transform_data(self._layout_defaults, layout_delta)
            removed_props = self._remove_overlapping_props(self._layout, self._layout_defaults)
            if removed_props:
                remove_props_msg = {'remove_props': removed_props}
                self._py2js_removeLayoutProps = remove_props_msg
                self._py2js_removeLayoutProps = None
            for proppath in delta_transform:
                prop = proppath[0]
                match = self.layout._subplot_re_match(prop)
                if match and prop not in self.layout:
                    self.layout[prop] = {}
            self._dispatch_layout_change_callbacks(delta_transform)
            self._layout_edit_in_process = False
            if not self._trace_edit_in_process:
                while self._waiting_edit_callbacks:
                    self._waiting_edit_callbacks.pop()()
        self._js2py_layoutDelta = None

    @observe('_js2py_restyle')
    def _handler_js2py_restyle(self, change):
        if False:
            print('Hello World!')
        '\n        Process Plotly.restyle message from the frontend\n        '
        restyle_msg = change['new']
        if not restyle_msg:
            self._js2py_restyle = None
            return
        style_data = restyle_msg['style_data']
        style_traces = restyle_msg['style_traces']
        source_view_id = restyle_msg['source_view_id']
        self.plotly_restyle(restyle_data=style_data, trace_indexes=style_traces, source_view_id=source_view_id)
        self._js2py_restyle = None

    @observe('_js2py_update')
    def _handler_js2py_update(self, change):
        if False:
            print('Hello World!')
        '\n        Process Plotly.update message from the frontend\n        '
        update_msg = change['new']
        if not update_msg:
            self._js2py_update = None
            return
        style = update_msg['style_data']
        trace_indexes = update_msg['style_traces']
        layout = update_msg['layout_data']
        source_view_id = update_msg['source_view_id']
        self.plotly_update(restyle_data=style, relayout_data=layout, trace_indexes=trace_indexes, source_view_id=source_view_id)
        self._js2py_update = None

    @observe('_js2py_relayout')
    def _handler_js2py_relayout(self, change):
        if False:
            return 10
        '\n        Process Plotly.relayout message from the frontend\n        '
        relayout_msg = change['new']
        if not relayout_msg:
            self._js2py_relayout = None
            return
        relayout_data = relayout_msg['relayout_data']
        source_view_id = relayout_msg['source_view_id']
        if 'lastInputTime' in relayout_data:
            relayout_data.pop('lastInputTime')
        self.plotly_relayout(relayout_data=relayout_data, source_view_id=source_view_id)
        self._js2py_relayout = None

    @observe('_js2py_pointsCallback')
    def _handler_js2py_pointsCallback(self, change):
        if False:
            i = 10
            return i + 15
        '\n        Process points callback message from the frontend\n        '
        callback_data = change['new']
        if not callback_data:
            self._js2py_pointsCallback = None
            return
        event_type = callback_data['event_type']
        if callback_data.get('selector', None):
            selector_data = callback_data['selector']
            selector_type = selector_data['type']
            selector_state = selector_data['selector_state']
            if selector_type == 'box':
                selector = BoxSelector(**selector_state)
            elif selector_type == 'lasso':
                selector = LassoSelector(**selector_state)
            else:
                raise ValueError('Unsupported selector type: %s' % selector_type)
        else:
            selector = None
        if callback_data.get('device_state', None):
            device_state_data = callback_data['device_state']
            state = InputDeviceState(**device_state_data)
        else:
            state = None
        points_data = callback_data['points']
        trace_points = {trace_ind: {'point_inds': [], 'xs': [], 'ys': [], 'trace_name': self._data_objs[trace_ind].name, 'trace_index': trace_ind} for trace_ind in range(len(self._data_objs))}
        for (x, y, point_ind, trace_ind) in zip(points_data['xs'], points_data['ys'], points_data['point_indexes'], points_data['trace_indexes']):
            trace_dict = trace_points[trace_ind]
            trace_dict['xs'].append(x)
            trace_dict['ys'].append(y)
            trace_dict['point_inds'].append(point_ind)
        for (trace_ind, trace_points_data) in trace_points.items():
            points = Points(**trace_points_data)
            trace = self.data[trace_ind]
            if event_type == 'plotly_click':
                trace._dispatch_on_click(points, state)
            elif event_type == 'plotly_hover':
                trace._dispatch_on_hover(points, state)
            elif event_type == 'plotly_unhover':
                trace._dispatch_on_unhover(points, state)
            elif event_type == 'plotly_selected':
                trace._dispatch_on_selection(points, selector)
            elif event_type == 'plotly_deselect':
                trace._dispatch_on_deselect(points)
        self._js2py_pointsCallback = None

    def _repr_html_(self):
        if False:
            i = 10
            return i + 15
        '\n        Customize html representation\n        '
        raise NotImplementedError

    def _repr_mimebundle_(self, include=None, exclude=None, validate=True, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Return mimebundle corresponding to default renderer.\n        '
        return {'application/vnd.jupyter.widget-view+json': {'version_major': 2, 'version_minor': 0, 'model_id': self._model_id}}

    def _ipython_display_(self):
        if False:
            print('Hello World!')
        '\n        Handle rich display of figures in ipython contexts\n        '
        raise NotImplementedError

    def on_edits_completed(self, fn):
        if False:
            while True:
                i = 10
        '\n        Register a function to be called after all pending trace and layout\n        edit operations have completed\n\n        If there are no pending edit operations then function is called\n        immediately\n\n        Parameters\n        ----------\n        fn : callable\n            Function of zero arguments to be called when all pending edit\n            operations have completed\n        '
        if self._layout_edit_in_process or self._trace_edit_in_process:
            self._waiting_edit_callbacks.append(fn)
        else:
            fn()

    @property
    def frames(self):
        if False:
            while True:
                i = 10
        return self._frame_objs

    @frames.setter
    def frames(self, new_frames):
        if False:
            print('Hello World!')
        if new_frames:
            BaseFigureWidget._display_frames_error()

    @staticmethod
    def _display_frames_error():
        if False:
            i = 10
            return i + 15
        '\n        Display an informative error when user attempts to set frames on a\n        FigureWidget\n\n        Raises\n        ------\n        ValueError\n            always\n        '
        msg = '\nFrames are not supported by the plotly.graph_objs.FigureWidget class.\nNote: Frames are supported by the plotly.graph_objs.Figure class'
        raise ValueError(msg)

    @staticmethod
    def _remove_overlapping_props(input_data, delta_data, prop_path=()):
        if False:
            print('Hello World!')
        "\n        Remove properties in input_data that are also in delta_data, and do so\n        recursively.\n\n        Exception: Never remove 'uid' from input_data, this property is used\n        to align traces\n\n        Parameters\n        ----------\n        input_data : dict|list\n        delta_data : dict|list\n\n        Returns\n        -------\n        list[tuple[str|int]]\n            List of removed property path tuples\n        "
        removed = []
        if isinstance(input_data, dict):
            assert isinstance(delta_data, dict)
            for (p, delta_val) in delta_data.items():
                if isinstance(delta_val, dict) or BaseFigure._is_dict_list(delta_val):
                    if p in input_data:
                        input_val = input_data[p]
                        recur_prop_path = prop_path + (p,)
                        recur_removed = BaseFigureWidget._remove_overlapping_props(input_val, delta_val, recur_prop_path)
                        removed.extend(recur_removed)
                        if not input_val:
                            input_data.pop(p)
                            removed.append(recur_prop_path)
                elif p in input_data and p != 'uid':
                    input_data.pop(p)
                    removed.append(prop_path + (p,))
        elif isinstance(input_data, list):
            assert isinstance(delta_data, list)
            for (i, delta_val) in enumerate(delta_data):
                if i >= len(input_data):
                    break
                input_val = input_data[i]
                if input_val is not None and isinstance(delta_val, dict) or BaseFigure._is_dict_list(delta_val):
                    recur_prop_path = prop_path + (i,)
                    recur_removed = BaseFigureWidget._remove_overlapping_props(input_val, delta_val, recur_prop_path)
                    removed.extend(recur_removed)
        return removed

    @staticmethod
    def _transform_data(to_data, from_data, should_remove=True, relayout_path=()):
        if False:
            while True:
                i = 10
        '\n        Transform to_data into from_data and return relayout-style\n        description of the transformation\n\n        Parameters\n        ----------\n        to_data : dict|list\n        from_data : dict|list\n\n        Returns\n        -------\n        dict\n            relayout-style description of the transformation\n        '
        relayout_data = {}
        if isinstance(to_data, dict):
            if not isinstance(from_data, dict):
                raise ValueError('Mismatched data types: {to_dict} {from_data}'.format(to_dict=to_data, from_data=from_data))
            for (from_prop, from_val) in from_data.items():
                if isinstance(from_val, dict) or BaseFigure._is_dict_list(from_val):
                    if from_prop not in to_data:
                        to_data[from_prop] = {} if isinstance(from_val, dict) else []
                    input_val = to_data[from_prop]
                    relayout_data.update(BaseFigureWidget._transform_data(input_val, from_val, should_remove=should_remove, relayout_path=relayout_path + (from_prop,)))
                elif from_prop not in to_data or not BasePlotlyType._vals_equal(to_data[from_prop], from_val):
                    to_data[from_prop] = from_val
                    relayout_path_prop = relayout_path + (from_prop,)
                    relayout_data[relayout_path_prop] = from_val
            if should_remove:
                for remove_prop in set(to_data.keys()).difference(set(from_data.keys())):
                    to_data.pop(remove_prop)
        elif isinstance(to_data, list):
            if not isinstance(from_data, list):
                raise ValueError('Mismatched data types: to_data: {to_data} {from_data}'.format(to_data=to_data, from_data=from_data))
            for (i, from_val) in enumerate(from_data):
                if i >= len(to_data):
                    to_data.append(None)
                input_val = to_data[i]
                if input_val is not None and (isinstance(from_val, dict) or BaseFigure._is_dict_list(from_val)):
                    relayout_data.update(BaseFigureWidget._transform_data(input_val, from_val, should_remove=should_remove, relayout_path=relayout_path + (i,)))
                elif not BasePlotlyType._vals_equal(to_data[i], from_val):
                    to_data[i] = from_val
                    relayout_data[relayout_path + (i,)] = from_val
        return relayout_data