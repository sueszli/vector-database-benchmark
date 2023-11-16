import ipywidgets
import traitlets
import IPython
import json
import functools
import open3d as o3d
from open3d.visualization.async_event_loop import async_event_loop
from open3d._build_config import _build_config
if not _build_config['BUILD_JUPYTER_EXTENSION']:
    raise RuntimeError('Open3D WebVisualizer Jupyter extension is not available. To use WebVisualizer, build Open3D with -DBUILD_JUPYTER_EXTENSION=ON.')

@ipywidgets.register
class WebVisualizer(ipywidgets.DOMWidget):
    """Open3D Web Visualizer based on WebRTC."""
    _view_name = traitlets.Unicode('WebVisualizerView').tag(sync=True)
    _model_name = traitlets.Unicode('WebVisualizerModel').tag(sync=True)
    _view_module = traitlets.Unicode('open3d').tag(sync=True)
    _model_module = traitlets.Unicode('open3d').tag(sync=True)
    _view_module_version = traitlets.Unicode('~@PROJECT_VERSION_THREE_NUMBER@').tag(sync=True)
    _model_module_version = traitlets.Unicode('~@PROJECT_VERSION_THREE_NUMBER@').tag(sync=True)
    window_uid = traitlets.Unicode('window_UNDEFINED', help='Window UID').tag(sync=True)
    pyjs_channel = traitlets.Unicode('Empty pyjs_channel.', help='Python->JS message channel.').tag(sync=True)
    jspy_channel = traitlets.Unicode('Empty jspy_channel.', help='JS->Python message channel.').tag(sync=True)

    def show(self):
        if False:
            i = 10
            return i + 15
        IPython.display.display(self)

    def _call_http_api(self, entry_point, query_string, data):
        if False:
            return 10
        return o3d.visualization.webrtc_server.call_http_api(entry_point, query_string, data)

    @traitlets.validate('window_uid')
    def _valid_window_uid(self, proposal):
        if False:
            while True:
                i = 10
        if proposal['value'][:7] != 'window_':
            raise traitlets.TraitError('window_uid must be "window_xxx".')
        return proposal['value']

    @traitlets.observe('jspy_channel')
    def _on_jspy_channel(self, change):
        if False:
            return 10
        if not hasattr(self, 'result_map'):
            self.result_map = dict()
        jspy_message = change['new']
        try:
            jspy_requests = json.loads(jspy_message)
            for (call_id, payload) in jspy_requests.items():
                if 'func' not in payload or payload['func'] != 'call_http_api':
                    raise ValueError(f'Invalid jspy function: {jspy_requests}')
                if 'args' not in payload or len(payload['args']) != 3:
                    raise ValueError(f'Invalid jspy function arguments: {jspy_requests}')
                if not call_id in self.result_map:
                    json_result = self._call_http_api(payload['args'][0], payload['args'][1], payload['args'][2])
                    self.result_map[call_id] = json_result
        except:
            print(f'jspy_message is not a function call, ignored: {jspy_message}')
        else:
            self.pyjs_channel = json.dumps(self.result_map)

def draw(geometry=None, title='Open3D', width=640, height=480, actions=None, lookat=None, eye=None, up=None, field_of_view=60.0, bg_color=(1.0, 1.0, 1.0, 1.0), bg_image=None, show_ui=None, point_size=None, animation_time_step=1.0, animation_duration=None, rpc_interface=False, on_init=None, on_animation_frame=None, on_animation_tick=None):
    if False:
        i = 10
        return i + 15
    'Draw in Jupyter Cell'
    window_uid = async_event_loop.run_sync(functools.partial(o3d.visualization.draw, geometry=geometry, title=title, width=width, height=height, actions=actions, lookat=lookat, eye=eye, up=up, field_of_view=field_of_view, bg_color=bg_color, bg_image=bg_image, show_ui=show_ui, point_size=point_size, animation_time_step=animation_time_step, animation_duration=animation_duration, rpc_interface=rpc_interface, on_init=on_init, on_animation_frame=on_animation_frame, on_animation_tick=on_animation_tick, non_blocking_and_return_uid=True))
    visualizer = WebVisualizer(window_uid=window_uid)
    visualizer.show()