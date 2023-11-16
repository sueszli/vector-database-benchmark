import os
import pydoc
from typing import Any, Callable
from lightning.app.frontend.utils import _reduce_to_flow_scope
from lightning.app.utilities.state import AppState

def _get_state() -> AppState:
    if False:
        i = 10
        return i + 15
    app_state = AppState()
    return _reduce_to_flow_scope(app_state, flow=os.environ['LIGHTNING_FLOW_NAME'])

def _webpage() -> Any:
    if False:
        i = 10
        return i + 15
    import justpy as jp
    wp = jp.WebPage()
    d = jp.Div(text='')
    wp.add(d)
    return wp

def _get_render_fn_from_environment() -> Callable:
    if False:
        return 10
    render_fn_name = os.environ['LIGHTNING_RENDER_FUNCTION']
    render_fn_module_file = os.environ['LIGHTNING_RENDER_MODULE_FILE']
    module = pydoc.importfile(render_fn_module_file)
    return getattr(module, render_fn_name)

def _main() -> None:
    if False:
        return 10
    'Run the render_fn with the current flow_state.'
    import justpy as jp
    flow_name = os.environ['LIGHTNING_FLOW_NAME']
    render_fn = _get_render_fn_from_environment()
    host = os.environ['LIGHTNING_HOST']
    port = int(os.environ['LIGHTNING_PORT'])
    entry_fn = render_fn(_get_state)
    if not isinstance(entry_fn, Callable):
        raise Exception('You need to return a function with JustPy Frontend.')
    jp.app.add_jproute(f'/{flow_name}', entry_fn)
    jp.justpy(_webpage, host=host, port=port)
if __name__ == '__main__':
    _main()