"""This file gets run by streamlit, which we launch within Lightning.

From here, we will call the render function that the user provided in ``configure_layout``.

"""
import os
import pydoc
from typing import Callable
from lightning.app.frontend.utils import _reduce_to_flow_scope
from lightning.app.utilities.app_helpers import StreamLitStatePlugin
from lightning.app.utilities.state import AppState

def _get_render_fn_from_environment() -> Callable:
    if False:
        for i in range(10):
            print('nop')
    render_fn_name = os.environ['LIGHTNING_RENDER_FUNCTION']
    render_fn_module_file = os.environ['LIGHTNING_RENDER_MODULE_FILE']
    module = pydoc.importfile(render_fn_module_file)
    return getattr(module, render_fn_name)

def _main():
    if False:
        return 10
    'Run the render_fn with the current flow_state.'
    app_state = AppState(plugin=StreamLitStatePlugin())
    flow_state = _reduce_to_flow_scope(app_state, flow=os.environ['LIGHTNING_FLOW_NAME'])
    render_fn = _get_render_fn_from_environment()
    render_fn(flow_state)
if __name__ == '__main__':
    _main()