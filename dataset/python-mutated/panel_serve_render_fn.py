"""This file gets run by Python to launch a Panel Server with Lightning.

We will call the ``render_fn`` that the user provided to the PanelFrontend.

It requires the following environment variables to be set


- LIGHTNING_RENDER_FUNCTION
- LIGHTNING_RENDER_MODULE_FILE

Example:

.. code-block:: bash

        python panel_serve_render_fn

"""
import inspect
import os
import pydoc
from typing import Callable
from lightning.app.frontend.panel.app_state_watcher import AppStateWatcher

def _get_render_fn_from_environment(render_fn_name: str, render_fn_module_file: str) -> Callable:
    if False:
        for i in range(10):
            print('nop')
    'Returns the render_fn function to serve in the Frontend.'
    module = pydoc.importfile(render_fn_module_file)
    return getattr(module, render_fn_name)

def _get_render_fn():
    if False:
        for i in range(10):
            print('nop')
    render_fn_name = os.environ['LIGHTNING_RENDER_FUNCTION']
    render_fn_module_file = os.environ['LIGHTNING_RENDER_MODULE_FILE']
    render_fn = _get_render_fn_from_environment(render_fn_name, render_fn_module_file)
    if inspect.signature(render_fn).parameters:

        def _render_fn_wrapper():
            if False:
                return 10
            app = AppStateWatcher()
            return render_fn(app)
        return _render_fn_wrapper
    return render_fn

def _main():
    if False:
        while True:
            i = 10
    import panel as pn
    if 'lightning_render_fn' not in pn.state.cache:
        pn.state.cache['lightning_render_fn'] = _get_render_fn()
    pn.state.cache['lightning_render_fn']()
if __name__.startswith('bokeh'):
    _main()