from __future__ import annotations
import sys
from contextlib import contextmanager
from contextvars import ContextVar
IS_WASM = sys.platform == 'emscripten'

class WasmUnsupportedError(Exception):
    pass
app_map = {}
_app_id_context_var: ContextVar[str | None] = ContextVar('app_id', default=None)

@contextmanager
def app_id_context(app_id: str):
    if False:
        i = 10
        return i + 15
    token = _app_id_context_var.set(app_id)
    yield
    _app_id_context_var.reset(token)

def register_app(_app):
    if False:
        print('Hello World!')
    global app_map
    app_id = _app_id_context_var.get()
    if app_id in app_map:
        app = app_map[app_id]
        app.blocks.close()
    app_map[app_id] = _app

def get_registered_app(app_id: str):
    if False:
        while True:
            i = 10
    global app_map
    return app_map[app_id]