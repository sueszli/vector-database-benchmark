from __future__ import annotations
from fastapi import FastAPI
from reactpy.backend import starlette
Options = starlette.Options
configure = starlette.configure

def create_development_app() -> FastAPI:
    if False:
        print('Hello World!')
    'Create a development ``FastAPI`` application instance.'
    return FastAPI(debug=True)
serve_development_app = starlette.serve_development_app
use_connection = starlette.use_connection
use_websocket = starlette.use_websocket