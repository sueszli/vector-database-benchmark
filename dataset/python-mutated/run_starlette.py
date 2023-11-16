from reactpy import run
from reactpy.backend import starlette as starlette_server
starlette_server.configure = lambda _, cmpt: run(cmpt)
from starlette.applications import Starlette
from reactpy import component, html
from reactpy.backend.starlette import configure

@component
def HelloWorld():
    if False:
        i = 10
        return i + 15
    return html.h1('Hello, world!')
app = Starlette()
configure(app, HelloWorld)