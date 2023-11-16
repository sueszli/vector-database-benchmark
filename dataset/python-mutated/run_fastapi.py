from reactpy import run
from reactpy.backend import fastapi as fastapi_server
fastapi_server.configure = lambda _, cmpt: run(cmpt)
from fastapi import FastAPI
from reactpy import component, html
from reactpy.backend.fastapi import configure

@component
def HelloWorld():
    if False:
        print('Hello World!')
    return html.h1('Hello, world!')
app = FastAPI()
configure(app, HelloWorld)