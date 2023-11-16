from reactpy import run
from reactpy.backend import flask as flask_server
flask_server.configure = lambda _, cmpt: run(cmpt)
from flask import Flask
from reactpy import component, html
from reactpy.backend.flask import configure

@component
def HelloWorld():
    if False:
        return 10
    return html.h1('Hello, world!')
app = Flask(__name__)
configure(app, HelloWorld)