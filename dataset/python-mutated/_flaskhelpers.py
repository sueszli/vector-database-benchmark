"""
This file contains the main functions used to implement a flask/gevent server
hosting a flexx application.

The chain of initialisation is the following:

# Import
from flexx import flx_flask
# Define one or multiple classes
class Example1(flx.Widget):
    ...
# Register the class to the server (you can define more than one)
flx_flask.serve(Example1)

# Instantiate the Socket class and then register all flexx apps.
# The flexx apps are individually registered as one Blueprint each.
sockets = Sockets(app)  # keep at the end
flx_flask.register_blueprints(app, sockets, static_folder='static')

# Start the flexx thread to manage the flexx asyncio worker loop.
flx_flask.start_thread()

# You can then start the flask/gevent server.

See the howtos/flask_server.py example for a working example.
"""
import flask
from ._app import manager, App
from ._server import current_server
flexxBlueprint = flask.Blueprint('FlexxApps', __name__, static_folder='static')
flexxWS = flask.Blueprint('flexxWS', __name__)
_blueprints_registered = False
import os
import inspect

def register_blueprints(app, sockets, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Register all flexx apps to flask. Flask will create one URL per application plus a\n    generic /flexx/ URL for serving assets and data.\n\n    see flexxamples/howtos/flask_server.py for a full example.\n    '
    global _blueprints_registered
    if _blueprints_registered:
        return
    frame = inspect.stack()[1]
    p = frame[0].f_code.co_filename
    caller_path = os.path.dirname(p)
    for (key, value) in kwargs.items():
        if key in ['static_folder', 'static_url_path', 'template_folder']:
            kwargs[key] = os.path.abspath(os.path.join(caller_path, value))
    for name in manager._appinfo.keys():
        appBlueprint = flask.Blueprint(f'Flexx_{name}', __name__, **kwargs)
        from ._flaskserver import AppHandler

        def app_handler():
            if False:
                i = 10
                return i + 15
            return AppHandler(flask.request).run()
        appBlueprint.route('/')(app_handler)
        app_handler.__name__ = name

        def app_static_handler(path):
            if False:
                while True:
                    i = 10
            return appBlueprint.send_static_file(path)
        appBlueprint.route('/<path:path>')(app_static_handler)
        app.register_blueprint(appBlueprint, url_prefix=f'/{name}')
    app.register_blueprint(flexxBlueprint, url_prefix='/flexx')
    sockets.register_blueprint(flexxWS, url_prefix='/flexx')
    _blueprints_registered = True
    return

def serve(cls):
    if False:
        for i in range(10):
            print('nop')
    '\n    This function registers the flexx Widget to the manager so the server can\n    serve them properly from the server.\n    '
    m = App(cls)
    if not m._is_served:
        m.serve()

def _start(loop):
    if False:
        for i in range(10):
            print('nop')
    '\n    Start the flexx event loop only. This function generally does not\n    return until the application is stopped.\n\n    In more detail, this calls ``run_forever()`` on the asyncio event loop\n    associated with the current server.\n    '
    server = current_server(backend='flask', loop=loop)
    server.start_serverless()

def start_thread():
    if False:
        print('Hello World!')
    '\n    Starts the flexx thread that manages the flexx asyncio worker loop.\n    '
    import threading
    import asyncio
    flexx_loop = asyncio.new_event_loop()

    def flexx_thread(loop):
        if False:
            print('Hello World!')
        '\n        Function to start a thread containing the main loop of flexx.\n        This is needed as flexx is an asyncio application which is not\n        compatible with flask/gevent.\n        '
        asyncio.set_event_loop(loop)
        _start(loop)
    thread1 = threading.Thread(target=flexx_thread, args=(flexx_loop,))
    thread1.daemon = True
    thread1.start()