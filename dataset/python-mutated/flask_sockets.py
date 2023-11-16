from werkzeug.routing import Map, Rule
from werkzeug.exceptions import NotFound
from werkzeug.http import parse_cookie
from flask import request
try:
    from geventwebsocket.gunicorn.workers import GeventWebSocketWorker as Worker
    from geventwebsocket.handler import WebSocketHandler
    from gunicorn.workers.ggevent import PyWSGIHandler
    import gevent
except ImportError:
    pass

class SocketMiddleware(object):

    def __init__(self, wsgi_app, app, socket):
        if False:
            i = 10
            return i + 15
        self.ws = socket
        self.app = app
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        if False:
            return 10
        adapter = self.ws.url_map.bind_to_environ(environ)
        try:
            (handler, values) = adapter.match()
            environment = environ['wsgi.websocket']
            cookie = None
            if 'HTTP_COOKIE' in environ:
                cookie = parse_cookie(environ['HTTP_COOKIE'])
            with self.app.app_context():
                with self.app.request_context(environ):
                    request.cookie = cookie
                    handler(environment, **values)
                    return []
        except (NotFound, KeyError):
            return self.wsgi_app(environ, start_response)

class Sockets(object):

    def __init__(self, app=None):
        if False:
            return 10
        self.url_map = Map()
        self.blueprints = {}
        self._blueprint_order = []
        if app:
            self.init_app(app)

    def init_app(self, app):
        if False:
            return 10
        app.wsgi_app = SocketMiddleware(app.wsgi_app, app, self)

    def route(self, rule, **options):
        if False:
            return 10

        def decorator(f):
            if False:
                for i in range(10):
                    print('nop')
            endpoint = options.pop('endpoint', None)
            self.add_url_rule(rule, endpoint, f, **options)
            return f
        return decorator

    def add_url_rule(self, rule, _, f, **options):
        if False:
            i = 10
            return i + 15
        self.url_map.add(Rule(rule, endpoint=f))

    def register_blueprint(self, blueprint, **options):
        if False:
            while True:
                i = 10
        "\n        Registers a blueprint for web sockets like for 'Flask' application.\n\n        Decorator :meth:`~flask.app.setupmethod` is not applied, because it\n        requires ``debug`` and ``_got_first_request`` attributes to be defined.\n        "
        first_registration = False
        if blueprint.name in self.blueprints:
            assert self.blueprints[blueprint.name] is blueprint, 'A blueprint\'s name collision occurred between %r and %r.  Both share the same name "%s".  Blueprints that are created on the fly need unique names.' % (blueprint, self.blueprints[blueprint.name], blueprint.name)
        else:
            self.blueprints[blueprint.name] = blueprint
            self._blueprint_order.append(blueprint)
            first_registration = True
        blueprint.register(self, options, first_registration)
if 'Worker' in locals() and 'PyWSGIHandler' in locals() and ('gevent' in locals()):

    class GunicornWebSocketHandler(PyWSGIHandler, WebSocketHandler):

        def log_request(self):
            if False:
                i = 10
                return i + 15
            if '101' not in self.status:
                super(GunicornWebSocketHandler, self).log_request()
    Worker.wsgi_handler = GunicornWebSocketHandler
    worker = Worker