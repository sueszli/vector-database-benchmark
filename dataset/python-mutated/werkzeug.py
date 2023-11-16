from werkzeug import run_simple
from ..gateway import Gateway
from .wsgi import WsgiGateway

def serve(gateway: Gateway, host='localhost', port=4566, use_reloader=True, **kwargs) -> None:
    if False:
        return 10
    '\n    Serve a Gateway as a WSGI application through werkzeug. This is mostly for development purposes.\n\n    :param gateway: the Gateway to serve\n    :param host: the host to expose the server to\n    :param port: the port to expose the server to\n    :param use_reloader: whether to autoreload the server on changes\n    :param kwargs: any other arguments that can be passed to `werkzeug.run_simple`\n    '
    kwargs['threaded'] = kwargs.get('threaded', True)
    run_simple(host, port, WsgiGateway(gateway), use_reloader=use_reloader, **kwargs)