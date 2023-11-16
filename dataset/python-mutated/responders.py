"""Default responder implementations."""
from falcon.errors import HTTPBadRequest
from falcon.errors import HTTPMethodNotAllowed
from falcon.errors import HTTPRouteNotFound
from falcon.status_codes import HTTP_200

def path_not_found(req, resp, **kwargs):
    if False:
        return 10
    'Raise 404 HTTPRouteNotFound error.'
    raise HTTPRouteNotFound()

async def path_not_found_async(req, resp, **kwargs):
    """Raise 404 HTTPRouteNotFound error."""
    raise HTTPRouteNotFound()

def bad_request(req, resp, **kwargs):
    if False:
        return 10
    'Raise 400 HTTPBadRequest error.'
    raise HTTPBadRequest(title='Bad request', description='Invalid HTTP method')

async def bad_request_async(req, resp, **kwargs):
    """Raise 400 HTTPBadRequest error."""
    raise HTTPBadRequest(title='Bad request', description='Invalid HTTP method')

def create_method_not_allowed(allowed_methods, asgi=False):
    if False:
        while True:
            i = 10
    'Create a responder for "405 Method Not Allowed".\n\n    Args:\n        allowed_methods: A list of HTTP methods (uppercase) that should be\n            returned in the Allow header.\n        asgi (bool): ``True`` if using an ASGI app, ``False`` otherwise\n            (default ``False``).\n    '
    if asgi:

        async def method_not_allowed_responder_async(req, resp, **kwargs):
            raise HTTPMethodNotAllowed(allowed_methods)
        return method_not_allowed_responder_async

    def method_not_allowed(req, resp, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        raise HTTPMethodNotAllowed(allowed_methods)
    return method_not_allowed

def create_default_options(allowed_methods, asgi=False):
    if False:
        return 10
    'Create a default responder for the OPTIONS method.\n\n    Args:\n        allowed_methods (iterable): An iterable of HTTP methods (uppercase)\n            that should be returned in the Allow header.\n        asgi (bool): ``True`` if using an ASGI app, ``False`` otherwise\n            (default ``False``).\n    '
    allowed = ', '.join(allowed_methods)
    if asgi:

        async def options_responder_async(req, resp, **kwargs):
            resp.status = HTTP_200
            resp.set_header('Allow', allowed)
            resp.set_header('Content-Length', '0')
        return options_responder_async

    def options_responder(req, resp, **kwargs):
        if False:
            print('Hello World!')
        resp.status = HTTP_200
        resp.set_header('Allow', allowed)
        resp.set_header('Content-Length', '0')
    return options_responder