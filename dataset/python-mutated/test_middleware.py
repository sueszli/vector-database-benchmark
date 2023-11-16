from starlette.middleware import Middleware

class CustomMiddleware:
    pass

def test_middleware_repr():
    if False:
        i = 10
        return i + 15
    middleware = Middleware(CustomMiddleware)
    assert repr(middleware) == 'Middleware(CustomMiddleware)'