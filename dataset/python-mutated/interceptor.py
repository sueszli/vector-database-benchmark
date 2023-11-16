"""APIs related to intercepting/blocking requests."""
from qutebrowser.extensions import interceptors
from qutebrowser.extensions.interceptors import Request
InterceptorType = interceptors.InterceptorType
ResourceType = interceptors.ResourceType

def register(interceptor: InterceptorType) -> None:
    if False:
        i = 10
        return i + 15
    "Register a request interceptor.\n\n    Whenever a request happens, the interceptor gets called with a\n    :class:`Request` object.\n\n    Example::\n\n        def intercept(request: interceptor.Request) -> None:\n            if request.request_url.host() == 'badhost.example.com':\n                request.block()\n    "
    interceptors.register(interceptor)