from functools import wraps
from asgiref.sync import iscoroutinefunction
from django.utils.cache import patch_vary_headers

def vary_on_headers(*headers):
    if False:
        return 10
    "\n    A view decorator that adds the specified headers to the Vary header of the\n    response. Usage:\n\n       @vary_on_headers('Cookie', 'Accept-language')\n       def index(request):\n           ...\n\n    Note that the header names are not case-sensitive.\n    "

    def decorator(func):
        if False:
            i = 10
            return i + 15
        if iscoroutinefunction(func):

            async def _view_wrapper(request, *args, **kwargs):
                response = await func(request, *args, **kwargs)
                patch_vary_headers(response, headers)
                return response
        else:

            def _view_wrapper(request, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                response = func(request, *args, **kwargs)
                patch_vary_headers(response, headers)
                return response
        return wraps(func)(_view_wrapper)
    return decorator
vary_on_cookie = vary_on_headers('Cookie')
vary_on_cookie.__doc__ = 'A view decorator that adds "Cookie" to the Vary header of a response. This indicates that a page\'s contents depends on cookies.'