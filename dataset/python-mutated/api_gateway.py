from __future__ import annotations
from typing import Any, Callable
from django.http.response import HttpResponseBase
from rest_framework.request import Request
from sentry.api_gateway import proxy_request_if_needed

class ApiGatewayMiddleware:
    """Proxy requests intended for remote silos"""

    def __init__(self, get_response: Callable[[Request], HttpResponseBase]):
        if False:
            return 10
        self.get_response = get_response

    def __call__(self, request: Request) -> HttpResponseBase:
        if False:
            i = 10
            return i + 15
        return self.get_response(request)

    def process_view(self, request: Request, view_func: Callable[..., HttpResponseBase], view_args: tuple[str], view_kwargs: dict[str, Any]) -> HttpResponseBase | None:
        if False:
            while True:
                i = 10
        proxy_response = proxy_request_if_needed(request, view_func, view_kwargs)
        if proxy_response is not None:
            return proxy_response
        else:
            return None