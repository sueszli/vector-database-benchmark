""" Bokeh Application Handler to look for Bokeh server request callbacks
in a specified Python module.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING, Any, Callable
from .handler import Handler
if TYPE_CHECKING:
    from tornado.httputil import HTTPServerRequest
__all__ = ('RequestHandler',)

class RequestHandler(Handler):
    """ Load a script which contains server request handler callbacks.

    """
    _process_request: Callable[[HTTPServerRequest], dict[str, Any]]

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._process_request = _return_empty

    def process_request(self, request: HTTPServerRequest) -> dict[str, Any]:
        if False:
            i = 10
            return i + 15
        ' Processes incoming HTTP request returning a dictionary of\n        additional data to add to the session_context.\n\n        Args:\n            request: HTTP request\n\n        Returns:\n            A dictionary of JSON serializable data to be included on\n            the session context.\n        '
        return self._process_request(request)

    @property
    def safe_to_fork(self) -> bool:
        if False:
            i = 10
            return i + 15
        return True

def _return_empty(request: HTTPServerRequest) -> dict[str, Any]:
    if False:
        print('Hello World!')
    return {}