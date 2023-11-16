""" Bokeh Application Handler to look for Bokeh server lifecycle callbacks
in a specified Python module.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import Any, Callable
from ...document import Document
from ..application import ServerContext, SessionContext
from .handler import Handler
__all__ = ('LifecycleHandler',)

class LifecycleHandler(Handler):
    """ Load a script which contains server lifecycle callbacks.

    """
    _on_server_loaded: Callable[[ServerContext], None]
    _on_server_unloaded: Callable[[ServerContext], None]
    _on_session_created: Callable[[SessionContext], None]
    _on_session_destroyed: Callable[[SessionContext], None]

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self._on_server_loaded = _do_nothing
        self._on_server_unloaded = _do_nothing
        self._on_session_created = _do_nothing
        self._on_session_destroyed = _do_nothing

    @property
    def safe_to_fork(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def modify_document(self, doc: Document) -> None:
        if False:
            while True:
                i = 10
        ' This handler does not make any modifications to the Document.\n\n        Args:\n            doc (Document) : A Bokeh Document to update in-place\n\n                *This handler does not modify the document*\n\n        Returns:\n            None\n\n        '
        pass

    def on_server_loaded(self, server_context: ServerContext) -> None:
        if False:
            while True:
                i = 10
        ' Execute `on_server_unloaded`` from the configured module (if\n        it is defined) when the server is first started.\n\n        Args:\n            server_context (ServerContext) :\n\n        '
        return self._on_server_loaded(server_context)

    def on_server_unloaded(self, server_context: ServerContext) -> None:
        if False:
            print('Hello World!')
        " Execute ``on_server_unloaded`` from the configured module (if\n        it is defined) when the server cleanly exits. (Before stopping the\n        server's ``IOLoop``.)\n\n        Args:\n            server_context (ServerContext) :\n\n        .. warning::\n            In practice this code may not run, since servers are often killed\n            by a signal.\n\n        "
        return self._on_server_unloaded(server_context)

    async def on_session_created(self, session_context: SessionContext) -> None:
        """ Execute ``on_session_created`` from the configured module (if
        it is defined) when a new session is created.

        Args:
            session_context (SessionContext) :

        """
        return self._on_session_created(session_context)

    async def on_session_destroyed(self, session_context: SessionContext) -> None:
        """ Execute ``on_session_destroyed`` from the configured module (if
        it is defined) when a new session is destroyed.

        Args:
            session_context (SessionContext) :

        """
        return self._on_session_destroyed(session_context)

def _do_nothing(ignored: Any) -> None:
    if False:
        return 10
    pass