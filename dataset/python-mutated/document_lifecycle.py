""" Bokeh Application Handler to execute on_session_destroyed callbacks defined
on the Document.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ..application import SessionContext
from .lifecycle import LifecycleHandler
__all__ = ('DocumentLifecycleHandler',)

class DocumentLifecycleHandler(LifecycleHandler):
    """ Calls on_session_destroyed callbacks defined on the Document.
    """

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self._on_session_destroyed = _on_session_destroyed

def _on_session_destroyed(session_context: SessionContext) -> None:
    if False:
        return 10
    '\n    Calls any on_session_destroyed callbacks defined on the Document\n    '
    callbacks = session_context._document.session_destroyed_callbacks
    session_context._document.session_destroyed_callbacks = set()
    for callback in callbacks:
        try:
            callback(session_context)
        except Exception as e:
            log.warning(f'DocumentLifeCycleHandler on_session_destroyed callback {callback} failed with following error: {e}')
    if callbacks:
        del callback
        del callbacks
        import gc
        gc.collect()