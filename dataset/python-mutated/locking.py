"""

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import asyncio
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, TypeVar, cast
if TYPE_CHECKING:
    from ..server.callbacks import NextTickCallback
    from .document import Callback, Document
__all__ = ('UnlockedDocumentProxy', 'without_document_lock')
F = TypeVar('F', bound=Callable[..., Any])

class NoLockCallback(Protocol[F]):
    __call__: F
    nolock: Literal[True]

def without_document_lock(func: F) -> NoLockCallback[F]:
    if False:
        for i in range(10):
            print('nop')
    ' Wrap a callback function to execute without first obtaining the\n    document lock.\n\n    Args:\n        func (callable) : The function to wrap\n\n    Returns:\n        callable : a function wrapped to execute without a |Document| lock.\n\n    While inside an unlocked callback, it is completely *unsafe* to modify\n    ``curdoc()``. The value of ``curdoc()`` inside the callback will be a\n    specially wrapped version of |Document| that only allows safe operations,\n    which are:\n\n    * :func:`~bokeh.document.Document.add_next_tick_callback`\n    * :func:`~bokeh.document.Document.remove_next_tick_callback`\n\n    Only these may be used safely without taking the document lock. To make\n    other changes to the document, you must add a next tick callback and make\n    your changes to ``curdoc()`` from that second callback.\n\n    Attempts to otherwise access or change the Document will result in an\n    exception being raised.\n\n    ``func`` can be a synchronous function, an async function, or a function\n    decorated with ``asyncio.coroutine``. The returned function will be an\n    async function if ``func`` is any of the latter two.\n\n    '
    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def _wrapper(*args: Any, **kw: Any) -> None:
            await func(*args, **kw)
    else:

        @wraps(func)
        def _wrapper(*args: Any, **kw: Any) -> None:
            if False:
                print('Hello World!')
            func(*args, **kw)
    wrapper = cast(NoLockCallback[F], _wrapper)
    wrapper.nolock = True
    return wrapper
UNSAFE_DOC_ATTR_USAGE_MSG = "Only 'add_next_tick_callback' may be used safely without taking the document lock; to make other changes to the document, add a next tick callback and make your changes from that callback."

class UnlockedDocumentProxy:
    """ Wrap a Document object so that only methods that can safely be used
    from unlocked callbacks or threads are exposed. Attempts to otherwise
    access or change the Document results in an exception.

    """

    def __init__(self, doc: Document) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n\n        '
        self._doc = doc

    def __getattr__(self, attr: str) -> Any:
        if False:
            while True:
                i = 10
        '\n\n        '
        raise AttributeError(UNSAFE_DOC_ATTR_USAGE_MSG)

    def add_next_tick_callback(self, callback: Callback) -> NextTickCallback:
        if False:
            while True:
                i = 10
        ' Add a "next tick" callback.\n\n        Args:\n            callback (callable) :\n\n        '
        return self._doc.add_next_tick_callback(callback)

    def remove_next_tick_callback(self, callback: NextTickCallback) -> None:
        if False:
            i = 10
            return i + 15
        ' Remove a "next tick" callback.\n\n        Args:\n            callback (callable) :\n\n        '
        self._doc.remove_next_tick_callback(callback)