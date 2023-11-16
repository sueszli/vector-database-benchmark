import inspect
import logging
import typing
import warnings
import weakref
from collections import Callable
from golem_messages import message
logger = logging.getLogger(__name__)

class DuplicatedHandler(UserWarning):
    pass

class HandlersLibrary:
    """Library of handlers for messages received from concent"""
    __slots__ = ('_handlers',)

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self._handlers: typing.Dict[message.base.Message, weakref.ref] = {}

    def register_handler(self, msg_cls: message.base.Message) -> Callable:
        if False:
            return 10

        def _wrapped(f) -> Callable:
            if False:
                for i in range(10):
                    print('nop')
            try:
                if self._handlers[msg_cls]() is not None:
                    warnings.warn('Duplicated handler for {msg_cls}. Replacing {current_handler} with {new_handler}'.format(msg_cls=msg_cls.__name__, current_handler=self._handlers[msg_cls](), new_handler=f), DuplicatedHandler)
            except KeyError:
                pass
            ref: typing.Optional[weakref.ref] = None
            if inspect.ismethod(f):
                ref = weakref.WeakMethod(f)
            else:
                ref = weakref.ref(f)
            self._handlers[msg_cls] = ref
            return f
        return _wrapped

    def interpret(self, msg, response_to: message.base.Message=None) -> None:
        if False:
            while True:
                i = 10
        try:
            ref = self._handlers[msg.__class__]
            handler = ref()
            if handler is None:
                raise KeyError('Handler was defined but it has been garbage collected')
        except KeyError:
            logger.debug('interpret(%r) err', msg, exc_info=True)
            logger.warning("I don't know how to handle %s. Ignoring %r", msg.__class__, msg)
            return
        if not response_to:
            handler(msg)
        else:
            handler(msg, response_to=response_to)
library = HandlersLibrary()