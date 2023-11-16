from typing import Any, List, Tuple
from pydispatch import dispatcher
from twisted.internet.defer import Deferred
from scrapy.utils import signal as _signal

class SignalManager:

    def __init__(self, sender: Any=dispatcher.Anonymous):
        if False:
            return 10
        self.sender: Any = sender

    def connect(self, receiver: Any, signal: Any, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Connect a receiver function to a signal.\n\n        The signal can be any object, although Scrapy comes with some\n        predefined signals that are documented in the :ref:`topics-signals`\n        section.\n\n        :param receiver: the function to be connected\n        :type receiver: collections.abc.Callable\n\n        :param signal: the signal to connect to\n        :type signal: object\n        '
        kwargs.setdefault('sender', self.sender)
        dispatcher.connect(receiver, signal, **kwargs)

    def disconnect(self, receiver: Any, signal: Any, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        '\n        Disconnect a receiver function from a signal. This has the\n        opposite effect of the :meth:`connect` method, and the arguments\n        are the same.\n        '
        kwargs.setdefault('sender', self.sender)
        dispatcher.disconnect(receiver, signal, **kwargs)

    def send_catch_log(self, signal: Any, **kwargs: Any) -> List[Tuple[Any, Any]]:
        if False:
            return 10
        '\n        Send a signal, catch exceptions and log them.\n\n        The keyword arguments are passed to the signal handlers (connected\n        through the :meth:`connect` method).\n        '
        kwargs.setdefault('sender', self.sender)
        return _signal.send_catch_log(signal, **kwargs)

    def send_catch_log_deferred(self, signal: Any, **kwargs: Any) -> Deferred:
        if False:
            for i in range(10):
                print('nop')
        '\n        Like :meth:`send_catch_log` but supports returning\n        :class:`~twisted.internet.defer.Deferred` objects from signal handlers.\n\n        Returns a Deferred that gets fired once all signal handlers\n        deferreds were fired. Send a signal, catch exceptions and log them.\n\n        The keyword arguments are passed to the signal handlers (connected\n        through the :meth:`connect` method).\n        '
        kwargs.setdefault('sender', self.sender)
        return _signal.send_catch_log_deferred(signal, **kwargs)

    def disconnect_all(self, signal: Any, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Disconnect all receivers from the given signal.\n\n        :param signal: the signal to disconnect from\n        :type signal: object\n        '
        kwargs.setdefault('sender', self.sender)
        _signal.disconnect_all(signal, **kwargs)