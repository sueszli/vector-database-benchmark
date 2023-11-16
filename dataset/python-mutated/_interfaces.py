"""
Logger interfaces.
"""
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
from zope.interface import Interface
if TYPE_CHECKING:
    from ._logger import Logger
LogEvent = Dict[str, Any]
LogTrace = List[Tuple['Logger', 'ILogObserver']]

class ILogObserver(Interface):
    """
    An observer which can handle log events.

    Unlike most interfaces within Twisted, an L{ILogObserver} I{must be
    thread-safe}.  Log observers may be called indiscriminately from many
    different threads, as any thread may wish to log a message at any time.
    """

    def __call__(event: LogEvent) -> None:
        if False:
            print('Hello World!')
        '\n        Log an event.\n\n        @param event: A dictionary with arbitrary keys as defined by the\n            application emitting logging events, as well as keys added by the\n            logging system.  The logging system reserves the right to set any\n            key beginning with the prefix C{"log_"}; applications should not\n            use any key so named.  Currently, the following keys are used by\n            the logging system in some way, if they are present (they are all\n            optional):\n\n                - C{"log_format"}: a PEP-3101-style format string which draws\n                  upon the keys in the event as its values, used to format the\n                  event for human consumption.\n\n                - C{"log_flattened"}: a dictionary mapping keys derived from\n                  the names and format values used in the C{"log_format"}\n                  string to their values.  This is used to preserve some\n                  structured information for use with\n                  L{twisted.logger.extractField}.\n\n                - C{"log_trace"}: A L{list} designed to capture information\n                  about which L{LogPublisher}s have observed the event.\n\n                - C{"log_level"}: a L{log level\n                  <twisted.logger.LogLevel>} constant, indicating the\n                  importance of and audience for this event.\n\n                - C{"log_namespace"}: a namespace for the emitter of the event,\n                  given as a L{str}.\n\n                - C{"log_system"}: a string indicating the network event or\n                  method call which resulted in the message being logged.\n        '