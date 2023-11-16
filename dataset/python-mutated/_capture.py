"""
Context manager for capturing logs.
"""
from contextlib import contextmanager
from typing import Iterator, List, Sequence, cast
from twisted.logger import globalLogPublisher
from ._interfaces import ILogObserver, LogEvent

@contextmanager
def capturedLogs() -> Iterator[Sequence[LogEvent]]:
    if False:
        i = 10
        return i + 15
    events: List[LogEvent] = []
    observer = cast(ILogObserver, events.append)
    globalLogPublisher.addObserver(observer)
    yield events
    globalLogPublisher.removeObserver(observer)