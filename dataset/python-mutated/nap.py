import time
import typing
if typing.TYPE_CHECKING:
    import threading

def sleep(seconds: float) -> None:
    if False:
        while True:
            i = 10
    '\n    Sleep strategy that delays execution for a given number of seconds.\n\n    This is the default strategy, and may be mocked out for unit testing.\n    '
    time.sleep(seconds)

class sleep_using_event:
    """Sleep strategy that waits on an event to be set."""

    def __init__(self, event: 'threading.Event') -> None:
        if False:
            i = 10
            return i + 15
        self.event = event

    def __call__(self, timeout: typing.Optional[float]) -> None:
        if False:
            return 10
        self.event.wait(timeout=timeout)