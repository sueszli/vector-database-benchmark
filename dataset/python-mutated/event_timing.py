import datetime
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional
logger = logging.getLogger('airbyte')

class EventTimer:
    """Simple nanosecond resolution event timer for debugging, initially intended to be used to record streams execution
    time for a source.
       Event nesting follows a LIFO pattern, so finish will apply to the last started event.
    """

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.events = {}
        self.count = 0
        self.stack = []

    def start_event(self, name):
        if False:
            return 10
        '\n        Start a new event and push it to the stack.\n        '
        self.events[name] = Event(name=name)
        self.count += 1
        self.stack.insert(0, self.events[name])

    def finish_event(self):
        if False:
            while True:
                i = 10
        '\n        Finish the current event and pop it from the stack.\n        '
        if self.stack:
            event = self.stack.pop(0)
            event.finish()
        else:
            logger.warning(f'{self.name} finish_event called without start_event')

    def report(self, order_by='name'):
        if False:
            return 10
        "\n        :param order_by: 'name' or 'duration'\n        "
        if order_by == 'name':
            events = sorted(self.events.values(), key=lambda event: event.name)
        elif order_by == 'duration':
            events = sorted(self.events.values(), key=lambda event: event.duration)
        text = f'{self.name} runtimes:\n'
        text += '\n'.join((str(event) for event in events))
        return text

@dataclass
class Event:
    name: str
    start: float = field(default_factory=time.perf_counter_ns)
    end: Optional[float] = field(default=None)

    @property
    def duration(self) -> float:
        if False:
            return 10
        'Returns the elapsed time in seconds or positive infinity if event was never finished'
        if self.end:
            return (self.end - self.start) / 1000000000.0
        return float('+inf')

    def __str__(self):
        if False:
            return 10
        return f'{self.name} {datetime.timedelta(seconds=self.duration)}'

    def finish(self):
        if False:
            for i in range(10):
                print('nop')
        self.end = time.perf_counter_ns()

@contextmanager
def create_timer(name):
    if False:
        return 10
    '\n    Creates a new EventTimer as a context manager to improve code readability.\n    '
    a_timer = EventTimer(name)
    yield a_timer