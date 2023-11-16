from __future__ import annotations
import time
from typing import TypeVar
from arroyo.processing.strategies import MessageRejected, ProcessingStrategy, RunTask
from arroyo.types import FilteredPayload, Message
from sentry import options
from sentry.processing.backpressure.health import is_consumer_healthy

class HealthChecker:

    def __init__(self, consumer_name: str='default'):
        if False:
            while True:
                i = 10
        self.consumer_name = consumer_name
        self.last_check: float = 0
        self.is_queue_healthy = True

    def is_healthy(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        now = time.time()
        if now - self.last_check >= options.get('backpressure.checking.interval'):
            self.is_queue_healthy = is_consumer_healthy(self.consumer_name)
            self.last_check = now
        return self.is_queue_healthy
TPayload = TypeVar('TPayload')

def create_backpressure_step(health_checker: HealthChecker, next_step: ProcessingStrategy[FilteredPayload | TPayload]) -> ProcessingStrategy[FilteredPayload | TPayload]:
    if False:
        return 10
    '\n    This creates a new arroyo `ProcessingStrategy` that will check the `HealthChecker`\n    and reject messages if the downstream step is not healthy.\n    This strategy can be chained in front of the `next_step` that will do the actual\n    processing.\n    '

    def ensure_healthy_queue(message: Message[TPayload]) -> TPayload:
        if False:
            while True:
                i = 10
        if not health_checker.is_healthy():
            raise MessageRejected()
        return message.payload
    return RunTask(function=ensure_healthy_queue, next_step=next_step)