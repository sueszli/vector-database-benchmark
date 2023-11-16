from __future__ import annotations
import datetime
import time
from typing import Union
from airflow.typing_compat import Protocol
DeltaType = Union[int, float, datetime.timedelta]

class TimerProtocol(Protocol):
    """Type protocol for StatsLogger.timer."""

    def __enter__(self) -> Timer:
        if False:
            for i in range(10):
                print('nop')
        ...

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if False:
            i = 10
            return i + 15
        ...

    def start(self) -> Timer:
        if False:
            return 10
        'Start the timer.'
        ...

    def stop(self, send: bool=True) -> None:
        if False:
            print('Hello World!')
        'Stop, and (by default) submit the timer to StatsD.'
        ...

class Timer(TimerProtocol):
    """
    Timer that records duration, and optional sends to StatsD backend.

    This class lets us have an accurate timer with the logic in one place (so
    that we don't use datetime math for duration -- it is error prone).

    Example usage:

    .. code-block:: python

        with Stats.timer() as t:
            # Something to time
            frob_the_foos()

        log.info("Frobbing the foos took %.2f", t.duration)

    Or without a context manager:

    .. code-block:: python

        timer = Stats.timer().start()

        # Something to time
        frob_the_foos()

        timer.end()

        log.info("Frobbing the foos took %.2f", timer.duration)

    To send a metric:

    .. code-block:: python

        with Stats.timer("foos.frob"):
            # Something to time
            frob_the_foos()

    Or both:

    .. code-block:: python

        with Stats.timer("foos.frob") as t:
            # Something to time
            frob_the_foos()

        log.info("Frobbing the foos took %.2f", t.duration)
    """
    _start_time: float | None
    duration: float | None

    def __init__(self, real_timer: Timer | None=None) -> None:
        if False:
            while True:
                i = 10
        self.real_timer = real_timer

    def __enter__(self) -> Timer:
        if False:
            i = 10
            return i + 15
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if False:
            while True:
                i = 10
        self.stop()

    def start(self) -> Timer:
        if False:
            print('Hello World!')
        'Start the timer.'
        if self.real_timer:
            self.real_timer.start()
        self._start_time = time.perf_counter()
        return self

    def stop(self, send: bool=True) -> None:
        if False:
            print('Hello World!')
        'Stop the timer, and optionally send it to stats backend.'
        if self._start_time is not None:
            self.duration = time.perf_counter() - self._start_time
        if send and self.real_timer:
            self.real_timer.stop()