"""Module to condition how Interactive Beam stops capturing data.

For internal use only; no backwards-compatibility guarantees.
"""
import threading
import pandas as pd
from apache_beam.portability.api import beam_interactive_api_pb2
from apache_beam.portability.api import beam_runner_api_pb2
from apache_beam.runners.interactive import interactive_environment as ie
from apache_beam.utils.windowed_value import WindowedValue

class Limiter:
    """Limits an aspect of the caching layer."""

    def is_triggered(self):
        if False:
            i = 10
            return i + 15
        'Returns True if the limiter has triggered, and caching should stop.'
        raise NotImplementedError

class ElementLimiter(Limiter):
    """A `Limiter` that limits reading from cache based on some property of an
  element.
  """

    def update(self, e):
        if False:
            while True:
                i = 10
        'Update the internal state based on some property of an element.\n\n    This is executed on every element that is read from cache.\n    '
        raise NotImplementedError

class SizeLimiter(Limiter):
    """Limits the cache size to a specified byte limit."""

    def __init__(self, size_limit):
        if False:
            for i in range(10):
                print('nop')
        self._size_limit = size_limit

    def is_triggered(self):
        if False:
            while True:
                i = 10
        total_capture_size = 0
        ie.current_env().track_user_pipelines()
        for user_pipeline in ie.current_env().tracked_user_pipelines:
            cache_manager = ie.current_env().get_cache_manager(user_pipeline)
            if hasattr(cache_manager, 'capture_size'):
                total_capture_size += cache_manager.capture_size
        return total_capture_size >= self._size_limit

class DurationLimiter(Limiter):
    """Limits the duration of the capture."""

    def __init__(self, duration_limit):
        if False:
            print('Hello World!')
        self._duration_limit = duration_limit
        self._timer = threading.Timer(duration_limit.total_seconds(), self._trigger)
        self._timer.daemon = True
        self._triggered = False
        self._timer.start()

    def _trigger(self):
        if False:
            while True:
                i = 10
        self._triggered = True

    def is_triggered(self):
        if False:
            while True:
                i = 10
        return self._triggered

class CountLimiter(ElementLimiter):
    """Limits by counting the number of elements seen."""

    def __init__(self, max_count):
        if False:
            for i in range(10):
                print('nop')
        self._max_count = max_count
        self._count = 0

    def update(self, e):
        if False:
            while True:
                i = 10
        if isinstance(e, beam_interactive_api_pb2.TestStreamFileRecord):
            if not e.recorded_event.element_event:
                return
            self._count += len(e.recorded_event.element_event.elements)
        elif not isinstance(e, beam_interactive_api_pb2.TestStreamFileHeader):
            if isinstance(e, WindowedValue) and isinstance(e.value, pd.DataFrame):
                self._count += len(e.value)
            else:
                self._count += 1

    def is_triggered(self):
        if False:
            while True:
                i = 10
        return self._count >= self._max_count

class ProcessingTimeLimiter(ElementLimiter):
    """Limits by how long the ProcessingTime passed in the element stream.

  Reads all elements from the timespan [start, start + duration).

  This measures the duration from the first element in the stream. Each
  subsequent element has a delta "advance_duration" that moves the internal
  clock forward. This triggers when the duration from the internal clock and
  the start exceeds the given duration.
  """

    def __init__(self, max_duration_secs):
        if False:
            i = 10
            return i + 15
        'Initialize the ProcessingTimeLimiter.'
        self._max_duration_us = max_duration_secs * 1000000.0
        self._start_us = 0
        self._cur_time_us = 0

    def update(self, e):
        if False:
            i = 10
            return i + 15
        if not isinstance(e, beam_runner_api_pb2.TestStreamPayload.Event):
            return
        if not e.HasField('processing_time_event'):
            return
        if self._start_us == 0:
            self._start_us = e.processing_time_event.advance_duration
        self._cur_time_us += e.processing_time_event.advance_duration

    def is_triggered(self):
        if False:
            return 10
        return self._cur_time_us - self._start_us >= self._max_duration_us