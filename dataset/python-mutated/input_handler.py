from abc import ABC
from enum import Enum
from typing import cast, Iterable
from pyflink.common import Row
from pyflink.common.constants import DEFAULT_OUTPUT_TAG
from pyflink.datastream.output_tag import OutputTag
from pyflink.fn_execution.datastream.process.timerservice_impl import InternalTimerServiceImpl

class TimerType(Enum):
    EVENT_TIME = 0
    PROCESSING_TIME = 1

class RunnerInputHandler(ABC):
    """
    Handler which handles normal input data.
    """

    def __init__(self, internal_timer_service: InternalTimerServiceImpl, process_element_func, has_side_output: bool):
        if False:
            i = 10
            return i + 15
        self._internal_timer_service = internal_timer_service
        self._process_element_func = process_element_func
        self._has_side_output = has_side_output

    def process_element(self, value) -> Iterable:
        if False:
            i = 10
            return i + 15
        timestamp = value[0]
        watermark = value[1]
        data = value[2]
        self._advance_watermark(watermark)
        yield from _emit_results(timestamp, watermark, self._process_element_func(data, timestamp), self._has_side_output)

    def _advance_watermark(self, watermark: int) -> None:
        if False:
            print('Hello World!')
        self._internal_timer_service.advance_watermark(watermark)

class TimerHandler(ABC):
    """
    Handler which handles normal input data.
    """

    def __init__(self, internal_timer_service: InternalTimerServiceImpl, on_event_time_func, on_processing_time_func, namespace_coder, has_side_output):
        if False:
            for i in range(10):
                print('nop')
        self._internal_timer_service = internal_timer_service
        self._on_event_time_func = on_event_time_func
        self._on_processing_time_func = on_processing_time_func
        self._namespace_coder = namespace_coder
        self._has_side_output = has_side_output

    def process_timer(self, timer_data) -> Iterable:
        if False:
            return 10
        timer_type = timer_data[0]
        watermark = timer_data[1]
        timestamp = timer_data[2]
        key = timer_data[3]
        serialized_namespace = timer_data[4]
        self._advance_watermark(watermark)
        if self._namespace_coder is not None:
            namespace = self._namespace_coder.decode(serialized_namespace)
        else:
            namespace = None
        if timer_type == TimerType.EVENT_TIME.value:
            yield from _emit_results(timestamp, watermark, self._on_event_time(timestamp, key, namespace), self._has_side_output)
        elif timer_type == TimerType.PROCESSING_TIME.value:
            yield from _emit_results(timestamp, watermark, self._on_processing_time(timestamp, key, namespace), self._has_side_output)
        else:
            raise Exception('Unsupported timer type: %d' % timer_type)

    def _on_event_time(self, timestamp, key, namespace) -> Iterable:
        if False:
            print('Hello World!')
        yield from self._on_event_time_func(timestamp, key, namespace)

    def _on_processing_time(self, timestamp, key, namespace) -> Iterable:
        if False:
            return 10
        yield from self._on_processing_time_func(timestamp, key, namespace)

    def _advance_watermark(self, watermark: int) -> None:
        if False:
            i = 10
            return i + 15
        self._internal_timer_service.advance_watermark(watermark)

def _emit_results(timestamp, watermark, results, has_side_output):
    if False:
        i = 10
        return i + 15
    if results:
        if has_side_output:
            for result in results:
                if isinstance(result, tuple) and isinstance(result[0], OutputTag):
                    yield (cast(OutputTag, result[0]).tag_id, Row(timestamp, watermark, result[1]))
                else:
                    yield (DEFAULT_OUTPUT_TAG, Row(timestamp, watermark, result))
        else:
            for result in results:
                yield Row(timestamp, watermark, result)