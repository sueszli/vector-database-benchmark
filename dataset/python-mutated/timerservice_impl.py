from pyflink.datastream import TimerService
from pyflink.fn_execution.datastream.timerservice import InternalTimerService, N

class TimerServiceImpl(TimerService):

    def __init__(self, j_timer_service):
        if False:
            for i in range(10):
                print('nop')
        self._j_timer_service = j_timer_service

    def current_processing_time(self):
        if False:
            for i in range(10):
                print('nop')
        return self._j_timer_service.currentProcessingTime()

    def current_watermark(self):
        if False:
            i = 10
            return i + 15
        return self._j_timer_service.currentWatermark()

    def register_processing_time_timer(self, timestamp: int):
        if False:
            for i in range(10):
                print('nop')
        self._j_timer_service.registerProcessingTimeTimer(timestamp)

    def register_event_time_timer(self, timestamp: int):
        if False:
            print('Hello World!')
        self._j_timer_service.registerEventTimeTimer(timestamp)

    def delete_processing_time_timer(self, timestamp: int):
        if False:
            while True:
                i = 10
        self._j_timer_service.deleteProcessingTimeTimer(timestamp)

    def delete_event_time_timer(self, timestamp: int):
        if False:
            print('Hello World!')
        self._j_timer_service.deleteEventTimeTimer(timestamp)

class InternalTimerServiceImpl(InternalTimerService[N]):

    def __init__(self, j_timer_service, window_converter):
        if False:
            return 10
        self._j_timer_service = j_timer_service
        self._window_converter = window_converter

    def current_processing_time(self):
        if False:
            return 10
        return self._j_timer_service.currentProcessingTime()

    def current_watermark(self):
        if False:
            return 10
        return self._j_timer_service.currentWatermark()

    def register_processing_time_timer(self, namespace: N, timestamp: int):
        if False:
            while True:
                i = 10
        window = self._window_converter.to_external(namespace)
        self._j_timer_service.registerProcessingTimeTimer(window, timestamp)

    def register_event_time_timer(self, namespace: N, timestamp: int):
        if False:
            print('Hello World!')
        window = self._window_converter.to_external(namespace)
        self._j_timer_service.registerEventTimeTimer(window, timestamp)

    def delete_event_time_timer(self, namespace: N, timestamp: int):
        if False:
            for i in range(10):
                print('nop')
        window = self._window_converter.to_external(namespace)
        self._j_timer_service.deleteEventTimeTimer(window, timestamp)

    def delete_processing_time_timer(self, namespace: N, timestamp: int):
        if False:
            for i in range(10):
                print('nop')
        window = self._window_converter.to_external(namespace)
        self._j_timer_service.deleteProcessingTimeTimer(window, timestamp)