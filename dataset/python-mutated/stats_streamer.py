import threading
import time
from glances.logger import logger

class StatsStreamer:
    """
    Utility class to stream an iterable using a background / daemon Thread

    Use `StatsStreamer.stats` to access the latest streamed results
    """

    def __init__(self, iterable, initial_stream_value=None, sleep_duration=0.1):
        if False:
            while True:
                i = 10
        '\n        iterable: an Iterable instance that needs to be streamed\n        '
        self._iterable = iterable
        self._raw_result = initial_stream_value
        self._thread = threading.Thread(target=self._stream_results, daemon=True)
        self._stopper = threading.Event()
        self.result_lock = threading.Lock()
        self._last_update_time = 0
        self._sleep_duration = sleep_duration
        self._thread.start()

    def stop(self):
        if False:
            return 10
        'Stop the thread.'
        self._stopper.set()

    def stopped(self):
        if False:
            for i in range(10):
                print('nop')
        'Return True is the thread is stopped.'
        return self._stopper.is_set()

    def _stream_results(self):
        if False:
            while True:
                i = 10
        'Grab the stats.\n\n        Infinite loop, should be stopped by calling the stop() method\n        '
        try:
            for res in self._iterable:
                self._pre_update_hook()
                self._raw_result = res
                self._post_update_hook()
                time.sleep(self._sleep_duration)
                if self.stopped():
                    break
        except Exception as e:
            logger.debug('docker plugin - Exception thrown during run ({})'.format(e))
            self.stop()

    def _pre_update_hook(self):
        if False:
            i = 10
            return i + 15
        'Hook that runs before worker thread updates the raw_stats'
        self.result_lock.acquire()

    def _post_update_hook(self):
        if False:
            for i in range(10):
                print('nop')
        'Hook that runs after worker thread updates the raw_stats'
        self._last_update_time = time.time()
        self.result_lock.release()

    @property
    def stats(self):
        if False:
            i = 10
            return i + 15
        'Raw Stats getter.'
        return self._raw_result

    @property
    def last_update_time(self):
        if False:
            for i in range(10):
                print('nop')
        'Raw Stats getter.'
        return self._last_update_time