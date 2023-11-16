from collections import namedtuple
from queue import Queue
from sacred.observers.base import RunObserver
from sacred.utils import IntervalTimer
import traceback
import logging
logger = logging.getLogger(__name__)
WrappedEvent = namedtuple('WrappedEvent', 'name args kwargs')

class QueueObserver(RunObserver):
    """Wraps any observer and puts processing of events in the background.

    If the covered observer fails to process an event, the queue observer
    will retry until it works. This is useful for observers that rely on
    external services like databases that might become temporarily
    unavailable.
    """

    def __init__(self, covered_observer: RunObserver, interval: float=20.0, retry_interval: float=10.0):
        if False:
            return 10
        'Initialize QueueObserver.\n\n        Parameters\n        ----------\n        covered_observer\n            The real observer that is being wrapped.\n        interval\n            The interval in seconds at which the background thread is woken up to process new events.\n        retry_interval\n            The interval in seconds to wait if an event failed to be processed.\n        '
        self._covered_observer = covered_observer
        self._retry_interval = retry_interval
        self._interval = interval
        self._queue = None
        self._worker = None
        self._stop_worker_event = None
        logger.debug('just testing')

    def queued_event(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self._queue.put(WrappedEvent('queued_event', args, kwargs))

    def started_event(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self._queue = Queue()
        (self._stop_worker_event, self._worker) = IntervalTimer.create(self._run, interval=self._interval)
        self._worker.start()
        return self._covered_observer.started_event(*args, **kwargs)

    def heartbeat_event(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self._queue.put(WrappedEvent('heartbeat_event', args, kwargs))

    def completed_event(self, *args, **kwargs):
        if False:
            return 10
        self._queue.put(WrappedEvent('completed_event', args, kwargs))
        self.join()

    def interrupted_event(self, *args, **kwargs):
        if False:
            return 10
        self._queue.put(WrappedEvent('interrupted_event', args, kwargs))
        self.join()

    def failed_event(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self._queue.put(WrappedEvent('failed_event', args, kwargs))
        self.join()

    def resource_event(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self._queue.put(WrappedEvent('resource_event', args, kwargs))

    def artifact_event(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self._queue.put(WrappedEvent('artifact_event', args, kwargs))

    def log_metrics(self, metrics_by_name, info):
        if False:
            print('Hello World!')
        for (metric_name, metric_values) in metrics_by_name.items():
            self._queue.put(WrappedEvent('log_metrics', [metric_name, metric_values, info], {}))

    def _run(self):
        if False:
            return 10
        'Empty the queue every interval.'
        while not self._queue.empty():
            try:
                event = self._queue.get()
            except IndexError:
                pass
            else:
                try:
                    method = getattr(self._covered_observer, event.name)
                except NameError:
                    self._queue.task_done()
                else:
                    while True:
                        try:
                            method(*event.args, **event.kwargs)
                        except:
                            logger.debug('Error while processing event. Trying again.\n{}'.format(traceback.format_exc()))
                            self._stop_worker_event.wait(self._retry_interval)
                            continue
                        else:
                            self._queue.task_done()
                            break

    def join(self):
        if False:
            for i in range(10):
                print('nop')
        if self._queue is not None:
            self._queue.join()
            self._stop_worker_event.set()
            self._worker.join(timeout=10)

    def __getattr__(self, item):
        if False:
            return 10
        return getattr(self._covered_observer, item)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self._covered_observer == other