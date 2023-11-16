"""
This module provides a python-land multithreaded mechanism for executing work.

Basic usage is as follows:
   coordinator = parallel_workers.init_workers(
      my_worker_fun,
      worker_name="train"
   )
   ...
   coordinator.start()

First argument is the function to run in a loop on potentially multiple threads.
It has the call signature
    worker_fun(worker_id)

Argument 'worker_name' is used to distinguish different workers,
such as workers processing train data or workers processing test data.

Optionally, one can define an "init function" that is called once before
threads start, and has call signature:
   my_init_fun(worker_coordinator, global_coordinator)

Note that for data_parallel_models, init_workers will be called
for each GPU. Note that the 'coordinator' returned by the function is same
each time.
"""
import logging
import threading
import atexit
import time
import collections
import traceback
from abc import ABCMeta, abstractmethod
log = logging.getLogger('parallel_workers')
log.setLevel(logging.INFO)
LOG_INT_SECS = 60

def init_workers(worker_fun, num_worker_threads=2, worker_name='train', init_fun=None, external_loggers=None, shutdown_fun=None):
    if False:
        for i in range(10):
            print('nop')
    global global_coordinator
    metrics = Metrics(external_loggers)
    worker_ids = [global_coordinator.get_new_worker_id() for i in range(num_worker_threads)]
    coordinator = WorkerCoordinator(worker_name, worker_ids, init_fun, shutdown_fun=shutdown_fun)
    workers = [threading.Thread(target=run_worker, name='parallel_workers worker id {}'.format(worker_id), args=[coordinator, Worker(coordinator, worker_id, worker_fun, metrics)]) for worker_id in worker_ids]
    coordinator._workers = workers
    global_coordinator.add(coordinator)
    return global_coordinator

class Metrics:

    def __init__(self, external_loggers):
        if False:
            return 10
        self._metrics = collections.defaultdict(lambda : 0)
        self._external_loggers = external_loggers

    def reset_metrics(self):
        if False:
            return 10
        self._metrics = collections.defaultdict(lambda : 0)

    def log_metrics(self):
        if False:
            return 10
        if not self._external_loggers:
            return
        for logger in self._external_loggers:
            try:
                logger.log(self._metrics)
            except Exception as e:
                print('Failed to call ExternalLogger: {}'.format(e))

    def put_metric(self, key, value, count=True):
        if False:
            return 10
        self._metrics[key] += value
        if count:
            count_key = '{}_count'.format(key)
            self._metrics[count_key] += 1

class State:
    __metaclass__ = ABCMeta

    @abstractmethod
    def start(self):
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def cleanup(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class WorkerCoordinator:

    def __init__(self, worker_name, worker_ids, init_fun, state=None, shutdown_fun=None):
        if False:
            return 10
        self._active = True
        self._started = False
        self._workers = []
        self._worker_name = worker_name
        self._worker_ids = worker_ids
        self._init_fun = init_fun
        self._state = state
        self._shutdown_fun = shutdown_fun

    def is_active(self):
        if False:
            for i in range(10):
                print('nop')
        return self._active

    def init(self, global_coordinator):
        if False:
            i = 10
            return i + 15
        if self._init_fun and (not self._started):
            data_coordinator = self
            self._init_fun(data_coordinator, global_coordinator)

    def _start(self):
        if False:
            i = 10
            return i + 15
        if self._started:
            return
        self._active = True
        self._started = True
        if self._state:
            self._state.start()
        for w in self._workers:
            w.daemon = True
            w.start()

    def _stop(self, reason=None):
        if False:
            for i in range(10):
                print('nop')
        self._active = False
        if reason is not None:
            log.error('Data input failed due to an error: {}'.format(reason))
        if self._shutdown_fun and self._started:
            self._shutdown_fun()
        if self._state:
            self._state.stop()
        self._started = False

    def _wait_finish(self, cleanup=None):
        if False:
            for i in range(10):
                print('nop')
        print('Wait for workers to die: {}'.format(self._worker_name))
        for w in self._workers:
            if w != threading.current_thread():
                w.join(5.0)
        success = True
        for w in self._workers:
            if w.is_alive():
                print('Worker {} failed to close while waiting'.format(w))
                success = False
        if success and self._state:
            self._state.cleanup()
        print('All workers terminated: {}'.format(success))
        return success

    def get_worker_ids(self):
        if False:
            i = 10
            return i + 15
        return self._worker_ids

class GlobalWorkerCoordinator:

    def __init__(self):
        if False:
            print('Hello World!')
        self._coordinators = []
        self._fetcher_id_seq = 0
        self._worker_ids = []
        self.register_shutdown_handler()

    def add(self, coordinator):
        if False:
            while True:
                i = 10
        self._coordinators.append(coordinator)

    def get_new_worker_id(self):
        if False:
            return 10
        worker_id = self._fetcher_id_seq
        self._worker_ids.append(worker_id)
        self._fetcher_id_seq += 1
        return worker_id

    def get_worker_ids(self):
        if False:
            return 10
        return self._worker_ids

    def start(self):
        if False:
            print('Hello World!')
        for c in self._coordinators:
            c.init(self)
        for c in self._coordinators:
            c._start()

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        all_success = True
        for c in self._coordinators:
            c._stop()
        for c in self._coordinators:
            success = c._wait_finish()
            all_success = all_success and success
        self._coordinators = []
        return all_success

    def stop_coordinator(self, worker_name):
        if False:
            i = 10
            return i + 15
        '\n        Stop a specific coordinator\n        '
        for c in self._coordinators:
            if c._worker_name == worker_name:
                c._stop()
                c._wait_finish()
        self._coordinators = [c for c in self._coordinators if c._worker_name != worker_name]

    def register_shutdown_handler(self):
        if False:
            i = 10
            return i + 15

        def cleanup():
            if False:
                i = 10
                return i + 15
            self.stop()
        atexit.register(cleanup)

class Worker:

    def __init__(self, coordinator, worker_id, worker_fun=None, metrics=None):
        if False:
            i = 10
            return i + 15
        self._coordinator = coordinator
        self._worker_id = worker_id
        self._worker_fun = worker_fun
        self._metrics = metrics

    def start(self):
        if False:
            return 10
        self._start_time = time.time()

    def run(self):
        if False:
            print('Hello World!')
        self._worker_fun(self._worker_id)

    def handle_exception(self, e):
        if False:
            for i in range(10):
                print('nop')
        traceback.print_exc()
        logging.exception('Exception in worker', e)
        self._coordinator._stop('Exception in worker {}: {}'.format(self._worker_id, e))

    def finish(self):
        if False:
            for i in range(10):
                print('nop')
        self._metrics.put_metric('worker_time', time.time() - self._start_time)
        self._metrics.log_metrics()
global_coordinator = GlobalWorkerCoordinator()

def run_worker(coordinator, worker):
    if False:
        while True:
            i = 10
    while coordinator.is_active():
        worker.start()
        try:
            worker.run()
        except Exception as e:
            worker.handle_exception(e)
        finally:
            worker.finish()