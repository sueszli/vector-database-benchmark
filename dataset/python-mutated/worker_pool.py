import contextlib
import errno
import logging
import os
import signal
import time
from enum import Enum
from multiprocessing import Process
from typing import Dict, List, NamedTuple, Optional, Type, Union
from uuid import uuid4
from redis import ConnectionPool, Redis
from rq.serializers import DefaultSerializer
from .connections import parse_connection
from .defaults import DEFAULT_LOGGING_DATE_FORMAT, DEFAULT_LOGGING_FORMAT
from .job import Job
from .logutils import setup_loghandlers
from .queue import Queue
from .utils import parse_names
from .worker import BaseWorker, Worker

class WorkerData(NamedTuple):
    name: str
    pid: int
    process: Process

class WorkerPool:

    class Status(Enum):
        IDLE = 1
        STARTED = 2
        STOPPED = 3

    def __init__(self, queues: List[Union[str, Queue]], connection: Redis, num_workers: int=1, worker_class: Type[BaseWorker]=Worker, serializer: Type[DefaultSerializer]=DefaultSerializer, job_class: Type[Job]=Job, *args, **kwargs):
        if False:
            print('Hello World!')
        self.num_workers: int = num_workers
        self._workers: List[Worker] = []
        setup_loghandlers('INFO', DEFAULT_LOGGING_DATE_FORMAT, DEFAULT_LOGGING_FORMAT, name=__name__)
        self.log: logging.Logger = logging.getLogger(__name__)
        self._queue_names: List[str] = parse_names(queues)
        self.connection = connection
        self.name: str = uuid4().hex
        self._burst: bool = True
        self._sleep: int = 0
        self.status: self.Status = self.Status.IDLE
        self.worker_class: Type[BaseWorker] = worker_class
        self.serializer: Type[DefaultSerializer] = serializer
        self.job_class: Type[Job] = job_class
        self.worker_dict: Dict[str, WorkerData] = {}
        (self._connection_class, self._pool_class, self._pool_kwargs) = parse_connection(connection)

    @property
    def queues(self) -> List[Queue]:
        if False:
            i = 10
            return i + 15
        'Returns a list of Queue objects'
        return [Queue(name, connection=self.connection) for name in self._queue_names]

    @property
    def number_of_active_workers(self) -> int:
        if False:
            return 10
        'Returns a list of Queue objects'
        return len(self.worker_dict)

    def _install_signal_handlers(self):
        if False:
            i = 10
            return i + 15
        'Installs signal handlers for handling SIGINT and SIGTERM\n        gracefully.\n        '
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

    def request_stop(self, signum=None, frame=None):
        if False:
            for i in range(10):
                print('nop')
        "Toggle self._stop_requested that's checked on every loop"
        self.log.info('Received SIGINT/SIGTERM, shutting down...')
        self.status = self.Status.STOPPED
        self.stop_workers()

    def all_workers_have_stopped(self) -> bool:
        if False:
            print('Hello World!')
        'Returns True if all workers have stopped.'
        self.reap_workers()
        return self.number_of_active_workers == 0

    def reap_workers(self):
        if False:
            for i in range(10):
                print('nop')
        'Removes dead workers from worker_dict'
        self.log.debug('Reaping dead workers')
        worker_datas = list(self.worker_dict.values())
        for data in worker_datas:
            data.process.join(0.1)
            if data.process.is_alive():
                self.log.debug('Worker %s with pid %d is alive', data.name, data.pid)
            else:
                self.handle_dead_worker(data)
                continue

    def handle_dead_worker(self, worker_data: WorkerData):
        if False:
            for i in range(10):
                print('nop')
        '\n        Handle a dead worker\n        '
        self.log.info('Worker %s with pid %d is dead', worker_data.name, worker_data.pid)
        with contextlib.suppress(KeyError):
            self.worker_dict.pop(worker_data.name)

    def check_workers(self, respawn: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Check whether workers are still alive\n        '
        self.log.debug('Checking worker processes')
        self.reap_workers()
        if respawn and self.status != self.Status.STOPPED:
            delta = self.num_workers - len(self.worker_dict)
            if delta:
                for i in range(delta):
                    self.start_worker(burst=self._burst, _sleep=self._sleep)

    def start_worker(self, count: Optional[int]=None, burst: bool=True, _sleep: float=0, logging_level: str='INFO'):
        if False:
            i = 10
            return i + 15
        '\n        Starts a worker and adds the data to worker_datas.\n        * sleep: waits for X seconds before creating worker, for testing purposes\n        '
        name = uuid4().hex
        process = Process(target=run_worker, args=(name, self._queue_names, self._connection_class, self._pool_class, self._pool_kwargs), kwargs={'_sleep': _sleep, 'burst': burst, 'logging_level': logging_level, 'worker_class': self.worker_class, 'job_class': self.job_class, 'serializer': self.serializer}, name=f'Worker {name} (WorkerPool {self.name})')
        process.start()
        worker_data = WorkerData(name=name, pid=process.pid, process=process)
        self.worker_dict[name] = worker_data
        self.log.debug('Spawned worker: %s with PID %d', name, process.pid)

    def start_workers(self, burst: bool=True, _sleep: float=0, logging_level: str='INFO'):
        if False:
            i = 10
            return i + 15
        '\n        Run the workers\n        * sleep: waits for X seconds before creating worker, only for testing purposes\n        '
        self.log.debug(f'Spawning {self.num_workers} workers')
        for i in range(self.num_workers):
            self.start_worker(i + 1, burst=burst, _sleep=_sleep, logging_level=logging_level)

    def stop_worker(self, worker_data: WorkerData, sig=signal.SIGINT):
        if False:
            while True:
                i = 10
        '\n        Send stop signal to worker and catch "No such process" error if the worker is already dead.\n        '
        try:
            os.kill(worker_data.pid, sig)
            self.log.info('Sent shutdown command to worker with %s', worker_data.pid)
        except OSError as e:
            if e.errno == errno.ESRCH:
                self.log.debug('Horse already dead')
            else:
                raise

    def stop_workers(self):
        if False:
            print('Hello World!')
        'Send SIGINT to all workers'
        self.log.info('Sending stop signal to %s workers', len(self.worker_dict))
        worker_datas = list(self.worker_dict.values())
        for worker_data in worker_datas:
            self.stop_worker(worker_data)

    def start(self, burst: bool=False, logging_level: str='INFO'):
        if False:
            return 10
        self._burst = burst
        respawn = not burst
        setup_loghandlers(logging_level, DEFAULT_LOGGING_DATE_FORMAT, DEFAULT_LOGGING_FORMAT, name=__name__)
        self.log.info(f'Starting worker pool {self.name} with pid %d...', os.getpid())
        self.status = self.Status.IDLE
        self.start_workers(burst=self._burst, logging_level=logging_level)
        self._install_signal_handlers()
        while True:
            if self.status == self.Status.STOPPED:
                if self.all_workers_have_stopped():
                    self.log.info('All workers stopped, exiting...')
                    break
                else:
                    self.log.info('Waiting for workers to shutdown...')
                    time.sleep(1)
                    continue
            else:
                self.check_workers(respawn=respawn)
                if burst and self.number_of_active_workers == 0:
                    self.log.info('All workers stopped, exiting...')
                    break
                time.sleep(1)

def run_worker(worker_name: str, queue_names: List[str], connection_class, connection_pool_class, connection_pool_kwargs: dict, worker_class: Type[BaseWorker]=Worker, serializer: Type[DefaultSerializer]=DefaultSerializer, job_class: Type[Job]=Job, burst: bool=True, logging_level: str='INFO', _sleep: int=0):
    if False:
        i = 10
        return i + 15
    connection = connection_class(connection_pool=ConnectionPool(connection_class=connection_pool_class, **connection_pool_kwargs))
    queues = [Queue(name, connection=connection) for name in queue_names]
    worker = worker_class(queues, name=worker_name, connection=connection, serializer=serializer, job_class=job_class)
    worker.log.info('Starting worker started with PID %s', os.getpid())
    time.sleep(_sleep)
    worker.work(burst=burst, logging_level=logging_level)