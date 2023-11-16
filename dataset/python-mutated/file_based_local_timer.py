import io
import json
import logging
import os
import select
import signal
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Set, Tuple
from torch.distributed.elastic.timer.api import TimerClient, TimerRequest
__all__ = ['FileTimerClient', 'FileTimerRequest', 'FileTimerServer']
log = logging.getLogger(__name__)

class FileTimerRequest(TimerRequest):
    """
    Data object representing a countdown timer acquisition and release
    that is used between the ``FileTimerClient`` and ``FileTimerServer``.
    A negative ``expiration_time`` should be interpreted as a "release"
    request.
    ``signal`` is the signal to reap the worker process from the server
    process.
    """
    __slots__ = ['version', 'worker_pid', 'scope_id', 'expiration_time', 'signal']

    def __init__(self, worker_pid: int, scope_id: str, expiration_time: float, signal: int=0) -> None:
        if False:
            return 10
        self.version = 1
        self.worker_pid = worker_pid
        self.scope_id = scope_id
        self.expiration_time = expiration_time
        self.signal = signal

    def __eq__(self, other) -> bool:
        if False:
            while True:
                i = 10
        if isinstance(other, FileTimerRequest):
            return self.version == other.version and self.worker_pid == other.worker_pid and (self.scope_id == other.scope_id) and (self.expiration_time == other.expiration_time) and (self.signal == other.signal)
        return False

    def to_json(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return json.dumps({'version': self.version, 'pid': self.worker_pid, 'scope_id': self.scope_id, 'expiration_time': self.expiration_time, 'signal': self.signal})

class FileTimerClient(TimerClient):
    """
    Client side of ``FileTimerServer``. This client is meant to be used
    on the same host that the ``FileTimerServer`` is running on and uses
    pid to uniquely identify a worker.
    This client uses a named_pipe to send timer requests to the
    ``FileTimerServer``. This client is a producer while the
    ``FileTimerServer`` is a consumer. Multiple clients can work with
    the same ``FileTimerServer``.

    Args:

        file_path: str, the path of a FIFO special file. ``FileTimerServer``
                        must have created it by calling os.mkfifo().

        signal: signal, the signal to use to kill the process. Using a
                        negative or zero signal will not kill the process.
    """

    def __init__(self, file_path: str, signal=signal.SIGKILL if sys.platform != 'win32' else signal.CTRL_C_EVENT) -> None:
        if False:
            return 10
        super().__init__()
        self._file_path = file_path
        self.signal = signal

    def _open_non_blocking(self) -> Optional[io.TextIOWrapper]:
        if False:
            print('Hello World!')
        try:
            fd = os.open(self._file_path, os.O_WRONLY | os.O_NONBLOCK)
            return os.fdopen(fd, 'wt')
        except Exception:
            return None

    def _send_request(self, request: FileTimerRequest) -> None:
        if False:
            i = 10
            return i + 15
        file = self._open_non_blocking()
        if file is None:
            raise BrokenPipeError('Could not send the FileTimerRequest because FileTimerServer is not available.')
        with file:
            json_request = request.to_json()
            if len(json_request) > select.PIPE_BUF:
                raise RuntimeError(f'FileTimerRequest larger than {select.PIPE_BUF} bytes is not supported: {json_request}')
            file.write(json_request + '\n')

    def acquire(self, scope_id: str, expiration_time: float) -> None:
        if False:
            print('Hello World!')
        self._send_request(request=FileTimerRequest(worker_pid=os.getpid(), scope_id=scope_id, expiration_time=expiration_time, signal=self.signal))

    def release(self, scope_id: str) -> None:
        if False:
            print('Hello World!')
        self._send_request(request=FileTimerRequest(worker_pid=os.getpid(), scope_id=scope_id, expiration_time=-1, signal=0))

class FileTimerServer:
    """
    Server that works with ``FileTimerClient``. Clients are expected to be
    running on the same host as the process that is running this server.
    Each host in the job is expected to start its own timer server locally
    and each server instance manages timers for local workers (running on
    processes on the same host).

    Args:

        file_path: str, the path of a FIFO special file to be created.

        max_interval: float, max interval in seconds for each watchdog loop.

        daemon: bool, running the watchdog thread in daemon mode or not.
                      A daemon thread will not block a process to stop.
        log_event: Callable[[Dict[str, str]], None], an optional callback for
                logging the events in JSON format.
    """

    def __init__(self, file_path: str, max_interval: float=10, daemon: bool=True, log_event: Optional[Callable[[str, Optional[FileTimerRequest]], None]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._file_path = file_path
        self._max_interval = max_interval
        self._daemon = daemon
        self._timers: Dict[Tuple[int, str], FileTimerRequest] = {}
        self._stop_signaled = False
        self._watchdog_thread: Optional[threading.Thread] = None
        if os.path.exists(self._file_path):
            os.remove(self._file_path)
        os.mkfifo(self._file_path)
        self._request_count = 0
        self._run_once = False
        self._log_event = log_event if log_event is not None else lambda name, request: None

    def start(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        log.info('Starting %s... max_interval=%s, daemon=%s', type(self).__name__, self._max_interval, self._daemon)
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=self._daemon)
        log.info('Starting watchdog thread...')
        self._watchdog_thread.start()
        self._log_event('watchdog started', None)

    def stop(self) -> None:
        if False:
            i = 10
            return i + 15
        log.info('Stopping %s', type(self).__name__)
        self._stop_signaled = True
        if self._watchdog_thread:
            log.info('Stopping watchdog thread...')
            self._watchdog_thread.join(self._max_interval)
            self._watchdog_thread = None
        else:
            log.info('No watchdog thread running, doing nothing')
        if os.path.exists(self._file_path):
            os.remove(self._file_path)
        self._log_event('watchdog stopped', None)

    def run_once(self) -> None:
        if False:
            return 10
        self._run_once = True
        if self._watchdog_thread:
            log.info('Stopping watchdog thread...')
            self._watchdog_thread.join()
            self._watchdog_thread = None
        else:
            log.info('No watchdog thread running, doing nothing')
        if os.path.exists(self._file_path):
            os.remove(self._file_path)

    def _watchdog_loop(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with open(self._file_path) as fd:
            while not self._stop_signaled:
                try:
                    run_once = self._run_once
                    self._run_watchdog(fd)
                    if run_once:
                        break
                except Exception as e:
                    log.error('Error running watchdog', exc_info=e)

    def _run_watchdog(self, fd: io.TextIOWrapper) -> None:
        if False:
            while True:
                i = 10
        timer_requests = self._get_requests(fd, self._max_interval)
        self.register_timers(timer_requests)
        now = time.time()
        reaped_worker_pids = set()
        for (worker_pid, expired_timers) in self.get_expired_timers(now).items():
            log.info('Reaping worker_pid=[%s]. Expired timers: %s', worker_pid, self._get_scopes(expired_timers))
            reaped_worker_pids.add(worker_pid)
            expired_timers.sort(key=lambda timer: timer.expiration_time)
            signal = 0
            expired_timer = None
            for timer in expired_timers:
                self._log_event('timer expired', timer)
                if timer.signal > 0:
                    signal = timer.signal
                    expired_timer = timer
                    break
            if signal <= 0:
                log.info('No signal specified with worker=[%s]. Do not reap it.', worker_pid)
                continue
            if self._reap_worker(worker_pid, signal):
                log.info('Successfully reaped worker=[%s] with signal=%s', worker_pid, signal)
                self._log_event('kill worker process', expired_timer)
            else:
                log.error('Error reaping worker=[%s]. Will retry on next watchdog.', worker_pid)
        self.clear_timers(reaped_worker_pids)

    def _get_scopes(self, timer_requests: List[FileTimerRequest]) -> List[str]:
        if False:
            i = 10
            return i + 15
        return [r.scope_id for r in timer_requests]

    def _get_requests(self, fd: io.TextIOWrapper, max_interval: float) -> List[FileTimerRequest]:
        if False:
            while True:
                i = 10
        start = time.time()
        requests = []
        while not self._stop_signaled or self._run_once:
            json_request = fd.readline()
            if len(json_request) == 0:
                if self._run_once:
                    break
                time.sleep(min(max_interval, 1))
            else:
                request = json.loads(json_request)
                pid = request['pid']
                scope_id = request['scope_id']
                expiration_time = request['expiration_time']
                signal = request['signal']
                requests.append(FileTimerRequest(worker_pid=pid, scope_id=scope_id, expiration_time=expiration_time, signal=signal))
            now = time.time()
            if now - start > max_interval:
                break
        return requests

    def register_timers(self, timer_requests: List[FileTimerRequest]) -> None:
        if False:
            i = 10
            return i + 15
        for request in timer_requests:
            pid = request.worker_pid
            scope_id = request.scope_id
            expiration_time = request.expiration_time
            self._request_count += 1
            key = (pid, scope_id)
            if expiration_time < 0:
                if key in self._timers:
                    del self._timers[key]
            else:
                self._timers[key] = request

    def clear_timers(self, worker_pids: Set[int]) -> None:
        if False:
            for i in range(10):
                print('nop')
        for (pid, scope_id) in list(self._timers.keys()):
            if pid in worker_pids:
                del self._timers[pid, scope_id]

    def get_expired_timers(self, deadline: float) -> Dict[int, List[FileTimerRequest]]:
        if False:
            while True:
                i = 10
        expired_timers: Dict[int, List[FileTimerRequest]] = {}
        for request in self._timers.values():
            if request.expiration_time <= deadline:
                expired_scopes = expired_timers.setdefault(request.worker_pid, [])
                expired_scopes.append(request)
        return expired_timers

    def _reap_worker(self, worker_pid: int, signal: int) -> bool:
        if False:
            print('Hello World!')
        try:
            os.kill(worker_pid, signal)
            return True
        except ProcessLookupError:
            log.info('Process with pid=%s does not exist. Skipping', worker_pid)
            return True
        except Exception as e:
            log.error('Error terminating pid=%s', worker_pid, exc_info=e)
        return False