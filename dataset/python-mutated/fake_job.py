"""
Base class for dummy jobs.
"""
from concurrent import futures
from qiskit.providers import JobV1
from qiskit.providers.jobstatus import JobStatus

class FakeJob(JobV1):
    """Fake simulator job"""
    _executor = futures.ThreadPoolExecutor()

    def __init__(self, backend, job_id, fn):
        if False:
            print('Hello World!')
        super().__init__(backend, job_id)
        self._backend = backend
        self._job_id = job_id
        self._future = None
        self._future_callback = fn

    def submit(self):
        if False:
            return 10
        self._future = self._executor.submit(self._future_callback)

    def result(self, timeout=None):
        if False:
            i = 10
            return i + 15
        return self._future.result(timeout=timeout)

    def cancel(self):
        if False:
            print('Hello World!')
        return self._future.cancel()

    def status(self):
        if False:
            print('Hello World!')
        if self._running:
            _status = JobStatus.RUNNING
        elif not self._done:
            _status = JobStatus.QUEUED
        elif self._cancelled:
            _status = JobStatus.CANCELLED
        elif self._done:
            _status = JobStatus.DONE
        elif self._error:
            _status = JobStatus.ERROR
        else:
            raise Exception(f'Unexpected state of {self.__class__.__name__}')
        _status_msg = None
        return {'status': _status, 'status_msg': _status_msg}

    def job_id(self):
        if False:
            for i in range(10):
                print('nop')
        return self._job_id

    def backend(self):
        if False:
            for i in range(10):
                print('nop')
        return self._backend

    @property
    def _cancelled(self):
        if False:
            print('Hello World!')
        return self._future.cancelled()

    @property
    def _done(self):
        if False:
            i = 10
            return i + 15
        return self._future.done()

    @property
    def _running(self):
        if False:
            while True:
                i = 10
        return self._future.running()

    @property
    def _error(self):
        if False:
            for i in range(10):
                print('nop')
        return self._future.exception(timeout=0)