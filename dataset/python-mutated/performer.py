from threading import Thread
import sys
from typing import Callable, Tuple, Union
from hscommon.jobprogress.job import Job, JobInProgressError, JobCancelled

class ThreadedJobPerformer:
    """Run threaded jobs and track progress.

    To run a threaded job, first create a job with _create_job(), then call _run_threaded(), with
    your work function as a parameter.

    Example:

    j = self._create_job()
    self._run_threaded(self.some_work_func, (arg1, arg2, j))
    """
    _job_running = False
    last_error = None

    def create_job(self) -> Job:
        if False:
            i = 10
            return i + 15
        if self._job_running:
            raise JobInProgressError()
        self.last_progress: Union[int, None] = -1
        self.last_desc = ''
        self.job_cancelled = False
        return Job(1, self._update_progress)

    def _async_run(self, *args) -> None:
        if False:
            while True:
                i = 10
        target = args[0]
        args = tuple(args[1:])
        self._job_running = True
        self.last_error = None
        try:
            target(*args)
        except JobCancelled:
            pass
        except Exception as e:
            self.last_error = e
            self.last_traceback = sys.exc_info()[2]
        finally:
            self._job_running = False
            self.last_progress = None

    def reraise_if_error(self) -> None:
        if False:
            return 10
        'Reraises the error that happened in the thread if any.\n\n        Call this after the caller of run_threaded detected that self._job_running returned to False\n        '
        if self.last_error is not None:
            raise self.last_error.with_traceback(self.last_traceback)

    def _update_progress(self, newprogress: int, newdesc: str='') -> bool:
        if False:
            while True:
                i = 10
        self.last_progress = newprogress
        if newdesc:
            self.last_desc = newdesc
        return not self.job_cancelled

    def run_threaded(self, target: Callable, args: Tuple=()) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._job_running:
            raise JobInProgressError()
        args = (target,) + args
        Thread(target=self._async_run, args=args).start()