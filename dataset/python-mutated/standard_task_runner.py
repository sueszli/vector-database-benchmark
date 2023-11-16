"""Standard task runner."""
from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING
import psutil
from setproctitle import setproctitle
from airflow.models.taskinstance import TaskReturnCode
from airflow.settings import CAN_FORK
from airflow.task.task_runner.base_task_runner import BaseTaskRunner
from airflow.utils.dag_parsing_context import _airflow_parsing_context_manager
from airflow.utils.process_utils import reap_process_group, set_new_process_group
if TYPE_CHECKING:
    from airflow.jobs.local_task_job_runner import LocalTaskJobRunner

class StandardTaskRunner(BaseTaskRunner):
    """Standard runner for all tasks."""

    def __init__(self, job_runner: LocalTaskJobRunner):
        if False:
            return 10
        super().__init__(job_runner=job_runner)
        self._rc = None
        self.dag = self._task_instance.task.dag

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        if CAN_FORK and (not self.run_as_user):
            self.process = self._start_by_fork()
        else:
            self.process = self._start_by_exec()

    def _start_by_exec(self) -> psutil.Process:
        if False:
            print('Hello World!')
        subprocess = self.run_command()
        self.process = psutil.Process(subprocess.pid)
        return self.process

    def _start_by_fork(self):
        if False:
            i = 10
            return i + 15
        pid = os.fork()
        if pid:
            self.log.info('Started process %d to run task', pid)
            return psutil.Process(pid)
        else:
            set_new_process_group()
            import signal
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            from airflow import settings
            from airflow.cli.cli_parser import get_parser
            from airflow.sentry import Sentry
            settings.engine.pool.dispose()
            settings.engine.dispose()
            parser = get_parser()
            args = parser.parse_args(self._command[1:])
            job_id = getattr(args, 'job_id', self._task_instance.job_id)
            self.log.info('Running: %s', self._command)
            self.log.info('Job %s: Subtask %s', job_id, self._task_instance.task_id)
            proc_title = 'airflow task runner: {0.dag_id} {0.task_id} {0.execution_date_or_run_id}'
            if job_id is not None:
                proc_title += ' {0.job_id}'
            setproctitle(proc_title.format(args))
            return_code = 0
            try:
                with _airflow_parsing_context_manager(dag_id=self._task_instance.dag_id, task_id=self._task_instance.task_id):
                    ret = args.func(args, dag=self.dag)
                    return_code = 0
                    if isinstance(ret, TaskReturnCode):
                        return_code = ret.value
            except Exception as exc:
                return_code = 1
                self.log.error('Failed to execute job %s for task %s (%s; %r)', job_id, self._task_instance.task_id, exc, os.getpid())
            except SystemExit as sys_ex:
                return_code = sys_ex.code
            except BaseException:
                return_code = 2
            finally:
                try:
                    Sentry.flush()
                    logging.shutdown()
                except BaseException:
                    pass
            os._exit(return_code)

    def return_code(self, timeout: float=0) -> int | None:
        if False:
            i = 10
            return i + 15
        if self._rc is not None or not self.process:
            return self._rc
        try:
            self._rc = self.process.wait(timeout=timeout)
            self.process = None
        except psutil.TimeoutExpired:
            pass
        return self._rc

    def terminate(self):
        if False:
            for i in range(10):
                print('nop')
        if self.process is None:
            return
        _ = self.return_code(timeout=0)
        if self.process and self.process.is_running():
            rcs = reap_process_group(self.process.pid, self.log)
            self._rc = rcs.get(self.process.pid)
        self.process = None
        if self._rc is None:
            self._rc = -9
        if self._rc == -9:
            self.log.error('Job %s was killed before it finished (likely due to running out of memory)', self._task_instance.job_id)

    def get_process_pid(self) -> int:
        if False:
            print('Hello World!')
        if self.process is None:
            raise RuntimeError('Process is not started yet')
        return self.process.pid