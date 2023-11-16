"""Base task runner."""
from __future__ import annotations
import os
import subprocess
import threading
from airflow.utils.dag_parsing_context import _airflow_parsing_context_manager
from airflow.utils.platform import IS_WINDOWS
if not IS_WINDOWS:
    from pwd import getpwnam
from typing import TYPE_CHECKING
from airflow.configuration import conf
from airflow.exceptions import AirflowConfigException
from airflow.utils.configuration import tmp_configuration_copy
from airflow.utils.log.logging_mixin import LoggingMixin
from airflow.utils.net import get_hostname
from airflow.utils.platform import getuser
if TYPE_CHECKING:
    from airflow.jobs.local_task_job_runner import LocalTaskJobRunner
PYTHONPATH_VAR = 'PYTHONPATH'

class BaseTaskRunner(LoggingMixin):
    """
    Runs Airflow task instances via CLI.

    Invoke the `airflow tasks run` command with raw mode enabled in a subprocess.

    :param job_runner: The LocalTaskJobRunner associated with the task runner
    """

    def __init__(self, job_runner: LocalTaskJobRunner):
        if False:
            print('Hello World!')
        self.job_runner = job_runner
        super().__init__(job_runner.task_instance)
        self._task_instance = job_runner.task_instance
        popen_prepend = []
        if self._task_instance.run_as_user:
            self.run_as_user: str | None = self._task_instance.run_as_user
        else:
            try:
                self.run_as_user = conf.get('core', 'default_impersonation')
            except AirflowConfigException:
                self.run_as_user = None
        self.log.debug('Planning to run as the %s user', self.run_as_user)
        if self.run_as_user and self.run_as_user != getuser():
            cfg_path = tmp_configuration_copy(chmod=384, include_env=True, include_cmds=True)
            subprocess.check_call(['sudo', 'chown', self.run_as_user, cfg_path], close_fds=True)
            pythonpath_value = os.environ.get(PYTHONPATH_VAR, '')
            popen_prepend = ['sudo', '-E', '-H', '-u', self.run_as_user]
            if pythonpath_value:
                popen_prepend.append(f'{PYTHONPATH_VAR}={pythonpath_value}')
        else:
            cfg_path = tmp_configuration_copy(chmod=384, include_env=False, include_cmds=False)
        self._cfg_path = cfg_path
        self._command = popen_prepend + self._task_instance.command_as_list(raw=True, pickle_id=self.job_runner.pickle_id, mark_success=self.job_runner.mark_success, job_id=self.job_runner.job.id, pool=self.job_runner.pool, cfg_path=cfg_path)
        self.process = None

    def _read_task_logs(self, stream):
        if False:
            return 10
        while True:
            line = stream.readline()
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            if not line:
                break
            self.log.info('Job %s: Subtask %s %s', self._task_instance.job_id, self._task_instance.task_id, line.rstrip('\n'))

    def run_command(self, run_with=None) -> subprocess.Popen:
        if False:
            return 10
        "\n        Run the task command.\n\n        :param run_with: list of tokens to run the task command with e.g. ``['bash', '-c']``\n        :return: the process that was run\n        "
        run_with = run_with or []
        full_cmd = run_with + self._command
        self.log.info('Running on host: %s', get_hostname())
        self.log.info('Running: %s', full_cmd)
        with _airflow_parsing_context_manager(dag_id=self._task_instance.dag_id, task_id=self._task_instance.task_id):
            if IS_WINDOWS:
                proc = subprocess.Popen(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, close_fds=True, env=os.environ.copy())
            else:
                proc = subprocess.Popen(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, close_fds=True, env=os.environ.copy(), preexec_fn=os.setsid)
        log_reader = threading.Thread(target=self._read_task_logs, args=(proc.stdout,))
        log_reader.daemon = True
        log_reader.start()
        return proc

    def start(self):
        if False:
            while True:
                i = 10
        'Start running the task instance in a subprocess.'
        raise NotImplementedError()

    def return_code(self, timeout: float=0.0) -> int | None:
        if False:
            print('Hello World!')
        '\n        Extract the return code.\n\n        :return: The return code associated with running the task instance or\n            None if the task is not yet done.\n        '
        raise NotImplementedError()

    def terminate(self) -> None:
        if False:
            while True:
                i = 10
        'Force kill the running task instance.'
        raise NotImplementedError()

    def on_finish(self) -> None:
        if False:
            while True:
                i = 10
        'Execute when this is done running.'
        if self._cfg_path and os.path.isfile(self._cfg_path):
            if self.run_as_user:
                subprocess.call(['sudo', 'rm', self._cfg_path], close_fds=True)
            else:
                os.remove(self._cfg_path)

    def get_process_pid(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Get the process pid.'
        if hasattr(self, 'process') and self.process is not None and hasattr(self.process, 'pid'):
            return self.process.pid
        raise NotImplementedError()