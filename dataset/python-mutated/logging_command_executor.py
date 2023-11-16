from __future__ import annotations
import shlex
import subprocess
from airflow.exceptions import AirflowException
from airflow.utils.log.logging_mixin import LoggingMixin

class CommandExecutionError(Exception):
    """Raise in case of error during command execution"""

class LoggingCommandExecutor(LoggingMixin):

    def execute_cmd(self, cmd, silent=False, cwd=None, env=None):
        if False:
            while True:
                i = 10
        if silent:
            self.log.info("Executing in silent mode: '%s'", ' '.join((shlex.quote(c) for c in cmd)))
            return subprocess.call(args=cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env, cwd=cwd)
        else:
            self.log.info("Executing: '%s'", ' '.join((shlex.quote(c) for c in cmd)))
            with subprocess.Popen(args=cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, cwd=cwd, env=env) as process:
                (output, err) = process.communicate()
                retcode = process.poll()
                self.log.info('Stdout: %s', output)
                self.log.info('Stderr: %s', err)
                if retcode:
                    self.log.error('Error when executing %s', ' '.join((shlex.quote(c) for c in cmd)))
                return retcode

    def check_output(self, cmd):
        if False:
            print('Hello World!')
        self.log.info("Executing for output: '%s'", ' '.join((shlex.quote(c) for c in cmd)))
        with subprocess.Popen(args=cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            (output, err) = process.communicate()
            retcode = process.poll()
            if retcode:
                self.log.error("Error when executing '%s'", ' '.join((shlex.quote(c) for c in cmd)))
                self.log.info('Stdout: %s', output)
                self.log.info('Stderr: %s', err)
                raise AirflowException(f"Retcode {retcode} on {' '.join(cmd)} with stdout: {output}, stderr: {err}")
            return output

class CommandExecutor(LoggingCommandExecutor):
    """
    Due to 'LoggingCommandExecutor' class just returns the status code of command execution
    ('execute_cmd' method) and continues to perform code with possible errors, separate
    inherited 'CommandExecutor' class was created to use it if you need to break code performing
    and immediately raise an exception in case of error during command execution,
    so re-written 'execute_cmd' method will raise 'CommandExecutionError' exception.
    """

    def execute_cmd(self, cmd, silent=False, cwd=None, env=None):
        if False:
            return 10
        if silent:
            self.log.info("Executing in silent mode: '%s'", ' '.join((shlex.quote(c) for c in cmd)))
            return subprocess.call(args=cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env, cwd=cwd)
        else:
            self.log.info("Executing: '%s'", ' '.join((shlex.quote(c) for c in cmd)))
            with subprocess.Popen(args=cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, cwd=cwd, env=env) as process:
                (output, err) = process.communicate()
                retcode = process.poll()
                if retcode:
                    raise CommandExecutionError(f"Error when executing '{' '.join(cmd)}' with stdout: {output}, stderr: {err}")
                self.log.info('Stdout: %s', output)
                self.log.info('Stderr: %s', err)

def get_executor() -> LoggingCommandExecutor:
    if False:
        return 10
    return LoggingCommandExecutor()