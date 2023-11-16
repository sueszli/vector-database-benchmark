from __future__ import annotations
import shlex
import subprocess
from typing import Any
import psutil

def run(cmd: list | str, split: bool=True, print_output: bool=False, **kwargs: Any) -> subprocess.CompletedProcess:
    if False:
        return 10
    'Run a shell command.\n\n    Args:\n        cmd: A command string, or a command followed by program\n            arguments that will be submitted to Popen to run.\n\n        split: Flag that splits command to provide as multiple *args\n            to Popen. Default is True.\n\n        print_output: If True will print previously captured stdout.\n            Default is False.\n\n        **kwargs: Extra options to pass to subprocess.\n\n    Example:\n    ::\n        "ls"\n        "ls -la"\n        "chmod 754 local/file"\n\n    Returns:\n        Result with attributes args, returncode, stdout and stderr.\n\n    '
    if isinstance(cmd, str) and split:
        cmd = shlex.split(cmd)
    result = subprocess.run(cmd, input='', capture_output=True, **kwargs)
    result.stdout = result.stdout.decode('utf-8')
    result.stderr = result.stderr.decode('utf-8')
    if print_output:
        print(result.stdout)
    return result

def check_run(cmd: list | str, print_output: bool=False) -> None:
    if False:
        return 10
    '\n    Run cmd using subprocess.check_call (throws error if non-zero value\n    returned)\n\n    Args:\n        cmd: command to be run\n        print_output: whether to print output\n    '
    if isinstance(cmd, str):
        split_cmd = shlex.split(cmd)
    else:
        split_cmd = cmd
    if print_output:
        subprocess.check_call(split_cmd)
    else:
        subprocess.check_call(split_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

class ChildTerminatingPopen(subprocess.Popen):
    """Extend subprocess.Popen class to automatically kill child processes
    when terminated.

    Note:
        On GNU/Linux child processes are not killed automatically if the parent
        dies (so-called orphan processes)
    """

    def __init__(self, cmd: list[str], **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Initializer pipes stderr and stdout.\n\n        Args:\n            cmd: command to be run.\n            **kwargs: keyword arguments such as env and cwd\n\n        '
        super().__init__(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)

    def terminate(self) -> None:
        if False:
            i = 10
            return i + 15
        'Terminate process and children.'
        try:
            proc = psutil.Process(self.pid)
            procs = [proc] + proc.children(recursive=True)
        except psutil.NoSuchProcess:
            pass
        else:
            for proc in reversed(procs):
                try:
                    proc.terminate()
                except psutil.NoSuchProcess:
                    pass
            alive = psutil.wait_procs(procs, timeout=3)[1]
            for proc in alive:
                proc.kill()