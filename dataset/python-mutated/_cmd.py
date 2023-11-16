import os
import subprocess
import tempfile
from typing import List, Mapping, Optional, Callable, Any
from .errors import create_command_error
OnOutput = Callable[[str], Any]

class CommandResult:

    def __init__(self, stdout: str, stderr: str, code: int) -> None:
        if False:
            while True:
                i = 10
        self.stdout = stdout
        self.stderr = stderr
        self.code = code

    def __repr__(self):
        if False:
            return 10
        return f'CommandResult(stdout={self.stdout!r}, stderr={self.stderr!r}, code={self.code!r})'

    def __str__(self) -> str:
        if False:
            return 10
        return f'\n code: {self.code}\n stdout: {self.stdout}\n stderr: {self.stderr}'

def _run_pulumi_cmd(args: List[str], cwd: str, additional_env: Mapping[str, str], on_output: Optional[OnOutput]=None) -> CommandResult:
    if False:
        while True:
            i = 10
    if '--non-interactive' not in args:
        args.append('--non-interactive')
    env = {**os.environ, **additional_env}
    cmd = ['pulumi']
    cmd.extend(args)
    stdout_chunks: List[str] = []
    with tempfile.TemporaryFile() as stderr_file:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr_file, cwd=cwd, env=env) as process:
            assert process.stdout is not None
            while True:
                output = process.stdout.readline().decode(encoding='utf-8')
                if output == '' and process.poll() is not None:
                    break
                if output:
                    text = output.rstrip()
                    if on_output:
                        on_output(text)
                    stdout_chunks.append(text)
            code = process.returncode
        stderr_file.seek(0)
        stderr_contents = stderr_file.read().decode('utf-8')
    result = CommandResult(stderr=stderr_contents, stdout='\n'.join(stdout_chunks), code=code)
    if code != 0:
        raise create_command_error(result)
    return result