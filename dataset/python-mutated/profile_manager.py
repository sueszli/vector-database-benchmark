import asyncio
import shutil
import subprocess
import sys
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
DARWIN_SET_CHOWN_CMD = 'sudo chown root: `which py-spy`'
LINUX_SET_CHOWN_CMD = 'sudo chown root:root `which py-spy`'
PYSPY_PERMISSIONS_ERROR_MESSAGE = '\nNote that this command requires `py-spy` to be installed with root permissions. You\ncan install `py-spy` and give it root permissions as follows:\n  $ pip install py-spy\n  $ {set_chown_command}\n  $ sudo chmod u+s `which py-spy`\n\nAlternatively, you can start Ray with passwordless sudo / root permissions.\n\n'

def _format_failed_pyspy_command(cmd, stdout, stderr) -> str:
    if False:
        print('Hello World!')
    stderr_str = stderr.decode('utf-8')
    extra_message = ''
    if 'permission' in stderr_str.lower():
        set_chown_command = DARWIN_SET_CHOWN_CMD if sys.platform == 'darwin' else LINUX_SET_CHOWN_CMD
        extra_message = PYSPY_PERMISSIONS_ERROR_MESSAGE.format(set_chown_command=set_chown_command)
    return f"Failed to execute `{cmd}`.\n{extra_message}\n=== stderr ===\n{stderr.decode('utf-8')}\n\n=== stdout ===\n{stdout.decode('utf-8')}\n"

async def _can_passwordless_sudo() -> bool:
    process = await asyncio.create_subprocess_exec('sudo', '-n', 'true', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (_, _) = await process.communicate()
    return process.returncode == 0

class CpuProfilingManager:

    def __init__(self, profile_dir_path: str):
        if False:
            while True:
                i = 10
        self.profile_dir_path = Path(profile_dir_path)
        self.profile_dir_path.mkdir(exist_ok=True)

    async def trace_dump(self, pid: int, native: bool=False) -> (bool, str):
        pyspy = shutil.which('py-spy')
        if pyspy is None:
            return (False, 'py-spy is not installed')
        cmd = [pyspy, 'dump', '-p', str(pid)]
        if sys.platform == 'linux' and native:
            cmd.append('--native')
        if await _can_passwordless_sudo():
            cmd = ['sudo', '-n'] + cmd
        process = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdout, stderr) = await process.communicate()
        if process.returncode != 0:
            return (False, _format_failed_pyspy_command(cmd, stdout, stderr))
        else:
            return (True, stdout.decode('utf-8'))

    async def cpu_profile(self, pid: int, format='flamegraph', duration: float=5, native: bool=False) -> (bool, str):
        pyspy = shutil.which('py-spy')
        if pyspy is None:
            return (False, 'py-spy is not installed')
        if format not in ('flamegraph', 'raw', 'speedscope'):
            return (False, f'Invalid format {format}, ' + 'must be [flamegraph, raw, speedscope]')
        if format == 'flamegraph':
            extension = 'svg'
        else:
            extension = 'txt'
        profile_file_path = self.profile_dir_path / f'{format}_{pid}_cpu_profiling.{extension}'
        cmd = [pyspy, 'record', '-o', profile_file_path, '-p', str(pid), '-d', str(duration), '-f', format]
        if sys.platform == 'linux' and native:
            cmd.append('--native')
        if await _can_passwordless_sudo():
            cmd = ['sudo', '-n'] + cmd
        process = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdout, stderr) = await process.communicate()
        if process.returncode != 0:
            return (False, _format_failed_pyspy_command(cmd, stdout, stderr))
        else:
            return (True, open(profile_file_path, 'rb').read())