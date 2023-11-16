import logging
import sys
import tempfile
from contextlib import asynccontextmanager, contextmanager
from functools import partial
from pathlib import Path
from subprocess import DEVNULL
from typing import AsyncContextManager, AsyncGenerator, Generator, List, Optional, Union
import trio
from streamlink.utils.path import resolve_executable
from streamlink.webbrowser.exceptions import WebbrowserError
log = logging.getLogger(__name__)

class Webbrowser:
    ERROR_RESOLVE = 'Could not find web browser executable'
    TIMEOUT = 10

    @classmethod
    def names(cls) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        return []

    @classmethod
    def fallback_paths(cls) -> List[Union[str, Path]]:
        if False:
            print('Hello World!')
        return []

    @classmethod
    def launch_args(cls) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        return []

    def __init__(self, executable: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        resolved = resolve_executable(executable, self.names(), self.fallback_paths())
        if not resolved:
            raise WebbrowserError(f'Invalid web browser executable: {executable}' if executable else f'{self.ERROR_RESOLVE}: Please set the path to a supported web browser using --webbrowser-executable')
        self.executable: Union[str, Path] = resolved
        self.arguments: List[str] = self.launch_args().copy()

    def launch(self, timeout: Optional[float]=None) -> AsyncContextManager[trio.Nursery]:
        if False:
            i = 10
            return i + 15
        return self._launch(self.executable, self.arguments, timeout=timeout)

    def _launch(self, executable: Union[str, Path], arguments: List[str], timeout: Optional[float]=None) -> AsyncContextManager[trio.Nursery]:
        if False:
            while True:
                i = 10
        if timeout is None:
            timeout = self.TIMEOUT
        launcher = _WebbrowserLauncher(executable, arguments, timeout)
        return launcher.launch()

    @staticmethod
    @contextmanager
    def _create_temp_dir() -> Generator[str, None, None]:
        if False:
            i = 10
            return i + 15
        kwargs = {'ignore_cleanup_errors': True} if sys.version_info >= (3, 10) else {}
        with tempfile.TemporaryDirectory(**kwargs) as temp_file:
            yield temp_file

class _WebbrowserLauncher:

    def __init__(self, executable: Union[str, Path], arguments: List[str], timeout: float):
        if False:
            return 10
        self.executable = executable
        self.arguments = arguments
        self.timeout = timeout
        self._process_ended_early = False

    @asynccontextmanager
    async def launch(self) -> AsyncGenerator[trio.Nursery, None]:
        async with trio.open_nursery() as nursery:
            log.info(f'Launching web browser: {self.executable}')
            run_process = partial(trio.run_process, [self.executable, *self.arguments], check=False, stdout=DEVNULL, stderr=DEVNULL)
            process: trio.Process = await nursery.start(run_process)
            nursery.start_soon(self._task_process_watcher, process, nursery)
            try:
                with trio.move_on_after(self.timeout) as cancel_scope:
                    yield nursery
            except BaseException:
                raise
            else:
                if cancel_scope.cancelled_caught:
                    log.warning('Web browser task group has timed out')
            finally:
                if not self._process_ended_early:
                    log.debug('Waiting for web browser process to terminate')
                nursery.cancel_scope.cancel()

    async def _task_process_watcher(self, process: trio.Process, nursery: trio.Nursery) -> None:
        """Task for cancelling the launch task group if the user closes the browser or if it exits early on its own"""
        await process.wait()
        if not nursery.cancel_scope.cancel_called:
            self._process_ended_early = True
            log.warning('Web browser process ended early')
            nursery.cancel_scope.cancel()