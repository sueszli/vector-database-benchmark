import logging
import os
import signal
import sys
import threading
from pathlib import Path
from socket import socket
from types import FrameType
from typing import Callable, Iterator, List, Optional
import click
from uvicorn._subprocess import get_subprocess
from uvicorn.config import Config
HANDLED_SIGNALS = (signal.SIGINT, signal.SIGTERM)
logger = logging.getLogger('uvicorn.error')

class BaseReload:

    def __init__(self, config: Config, target: Callable[[Optional[List[socket]]], None], sockets: List[socket]) -> None:
        if False:
            i = 10
            return i + 15
        self.config = config
        self.target = target
        self.sockets = sockets
        self.should_exit = threading.Event()
        self.pid = os.getpid()
        self.is_restarting = False
        self.reloader_name: Optional[str] = None

    def signal_handler(self, sig: int, frame: Optional[FrameType]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        A signal handler that is registered with the parent process.\n        '
        if sys.platform == 'win32' and self.is_restarting:
            self.is_restarting = False
        else:
            self.should_exit.set()

    def run(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.startup()
        for changes in self:
            if changes:
                logger.warning('%s detected changes in %s. Reloading...', self.reloader_name, ', '.join(map(_display_path, changes)))
                self.restart()
        self.shutdown()

    def pause(self) -> None:
        if False:
            return 10
        if self.should_exit.wait(self.config.reload_delay):
            raise StopIteration()

    def __iter__(self) -> Iterator[Optional[List[Path]]]:
        if False:
            for i in range(10):
                print('nop')
        return self

    def __next__(self) -> Optional[List[Path]]:
        if False:
            i = 10
            return i + 15
        return self.should_restart()

    def startup(self) -> None:
        if False:
            while True:
                i = 10
        message = f'Started reloader process [{self.pid}] using {self.reloader_name}'
        color_message = 'Started reloader process [{}] using {}'.format(click.style(str(self.pid), fg='cyan', bold=True), click.style(str(self.reloader_name), fg='cyan', bold=True))
        logger.info(message, extra={'color_message': color_message})
        for sig in HANDLED_SIGNALS:
            signal.signal(sig, self.signal_handler)
        self.process = get_subprocess(config=self.config, target=self.target, sockets=self.sockets)
        self.process.start()

    def restart(self) -> None:
        if False:
            while True:
                i = 10
        if sys.platform == 'win32':
            self.is_restarting = True
            assert self.process.pid is not None
            os.kill(self.process.pid, signal.CTRL_C_EVENT)
        else:
            self.process.terminate()
        self.process.join()
        self.process = get_subprocess(config=self.config, target=self.target, sockets=self.sockets)
        self.process.start()

    def shutdown(self) -> None:
        if False:
            while True:
                i = 10
        if sys.platform == 'win32':
            self.should_exit.set()
        else:
            self.process.terminate()
        self.process.join()
        for sock in self.sockets:
            sock.close()
        message = 'Stopping reloader process [{}]'.format(str(self.pid))
        color_message = 'Stopping reloader process [{}]'.format(click.style(str(self.pid), fg='cyan', bold=True))
        logger.info(message, extra={'color_message': color_message})

    def should_restart(self) -> Optional[List[Path]]:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Reload strategies should override should_restart()')

def _display_path(path: Path) -> str:
    if False:
        return 10
    try:
        return f"'{path.relative_to(Path.cwd())}'"
    except ValueError:
        return f"'{path}'"