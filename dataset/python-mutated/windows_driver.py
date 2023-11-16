from __future__ import annotations
import asyncio
import sys
from threading import Event, Thread
from typing import TYPE_CHECKING, Callable
from ..driver import Driver
from . import win32
from ._writer_thread import WriterThread
if TYPE_CHECKING:
    from ..app import App

class WindowsDriver(Driver):
    """Powers display and input for Windows."""

    def __init__(self, app: App, *, debug: bool=False, size: tuple[int, int] | None=None) -> None:
        if False:
            while True:
                i = 10
        'Initialize Windows driver.\n\n        Args:\n            app: The App instance.\n            debug: Enable debug mode.\n            size: Initial size of the terminal or `None` to detect.\n        '
        super().__init__(app, debug=debug, size=size)
        self._file = sys.__stdout__
        self.exit_event = Event()
        self._event_thread: Thread | None = None
        self._restore_console: Callable[[], None] | None = None
        self._writer_thread: WriterThread | None = None

    def write(self, data: str) -> None:
        if False:
            return 10
        'Write data to the output device.\n\n        Args:\n            data: Raw data.\n        '
        assert self._writer_thread is not None, 'Driver must be in application mode'
        self._writer_thread.write(data)

    def _enable_mouse_support(self) -> None:
        if False:
            i = 10
            return i + 15
        'Enable reporting of mouse events.'
        write = self.write
        write('\x1b[?1000h')
        write('\x1b[?1003h')
        write('\x1b[?1015h')
        write('\x1b[?1006h')
        self.flush()

    def _disable_mouse_support(self) -> None:
        if False:
            while True:
                i = 10
        'Disable reporting of mouse events.'
        write = self.write
        write('\x1b[?1000l')
        write('\x1b[?1003l')
        write('\x1b[?1015l')
        write('\x1b[?1006l')
        self.flush()

    def _enable_bracketed_paste(self) -> None:
        if False:
            i = 10
            return i + 15
        'Enable bracketed paste mode.'
        self.write('\x1b[?2004h')

    def _disable_bracketed_paste(self) -> None:
        if False:
            print('Hello World!')
        'Disable bracketed paste mode.'
        self.write('\x1b[?2004l')

    def start_application_mode(self) -> None:
        if False:
            i = 10
            return i + 15
        'Start application mode.'
        loop = asyncio.get_running_loop()
        self._restore_console = win32.enable_application_mode()
        self._writer_thread = WriterThread(self._file)
        self._writer_thread.start()
        self.write('\x1b[?1049h')
        self._enable_mouse_support()
        self.write('\x1b[?25l')
        self.write('\x1b[?1003h\n')
        self._enable_bracketed_paste()
        self._event_thread = win32.EventMonitor(loop, self._app, self.exit_event, self.process_event)
        self._event_thread.start()

    def disable_input(self) -> None:
        if False:
            i = 10
            return i + 15
        'Disable further input.'
        try:
            if not self.exit_event.is_set():
                self._disable_mouse_support()
                self.exit_event.set()
                if self._event_thread is not None:
                    self._event_thread.join()
                    self._event_thread = None
                self.exit_event.clear()
        except Exception as error:
            pass

    def stop_application_mode(self) -> None:
        if False:
            print('Hello World!')
        'Stop application mode, restore state.'
        self._disable_bracketed_paste()
        self.disable_input()
        self.write('\x1b[?1049l' + '\x1b[?25h')
        self.flush()

    def close(self) -> None:
        if False:
            i = 10
            return i + 15
        'Perform cleanup.'
        if self._writer_thread is not None:
            self._writer_thread.stop()
        if self._restore_console:
            self._restore_console()