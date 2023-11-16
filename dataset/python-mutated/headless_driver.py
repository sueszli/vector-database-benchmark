from __future__ import annotations
import asyncio
from .. import events
from ..driver import Driver
from ..geometry import Size

class HeadlessDriver(Driver):
    """A do-nothing driver for testing."""

    @property
    def is_headless(self) -> bool:
        if False:
            while True:
                i = 10
        "Is the driver running in 'headless' mode?"
        return True

    def _get_terminal_size(self) -> tuple[int, int]:
        if False:
            for i in range(10):
                print('nop')
        if self._size is not None:
            return self._size
        width: int | None = 80
        height: int | None = 25
        import shutil
        try:
            (width, height) = shutil.get_terminal_size()
        except (AttributeError, ValueError, OSError):
            try:
                (width, height) = shutil.get_terminal_size()
            except (AttributeError, ValueError, OSError):
                pass
        width = width or 80
        height = height or 25
        return (width, height)

    def write(self, data: str) -> None:
        if False:
            while True:
                i = 10
        'Write data to the output device.\n\n        Args:\n            data: Raw data.\n        '

    def start_application_mode(self) -> None:
        if False:
            print('Hello World!')
        'Start application mode.'
        loop = asyncio.get_running_loop()

        def send_size_event() -> None:
            if False:
                for i in range(10):
                    print('nop')
            'Send first resize event.'
            terminal_size = self._get_terminal_size()
            (width, height) = terminal_size
            textual_size = Size(width, height)
            event = events.Resize(textual_size, textual_size)
            asyncio.run_coroutine_threadsafe(self._app._post_message(event), loop=loop)
        send_size_event()

    def disable_input(self) -> None:
        if False:
            i = 10
            return i + 15
        'Disable further input.'

    def stop_application_mode(self) -> None:
        if False:
            i = 10
            return i + 15
        'Stop application mode, restore state.'