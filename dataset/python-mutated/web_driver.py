"""

The Remote driver uses the following packet stricture.

1 byte for packet type. "D" for data, "M" for meta.
4 byte little endian integer for the size of the payload.
Arbitrary payload.


"""
from __future__ import annotations
import asyncio
import json
import os
import platform
import signal
import sys
from codecs import getincrementaldecoder
from functools import partial
from threading import Event, Thread
from .. import events, log, messages
from .._xterm_parser import XTermParser
from ..app import App
from ..driver import Driver
from ..geometry import Size
from ._byte_stream import ByteStream
from ._input_reader import InputReader
WINDOWS = platform.system() == 'Windows'

class _ExitInput(Exception):
    """Internal exception to force exit of input loop."""

class WebDriver(Driver):
    """A headless driver that may be run remotely."""

    def __init__(self, app: App, *, debug: bool=False, size: tuple[int, int] | None=None):
        if False:
            while True:
                i = 10
        if size is None:
            try:
                width = int(os.environ.get('COLUMNS', 80))
                height = int(os.environ.get('ROWS', 24))
            except ValueError:
                pass
            else:
                size = (width, height)
        super().__init__(app, debug=debug, size=size)
        self.stdout = sys.__stdout__
        self.fileno = sys.__stdout__.fileno()
        self._write = partial(os.write, self.fileno)
        self.exit_event = Event()
        self._key_thread: Thread = Thread(target=self.run_input_thread)
        self._input_reader = InputReader()

    def write(self, data: str) -> None:
        if False:
            i = 10
            return i + 15
        'Write data to the output device.\n\n        Args:\n            data: Raw data.\n        '
        data_bytes = data.encode('utf-8')
        self._write(b'D%s%s' % (len(data_bytes).to_bytes(4, 'big'), data_bytes))

    def write_meta(self, data: dict[str, object]) -> None:
        if False:
            print('Hello World!')
        'Write meta to the controlling process (i.e. textual-web)\n\n        Args:\n            data: Meta dict.\n        '
        meta_bytes = json.dumps(data).encode('utf-8', errors='ignore')
        self._write(b'M%s%s' % (len(meta_bytes).to_bytes(4, 'big'), meta_bytes))

    def flush(self) -> None:
        if False:
            while True:
                i = 10
        pass

    def _enable_mouse_support(self) -> None:
        if False:
            return 10
        'Enable reporting of mouse events.'
        write = self.write
        write('\x1b[?1000h')
        write('\x1b[?1003h')
        write('\x1b[?1015h')
        write('\x1b[?1006h')

    def _enable_bracketed_paste(self) -> None:
        if False:
            return 10
        'Enable bracketed paste mode.'
        self.write('\x1b[?2004h')

    def _disable_bracketed_paste(self) -> None:
        if False:
            while True:
                i = 10
        'Disable bracketed paste mode.'
        self.write('\x1b[?2004l')

    def _disable_mouse_support(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Disable reporting of mouse events.'
        write = self.write
        write('\x1b[?1000l')
        write('\x1b[?1003l')
        write('\x1b[?1015l')
        write('\x1b[?1006l')

    def _request_terminal_sync_mode_support(self) -> None:
        if False:
            return 10
        'Writes an escape sequence to query the terminal support for the sync protocol.'
        self.write('\x1b[?2026$p')

    def start_application_mode(self) -> None:
        if False:
            while True:
                i = 10
        'Start application mode.'
        loop = asyncio.get_running_loop()

        def do_exit() -> None:
            if False:
                return 10
            'Callback to force exit.'
            asyncio.run_coroutine_threadsafe(self._app._post_message(messages.ExitApp()), loop=loop)
        if not WINDOWS:
            for _signal in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(_signal, do_exit)
        self._write(b'__GANGLION__\n')
        self.write('\x1b[?1049h')
        self._enable_mouse_support()
        self.write('\x1b[?25l')
        self.write('\x1b[?1003h\n')
        size = Size(80, 24) if self._size is None else Size(*self._size)
        event = events.Resize(size, size)
        asyncio.run_coroutine_threadsafe(self._app._post_message(event), loop=loop)
        self._request_terminal_sync_mode_support()
        self._enable_bracketed_paste()
        self.flush()
        self._key_thread.start()

    def disable_input(self) -> None:
        if False:
            while True:
                i = 10
        'Disable further input.'

    def stop_application_mode(self) -> None:
        if False:
            print('Hello World!')
        'Stop application mode, restore state.'
        self.exit_event.set()
        self._input_reader.close()
        self.write_meta({'type': 'exit'})

    def run_input_thread(self) -> None:
        if False:
            i = 10
            return i + 15
        'Wait for input and dispatch events.'
        input_reader = self._input_reader
        parser = XTermParser(input_reader.more_data, debug=self._debug)
        utf8_decoder = getincrementaldecoder('utf-8')().decode
        decode = utf8_decoder
        byte_stream = ByteStream()
        try:
            for data in input_reader:
                for (packet_type, payload) in byte_stream.feed(data):
                    if packet_type == 'D':
                        for event in parser.feed(decode(payload)):
                            self.process_event(event)
                    else:
                        self._on_meta(packet_type, payload)
        except _ExitInput:
            pass
        except Exception:
            from traceback import format_exc
            log(format_exc())
        finally:
            input_reader.close()

    def _on_meta(self, packet_type: str, payload: bytes) -> None:
        if False:
            while True:
                i = 10
        'Private method to dispatch meta.\n\n        Args:\n            packet_type: Packet type (currently always "M")\n            payload: Meta payload (JSON encoded as bytes).\n        '
        payload_map = json.loads(payload)
        _type = payload_map.get('type')
        if isinstance(payload_map, dict):
            self.on_meta(_type, payload_map)

    def on_meta(self, packet_type: str, payload: dict) -> None:
        if False:
            print('Hello World!')
        'Process meta information.\n\n        Args:\n            packet_type: The type of the packet.\n            payload: meta dict.\n        '
        if packet_type == 'resize':
            self._size = (payload['width'], payload['height'])
            size = Size(*self._size)
            self._app.post_message(events.Resize(size, size))
        elif packet_type == 'quit':
            self._app.post_message(messages.ExitApp())
        elif packet_type == 'exit':
            raise _ExitInput()