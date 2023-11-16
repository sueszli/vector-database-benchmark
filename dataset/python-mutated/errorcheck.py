import asyncio
import logging
import sys
from mitmproxy import log

class ErrorCheck:
    """Monitor startup for error log entries, and terminate immediately if there are some."""
    repeat_errors_on_stderr: bool
    '\n    Repeat all errors on stderr before exiting.\n    This is useful for the console UI, which otherwise swallows all output.\n    '

    def __init__(self, repeat_errors_on_stderr: bool=False) -> None:
        if False:
            while True:
                i = 10
        self.repeat_errors_on_stderr = repeat_errors_on_stderr
        self.logger = ErrorCheckHandler()
        self.logger.install()

    def finish(self):
        if False:
            while True:
                i = 10
        self.logger.uninstall()

    async def shutdown_if_errored(self):
        await asyncio.sleep(0)
        if self.logger.has_errored:
            plural = 's' if len(self.logger.has_errored) > 1 else ''
            if self.repeat_errors_on_stderr:
                msg = '\n'.join((r.msg for r in self.logger.has_errored))
                print(f'Error{plural} logged during startup: {msg}', file=sys.stderr)
            else:
                print(f'Error{plural} logged during startup, exiting...', file=sys.stderr)
            sys.exit(1)

class ErrorCheckHandler(log.MitmLogHandler):

    def __init__(self) -> None:
        if False:
            return 10
        super().__init__(logging.ERROR)
        self.has_errored: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if False:
            while True:
                i = 10
        self.has_errored.append(record)