from __future__ import annotations
import asyncio
import logging
import sys
from typing import IO
from mitmproxy import ctx
from mitmproxy import log
from mitmproxy.utils import vt_codes

class TermLog:
    _teardown_task: asyncio.Task | None = None

    def __init__(self, out: IO[str] | None=None):
        if False:
            return 10
        self.logger = TermLogHandler(out)
        self.logger.install()

    def load(self, loader):
        if False:
            print('Hello World!')
        loader.add_option('termlog_verbosity', str, 'info', 'Log verbosity.', choices=log.LogLevels)
        self.logger.setLevel(logging.INFO)

    def configure(self, updated):
        if False:
            print('Hello World!')
        if 'termlog_verbosity' in updated:
            self.logger.setLevel(ctx.options.termlog_verbosity.upper())

    def done(self):
        if False:
            i = 10
            return i + 15
        t = self._teardown()
        try:
            self._teardown_task = asyncio.create_task(t)
        except RuntimeError:
            asyncio.run(t)

    async def _teardown(self):
        self.logger.uninstall()

class TermLogHandler(log.MitmLogHandler):

    def __init__(self, out: IO[str] | None=None):
        if False:
            return 10
        super().__init__()
        self.file: IO[str] = out or sys.stdout
        self.has_vt_codes = vt_codes.ensure_supported(self.file)
        self.formatter = log.MitmFormatter(self.has_vt_codes)

    def emit(self, record: logging.LogRecord) -> None:
        if False:
            while True:
                i = 10
        try:
            print(self.format(record), file=self.file)
        except OSError:
            sys.exit(1)