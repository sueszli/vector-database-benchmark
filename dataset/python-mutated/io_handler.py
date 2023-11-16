from __future__ import annotations
import logging
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from logging import LogRecord
    from cleo.io.io import IO

class IOHandler(logging.Handler):

    def __init__(self, io: IO) -> None:
        if False:
            print('Hello World!')
        self._io = io
        super().__init__()

    def emit(self, record: LogRecord) -> None:
        if False:
            return 10
        try:
            msg = self.format(record)
            level = record.levelname.lower()
            err = level in ('warning', 'error', 'exception', 'critical')
            if err:
                self._io.write_error_line(msg)
            else:
                self._io.write_line(msg)
        except Exception:
            self.handleError(record)