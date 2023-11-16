from __future__ import annotations
import logging
from pre_commit import color
from pre_commit.logging_handler import LoggingHandler

def _log_record(message, level):
    if False:
        while True:
            i = 10
    return logging.LogRecord('name', level, '', 1, message, {}, None)

def test_logging_handler_color(cap_out):
    if False:
        while True:
            i = 10
    handler = LoggingHandler(True)
    handler.emit(_log_record('hi', logging.WARNING))
    ret = cap_out.get()
    assert ret == f'{color.YELLOW}[WARNING]{color.NORMAL} hi\n'

def test_logging_handler_no_color(cap_out):
    if False:
        i = 10
        return i + 15
    handler = LoggingHandler(False)
    handler.emit(_log_record('hi', logging.WARNING))
    assert cap_out.get() == '[WARNING] hi\n'