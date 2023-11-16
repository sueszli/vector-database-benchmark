"""Class responsible for colouring logs based on log level."""
from __future__ import annotations
import sys
from typing import TYPE_CHECKING, Any
import re2
from colorlog import TTYColoredFormatter
from colorlog.escape_codes import esc, escape_codes
from airflow.utils.log.timezone_aware import TimezoneAware
if TYPE_CHECKING:
    from logging import LogRecord
DEFAULT_COLORS = {'DEBUG': 'green', 'INFO': '', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'red'}
BOLD_ON = escape_codes['bold']
BOLD_OFF = esc('22')

class CustomTTYColoredFormatter(TTYColoredFormatter, TimezoneAware):
    """
    Custom log formatter.

    Extends `colored.TTYColoredFormatter` by adding attributes
    to message arguments and coloring error traceback.
    """

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        kwargs['stream'] = sys.stdout or kwargs.get('stream')
        kwargs['log_colors'] = DEFAULT_COLORS
        super().__init__(*args, **kwargs)

    @staticmethod
    def _color_arg(arg: Any) -> str | float | int:
        if False:
            return 10
        if isinstance(arg, (int, float)):
            return arg
        return BOLD_ON + str(arg) + BOLD_OFF

    @staticmethod
    def _count_number_of_arguments_in_message(record: LogRecord) -> int:
        if False:
            i = 10
            return i + 15
        matches = re2.findall('%.', record.msg)
        return len(matches) if matches else 0

    def _color_record_args(self, record: LogRecord) -> LogRecord:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(record.args, (tuple, list)):
            record.args = tuple((self._color_arg(arg) for arg in record.args))
        elif isinstance(record.args, dict):
            if self._count_number_of_arguments_in_message(record) > 1:
                record.args = {key: self._color_arg(value) for (key, value) in record.args.items()}
            else:
                record.args = self._color_arg(record.args)
        elif isinstance(record.args, str):
            record.args = self._color_arg(record.args)
        return record

    def _color_record_traceback(self, record: LogRecord) -> LogRecord:
        if False:
            while True:
                i = 10
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            if record.exc_text:
                record.exc_text = self.color(self.log_colors, record.levelname) + record.exc_text + escape_codes['reset']
        return record

    def format(self, record: LogRecord) -> str:
        if False:
            while True:
                i = 10
        try:
            if self.stream.isatty():
                record = self._color_record_args(record)
                record = self._color_record_traceback(record)
            return super().format(record)
        except ValueError:
            from logging import Formatter
            return Formatter().format(record)