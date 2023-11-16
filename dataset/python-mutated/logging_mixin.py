from __future__ import annotations
import abc
import enum
import logging
import sys
from io import IOBase
from logging import Handler, StreamHandler
from typing import IO, TYPE_CHECKING, Any, Optional, TypeVar, cast
import re2
if TYPE_CHECKING:
    from logging import Logger
ANSI_ESCAPE = re2.compile('\\x1B[@-_][0-?]*[ -/]*[@-~]')

class SetContextPropagate(enum.Enum):
    """Sentinel objects for log propagation contexts.

    :meta private:
    """
    MAINTAIN_PROPAGATE = object()
    DISABLE_PROPAGATE = object()

def __getattr__(name):
    if False:
        print('Hello World!')
    if name in ('DISABLE_PROPOGATE', 'DISABLE_PROPAGATE'):
        return SetContextPropagate.DISABLE_PROPAGATE
    raise AttributeError(f'module {__name__} has no attribute {name}')

def remove_escape_codes(text: str) -> str:
    if False:
        print('Hello World!')
    'Remove ANSI escapes codes from string; used to remove "colors" from log messages.'
    return ANSI_ESCAPE.sub('', text)
_T = TypeVar('_T')

class LoggingMixin:
    """Convenience super-class to have a logger configured with the class name."""
    _log: logging.Logger | None = None
    _log_config_logger_name: Optional[str] = None
    _logger_name: Optional[str] = None

    def __init__(self, context=None):
        if False:
            return 10
        self._set_context(context)

    @staticmethod
    def _create_logger_name(logged_class: type[_T], log_config_logger_name: str | None=None, class_logger_name: str | None=None) -> str:
        if False:
            i = 10
            return i + 15
        'Generate a logger name for the given `logged_class`.\n\n        By default, this function returns the `class_logger_name` as logger name. If it is not provided,\n        the {class.__module__}.{class.__name__} is returned instead. When a `parent_logger_name` is provided,\n        it will prefix the logger name with a separating dot.\n        '
        logger_name: str = class_logger_name if class_logger_name is not None else f'{logged_class.__module__}.{logged_class.__name__}'
        if log_config_logger_name:
            return f'{log_config_logger_name}.{logger_name}' if logger_name else log_config_logger_name
        return logger_name

    @classmethod
    def _get_log(cls, obj: Any, clazz: type[_T]) -> Logger:
        if False:
            return 10
        if obj._log is None:
            logger_name: str = cls._create_logger_name(logged_class=clazz, log_config_logger_name=obj._log_config_logger_name, class_logger_name=obj._logger_name)
            obj._log = logging.getLogger(logger_name)
        return obj._log

    @classmethod
    def logger(cls) -> Logger:
        if False:
            print('Hello World!')
        'Return a logger.'
        return LoggingMixin._get_log(cls, cls)

    @property
    def log(self) -> Logger:
        if False:
            i = 10
            return i + 15
        'Return a logger.'
        return LoggingMixin._get_log(self, self.__class__)

    def _set_context(self, context):
        if False:
            i = 10
            return i + 15
        if context is not None:
            set_context(self.log, context)

class ExternalLoggingMixin:
    """Define a log handler based on an external service (e.g. ELK, StackDriver)."""

    @property
    @abc.abstractmethod
    def log_name(self) -> str:
        if False:
            i = 10
            return i + 15
        'Return log name.'

    @abc.abstractmethod
    def get_external_log_url(self, task_instance, try_number) -> str:
        if False:
            while True:
                i = 10
        'Return the URL for log visualization in the external service.'

    @property
    @abc.abstractmethod
    def supports_external_link(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Return whether handler is able to support external links.'

class StreamLogWriter(IOBase, IO[str]):
    """
    Allows to redirect stdout and stderr to logger.

    :param log: The log level method to write to, ie. log.debug, log.warning
    """
    encoding: None = None

    def __init__(self, logger, level):
        if False:
            print('Hello World!')
        self.logger = logger
        self.level = level
        self._buffer = ''

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Provide close method, for compatibility with the io.IOBase interface.\n\n        This is a no-op method.\n        '

    @property
    def closed(self):
        if False:
            i = 10
            return i + 15
        "\n        Return False to indicate that the stream is not closed.\n\n        Streams will be open for the duration of Airflow's lifecycle.\n\n        For compatibility with the io.IOBase interface.\n        "
        return False

    def _propagate_log(self, message):
        if False:
            return 10
        'Propagate message removing escape codes.'
        self.logger.log(self.level, remove_escape_codes(message))

    def write(self, message):
        if False:
            print('Hello World!')
        '\n        Do whatever it takes to actually log the specified logging record.\n\n        :param message: message to log\n        '
        if not message.endswith('\n'):
            self._buffer += message
        else:
            self._buffer += message.rstrip()
            self.flush()

    def flush(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure all logging output has been flushed.'
        buf = self._buffer
        if buf:
            self._buffer = ''
            self._propagate_log(buf)

    def isatty(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return False to indicate the fd is not connected to a tty(-like) device.\n\n        For compatibility reasons.\n        '
        return False

class RedirectStdHandler(StreamHandler):
    """
    Custom StreamHandler that uses current sys.stderr/stdout as the stream for logging.

    This class is like a StreamHandler using sys.stderr/stdout, but uses
    whatever sys.stderr/stdout is currently set to rather than the value of
    sys.stderr/stdout at handler construction time, except when running a
    task in a kubernetes executor pod.
    """

    def __init__(self, stream):
        if False:
            return 10
        if not isinstance(stream, str):
            raise Exception("Cannot use file like objects. Use 'stdout' or 'stderr' as a str and without 'ext://'.")
        self._use_stderr = True
        if 'stdout' in stream:
            self._use_stderr = False
            self._orig_stream = sys.stdout
        else:
            self._orig_stream = sys.stderr
        Handler.__init__(self)

    @property
    def stream(self):
        if False:
            i = 10
            return i + 15
        'Returns current stream.'
        from airflow.settings import IS_EXECUTOR_CONTAINER, IS_K8S_EXECUTOR_POD
        if IS_K8S_EXECUTOR_POD or IS_EXECUTOR_CONTAINER:
            return self._orig_stream
        if self._use_stderr:
            return sys.stderr
        return sys.stdout

def set_context(logger, value):
    if False:
        i = 10
        return i + 15
    '\n    Walk the tree of loggers and try to set the context for each handler.\n\n    :param logger: logger\n    :param value: value to set\n    '
    while logger:
        orig_propagate = logger.propagate
        for handler in logger.handlers:
            if hasattr(handler, 'set_context'):
                from airflow.utils.log.file_task_handler import FileTaskHandler
                flag = cast(FileTaskHandler, handler).set_context(value)
                if flag is not SetContextPropagate.MAINTAIN_PROPAGATE:
                    logger.propagate = False
        if orig_propagate is True:
            logger = logger.parent
        else:
            break