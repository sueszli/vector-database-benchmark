import sys
from datetime import datetime
from typing import Callable, Literal
from robot.errors import DataError
from robot.model import Message as BaseMessage, MessageLevel
from robot.utils import console_encode, safe_str
LEVELS = {'NONE': 7, 'SKIP': 6, 'FAIL': 5, 'ERROR': 4, 'WARN': 3, 'INFO': 2, 'DEBUG': 1, 'TRACE': 0}
PseudoLevel = Literal['HTML', 'CONSOLE']

def write_to_console(msg, newline=True, stream='stdout'):
    if False:
        for i in range(10):
            print('nop')
    msg = str(msg)
    if newline:
        msg += '\n'
    stream = sys.__stdout__ if stream.lower() != 'stderr' else sys.__stderr__
    stream.write(console_encode(msg, stream=stream))
    stream.flush()

class AbstractLogger:

    def __init__(self, level='TRACE'):
        if False:
            return 10
        self._is_logged = IsLogged(level)

    def set_level(self, level):
        if False:
            while True:
                i = 10
        return self._is_logged.set_level(level)

    def trace(self, msg):
        if False:
            return 10
        self.write(msg, 'TRACE')

    def debug(self, msg):
        if False:
            for i in range(10):
                print('nop')
        self.write(msg, 'DEBUG')

    def info(self, msg):
        if False:
            for i in range(10):
                print('nop')
        self.write(msg, 'INFO')

    def warn(self, msg):
        if False:
            return 10
        self.write(msg, 'WARN')

    def fail(self, msg):
        if False:
            for i in range(10):
                print('nop')
        html = False
        if msg.startswith('*HTML*'):
            html = True
            msg = msg[6:].lstrip()
        self.write(msg, 'FAIL', html)

    def skip(self, msg):
        if False:
            print('Hello World!')
        html = False
        if msg.startswith('*HTML*'):
            html = True
            msg = msg[6:].lstrip()
        self.write(msg, 'SKIP', html)

    def error(self, msg):
        if False:
            print('Hello World!')
        self.write(msg, 'ERROR')

    def write(self, message, level, html=False):
        if False:
            return 10
        self.message(Message(message, level, html))

    def message(self, msg):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError(self.__class__)

class Message(BaseMessage):
    __slots__ = ['_message']

    def __init__(self, message: 'str|Callable[[], str]', level: 'MessageLevel|PseudoLevel'='INFO', html: bool=False, timestamp: 'datetime|str|None'=None):
        if False:
            while True:
                i = 10
        (level, html) = self._get_level_and_html(level, html)
        super().__init__(message, level, html, timestamp or datetime.now())

    def _get_level_and_html(self, level, html) -> 'tuple[MessageLevel, bool]':
        if False:
            while True:
                i = 10
        level = level.upper()
        if level == 'HTML':
            return ('INFO', True)
        if level == 'CONSOLE':
            return ('INFO', html)
        if level in LEVELS:
            return (level, html)
        raise DataError(f"Invalid log level '{level}'.")

    @property
    def message(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        self.resolve_delayed_message()
        return self._message

    @message.setter
    def message(self, message: 'str|Callable[[], str]'):
        if False:
            while True:
                i = 10
        if not callable(message):
            if not isinstance(message, str):
                message = safe_str(message)
            if '\r\n' in message:
                message = message.replace('\r\n', '\n')
        self._message = message

    def resolve_delayed_message(self):
        if False:
            while True:
                i = 10
        if callable(self._message):
            self._message = self._message()

class IsLogged:

    def __init__(self, level):
        if False:
            for i in range(10):
                print('nop')
        self.level = level.upper()
        self._int_level = self._level_to_int(level)

    def __call__(self, level):
        if False:
            print('Hello World!')
        return self._level_to_int(level) >= self._int_level

    def set_level(self, level):
        if False:
            i = 10
            return i + 15
        old = self.level
        self.__init__(level)
        return old

    def _level_to_int(self, level):
        if False:
            i = 10
            return i + 15
        try:
            return LEVELS[level.upper()]
        except KeyError:
            raise DataError("Invalid log level '%s'." % level)