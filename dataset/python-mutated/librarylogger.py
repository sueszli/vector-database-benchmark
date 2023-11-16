"""Implementation of the public logging API for libraries.

This is exposed via :py:mod:`robot.api.logger`. Implementation must reside
here to avoid cyclic imports.
"""
import threading
from .logger import LOGGER
from .loggerhelper import Message, write_to_console
LOGGING_THREADS = ('MainThread', 'RobotFrameworkTimeoutThread')

def write(msg, level, html=False):
    if False:
        print('Hello World!')
    if callable(msg):
        msg = str(msg)
    if level.upper() not in ('TRACE', 'DEBUG', 'INFO', 'HTML', 'WARN', 'ERROR'):
        if level.upper() == 'CONSOLE':
            level = 'INFO'
            console(msg)
        else:
            raise RuntimeError("Invalid log level '%s'." % level)
    if threading.current_thread().name in LOGGING_THREADS:
        LOGGER.log_message(Message(msg, level, html))

def trace(msg, html=False):
    if False:
        for i in range(10):
            print('nop')
    write(msg, 'TRACE', html)

def debug(msg, html=False):
    if False:
        return 10
    write(msg, 'DEBUG', html)

def info(msg, html=False, also_console=False):
    if False:
        print('Hello World!')
    write(msg, 'INFO', html)
    if also_console:
        console(msg)

def warn(msg, html=False):
    if False:
        print('Hello World!')
    write(msg, 'WARN', html)

def error(msg, html=False):
    if False:
        while True:
            i = 10
    write(msg, 'ERROR', html)

def console(msg, newline=True, stream='stdout'):
    if False:
        return 10
    write_to_console(msg, newline, stream)