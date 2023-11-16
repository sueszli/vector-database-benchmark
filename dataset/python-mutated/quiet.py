import sys
from .highlighting import HighlightingStream
from ..loggerapi import LoggerApi

class QuietOutput(LoggerApi):

    def __init__(self, colors='AUTO', stderr=None):
        if False:
            for i in range(10):
                print('nop')
        self._stderr = HighlightingStream(stderr or sys.__stderr__, colors)

    def message(self, msg):
        if False:
            print('Hello World!')
        if msg.level in ('WARN', 'ERROR'):
            self._stderr.error(msg.message, msg.level)

class NoOutput(LoggerApi):
    pass