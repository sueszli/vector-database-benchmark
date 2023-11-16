import logging
from typing import List, Union, Dict
from types import TracebackType
LogMessage = List[Union[str, Dict[str, TracebackType]]]

class MockedLoggingHandler(logging.Handler):
    debug: List[LogMessage] = []
    warning: List[LogMessage] = []
    info: List[LogMessage] = []
    error: List[LogMessage] = []
    critical: List[LogMessage] = []

    def emit(self, record):
        if False:
            for i in range(10):
                print('nop')
        if record.exc_info:
            value = {'message': record.getMessage(), 'exc_info': record.exc_info}
        else:
            value = record.getMessage()
        getattr(self.__class__, record.levelname.lower()).append(value)

    @classmethod
    def reset(cls):
        if False:
            i = 10
            return i + 15
        for attr in dir(cls):
            if isinstance(getattr(cls, attr), list):
                setattr(cls, attr, [])