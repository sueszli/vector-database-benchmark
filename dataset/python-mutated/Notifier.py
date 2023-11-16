"""
Notifier module: contains methods for handling information output
for the programmer/user
"""
from __future__ import annotations
from .Logger import Logger
from .LoggerGlobal import defaultLogger
from direct.showbase import PythonUtil
from panda3d.core import ConfigVariableBool, NotifyCategory, StreamWriter, Notify
import time
import sys
from typing import NoReturn

class NotifierException(Exception):
    pass

class Notifier:
    serverDelta = 0
    streamWriter: StreamWriter | None = None
    if ConfigVariableBool('notify-integrate', True):
        streamWriter = StreamWriter(Notify.out(), False)
    showTime = ConfigVariableBool('notify-timestamp', False)

    def __init__(self, name: str, logger: Logger | None=None) -> None:
        if False:
            return 10
        '\n        Parameters:\n            name (str): a string name given to this Notifier instance.\n            logger (Logger, optional): an optional Logger object for\n                piping output to.  If none is specified, the global\n                :data:`~.LoggerGlobal.defaultLogger` is used.\n        '
        self.__name = name
        if logger is None:
            self.__logger = defaultLogger
        else:
            self.__logger = logger
        self.__info = True
        self.__warning = True
        self.__debug = False
        self.__logging = False

    def setServerDelta(self, delta: float, timezone: int) -> None:
        if False:
            while True:
                i = 10
        "\n        Call this method on any Notify object to globally change the\n        timestamp printed for each line of all Notify objects.\n\n        This synchronizes the timestamp with the server's known time\n        of day, and also switches into the server's timezone.\n        "
        delta = int(round(delta))
        Notifier.serverDelta = delta + time.timezone - timezone
        NotifyCategory.setServerDelta(self.serverDelta)
        self.info('Notify clock adjusted by %s (and timezone adjusted by %s hours) to synchronize with server.' % (PythonUtil.formatElapsedSeconds(delta), (time.timezone - timezone) / 3600))

    def getTime(self) -> str:
        if False:
            return 10
        '\n        Return the time as a string suitable for printing at the\n        head of any notify message\n        '
        return time.strftime(':%m-%d-%Y %H:%M:%S ', time.localtime(time.time() + self.serverDelta))

    def getOnlyTime(self) -> str:
        if False:
            return 10
        '\n        Return the time as a string.\n        The Only in the name is referring to not showing the date.\n        '
        return time.strftime('%H:%M:%S', time.localtime(time.time() + self.serverDelta))

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        '\n        Print handling routine\n        '
        return '%s: info = %d, warning = %d, debug = %d, logging = %d' % (self.__name, self.__info, self.__warning, self.__debug, self.__logging)

    def setSeverity(self, severity: int) -> None:
        if False:
            while True:
                i = 10
        from panda3d.core import NSDebug, NSInfo, NSWarning, NSError
        if severity >= NSError:
            self.setWarning(False)
            self.setInfo(False)
            self.setDebug(False)
        elif severity == NSWarning:
            self.setWarning(True)
            self.setInfo(False)
            self.setDebug(False)
        elif severity == NSInfo:
            self.setWarning(True)
            self.setInfo(True)
            self.setDebug(False)
        elif severity <= NSDebug:
            self.setWarning(True)
            self.setInfo(True)
            self.setDebug(True)

    def getSeverity(self) -> int:
        if False:
            return 10
        from panda3d.core import NSDebug, NSInfo, NSWarning, NSError
        if self.getDebug():
            return NSDebug
        elif self.getInfo():
            return NSInfo
        elif self.getWarning():
            return NSWarning
        else:
            return NSError

    def error(self, errorString: object, exception: type[Exception]=NotifierException) -> NoReturn:
        if False:
            for i in range(10):
                print('nop')
        '\n        Raise an exception with given string and optional type:\n        Exception: error\n        '
        message = str(errorString)
        if Notifier.showTime:
            string = f'{self.getTime()}{exception!s}: {self.__name}(error): {message}'
        else:
            string = f'{exception!s}: {self.__name}(error): {message}'
        self.__log(string)
        raise exception(errorString)

    def warning(self, warningString: object) -> int:
        if False:
            return 10
        '\n        Issue the warning message if warn flag is on\n        '
        if self.__warning:
            message = str(warningString)
            if Notifier.showTime:
                string = f'{self.getTime()}{self.__name}(warning): {message}'
            else:
                string = f':{self.__name}(warning): {message}'
            self.__log(string)
            self.__print(string)
        return 1

    def setWarning(self, enable: bool) -> None:
        if False:
            while True:
                i = 10
        '\n        Enable/Disable the printing of warning messages\n        '
        self.__warning = enable

    def getWarning(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Return whether the printing of warning messages is on or off\n        '
        return self.__warning

    def debug(self, debugString: object) -> int:
        if False:
            return 10
        '\n        Issue the debug message if debug flag is on\n        '
        if self.__debug:
            message = str(debugString)
            if Notifier.showTime:
                string = f'{self.getTime()}{self.__name}(debug): {message}'
            else:
                string = f':{self.__name}(debug): {message}'
            self.__log(string)
            self.__print(string)
        return 1

    def setDebug(self, enable: bool) -> None:
        if False:
            return 10
        '\n        Enable/Disable the printing of debug messages\n        '
        self.__debug = enable

    def getDebug(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Return whether the printing of debug messages is on or off\n        '
        return self.__debug

    def info(self, infoString: object) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Print the given informational string, if info flag is on\n        '
        if self.__info:
            message = str(infoString)
            if Notifier.showTime:
                string = f'{self.getTime()}{self.__name}: {message}'
            else:
                string = f':{self.__name}: {message}'
            self.__log(string)
            self.__print(string)
        return 1

    def getInfo(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return whether the printing of info messages is on or off\n        '
        return self.__info

    def setInfo(self, enable: bool) -> None:
        if False:
            return 10
        '\n        Enable/Disable informational message  printing\n        '
        self.__info = enable

    def __log(self, logEntry: str) -> None:
        if False:
            print('Hello World!')
        '\n        Determine whether to send informational message to the logger\n        '
        if self.__logging:
            self.__logger.log(logEntry)

    def getLogging(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return 1 if logging enabled, 0 otherwise\n        '
        return self.__logging

    def setLogging(self, enable: bool) -> None:
        if False:
            print('Hello World!')
        '\n        Set the logging flag to int (1=on, 0=off)\n        '
        self.__logging = enable

    def __print(self, string: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Prints the string to output followed by a newline.\n        '
        if self.streamWriter:
            self.streamWriter.write(string + '\n')
        else:
            sys.stderr.write(string + '\n')

    def debugStateCall(self, obj=None, fsmMemberName='fsm', secondaryFsm='secondaryFSM'):
        if False:
            print('Hello World!')
        '\n        If this notify is in debug mode, print the time of the\n        call followed by the [fsm state] notifier category and\n        the function call (with parameters).\n        '
        if __debug__ and self.__debug:
            state = ''
            doId = ''
            if obj is not None:
                fsm = obj.__dict__.get(fsmMemberName)
                if fsm is not None:
                    stateObj = fsm.getCurrentState()
                    if stateObj is not None:
                        state = stateObj.getName()
                fsm = obj.__dict__.get(secondaryFsm)
                if fsm is not None:
                    stateObj = fsm.getCurrentState()
                    if stateObj is not None:
                        state = '%s, %s' % (state, stateObj.getName())
                if hasattr(obj, 'doId'):
                    doId = f' doId:{obj.doId}'
            string = ':%s:%s [%-7s] id(%s)%s %s' % (self.getOnlyTime(), self.__name, state, id(obj), doId, PythonUtil.traceParentCall())
            self.__log(string)
            self.__print(string)
        return 1

    def debugCall(self, debugString: object='') -> int:
        if False:
            while True:
                i = 10
        '\n        If this notify is in debug mode, print the time of the\n        call followed by the notifier category and\n        the function call (with parameters).\n        '
        if __debug__ and self.__debug:
            message = str(debugString)
            string = ':%s:%s "%s" %s' % (self.getOnlyTime(), self.__name, message, PythonUtil.traceParentCall())
            self.__log(string)
            self.__print(string)
        return 1