"""Logger module: contains the logger class which creates and writes
   data to log files on disk"""
from __future__ import annotations
import io
import time
import math

class Logger:

    def __init__(self, fileName: str='log') -> None:
        if False:
            i = 10
            return i + 15
        '\n        Logger constructor\n        '
        self.__timeStamp = True
        self.__startTime = 0.0
        self.__logFile: io.TextIOWrapper | None = None
        self.__logFileName = fileName

    def setTimeStamp(self, enable: bool) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Toggle time stamp printing with log entries on and off\n        '
        self.__timeStamp = enable

    def getTimeStamp(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Return whether or not we are printing time stamps with log entries\n        '
        return self.__timeStamp

    def resetStartTime(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Reset the start time of the log file for time stamps\n        '
        self.__startTime = time.time()

    def log(self, entryString: str) -> None:
        if False:
            while True:
                i = 10
        'log(self, string)\n        Print the given string to the log file'
        if self.__logFile is None:
            self.__openLogFile()
        assert self.__logFile is not None
        if self.__timeStamp:
            self.__logFile.write(self.__getTimeStamp())
        self.__logFile.write(entryString + '\n')

    def __openLogFile(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Open a file for logging error/warning messages\n        '
        self.resetStartTime()
        t = time.localtime(self.__startTime)
        st = time.strftime('%m-%d-%Y-%H-%M-%S', t)
        logFileName = self.__logFileName + '.' + st
        self.__logFile = open(logFileName, 'w')

    def __closeLogFile(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Close the error/warning output file\n        '
        if self.__logFile is not None:
            self.__logFile.close()

    def __getTimeStamp(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Return the offset between current time and log file startTime\n        '
        t = time.time()
        dt = t - self.__startTime
        (days, dt) = divmod(dt, 86400)
        (hours, dt) = divmod(dt, 3600)
        (minutes, dt) = divmod(dt, 60)
        seconds = int(math.ceil(dt))
        return '%02d:%02d:%02d:%02d: ' % (days, hours, minutes, seconds)