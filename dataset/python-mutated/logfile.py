"""
A rotating, browsable log file.
"""
import glob
import os
import stat
import time
from typing import BinaryIO, Optional, cast
from twisted.python import threadable

class BaseLogFile:
    """
    The base class for a log file that can be rotated.
    """
    synchronized = ['write', 'rotate']

    def __init__(self, name: str, directory: str, defaultMode: Optional[int]=None) -> None:
        if False:
            while True:
                i = 10
        '\n        Create a log file.\n\n        @param name: name of the file\n        @param directory: directory holding the file\n        @param defaultMode: permissions used to create the file. Default to\n        current permissions of the file if the file exists.\n        '
        self.directory = directory
        self.name = name
        self.path = os.path.join(directory, name)
        if defaultMode is None and os.path.exists(self.path):
            self.defaultMode: Optional[int] = stat.S_IMODE(os.stat(self.path)[stat.ST_MODE])
        else:
            self.defaultMode = defaultMode
        self._openFile()

    @classmethod
    def fromFullPath(cls, filename, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Construct a log file from a full file path.\n        '
        logPath = os.path.abspath(filename)
        return cls(os.path.basename(logPath), os.path.dirname(logPath), *args, **kwargs)

    def shouldRotate(self):
        if False:
            while True:
                i = 10
        '\n        Override with a method to that returns true if the log\n        should be rotated.\n        '
        raise NotImplementedError

    def _openFile(self):
        if False:
            while True:
                i = 10
        '\n        Open the log file.\n\n        The log file is always opened in binary mode.\n        '
        self.closed = False
        if os.path.exists(self.path):
            self._file = cast(BinaryIO, open(self.path, 'rb+', 0))
            self._file.seek(0, 2)
        elif self.defaultMode is not None:
            oldUmask = os.umask(511)
            try:
                self._file = cast(BinaryIO, open(self.path, 'wb+', 0))
            finally:
                os.umask(oldUmask)
        else:
            self._file = cast(BinaryIO, open(self.path, 'wb+', 0))
        if self.defaultMode is not None:
            try:
                os.chmod(self.path, self.defaultMode)
            except OSError:
                pass

    def write(self, data):
        if False:
            print('Hello World!')
        '\n        Write some data to the file.\n\n        @param data: The data to write.  Text will be encoded as UTF-8.\n        @type data: L{bytes} or L{unicode}\n        '
        if self.shouldRotate():
            self.flush()
            self.rotate()
        if isinstance(data, str):
            data = data.encode('utf8')
        self._file.write(data)

    def flush(self):
        if False:
            i = 10
            return i + 15
        '\n        Flush the file.\n        '
        self._file.flush()

    def close(self):
        if False:
            i = 10
            return i + 15
        '\n        Close the file.\n\n        The file cannot be used once it has been closed.\n        '
        self.closed = True
        self._file.close()
        del self._file

    def reopen(self):
        if False:
            print('Hello World!')
        "\n        Reopen the log file. This is mainly useful if you use an external log\n        rotation tool, which moves under your feet.\n\n        Note that on Windows you probably need a specific API to rename the\n        file, as it's not supported to simply use os.rename, for example.\n        "
        self.close()
        self._openFile()

    def getCurrentLog(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a LogReader for the current log file.\n        '
        return LogReader(self.path)

class LogFile(BaseLogFile):
    """
    A log file that can be rotated.

    A rotateLength of None disables automatic log rotation.
    """

    def __init__(self, name, directory, rotateLength=1000000, defaultMode=None, maxRotatedFiles=None):
        if False:
            i = 10
            return i + 15
        '\n        Create a log file rotating on length.\n\n        @param name: file name.\n        @type name: C{str}\n        @param directory: path of the log file.\n        @type directory: C{str}\n        @param rotateLength: size of the log file where it rotates. Default to\n            1M.\n        @type rotateLength: C{int}\n        @param defaultMode: mode used to create the file.\n        @type defaultMode: C{int}\n        @param maxRotatedFiles: if not None, max number of log files the class\n            creates. Warning: it removes all log files above this number.\n        @type maxRotatedFiles: C{int}\n        '
        BaseLogFile.__init__(self, name, directory, defaultMode)
        self.rotateLength = rotateLength
        self.maxRotatedFiles = maxRotatedFiles

    def _openFile(self):
        if False:
            print('Hello World!')
        BaseLogFile._openFile(self)
        self.size = self._file.tell()

    def shouldRotate(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Rotate when the log file size is larger than rotateLength.\n        '
        return self.rotateLength and self.size >= self.rotateLength

    def getLog(self, identifier):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given an integer, return a LogReader for an old log file.\n        '
        filename = '%s.%d' % (self.path, identifier)
        if not os.path.exists(filename):
            raise ValueError('no such logfile exists')
        return LogReader(filename)

    def write(self, data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Write some data to the file.\n        '
        BaseLogFile.write(self, data)
        self.size += len(data)

    def rotate(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Rotate the file and create a new one.\n\n        If it's not possible to open new logfile, this will fail silently,\n        and continue logging to old logfile.\n        "
        if not (os.access(self.directory, os.W_OK) and os.access(self.path, os.W_OK)):
            return
        logs = self.listLogs()
        logs.reverse()
        for i in logs:
            if self.maxRotatedFiles is not None and i >= self.maxRotatedFiles:
                os.remove('%s.%d' % (self.path, i))
            else:
                os.rename('%s.%d' % (self.path, i), '%s.%d' % (self.path, i + 1))
        self._file.close()
        os.rename(self.path, '%s.1' % self.path)
        self._openFile()

    def listLogs(self):
        if False:
            return 10
        "\n        Return sorted list of integers - the old logs' identifiers.\n        "
        result = []
        for name in glob.glob('%s.*' % self.path):
            try:
                counter = int(name.split('.')[-1])
                if counter:
                    result.append(counter)
            except ValueError:
                pass
        result.sort()
        return result

    def __getstate__(self):
        if False:
            print('Hello World!')
        state = BaseLogFile.__getstate__(self)
        del state['size']
        return state
threadable.synchronize(LogFile)

class DailyLogFile(BaseLogFile):
    """A log file that is rotated daily (at or after midnight localtime)"""

    def _openFile(self):
        if False:
            return 10
        BaseLogFile._openFile(self)
        self.lastDate = self.toDate(os.stat(self.path)[8])

    def shouldRotate(self):
        if False:
            i = 10
            return i + 15
        'Rotate when the date has changed since last write'
        return self.toDate() > self.lastDate

    def toDate(self, *args):
        if False:
            return 10
        'Convert a unixtime to (year, month, day) localtime tuple,\n        or return the current (year, month, day) localtime tuple.\n\n        This function primarily exists so you may overload it with\n        gmtime, or some cruft to make unit testing possible.\n        '
        return time.localtime(*args)[:3]

    def suffix(self, tupledate):
        if False:
            print('Hello World!')
        'Return the suffix given a (year, month, day) tuple or unixtime'
        try:
            return '_'.join(map(str, tupledate))
        except BaseException:
            return '_'.join(map(str, self.toDate(tupledate)))

    def getLog(self, identifier):
        if False:
            for i in range(10):
                print('nop')
        'Given a unix time, return a LogReader for an old log file.'
        if self.toDate(identifier) == self.lastDate:
            return self.getCurrentLog()
        filename = f'{self.path}.{self.suffix(identifier)}'
        if not os.path.exists(filename):
            raise ValueError('no such logfile exists')
        return LogReader(filename)

    def write(self, data):
        if False:
            for i in range(10):
                print('nop')
        'Write some data to the log file'
        BaseLogFile.write(self, data)
        self.lastDate = max(self.lastDate, self.toDate())

    def rotate(self):
        if False:
            while True:
                i = 10
        "Rotate the file and create a new one.\n\n        If it's not possible to open new logfile, this will fail silently,\n        and continue logging to old logfile.\n        "
        if not (os.access(self.directory, os.W_OK) and os.access(self.path, os.W_OK)):
            return
        newpath = f'{self.path}.{self.suffix(self.lastDate)}'
        if os.path.exists(newpath):
            return
        self._file.close()
        os.rename(self.path, newpath)
        self._openFile()

    def __getstate__(self):
        if False:
            while True:
                i = 10
        state = BaseLogFile.__getstate__(self)
        del state['lastDate']
        return state
threadable.synchronize(DailyLogFile)

class LogReader:
    """Read from a log file."""

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        '\n        Open the log file for reading.\n\n        The comments about binary-mode for L{BaseLogFile._openFile} also apply\n        here.\n        '
        self._file = open(name)

    def readLines(self, lines=10):
        if False:
            return 10
        "Read a list of lines from the log file.\n\n        This doesn't returns all of the files lines - call it multiple times.\n        "
        result = []
        for i in range(lines):
            line = self._file.readline()
            if not line:
                break
            result.append(line)
        return result

    def close(self):
        if False:
            i = 10
            return i + 15
        self._file.close()