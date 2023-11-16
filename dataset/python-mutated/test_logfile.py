from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase

class LogFileTests(TestCase):
    """
    Test the rotating log file.
    """

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        self.dir = self.mktemp()
        os.makedirs(self.dir)
        self.name = 'test.log'
        self.path = os.path.join(self.dir, self.name)

    def tearDown(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Restore back write rights on created paths: if tests modified the\n        rights, that will allow the paths to be removed easily afterwards.\n        '
        os.chmod(self.dir, 511)
        if os.path.exists(self.path):
            os.chmod(self.path, 511)

    def test_abstractShouldRotate(self) -> None:
        if False:
            while True:
                i = 10
        '\n        L{BaseLogFile.shouldRotate} is abstract and must be implemented by\n        subclass.\n        '
        log = logfile.BaseLogFile(self.name, self.dir)
        self.addCleanup(log.close)
        self.assertRaises(NotImplementedError, log.shouldRotate)

    def test_writing(self) -> None:
        if False:
            return 10
        '\n        Log files can be written to, flushed and closed. Closing a log file\n        also flushes it.\n        '
        with contextlib.closing(logfile.LogFile(self.name, self.dir)) as log:
            log.write('123')
            log.write('456')
            log.flush()
            log.write('7890')
        with open(self.path) as f:
            self.assertEqual(f.read(), '1234567890')

    def test_rotation(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Rotating log files autorotate after a period of time, and can also be\n        manually rotated.\n        '
        with contextlib.closing(logfile.LogFile(self.name, self.dir, rotateLength=10)) as log:
            log.write('123')
            log.write('4567890')
            log.write('1' * 11)
            self.assertTrue(os.path.exists(f'{self.path}.1'))
            self.assertFalse(os.path.exists(f'{self.path}.2'))
            log.write('')
            self.assertTrue(os.path.exists(f'{self.path}.1'))
            self.assertTrue(os.path.exists(f'{self.path}.2'))
            self.assertFalse(os.path.exists(f'{self.path}.3'))
            log.write('3')
            self.assertFalse(os.path.exists(f'{self.path}.3'))
            log.rotate()
            self.assertTrue(os.path.exists(f'{self.path}.3'))
            self.assertFalse(os.path.exists(f'{self.path}.4'))
        self.assertEqual(log.listLogs(), [1, 2, 3])

    def test_append(self) -> None:
        if False:
            print('Hello World!')
        '\n        Log files can be written to, closed. Their size is the number of\n        bytes written to them. Everything that was written to them can\n        be read, even if the writing happened on separate occasions,\n        and even if the log file was closed in between.\n        '
        with contextlib.closing(logfile.LogFile(self.name, self.dir)) as log:
            log.write('0123456789')
        log = logfile.LogFile(self.name, self.dir)
        self.addCleanup(log.close)
        self.assertEqual(log.size, 10)
        self.assertEqual(log._file.tell(), log.size)
        log.write('abc')
        self.assertEqual(log.size, 13)
        self.assertEqual(log._file.tell(), log.size)
        f = log._file
        f.seek(0, 0)
        self.assertEqual(f.read(), b'0123456789abc')

    def test_logReader(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Various tests for log readers.\n\n        First of all, log readers can get logs by number and read what\n        was written to those log files. Getting nonexistent log files\n        raises C{ValueError}. Using anything other than an integer\n        index raises C{TypeError}. As logs get older, their log\n        numbers increase.\n        '
        log = logfile.LogFile(self.name, self.dir)
        self.addCleanup(log.close)
        log.write('abc\n')
        log.write('def\n')
        log.rotate()
        log.write('ghi\n')
        log.flush()
        self.assertEqual(log.listLogs(), [1])
        with contextlib.closing(log.getCurrentLog()) as reader:
            reader._file.seek(0)
            self.assertEqual(reader.readLines(), ['ghi\n'])
            self.assertEqual(reader.readLines(), [])
        with contextlib.closing(log.getLog(1)) as reader:
            self.assertEqual(reader.readLines(), ['abc\n', 'def\n'])
            self.assertEqual(reader.readLines(), [])
        self.assertRaises(ValueError, log.getLog, 2)
        self.assertRaises(TypeError, log.getLog, '1')
        log.rotate()
        self.assertEqual(log.listLogs(), [1, 2])
        with contextlib.closing(log.getLog(1)) as reader:
            reader._file.seek(0)
            self.assertEqual(reader.readLines(), ['ghi\n'])
            self.assertEqual(reader.readLines(), [])
        with contextlib.closing(log.getLog(2)) as reader:
            self.assertEqual(reader.readLines(), ['abc\n', 'def\n'])
            self.assertEqual(reader.readLines(), [])

    def test_LogReaderReadsZeroLine(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{LogReader.readLines} supports reading no line.\n        '
        with open(self.path, 'w'):
            pass
        reader = logfile.LogReader(self.path)
        self.addCleanup(reader.close)
        self.assertEqual([], reader.readLines(0))

    def test_modePreservation(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Check rotated files have same permissions as original.\n        '
        open(self.path, 'w').close()
        os.chmod(self.path, 455)
        mode = os.stat(self.path)[stat.ST_MODE]
        log = logfile.LogFile(self.name, self.dir)
        self.addCleanup(log.close)
        log.write('abc')
        log.rotate()
        self.assertEqual(mode, os.stat(self.path)[stat.ST_MODE])

    def test_noPermission(self) -> None:
        if False:
            return 10
        '\n        Check it keeps working when permission on dir changes.\n        '
        log = logfile.LogFile(self.name, self.dir)
        self.addCleanup(log.close)
        log.write('abc')
        os.chmod(self.dir, 365)
        try:
            f = open(os.path.join(self.dir, 'xxx'), 'w')
        except OSError:
            pass
        else:
            f.close()
            return
        log.rotate()
        log.write('def')
        log.flush()
        f = log._file
        self.assertEqual(f.tell(), 6)
        f.seek(0, 0)
        self.assertEqual(f.read(), b'abcdef')

    def test_maxNumberOfLog(self) -> None:
        if False:
            return 10
        '\n        Test it respect the limit on the number of files when maxRotatedFiles\n        is not None.\n        '
        log = logfile.LogFile(self.name, self.dir, rotateLength=10, maxRotatedFiles=3)
        self.addCleanup(log.close)
        log.write('1' * 11)
        log.write('2' * 11)
        self.assertTrue(os.path.exists(f'{self.path}.1'))
        log.write('3' * 11)
        self.assertTrue(os.path.exists(f'{self.path}.2'))
        log.write('4' * 11)
        self.assertTrue(os.path.exists(f'{self.path}.3'))
        with open(f'{self.path}.3') as fp:
            self.assertEqual(fp.read(), '1' * 11)
        log.write('5' * 11)
        with open(f'{self.path}.3') as fp:
            self.assertEqual(fp.read(), '2' * 11)
        self.assertFalse(os.path.exists(f'{self.path}.4'))

    def test_fromFullPath(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test the fromFullPath method.\n        '
        log1 = logfile.LogFile(self.name, self.dir, 10, defaultMode=511)
        self.addCleanup(log1.close)
        log2 = logfile.LogFile.fromFullPath(self.path, 10, defaultMode=511)
        self.addCleanup(log2.close)
        self.assertEqual(log1.name, log2.name)
        self.assertEqual(os.path.abspath(log1.path), log2.path)
        self.assertEqual(log1.rotateLength, log2.rotateLength)
        self.assertEqual(log1.defaultMode, log2.defaultMode)

    def test_defaultPermissions(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test the default permission of the log file: if the file exist, it\n        should keep the permission.\n        '
        with open(self.path, 'wb'):
            os.chmod(self.path, 455)
            currentMode = stat.S_IMODE(os.stat(self.path)[stat.ST_MODE])
        log1 = logfile.LogFile(self.name, self.dir)
        self.assertEqual(stat.S_IMODE(os.stat(self.path)[stat.ST_MODE]), currentMode)
        self.addCleanup(log1.close)

    def test_specifiedPermissions(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test specifying the permissions used on the log file.\n        '
        log1 = logfile.LogFile(self.name, self.dir, defaultMode=54)
        self.addCleanup(log1.close)
        mode = stat.S_IMODE(os.stat(self.path)[stat.ST_MODE])
        if runtime.platform.isWindows():
            self.assertEqual(mode, 292)
        else:
            self.assertEqual(mode, 54)

    @skipIf(runtime.platform.isWindows(), "Can't test reopen on Windows")
    def test_reopen(self) -> None:
        if False:
            while True:
                i = 10
        '\n        L{logfile.LogFile.reopen} allows to rename the currently used file and\n        make L{logfile.LogFile} create a new file.\n        '
        with contextlib.closing(logfile.LogFile(self.name, self.dir)) as log1:
            log1.write('hello1')
            savePath = os.path.join(self.dir, 'save.log')
            os.rename(self.path, savePath)
            log1.reopen()
            log1.write('hello2')
        with open(self.path) as f:
            self.assertEqual(f.read(), 'hello2')
        with open(savePath) as f:
            self.assertEqual(f.read(), 'hello1')

    def test_nonExistentDir(self) -> None:
        if False:
            print('Hello World!')
        '\n        Specifying an invalid directory to L{LogFile} raises C{IOError}.\n        '
        e = self.assertRaises(IOError, logfile.LogFile, self.name, 'this_dir_does_not_exist')
        self.assertEqual(e.errno, errno.ENOENT)

    def test_cantChangeFileMode(self) -> None:
        if False:
            while True:
                i = 10
        "\n        Opening a L{LogFile} which can be read and write but whose mode can't\n        be changed doesn't trigger an error.\n        "
        if runtime.platform.isWindows():
            (name, directory) = ('NUL', '')
            expectedPath = 'NUL'
        else:
            (name, directory) = ('null', '/dev')
            expectedPath = '/dev/null'
        log = logfile.LogFile(name, directory, defaultMode=365)
        self.addCleanup(log.close)
        self.assertEqual(log.path, expectedPath)
        self.assertEqual(log.defaultMode, 365)

    def test_listLogsWithBadlyNamedFiles(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        L{LogFile.listLogs} doesn't choke if it encounters a file with an\n        unexpected name.\n        "
        log = logfile.LogFile(self.name, self.dir)
        self.addCleanup(log.close)
        with open(f'{log.path}.1', 'w') as fp:
            fp.write('123')
        with open(f'{log.path}.bad-file', 'w') as fp:
            fp.write('123')
        self.assertEqual([1], log.listLogs())

    def test_listLogsIgnoresZeroSuffixedFiles(self) -> None:
        if False:
            print('Hello World!')
        '\n        L{LogFile.listLogs} ignores log files which rotated suffix is 0.\n        '
        log = logfile.LogFile(self.name, self.dir)
        self.addCleanup(log.close)
        for i in range(0, 3):
            with open(f'{log.path}.{i}', 'w') as fp:
                fp.write('123')
        self.assertEqual([1, 2], log.listLogs())

class RiggedDailyLogFile(logfile.DailyLogFile):
    _clock = 0.0

    def _openFile(self) -> None:
        if False:
            i = 10
            return i + 15
        logfile.DailyLogFile._openFile(self)
        self.lastDate = self.toDate()

    def toDate(self, *args: float) -> tuple[int, int, int]:
        if False:
            i = 10
            return i + 15
        if args:
            return time.gmtime(*args)[:3]
        return time.gmtime(self._clock)[:3]

class DailyLogFileTests(TestCase):
    """
    Test rotating log file.
    """

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        self.dir = self.mktemp()
        os.makedirs(self.dir)
        self.name = 'testdaily.log'
        self.path = os.path.join(self.dir, self.name)

    def test_writing(self) -> None:
        if False:
            return 10
        '\n        A daily log file can be written to like an ordinary log file.\n        '
        with contextlib.closing(RiggedDailyLogFile(self.name, self.dir)) as log:
            log.write('123')
            log.write('456')
            log.flush()
            log.write('7890')
        with open(self.path) as f:
            self.assertEqual(f.read(), '1234567890')

    def test_rotation(self) -> None:
        if False:
            return 10
        '\n        Daily log files rotate daily.\n        '
        log = RiggedDailyLogFile(self.name, self.dir)
        self.addCleanup(log.close)
        days = [self.path + '.' + log.suffix(day * 86400) for day in range(3)]
        log._clock = 0.0
        log.write('123')
        log._clock = 43200
        log.write('4567890')
        log._clock = 86400
        log.write('1' * 11)
        self.assertTrue(os.path.exists(days[0]))
        self.assertFalse(os.path.exists(days[1]))
        log._clock = 172800
        log.write('')
        self.assertTrue(os.path.exists(days[0]))
        self.assertTrue(os.path.exists(days[1]))
        self.assertFalse(os.path.exists(days[2]))
        log._clock = 259199
        log.write('3')
        self.assertFalse(os.path.exists(days[2]))

    def test_getLog(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test retrieving log files with L{DailyLogFile.getLog}.\n        '
        data = ['1\n', '2\n', '3\n']
        log = RiggedDailyLogFile(self.name, self.dir)
        self.addCleanup(log.close)
        for d in data:
            log.write(d)
        log.flush()
        r = log.getLog(0.0)
        self.addCleanup(r.close)
        self.assertEqual(data, r.readLines())
        self.assertRaises(ValueError, log.getLog, 86400)
        log._clock = 86401
        r.close()
        log.rotate()
        r = log.getLog(0)
        self.addCleanup(r.close)
        self.assertEqual(data, r.readLines())

    def test_rotateAlreadyExists(self) -> None:
        if False:
            while True:
                i = 10
        "\n        L{DailyLogFile.rotate} doesn't do anything if they new log file already\n        exists on the disk.\n        "
        log = RiggedDailyLogFile(self.name, self.dir)
        self.addCleanup(log.close)
        newFilePath = f'{log.path}.{log.suffix(log.lastDate)}'
        with open(newFilePath, 'w') as fp:
            fp.write('123')
        previousFile = log._file
        log.rotate()
        self.assertEqual(previousFile, log._file)

    @skipIf(runtime.platform.isWindows(), 'Making read-only directories on Windows is too complex for this test to reasonably do.')
    def test_rotatePermissionDirectoryNotOk(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        L{DailyLogFile.rotate} doesn't do anything if the directory containing\n        the log files can't be written to.\n        "
        log = logfile.DailyLogFile(self.name, self.dir)
        self.addCleanup(log.close)
        os.chmod(log.directory, 292)
        self.addCleanup(os.chmod, log.directory, 493)
        previousFile = log._file
        log.rotate()
        self.assertEqual(previousFile, log._file)

    def test_rotatePermissionFileNotOk(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        L{DailyLogFile.rotate} doesn't do anything if the log file can't be\n        written to.\n        "
        log = logfile.DailyLogFile(self.name, self.dir)
        self.addCleanup(log.close)
        os.chmod(log.path, 292)
        previousFile = log._file
        log.rotate()
        self.assertEqual(previousFile, log._file)

    def test_toDate(self) -> None:
        if False:
            return 10
        '\n        Test that L{DailyLogFile.toDate} converts its timestamp argument to a\n        time tuple (year, month, day).\n        '
        log = logfile.DailyLogFile(self.name, self.dir)
        self.addCleanup(log.close)
        timestamp = time.mktime((2000, 1, 1, 0, 0, 0, 0, 0, 0))
        self.assertEqual((2000, 1, 1), log.toDate(timestamp))

    def test_toDateDefaultToday(self) -> None:
        if False:
            print('Hello World!')
        "\n        Test that L{DailyLogFile.toDate} returns today's date by default.\n\n        By mocking L{time.localtime}, we ensure that L{DailyLogFile.toDate}\n        returns the first 3 values of L{time.localtime} which is the current\n        date.\n\n        Note that we don't compare the *real* result of L{DailyLogFile.toDate}\n        to the *real* current date, as there's a slight possibility that the\n        date changes between the 2 function calls.\n        "

        def mock_localtime(*args: object) -> list[int]:
            if False:
                return 10
            self.assertEqual((), args)
            return list(range(0, 9))
        log = logfile.DailyLogFile(self.name, self.dir)
        self.addCleanup(log.close)
        self.patch(time, 'localtime', mock_localtime)
        logDate = log.toDate()
        self.assertEqual([0, 1, 2], logDate)

    def test_toDateUsesArgumentsToMakeADate(self) -> None:
        if False:
            return 10
        '\n        Test that L{DailyLogFile.toDate} uses its arguments to create a new\n        date.\n        '
        log = logfile.DailyLogFile(self.name, self.dir)
        self.addCleanup(log.close)
        date = (2014, 10, 22)
        seconds = time.mktime(date + (0,) * 6)
        logDate = log.toDate(seconds)
        self.assertEqual(date, logDate)