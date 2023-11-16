"""
Tests for L{twisted.conch.scripts.cftp}.
"""
import getpass
import locale
import operator
import os
import struct
import sys
import time
from io import BytesIO, TextIOWrapper
from unittest import skipIf
from zope.interface import implementer
from twisted.conch import ls
from twisted.conch.interfaces import ISFTPFile
from twisted.conch.test.test_filetransfer import FileTransferTestAvatar, SFTPTestBase
from twisted.cred import portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport
from twisted.internet.utils import getProcessOutputAndValue, getProcessValue
from twisted.python import log
from twisted.python.fakepwd import UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.procutils import which
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
cryptography = requireModule('cryptography')
unix = requireModule('twisted.conch.unix')
if cryptography:
    try:
        from twisted.conch.scripts import cftp
        from twisted.conch.scripts.cftp import SSHSession
        from twisted.conch.ssh import filetransfer
        from twisted.conch.ssh.connection import EXTENDED_DATA_STDERR
        from twisted.conch.test import test_conch, test_ssh
        from twisted.conch.test.test_conch import FakeStdio
        from twisted.conch.test.test_filetransfer import FileTransferForTestAvatar
    except ImportError:
        pass
skipTests = False
if None in (unix, cryptography, interfaces.IReactorProcess(reactor, None)):
    skipTests = True

@skipIf(skipTests, "don't run w/o spawnProcess or cryptography")
class SSHSessionTests(TestCase):
    """
    Tests for L{twisted.conch.scripts.cftp.SSHSession}.
    """

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.stdio = FakeStdio()
        self.channel = SSHSession()
        self.channel.stdio = self.stdio
        self.stderrBuffer = BytesIO()
        self.stderr = TextIOWrapper(self.stderrBuffer)
        self.channel.stderr = self.stderr

    def test_eofReceived(self) -> None:
        if False:
            print('Hello World!')
        '\n        L{twisted.conch.scripts.cftp.SSHSession.eofReceived} loses the write\n        half of its stdio connection.\n        '
        self.channel.eofReceived()
        self.assertTrue(self.stdio.writeConnLost)

    def test_extReceivedStderr(self) -> None:
        if False:
            while True:
                i = 10
        '\n        L{twisted.conch.scripts.cftp.SSHSession.extReceived} decodes\n        stderr data using UTF-8 with the "backslashescape" error handling and\n        writes the result to its own stderr.\n        '
        errorText = '☃'
        errorBytes = errorText.encode('utf-8')
        self.channel.extReceived(EXTENDED_DATA_STDERR, errorBytes + b'\xff')
        self.assertEqual(self.stderrBuffer.getvalue(), errorBytes + b'\\xff')

class ListingTests(TestCase):
    """
    Tests for L{lsLine}, the function which generates an entry for a file or
    directory in an SFTP I{ls} command's output.
    """
    if getattr(time, 'tzset', None) is None:
        skip = 'Cannot test timestamp formatting code without time.tzset'

    def setUp(self):
        if False:
            return 10
        "\n        Patch the L{ls} module's time function so the results of L{lsLine} are\n        deterministic.\n        "
        self.now = 123456789

        def fakeTime():
            if False:
                print('Hello World!')
            return self.now
        self.patch(ls, 'time', fakeTime)
        if 'TZ' in os.environ:
            self.addCleanup(operator.setitem, os.environ, 'TZ', os.environ['TZ'])
            self.addCleanup(time.tzset)
        else:

            def cleanup():
                if False:
                    i = 10
                    return i + 15
                try:
                    del os.environ['TZ']
                except KeyError:
                    pass
                time.tzset()
            self.addCleanup(cleanup)

    def _lsInTimezone(self, timezone, stat):
        if False:
            print('Hello World!')
        '\n        Call L{ls.lsLine} after setting the timezone to C{timezone} and return\n        the result.\n        '
        os.environ['TZ'] = timezone
        time.tzset()
        return ls.lsLine('foo', stat)

    def test_oldFile(self):
        if False:
            print('Hello World!')
        '\n        A file with an mtime six months (approximately) or more in the past has\n        a listing including a low-resolution timestamp.\n        '
        then = self.now - 60 * 60 * 24 * 31 * 7
        stat = os.stat_result((0, 0, 0, 0, 0, 0, 0, 0, then, 0))
        self.assertEqual(self._lsInTimezone('America/New_York', stat), '!---------    0 0        0               0 Apr 26  1973 foo')
        self.assertEqual(self._lsInTimezone('Pacific/Auckland', stat), '!---------    0 0        0               0 Apr 27  1973 foo')

    def test_oldSingleDigitDayOfMonth(self):
        if False:
            i = 10
            return i + 15
        '\n        A file with a high-resolution timestamp which falls on a day of the\n        month which can be represented by one decimal digit is formatted with\n        one padding 0 to preserve the columns which come after it.\n        '
        then = self.now - 60 * 60 * 24 * 31 * 7 + 60 * 60 * 24 * 5
        stat = os.stat_result((0, 0, 0, 0, 0, 0, 0, 0, then, 0))
        self.assertEqual(self._lsInTimezone('America/New_York', stat), '!---------    0 0        0               0 May 01  1973 foo')
        self.assertEqual(self._lsInTimezone('Pacific/Auckland', stat), '!---------    0 0        0               0 May 02  1973 foo')

    def test_newFile(self):
        if False:
            return 10
        '\n        A file with an mtime fewer than six months (approximately) in the past\n        has a listing including a high-resolution timestamp excluding the year.\n        '
        then = self.now - 60 * 60 * 24 * 31 * 3
        stat = os.stat_result((0, 0, 0, 0, 0, 0, 0, 0, then, 0))
        self.assertEqual(self._lsInTimezone('America/New_York', stat), '!---------    0 0        0               0 Aug 28 17:33 foo')
        self.assertEqual(self._lsInTimezone('Pacific/Auckland', stat), '!---------    0 0        0               0 Aug 29 09:33 foo')
    currentLocale = locale.getlocale()
    try:
        try:
            locale.setlocale(locale.LC_ALL, 'es_AR.UTF8')
        except locale.Error:
            localeSkip = True
        else:
            localeSkip = False
    finally:
        locale.setlocale(locale.LC_ALL, currentLocale)

    @skipIf(localeSkip, 'The es_AR.UTF8 locale is not installed.')
    def test_localeIndependent(self):
        if False:
            return 10
        '\n        The month name in the date is locale independent.\n        '
        then = self.now - 60 * 60 * 24 * 31 * 3
        stat = os.stat_result((0, 0, 0, 0, 0, 0, 0, 0, then, 0))
        currentLocale = locale.getlocale()
        locale.setlocale(locale.LC_ALL, 'es_AR.UTF8')
        self.addCleanup(locale.setlocale, locale.LC_ALL, currentLocale)
        self.assertEqual(self._lsInTimezone('America/New_York', stat), '!---------    0 0        0               0 Aug 28 17:33 foo')
        self.assertEqual(self._lsInTimezone('Pacific/Auckland', stat), '!---------    0 0        0               0 Aug 29 09:33 foo')

    def test_newSingleDigitDayOfMonth(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A file with a high-resolution timestamp which falls on a day of the\n        month which can be represented by one decimal digit is formatted with\n        one padding 0 to preserve the columns which come after it.\n        '
        then = self.now - 60 * 60 * 24 * 31 * 3 + 60 * 60 * 24 * 4
        stat = os.stat_result((0, 0, 0, 0, 0, 0, 0, 0, then, 0))
        self.assertEqual(self._lsInTimezone('America/New_York', stat), '!---------    0 0        0               0 Sep 01 17:33 foo')
        self.assertEqual(self._lsInTimezone('Pacific/Auckland', stat), '!---------    0 0        0               0 Sep 02 09:33 foo')

class InMemorySSHChannel(StringTransport):
    """
    Minimal implementation of a L{SSHChannel} like class which only reads and
    writes data from memory.
    """

    def __init__(self, conn):
        if False:
            for i in range(10):
                print('nop')
        '\n        @param conn: The SSH connection associated with this channel.\n        @type conn: L{SSHConnection}\n        '
        self.conn = conn
        self.localClosed = 0
        super().__init__()

class FilesystemAccessExpectations:
    """
    A test helper used to support expected filesystem access.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._cache = {}

    def put(self, path, flags, stream):
        if False:
            return 10
        '\n\n        @param path: Path at which the stream is requested.\n        @type path: L{str}\n\n        @param path: Flags with which the stream is requested.\n        @type path: L{str}\n\n        @param stream: A stream.\n        @type stream: C{File}\n        '
        self._cache[path, flags] = stream

    def pop(self, path, flags):
        if False:
            while True:
                i = 10
        '\n        Remove a stream from the memory.\n\n        @param path: Path at which the stream is requested.\n        @type path: L{str}\n\n        @param path: Flags with which the stream is requested.\n        @type path: L{str}\n\n        @return: A stream.\n        @rtype: C{File}\n        '
        return self._cache.pop((path, flags))

class InMemorySFTPClient:
    """
    A L{filetransfer.FileTransferClient} which does filesystem operations in
    memory, without touching the local disc or the network interface.

    @ivar _availableFiles: File like objects which are available to the SFTP
        client.
    @type _availableFiles: L{FilesystemRegister}
    """

    def __init__(self, availableFiles):
        if False:
            i = 10
            return i + 15
        self.transport = InMemorySSHChannel(self)
        self.options = {'requests': 1, 'buffersize': 10}
        self._availableFiles = availableFiles

    def openFile(self, filename, flags, attrs):
        if False:
            while True:
                i = 10
        '\n        @see: L{filetransfer.FileTransferClient.openFile}.\n\n        Retrieve and remove cached file based on flags.\n        '
        return self._availableFiles.pop(filename, flags)

@implementer(ISFTPFile)
class InMemoryRemoteFile(BytesIO):
    """
    An L{ISFTPFile} which handles all data in memory.
    """

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        '\n        @param name: Name of this file.\n        @type name: L{str}\n        '
        self.name = name
        BytesIO.__init__(self)

    def writeChunk(self, start, data):
        if False:
            while True:
                i = 10
        '\n        @see: L{ISFTPFile.writeChunk}\n        '
        self.seek(start)
        self.write(data)
        return defer.succeed(self)

    def close(self):
        if False:
            i = 10
            return i + 15
        '\n        @see: L{ISFTPFile.writeChunk}\n\n        Keeps data after file was closed to help with testing.\n        '
        self._closed = True

    def getAttrs(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def readChunk(self, offset, length):
        if False:
            i = 10
            return i + 15
        pass

    def setAttrs(self, attrs):
        if False:
            print('Hello World!')
        pass

    def getvalue(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get current data of file.\n\n        Allow reading data event when file is closed.\n        '
        return BytesIO.getvalue(self)

@skipIf(skipTests, "don't run w/o spawnProcess or cryptography")
class StdioClientTests(TestCase):
    """
    Tests for L{cftp.StdioClient}.
    """

    def setUp(self):
        if False:
            return 10
        '\n        Create a L{cftp.StdioClient} hooked up to dummy transport and a fake\n        user database.\n        '
        self.fakeFilesystem = FilesystemAccessExpectations()
        sftpClient = InMemorySFTPClient(self.fakeFilesystem)
        self.client = cftp.StdioClient(sftpClient)
        self.client.currentDirectory = '/'
        self.database = self.client._pwd = UserDatabase()
        self.setKnownConsoleSize(500, 24)
        self.client.transport = self.client.client.transport

    def test_exec(self):
        if False:
            return 10
        "\n        The I{exec} command runs its arguments locally in a child process\n        using the user's shell.\n        "
        self.database.addUser(getpass.getuser(), 'secret', os.getuid(), 1234, 'foo', 'bar', sys.executable)
        d = self.client._dispatchCommand('exec print(1 + 2)')
        d.addCallback(self.assertEqual, b'3\n')
        return d

    def test_execWithoutShell(self):
        if False:
            i = 10
            return i + 15
        '\n        If the local user has no shell, the I{exec} command runs its arguments\n        using I{/bin/sh}.\n        '
        self.database.addUser(getpass.getuser(), 'secret', os.getuid(), 1234, 'foo', 'bar', '')
        d = self.client._dispatchCommand('exec echo hello')
        d.addCallback(self.assertEqual, b'hello\n')
        return d

    def test_bang(self):
        if False:
            print('Hello World!')
        '\n        The I{exec} command is run for lines which start with C{"!"}.\n        '
        self.database.addUser(getpass.getuser(), 'secret', os.getuid(), 1234, 'foo', 'bar', '/bin/sh')
        d = self.client._dispatchCommand('!echo hello')
        d.addCallback(self.assertEqual, b'hello\n')
        return d

    def setKnownConsoleSize(self, width, height):
        if False:
            i = 10
            return i + 15
        "\n        For the duration of this test, patch C{cftp}'s C{fcntl} module to return\n        a fixed width and height.\n\n        @param width: the width in characters\n        @type width: L{int}\n        @param height: the height in characters\n        @type height: L{int}\n        "
        import tty

        class FakeFcntl:

            def ioctl(self, fd, opt, mutate):
                if False:
                    for i in range(10):
                        print('nop')
                if opt != tty.TIOCGWINSZ:
                    self.fail('Only window-size queries supported.')
                return struct.pack('4H', height, width, 0, 0)
        self.patch(cftp, 'fcntl', FakeFcntl())

    def test_printProgressBarReporting(self):
        if False:
            print('Hello World!')
        "\n        L{StdioClient._printProgressBar} prints a progress description,\n        including percent done, amount transferred, transfer rate, and time\n        remaining, all based the given start time, the given L{FileWrapper}'s\n        progress information and the reactor's current time.\n        "
        self.setKnownConsoleSize(10, 34)
        clock = self.client.reactor = Clock()
        wrapped = BytesIO(b'x')
        wrapped.name = b'sample'
        wrapper = cftp.FileWrapper(wrapped)
        wrapper.size = 1024 * 10
        startTime = clock.seconds()
        clock.advance(2.0)
        wrapper.total += 4096
        self.client._printProgressBar(wrapper, startTime)
        result = b"\rb'sample' 40% 4.0kB 2.0kBps 00:03 "
        self.assertEqual(self.client.transport.value(), result)

    def test_printProgressBarNoProgress(self):
        if False:
            return 10
        '\n        L{StdioClient._printProgressBar} prints a progress description that\n        indicates 0 bytes transferred if no bytes have been transferred and no\n        time has passed.\n        '
        self.setKnownConsoleSize(10, 34)
        clock = self.client.reactor = Clock()
        wrapped = BytesIO(b'x')
        wrapped.name = b'sample'
        wrapper = cftp.FileWrapper(wrapped)
        startTime = clock.seconds()
        self.client._printProgressBar(wrapper, startTime)
        result = b"\rb'sample'  0% 0.0B 0.0Bps 00:00 "
        self.assertEqual(self.client.transport.value(), result)

    def test_printProgressBarEmptyFile(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Print the progress for empty files.\n        '
        self.setKnownConsoleSize(10, 34)
        wrapped = BytesIO()
        wrapped.name = b'empty-file'
        wrapper = cftp.FileWrapper(wrapped)
        self.client._printProgressBar(wrapper, 0)
        result = b"\rb'empty-file'100% 0.0B 0.0Bps 00:00 "
        self.assertEqual(result, self.client.transport.value())

    def test_getFilenameEmpty(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns empty value for both filename and remaining data.\n        '
        result = self.client._getFilename('  ')
        self.assertEqual(('', ''), result)

    def test_getFilenameOnlyLocal(self):
        if False:
            print('Hello World!')
        '\n        Returns empty value for remaining data when line contains\n        only a filename.\n        '
        result = self.client._getFilename('only-local')
        self.assertEqual(('only-local', ''), result)

    def test_getFilenameNotQuoted(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns filename and remaining data striped of leading and trailing\n        spaces.\n        '
        result = self.client._getFilename(' local  remote file  ')
        self.assertEqual(('local', 'remote file'), result)

    def test_getFilenameQuoted(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns filename and remaining data not striped of leading and trailing\n        spaces when quoted paths are requested.\n        '
        result = self.client._getFilename(' " local file "  " remote  file " ')
        self.assertEqual((' local file ', '" remote  file "'), result)

    def makeFile(self, path=None, content=b''):
        if False:
            while True:
                i = 10
        '\n        Create a local file and return its path.\n\n        When `path` is L{None}, it will create a new temporary file.\n\n        @param path: Optional path for the new file.\n        @type path: L{str}\n\n        @param content: Content to be written in the new file.\n        @type content: L{bytes}\n\n        @return: Path to the newly create file.\n        '
        if path is None:
            path = self.mktemp()
        with open(path, 'wb') as file:
            file.write(content)
        return path

    def checkPutMessage(self, transfers, randomOrder=False):
        if False:
            i = 10
            return i + 15
        '\n        Check output of cftp client for a put request.\n\n\n        @param transfers: List with tuple of (local, remote, progress).\n        @param randomOrder: When set to C{True}, it will ignore the order\n            in which put reposes are received\n\n        '
        output = self.client.transport.value()
        output = output.decode('utf-8')
        output = output.split('\n\r')
        expectedOutput = []
        actualOutput = []
        for (local, remote, expected) in transfers:
            expectedTransfer = []
            for line in expected:
                expectedTransfer.append(f'{local} {line}')
            expectedTransfer.append(f'Transferred {local} to {remote}')
            expectedOutput.append(expectedTransfer)
            progressParts = output.pop(0).strip('\r').split('\r')
            actual = progressParts[:-1]
            last = progressParts[-1].strip('\n').split('\n')
            actual.extend(last)
            actualTransfer = []
            for line in actual[:-1]:
                line = line.strip().rsplit(' ', 2)[0]
                line = line.strip().split(' ', 1)
                actualTransfer.append(f'{line[0]} {line[1].strip()}')
            actualTransfer.append(actual[-1])
            actualOutput.append(actualTransfer)
        if randomOrder:
            self.assertEqual(sorted(expectedOutput), sorted(actualOutput))
        else:
            self.assertEqual(expectedOutput, actualOutput)
        self.assertEqual(0, len(output), 'There are still put responses which were not checked.')

    def test_cmd_PUTSingleNoRemotePath(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A name based on local path is used when remote path is not\n        provided.\n\n        The progress is updated while chunks are transferred.\n        '
        content = b'Test\r\nContent'
        localPath = self.makeFile(content=content)
        flags = filetransfer.FXF_WRITE | filetransfer.FXF_CREAT | filetransfer.FXF_TRUNC
        remoteName = os.path.join('/', os.path.basename(localPath))
        remoteFile = InMemoryRemoteFile(remoteName)
        self.fakeFilesystem.put(remoteName, flags, defer.succeed(remoteFile))
        self.client.client.options['buffersize'] = 10
        deferred = self.client.cmd_PUT(localPath)
        self.successResultOf(deferred)
        self.assertEqual(content, remoteFile.getvalue())
        self.assertTrue(remoteFile._closed)
        self.checkPutMessage([(localPath, remoteName, ['76% 10.0B', '100% 13.0B', '100% 13.0B'])])

    def test_cmd_PUTSingleRemotePath(self):
        if False:
            return 10
        '\n        Remote path is extracted from first filename after local file.\n\n        Any other data in the line is ignored.\n        '
        localPath = self.makeFile()
        flags = filetransfer.FXF_WRITE | filetransfer.FXF_CREAT | filetransfer.FXF_TRUNC
        remoteName = '/remote-path'
        remoteFile = InMemoryRemoteFile(remoteName)
        self.fakeFilesystem.put(remoteName, flags, defer.succeed(remoteFile))
        deferred = self.client.cmd_PUT(f'{localPath} {remoteName} ignored')
        self.successResultOf(deferred)
        self.checkPutMessage([(localPath, remoteName, ['100% 0.0B'])])
        self.assertTrue(remoteFile._closed)
        self.assertEqual(b'', remoteFile.getvalue())

    def test_cmd_PUTMultipleNoRemotePath(self):
        if False:
            print('Hello World!')
        '\n        When a gobbing expression is used local files are transferred with\n        remote file names based on local names.\n        '
        first = self.makeFile()
        firstName = os.path.basename(first)
        secondName = 'second-name'
        parent = os.path.dirname(first)
        second = self.makeFile(path=os.path.join(parent, secondName))
        flags = filetransfer.FXF_WRITE | filetransfer.FXF_CREAT | filetransfer.FXF_TRUNC
        firstRemotePath = f'/{firstName}'
        secondRemotePath = f'/{secondName}'
        firstRemoteFile = InMemoryRemoteFile(firstRemotePath)
        secondRemoteFile = InMemoryRemoteFile(secondRemotePath)
        self.fakeFilesystem.put(firstRemotePath, flags, defer.succeed(firstRemoteFile))
        self.fakeFilesystem.put(secondRemotePath, flags, defer.succeed(secondRemoteFile))
        deferred = self.client.cmd_PUT(os.path.join(parent, '*'))
        self.successResultOf(deferred)
        self.assertTrue(firstRemoteFile._closed)
        self.assertEqual(b'', firstRemoteFile.getvalue())
        self.assertTrue(secondRemoteFile._closed)
        self.assertEqual(b'', secondRemoteFile.getvalue())
        self.checkPutMessage([(first, firstRemotePath, ['100% 0.0B']), (second, secondRemotePath, ['100% 0.0B'])], randomOrder=True)

    def test_cmd_PUTMultipleWithRemotePath(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When a gobbing expression is used local files are transferred with\n        remote file names based on local names.\n        when a remote folder is requested remote paths are composed from\n        remote path and local filename.\n        '
        first = self.makeFile()
        firstName = os.path.basename(first)
        secondName = 'second-name'
        parent = os.path.dirname(first)
        second = self.makeFile(path=os.path.join(parent, secondName))
        flags = filetransfer.FXF_WRITE | filetransfer.FXF_CREAT | filetransfer.FXF_TRUNC
        firstRemoteFile = InMemoryRemoteFile(firstName)
        secondRemoteFile = InMemoryRemoteFile(secondName)
        firstRemotePath = f'/remote/{firstName}'
        secondRemotePath = f'/remote/{secondName}'
        self.fakeFilesystem.put(firstRemotePath, flags, defer.succeed(firstRemoteFile))
        self.fakeFilesystem.put(secondRemotePath, flags, defer.succeed(secondRemoteFile))
        deferred = self.client.cmd_PUT('{} remote'.format(os.path.join(parent, '*')))
        self.successResultOf(deferred)
        self.assertTrue(firstRemoteFile._closed)
        self.assertEqual(b'', firstRemoteFile.getvalue())
        self.assertTrue(secondRemoteFile._closed)
        self.assertEqual(b'', secondRemoteFile.getvalue())
        self.checkPutMessage([(first, firstName, ['100% 0.0B']), (second, secondName, ['100% 0.0B'])], randomOrder=True)

class FileTransferTestRealm:

    def __init__(self, testDir):
        if False:
            for i in range(10):
                print('nop')
        self.testDir = testDir

    def requestAvatar(self, avatarID, mind, *interfaces):
        if False:
            print('Hello World!')
        a = FileTransferTestAvatar(self.testDir)
        return (interfaces[0], a, lambda : None)

class SFTPTestProcess(protocol.ProcessProtocol):
    """
    Protocol for testing cftp. Provides an interface between Python (where all
    the tests are) and the cftp client process (which does the work that is
    being tested).
    """

    def __init__(self, onOutReceived):
        if False:
            print('Hello World!')
        '\n        @param onOutReceived: A L{Deferred} to be fired as soon as data is\n        received from stdout.\n        '
        self.clearBuffer()
        self.onOutReceived = onOutReceived
        self.onProcessEnd = None
        self._expectingCommand = None
        self._processEnded = False

    def clearBuffer(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Clear any buffered data received from stdout. Should be private.\n        '
        self.buffer = b''
        self._linesReceived = []
        self._lineBuffer = b''

    def outReceived(self, data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called by Twisted when the cftp client prints data to stdout.\n        '
        log.msg('got %r' % data)
        lines = (self._lineBuffer + data).split(b'\n')
        self._lineBuffer = lines.pop(-1)
        self._linesReceived.extend(lines)
        if self.onOutReceived is not None:
            (d, self.onOutReceived) = (self.onOutReceived, None)
            d.callback(data)
        self.buffer += data
        self._checkForCommand()

    def _checkForCommand(self):
        if False:
            return 10
        prompt = b'cftp> '
        if self._expectingCommand and self._lineBuffer == prompt:
            buf = b'\n'.join(self._linesReceived)
            if buf.startswith(prompt):
                buf = buf[len(prompt):]
            self.clearBuffer()
            (d, self._expectingCommand) = (self._expectingCommand, None)
            d.callback(buf)

    def errReceived(self, data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called by Twisted when the cftp client prints data to stderr.\n        '
        log.msg('err: %s' % data)

    def getBuffer(self):
        if False:
            print('Hello World!')
        '\n        Return the contents of the buffer of data received from stdout.\n        '
        return self.buffer

    def runCommand(self, command):
        if False:
            while True:
                i = 10
        "\n        Issue the given command via the cftp client. Return a C{Deferred} that\n        fires when the server returns a result. Note that the C{Deferred} will\n        callback even if the server returns some kind of error.\n\n        @param command: A string containing an sftp command.\n\n        @return: A C{Deferred} that fires when the sftp server returns a\n        result. The payload is the server's response string.\n        "
        self._expectingCommand = defer.Deferred()
        self.clearBuffer()
        if isinstance(command, str):
            command = command.encode('utf-8')
        self.transport.write(command + b'\n')
        return self._expectingCommand

    def runScript(self, commands):
        if False:
            while True:
                i = 10
        '\n        Run each command in sequence and return a Deferred that fires when all\n        commands are completed.\n\n        @param commands: A list of strings containing sftp commands.\n\n        @return: A C{Deferred} that fires when all commands are completed. The\n        payload is a list of response strings from the server, in the same\n        order as the commands.\n        '
        sem = defer.DeferredSemaphore(1)
        dl = [sem.run(self.runCommand, command) for command in commands]
        return defer.gatherResults(dl)

    def killProcess(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Kill the process if it is still running.\n\n        If the process is still running, sends a KILL signal to the transport\n        and returns a C{Deferred} which fires when L{processEnded} is called.\n\n        @return: a C{Deferred}.\n        '
        if self._processEnded:
            return defer.succeed(None)
        self.onProcessEnd = defer.Deferred()
        self.transport.signalProcess('KILL')
        return self.onProcessEnd

    def processEnded(self, reason):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called by Twisted when the cftp client process ends.\n        '
        self._processEnded = True
        if self.onProcessEnd:
            (d, self.onProcessEnd) = (self.onProcessEnd, None)
            d.callback(None)

class CFTPClientTestBase(SFTPTestBase):

    def setUp(self):
        if False:
            print('Hello World!')
        with open('dsa_test.pub', 'wb') as f:
            f.write(test_ssh.publicDSA_openssh)
        with open('dsa_test', 'wb') as f:
            f.write(test_ssh.privateDSA_openssh)
        os.chmod('dsa_test', 33152)
        with open('kh_test', 'wb') as f:
            f.write(b'127.0.0.1 ' + test_ssh.publicRSA_openssh)
        return SFTPTestBase.setUp(self)

    def startServer(self):
        if False:
            for i in range(10):
                print('nop')
        realm = FileTransferTestRealm(self.testDir)
        p = portal.Portal(realm)
        p.registerChecker(test_ssh.conchTestPublicKeyChecker())
        fac = test_ssh.ConchTestServerFactory()
        fac.portal = p
        self.server = reactor.listenTCP(0, fac, interface='127.0.0.1')

    def stopServer(self):
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(self.server.factory, 'proto'):
            return self._cbStopServer(None)
        self.server.factory.proto.expectedLoseConnection = 1
        d = defer.maybeDeferred(self.server.factory.proto.transport.loseConnection)
        d.addCallback(self._cbStopServer)
        return d

    def _cbStopServer(self, ignored):
        if False:
            for i in range(10):
                print('nop')
        return defer.maybeDeferred(self.server.stopListening)

    def tearDown(self):
        if False:
            print('Hello World!')
        for f in ['dsa_test.pub', 'dsa_test', 'kh_test']:
            try:
                os.remove(f)
            except BaseException:
                pass
        return SFTPTestBase.tearDown(self)

@skipIf(skipTests, "don't run w/o spawnProcess or cryptography")
class OurServerCmdLineClientTests(CFTPClientTestBase):
    """
    Functional tests which launch a SFTP server over TCP on localhost and check
    cftp command line interface using a spawned process.

    Due to the spawned process you can not add a debugger breakpoint for the
    client code.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        CFTPClientTestBase.setUp(self)
        self.startServer()
        cmds = '-p %i -l testuser --known-hosts kh_test --user-authentications publickey --host-key-algorithms ssh-rsa -i dsa_test -a -v 127.0.0.1'
        port = self.server.getHost().port
        cmds = test_conch._makeArgs((cmds % port).split(), mod='cftp')
        log.msg(f'running {sys.executable} {cmds}')
        d = defer.Deferred()
        self.processProtocol = SFTPTestProcess(d)
        d.addCallback(lambda _: self.processProtocol.clearBuffer())
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(sys.path)
        encodedCmds = []
        encodedEnv = {}
        for cmd in cmds:
            if isinstance(cmd, str):
                cmd = cmd.encode('utf-8')
            encodedCmds.append(cmd)
        for var in env:
            val = env[var]
            if isinstance(var, str):
                var = var.encode('utf-8')
            if isinstance(val, str):
                val = val.encode('utf-8')
            encodedEnv[var] = val
        log.msg(encodedCmds)
        log.msg(encodedEnv)
        reactor.spawnProcess(self.processProtocol, sys.executable, encodedCmds, env=encodedEnv)
        return d

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        d = self.stopServer()
        d.addCallback(lambda _: self.processProtocol.killProcess())
        return d

    def _killProcess(self, ignored):
        if False:
            return 10
        try:
            self.processProtocol.transport.signalProcess('KILL')
        except error.ProcessExitedAlready:
            pass

    def runCommand(self, command):
        if False:
            print('Hello World!')
        "\n        Run the given command with the cftp client. Return a C{Deferred} that\n        fires when the command is complete. Payload is the server's output for\n        that command.\n        "
        return self.processProtocol.runCommand(command)

    def runScript(self, *commands):
        if False:
            while True:
                i = 10
        "\n        Run the given commands with the cftp client. Returns a C{Deferred}\n        that fires when the commands are all complete. The C{Deferred}'s\n        payload is a list of output for each command.\n        "
        return self.processProtocol.runScript(commands)

    def testCdPwd(self):
        if False:
            print('Hello World!')
        "\n        Test that 'pwd' reports the current remote directory, that 'lpwd'\n        reports the current local directory, and that changing to a\n        subdirectory then changing to its parent leaves you in the original\n        remote directory.\n        "
        homeDir = self.testDir
        d = self.runScript('pwd', 'lpwd', 'cd testDirectory', 'cd ..', 'pwd')

        def cmdOutput(output):
            if False:
                print('Hello World!')
            '\n            Callback function for handling command output.\n            '
            cmds = []
            for cmd in output:
                if isinstance(cmd, bytes):
                    cmd = cmd.decode('utf-8')
                cmds.append(cmd)
            return cmds[:3] + cmds[4:]
        d.addCallback(cmdOutput)
        d.addCallback(self.assertEqual, [homeDir.path, os.getcwd(), '', homeDir.path])
        return d

    def testChAttrs(self):
        if False:
            print('Hello World!')
        "\n        Check that 'ls -l' output includes the access permissions and that\n        this output changes appropriately with 'chmod'.\n        "

        def _check(results):
            if False:
                i = 10
                return i + 15
            self.flushLoggedErrors()
            self.assertTrue(results[0].startswith(b'-rw-r--r--'))
            self.assertEqual(results[1], b'')
            self.assertTrue(results[2].startswith(b'----------'), results[2])
            self.assertEqual(results[3], b'')
        d = self.runScript('ls -l testfile1', 'chmod 0 testfile1', 'ls -l testfile1', 'chmod 644 testfile1')
        return d.addCallback(_check)

    def testList(self):
        if False:
            while True:
                i = 10
        "\n        Check 'ls' works as expected. Checks for wildcards, hidden files,\n        listing directories and listing empty directories.\n        "

        def _check(results):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(results[0], [b'testDirectory', b'testRemoveFile', b'testRenameFile', b'testfile1'])
            self.assertEqual(results[1], [b'testDirectory', b'testRemoveFile', b'testRenameFile', b'testfile1'])
            self.assertEqual(results[2], [b'testRemoveFile', b'testRenameFile'])
            self.assertEqual(results[3], [b'.testHiddenFile', b'testRemoveFile', b'testRenameFile'])
            self.assertEqual(results[4], [b''])
        d = self.runScript('ls', 'ls ../' + self.testDir.basename(), 'ls *File', 'ls -a *File', 'ls -l testDirectory')
        d.addCallback(lambda xs: [x.split(b'\n') for x in xs])
        return d.addCallback(_check)

    def testHelp(self):
        if False:
            print('Hello World!')
        "\n        Check that running the '?' command returns help.\n        "
        d = self.runCommand('?')
        helpText = cftp.StdioClient(None).cmd_HELP('').strip()
        if isinstance(helpText, str):
            helpText = helpText.encode('utf-8')
        d.addCallback(self.assertEqual, helpText)
        return d

    def assertFilesEqual(self, name1, name2, msg=None):
        if False:
            return 10
        '\n        Assert that the files at C{name1} and C{name2} contain exactly the\n        same data.\n        '
        self.assertEqual(name1.getContent(), name2.getContent(), msg)

    def testGet(self):
        if False:
            i = 10
            return i + 15
        "\n        Test that 'get' saves the remote file to the correct local location,\n        that the output of 'get' is correct and that 'rm' actually removes\n        the file.\n        "
        expectedOutput = 'Transferred {}/testfile1 to {}/test file2'.format(self.testDir.path, self.testDir.path)
        if isinstance(expectedOutput, str):
            expectedOutput = expectedOutput.encode('utf-8')

        def _checkGet(result):
            if False:
                for i in range(10):
                    print('nop')
            self.assertTrue(result.endswith(expectedOutput))
            self.assertFilesEqual(self.testDir.child('testfile1'), self.testDir.child('test file2'), 'get failed')
            return self.runCommand('rm "test file2"')
        d = self.runCommand(f'get testfile1 "{self.testDir.path}/test file2"')
        d.addCallback(_checkGet)
        d.addCallback(lambda _: self.assertFalse(self.testDir.child('test file2').exists()))
        return d

    def testWildcardGet(self):
        if False:
            return 10
        "\n        Test that 'get' works correctly when given wildcard parameters.\n        "

        def _check(ignored):
            if False:
                for i in range(10):
                    print('nop')
            self.assertFilesEqual(self.testDir.child('testRemoveFile'), FilePath('testRemoveFile'), 'testRemoveFile get failed')
            self.assertFilesEqual(self.testDir.child('testRenameFile'), FilePath('testRenameFile'), 'testRenameFile get failed')
        d = self.runCommand('get testR*')
        return d.addCallback(_check)

    def testPut(self):
        if False:
            i = 10
            return i + 15
        "\n        Check that 'put' uploads files correctly and that they can be\n        successfully removed. Also check the output of the put command.\n        "
        expectedOutput = b'Transferred ' + self.testDir.asBytesMode().path + b'/testfile1 to ' + self.testDir.asBytesMode().path + b'/test"file2'

        def _checkPut(result):
            if False:
                i = 10
                return i + 15
            self.assertFilesEqual(self.testDir.child('testfile1'), self.testDir.child('test"file2'))
            self.assertTrue(result.endswith(expectedOutput))
            return self.runCommand('rm "test\\"file2"')
        d = self.runCommand(f'put {self.testDir.path}/testfile1 "test\\"file2"')
        d.addCallback(_checkPut)
        d.addCallback(lambda _: self.assertFalse(self.testDir.child('test"file2').exists()))
        return d

    def test_putOverLongerFile(self):
        if False:
            return 10
        "\n        Check that 'put' uploads files correctly when overwriting a longer\n        file.\n        "
        with self.testDir.child('shorterFile').open(mode='w') as f:
            f.write(b'a')
        with self.testDir.child('longerFile').open(mode='w') as f:
            f.write(b'bb')

        def _checkPut(result):
            if False:
                print('Hello World!')
            self.assertFilesEqual(self.testDir.child('shorterFile'), self.testDir.child('longerFile'))
        d = self.runCommand(f'put {self.testDir.path}/shorterFile longerFile')
        d.addCallback(_checkPut)
        return d

    def test_putMultipleOverLongerFile(self):
        if False:
            print('Hello World!')
        "\n        Check that 'put' uploads files correctly when overwriting a longer\n        file and you use a wildcard to specify the files to upload.\n        "
        someDir = self.testDir.child('dir')
        someDir.createDirectory()
        with someDir.child('file').open(mode='w') as f:
            f.write(b'a')
        with self.testDir.child('file').open(mode='w') as f:
            f.write(b'bb')

        def _checkPut(result):
            if False:
                i = 10
                return i + 15
            self.assertFilesEqual(someDir.child('file'), self.testDir.child('file'))
        d = self.runCommand(f'put {self.testDir.path}/dir/*')
        d.addCallback(_checkPut)
        return d

    def testWildcardPut(self):
        if False:
            while True:
                i = 10
        "\n        What happens if you issue a 'put' command and include a wildcard (i.e.\n        '*') in parameter? Check that all files matching the wildcard are\n        uploaded to the correct directory.\n        "

        def check(results):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(results[0], b'')
            self.assertEqual(results[2], b'')
            self.assertFilesEqual(self.testDir.child('testRemoveFile'), self.testDir.parent().child('testRemoveFile'), 'testRemoveFile get failed')
            self.assertFilesEqual(self.testDir.child('testRenameFile'), self.testDir.parent().child('testRenameFile'), 'testRenameFile get failed')
        d = self.runScript('cd ..', f'put {self.testDir.path}/testR*', 'cd %s' % self.testDir.basename())
        d.addCallback(check)
        return d

    def testLink(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test that 'ln' creates a file which appears as a link in the output of\n        'ls'. Check that removing the new file succeeds without output.\n        "

        def _check(results):
            if False:
                print('Hello World!')
            self.flushLoggedErrors()
            self.assertEqual(results[0], b'')
            self.assertTrue(results[1].startswith(b'l'), 'link failed')
            return self.runCommand('rm testLink')
        d = self.runScript('ln testLink testfile1', 'ls -l testLink')
        d.addCallback(_check)
        d.addCallback(self.assertEqual, b'')
        return d

    def testRemoteDirectory(self):
        if False:
            while True:
                i = 10
        '\n        Test that we can create and remove directories with the cftp client.\n        '

        def _check(results):
            if False:
                print('Hello World!')
            self.assertEqual(results[0], b'')
            self.assertTrue(results[1].startswith(b'd'))
            return self.runCommand('rmdir testMakeDirectory')
        d = self.runScript('mkdir testMakeDirectory', 'ls -l testMakeDirector?')
        d.addCallback(_check)
        d.addCallback(self.assertEqual, b'')
        return d

    def test_existingRemoteDirectory(self):
        if False:
            print('Hello World!')
        "\n        Test that a C{mkdir} on an existing directory fails with the\n        appropriate error, and doesn't log an useless error server side.\n        "

        def _check(results):
            if False:
                return 10
            self.assertEqual(results[0], b'')
            self.assertEqual(results[1], b'remote error 11: mkdir failed')
        d = self.runScript('mkdir testMakeDirectory', 'mkdir testMakeDirectory')
        d.addCallback(_check)
        return d

    def testLocalDirectory(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test that we can create a directory locally and remove it with the\n        cftp client. This test works because the 'remote' server is running\n        out of a local directory.\n        "
        d = self.runCommand(f'lmkdir {self.testDir.path}/testLocalDirectory')
        d.addCallback(self.assertEqual, b'')
        d.addCallback(lambda _: self.runCommand('rmdir testLocalDirectory'))
        d.addCallback(self.assertEqual, b'')
        return d

    def testRename(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that we can rename a file.\n        '

        def _check(results):
            if False:
                return 10
            self.assertEqual(results[0], b'')
            self.assertEqual(results[1], b'testfile2')
            return self.runCommand('rename testfile2 testfile1')
        d = self.runScript('rename testfile1 testfile2', 'ls testfile?')
        d.addCallback(_check)
        d.addCallback(self.assertEqual, b'')
        return d

@skipIf(skipTests, "don't run w/o spawnProcess or cryptography")
class OurServerBatchFileTests(CFTPClientTestBase):
    """
    Functional tests which launch a SFTP server over localhost and checks csftp
    in batch interface.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        CFTPClientTestBase.setUp(self)
        self.startServer()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        CFTPClientTestBase.tearDown(self)
        return self.stopServer()

    def _getBatchOutput(self, f):
        if False:
            for i in range(10):
                print('nop')
        fn = self.mktemp()
        with open(fn, 'w') as fp:
            fp.write(f)
        port = self.server.getHost().port
        cmds = '-p %i -l testuser --known-hosts kh_test --user-authentications publickey --host-key-algorithms ssh-rsa -i dsa_test -a -v -b %s 127.0.0.1' % (port, fn)
        cmds = test_conch._makeArgs(cmds.split(), mod='cftp')[1:]
        log.msg(f'running {sys.executable} {cmds}')
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(sys.path)
        self.server.factory.expectedLoseConnection = 1
        d = getProcessOutputAndValue(sys.executable, cmds, env=env)

        def _cleanup(res):
            if False:
                i = 10
                return i + 15
            os.remove(fn)
            return res
        d.addCallback(lambda res: res[0])
        d.addBoth(_cleanup)
        return d

    def testBatchFile(self):
        if False:
            i = 10
            return i + 15
        "\n        Test whether batch file function of cftp ('cftp -b batchfile').\n        This works by treating the file as a list of commands to be run.\n        "
        cmds = 'pwd\nls\nexit\n'

        def _cbCheckResult(res):
            if False:
                for i in range(10):
                    print('nop')
            res = res.split(b'\n')
            log.msg('RES %s' % repr(res))
            self.assertIn(self.testDir.asBytesMode().path, res[1])
            self.assertEqual(res[3:-2], [b'testDirectory', b'testRemoveFile', b'testRenameFile', b'testfile1'])
        d = self._getBatchOutput(cmds)
        d.addCallback(_cbCheckResult)
        return d

    def testError(self):
        if False:
            print('Hello World!')
        '\n        Test that an error in the batch file stops running the batch.\n        '
        cmds = 'chown 0 missingFile\npwd\nexit\n'

        def _cbCheckResult(res):
            if False:
                return 10
            self.assertNotIn(self.testDir.asBytesMode().path, res)
        d = self._getBatchOutput(cmds)
        d.addCallback(_cbCheckResult)
        return d

    def testIgnoredError(self):
        if False:
            while True:
                i = 10
        "\n        Test that a minus sign '-' at the front of a line ignores\n        any errors.\n        "
        cmds = '-chown 0 missingFile\npwd\nexit\n'

        def _cbCheckResult(res):
            if False:
                return 10
            self.assertIn(self.testDir.asBytesMode().path, res)
        d = self._getBatchOutput(cmds)
        d.addCallback(_cbCheckResult)
        return d

@skipIf(skipTests, "don't run w/o spawnProcess or cryptography")
@skipIf(not which('ssh'), 'no ssh command-line client available')
@skipIf(not which('sftp'), 'no sftp command-line client available')
class OurServerSftpClientTests(CFTPClientTestBase):
    """
    Test the sftp server against sftp command line client.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        CFTPClientTestBase.setUp(self)
        return self.startServer()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        return self.stopServer()

    def test_extendedAttributes(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the return of extended attributes by the server: the sftp client\n        should ignore them, but still be able to parse the response correctly.\n\n        This test is mainly here to check that\n        L{filetransfer.FILEXFER_ATTR_EXTENDED} has the correct value.\n        '
        env = dict(os.environ)
        fn = self.mktemp()
        with open(fn, 'w') as f:
            f.write('ls .\nexit')
        port = self.server.getHost().port
        oldGetAttr = FileTransferForTestAvatar._getAttrs

        def _getAttrs(self, s):
            if False:
                while True:
                    i = 10
            attrs = oldGetAttr(self, s)
            attrs['ext_foo'] = 'bar'
            return attrs
        self.patch(FileTransferForTestAvatar, '_getAttrs', _getAttrs)
        self.server.factory.expectedLoseConnection = True
        d = getProcessValue('ssh', ('-o', 'PubkeyAcceptedKeyTypes=ssh-dss', '-V'), env)

        def hasPAKT(status):
            if False:
                i = 10
                return i + 15
            if status == 0:
                args = ('-o', 'PubkeyAcceptedKeyTypes=ssh-dss')
            else:
                args = ()
            args += ('-F', '/dev/null', '-o', 'IdentityFile=dsa_test', '-o', 'UserKnownHostsFile=kh_test', '-o', 'HostKeyAlgorithms=ssh-rsa', '-o', 'Port=%i' % (port,), '-b', fn, 'testuser@127.0.0.1')
            return args

        def check(result):
            if False:
                print('Hello World!')
            self.assertEqual(result[2], 0, result[1].decode('ascii'))
            for i in [b'testDirectory', b'testRemoveFile', b'testRenameFile', b'testfile1']:
                self.assertIn(i, result[0])
        d.addCallback(hasPAKT)
        d.addCallback(lambda args: getProcessOutputAndValue('sftp', args, env))
        return d.addCallback(check)