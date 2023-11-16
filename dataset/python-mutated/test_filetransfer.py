"""
Tests for L{twisted.conch.ssh.filetransfer}.
"""
import os
import re
import struct
from unittest import skipIf
from hamcrest import assert_that, equal_to
from twisted.internet import defer
from twisted.internet.error import ConnectionLost
from twisted.internet.testing import StringTransport
from twisted.protocols import loopback
from twisted.python import components
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
try:
    from twisted.conch import unix as _unix
except ImportError:
    unix = None
else:
    unix = _unix
try:
    from twisted.conch.unix import SFTPServerForUnixConchUser as _SFTPServerForUnixConchUser
except ImportError:
    SFTPServerForUnixConchUser = None
else:
    SFTPServerForUnixConchUser = _SFTPServerForUnixConchUser
try:
    import cryptography as _cryptography
except ImportError:
    cryptography = None
else:
    cryptography = _cryptography
try:
    from twisted.conch.avatar import ConchUser as _ConchUser
except ImportError:
    ConchUser = object
else:
    ConchUser = _ConchUser
try:
    from twisted.conch.ssh import common, connection, filetransfer, session
except ImportError:
    pass

class TestAvatar(ConchUser):

    def __init__(self):
        if False:
            print('Hello World!')
        ConchUser.__init__(self)
        self.channelLookup[b'session'] = session.SSHSession
        self.subsystemLookup[b'sftp'] = filetransfer.FileTransferServer

    def _runAsUser(self, f, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        try:
            f = iter(f)
        except TypeError:
            f = [(f, args, kw)]
        for i in f:
            func = i[0]
            args = len(i) > 1 and i[1] or ()
            kw = len(i) > 2 and i[2] or {}
            r = func(*args, **kw)
        return r

class FileTransferTestAvatar(TestAvatar):

    def __init__(self, homeDir):
        if False:
            print('Hello World!')
        TestAvatar.__init__(self)
        self.homeDir = homeDir

    def getHomeDir(self):
        if False:
            while True:
                i = 10
        return FilePath(os.getcwd()).preauthChild(self.homeDir.path)

class ConchSessionForTestAvatar:

    def __init__(self, avatar):
        if False:
            print('Hello World!')
        self.avatar = avatar
if SFTPServerForUnixConchUser is None:
    import warnings
    warnings.warn("twisted.conch.unix imported %r, but doesn't define SFTPServerForUnixConchUser'" % (unix,))
else:

    class FileTransferForTestAvatar(SFTPServerForUnixConchUser):

        def gotVersion(self, version, otherExt):
            if False:
                for i in range(10):
                    print('nop')
            return {b'conchTest': b'ext data'}

        def extendedRequest(self, extName, extData):
            if False:
                return 10
            if extName == b'testExtendedRequest':
                return b'bar'
            raise NotImplementedError
    components.registerAdapter(FileTransferForTestAvatar, TestAvatar, filetransfer.ISFTPServer)

class SFTPTestBase(TestCase):

    def setUp(self):
        if False:
            return 10
        self.testDir = FilePath(self.mktemp())
        self.testDir = self.testDir.child('extra')
        self.testDir.child('testDirectory').makedirs(True)
        with self.testDir.child('testfile1').open(mode='wb') as f:
            f.write(b'a' * 10 + b'b' * 10)
            with open('/dev/urandom', 'rb') as f2:
                f.write(f2.read(1024 * 64))
        self.testDir.child('testfile1').chmod(420)
        with self.testDir.child('testRemoveFile').open(mode='wb') as f:
            f.write(b'a')
        with self.testDir.child('testRenameFile').open(mode='wb') as f:
            f.write(b'a')
        with self.testDir.child('.testHiddenFile').open(mode='wb') as f:
            f.write(b'a')

@skipIf(not unix, "can't run on non-posix computers")
class OurServerOurClientTests(SFTPTestBase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        SFTPTestBase.setUp(self)
        self.avatar = FileTransferTestAvatar(self.testDir)
        self.server = filetransfer.FileTransferServer(avatar=self.avatar)
        clientTransport = loopback.LoopbackRelay(self.server)
        self.client = filetransfer.FileTransferClient()
        self._serverVersion = None
        self._extData = None

        def _(serverVersion, extData):
            if False:
                while True:
                    i = 10
            self._serverVersion = serverVersion
            self._extData = extData
        self.client.gotServerVersion = _
        serverTransport = loopback.LoopbackRelay(self.client)
        self.client.makeConnection(clientTransport)
        self.server.makeConnection(serverTransport)
        self.clientTransport = clientTransport
        self.serverTransport = serverTransport
        self._emptyBuffers()

    def _emptyBuffers(self):
        if False:
            for i in range(10):
                print('nop')
        while self.serverTransport.buffer or self.clientTransport.buffer:
            self.serverTransport.clearBuffer()
            self.clientTransport.clearBuffer()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.serverTransport.loseConnection()
        self.clientTransport.loseConnection()
        self.serverTransport.clearBuffer()
        self.clientTransport.clearBuffer()

    def test_serverVersion(self):
        if False:
            return 10
        self.assertEqual(self._serverVersion, 3)
        self.assertEqual(self._extData, {b'conchTest': b'ext data'})

    def test_interface_implementation(self):
        if False:
            while True:
                i = 10
        '\n        It implements the ISFTPServer interface.\n        '
        self.assertTrue(filetransfer.ISFTPServer.providedBy(self.server.client), f'ISFTPServer not provided by {self.server.client!r}')

    def test_openedFileClosedWithConnection(self):
        if False:
            i = 10
            return i + 15
        '\n        A file opened with C{openFile} is closed when the connection is lost.\n        '
        d = self.client.openFile(b'testfile1', filetransfer.FXF_READ | filetransfer.FXF_WRITE, {})
        self._emptyBuffers()
        oldClose = os.close
        closed = []

        def close(fd):
            if False:
                return 10
            closed.append(fd)
            oldClose(fd)
        self.patch(os, 'close', close)

        def _fileOpened(openFile):
            if False:
                for i in range(10):
                    print('nop')
            fd = self.server.openFiles[openFile.handle[4:]].fd
            self.serverTransport.loseConnection()
            self.clientTransport.loseConnection()
            self.serverTransport.clearBuffer()
            self.clientTransport.clearBuffer()
            self.assertEqual(self.server.openFiles, {})
            self.assertIn(fd, closed)
        d.addCallback(_fileOpened)
        return d

    def test_openedDirectoryClosedWithConnection(self):
        if False:
            while True:
                i = 10
        '\n        A directory opened with C{openDirectory} is close when the connection\n        is lost.\n        '
        d = self.client.openDirectory('')
        self._emptyBuffers()

        def _getFiles(openDir):
            if False:
                i = 10
                return i + 15
            self.serverTransport.loseConnection()
            self.clientTransport.loseConnection()
            self.serverTransport.clearBuffer()
            self.clientTransport.clearBuffer()
            self.assertEqual(self.server.openDirs, {})
        d.addCallback(_getFiles)
        return d

    def test_openFileIO(self):
        if False:
            for i in range(10):
                print('nop')
        d = self.client.openFile(b'testfile1', filetransfer.FXF_READ | filetransfer.FXF_WRITE, {})
        self._emptyBuffers()

        def _fileOpened(openFile):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(openFile, filetransfer.ISFTPFile(openFile))
            d = _readChunk(openFile)
            d.addCallback(_writeChunk, openFile)
            return d

        def _readChunk(openFile):
            if False:
                return 10
            d = openFile.readChunk(0, 20)
            self._emptyBuffers()
            d.addCallback(self.assertEqual, b'a' * 10 + b'b' * 10)
            return d

        def _writeChunk(_, openFile):
            if False:
                while True:
                    i = 10
            d = openFile.writeChunk(20, b'c' * 10)
            self._emptyBuffers()
            d.addCallback(_readChunk2, openFile)
            return d

        def _readChunk2(_, openFile):
            if False:
                i = 10
                return i + 15
            d = openFile.readChunk(0, 30)
            self._emptyBuffers()
            d.addCallback(self.assertEqual, b'a' * 10 + b'b' * 10 + b'c' * 10)
            return d
        d.addCallback(_fileOpened)
        return d

    def test_closedFileGetAttrs(self):
        if False:
            for i in range(10):
                print('nop')
        d = self.client.openFile(b'testfile1', filetransfer.FXF_READ | filetransfer.FXF_WRITE, {})
        self._emptyBuffers()

        def _getAttrs(_, openFile):
            if False:
                return 10
            d = openFile.getAttrs()
            self._emptyBuffers()
            return d

        def _err(f):
            if False:
                print('Hello World!')
            self.flushLoggedErrors()
            return f

        def _close(openFile):
            if False:
                print('Hello World!')
            d = openFile.close()
            self._emptyBuffers()
            d.addCallback(_getAttrs, openFile)
            d.addErrback(_err)
            return self.assertFailure(d, filetransfer.SFTPError)
        d.addCallback(_close)
        return d

    def test_openFileAttributes(self):
        if False:
            return 10
        d = self.client.openFile(b'testfile1', filetransfer.FXF_READ | filetransfer.FXF_WRITE, {})
        self._emptyBuffers()

        def _getAttrs(openFile):
            if False:
                print('Hello World!')
            d = openFile.getAttrs()
            self._emptyBuffers()
            d.addCallback(_getAttrs2)
            return d

        def _getAttrs2(attrs1):
            if False:
                return 10
            d = self.client.getAttrs(b'testfile1')
            self._emptyBuffers()
            d.addCallback(self.assertEqual, attrs1)
            return d
        return d.addCallback(_getAttrs)

    def test_openFileSetAttrs(self):
        if False:
            return 10
        d = self.client.openFile(b'testfile1', filetransfer.FXF_READ | filetransfer.FXF_WRITE, {})
        self._emptyBuffers()

        def _getAttrs(openFile):
            if False:
                while True:
                    i = 10
            d = openFile.getAttrs()
            self._emptyBuffers()
            d.addCallback(_setAttrs)
            return d

        def _setAttrs(attrs):
            if False:
                while True:
                    i = 10
            attrs['atime'] = 0
            d = self.client.setAttrs(b'testfile1', attrs)
            self._emptyBuffers()
            d.addCallback(_getAttrs2)
            d.addCallback(self.assertEqual, attrs)
            return d

        def _getAttrs2(_):
            if False:
                print('Hello World!')
            d = self.client.getAttrs(b'testfile1')
            self._emptyBuffers()
            return d
        d.addCallback(_getAttrs)
        return d

    def test_openFileExtendedAttributes(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check that L{filetransfer.FileTransferClient.openFile} can send\n        extended attributes, that should be extracted server side. By default,\n        they are ignored, so we just verify they are correctly parsed.\n        '
        savedAttributes = {}
        oldOpenFile = self.server.client.openFile

        def openFile(filename, flags, attrs):
            if False:
                while True:
                    i = 10
            savedAttributes.update(attrs)
            return oldOpenFile(filename, flags, attrs)
        self.server.client.openFile = openFile
        d = self.client.openFile(b'testfile1', filetransfer.FXF_READ | filetransfer.FXF_WRITE, {'ext_foo': b'bar'})
        self._emptyBuffers()

        def check(ign):
            if False:
                print('Hello World!')
            self.assertEqual(savedAttributes, {'ext_foo': b'bar'})
        return d.addCallback(check)

    def test_removeFile(self):
        if False:
            for i in range(10):
                print('nop')
        d = self.client.getAttrs(b'testRemoveFile')
        self._emptyBuffers()

        def _removeFile(ignored):
            if False:
                i = 10
                return i + 15
            d = self.client.removeFile(b'testRemoveFile')
            self._emptyBuffers()
            return d
        d.addCallback(_removeFile)
        d.addCallback(_removeFile)
        return self.assertFailure(d, filetransfer.SFTPError)

    def test_renameFile(self):
        if False:
            i = 10
            return i + 15
        d = self.client.getAttrs(b'testRenameFile')
        self._emptyBuffers()

        def _rename(attrs):
            if False:
                for i in range(10):
                    print('nop')
            d = self.client.renameFile(b'testRenameFile', b'testRenamedFile')
            self._emptyBuffers()
            d.addCallback(_testRenamed, attrs)
            return d

        def _testRenamed(_, attrs):
            if False:
                while True:
                    i = 10
            d = self.client.getAttrs(b'testRenamedFile')
            self._emptyBuffers()
            d.addCallback(self.assertEqual, attrs)
        return d.addCallback(_rename)

    def test_directoryBad(self):
        if False:
            return 10
        d = self.client.getAttrs(b'testMakeDirectory')
        self._emptyBuffers()
        return self.assertFailure(d, filetransfer.SFTPError)

    def test_directoryCreation(self):
        if False:
            while True:
                i = 10
        d = self.client.makeDirectory(b'testMakeDirectory', {})
        self._emptyBuffers()

        def _getAttrs(_):
            if False:
                i = 10
                return i + 15
            d = self.client.getAttrs(b'testMakeDirectory')
            self._emptyBuffers()
            return d

        def _removeDirectory(_):
            if False:
                i = 10
                return i + 15
            d = self.client.removeDirectory(b'testMakeDirectory')
            self._emptyBuffers()
            return d
        d.addCallback(_getAttrs)
        d.addCallback(_removeDirectory)
        d.addCallback(_getAttrs)
        return self.assertFailure(d, filetransfer.SFTPError)

    def test_openDirectory(self):
        if False:
            for i in range(10):
                print('nop')
        d = self.client.openDirectory(b'')
        self._emptyBuffers()
        files = []

        def _getFiles(openDir):
            if False:
                for i in range(10):
                    print('nop')

            def append(f):
                if False:
                    while True:
                        i = 10
                files.append(f)
                return openDir
            d = defer.maybeDeferred(openDir.next)
            self._emptyBuffers()
            d.addCallback(append)
            d.addCallback(_getFiles)
            d.addErrback(_close, openDir)
            return d

        def _checkFiles(ignored):
            if False:
                while True:
                    i = 10
            fs = list(list(zip(*files))[0])
            fs.sort()
            self.assertEqual(fs, [b'.testHiddenFile', b'testDirectory', b'testRemoveFile', b'testRenameFile', b'testfile1'])

        def _close(_, openDir):
            if False:
                print('Hello World!')
            d = openDir.close()
            self._emptyBuffers()
            return d
        d.addCallback(_getFiles)
        d.addCallback(_checkFiles)
        return d

    def test_linkDoesntExist(self):
        if False:
            print('Hello World!')
        d = self.client.getAttrs(b'testLink')
        self._emptyBuffers()
        return self.assertFailure(d, filetransfer.SFTPError)

    def test_linkSharesAttrs(self):
        if False:
            for i in range(10):
                print('nop')
        d = self.client.makeLink(b'testLink', b'testfile1')
        self._emptyBuffers()

        def _getFirstAttrs(_):
            if False:
                i = 10
                return i + 15
            d = self.client.getAttrs(b'testLink', 1)
            self._emptyBuffers()
            return d

        def _getSecondAttrs(firstAttrs):
            if False:
                return 10
            d = self.client.getAttrs(b'testfile1')
            self._emptyBuffers()
            d.addCallback(self.assertEqual, firstAttrs)
            return d
        d.addCallback(_getFirstAttrs)
        return d.addCallback(_getSecondAttrs)

    def test_linkPath(self):
        if False:
            while True:
                i = 10
        d = self.client.makeLink(b'testLink', b'testfile1')
        self._emptyBuffers()

        def _readLink(_):
            if False:
                print('Hello World!')
            d = self.client.readLink(b'testLink')
            self._emptyBuffers()
            testFile = FilePath(os.getcwd()).preauthChild(self.testDir.path)
            testFile = testFile.child('testfile1')
            d.addCallback(self.assertEqual, testFile.path)
            return d

        def _realPath(_):
            if False:
                for i in range(10):
                    print('nop')
            d = self.client.realPath(b'testLink')
            self._emptyBuffers()
            testLink = FilePath(os.getcwd()).preauthChild(self.testDir.path)
            testLink = testLink.child('testfile1')
            d.addCallback(self.assertEqual, testLink.path)
            return d
        d.addCallback(_readLink)
        d.addCallback(_realPath)
        return d

    def test_extendedRequest(self):
        if False:
            return 10
        d = self.client.extendedRequest(b'testExtendedRequest', b'foo')
        self._emptyBuffers()
        d.addCallback(self.assertEqual, b'bar')
        d.addCallback(self._cbTestExtendedRequest)
        return d

    def _cbTestExtendedRequest(self, ignored):
        if False:
            while True:
                i = 10
        d = self.client.extendedRequest(b'testBadRequest', b'')
        self._emptyBuffers()
        return self.assertFailure(d, NotImplementedError)

    @defer.inlineCallbacks
    def test_openDirectoryIteratorDeprecated(self):
        if False:
            return 10
        '\n        Using client.openDirectory as an iterator is deprecated.\n        '
        d = self.client.openDirectory(b'')
        self._emptyBuffers()
        openDir = (yield d)
        oneFile = openDir.next()
        self._emptyBuffers()
        yield oneFile
        warnings = self.flushWarnings()
        message = 'Using twisted.conch.ssh.filetransfer.ClientDirectory as an iterator was deprecated in Twisted 18.9.0.'
        self.assertEqual(1, len(warnings))
        self.assertEqual(DeprecationWarning, warnings[0]['category'])
        self.assertEqual(message, warnings[0]['message'])

    @defer.inlineCallbacks
    def test_closedConnectionCancelsRequests(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If there are requests outstanding when the connection\n        is closed for any reason, they should fail.\n        '
        d = self.client.openFile(b'testfile1', filetransfer.FXF_READ, {})
        self._emptyBuffers()
        fh = (yield d)
        gotReadRequest = []

        def _slowRead(offset, length):
            if False:
                print('Hello World!')
            self.assertEqual(gotReadRequest, [])
            d = defer.Deferred()
            gotReadRequest.append(offset)
            return d
        [serverSideFh] = self.server.openFiles.values()
        serverSideFh.readChunk = _slowRead
        del serverSideFh
        d = fh.readChunk(100, 200)
        self._emptyBuffers()
        self.assertEqual(len(gotReadRequest), 1)
        self.assertNoResult(d)
        self.serverTransport.loseConnection()
        self.serverTransport.clearBuffer()
        self.clientTransport.clearBuffer()
        self._emptyBuffers()
        self.assertFalse(self.client.connected)
        self.failureResultOf(d, ConnectionLost)
        d = fh.getAttrs()
        self.failureResultOf(d, ConnectionLost)

class FakeConn:

    def sendClose(self, channel):
        if False:
            while True:
                i = 10
        pass

@skipIf(not unix, "can't run on non-posix computers")
class FileTransferCloseTests(TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.avatar = TestAvatar()

    def buildServerConnection(self):
        if False:
            i = 10
            return i + 15
        conn = connection.SSHConnection()

        class DummyTransport:

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.transport = self

            def sendPacket(self, kind, data):
                if False:
                    return 10
                pass

            def logPrefix(self):
                if False:
                    i = 10
                    return i + 15
                return 'dummy transport'
        conn.transport = DummyTransport()
        conn.transport.avatar = self.avatar
        return conn

    def interceptConnectionLost(self, sftpServer):
        if False:
            return 10
        self.connectionLostFired = False
        origConnectionLost = sftpServer.connectionLost

        def connectionLost(reason):
            if False:
                i = 10
                return i + 15
            self.connectionLostFired = True
            origConnectionLost(reason)
        sftpServer.connectionLost = connectionLost

    def assertSFTPConnectionLost(self):
        if False:
            print('Hello World!')
        self.assertTrue(self.connectionLostFired, "sftpServer's connectionLost was not called")

    def test_sessionClose(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Closing a session should notify an SFTP subsystem launched by that\n        session.\n        '
        testSession = session.SSHSession(conn=FakeConn(), avatar=self.avatar)
        testSession.request_subsystem(common.NS(b'sftp'))
        sftpServer = testSession.client.transport.proto
        self.interceptConnectionLost(sftpServer)
        testSession.closeReceived()
        self.assertSFTPConnectionLost()

    def test_clientClosesChannelOnConnnection(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A client sending CHANNEL_CLOSE should trigger closeReceived on the\n        associated channel instance.\n        '
        conn = self.buildServerConnection()
        packet = common.NS(b'session') + struct.pack('>L', 0) * 3
        conn.ssh_CHANNEL_OPEN(packet)
        sessionChannel = conn.channels[0]
        sessionChannel.request_subsystem(common.NS(b'sftp'))
        sftpServer = sessionChannel.client.transport.proto
        self.interceptConnectionLost(sftpServer)
        self.interceptConnectionLost(sftpServer)
        conn.ssh_CHANNEL_CLOSE(struct.pack('>L', 0))
        self.assertSFTPConnectionLost()

    def test_stopConnectionServiceClosesChannel(self):
        if False:
            print('Hello World!')
        '\n        Closing an SSH connection should close all sessions within it.\n        '
        conn = self.buildServerConnection()
        packet = common.NS(b'session') + struct.pack('>L', 0) * 3
        conn.ssh_CHANNEL_OPEN(packet)
        sessionChannel = conn.channels[0]
        sessionChannel.request_subsystem(common.NS(b'sftp'))
        sftpServer = sessionChannel.client.transport.proto
        self.interceptConnectionLost(sftpServer)
        conn.serviceStopped()
        self.assertSFTPConnectionLost()

@skipIf(not cryptography, 'Cannot run without cryptography')
class ConstantsTests(TestCase):
    """
    Tests for the constants used by the SFTP protocol implementation.

    @ivar filexferSpecExcerpts: Excerpts from the
        draft-ietf-secsh-filexfer-02.txt (draft) specification of the SFTP
        protocol.  There are more recent drafts of the specification, but this
        one describes version 3, which is what conch (and OpenSSH) implements.
    """
    filexferSpecExcerpts = ["\n           The following values are defined for packet types.\n\n                #define SSH_FXP_INIT                1\n                #define SSH_FXP_VERSION             2\n                #define SSH_FXP_OPEN                3\n                #define SSH_FXP_CLOSE               4\n                #define SSH_FXP_READ                5\n                #define SSH_FXP_WRITE               6\n                #define SSH_FXP_LSTAT               7\n                #define SSH_FXP_FSTAT               8\n                #define SSH_FXP_SETSTAT             9\n                #define SSH_FXP_FSETSTAT           10\n                #define SSH_FXP_OPENDIR            11\n                #define SSH_FXP_READDIR            12\n                #define SSH_FXP_REMOVE             13\n                #define SSH_FXP_MKDIR              14\n                #define SSH_FXP_RMDIR              15\n                #define SSH_FXP_REALPATH           16\n                #define SSH_FXP_STAT               17\n                #define SSH_FXP_RENAME             18\n                #define SSH_FXP_READLINK           19\n                #define SSH_FXP_SYMLINK            20\n                #define SSH_FXP_STATUS            101\n                #define SSH_FXP_HANDLE            102\n                #define SSH_FXP_DATA              103\n                #define SSH_FXP_NAME              104\n                #define SSH_FXP_ATTRS             105\n                #define SSH_FXP_EXTENDED          200\n                #define SSH_FXP_EXTENDED_REPLY    201\n\n           Additional packet types should only be defined if the protocol\n           version number (see Section ``Protocol Initialization'') is\n           incremented, and their use MUST be negotiated using the version\n           number.  However, the SSH_FXP_EXTENDED and SSH_FXP_EXTENDED_REPLY\n           packets can be used to implement vendor-specific extensions.  See\n           Section ``Vendor-Specific-Extensions'' for more details.\n        ", '\n            The flags bits are defined to have the following values:\n\n                #define SSH_FILEXFER_ATTR_SIZE          0x00000001\n                #define SSH_FILEXFER_ATTR_UIDGID        0x00000002\n                #define SSH_FILEXFER_ATTR_PERMISSIONS   0x00000004\n                #define SSH_FILEXFER_ATTR_ACMODTIME     0x00000008\n                #define SSH_FILEXFER_ATTR_EXTENDED      0x80000000\n\n        ', "\n            The `pflags' field is a bitmask.  The following bits have been\n           defined.\n\n                #define SSH_FXF_READ            0x00000001\n                #define SSH_FXF_WRITE           0x00000002\n                #define SSH_FXF_APPEND          0x00000004\n                #define SSH_FXF_CREAT           0x00000008\n                #define SSH_FXF_TRUNC           0x00000010\n                #define SSH_FXF_EXCL            0x00000020\n        ", '\n            Currently, the following values are defined (other values may be\n           defined by future versions of this protocol):\n\n                #define SSH_FX_OK                            0\n                #define SSH_FX_EOF                           1\n                #define SSH_FX_NO_SUCH_FILE                  2\n                #define SSH_FX_PERMISSION_DENIED             3\n                #define SSH_FX_FAILURE                       4\n                #define SSH_FX_BAD_MESSAGE                   5\n                #define SSH_FX_NO_CONNECTION                 6\n                #define SSH_FX_CONNECTION_LOST               7\n                #define SSH_FX_OP_UNSUPPORTED                8\n        ']

    def test_constantsAgainstSpec(self):
        if False:
            while True:
                i = 10
        '\n        The constants used by the SFTP protocol implementation match those\n        found by searching through the spec.\n        '
        constants = {}
        for excerpt in self.filexferSpecExcerpts:
            for line in excerpt.splitlines():
                m = re.match('^\\s*#define SSH_([A-Z_]+)\\s+([0-9x]*)\\s*$', line)
                if m:
                    constants[m.group(1)] = int(m.group(2), 0)
        self.assertTrue(len(constants) > 0, 'No constants found (the test must be buggy).')
        for (k, v) in constants.items():
            self.assertEqual(v, getattr(filetransfer, k))

@skipIf(not unix, "can't run on non-posix computers")
@skipIf(not cryptography, 'Cannot run without cryptography')
class RawPacketDataServerTests(TestCase):
    """
    Tests for L{filetransfer.FileTransferServer} which explicitly craft
    certain less common situations to exercise their handling.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.fts = filetransfer.FileTransferServer(avatar=TestAvatar())

    def test_closeInvalidHandle(self):
        if False:
            i = 10
            return i + 15
        '\n        A close request with an unknown handle receives an FX_NO_SUCH_FILE error\n        response.\n        '
        transport = StringTransport()
        self.fts.makeConnection(transport)
        requestId = b'1234'
        handle = b'invalid handle'
        close = common.NS(bytes([4]) + requestId + common.NS(handle))
        self.fts.dataReceived(close)
        expected = common.NS(bytes([101]) + requestId + bytes([0, 0, 0, 2]) + common.NS(b'No such file or directory') + common.NS(b''))
        assert_that(transport.value(), equal_to(expected))

@skipIf(not cryptography, 'Cannot run without cryptography')
class RawPacketDataTests(TestCase):
    """
    Tests for L{filetransfer.FileTransferClient} which explicitly craft certain
    less common protocol messages to exercise their handling.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.ftc = filetransfer.FileTransferClient()

    def test_packetSTATUS(self):
        if False:
            return 10
        '\n        A STATUS packet containing a result code, a message, and a language is\n        parsed to produce the result of an outstanding request L{Deferred}.\n\n        @see: U{section 9.1<http://tools.ietf.org/html/draft-ietf-secsh-filexfer-13#section-9.1>}\n            of the SFTP Internet-Draft.\n        '
        d = defer.Deferred()
        d.addCallback(self._cbTestPacketSTATUS)
        self.ftc.openRequests[1] = d
        data = struct.pack('!LL', 1, filetransfer.FX_OK) + common.NS(b'msg') + common.NS(b'lang')
        self.ftc.packet_STATUS(data)
        return d

    def _cbTestPacketSTATUS(self, result):
        if False:
            while True:
                i = 10
        '\n        Assert that the result is a two-tuple containing the message and\n        language from the STATUS packet.\n        '
        self.assertEqual(result[0], b'msg')
        self.assertEqual(result[1], b'lang')

    def test_packetSTATUSShort(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A STATUS packet containing only a result code can also be parsed to\n        produce the result of an outstanding request L{Deferred}.  Such packets\n        are sent by some SFTP implementations, though not strictly legal.\n\n        @see: U{section 9.1<http://tools.ietf.org/html/draft-ietf-secsh-filexfer-13#section-9.1>}\n            of the SFTP Internet-Draft.\n        '
        d = defer.Deferred()
        d.addCallback(self._cbTestPacketSTATUSShort)
        self.ftc.openRequests[1] = d
        data = struct.pack('!LL', 1, filetransfer.FX_OK)
        self.ftc.packet_STATUS(data)
        return d

    def _cbTestPacketSTATUSShort(self, result):
        if False:
            while True:
                i = 10
        '\n        Assert that the result is a two-tuple containing empty strings, since\n        the STATUS packet had neither a message nor a language.\n        '
        self.assertEqual(result[0], b'')
        self.assertEqual(result[1], b'')

    def test_packetSTATUSWithoutLang(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A STATUS packet containing a result code and a message but no language\n        can also be parsed to produce the result of an outstanding request\n        L{Deferred}.  Such packets are sent by some SFTP implementations, though\n        not strictly legal.\n\n        @see: U{section 9.1<http://tools.ietf.org/html/draft-ietf-secsh-filexfer-13#section-9.1>}\n            of the SFTP Internet-Draft.\n        '
        d = defer.Deferred()
        d.addCallback(self._cbTestPacketSTATUSWithoutLang)
        self.ftc.openRequests[1] = d
        data = struct.pack('!LL', 1, filetransfer.FX_OK) + common.NS(b'msg')
        self.ftc.packet_STATUS(data)
        return d

    def _cbTestPacketSTATUSWithoutLang(self, result):
        if False:
            i = 10
            return i + 15
        '\n        Assert that the result is a two-tuple containing the message from the\n        STATUS packet and an empty string, since the language was missing.\n        '
        self.assertEqual(result[0], b'msg')
        self.assertEqual(result[1], b'')