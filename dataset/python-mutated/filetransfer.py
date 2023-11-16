import errno
import os
import struct
import warnings
from typing import Dict
from zope.interface import implementer
from twisted.conch.interfaces import ISFTPFile, ISFTPServer
from twisted.conch.ssh.common import NS, getNS
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure
from twisted.python.compat import nativeString, networkString

class FileTransferBase(protocol.Protocol):
    _log = Logger()
    versions = (3,)
    packetTypes: Dict[int, str] = {}

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.buf = b''
        self.otherVersion = None

    def sendPacket(self, kind, data):
        if False:
            return 10
        self.transport.write(struct.pack('!LB', len(data) + 1, kind) + data)

    def dataReceived(self, data):
        if False:
            while True:
                i = 10
        self.buf += data
        while len(self.buf) >= 9:
            header = self.buf[:9]
            (length, kind, reqId) = struct.unpack('!LBL', header)
            if len(self.buf) < 4 + length:
                return
            (data, self.buf) = (self.buf[5:4 + length], self.buf[4 + length:])
            packetType = self.packetTypes.get(kind, None)
            if not packetType:
                self._log.info('no packet type for {kind}', kind=kind)
                continue
            f = getattr(self, f'packet_{packetType}', None)
            if not f:
                self._log.info('not implemented: {packetType} data={data!r}', packetType=packetType, data=data[4:])
                self._sendStatus(reqId, FX_OP_UNSUPPORTED, f"don't understand {packetType}")
                continue
            self._log.info('dispatching: {packetType} requestId={reqId}', packetType=packetType, reqId=reqId)
            try:
                f(data)
            except Exception:
                self._log.failure('Failed to handle packet of type {packetType}', packetType=packetType)
                continue

    def _parseAttributes(self, data):
        if False:
            while True:
                i = 10
        (flags,) = struct.unpack('!L', data[:4])
        attrs = {}
        data = data[4:]
        if flags & FILEXFER_ATTR_SIZE == FILEXFER_ATTR_SIZE:
            (size,) = struct.unpack('!Q', data[:8])
            attrs['size'] = size
            data = data[8:]
        if flags & FILEXFER_ATTR_OWNERGROUP == FILEXFER_ATTR_OWNERGROUP:
            (uid, gid) = struct.unpack('!2L', data[:8])
            attrs['uid'] = uid
            attrs['gid'] = gid
            data = data[8:]
        if flags & FILEXFER_ATTR_PERMISSIONS == FILEXFER_ATTR_PERMISSIONS:
            (perms,) = struct.unpack('!L', data[:4])
            attrs['permissions'] = perms
            data = data[4:]
        if flags & FILEXFER_ATTR_ACMODTIME == FILEXFER_ATTR_ACMODTIME:
            (atime, mtime) = struct.unpack('!2L', data[:8])
            attrs['atime'] = atime
            attrs['mtime'] = mtime
            data = data[8:]
        if flags & FILEXFER_ATTR_EXTENDED == FILEXFER_ATTR_EXTENDED:
            (extendedCount,) = struct.unpack('!L', data[:4])
            data = data[4:]
            for i in range(extendedCount):
                (extendedType, data) = getNS(data)
                (extendedData, data) = getNS(data)
                attrs[f'ext_{nativeString(extendedType)}'] = extendedData
        return (attrs, data)

    def _packAttributes(self, attrs):
        if False:
            i = 10
            return i + 15
        flags = 0
        data = b''
        if 'size' in attrs:
            data += struct.pack('!Q', attrs['size'])
            flags |= FILEXFER_ATTR_SIZE
        if 'uid' in attrs and 'gid' in attrs:
            data += struct.pack('!2L', attrs['uid'], attrs['gid'])
            flags |= FILEXFER_ATTR_OWNERGROUP
        if 'permissions' in attrs:
            data += struct.pack('!L', attrs['permissions'])
            flags |= FILEXFER_ATTR_PERMISSIONS
        if 'atime' in attrs and 'mtime' in attrs:
            data += struct.pack('!2L', attrs['atime'], attrs['mtime'])
            flags |= FILEXFER_ATTR_ACMODTIME
        extended = []
        for k in attrs:
            if k.startswith('ext_'):
                extType = NS(networkString(k[4:]))
                extData = NS(attrs[k])
                extended.append(extType + extData)
        if extended:
            data += struct.pack('!L', len(extended))
            data += b''.join(extended)
            flags |= FILEXFER_ATTR_EXTENDED
        return struct.pack('!L', flags) + data

    def connectionLost(self, reason):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called when connection to the remote subsystem was lost.\n        '
        super().connectionLost(reason)
        self.connected = False

class FileTransferServer(FileTransferBase):

    def __init__(self, data=None, avatar=None):
        if False:
            while True:
                i = 10
        FileTransferBase.__init__(self)
        self.client = ISFTPServer(avatar)
        self.openFiles = {}
        self.openDirs = {}

    def packet_INIT(self, data):
        if False:
            for i in range(10):
                print('nop')
        (version,) = struct.unpack('!L', data[:4])
        self.version = min(list(self.versions) + [version])
        data = data[4:]
        ext = {}
        while data:
            (extName, data) = getNS(data)
            (extData, data) = getNS(data)
            ext[extName] = extData
        ourExt = self.client.gotVersion(version, ext)
        ourExtData = b''
        for (k, v) in ourExt.items():
            ourExtData += NS(k) + NS(v)
        self.sendPacket(FXP_VERSION, struct.pack('!L', self.version) + ourExtData)

    def packet_OPEN(self, data):
        if False:
            i = 10
            return i + 15
        requestId = data[:4]
        data = data[4:]
        (filename, data) = getNS(data)
        (flags,) = struct.unpack('!L', data[:4])
        data = data[4:]
        (attrs, data) = self._parseAttributes(data)
        assert data == b'', f'still have data in OPEN: {data!r}'
        d = defer.maybeDeferred(self.client.openFile, filename, flags, attrs)
        d.addCallback(self._cbOpenFile, requestId)
        d.addErrback(self._ebStatus, requestId, b'open failed')

    def _cbOpenFile(self, fileObj, requestId):
        if False:
            while True:
                i = 10
        fileId = networkString(str(hash(fileObj)))
        if fileId in self.openFiles:
            raise KeyError('id already open')
        self.openFiles[fileId] = fileObj
        self.sendPacket(FXP_HANDLE, requestId + NS(fileId))

    def packet_CLOSE(self, data):
        if False:
            print('Hello World!')
        requestId = data[:4]
        data = data[4:]
        (handle, data) = getNS(data)
        self._log.info('closing: {requestId!r} {handle!r}', requestId=requestId, handle=handle)
        assert data == b'', f'still have data in CLOSE: {data!r}'
        if handle in self.openFiles:
            fileObj = self.openFiles[handle]
            d = defer.maybeDeferred(fileObj.close)
            d.addCallback(self._cbClose, handle, requestId)
            d.addErrback(self._ebStatus, requestId, b'close failed')
        elif handle in self.openDirs:
            dirObj = self.openDirs[handle][0]
            d = defer.maybeDeferred(dirObj.close)
            d.addCallback(self._cbClose, handle, requestId, 1)
            d.addErrback(self._ebStatus, requestId, b'close failed')
        else:
            code = errno.ENOENT
            text = os.strerror(code)
            err = OSError(code, text)
            self._ebStatus(failure.Failure(err), requestId)

    def _cbClose(self, result, handle, requestId, isDir=0):
        if False:
            i = 10
            return i + 15
        if isDir:
            del self.openDirs[handle]
        else:
            del self.openFiles[handle]
        self._sendStatus(requestId, FX_OK, b'file closed')

    def packet_READ(self, data):
        if False:
            print('Hello World!')
        requestId = data[:4]
        data = data[4:]
        (handle, data) = getNS(data)
        ((offset, length), data) = (struct.unpack('!QL', data[:12]), data[12:])
        assert data == b'', f'still have data in READ: {data!r}'
        if handle not in self.openFiles:
            self._ebRead(failure.Failure(KeyError()), requestId)
        else:
            fileObj = self.openFiles[handle]
            d = defer.maybeDeferred(fileObj.readChunk, offset, length)
            d.addCallback(self._cbRead, requestId)
            d.addErrback(self._ebStatus, requestId, b'read failed')

    def _cbRead(self, result, requestId):
        if False:
            print('Hello World!')
        if result == b'':
            raise EOFError()
        self.sendPacket(FXP_DATA, requestId + NS(result))

    def packet_WRITE(self, data):
        if False:
            while True:
                i = 10
        requestId = data[:4]
        data = data[4:]
        (handle, data) = getNS(data)
        (offset,) = struct.unpack('!Q', data[:8])
        data = data[8:]
        (writeData, data) = getNS(data)
        assert data == b'', f'still have data in WRITE: {data!r}'
        if handle not in self.openFiles:
            self._ebWrite(failure.Failure(KeyError()), requestId)
        else:
            fileObj = self.openFiles[handle]
            d = defer.maybeDeferred(fileObj.writeChunk, offset, writeData)
            d.addCallback(self._cbStatus, requestId, b'write succeeded')
            d.addErrback(self._ebStatus, requestId, b'write failed')

    def packet_REMOVE(self, data):
        if False:
            i = 10
            return i + 15
        requestId = data[:4]
        data = data[4:]
        (filename, data) = getNS(data)
        assert data == b'', f'still have data in REMOVE: {data!r}'
        d = defer.maybeDeferred(self.client.removeFile, filename)
        d.addCallback(self._cbStatus, requestId, b'remove succeeded')
        d.addErrback(self._ebStatus, requestId, b'remove failed')

    def packet_RENAME(self, data):
        if False:
            i = 10
            return i + 15
        requestId = data[:4]
        data = data[4:]
        (oldPath, data) = getNS(data)
        (newPath, data) = getNS(data)
        assert data == b'', f'still have data in RENAME: {data!r}'
        d = defer.maybeDeferred(self.client.renameFile, oldPath, newPath)
        d.addCallback(self._cbStatus, requestId, b'rename succeeded')
        d.addErrback(self._ebStatus, requestId, b'rename failed')

    def packet_MKDIR(self, data):
        if False:
            while True:
                i = 10
        requestId = data[:4]
        data = data[4:]
        (path, data) = getNS(data)
        (attrs, data) = self._parseAttributes(data)
        assert data == b'', f'still have data in MKDIR: {data!r}'
        d = defer.maybeDeferred(self.client.makeDirectory, path, attrs)
        d.addCallback(self._cbStatus, requestId, b'mkdir succeeded')
        d.addErrback(self._ebStatus, requestId, b'mkdir failed')

    def packet_RMDIR(self, data):
        if False:
            print('Hello World!')
        requestId = data[:4]
        data = data[4:]
        (path, data) = getNS(data)
        assert data == b'', f'still have data in RMDIR: {data!r}'
        d = defer.maybeDeferred(self.client.removeDirectory, path)
        d.addCallback(self._cbStatus, requestId, b'rmdir succeeded')
        d.addErrback(self._ebStatus, requestId, b'rmdir failed')

    def packet_OPENDIR(self, data):
        if False:
            print('Hello World!')
        requestId = data[:4]
        data = data[4:]
        (path, data) = getNS(data)
        assert data == b'', f'still have data in OPENDIR: {data!r}'
        d = defer.maybeDeferred(self.client.openDirectory, path)
        d.addCallback(self._cbOpenDirectory, requestId)
        d.addErrback(self._ebStatus, requestId, b'opendir failed')

    def _cbOpenDirectory(self, dirObj, requestId):
        if False:
            while True:
                i = 10
        handle = networkString(str(hash(dirObj)))
        if handle in self.openDirs:
            raise KeyError('already opened this directory')
        self.openDirs[handle] = [dirObj, iter(dirObj)]
        self.sendPacket(FXP_HANDLE, requestId + NS(handle))

    def packet_READDIR(self, data):
        if False:
            return 10
        requestId = data[:4]
        data = data[4:]
        (handle, data) = getNS(data)
        assert data == b'', f'still have data in READDIR: {data!r}'
        if handle not in self.openDirs:
            self._ebStatus(failure.Failure(KeyError()), requestId)
        else:
            (dirObj, dirIter) = self.openDirs[handle]
            d = defer.maybeDeferred(self._scanDirectory, dirIter, [])
            d.addCallback(self._cbSendDirectory, requestId)
            d.addErrback(self._ebStatus, requestId, b'scan directory failed')

    def _scanDirectory(self, dirIter, f):
        if False:
            return 10
        while len(f) < 250:
            try:
                info = next(dirIter)
            except StopIteration:
                if not f:
                    raise EOFError
                return f
            if isinstance(info, defer.Deferred):
                info.addCallback(self._cbScanDirectory, dirIter, f)
                return
            else:
                f.append(info)
        return f

    def _cbScanDirectory(self, result, dirIter, f):
        if False:
            print('Hello World!')
        f.append(result)
        return self._scanDirectory(dirIter, f)

    def _cbSendDirectory(self, result, requestId):
        if False:
            for i in range(10):
                print('nop')
        data = b''
        for (filename, longname, attrs) in result:
            data += NS(filename)
            data += NS(longname)
            data += self._packAttributes(attrs)
        self.sendPacket(FXP_NAME, requestId + struct.pack('!L', len(result)) + data)

    def packet_STAT(self, data, followLinks=1):
        if False:
            print('Hello World!')
        requestId = data[:4]
        data = data[4:]
        (path, data) = getNS(data)
        assert data == b'', f'still have data in STAT/LSTAT: {data!r}'
        d = defer.maybeDeferred(self.client.getAttrs, path, followLinks)
        d.addCallback(self._cbStat, requestId)
        d.addErrback(self._ebStatus, requestId, b'stat/lstat failed')

    def packet_LSTAT(self, data):
        if False:
            return 10
        self.packet_STAT(data, 0)

    def packet_FSTAT(self, data):
        if False:
            return 10
        requestId = data[:4]
        data = data[4:]
        (handle, data) = getNS(data)
        assert data == b'', f'still have data in FSTAT: {data!r}'
        if handle not in self.openFiles:
            self._ebStatus(failure.Failure(KeyError(f'{handle} not in self.openFiles')), requestId)
        else:
            fileObj = self.openFiles[handle]
            d = defer.maybeDeferred(fileObj.getAttrs)
            d.addCallback(self._cbStat, requestId)
            d.addErrback(self._ebStatus, requestId, b'fstat failed')

    def _cbStat(self, result, requestId):
        if False:
            return 10
        data = requestId + self._packAttributes(result)
        self.sendPacket(FXP_ATTRS, data)

    def packet_SETSTAT(self, data):
        if False:
            while True:
                i = 10
        requestId = data[:4]
        data = data[4:]
        (path, data) = getNS(data)
        (attrs, data) = self._parseAttributes(data)
        if data != b'':
            self._log.warn('Still have data in SETSTAT: {data!r}', data=data)
        d = defer.maybeDeferred(self.client.setAttrs, path, attrs)
        d.addCallback(self._cbStatus, requestId, b'setstat succeeded')
        d.addErrback(self._ebStatus, requestId, b'setstat failed')

    def packet_FSETSTAT(self, data):
        if False:
            print('Hello World!')
        requestId = data[:4]
        data = data[4:]
        (handle, data) = getNS(data)
        (attrs, data) = self._parseAttributes(data)
        assert data == b'', f'still have data in FSETSTAT: {data!r}'
        if handle not in self.openFiles:
            self._ebStatus(failure.Failure(KeyError()), requestId)
        else:
            fileObj = self.openFiles[handle]
            d = defer.maybeDeferred(fileObj.setAttrs, attrs)
            d.addCallback(self._cbStatus, requestId, b'fsetstat succeeded')
            d.addErrback(self._ebStatus, requestId, b'fsetstat failed')

    def packet_READLINK(self, data):
        if False:
            return 10
        requestId = data[:4]
        data = data[4:]
        (path, data) = getNS(data)
        assert data == b'', f'still have data in READLINK: {data!r}'
        d = defer.maybeDeferred(self.client.readLink, path)
        d.addCallback(self._cbReadLink, requestId)
        d.addErrback(self._ebStatus, requestId, b'readlink failed')

    def _cbReadLink(self, result, requestId):
        if False:
            return 10
        self._cbSendDirectory([(result, b'', {})], requestId)

    def packet_SYMLINK(self, data):
        if False:
            print('Hello World!')
        requestId = data[:4]
        data = data[4:]
        (linkPath, data) = getNS(data)
        (targetPath, data) = getNS(data)
        d = defer.maybeDeferred(self.client.makeLink, linkPath, targetPath)
        d.addCallback(self._cbStatus, requestId, b'symlink succeeded')
        d.addErrback(self._ebStatus, requestId, b'symlink failed')

    def packet_REALPATH(self, data):
        if False:
            i = 10
            return i + 15
        requestId = data[:4]
        data = data[4:]
        (path, data) = getNS(data)
        assert data == b'', f'still have data in REALPATH: {data!r}'
        d = defer.maybeDeferred(self.client.realPath, path)
        d.addCallback(self._cbReadLink, requestId)
        d.addErrback(self._ebStatus, requestId, b'realpath failed')

    def packet_EXTENDED(self, data):
        if False:
            while True:
                i = 10
        requestId = data[:4]
        data = data[4:]
        (extName, extData) = getNS(data)
        d = defer.maybeDeferred(self.client.extendedRequest, extName, extData)
        d.addCallback(self._cbExtended, requestId)
        d.addErrback(self._ebStatus, requestId, b'extended ' + extName + b' failed')

    def _cbExtended(self, data, requestId):
        if False:
            return 10
        self.sendPacket(FXP_EXTENDED_REPLY, requestId + data)

    def _cbStatus(self, result, requestId, msg=b'request succeeded'):
        if False:
            while True:
                i = 10
        self._sendStatus(requestId, FX_OK, msg)

    def _ebStatus(self, reason, requestId, msg=b'request failed'):
        if False:
            while True:
                i = 10
        code = FX_FAILURE
        message = msg
        if isinstance(reason.value, (IOError, OSError)):
            if reason.value.errno == errno.ENOENT:
                code = FX_NO_SUCH_FILE
                message = networkString(reason.value.strerror)
            elif reason.value.errno == errno.EACCES:
                code = FX_PERMISSION_DENIED
                message = networkString(reason.value.strerror)
            elif reason.value.errno == errno.EEXIST:
                code = FX_FILE_ALREADY_EXISTS
            else:
                self._log.failure('Request {requestId} failed: {message}', failure=reason, requestId=requestId, message=message)
        elif isinstance(reason.value, EOFError):
            code = FX_EOF
            if reason.value.args:
                message = networkString(reason.value.args[0])
        elif isinstance(reason.value, NotImplementedError):
            code = FX_OP_UNSUPPORTED
            if reason.value.args:
                message = networkString(reason.value.args[0])
        elif isinstance(reason.value, SFTPError):
            code = reason.value.code
            message = networkString(reason.value.message)
        else:
            self._log.failure('Request {requestId} failed with unknown error: {message}', failure=reason, requestId=requestId, message=message)
        self._sendStatus(requestId, code, message)

    def _sendStatus(self, requestId, code, message, lang=b''):
        if False:
            while True:
                i = 10
        '\n        Helper method to send a FXP_STATUS message.\n        '
        data = requestId + struct.pack('!L', code)
        data += NS(message)
        data += NS(lang)
        self.sendPacket(FXP_STATUS, data)

    def connectionLost(self, reason):
        if False:
            return 10
        '\n        Called when connection to the remote subsystem was lost.\n\n        Clean all opened files and directories.\n        '
        FileTransferBase.connectionLost(self, reason)
        for fileObj in self.openFiles.values():
            fileObj.close()
        self.openFiles = {}
        for (dirObj, dirIter) in self.openDirs.values():
            dirObj.close()
        self.openDirs = {}

class FileTransferClient(FileTransferBase):

    def __init__(self, extData={}):
        if False:
            for i in range(10):
                print('nop')
        '\n        @param extData: a dict of extended_name : extended_data items\n        to be sent to the server.\n        '
        FileTransferBase.__init__(self)
        self.extData = {}
        self.counter = 0
        self.openRequests = {}

    def connectionMade(self):
        if False:
            i = 10
            return i + 15
        data = struct.pack('!L', max(self.versions))
        for (k, v) in self.extData.values():
            data += NS(k) + NS(v)
        self.sendPacket(FXP_INIT, data)

    def connectionLost(self, reason):
        if False:
            return 10
        '\n        Called when connection to the remote subsystem was lost.\n\n        Any pending requests are aborted.\n        '
        FileTransferBase.connectionLost(self, reason)
        if self.openRequests:
            requestError = error.ConnectionLost()
            requestError.__cause__ = reason.value
            requestFailure = failure.Failure(requestError)
            while self.openRequests:
                (_, deferred) = self.openRequests.popitem()
                deferred.errback(requestFailure)

    def _sendRequest(self, msg, data):
        if False:
            print('Hello World!')
        '\n        Send a request and return a deferred which waits for the result.\n\n        @type msg: L{int}\n        @param msg: The request type (e.g., C{FXP_READ}).\n\n        @type data: L{bytes}\n        @param data: The body of the request.\n        '
        if not self.connected:
            return defer.fail(error.ConnectionLost())
        data = struct.pack('!L', self.counter) + data
        d = defer.Deferred()
        self.openRequests[self.counter] = d
        self.counter += 1
        self.sendPacket(msg, data)
        return d

    def _parseRequest(self, data):
        if False:
            print('Hello World!')
        (id,) = struct.unpack('!L', data[:4])
        d = self.openRequests[id]
        del self.openRequests[id]
        return (d, data[4:])

    def openFile(self, filename, flags, attrs):
        if False:
            i = 10
            return i + 15
        '\n        Open a file.\n\n        This method returns a L{Deferred} that is called back with an object\n        that provides the L{ISFTPFile} interface.\n\n        @type filename: L{bytes}\n        @param filename: a string representing the file to open.\n\n        @param flags: an integer of the flags to open the file with, ORed together.\n        The flags and their values are listed at the bottom of this file.\n\n        @param attrs: a list of attributes to open the file with.  It is a\n        dictionary, consisting of 0 or more keys.  The possible keys are::\n\n            size: the size of the file in bytes\n            uid: the user ID of the file as an integer\n            gid: the group ID of the file as an integer\n            permissions: the permissions of the file with as an integer.\n            the bit representation of this field is defined by POSIX.\n            atime: the access time of the file as seconds since the epoch.\n            mtime: the modification time of the file as seconds since the epoch.\n            ext_*: extended attributes.  The server is not required to\n            understand this, but it may.\n\n        NOTE: there is no way to indicate text or binary files.  it is up\n        to the SFTP client to deal with this.\n        '
        data = NS(filename) + struct.pack('!L', flags) + self._packAttributes(attrs)
        d = self._sendRequest(FXP_OPEN, data)
        d.addCallback(self._cbOpenHandle, ClientFile, filename)
        return d

    def _cbOpenHandle(self, handle, handleClass, name):
        if False:
            while True:
                i = 10
        '\n        Callback invoked when an OPEN or OPENDIR request succeeds.\n\n        @param handle: The handle returned by the server\n        @type handle: L{bytes}\n        @param handleClass: The class that will represent the\n        newly-opened file or directory to the user (either L{ClientFile} or\n        L{ClientDirectory}).\n        @param name: The name of the file or directory represented\n        by C{handle}.\n        @type name: L{bytes}\n        '
        cb = handleClass(self, handle)
        cb.name = name
        return cb

    def removeFile(self, filename):
        if False:
            while True:
                i = 10
        '\n        Remove the given file.\n\n        This method returns a Deferred that is called back when it succeeds.\n\n        @type filename: L{bytes}\n        @param filename: the name of the file as a string.\n        '
        return self._sendRequest(FXP_REMOVE, NS(filename))

    def renameFile(self, oldpath, newpath):
        if False:
            return 10
        '\n        Rename the given file.\n\n        This method returns a Deferred that is called back when it succeeds.\n\n        @type oldpath: L{bytes}\n        @param oldpath: the current location of the file.\n        @type newpath: L{bytes}\n        @param newpath: the new file name.\n        '
        return self._sendRequest(FXP_RENAME, NS(oldpath) + NS(newpath))

    def makeDirectory(self, path, attrs):
        if False:
            i = 10
            return i + 15
        '\n        Make a directory.\n\n        This method returns a Deferred that is called back when it is\n        created.\n\n        @type path: L{bytes}\n        @param path: the name of the directory to create as a string.\n\n        @param attrs: a dictionary of attributes to create the directory\n        with.  Its meaning is the same as the attrs in the openFile method.\n        '
        return self._sendRequest(FXP_MKDIR, NS(path) + self._packAttributes(attrs))

    def removeDirectory(self, path):
        if False:
            print('Hello World!')
        '\n        Remove a directory (non-recursively)\n\n        It is an error to remove a directory that has files or directories in\n        it.\n\n        This method returns a Deferred that is called back when it is removed.\n\n        @type path: L{bytes}\n        @param path: the directory to remove.\n        '
        return self._sendRequest(FXP_RMDIR, NS(path))

    def openDirectory(self, path):
        if False:
            return 10
        "\n        Open a directory for scanning.\n\n        This method returns a Deferred that is called back with an iterable\n        object that has a close() method.\n\n        The close() method is called when the client is finished reading\n        from the directory.  At this point, the iterable will no longer\n        be used.\n\n        The iterable returns triples of the form (filename, longname, attrs)\n        or a Deferred that returns the same.  The sequence must support\n        __getitem__, but otherwise may be any 'sequence-like' object.\n\n        filename is the name of the file relative to the directory.\n        logname is an expanded format of the filename.  The recommended format\n        is:\n        -rwxr-xr-x   1 mjos     staff      348911 Mar 25 14:29 t-filexfer\n        1234567890 123 12345678 12345678 12345678 123456789012\n\n        The first line is sample output, the second is the length of the field.\n        The fields are: permissions, link count, user owner, group owner,\n        size in bytes, modification time.\n\n        attrs is a dictionary in the format of the attrs argument to openFile.\n\n        @type path: L{bytes}\n        @param path: the directory to open.\n        "
        d = self._sendRequest(FXP_OPENDIR, NS(path))
        d.addCallback(self._cbOpenHandle, ClientDirectory, path)
        return d

    def getAttrs(self, path, followLinks=0):
        if False:
            i = 10
            return i + 15
        '\n        Return the attributes for the given path.\n\n        This method returns a dictionary in the same format as the attrs\n        argument to openFile or a Deferred that is called back with same.\n\n        @type path: L{bytes}\n        @param path: the path to return attributes for as a string.\n        @param followLinks: a boolean.  if it is True, follow symbolic links\n        and return attributes for the real path at the base.  if it is False,\n        return attributes for the specified path.\n        '
        if followLinks:
            m = FXP_STAT
        else:
            m = FXP_LSTAT
        return self._sendRequest(m, NS(path))

    def setAttrs(self, path, attrs):
        if False:
            while True:
                i = 10
        '\n        Set the attributes for the path.\n\n        This method returns when the attributes are set or a Deferred that is\n        called back when they are.\n\n        @type path: L{bytes}\n        @param path: the path to set attributes for as a string.\n        @param attrs: a dictionary in the same format as the attrs argument to\n        openFile.\n        '
        data = NS(path) + self._packAttributes(attrs)
        return self._sendRequest(FXP_SETSTAT, data)

    def readLink(self, path):
        if False:
            return 10
        '\n        Find the root of a set of symbolic links.\n\n        This method returns the target of the link, or a Deferred that\n        returns the same.\n\n        @type path: L{bytes}\n        @param path: the path of the symlink to read.\n        '
        d = self._sendRequest(FXP_READLINK, NS(path))
        return d.addCallback(self._cbRealPath)

    def makeLink(self, linkPath, targetPath):
        if False:
            i = 10
            return i + 15
        '\n        Create a symbolic link.\n\n        This method returns when the link is made, or a Deferred that\n        returns the same.\n\n        @type linkPath: L{bytes}\n        @param linkPath: the pathname of the symlink as a string\n        @type targetPath: L{bytes}\n        @param targetPath: the path of the target of the link as a string.\n        '
        return self._sendRequest(FXP_SYMLINK, NS(linkPath) + NS(targetPath))

    def realPath(self, path):
        if False:
            print('Hello World!')
        '\n        Convert any path to an absolute path.\n\n        This method returns the absolute path as a string, or a Deferred\n        that returns the same.\n\n        @type path: L{bytes}\n        @param path: the path to convert as a string.\n        '
        d = self._sendRequest(FXP_REALPATH, NS(path))
        return d.addCallback(self._cbRealPath)

    def _cbRealPath(self, result):
        if False:
            for i in range(10):
                print('nop')
        (name, longname, attrs) = result[0]
        name = name.decode('utf-8')
        return name

    def extendedRequest(self, request, data):
        if False:
            return 10
        '\n        Make an extended request of the server.\n\n        The method returns a Deferred that is called back with\n        the result of the extended request.\n\n        @type request: L{bytes}\n        @param request: the name of the extended request to make.\n        @type data: L{bytes}\n        @param data: any other data that goes along with the request.\n        '
        return self._sendRequest(FXP_EXTENDED, NS(request) + data)

    def packet_VERSION(self, data):
        if False:
            while True:
                i = 10
        (version,) = struct.unpack('!L', data[:4])
        data = data[4:]
        d = {}
        while data:
            (k, data) = getNS(data)
            (v, data) = getNS(data)
            d[k] = v
        self.version = version
        self.gotServerVersion(version, d)

    def packet_STATUS(self, data):
        if False:
            i = 10
            return i + 15
        (d, data) = self._parseRequest(data)
        (code,) = struct.unpack('!L', data[:4])
        data = data[4:]
        if len(data) >= 4:
            (msg, data) = getNS(data)
            if len(data) >= 4:
                (lang, data) = getNS(data)
            else:
                lang = b''
        else:
            msg = b''
            lang = b''
        if code == FX_OK:
            d.callback((msg, lang))
        elif code == FX_EOF:
            d.errback(EOFError(msg))
        elif code == FX_OP_UNSUPPORTED:
            d.errback(NotImplementedError(msg))
        else:
            d.errback(SFTPError(code, nativeString(msg), lang))

    def packet_HANDLE(self, data):
        if False:
            i = 10
            return i + 15
        (d, data) = self._parseRequest(data)
        (handle, _) = getNS(data)
        d.callback(handle)

    def packet_DATA(self, data):
        if False:
            i = 10
            return i + 15
        (d, data) = self._parseRequest(data)
        d.callback(getNS(data)[0])

    def packet_NAME(self, data):
        if False:
            print('Hello World!')
        (d, data) = self._parseRequest(data)
        (count,) = struct.unpack('!L', data[:4])
        data = data[4:]
        files = []
        for i in range(count):
            (filename, data) = getNS(data)
            (longname, data) = getNS(data)
            (attrs, data) = self._parseAttributes(data)
            files.append((filename, longname, attrs))
        d.callback(files)

    def packet_ATTRS(self, data):
        if False:
            for i in range(10):
                print('nop')
        (d, data) = self._parseRequest(data)
        d.callback(self._parseAttributes(data)[0])

    def packet_EXTENDED_REPLY(self, data):
        if False:
            for i in range(10):
                print('nop')
        (d, data) = self._parseRequest(data)
        d.callback(data)

    def gotServerVersion(self, serverVersion, extData):
        if False:
            return 10
        '\n        Called when the client sends their version info.\n\n        @param serverVersion: an integer representing the version of the SFTP\n        protocol they are claiming.\n        @param extData: a dictionary of extended_name : extended_data items.\n        These items are sent by the client to indicate additional features.\n        '

@implementer(ISFTPFile)
class ClientFile:

    def __init__(self, parent, handle):
        if False:
            return 10
        self.parent = parent
        self.handle = NS(handle)

    def close(self):
        if False:
            print('Hello World!')
        return self.parent._sendRequest(FXP_CLOSE, self.handle)

    def readChunk(self, offset, length):
        if False:
            i = 10
            return i + 15
        data = self.handle + struct.pack('!QL', offset, length)
        return self.parent._sendRequest(FXP_READ, data)

    def writeChunk(self, offset, chunk):
        if False:
            for i in range(10):
                print('nop')
        data = self.handle + struct.pack('!Q', offset) + NS(chunk)
        return self.parent._sendRequest(FXP_WRITE, data)

    def getAttrs(self):
        if False:
            print('Hello World!')
        return self.parent._sendRequest(FXP_FSTAT, self.handle)

    def setAttrs(self, attrs):
        if False:
            i = 10
            return i + 15
        data = self.handle + self.parent._packAttributes(attrs)
        return self.parent._sendRequest(FXP_FSTAT, data)

class ClientDirectory:

    def __init__(self, parent, handle):
        if False:
            i = 10
            return i + 15
        self.parent = parent
        self.handle = NS(handle)
        self.filesCache = []

    def read(self):
        if False:
            return 10
        return self.parent._sendRequest(FXP_READDIR, self.handle)

    def close(self):
        if False:
            i = 10
            return i + 15
        if self.handle is None:
            return defer.succeed(None)
        d = self.parent._sendRequest(FXP_CLOSE, self.handle)
        self.handle = None
        return d

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __next__(self):
        if False:
            i = 10
            return i + 15
        warnings.warn('Using twisted.conch.ssh.filetransfer.ClientDirectory as an iterator was deprecated in Twisted 18.9.0.', category=DeprecationWarning, stacklevel=2)
        if self.filesCache:
            return self.filesCache.pop(0)
        if self.filesCache is None:
            raise StopIteration()
        d = self.read()
        d.addCallbacks(self._cbReadDir, self._ebReadDir)
        return d
    next = __next__

    def _cbReadDir(self, names):
        if False:
            return 10
        self.filesCache = names[1:]
        return names[0]

    def _ebReadDir(self, reason):
        if False:
            i = 10
            return i + 15
        reason.trap(EOFError)
        self.filesCache = None
        return failure.Failure(StopIteration())

class SFTPError(Exception):

    def __init__(self, errorCode, errorMessage, lang=''):
        if False:
            while True:
                i = 10
        Exception.__init__(self)
        self.code = errorCode
        self._message = errorMessage
        self.lang = lang

    @property
    def message(self):
        if False:
            while True:
                i = 10
        '\n        A string received over the network that explains the error to a human.\n        '
        return self._message

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        return f'SFTPError {self.code}: {self.message}'
FXP_INIT = 1
FXP_VERSION = 2
FXP_OPEN = 3
FXP_CLOSE = 4
FXP_READ = 5
FXP_WRITE = 6
FXP_LSTAT = 7
FXP_FSTAT = 8
FXP_SETSTAT = 9
FXP_FSETSTAT = 10
FXP_OPENDIR = 11
FXP_READDIR = 12
FXP_REMOVE = 13
FXP_MKDIR = 14
FXP_RMDIR = 15
FXP_REALPATH = 16
FXP_STAT = 17
FXP_RENAME = 18
FXP_READLINK = 19
FXP_SYMLINK = 20
FXP_STATUS = 101
FXP_HANDLE = 102
FXP_DATA = 103
FXP_NAME = 104
FXP_ATTRS = 105
FXP_EXTENDED = 200
FXP_EXTENDED_REPLY = 201
FILEXFER_ATTR_SIZE = 1
FILEXFER_ATTR_UIDGID = 2
FILEXFER_ATTR_OWNERGROUP = FILEXFER_ATTR_UIDGID
FILEXFER_ATTR_PERMISSIONS = 4
FILEXFER_ATTR_ACMODTIME = 8
FILEXFER_ATTR_EXTENDED = 2147483648
FILEXFER_TYPE_REGULAR = 1
FILEXFER_TYPE_DIRECTORY = 2
FILEXFER_TYPE_SYMLINK = 3
FILEXFER_TYPE_SPECIAL = 4
FILEXFER_TYPE_UNKNOWN = 5
FXF_READ = 1
FXF_WRITE = 2
FXF_APPEND = 4
FXF_CREAT = 8
FXF_TRUNC = 16
FXF_EXCL = 32
FXF_TEXT = 64
FX_OK = 0
FX_EOF = 1
FX_NO_SUCH_FILE = 2
FX_PERMISSION_DENIED = 3
FX_FAILURE = 4
FX_BAD_MESSAGE = 5
FX_NO_CONNECTION = 6
FX_CONNECTION_LOST = 7
FX_OP_UNSUPPORTED = 8
FX_FILE_ALREADY_EXISTS = 11
FX_NOT_A_DIRECTORY = FX_FAILURE
FX_FILE_IS_A_DIRECTORY = FX_FAILURE
g = globals()
for name in list(g.keys()):
    if name.startswith('FXP_'):
        value = g[name]
        FileTransferBase.packetTypes[value] = name[4:]
del g, name, value