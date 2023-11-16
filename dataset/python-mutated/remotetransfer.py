"""
module for regrouping all FileWriterImpl and FileReaderImpl away from steps
"""
import os
import tarfile
import tempfile
from io import BytesIO
from buildbot.util import bytes2unicode
from buildbot.util import unicode2bytes
from buildbot.worker.protocols import base

class FileWriter(base.FileWriterImpl):
    """
    Helper class that acts as a file-object with write access
    """

    def __init__(self, destfile, maxsize, mode):
        if False:
            while True:
                i = 10
        destfile = os.path.abspath(destfile)
        dirname = os.path.dirname(destfile)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.destfile = destfile
        self.mode = mode
        (fd, self.tmpname) = tempfile.mkstemp(dir=dirname, prefix='buildbot-transfer-')
        self.fp = os.fdopen(fd, 'wb')
        self.remaining = maxsize

    def remote_write(self, data):
        if False:
            i = 10
            return i + 15
        '\n        Called from remote worker to write L{data} to L{fp} within boundaries\n        of L{maxsize}\n\n        @type  data: C{string}\n        @param data: String of data to write\n        '
        data = unicode2bytes(data)
        if self.remaining is not None:
            if len(data) > self.remaining:
                data = data[:self.remaining]
            self.fp.write(data)
            self.remaining = self.remaining - len(data)
        else:
            self.fp.write(data)

    def remote_utime(self, accessed_modified):
        if False:
            while True:
                i = 10
        os.utime(self.destfile, accessed_modified)

    def remote_close(self):
        if False:
            i = 10
            return i + 15
        '\n        Called by remote worker to state that no more data will be transferred\n        '
        self.fp.close()
        self.fp = None
        if os.path.exists(self.destfile):
            os.unlink(self.destfile)
        os.rename(self.tmpname, self.destfile)
        self.tmpname = None
        if self.mode is not None:
            os.chmod(self.destfile, self.mode)

    def cancel(self):
        if False:
            print('Hello World!')
        fp = getattr(self, 'fp', None)
        if fp:
            fp.close()
            if self.destfile and os.path.exists(self.destfile):
                os.unlink(self.destfile)
            if self.tmpname and os.path.exists(self.tmpname):
                os.unlink(self.tmpname)

class DirectoryWriter(FileWriter):
    """
    A DirectoryWriter is implemented as a FileWriter, with an added post-processing
    step to unpack the archive, once the transfer has completed.
    """

    def __init__(self, destroot, maxsize, compress, mode):
        if False:
            for i in range(10):
                print('nop')
        self.destroot = destroot
        self.compress = compress
        (self.fd, self.tarname) = tempfile.mkstemp(prefix='buildbot-transfer-')
        os.close(self.fd)
        super().__init__(self.tarname, maxsize, mode)

    def remote_unpack(self):
        if False:
            print('Hello World!')
        '\n        Called by remote worker to state that no more data will be transferred\n        '
        self.remote_close()
        if self.compress == 'bz2':
            mode = 'r|bz2'
        elif self.compress == 'gz':
            mode = 'r|gz'
        else:
            mode = 'r'
        with tarfile.open(name=self.tarname, mode=mode) as archive:
            archive.extractall(path=self.destroot)
        os.remove(self.tarname)

class FileReader(base.FileReaderImpl):
    """
    Helper class that acts as a file-object with read access
    """

    def __init__(self, fp):
        if False:
            while True:
                i = 10
        self.fp = fp

    def remote_read(self, maxlength):
        if False:
            i = 10
            return i + 15
        '\n        Called from remote worker to read at most L{maxlength} bytes of data\n\n        @type  maxlength: C{integer}\n        @param maxlength: Maximum number of data bytes that can be returned\n\n        @return: Data read from L{fp}\n        @rtype: C{string} of bytes read from file\n        '
        if self.fp is None:
            return ''
        data = self.fp.read(maxlength)
        return data

    def remote_close(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called by remote worker to state that no more data will be transferred\n        '
        if self.fp is not None:
            self.fp.close()
            self.fp = None

class StringFileWriter(base.FileWriterImpl):
    """
    FileWriter class that just puts received data into a buffer.

    Used to upload a file from worker for inline processing rather than
    writing into a file on master.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.buffer = ''

    def remote_write(self, data):
        if False:
            return 10
        self.buffer += bytes2unicode(data)

    def remote_close(self):
        if False:
            print('Hello World!')
        pass

class StringFileReader(FileReader):
    """
    FileWriter class that just buid send data from a string.

    Used to download a file to worker from local string rather than first
    writing into a file on master.
    """

    def __init__(self, s):
        if False:
            while True:
                i = 10
        s = unicode2bytes(s)
        super().__init__(BytesIO(s))