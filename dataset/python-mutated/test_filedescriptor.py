"""
Whitebox tests for L{twisted.internet.abstract.FileDescriptor}.
"""
from zope.interface.verify import verifyClass
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IPushProducer
from twisted.trial.unittest import SynchronousTestCase

class MemoryFile(FileDescriptor):
    """
    A L{FileDescriptor} customization which writes to a Python list in memory
    with certain limitations.

    @ivar _written: A C{list} of C{bytes} which have been accepted as written.

    @ivar _freeSpace: A C{int} giving the number of bytes which will be accepted
        by future writes.
    """
    connected = True

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        FileDescriptor.__init__(self, reactor=object())
        self._written = []
        self._freeSpace = 0

    def startWriting(self):
        if False:
            return 10
        pass

    def stopWriting(self):
        if False:
            while True:
                i = 10
        pass

    def writeSomeData(self, data):
        if False:
            i = 10
            return i + 15
        '\n        Copy at most C{self._freeSpace} bytes from C{data} into C{self._written}.\n\n        @return: A C{int} indicating how many bytes were copied from C{data}.\n        '
        acceptLength = min(self._freeSpace, len(data))
        if acceptLength:
            self._freeSpace -= acceptLength
            self._written.append(data[:acceptLength])
        return acceptLength

class FileDescriptorTests(SynchronousTestCase):
    """
    Tests for L{FileDescriptor}.
    """

    def test_writeWithUnicodeRaisesException(self):
        if False:
            return 10
        "\n        L{FileDescriptor.write} doesn't accept unicode data.\n        "
        fileDescriptor = FileDescriptor(reactor=object())
        self.assertRaises(TypeError, fileDescriptor.write, 'foo')

    def test_writeSequenceWithUnicodeRaisesException(self):
        if False:
            i = 10
            return i + 15
        "\n        L{FileDescriptor.writeSequence} doesn't accept unicode data.\n        "
        fileDescriptor = FileDescriptor(reactor=object())
        self.assertRaises(TypeError, fileDescriptor.writeSequence, [b'foo', 'bar', b'baz'])

    def test_implementInterfaceIPushProducer(self):
        if False:
            i = 10
            return i + 15
        '\n        L{FileDescriptor} should implement L{IPushProducer}.\n        '
        self.assertTrue(verifyClass(IPushProducer, FileDescriptor))

class WriteDescriptorTests(SynchronousTestCase):
    """
    Tests for L{FileDescriptor}'s implementation of L{IWriteDescriptor}.
    """

    def test_kernelBufferFull(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When L{FileDescriptor.writeSomeData} returns C{0} to indicate no more\n        data can be written immediately, L{FileDescriptor.doWrite} returns\n        L{None}.\n        '
        descriptor = MemoryFile()
        descriptor.write(b'hello, world')
        self.assertIsNone(descriptor.doWrite())