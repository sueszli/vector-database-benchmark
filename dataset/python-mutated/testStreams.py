import unittest
import pythoncom
import win32com.server.util
import win32com.test.util

class Persists:
    _public_methods_ = ['GetClassID', 'IsDirty', 'Load', 'Save', 'GetSizeMax', 'InitNew']
    _com_interfaces_ = [pythoncom.IID_IPersistStreamInit]

    def __init__(self):
        if False:
            print('Hello World!')
        self.data = b'abcdefg'
        self.dirty = 1

    def GetClassID(self):
        if False:
            for i in range(10):
                print('nop')
        return pythoncom.IID_NULL

    def IsDirty(self):
        if False:
            while True:
                i = 10
        return self.dirty

    def Load(self, stream):
        if False:
            print('Hello World!')
        self.data = stream.Read(26)

    def Save(self, stream, clearDirty):
        if False:
            while True:
                i = 10
        stream.Write(self.data)
        if clearDirty:
            self.dirty = 0

    def GetSizeMax(self):
        if False:
            print('Hello World!')
        return 1024

    def InitNew(self):
        if False:
            print('Hello World!')
        pass

class Stream:
    _public_methods_ = ['Read', 'Write', 'Seek']
    _com_interfaces_ = [pythoncom.IID_IStream]

    def __init__(self, data):
        if False:
            print('Hello World!')
        self.data = data
        self.index = 0

    def Read(self, amount):
        if False:
            for i in range(10):
                print('nop')
        result = self.data[self.index:self.index + amount]
        self.index = self.index + amount
        return result

    def Write(self, data):
        if False:
            i = 10
            return i + 15
        self.data = data
        self.index = 0
        return len(data)

    def Seek(self, dist, origin):
        if False:
            while True:
                i = 10
        if origin == pythoncom.STREAM_SEEK_SET:
            self.index = dist
        elif origin == pythoncom.STREAM_SEEK_CUR:
            self.index = self.index + dist
        elif origin == pythoncom.STREAM_SEEK_END:
            self.index = len(self.data) + dist
        else:
            raise ValueError('Unknown Seek type: ' + str(origin))
        if self.index < 0:
            self.index = 0
        else:
            self.index = min(self.index, len(self.data))
        return self.index

class BadStream(Stream):
    """PyGStream::Read could formerly overflow buffer if the python implementation
    returned more data than requested.
    """

    def Read(self, amount):
        if False:
            while True:
                i = 10
        return b'x' * (amount + 1)

class StreamTest(win32com.test.util.TestCase):

    def _readWrite(self, data, write_stream, read_stream=None):
        if False:
            i = 10
            return i + 15
        if read_stream is None:
            read_stream = write_stream
        write_stream.Write(data)
        read_stream.Seek(0, pythoncom.STREAM_SEEK_SET)
        got = read_stream.Read(len(data))
        self.assertEqual(data, got)
        read_stream.Seek(1, pythoncom.STREAM_SEEK_SET)
        got = read_stream.Read(len(data) - 2)
        self.assertEqual(data[1:-1], got)

    def testit(self):
        if False:
            i = 10
            return i + 15
        mydata = b'abcdefghijklmnopqrstuvwxyz'
        s = Stream(mydata)
        p = Persists()
        p.Load(s)
        p.Save(s, 0)
        self.assertEqual(s.data, mydata)
        s2 = win32com.server.util.wrap(s, pythoncom.IID_IStream)
        p2 = win32com.server.util.wrap(p, pythoncom.IID_IPersistStreamInit)
        self._readWrite(mydata, s, s)
        self._readWrite(mydata, s, s2)
        self._readWrite(mydata, s2, s)
        self._readWrite(mydata, s2, s2)
        self._readWrite(b'string with\x00a NULL', s2, s2)
        s.Write(mydata)
        p2.Load(s2)
        p2.Save(s2, 0)
        self.assertEqual(s.data, mydata)

    def testseek(self):
        if False:
            print('Hello World!')
        s = Stream(b'yo')
        s = win32com.server.util.wrap(s, pythoncom.IID_IStream)
        s.Seek(4294967296, pythoncom.STREAM_SEEK_SET)

    def testerrors(self):
        if False:
            return 10
        (records, old_log) = win32com.test.util.setup_test_logger()
        badstream = BadStream('Check for buffer overflow')
        badstream2 = win32com.server.util.wrap(badstream, pythoncom.IID_IStream)
        self.assertRaises(pythoncom.com_error, badstream2.Read, 10)
        win32com.test.util.restore_test_logger(old_log)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].msg.startswith('pythoncom error'))
if __name__ == '__main__':
    unittest.main()