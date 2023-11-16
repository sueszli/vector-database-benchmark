import threading
import time
import unittest
import pywintypes
import win32con
import win32event
import win32file
import win32pipe
import winerror

class PipeTests(unittest.TestCase):
    pipename = '\\\\.\\pipe\\python_test_pipe'

    def _serverThread(self, pipe_handle, event, wait_time):
        if False:
            for i in range(10):
                print('nop')
        hr = win32pipe.ConnectNamedPipe(pipe_handle)
        self.assertTrue(hr in (0, winerror.ERROR_PIPE_CONNECTED), f'Got error code 0x{hr:x}')
        (hr, got) = win32file.ReadFile(pipe_handle, 100)
        self.assertEqual(got, b'foo\x00bar')
        time.sleep(wait_time)
        win32file.WriteFile(pipe_handle, b'bar\x00foo')
        pipe_handle.Close()
        event.set()

    def startPipeServer(self, event, wait_time=0):
        if False:
            for i in range(10):
                print('nop')
        openMode = win32pipe.PIPE_ACCESS_DUPLEX
        pipeMode = win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_WAIT
        sa = pywintypes.SECURITY_ATTRIBUTES()
        sa.SetSecurityDescriptorDacl(1, None, 0)
        pipe_handle = win32pipe.CreateNamedPipe(self.pipename, openMode, pipeMode, win32pipe.PIPE_UNLIMITED_INSTANCES, 0, 0, 2000, sa)
        threading.Thread(target=self._serverThread, args=(pipe_handle, event, wait_time)).start()

    def testCallNamedPipe(self):
        if False:
            i = 10
            return i + 15
        event = threading.Event()
        self.startPipeServer(event)
        got = win32pipe.CallNamedPipe(self.pipename, b'foo\x00bar', 1024, win32pipe.NMPWAIT_WAIT_FOREVER)
        self.assertEqual(got, b'bar\x00foo')
        event.wait(5)
        self.assertTrue(event.isSet(), "Pipe server thread didn't terminate")

    def testTransactNamedPipeBlocking(self):
        if False:
            print('Hello World!')
        event = threading.Event()
        self.startPipeServer(event)
        open_mode = win32con.GENERIC_READ | win32con.GENERIC_WRITE
        hpipe = win32file.CreateFile(self.pipename, open_mode, 0, None, win32con.OPEN_EXISTING, 0, None)
        win32pipe.SetNamedPipeHandleState(hpipe, win32pipe.PIPE_READMODE_MESSAGE, None, None)
        (hr, got) = win32pipe.TransactNamedPipe(hpipe, b'foo\x00bar', 1024, None)
        self.assertEqual(got, b'bar\x00foo')
        event.wait(5)
        self.assertTrue(event.isSet(), "Pipe server thread didn't terminate")

    def testTransactNamedPipeBlockingBuffer(self):
        if False:
            return 10
        event = threading.Event()
        self.startPipeServer(event)
        open_mode = win32con.GENERIC_READ | win32con.GENERIC_WRITE
        hpipe = win32file.CreateFile(self.pipename, open_mode, 0, None, win32con.OPEN_EXISTING, 0, None)
        win32pipe.SetNamedPipeHandleState(hpipe, win32pipe.PIPE_READMODE_MESSAGE, None, None)
        buffer = win32file.AllocateReadBuffer(1024)
        (hr, got) = win32pipe.TransactNamedPipe(hpipe, b'foo\x00bar', buffer, None)
        self.assertEqual(got, b'bar\x00foo')
        event.wait(5)
        self.assertTrue(event.isSet(), "Pipe server thread didn't terminate")

    def testTransactNamedPipeAsync(self):
        if False:
            for i in range(10):
                print('nop')
        event = threading.Event()
        overlapped = pywintypes.OVERLAPPED()
        overlapped.hEvent = win32event.CreateEvent(None, 0, 0, None)
        self.startPipeServer(event, 0.5)
        open_mode = win32con.GENERIC_READ | win32con.GENERIC_WRITE
        hpipe = win32file.CreateFile(self.pipename, open_mode, 0, None, win32con.OPEN_EXISTING, win32con.FILE_FLAG_OVERLAPPED, None)
        win32pipe.SetNamedPipeHandleState(hpipe, win32pipe.PIPE_READMODE_MESSAGE, None, None)
        buffer = win32file.AllocateReadBuffer(1024)
        (hr, got) = win32pipe.TransactNamedPipe(hpipe, b'foo\x00bar', buffer, overlapped)
        self.assertEqual(hr, winerror.ERROR_IO_PENDING)
        nbytes = win32file.GetOverlappedResult(hpipe, overlapped, True)
        got = buffer[:nbytes]
        self.assertEqual(got, b'bar\x00foo')
        event.wait(5)
        self.assertTrue(event.isSet(), "Pipe server thread didn't terminate")
if __name__ == '__main__':
    unittest.main()