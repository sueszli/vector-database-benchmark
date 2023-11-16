import datetime
import os
import random
import shutil
import socket
import tempfile
import threading
import time
import unittest
import ntsecuritycon
import pywintypes
import win32api
import win32con
import win32event
import win32file
import win32pipe
import win32timezone
import winerror
from pywin32_testutil import TestSkipped, testmain

class TestReadBuffer(unittest.TestCase):

    def testLen(self):
        if False:
            i = 10
            return i + 15
        buffer = win32file.AllocateReadBuffer(1)
        self.assertEqual(len(buffer), 1)

    def testSimpleIndex(self):
        if False:
            for i in range(10):
                print('nop')
        buffer = win32file.AllocateReadBuffer(1)
        buffer[0] = 255
        self.assertEqual(buffer[0], 255)

    def testSimpleSlice(self):
        if False:
            i = 10
            return i + 15
        buffer = win32file.AllocateReadBuffer(2)
        val = b'\x00\x00'
        buffer[:2] = val
        self.assertEqual(buffer[0:2], val)

class TestSimpleOps(unittest.TestCase):

    def testSimpleFiles(self):
        if False:
            print('Hello World!')
        (fd, filename) = tempfile.mkstemp()
        os.close(fd)
        os.unlink(filename)
        handle = win32file.CreateFile(filename, win32file.GENERIC_WRITE, 0, None, win32con.CREATE_NEW, 0, None)
        test_data = b'Hello\x00there'
        try:
            win32file.WriteFile(handle, test_data)
            handle.Close()
            handle = win32file.CreateFile(filename, win32file.GENERIC_READ, 0, None, win32con.OPEN_EXISTING, 0, None)
            (rc, data) = win32file.ReadFile(handle, 1024)
            self.assertEqual(data, test_data)
        finally:
            handle.Close()
            try:
                os.unlink(filename)
            except OSError:
                pass

    def testMoreFiles(self):
        if False:
            return 10
        testName = os.path.join(win32api.GetTempPath(), 'win32filetest.dat')
        desiredAccess = win32file.GENERIC_READ | win32file.GENERIC_WRITE
        fileFlags = win32file.FILE_FLAG_DELETE_ON_CLOSE
        h = win32file.CreateFile(testName, desiredAccess, win32file.FILE_SHARE_READ, None, win32file.CREATE_ALWAYS, fileFlags, 0)
        data = b'z' * 1025
        win32file.WriteFile(h, data)
        self.assertTrue(win32file.GetFileSize(h) == len(data), 'WARNING: Written file does not have the same size as the length of the data in it!')
        win32file.SetFilePointer(h, 0, win32file.FILE_BEGIN)
        (hr, read_data) = win32file.ReadFile(h, len(data) + 10)
        self.assertTrue(hr == 0, 'Readfile returned %d' % hr)
        self.assertTrue(read_data == data, 'Read data is not what we wrote!')
        newSize = len(data) // 2
        win32file.SetFilePointer(h, newSize, win32file.FILE_BEGIN)
        win32file.SetEndOfFile(h)
        self.assertEqual(win32file.GetFileSize(h), newSize)
        self.assertEqual(win32file.GetFileAttributesEx(testName), win32file.GetFileAttributesExW(testName))
        (attr, ct, at, wt, size) = win32file.GetFileAttributesEx(testName)
        self.assertTrue(size == newSize, 'Expected GetFileAttributesEx to return the same size as GetFileSize()')
        self.assertTrue(attr == win32file.GetFileAttributes(testName), 'Expected GetFileAttributesEx to return the same attributes as GetFileAttributes')
        h = None
        self.assertTrue(not os.path.isfile(testName), 'After closing the file, it still exists!')

    def testFilePointer(self):
        if False:
            for i in range(10):
                print('nop')
        filename = os.path.join(win32api.GetTempPath(), 'win32filetest.dat')
        f = win32file.CreateFile(filename, win32file.GENERIC_READ | win32file.GENERIC_WRITE, 0, None, win32file.CREATE_ALWAYS, win32file.FILE_ATTRIBUTE_NORMAL, 0)
        try:
            data = b'Some data'
            (res, written) = win32file.WriteFile(f, data)
            self.assertFalse(res)
            self.assertEqual(written, len(data))
            win32file.SetFilePointer(f, 0, win32file.FILE_BEGIN)
            (res, s) = win32file.ReadFile(f, len(data))
            self.assertFalse(res)
            self.assertEqual(s, data)
            win32file.SetFilePointer(f, -len(data), win32file.FILE_END)
            (res, s) = win32file.ReadFile(f, len(data))
            self.assertFalse(res)
            self.assertEqual(s, data)
        finally:
            f.Close()
            os.unlink(filename)

    def testFileTimesTimezones(self):
        if False:
            while True:
                i = 10
        filename = tempfile.mktemp('-testFileTimes')
        now_utc = win32timezone.utcnow().replace(microsecond=0)
        now_local = now_utc.astimezone(win32timezone.TimeZoneInfo.local())
        h = win32file.CreateFile(filename, win32file.GENERIC_READ | win32file.GENERIC_WRITE, 0, None, win32file.CREATE_ALWAYS, 0, 0)
        try:
            win32file.SetFileTime(h, now_utc, now_utc, now_utc)
            (ct, at, wt) = win32file.GetFileTime(h)
            self.assertEqual(now_local, ct)
            self.assertEqual(now_local, at)
            self.assertEqual(now_local, wt)
            win32file.SetFileTime(h, now_local, now_local, now_local)
            (ct, at, wt) = win32file.GetFileTime(h)
            self.assertEqual(now_utc, ct)
            self.assertEqual(now_utc, at)
            self.assertEqual(now_utc, wt)
        finally:
            h.close()
            os.unlink(filename)

    def testFileTimes(self):
        if False:
            while True:
                i = 10
        from win32timezone import TimeZoneInfo
        now = datetime.datetime.now(tz=TimeZoneInfo.utc()).replace(microsecond=0)
        nowish = now + datetime.timedelta(seconds=1)
        later = now + datetime.timedelta(seconds=120)
        filename = tempfile.mktemp('-testFileTimes')
        open(filename, 'w').close()
        f = win32file.CreateFile(filename, win32file.GENERIC_READ | win32file.GENERIC_WRITE, 0, None, win32con.OPEN_EXISTING, 0, None)
        try:
            (ct, at, wt) = win32file.GetFileTime(f)
            self.assertTrue(ct >= now, f'File was created in the past - now={now}, created={ct}')
            self.assertTrue(now <= ct <= nowish, (now, ct))
            self.assertTrue(wt >= now, f'File was written-to in the past now={now}, written={wt}')
            self.assertTrue(now <= wt <= nowish, (now, wt))
            win32file.SetFileTime(f, later, later, later, UTCTimes=True)
            (ct, at, wt) = win32file.GetFileTime(f)
            self.assertEqual(ct, later)
            self.assertEqual(at, later)
            self.assertEqual(wt, later)
        finally:
            f.Close()
            os.unlink(filename)

class TestGetFileInfoByHandleEx(unittest.TestCase):
    __handle = __filename = None

    def setUp(self):
        if False:
            i = 10
            return i + 15
        (fd, self.__filename) = tempfile.mkstemp()
        os.close(fd)

    def tearDown(self):
        if False:
            print('Hello World!')
        if self.__handle is not None:
            self.__handle.Close()
        if self.__filename is not None:
            try:
                os.unlink(self.__filename)
            except OSError:
                pass
        self.__handle = self.__filename = None

    def testFileBasicInfo(self):
        if False:
            return 10
        attr = win32file.GetFileAttributes(self.__filename)
        f = win32file.CreateFile(self.__filename, win32file.GENERIC_READ, 0, None, win32con.OPEN_EXISTING, 0, None)
        self.__handle = f
        (ct, at, wt) = win32file.GetFileTime(f)
        basic_info = win32file.GetFileInformationByHandleEx(f, win32file.FileBasicInfo)
        self.assertEqual(ct, basic_info['CreationTime'])
        self.assertEqual(at, basic_info['LastAccessTime'])
        self.assertEqual(wt, basic_info['LastWriteTime'])
        self.assertEqual(attr, basic_info['FileAttributes'])

class TestOverlapped(unittest.TestCase):

    def testSimpleOverlapped(self):
        if False:
            print('Hello World!')
        import win32event
        testName = os.path.join(win32api.GetTempPath(), 'win32filetest.dat')
        desiredAccess = win32file.GENERIC_WRITE
        overlapped = pywintypes.OVERLAPPED()
        evt = win32event.CreateEvent(None, 0, 0, None)
        overlapped.hEvent = evt
        h = win32file.CreateFile(testName, desiredAccess, 0, None, win32file.CREATE_ALWAYS, 0, 0)
        chunk_data = b'z' * 32768
        num_loops = 512
        expected_size = num_loops * len(chunk_data)
        for i in range(num_loops):
            win32file.WriteFile(h, chunk_data, overlapped)
            win32event.WaitForSingleObject(overlapped.hEvent, win32event.INFINITE)
            overlapped.Offset = overlapped.Offset + len(chunk_data)
        h.Close()
        overlapped = pywintypes.OVERLAPPED()
        evt = win32event.CreateEvent(None, 0, 0, None)
        overlapped.hEvent = evt
        desiredAccess = win32file.GENERIC_READ
        h = win32file.CreateFile(testName, desiredAccess, 0, None, win32file.OPEN_EXISTING, 0, 0)
        buffer = win32file.AllocateReadBuffer(65535)
        while 1:
            try:
                (hr, data) = win32file.ReadFile(h, buffer, overlapped)
                win32event.WaitForSingleObject(overlapped.hEvent, win32event.INFINITE)
                overlapped.Offset = overlapped.Offset + len(data)
                if not data is buffer:
                    self.fail('Unexpected result from ReadFile - should be the same buffer we passed it')
            except win32api.error:
                break
        h.Close()

    def testCompletionPortsMultiple(self):
        if False:
            i = 10
            return i + 15
        ioport = win32file.CreateIoCompletionPort(win32file.INVALID_HANDLE_VALUE, 0, 0, 0)
        socks = []
        for PORT in range(9123, 9125):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', PORT))
            sock.listen(1)
            socks.append(sock)
            new = win32file.CreateIoCompletionPort(sock.fileno(), ioport, PORT, 0)
            assert new is ioport
        for s in socks:
            s.close()
        hv = int(ioport)
        ioport = new = None
        try:
            win32file.CloseHandle(hv)
            raise RuntimeError('Expected close to fail!')
        except win32file.error as details:
            self.assertEqual(details.winerror, winerror.ERROR_INVALID_HANDLE)

    def testCompletionPortsQueued(self):
        if False:
            while True:
                i = 10

        class Foo:
            pass
        io_req_port = win32file.CreateIoCompletionPort(-1, None, 0, 0)
        overlapped = pywintypes.OVERLAPPED()
        overlapped.object = Foo()
        win32file.PostQueuedCompletionStatus(io_req_port, 0, 99, overlapped)
        (errCode, bytes, key, overlapped) = win32file.GetQueuedCompletionStatus(io_req_port, win32event.INFINITE)
        self.assertEqual(errCode, 0)
        self.assertTrue(isinstance(overlapped.object, Foo))

    def _IOCPServerThread(self, handle, port, drop_overlapped_reference):
        if False:
            i = 10
            return i + 15
        overlapped = pywintypes.OVERLAPPED()
        win32pipe.ConnectNamedPipe(handle, overlapped)
        if drop_overlapped_reference:
            overlapped = None
            try:
                self.assertRaises(RuntimeError, win32file.GetQueuedCompletionStatus, port, -1)
            finally:
                handle.Close()
            return
        result = win32file.GetQueuedCompletionStatus(port, -1)
        ol2 = result[-1]
        self.assertTrue(ol2 is overlapped)
        data = win32file.ReadFile(handle, 512)[1]
        win32file.WriteFile(handle, data)

    def testCompletionPortsNonQueued(self, test_overlapped_death=0):
        if False:
            while True:
                i = 10
        BUFSIZE = 512
        pipe_name = '\\\\.\\pipe\\pywin32_test_pipe'
        handle = win32pipe.CreateNamedPipe(pipe_name, win32pipe.PIPE_ACCESS_DUPLEX | win32file.FILE_FLAG_OVERLAPPED, win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT, 1, BUFSIZE, BUFSIZE, win32pipe.NMPWAIT_WAIT_FOREVER, None)
        port = win32file.CreateIoCompletionPort(-1, 0, 0, 0)
        win32file.CreateIoCompletionPort(handle, port, 1, 0)
        t = threading.Thread(target=self._IOCPServerThread, args=(handle, port, test_overlapped_death))
        t.setDaemon(True)
        t.start()
        try:
            time.sleep(0.1)
            try:
                win32pipe.CallNamedPipe('\\\\.\\pipe\\pywin32_test_pipe', b'Hello there', BUFSIZE, 0)
            except win32pipe.error:
                if not test_overlapped_death:
                    raise
        finally:
            if not test_overlapped_death:
                handle.Close()
            t.join(3)
            self.assertFalse(t.is_alive(), "thread didn't finish")

    def testCompletionPortsNonQueuedBadReference(self):
        if False:
            while True:
                i = 10
        self.testCompletionPortsNonQueued(True)

    def testHashable(self):
        if False:
            for i in range(10):
                print('nop')
        overlapped = pywintypes.OVERLAPPED()
        d = {}
        d[overlapped] = 'hello'
        self.assertEqual(d[overlapped], 'hello')

    def testComparable(self):
        if False:
            i = 10
            return i + 15
        overlapped = pywintypes.OVERLAPPED()
        self.assertEqual(overlapped, overlapped)
        self.assertTrue(overlapped == overlapped)
        self.assertFalse(overlapped != overlapped)

    def testComparable2(self):
        if False:
            return 10
        overlapped1 = pywintypes.OVERLAPPED()
        overlapped2 = pywintypes.OVERLAPPED()
        self.assertEqual(overlapped1, overlapped2)
        self.assertTrue(overlapped1 == overlapped2)
        self.assertFalse(overlapped1 != overlapped2)
        overlapped1.hEvent = 1
        self.assertNotEqual(overlapped1, overlapped2)
        self.assertFalse(overlapped1 == overlapped2)
        self.assertTrue(overlapped1 != overlapped2)

class TestSocketExtensions(unittest.TestCase):

    def acceptWorker(self, port, running_event, stopped_event):
        if False:
            while True:
                i = 10
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(('', port))
        listener.listen(200)
        accepter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        overlapped = pywintypes.OVERLAPPED()
        overlapped.hEvent = win32event.CreateEvent(None, 0, 0, None)
        buffer = ' ' * 1024
        self.assertRaises(TypeError, win32file.AcceptEx, listener, accepter, buffer, overlapped)
        buffer = win32file.AllocateReadBuffer(1024)
        rc = win32file.AcceptEx(listener, accepter, buffer, overlapped)
        self.assertEqual(rc, winerror.ERROR_IO_PENDING)
        running_event.set()
        rc = win32event.WaitForSingleObject(overlapped.hEvent, 2000)
        if rc == win32event.WAIT_TIMEOUT:
            self.fail('timed out waiting for a connection')
        nbytes = win32file.GetOverlappedResult(listener.fileno(), overlapped, False)
        accepter.send(buffer[:nbytes])
        stopped_event.set()

    def testAcceptEx(self):
        if False:
            while True:
                i = 10
        port = 4680
        running = threading.Event()
        stopped = threading.Event()
        t = threading.Thread(target=self.acceptWorker, args=(port, running, stopped))
        t.start()
        running.wait(2)
        if not running.isSet():
            self.fail('AcceptEx Worker thread failed to start')
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('127.0.0.1', port))
        win32file.WSASend(s, b'hello', None)
        overlapped = pywintypes.OVERLAPPED()
        overlapped.hEvent = win32event.CreateEvent(None, 0, 0, None)
        buffer = ' ' * 10
        self.assertRaises(TypeError, win32file.WSARecv, s, buffer, overlapped)
        buffer = win32file.AllocateReadBuffer(10)
        win32file.WSARecv(s, buffer, overlapped)
        nbytes = win32file.GetOverlappedResult(s.fileno(), overlapped, True)
        got = buffer[:nbytes]
        self.assertEqual(got, b'hello')
        stopped.wait(2)
        if not stopped.isSet():
            self.fail('AcceptEx Worker thread failed to successfully stop')

class TestFindFiles(unittest.TestCase):

    def testIter(self):
        if False:
            i = 10
            return i + 15
        dir = os.path.join(os.getcwd(), '*')
        files = win32file.FindFilesW(dir)
        set1 = set()
        set1.update(files)
        set2 = set()
        for file in win32file.FindFilesIterator(dir):
            set2.add(file)
        assert len(set2) > 5, 'This directory has less than 5 files!?'
        self.assertEqual(set1, set2)

    def testBadDir(self):
        if False:
            print('Hello World!')
        dir = os.path.join(os.getcwd(), 'a dir that doesnt exist', '*')
        self.assertRaises(win32file.error, win32file.FindFilesIterator, dir)

    def testEmptySpec(self):
        if False:
            i = 10
            return i + 15
        spec = os.path.join(os.getcwd(), '*.foo_bar')
        num = 0
        for i in win32file.FindFilesIterator(spec):
            num += 1
        self.assertEqual(0, num)

    def testEmptyDir(self):
        if False:
            return 10
        test_path = os.path.join(win32api.GetTempPath(), 'win32file_test_directory')
        try:
            os.rmdir(test_path)
        except OSError:
            pass
        os.mkdir(test_path)
        try:
            num = 0
            for i in win32file.FindFilesIterator(os.path.join(test_path, '*')):
                num += 1
            self.assertEqual(2, num)
        finally:
            os.rmdir(test_path)

class TestDirectoryChanges(unittest.TestCase):
    num_test_dirs = 1

    def setUp(self):
        if False:
            print('Hello World!')
        self.watcher_threads = []
        self.watcher_thread_changes = []
        self.dir_names = []
        self.dir_handles = []
        for i in range(self.num_test_dirs):
            td = tempfile.mktemp('-test-directory-changes-%d' % i)
            os.mkdir(td)
            self.dir_names.append(td)
            hdir = win32file.CreateFile(td, ntsecuritycon.FILE_LIST_DIRECTORY, win32con.FILE_SHARE_READ, None, win32con.OPEN_EXISTING, win32con.FILE_FLAG_BACKUP_SEMANTICS | win32con.FILE_FLAG_OVERLAPPED, None)
            self.dir_handles.append(hdir)
            changes = []
            t = threading.Thread(target=self._watcherThreadOverlapped, args=(td, hdir, changes))
            t.start()
            self.watcher_threads.append(t)
            self.watcher_thread_changes.append(changes)

    def _watcherThread(self, dn, dh, changes):
        if False:
            while True:
                i = 10
        flags = win32con.FILE_NOTIFY_CHANGE_FILE_NAME
        while 1:
            try:
                print('waiting', dh)
                changes = win32file.ReadDirectoryChangesW(dh, 8192, False, flags)
                print('got', changes)
            except:
                raise
            changes.extend(changes)

    def _watcherThreadOverlapped(self, dn, dh, changes):
        if False:
            for i in range(10):
                print('nop')
        flags = win32con.FILE_NOTIFY_CHANGE_FILE_NAME
        buf = win32file.AllocateReadBuffer(8192)
        overlapped = pywintypes.OVERLAPPED()
        overlapped.hEvent = win32event.CreateEvent(None, 0, 0, None)
        while 1:
            win32file.ReadDirectoryChangesW(dh, buf, False, flags, overlapped)
            rc = win32event.WaitForSingleObject(overlapped.hEvent, 5000)
            if rc == win32event.WAIT_OBJECT_0:
                nbytes = win32file.GetOverlappedResult(dh, overlapped, True)
                if nbytes:
                    bits = win32file.FILE_NOTIFY_INFORMATION(buf, nbytes)
                    changes.extend(bits)
                else:
                    return
            else:
                print('ERROR: Watcher thread timed-out!')
                return

    def tearDown(self):
        if False:
            print('Hello World!')
        for h in self.dir_handles:
            h.Close()
        for dn in self.dir_names:
            try:
                shutil.rmtree(dn)
            except OSError:
                print('FAILED to remove directory', dn)
        for t in self.watcher_threads:
            t.join(5)
            if t.is_alive():
                print('FAILED to wait for thread termination')

    def stablize(self):
        if False:
            i = 10
            return i + 15
        time.sleep(0.5)

    def testSimple(self):
        if False:
            return 10
        self.stablize()
        for dn in self.dir_names:
            fn = os.path.join(dn, 'test_file')
            open(fn, 'w').close()
        self.stablize()
        changes = self.watcher_thread_changes[0]
        self.assertEqual(changes, [(1, 'test_file')])

    def testSmall(self):
        if False:
            print('Hello World!')
        self.stablize()
        for dn in self.dir_names:
            fn = os.path.join(dn, 'x')
            open(fn, 'w').close()
        self.stablize()
        changes = self.watcher_thread_changes[0]
        self.assertEqual(changes, [(1, 'x')])

class TestEncrypt(unittest.TestCase):

    def testEncrypt(self):
        if False:
            print('Hello World!')
        fname = tempfile.mktemp('win32file_test')
        f = open(fname, 'wb')
        f.write(b'hello')
        f.close()
        f = None
        try:
            try:
                win32file.EncryptFile(fname)
            except win32file.error as details:
                if details.winerror != winerror.ERROR_ACCESS_DENIED:
                    raise
                print('It appears this is not NTFS - cant encrypt/decrypt')
            win32file.DecryptFile(fname)
        finally:
            if f is not None:
                f.close()
            os.unlink(fname)

class TestConnect(unittest.TestCase):

    def connect_thread_runner(self, expect_payload, giveup_event):
        if False:
            i = 10
            return i + 15
        listener = socket.socket()
        self.addr = ('localhost', random.randint(10000, 64000))
        listener.bind(self.addr)
        listener.listen(1)
        accepter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        overlapped = pywintypes.OVERLAPPED()
        overlapped.hEvent = win32event.CreateEvent(None, 0, 0, None)
        if expect_payload:
            buf_size = 1024
        else:
            buf_size = win32file.CalculateSocketEndPointSize(listener)
        buffer = win32file.AllocateReadBuffer(buf_size)
        win32file.AcceptEx(listener, accepter, buffer, overlapped)
        events = (giveup_event, overlapped.hEvent)
        rc = win32event.WaitForMultipleObjects(events, False, 2000)
        if rc == win32event.WAIT_TIMEOUT:
            self.fail('timed out waiting for a connection')
        if rc == win32event.WAIT_OBJECT_0:
            return
        nbytes = win32file.GetOverlappedResult(listener.fileno(), overlapped, False)
        if expect_payload:
            self.request = buffer[:nbytes]
        accepter.send(b'some expected response')

    def test_connect_with_payload(self):
        if False:
            return 10
        giveup_event = win32event.CreateEvent(None, 0, 0, None)
        t = threading.Thread(target=self.connect_thread_runner, args=(True, giveup_event))
        t.start()
        time.sleep(0.1)
        s2 = socket.socket()
        ol = pywintypes.OVERLAPPED()
        s2.bind(('0.0.0.0', 0))
        try:
            win32file.ConnectEx(s2, self.addr, ol, b'some expected request')
        except win32file.error as exc:
            win32event.SetEvent(giveup_event)
            if exc.winerror == 10022:
                raise TestSkipped('ConnectEx is not available on this platform')
            raise
        try:
            win32file.GetOverlappedResult(s2.fileno(), ol, 1)
        except win32file.error as exc:
            win32event.SetEvent(giveup_event)
            if exc.winerror == winerror.ERROR_CONNECTION_REFUSED:
                raise TestSkipped('Assuming ERROR_CONNECTION_REFUSED is transient')
            raise
        ol = pywintypes.OVERLAPPED()
        buff = win32file.AllocateReadBuffer(1024)
        win32file.WSARecv(s2, buff, ol, 0)
        length = win32file.GetOverlappedResult(s2.fileno(), ol, 1)
        self.response = buff[:length]
        self.assertEqual(self.response, b'some expected response')
        self.assertEqual(self.request, b'some expected request')
        t.join(5)
        self.assertFalse(t.is_alive(), "worker thread didn't terminate")

    def test_connect_without_payload(self):
        if False:
            for i in range(10):
                print('nop')
        giveup_event = win32event.CreateEvent(None, 0, 0, None)
        t = threading.Thread(target=self.connect_thread_runner, args=(False, giveup_event))
        t.start()
        time.sleep(0.1)
        s2 = socket.socket()
        ol = pywintypes.OVERLAPPED()
        s2.bind(('0.0.0.0', 0))
        try:
            win32file.ConnectEx(s2, self.addr, ol)
        except win32file.error as exc:
            win32event.SetEvent(giveup_event)
            if exc.winerror == 10022:
                raise TestSkipped('ConnectEx is not available on this platform')
            raise
        try:
            win32file.GetOverlappedResult(s2.fileno(), ol, 1)
        except win32file.error as exc:
            win32event.SetEvent(giveup_event)
            if exc.winerror == winerror.ERROR_CONNECTION_REFUSED:
                raise TestSkipped('Assuming ERROR_CONNECTION_REFUSED is transient')
            raise
        ol = pywintypes.OVERLAPPED()
        buff = win32file.AllocateReadBuffer(1024)
        win32file.WSARecv(s2, buff, ol, 0)
        length = win32file.GetOverlappedResult(s2.fileno(), ol, 1)
        self.response = buff[:length]
        self.assertEqual(self.response, b'some expected response')
        t.join(5)
        self.assertFalse(t.is_alive(), "worker thread didn't terminate")

class TestTransmit(unittest.TestCase):

    def test_transmit(self):
        if False:
            i = 10
            return i + 15
        import binascii
        bytes = os.urandom(1024 * 1024)
        val = binascii.hexlify(bytes)
        val_length = len(val)
        f = tempfile.TemporaryFile()
        f.write(val)

        def runner():
            if False:
                for i in range(10):
                    print('nop')
            s1 = socket.socket()
            for i in range(5):
                self.addr = ('localhost', random.randint(10000, 64000))
                try:
                    s1.bind(self.addr)
                    break
                except OSError as exc:
                    if exc.winerror != 10013:
                        raise
                    print('Failed to use port', self.addr, 'trying another random one')
            else:
                raise RuntimeError('Failed to find an available port to bind to.')
            s1.listen(1)
            (cli, addr) = s1.accept()
            buf = 1
            self.request = []
            while buf:
                buf = cli.recv(1024 * 100)
                self.request.append(buf)
        th = threading.Thread(target=runner)
        th.start()
        time.sleep(0.5)
        s2 = socket.socket()
        s2.connect(self.addr)
        length = 0
        aaa = b'[AAA]'
        bbb = b'[BBB]'
        ccc = b'[CCC]'
        ddd = b'[DDD]'
        empty = b''
        ol = pywintypes.OVERLAPPED()
        f.seek(0)
        win32file.TransmitFile(s2, win32file._get_osfhandle(f.fileno()), val_length, 0, ol, 0)
        length += win32file.GetOverlappedResult(s2.fileno(), ol, 1)
        ol = pywintypes.OVERLAPPED()
        f.seek(0)
        win32file.TransmitFile(s2, win32file._get_osfhandle(f.fileno()), val_length, 0, ol, 0, aaa, bbb)
        length += win32file.GetOverlappedResult(s2.fileno(), ol, 1)
        ol = pywintypes.OVERLAPPED()
        f.seek(0)
        win32file.TransmitFile(s2, win32file._get_osfhandle(f.fileno()), val_length, 0, ol, 0, empty, empty)
        length += win32file.GetOverlappedResult(s2.fileno(), ol, 1)
        ol = pywintypes.OVERLAPPED()
        f.seek(0)
        win32file.TransmitFile(s2, win32file._get_osfhandle(f.fileno()), val_length, 0, ol, 0, None, ccc)
        length += win32file.GetOverlappedResult(s2.fileno(), ol, 1)
        ol = pywintypes.OVERLAPPED()
        f.seek(0)
        win32file.TransmitFile(s2, win32file._get_osfhandle(f.fileno()), val_length, 0, ol, 0, ddd)
        length += win32file.GetOverlappedResult(s2.fileno(), ol, 1)
        s2.close()
        th.join()
        buf = b''.join(self.request)
        self.assertEqual(length, len(buf))
        expected = val + aaa + val + bbb + val + val + ccc + ddd + val
        self.assertEqual(type(expected), type(buf))
        self.assertEqual(expected, buf)

class TestWSAEnumNetworkEvents(unittest.TestCase):

    def test_basics(self):
        if False:
            i = 10
            return i + 15
        s = socket.socket()
        e = win32event.CreateEvent(None, 1, 0, None)
        win32file.WSAEventSelect(s, e, 0)
        self.assertEqual(win32file.WSAEnumNetworkEvents(s), {})
        self.assertEqual(win32file.WSAEnumNetworkEvents(s, e), {})
        self.assertRaises(TypeError, win32file.WSAEnumNetworkEvents, s, e, 3)
        self.assertRaises(TypeError, win32file.WSAEnumNetworkEvents, s, 'spam')
        self.assertRaises(TypeError, win32file.WSAEnumNetworkEvents, 'spam', e)
        self.assertRaises(TypeError, win32file.WSAEnumNetworkEvents, 'spam')
        f = open('NUL')
        h = win32file._get_osfhandle(f.fileno())
        self.assertRaises(win32file.error, win32file.WSAEnumNetworkEvents, h)
        self.assertRaises(win32file.error, win32file.WSAEnumNetworkEvents, s, h)
        try:
            win32file.WSAEnumNetworkEvents(h)
        except win32file.error as e:
            self.assertEqual(e.winerror, win32file.WSAENOTSOCK)
        try:
            win32file.WSAEnumNetworkEvents(s, h)
        except win32file.error as e:
            self.assertEqual(e.winerror, win32file.WSAENOTSOCK)

    def test_functional(self):
        if False:
            i = 10
            return i + 15
        port = socket.socket()
        port.setblocking(0)
        port_event = win32event.CreateEvent(None, 0, 0, None)
        win32file.WSAEventSelect(port, port_event, win32file.FD_ACCEPT | win32file.FD_CLOSE)
        port.bind(('127.0.0.1', 0))
        port.listen(10)
        client = socket.socket()
        client.setblocking(0)
        client_event = win32event.CreateEvent(None, 0, 0, None)
        win32file.WSAEventSelect(client, client_event, win32file.FD_CONNECT | win32file.FD_READ | win32file.FD_WRITE | win32file.FD_CLOSE)
        err = client.connect_ex(port.getsockname())
        self.assertEqual(err, win32file.WSAEWOULDBLOCK)
        res = win32event.WaitForSingleObject(port_event, 1000)
        self.assertEqual(res, win32event.WAIT_OBJECT_0)
        events = win32file.WSAEnumNetworkEvents(port, port_event)
        self.assertEqual(events, {win32file.FD_ACCEPT: 0})
        (server, addr) = port.accept()
        server.setblocking(0)
        server_event = win32event.CreateEvent(None, 1, 0, None)
        win32file.WSAEventSelect(server, server_event, win32file.FD_READ | win32file.FD_WRITE | win32file.FD_CLOSE)
        res = win32event.WaitForSingleObject(server_event, 1000)
        self.assertEqual(res, win32event.WAIT_OBJECT_0)
        events = win32file.WSAEnumNetworkEvents(server, server_event)
        self.assertEqual(events, {win32file.FD_WRITE: 0})
        res = win32event.WaitForSingleObject(client_event, 1000)
        self.assertEqual(res, win32event.WAIT_OBJECT_0)
        events = win32file.WSAEnumNetworkEvents(client, client_event)
        self.assertEqual(events, {win32file.FD_CONNECT: 0, win32file.FD_WRITE: 0})
        sent = 0
        data = b'x' * 16 * 1024
        while sent < 16 * 1024 * 1024:
            try:
                sent += client.send(data)
            except OSError as e:
                if e.args[0] == win32file.WSAEINTR:
                    continue
                elif e.args[0] in (win32file.WSAEWOULDBLOCK, win32file.WSAENOBUFS):
                    break
                else:
                    raise
        else:
            self.fail('could not find socket buffer limit')
        events = win32file.WSAEnumNetworkEvents(client)
        self.assertEqual(events, {})
        res = win32event.WaitForSingleObject(server_event, 1000)
        self.assertEqual(res, win32event.WAIT_OBJECT_0)
        events = win32file.WSAEnumNetworkEvents(server, server_event)
        self.assertEqual(events, {win32file.FD_READ: 0})
        received = 0
        while received < sent:
            try:
                received += len(server.recv(16 * 1024))
            except OSError as e:
                if e.args[0] in [win32file.WSAEINTR, win32file.WSAEWOULDBLOCK]:
                    continue
                else:
                    raise
        self.assertEqual(received, sent)
        events = win32file.WSAEnumNetworkEvents(server)
        self.assertEqual(events, {})
        res = win32event.WaitForSingleObject(client_event, 1000)
        self.assertEqual(res, win32event.WAIT_OBJECT_0)
        events = win32file.WSAEnumNetworkEvents(client, client_event)
        self.assertEqual(events, {win32file.FD_WRITE: 0})
        client.shutdown(socket.SHUT_WR)
        res = win32event.WaitForSingleObject(server_event, 1000)
        self.assertEqual(res, win32event.WAIT_OBJECT_0)
        for i in range(5):
            events = win32file.WSAEnumNetworkEvents(server, server_event)
            if events:
                break
            win32api.Sleep(100)
        else:
            raise AssertionError('failed to get events')
        self.assertEqual(events, {win32file.FD_CLOSE: 0})
        events = win32file.WSAEnumNetworkEvents(client)
        self.assertEqual(events, {})
        server.close()
        res = win32event.WaitForSingleObject(client_event, 1000)
        self.assertEqual(res, win32event.WAIT_OBJECT_0)
        events = win32file.WSAEnumNetworkEvents(client, client_event)
        self.assertEqual(events, {win32file.FD_CLOSE: 0})
        client.close()
        events = win32file.WSAEnumNetworkEvents(port)
        self.assertEqual(events, {})
if __name__ == '__main__':
    testmain()