import os
import sys
import threading
import time
import unittest
import win32trace
from pywin32_testutil import TestSkipped
if __name__ == '__main__':
    this_file = sys.argv[0]
else:
    this_file = __file__

def SkipIfCI():
    if False:
        i = 10
        return i + 15
    if 'CI' in os.environ:
        raise TestSkipped('We skip this test on CI')

def CheckNoOtherReaders():
    if False:
        return 10
    win32trace.write('Hi')
    time.sleep(0.05)
    if win32trace.read() != 'Hi':
        win32trace.TermRead()
        win32trace.TermWrite()
        raise RuntimeError('An existing win32trace reader appears to be running - please stop this process and try again')

class TestInitOps(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        SkipIfCI()
        win32trace.InitRead()
        win32trace.read()
        win32trace.TermRead()

    def tearDown(self):
        if False:
            print('Hello World!')
        try:
            win32trace.TermRead()
        except win32trace.error:
            pass
        try:
            win32trace.TermWrite()
        except win32trace.error:
            pass

    def testInitTermRead(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(win32trace.error, win32trace.read)
        win32trace.InitRead()
        result = win32trace.read()
        self.assertEqual(result, '')
        win32trace.TermRead()
        self.assertRaises(win32trace.error, win32trace.read)
        win32trace.InitRead()
        self.assertRaises(win32trace.error, win32trace.InitRead)
        win32trace.InitWrite()
        self.assertRaises(win32trace.error, win32trace.InitWrite)
        win32trace.TermWrite()
        win32trace.TermRead()

    def testInitTermWrite(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(win32trace.error, win32trace.write, 'Hei')
        win32trace.InitWrite()
        win32trace.write('Johan Galtung')
        win32trace.TermWrite()
        self.assertRaises(win32trace.error, win32trace.write, 'Hei')

    def testTermSematics(self):
        if False:
            while True:
                i = 10
        win32trace.InitWrite()
        win32trace.write('Ta da')
        win32trace.TermWrite()
        win32trace.InitRead()
        self.assertTrue(win32trace.read() in ('Ta da', ''))
        win32trace.TermRead()
        win32trace.InitWrite()
        win32trace.write('Ta da')
        win32trace.InitRead()
        win32trace.TermWrite()
        self.assertEqual('Ta da', win32trace.read())
        win32trace.TermRead()

class BasicSetupTearDown(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        SkipIfCI()
        win32trace.InitRead()
        win32trace.read()
        win32trace.InitWrite()

    def tearDown(self):
        if False:
            while True:
                i = 10
        win32trace.TermWrite()
        win32trace.TermRead()

class TestModuleOps(BasicSetupTearDown):

    def testRoundTrip(self):
        if False:
            return 10
        win32trace.write('Syver Enstad')
        syverEnstad = win32trace.read()
        self.assertEqual('Syver Enstad', syverEnstad)

    def testRoundTripUnicode(self):
        if False:
            return 10
        win32trace.write('©opyright Syver Enstad')
        syverEnstad = win32trace.read()
        self.assertEqual('©opyright Syver Enstad', syverEnstad)

    def testBlockingRead(self):
        if False:
            for i in range(10):
                print('nop')
        win32trace.write('Syver Enstad')
        self.assertEqual('Syver Enstad', win32trace.blockingread())

    def testBlockingReadUnicode(self):
        if False:
            for i in range(10):
                print('nop')
        win32trace.write('©opyright Syver Enstad')
        self.assertEqual('©opyright Syver Enstad', win32trace.blockingread())

    def testFlush(self):
        if False:
            print('Hello World!')
        win32trace.flush()

class TestTraceObjectOps(BasicSetupTearDown):

    def testInit(self):
        if False:
            i = 10
            return i + 15
        win32trace.TermRead()
        win32trace.TermWrite()
        traceObject = win32trace.GetTracer()
        self.assertRaises(win32trace.error, traceObject.read)
        self.assertRaises(win32trace.error, traceObject.write, '')
        win32trace.InitRead()
        win32trace.InitWrite()
        self.assertEqual('', traceObject.read())
        traceObject.write('Syver')

    def testFlush(self):
        if False:
            i = 10
            return i + 15
        traceObject = win32trace.GetTracer()
        traceObject.flush()

    def testIsatty(self):
        if False:
            print('Hello World!')
        tracer = win32trace.GetTracer()
        assert tracer.isatty() == False

    def testRoundTrip(self):
        if False:
            return 10
        traceObject = win32trace.GetTracer()
        traceObject.write('Syver Enstad')
        self.assertEqual('Syver Enstad', traceObject.read())

class WriterThread(threading.Thread):

    def run(self):
        if False:
            return 10
        self.writeCount = 0
        for each in range(self.BucketCount):
            win32trace.write(str(each))
        self.writeCount = self.BucketCount

    def verifyWritten(self):
        if False:
            i = 10
            return i + 15
        return self.writeCount == self.BucketCount

class TestMultipleThreadsWriting(unittest.TestCase):
    FullBucket = 50
    BucketCount = 9

    def setUp(self):
        if False:
            i = 10
            return i + 15
        SkipIfCI()
        WriterThread.BucketCount = self.BucketCount
        win32trace.InitRead()
        win32trace.read()
        win32trace.InitWrite()
        CheckNoOtherReaders()
        self.threads = [WriterThread() for each in range(self.FullBucket)]
        self.buckets = list(range(self.BucketCount))
        for each in self.buckets:
            self.buckets[each] = 0

    def tearDown(self):
        if False:
            return 10
        win32trace.TermRead()
        win32trace.TermWrite()

    def areBucketsFull(self):
        if False:
            print('Hello World!')
        bucketsAreFull = True
        for each in self.buckets:
            assert each <= self.FullBucket, each
            if each != self.FullBucket:
                bucketsAreFull = False
                break
        return bucketsAreFull

    def read(self):
        if False:
            print('Hello World!')
        while 1:
            readString = win32trace.blockingread()
            for ch in readString:
                integer = int(ch)
                count = self.buckets[integer]
                assert count != -1
                self.buckets[integer] = count + 1
                if self.buckets[integer] == self.FullBucket:
                    if self.areBucketsFull():
                        return

    def testThreads(self):
        if False:
            i = 10
            return i + 15
        for each in self.threads:
            each.start()
        self.read()
        for each in self.threads:
            each.join()
        for each in self.threads:
            assert each.verifyWritten()
        assert self.areBucketsFull()

class TestHugeChunks(unittest.TestCase):
    BiggestChunk = 2 ** 16

    def setUp(self):
        if False:
            while True:
                i = 10
        SkipIfCI()
        win32trace.InitRead()
        win32trace.read()
        win32trace.InitWrite()

    def testHugeChunks(self):
        if False:
            while True:
                i = 10
        data = '*' * 1023 + '\n'
        while len(data) <= self.BiggestChunk:
            win32trace.write(data)
            data = data + data

    def tearDown(self):
        if False:
            print('Hello World!')
        win32trace.TermRead()
        win32trace.TermWrite()
import win32event
import win32process

class TraceWriteProcess:

    def __init__(self, threadCount):
        if False:
            return 10
        self.exitCode = -1
        self.threadCount = threadCount

    def start(self):
        if False:
            i = 10
            return i + 15
        (procHandle, threadHandle, procId, threadId) = win32process.CreateProcess(None, 'python.exe "{}" /run_test_process {} {}'.format(this_file, self.BucketCount, self.threadCount), None, None, 0, win32process.NORMAL_PRIORITY_CLASS, None, None, win32process.STARTUPINFO())
        self.processHandle = procHandle

    def join(self):
        if False:
            for i in range(10):
                print('nop')
        win32event.WaitForSingleObject(self.processHandle, win32event.INFINITE)
        self.exitCode = win32process.GetExitCodeProcess(self.processHandle)

    def verifyWritten(self):
        if False:
            return 10
        return self.exitCode == 0

class TestOutofProcess(unittest.TestCase):
    BucketCount = 9
    FullBucket = 50

    def setUp(self):
        if False:
            while True:
                i = 10
        SkipIfCI()
        win32trace.InitRead()
        TraceWriteProcess.BucketCount = self.BucketCount
        self.setUpWriters()
        self.buckets = list(range(self.BucketCount))
        for each in self.buckets:
            self.buckets[each] = 0

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        win32trace.TermRead()

    def setUpWriters(self):
        if False:
            for i in range(10):
                print('nop')
        self.processes = []
        (quot, remainder) = divmod(self.FullBucket, 5)
        for each in range(5):
            self.processes.append(TraceWriteProcess(quot))
        if remainder:
            self.processes.append(TraceWriteProcess(remainder))

    def areBucketsFull(self):
        if False:
            i = 10
            return i + 15
        bucketsAreFull = True
        for each in self.buckets:
            assert each <= self.FullBucket, each
            if each != self.FullBucket:
                bucketsAreFull = False
                break
        return bucketsAreFull

    def read(self):
        if False:
            print('Hello World!')
        while 1:
            readString = win32trace.blockingread()
            for ch in readString:
                integer = int(ch)
                count = self.buckets[integer]
                assert count != -1
                self.buckets[integer] = count + 1
                if self.buckets[integer] == self.FullBucket:
                    if self.areBucketsFull():
                        return

    def testProcesses(self):
        if False:
            i = 10
            return i + 15
        for each in self.processes:
            each.start()
        self.read()
        for each in self.processes:
            each.join()
        for each in self.processes:
            assert each.verifyWritten()
        assert self.areBucketsFull()

def _RunAsTestProcess():
    if False:
        for i in range(10):
            print('nop')
    WriterThread.BucketCount = int(sys.argv[2])
    threadCount = int(sys.argv[3])
    threads = [WriterThread() for each in range(threadCount)]
    win32trace.InitWrite()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for t in threads:
        if not t.verifyWritten():
            sys.exit(-1)
if __name__ == '__main__':
    if sys.argv[1:2] == ['/run_test_process']:
        _RunAsTestProcess()
        sys.exit(0)
    win32trace.InitRead()
    win32trace.InitWrite()
    CheckNoOtherReaders()
    win32trace.TermRead()
    win32trace.TermWrite()
    unittest.main()