"""Testing pasing object between multiple COM threads

Uses standard COM marshalling to pass objects between threads.  Even
though Python generally seems to work when you just pass COM objects
between threads, it shouldnt.

This shows the "correct" way to do it.

It shows that although we create new threads to use the Python.Interpreter,
COM marshalls back all calls to that object to the main Python thread,
which must be running a message loop (as this sample does).

When this test is run in "free threaded" mode (at this stage, you must
manually mark the COM objects as "ThreadingModel=Free", or run from a
service which has marked itself as free-threaded), then no marshalling
is done, and the Python.Interpreter object start doing the "expected" thing
- ie, it reports being on the same thread as its caller!

Python.exe needs a good way to mark itself as FreeThreaded - at the moment
this is a pain in the but!

"""
import threading
import unittest
import pythoncom
import win32api
import win32com.client
import win32event
from .testServers import InterpCase
freeThreaded = 1

class ThreadInterpCase(InterpCase):

    def _testInterpInThread(self, stopEvent, interp):
        if False:
            print('Hello World!')
        try:
            self._doTestInThread(interp)
        finally:
            win32event.SetEvent(stopEvent)

    def _doTestInThread(self, interp):
        if False:
            print('Hello World!')
        pythoncom.CoInitialize()
        myThread = win32api.GetCurrentThreadId()
        if freeThreaded:
            interp = pythoncom.CoGetInterfaceAndReleaseStream(interp, pythoncom.IID_IDispatch)
            interp = win32com.client.Dispatch(interp)
        interp.Exec('import win32api')
        pythoncom.CoUninitialize()

    def BeginThreadsSimpleMarshal(self, numThreads):
        if False:
            i = 10
            return i + 15
        'Creates multiple threads using simple (but slower) marshalling.\n\n        Single interpreter object, but a new stream is created per thread.\n\n        Returns the handles the threads will set when complete.\n        '
        interp = win32com.client.Dispatch('Python.Interpreter')
        events = []
        threads = []
        for i in range(numThreads):
            hEvent = win32event.CreateEvent(None, 0, 0, None)
            events.append(hEvent)
            interpStream = pythoncom.CoMarshalInterThreadInterfaceInStream(pythoncom.IID_IDispatch, interp._oleobj_)
            t = threading.Thread(target=self._testInterpInThread, args=(hEvent, interpStream))
            t.setDaemon(1)
            t.start()
            threads.append(t)
        interp = None
        return (threads, events)

    def BeginThreadsFastMarshal(self, numThreads):
        if False:
            for i in range(10):
                print('nop')
        'Creates multiple threads using fast (but complex) marshalling.\n\n        The marshal stream is created once, and each thread uses the same stream\n\n        Returns the handles the threads will set when complete.\n        '
        interp = win32com.client.Dispatch('Python.Interpreter')
        if freeThreaded:
            interp = pythoncom.CoMarshalInterThreadInterfaceInStream(pythoncom.IID_IDispatch, interp._oleobj_)
        events = []
        threads = []
        for i in range(numThreads):
            hEvent = win32event.CreateEvent(None, 0, 0, None)
            t = threading.Thread(target=self._testInterpInThread, args=(hEvent, interp))
            t.setDaemon(1)
            t.start()
            events.append(hEvent)
            threads.append(t)
        return (threads, events)

    def _DoTestMarshal(self, fn, bCoWait=0):
        if False:
            print('Hello World!')
        (threads, events) = fn(2)
        numFinished = 0
        while 1:
            try:
                if bCoWait:
                    rc = pythoncom.CoWaitForMultipleHandles(0, 2000, events)
                else:
                    rc = win32event.MsgWaitForMultipleObjects(events, 0, 2000, win32event.QS_ALLINPUT)
                if rc >= win32event.WAIT_OBJECT_0 and rc < win32event.WAIT_OBJECT_0 + len(events):
                    numFinished = numFinished + 1
                    if numFinished >= len(events):
                        break
                elif rc == win32event.WAIT_OBJECT_0 + len(events):
                    pythoncom.PumpWaitingMessages()
                else:
                    print('Waiting for thread to stop with interfaces=%d, gateways=%d' % (pythoncom._GetInterfaceCount(), pythoncom._GetGatewayCount()))
            except KeyboardInterrupt:
                break
        for t in threads:
            t.join(2)
            self.assertFalse(t.is_alive(), 'thread failed to stop!?')
        threads = None

    def testSimpleMarshal(self):
        if False:
            print('Hello World!')
        self._DoTestMarshal(self.BeginThreadsSimpleMarshal)

    def testSimpleMarshalCoWait(self):
        if False:
            print('Hello World!')
        self._DoTestMarshal(self.BeginThreadsSimpleMarshal, 1)
if __name__ == '__main__':
    unittest.main('testMarshal')