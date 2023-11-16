"""An ISAPI extension base class implemented using a thread-pool."""
import sys
import threading
import time
import traceback
from pywintypes import OVERLAPPED
from win32event import INFINITE
from win32file import CloseHandle, CreateIoCompletionPort, GetQueuedCompletionStatus, PostQueuedCompletionStatus
from win32security import SetThreadToken
import isapi.simple
from isapi import ExtensionError, isapicon
ISAPI_REQUEST = 1
ISAPI_SHUTDOWN = 2

class WorkerThread(threading.Thread):

    def __init__(self, extension, io_req_port):
        if False:
            return 10
        self.running = False
        self.io_req_port = io_req_port
        self.extension = extension
        threading.Thread.__init__(self)
        self.setDaemon(True)

    def run(self):
        if False:
            i = 10
            return i + 15
        self.running = True
        while self.running:
            (errCode, bytes, key, overlapped) = GetQueuedCompletionStatus(self.io_req_port, INFINITE)
            if key == ISAPI_SHUTDOWN and overlapped is None:
                break
            dispatcher = self.extension.dispatch_map.get(key)
            if dispatcher is None:
                raise RuntimeError(f"Bad request '{key}'")
            dispatcher(errCode, bytes, key, overlapped)

    def call_handler(self, cblock):
        if False:
            while True:
                i = 10
        self.extension.Dispatch(cblock)

class ThreadPoolExtension(isapi.simple.SimpleExtension):
    """Base class for an ISAPI extension based around a thread-pool"""
    max_workers = 20
    worker_shutdown_wait = 15000

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.workers = []
        self.dispatch_map = {ISAPI_REQUEST: self.DispatchConnection}

    def GetExtensionVersion(self, vi):
        if False:
            i = 10
            return i + 15
        isapi.simple.SimpleExtension.GetExtensionVersion(self, vi)
        self.io_req_port = CreateIoCompletionPort(-1, None, 0, 0)
        self.workers = []
        for i in range(self.max_workers):
            worker = WorkerThread(self, self.io_req_port)
            worker.start()
            self.workers.append(worker)

    def HttpExtensionProc(self, control_block):
        if False:
            while True:
                i = 10
        overlapped = OVERLAPPED()
        overlapped.object = control_block
        PostQueuedCompletionStatus(self.io_req_port, 0, ISAPI_REQUEST, overlapped)
        return isapicon.HSE_STATUS_PENDING

    def TerminateExtension(self, status):
        if False:
            for i in range(10):
                print('nop')
        for worker in self.workers:
            worker.running = False
        for worker in self.workers:
            PostQueuedCompletionStatus(self.io_req_port, 0, ISAPI_SHUTDOWN, None)
        end_time = time.time() + self.worker_shutdown_wait / 1000
        alive = self.workers
        while alive:
            if time.time() > end_time:
                break
            time.sleep(0.2)
            alive = [w for w in alive if w.is_alive()]
        self.dispatch_map = {}
        CloseHandle(self.io_req_port)

    def DispatchConnection(self, errCode, bytes, key, overlapped):
        if False:
            return 10
        control_block = overlapped.object
        hRequestToken = control_block.GetImpersonationToken()
        SetThreadToken(None, hRequestToken)
        try:
            try:
                self.Dispatch(control_block)
            except:
                self.HandleDispatchError(control_block)
        finally:
            SetThreadToken(None, None)

    def Dispatch(self, ecb):
        if False:
            while True:
                i = 10
        'Overridden by the sub-class to handle connection requests.\n\n        This class creates a thread-pool using a Windows completion port,\n        and dispatches requests via this port.  Sub-classes can generally\n        implement each connection request using blocking reads and writes, and\n        the thread-pool will still provide decent response to the end user.\n\n        The sub-class can set a max_workers attribute (default is 20).  Note\n        that this generally does *not* mean 20 threads will all be concurrently\n        running, via the magic of Windows completion ports.\n\n        There is no default implementation - sub-classes must implement this.\n        '
        raise NotImplementedError('sub-classes should override Dispatch')

    def HandleDispatchError(self, ecb):
        if False:
            i = 10
            return i + 15
        'Handles errors in the Dispatch method.\n\n        When a Dispatch method call fails, this method is called to handle\n        the exception.  The default implementation formats the traceback\n        in the browser.\n        '
        ecb.HttpStatusCode = isapicon.HSE_STATUS_ERROR
        (exc_typ, exc_val, exc_tb) = sys.exc_info()
        limit = None
        try:
            try:
                import cgi
                ecb.SendResponseHeaders('200 OK', 'Content-type: text/html\r\n\r\n', False)
                print(file=ecb)
                print('<H3>Traceback (most recent call last):</H3>', file=ecb)
                list = traceback.format_tb(exc_tb, limit) + traceback.format_exception_only(exc_typ, exc_val)
                print('<PRE>{}<B>{}</B></PRE>'.format(cgi.escape(''.join(list[:-1])), cgi.escape(list[-1])), file=ecb)
            except ExtensionError:
                pass
            except:
                print('FAILED to render the error message!')
                traceback.print_exc()
                print('ORIGINAL extension error:')
                traceback.print_exception(exc_typ, exc_val, exc_tb)
        finally:
            exc_tb = None
            ecb.DoneWithSession()