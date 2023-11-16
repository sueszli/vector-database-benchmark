import _thread
import win32api
import win32event
import win32trace
from pywin.framework import winout
outputWindow = None

def CollectorThread(stopEvent, file):
    if False:
        for i in range(10):
            print('nop')
    win32trace.InitRead()
    handle = win32trace.GetHandle()
    import win32process
    win32process.SetThreadPriority(win32api.GetCurrentThread(), win32process.THREAD_PRIORITY_BELOW_NORMAL)
    try:
        while 1:
            rc = win32event.WaitForMultipleObjects((handle, stopEvent), 0, win32event.INFINITE)
            if rc == win32event.WAIT_OBJECT_0:
                file.write(win32trace.read().replace('\x00', '<null>'))
            else:
                break
    finally:
        win32trace.TermRead()
        print('Thread dieing')

class WindowOutput(winout.WindowOutput):

    def __init__(self, *args):
        if False:
            return 10
        winout.WindowOutput.__init__(*(self,) + args)
        self.hStopThread = win32event.CreateEvent(None, 0, 0, None)
        _thread.start_new(CollectorThread, (self.hStopThread, self))

    def _StopThread(self):
        if False:
            for i in range(10):
                print('nop')
        win32event.SetEvent(self.hStopThread)
        self.hStopThread = None

    def Close(self):
        if False:
            while True:
                i = 10
        self._StopThread()
        winout.WindowOutput.Close(self)
        return rc

def MakeOutputWindow():
    if False:
        i = 10
        return i + 15
    global outputWindow
    if outputWindow is None:
        title = 'Python Trace Collector'
        outputWindow = WindowOutput(title, title)
        msg = "# This window will display output from any programs that import win32traceutil\n# win32com servers registered with '--debug' are in this category.\n"
        outputWindow.write(msg)
    outputWindow.write('')
    return outputWindow
if __name__ == '__main__':
    MakeOutputWindow()