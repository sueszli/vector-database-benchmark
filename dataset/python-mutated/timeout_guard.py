import contextlib
import threading
import os
import time
import signal
import logging
"\nSometimes CUDA devices can get stuck, 'deadlock'. In this case it is often\nbetter just the kill the process automatically. Use this guard to set a\nmaximum timespan for a python call, such as RunNet(). If it does not complete\nin time, process is killed.\n\nExample usage:\n    with timeout_guard.CompleteInTimeOrDie(10.0):\n        core.RunNet(...)\n"

class WatcherThread(threading.Thread):

    def __init__(self, timeout_secs):
        if False:
            i = 10
            return i + 15
        threading.Thread.__init__(self)
        self.timeout_secs = timeout_secs
        self.completed = False
        self.condition = threading.Condition()
        self.daemon = True
        self.caller_thread = threading.current_thread()

    def run(self):
        if False:
            i = 10
            return i + 15
        started = time.time()
        self.condition.acquire()
        while time.time() - started < self.timeout_secs and (not self.completed):
            self.condition.wait(self.timeout_secs - (time.time() - started))
        self.condition.release()
        if not self.completed:
            log = logging.getLogger('timeout_guard')
            log.error('Call did not finish in time. Timeout:{}s PID: {}'.format(self.timeout_secs, os.getpid()))

            def forcequit():
                if False:
                    print('Hello World!')
                time.sleep(10.0)
                log.info('Prepared output, dumping threads. ')
                print('Caller thread was: {}'.format(self.caller_thread))
                print('-----After force------')
                log.info('-----After force------')
                import sys
                import traceback
                code = []
                for (threadId, stack) in sys._current_frames().items():
                    if threadId == self.caller_thread.ident:
                        code.append('\n# ThreadID: %s' % threadId)
                        for (filename, lineno, name, line) in traceback.extract_stack(stack):
                            code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
                            if line:
                                code.append('  %s' % line.strip())
                print('\n'.join(code))
                log.info('\n'.join(code))
                log.error('Process did not terminate cleanly in 10 s, forcing')
                os.abort()
            forcet = threading.Thread(target=forcequit, args=())
            forcet.daemon = True
            forcet.start()
            print('Caller thread was: {}'.format(self.caller_thread))
            print('-----Before forcing------')
            import sys
            import traceback
            code = []
            for (threadId, stack) in sys._current_frames().items():
                code.append('\n# ThreadID: %s' % threadId)
                for (filename, lineno, name, line) in traceback.extract_stack(stack):
                    code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
                    if line:
                        code.append('  %s' % line.strip())
            print('\n'.join(code))
            log.info('\n'.join(code))
            os.kill(os.getpid(), signal.SIGINT)

@contextlib.contextmanager
def CompleteInTimeOrDie(timeout_secs):
    if False:
        print('Hello World!')
    watcher = WatcherThread(timeout_secs)
    watcher.start()
    yield
    watcher.completed = True
    watcher.condition.acquire()
    watcher.condition.notify()
    watcher.condition.release()

def EuthanizeIfNecessary(timeout_secs=120):
    if False:
        i = 10
        return i + 15
    '\n    Call this if you have problem with process getting stuck at shutdown.\n    It will kill the process if it does not terminate in timeout_secs.\n    '
    watcher = WatcherThread(timeout_secs)
    watcher.start()