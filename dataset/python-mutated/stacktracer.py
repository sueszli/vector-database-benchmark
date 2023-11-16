"""Stack tracer for multi-threaded applications.
Useful for debugging deadlocks and hangs.

Usage:
    import stacktracer
    stacktracer.trace_start("trace.html", interval=5)
    ...
    stacktracer.trace_stop()

This will create a file named "trace.html" showing the stack traces of all threads,
updated every 5 seconds.
"""
import os
import sys
import threading
import time
import traceback
from typing import Optional
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

def _thread_from_id(ident) -> Optional[threading.Thread]:
    if False:
        for i in range(10):
            print('nop')
    return threading._active.get(ident)

def stacktraces():
    if False:
        print('Hello World!')
    'Taken from http://bzimmer.ziclix.com/2008/12/17/python-thread-dumps/'
    code = []
    for (thread_id, stack) in sys._current_frames().items():
        thread = _thread_from_id(thread_id)
        code.append(f'\n# thread_id={thread_id}. thread={thread}')
        for (filename, lineno, name, line) in traceback.extract_stack(stack):
            code.append(f'File: "{filename}", line {lineno}, in {name}')
            if line:
                code.append('  %s' % line.strip())
    return highlight('\n'.join(code), PythonLexer(), HtmlFormatter(full=False, noclasses=True))

class TraceDumper(threading.Thread):
    """Dump stack traces into a given file periodically.

    # written by nagylzs
    """

    def __init__(self, fpath, interval, auto):
        if False:
            i = 10
            return i + 15
        '\n        @param fpath: File path to output HTML (stack trace file)\n        @param auto: Set flag (True) to update trace continuously.\n            Clear flag (False) to update only if file not exists.\n            (Then delete the file to force update.)\n        @param interval: In seconds: how often to update the trace file.\n        '
        assert interval > 0.1
        self.auto = auto
        self.interval = interval
        self.fpath = os.path.abspath(fpath)
        self.stop_requested = threading.Event()
        threading.Thread.__init__(self)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        while not self.stop_requested.is_set():
            time.sleep(self.interval)
            if self.auto or not os.path.isfile(self.fpath):
                self.dump_stacktraces()

    def stop(self):
        if False:
            i = 10
            return i + 15
        self.stop_requested.set()
        self.join()
        try:
            if os.path.isfile(self.fpath):
                os.unlink(self.fpath)
        except OSError:
            pass

    def dump_stacktraces(self):
        if False:
            for i in range(10):
                print('nop')
        with open(self.fpath, 'w+') as fout:
            fout.write(stacktraces())
_tracer = None

def trace_start(fpath, interval=5, *, auto=True):
    if False:
        i = 10
        return i + 15
    'Start tracing into the given file.'
    global _tracer
    if _tracer is None:
        _tracer = TraceDumper(fpath, interval, auto)
        _tracer.daemon = True
        _tracer.start()
    else:
        raise Exception('Already tracing to %s' % _tracer.fpath)

def trace_stop():
    if False:
        while True:
            i = 10
    'Stop tracing.'
    global _tracer
    if _tracer is None:
        raise Exception('Not tracing, cannot stop.')
    else:
        _tracer.stop()
        _tracer = None