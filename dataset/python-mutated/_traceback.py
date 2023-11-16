from types import TracebackType
from typing import List, Optional
import tempfile
import traceback
import contextlib
import inspect
import os.path

@contextlib.contextmanager
def report_compile_source_on_error():
    if False:
        i = 10
        return i + 15
    try:
        yield
    except Exception as exc:
        tb = exc.__traceback__
        stack = []
        while tb is not None:
            filename = tb.tb_frame.f_code.co_filename
            source = tb.tb_frame.f_globals.get('__compile_source__')
            if filename == '<string>' and source is not None:
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
                    f.write(source)
                frame = tb.tb_frame
                code = compile('__inspect_currentframe()', f.name, 'eval')
                code = code.replace(co_name=frame.f_code.co_name)
                if hasattr(frame.f_code, 'co_linetable'):
                    code = code.replace(co_linetable=frame.f_code.co_linetable, co_firstlineno=frame.f_code.co_firstlineno)
                fake_frame = eval(code, frame.f_globals, {**frame.f_locals, '__inspect_currentframe': inspect.currentframe})
                fake_tb = TracebackType(None, fake_frame, tb.tb_lasti, tb.tb_lineno)
                stack.append(fake_tb)
            else:
                stack.append(tb)
            tb = tb.tb_next
        tb_next = None
        for tb in reversed(stack):
            tb.tb_next = tb_next
            tb_next = tb
        raise exc.with_traceback(tb_next)

def shorten_filename(fn, *, base=None):
    if False:
        i = 10
        return i + 15
    "Shorten a source filepath, with the assumption that torch/ subdirectories don't need to be shown to user."
    if base is None:
        base = os.path.dirname(os.path.dirname(__file__))
    try:
        prefix = os.path.commonpath([fn, base])
    except ValueError:
        return fn
    else:
        return fn[len(prefix) + 1:]

def format_frame(frame, *, base=None, line=False):
    if False:
        i = 10
        return i + 15
    '\n    Format a FrameSummary in a short way, without printing full absolute path or code.\n\n    The idea is the result fits on a single line.\n    '
    extra_line = ''
    if line:
        extra_line = f'{frame.line}  # '
    return f'{extra_line}{shorten_filename(frame.filename, base=base)}:{frame.lineno} in {frame.name}'

def format_traceback_short(tb):
    if False:
        return 10
    'Format a TracebackType in a short way, printing only the inner-most frame.'
    return format_frame(traceback.extract_tb(tb)[-1])

class CapturedTraceback:
    __slots__ = ['tb', 'skip']

    def __init__(self, tb, skip=0):
        if False:
            for i in range(10):
                print('nop')
        self.tb = tb
        self.skip = skip

    def cleanup(self):
        if False:
            i = 10
            return i + 15
        self.tb = None

    def summary(self):
        if False:
            for i in range(10):
                print('nop')
        import torch._C._profiler
        if self.tb is None:
            return traceback.StackSummary()
        return _extract_symbolized_tb(torch._C._profiler.symbolize_tracebacks([self.tb])[0], self.skip)

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        return (None, {'tb': None, 'skip': self.skip})

    @staticmethod
    def extract(*, script=False, cpp=False, skip=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Like traceback.extract_stack(), but faster (approximately 20x faster); it\n        is fast enough that you can unconditionally log stacks this way as part of\n        normal execution.  It returns a torch._C._profiler.CapturedTraceback\n        object that must be formatted specially with format_captured_tb.\n\n        By default, this only reports Python backtraces (like extract_stack).  You\n        can set the script/cpp kwargs to also turn on TorchScript/C++ trace\n        reporting.\n        '
        import torch._C._profiler
        if script or cpp:
            assert skip == 0, 'skip with script/cpp NYI'
        return CapturedTraceback(torch._C._profiler.gather_traceback(python=True, script=script, cpp=cpp), 0 if script or cpp else skip + 1)

    def format(self):
        if False:
            while True:
                i = 10
        '\n        Formats a single torch._C._profiler.CapturedTraceback into a list of\n        strings equivalent to the output of traceback.format_list.  Note that if\n        pass it CapturedTraceback with C++ traces,  it is better not to use this\n        function and use the batch formatting API format_captured_tbs to amortize\n        the cost of symbolization\n        '
        return traceback.format_list(self.summary())

    @staticmethod
    def format_all(tbs):
        if False:
            while True:
                i = 10
        '\n        Bulk version of CapturedTraceback.format.  Returns a list of list of strings.\n        '
        import torch._C._profiler
        rs: List[Optional[List[str]]] = []
        delayed_idxs = []
        for (i, tb) in enumerate(tbs):
            if tb.tb is None:
                rs.append([])
            else:
                rs.append(None)
                delayed_idxs.append(i)
        stbs = torch._C._profiler.symbolize_tracebacks([tbs[i].tb for i in delayed_idxs])
        for (i, stb) in zip(delayed_idxs, stbs):
            rs[i] = traceback.format_list(tbs[i].summary())
        return rs

def _extract_symbolized_tb(tb, skip):
    if False:
        print('Hello World!')
    '\n    Given a symbolized traceback from symbolize_tracebacks, return a StackSummary object of\n    pre-processed stack trace entries.\n    '
    stack = traceback.StackSummary()
    for f in reversed(tb[skip:]):
        stack.append(traceback.FrameSummary(f['filename'], f['line'], f['name']))
    return stack