"""
Profiling hooks

This module contains a couple of decorators (`profile` and `coverage`) that
can be used to wrap functions and/or methods to produce profiles and line
coverage reports.  There's a third convenient decorator (`timecall`) that
measures the duration of function execution without the extra profiling
overhead.

Usage example::

    from profilehooks import profile, coverage

    @profile    # or @coverage
    def fn(n):
        if n < 2: return 1
        else: return n * fn(n-1)

    print(fn(42))

Or without imports, with some hack

    $ python -m profilehooks yourmodule

    @profile    # or @coverage
    def fn(n):
        if n < 2: return 1
        else: return n * fn(n-1)

    print(fn(42))

Reports for all thusly decorated functions will be printed to sys.stdout
on program termination.  You can alternatively request for immediate
reports for each call by passing immediate=True to the profile decorator.

There's also a @timecall decorator for printing the time to sys.stderr
every time a function is called, when you just want to get a rough measure
instead of a detailed (but costly) profile.

Caveats

  A thread on python-dev convinced me that hotshot produces bogus numbers.
  See https://mail.python.org/pipermail/python-dev/2005-November/058264.html

  I don't know what will happen if a decorated function will try to call
  another decorated function.  All decorators probably need to explicitly
  support nested profiling (currently TraceFuncCoverage is the only one
  that supports this, while HotShotFuncProfile has support for recursive
  functions.)

  Profiling with hotshot creates temporary files (*.prof for profiling,
  *.cprof for coverage) in the current directory.  These files are not
  cleaned up.  Exception: when you specify a filename to the profile
  decorator (to store the pstats.Stats object for later inspection),
  the temporary file will be the filename you specified with '.raw'
  appended at the end.

  Coverage analysis with hotshot seems to miss some executions resulting
  in lower line counts and some lines errorneously marked as never
  executed.  For this reason coverage analysis now uses trace.py which is
  slower, but more accurate.

Copyright (c) 2004--2020 Marius Gedminas <marius@gedmin.as>
Copyright (c) 2007 Hanno Schlichting
Copyright (c) 2008 Florian Schulze

Released under the MIT licence since December 2006:

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

(Previously it was distributed under the GNU General Public Licence.)
"""
from __future__ import print_function
__author__ = 'Marius Gedminas <marius@gedmin.as>'
__copyright__ = 'Copyright 2004-2020 Marius Gedminas and contributors'
__license__ = 'MIT'
__version__ = '1.12.0'
__date__ = '2020-08-20'
import atexit
import functools
import inspect
import logging
import os
import re
import sys
from profile import Profile
import pstats
import timeit
try:
    import hotshot
    import hotshot.stats
except ImportError:
    hotshot = None
import trace
import dis
import token
import tokenize
if hotshot is not None:
    import _hotshot
    import hotshot.log
try:
    import cProfile
except ImportError:
    cProfile = None
AVAILABLE_PROFILERS = {}
__all__ = ['coverage', 'coverage_with_hotshot', 'profile', 'timecall']
tokenize_open = getattr(tokenize, 'open', open)

def _unwrap(fn):
    if False:
        for i in range(10):
            print('nop')
    if not hasattr(fn, '__wrapped__'):
        return fn
    else:
        return _unwrap(fn.__wrapped__)

def _identify(fn):
    if False:
        for i in range(10):
            print('nop')
    fn = _unwrap(fn)
    funcname = fn.__name__
    filename = fn.__code__.co_filename
    lineno = fn.__code__.co_firstlineno
    return (funcname, filename, lineno)

def _is_file_like(o):
    if False:
        return 10
    return hasattr(o, 'write')

def profile(fn=None, skip=0, filename=None, immediate=False, dirs=False, sort=None, entries=40, profiler=('cProfile', 'profile', 'hotshot'), stdout=True):
    if False:
        for i in range(10):
            print('nop')
    "Mark `fn` for profiling.\n\n    If `skip` is > 0, first `skip` calls to `fn` will not be profiled.\n\n    If `stdout` is not file-like and truthy, output will be printed to\n    sys.stdout. If it is a file-like object, output will be printed to it\n    instead. `stdout` must be writable in text mode (as opposed to binary)\n    if it is file-like.\n\n    If `immediate` is False, profiling results will be printed to\n    self.stdout on program termination.  Otherwise results will be printed\n    after each call.  (If you don't want this, set stdout=False and specify a\n    `filename` to store profile data.)\n\n    If `dirs` is False only the name of the file will be printed.\n    Otherwise the full path is used.\n\n    `sort` can be a list of sort keys (defaulting to ['cumulative',\n    'time', 'calls']).  The following ones are recognized::\n\n        'calls'      -- call count\n        'cumulative' -- cumulative time\n        'file'       -- file name\n        'line'       -- line number\n        'module'     -- file name\n        'name'       -- function name\n        'nfl'        -- name/file/line\n        'pcalls'     -- call count\n        'stdname'    -- standard name\n        'time'       -- internal time\n\n    `entries` limits the output to the first N entries.\n\n    `profiler` can be used to select the preferred profiler, or specify a\n    sequence of them, in order of preference.  The default is ('cProfile'.\n    'profile', 'hotshot').\n\n    If `filename` is specified, the profile stats will be stored in the\n    named file.  You can load them with pstats.Stats(filename) or use a\n    visualization tool like RunSnakeRun.\n\n    Usage::\n\n        def fn(...):\n            ...\n        fn = profile(fn, skip=1)\n\n    If you are using Python 2.4, you should be able to use the decorator\n    syntax::\n\n        @profile(skip=3)\n        def fn(...):\n            ...\n\n    or just ::\n\n        @profile\n        def fn(...):\n            ...\n\n    "
    if fn is None:

        def decorator(fn):
            if False:
                return 10
            return profile(fn, skip=skip, filename=filename, immediate=immediate, dirs=dirs, sort=sort, entries=entries, profiler=profiler, stdout=stdout)
        return decorator
    if isinstance(profiler, str):
        profiler = [profiler]
    for p in profiler:
        if p in AVAILABLE_PROFILERS:
            profiler_class = AVAILABLE_PROFILERS[p]
            break
    else:
        raise ValueError('only these profilers are available: %s' % ', '.join(sorted(AVAILABLE_PROFILERS)))
    fp = profiler_class(fn, skip=skip, filename=filename, immediate=immediate, dirs=dirs, sort=sort, entries=entries, stdout=stdout)

    @functools.wraps(fn)
    def new_fn(*args, **kw):
        if False:
            for i in range(10):
                print('nop')
        return fp(*args, **kw)
    return new_fn

def coverage(fn):
    if False:
        while True:
            i = 10
    'Mark `fn` for line coverage analysis.\n\n    Results will be printed to sys.stdout on program termination.\n\n    Usage::\n\n        def fn(...):\n            ...\n        fn = coverage(fn)\n\n    If you are using Python 2.4, you should be able to use the decorator\n    syntax::\n\n        @coverage\n        def fn(...):\n            ...\n\n    '
    fp = TraceFuncCoverage(fn)

    @functools.wraps(fn)
    def new_fn(*args, **kw):
        if False:
            return 10
        return fp(*args, **kw)
    return new_fn

def coverage_with_hotshot(fn):
    if False:
        while True:
            i = 10
    "Mark `fn` for line coverage analysis.\n\n    Uses the 'hotshot' module for fast coverage analysis.\n\n    BUG: Produces inaccurate results.\n\n    See the docstring of `coverage` for usage examples.\n    "
    fp = HotShotFuncCoverage(fn)

    @functools.wraps(fn)
    def new_fn(*args, **kw):
        if False:
            return 10
        return fp(*args, **kw)
    return new_fn

class FuncProfile(object):
    """Profiler for a function (uses profile)."""
    in_profiler = False
    Profile = Profile

    def __init__(self, fn, skip=0, filename=None, immediate=False, dirs=False, sort=None, entries=40, stdout=True):
        if False:
            for i in range(10):
                print('nop')
        'Creates a profiler for a function.\n\n        Every profiler has its own log file (the name of which is derived\n        from the function name).\n\n        FuncProfile registers an atexit handler that prints profiling\n        information to sys.stderr when the program terminates.\n        '
        self.fn = fn
        self.skip = skip
        self.filename = filename
        self._immediate = immediate
        self.stdout = stdout
        self._stdout_is_fp = self.stdout and _is_file_like(self.stdout)
        self.dirs = dirs
        self.sort = sort or ('cumulative', 'time', 'calls')
        if isinstance(self.sort, str):
            self.sort = (self.sort,)
        self.entries = entries
        self.reset_stats()
        if not self.immediate:
            atexit.register(self.atexit)

    @property
    def immediate(self):
        if False:
            for i in range(10):
                print('nop')
        return self._immediate

    def __call__(self, *args, **kw):
        if False:
            i = 10
            return i + 15
        'Profile a singe call to the function.'
        self.ncalls += 1
        if self.skip > 0:
            self.skip -= 1
            self.skipped += 1
            return self.fn(*args, **kw)
        if FuncProfile.in_profiler:
            return self.fn(*args, **kw)
        profiler = self.Profile()
        try:
            FuncProfile.in_profiler = True
            return profiler.runcall(self.fn, *args, **kw)
        finally:
            FuncProfile.in_profiler = False
            self.stats.add(profiler)
            if self.immediate:
                self.print_stats()
                self.reset_stats()

    def print_stats(self):
        if False:
            while True:
                i = 10
        'Print profile information to sys.stdout.'
        stats = self.stats
        if self.filename:
            stats.dump_stats(self.filename)
        if self.stdout:
            (funcname, filename, lineno) = _identify(self.fn)
            print_f = print
            if self._stdout_is_fp:
                print_f = functools.partial(print, file=self.stdout)
            print_f('')
            print_f('*** PROFILER RESULTS ***')
            print_f('%s (%s:%s)' % (funcname, filename, lineno))
            if self.skipped:
                skipped = ' (%d calls not profiled)' % self.skipped
            else:
                skipped = ''
            print_f('function called %d times%s' % (self.ncalls, skipped))
            print_f('')
            if not self.dirs:
                stats.strip_dirs()
            stats.sort_stats(*self.sort)
            stats.print_stats(self.entries)

    def reset_stats(self):
        if False:
            return 10
        'Reset accumulated profiler statistics.'
        stream = self.stdout if self._stdout_is_fp else sys.stdout
        self.stats = pstats.Stats(Profile(), stream=stream)
        self.ncalls = 0
        self.skipped = 0

    def atexit(self):
        if False:
            return 10
        'Stop profiling and print profile information to sys.stdout or self.stdout.\n\n        This function is registered as an atexit hook.\n        '
        self.print_stats()
AVAILABLE_PROFILERS['profile'] = FuncProfile
if cProfile is not None:

    class CProfileFuncProfile(FuncProfile):
        """Profiler for a function (uses cProfile)."""
        Profile = cProfile.Profile
    AVAILABLE_PROFILERS['cProfile'] = CProfileFuncProfile
if hotshot is not None:

    class HotShotFuncProfile(FuncProfile):
        """Profiler for a function (uses hotshot)."""
        in_profiler = False

        def __init__(self, fn, skip=0, filename=None, immediate=False, dirs=False, sort=None, entries=40, stdout=True):
            if False:
                return 10
            'Creates a profiler for a function.\n\n            Every profiler has its own log file (the name of which is derived\n            from the function name).\n\n            HotShotFuncProfile registers an atexit handler that prints\n            profiling information to sys.stderr when the program terminates.\n\n            The log file is not removed and remains there to clutter the\n            current working directory.\n            '
            if filename:
                self.logfilename = filename + '.raw'
            else:
                self.logfilename = '%s.%d.prof' % (fn.__name__, os.getpid())
            super(HotShotFuncProfile, self).__init__(fn, skip=skip, filename=filename, immediate=immediate, dirs=dirs, sort=sort, entries=entries, stdout=stdout)

        def __call__(self, *args, **kw):
            if False:
                while True:
                    i = 10
            'Profile a singe call to the function.'
            self.ncalls += 1
            if self.skip > 0:
                self.skip -= 1
                self.skipped += 1
                return self.fn(*args, **kw)
            if HotShotFuncProfile.in_profiler:
                return self.fn(*args, **kw)
            if self.profiler is None:
                self.profiler = hotshot.Profile(self.logfilename)
            try:
                HotShotFuncProfile.in_profiler = True
                return self.profiler.runcall(self.fn, *args, **kw)
            finally:
                HotShotFuncProfile.in_profiler = False
                if self.immediate:
                    self.print_stats()
                    self.reset_stats()

        def print_stats(self):
            if False:
                i = 10
                return i + 15
            if self.profiler is None:
                self.stats = pstats.Stats(Profile())
            else:
                self.profiler.close()
                self.stats = hotshot.stats.load(self.logfilename)
            super(HotShotFuncProfile, self).print_stats()

        def reset_stats(self):
            if False:
                while True:
                    i = 10
            self.profiler = None
            self.ncalls = 0
            self.skipped = 0
    AVAILABLE_PROFILERS['hotshot'] = HotShotFuncProfile

    class HotShotFuncCoverage:
        """Coverage analysis for a function (uses _hotshot).

        HotShot coverage is reportedly faster than trace.py, but it appears to
        have problems with exceptions; also line counts in coverage reports
        are generally lower from line counts produced by TraceFuncCoverage.
        Is this my bug, or is it a problem with _hotshot?
        """

        def __init__(self, fn):
            if False:
                for i in range(10):
                    print('nop')
            'Creates a profiler for a function.\n\n            Every profiler has its own log file (the name of which is derived\n            from the function name).\n\n            HotShotFuncCoverage registers an atexit handler that prints\n            profiling information to sys.stderr when the program terminates.\n\n            The log file is not removed and remains there to clutter the\n            current working directory.\n            '
            self.fn = fn
            self.logfilename = '%s.%d.cprof' % (fn.__name__, os.getpid())
            self.profiler = _hotshot.coverage(self.logfilename)
            self.ncalls = 0
            atexit.register(self.atexit)

        def __call__(self, *args, **kw):
            if False:
                i = 10
                return i + 15
            'Profile a singe call to the function.'
            self.ncalls += 1
            old_trace = sys.gettrace()
            try:
                return self.profiler.runcall(self.fn, args, kw)
            finally:
                sys.settrace(old_trace)

        def atexit(self):
            if False:
                print('Hello World!')
            'Stop profiling and print profile information to sys.stderr.\n\n            This function is registered as an atexit hook.\n            '
            self.profiler.close()
            (funcname, filename, lineno) = _identify(self.fn)
            print('')
            print('*** COVERAGE RESULTS ***')
            print('%s (%s:%s)' % (funcname, filename, lineno))
            print('function called %d times' % self.ncalls)
            print('')
            fs = FuncSource(self.fn)
            reader = hotshot.log.LogReader(self.logfilename)
            for (what, (filename, lineno, funcname), tdelta) in reader:
                if filename != fs.filename:
                    continue
                if what == hotshot.log.LINE:
                    fs.mark(lineno)
                if what == hotshot.log.ENTER:
                    if lineno == fs.firstlineno:
                        lineno = fs.firstcodelineno
                    fs.mark(lineno)
            reader.close()
            print(fs)
            never_executed = fs.count_never_executed()
            if never_executed:
                print('%d lines were not executed.' % never_executed)

class TraceFuncCoverage:
    """Coverage analysis for a function (uses trace module).

    HotShot coverage analysis is reportedly faster, but it appears to have
    problems with exceptions.
    """
    tracer = trace.Trace(count=True, trace=False, ignoredirs=[sys.prefix, sys.exec_prefix])
    tracing = False

    def __init__(self, fn):
        if False:
            print('Hello World!')
        'Creates a profiler for a function.\n\n        Every profiler has its own log file (the name of which is derived\n        from the function name).\n\n        TraceFuncCoverage registers an atexit handler that prints\n        profiling information to sys.stderr when the program terminates.\n\n        The log file is not removed and remains there to clutter the\n        current working directory.\n        '
        self.fn = fn
        self.logfilename = '%s.%d.cprof' % (fn.__name__, os.getpid())
        self.ncalls = 0
        atexit.register(self.atexit)

    def __call__(self, *args, **kw):
        if False:
            while True:
                i = 10
        'Profile a singe call to the function.'
        self.ncalls += 1
        if TraceFuncCoverage.tracing:
            return self.fn(*args, **kw)
        old_trace = sys.gettrace()
        try:
            TraceFuncCoverage.tracing = True
            return self.tracer.runfunc(self.fn, *args, **kw)
        finally:
            sys.settrace(old_trace)
            TraceFuncCoverage.tracing = False

    def atexit(self):
        if False:
            print('Hello World!')
        'Stop profiling and print profile information to sys.stderr.\n\n        This function is registered as an atexit hook.\n        '
        (funcname, filename, lineno) = _identify(self.fn)
        print('')
        print('*** COVERAGE RESULTS ***')
        print('%s (%s:%s)' % (funcname, filename, lineno))
        print('function called %d times' % self.ncalls)
        print('')
        fs = FuncSource(self.fn)
        for ((filename, lineno), count) in self.tracer.counts.items():
            if filename != fs.filename:
                continue
            fs.mark(lineno, count)
        print(fs)
        never_executed = fs.count_never_executed()
        if never_executed:
            print('%d lines were not executed.' % never_executed)

class FuncSource:
    """Source code annotator for a function."""
    blank_rx = re.compile('^\\s*finally:\\s*(#.*)?$')

    def __init__(self, fn):
        if False:
            return 10
        self.fn = fn
        self.filename = inspect.getsourcefile(fn)
        self.sourcelines = {}
        self.source = []
        self.firstlineno = self.firstcodelineno = 0
        try:
            (self.source, self.firstlineno) = inspect.getsourcelines(fn)
            self.firstcodelineno = self.firstlineno
            self.find_source_lines()
        except IOError:
            self.filename = None

    def find_source_lines(self):
        if False:
            i = 10
            return i + 15
        'Mark all executable source lines in fn as executed 0 times.'
        if self.filename is None:
            return
        strs = self._find_docstrings(self.filename)
        lines = {ln for (off, ln) in dis.findlinestarts(_unwrap(self.fn).__code__) if ln not in strs}
        for lineno in lines:
            self.sourcelines.setdefault(lineno, 0)
        if lines:
            self.firstcodelineno = min(lines)
        else:
            self.firstcodelineno = self.firstlineno

    def _find_docstrings(self, filename):
        if False:
            i = 10
            return i + 15
        strs = set()
        prev = token.INDENT
        with tokenize_open(filename) as f:
            tokens = tokenize.generate_tokens(f.readline)
            for (ttype, tstr, start, end, line) in tokens:
                if ttype == token.STRING and prev == token.INDENT:
                    strs.update(range(start[0], end[0] + 1))
                prev = ttype
        return strs

    def mark(self, lineno, count=1):
        if False:
            while True:
                i = 10
        'Mark a given source line as executed count times.\n\n        Multiple calls to mark for the same lineno add up.\n        '
        self.sourcelines[lineno] = self.sourcelines.get(lineno, 0) + count

    def count_never_executed(self):
        if False:
            while True:
                i = 10
        'Count statements that were never executed.'
        lineno = self.firstlineno
        counter = 0
        for line in self.source:
            if self.sourcelines.get(lineno) == 0:
                if not self.blank_rx.match(line):
                    counter += 1
            lineno += 1
        return counter

    def __str__(self):
        if False:
            while True:
                i = 10
        'Return annotated source code for the function.'
        if self.filename is None:
            return 'cannot show coverage data since co_filename is None'
        lines = []
        lineno = self.firstlineno
        for line in self.source:
            counter = self.sourcelines.get(lineno)
            if counter is None:
                prefix = ' ' * 7
            elif counter == 0:
                if self.blank_rx.match(line):
                    prefix = ' ' * 7
                else:
                    prefix = '>' * 6 + ' '
            else:
                prefix = '%5d: ' % counter
            lines.append(prefix + line)
            lineno += 1
        return ''.join(lines)

def timecall(fn=None, immediate=True, timer=None, log_name=None, log_level=logging.DEBUG):
    if False:
        while True:
            i = 10
    "Wrap `fn` and print its execution time.\n\n    Example::\n\n        @timecall\n        def somefunc(x, y):\n            time.sleep(x * y)\n\n        somefunc(2, 3)\n\n    will print the time taken by somefunc on every call.  If you want just\n    a summary at program termination, use ::\n\n        @timecall(immediate=False)\n\n    You can also choose a timing method other than the default\n    ``timeit.default_timer()``, e.g.::\n\n        @timecall(timer=time.clock)\n\n    You can also log the output to a logger by specifying the name and level\n    of the logger to use, eg:\n\n        @timecall(immediate=True,\n                  log_name='profile_log',\n                  log_level=logging.DEBUG)\n\n    "
    if fn is None:

        def decorator(fn):
            if False:
                return 10
            return timecall(fn, immediate=immediate, timer=timer, log_name=log_name, log_level=log_level)
        return decorator
    if timer is None:
        timer = timeit.default_timer
    fp = FuncTimer(fn, immediate=immediate, timer=timer, log_name=log_name, log_level=log_level)

    @functools.wraps(fn)
    def new_fn(*args, **kw):
        if False:
            while True:
                i = 10
        return fp(*args, **kw)
    return new_fn

class FuncTimer(object):

    def __init__(self, fn, immediate, timer, log_name=None, log_level=logging.DEBUG):
        if False:
            while True:
                i = 10
        self.logger = None
        if log_name:
            self.logger = logging.getLogger(log_name)
        self.log_level = log_level
        self.fn = fn
        self.ncalls = 0
        self.totaltime = 0
        self.immediate = immediate
        self.timer = timer
        if not immediate:
            atexit.register(self.atexit)

    def __call__(self, *args, **kw):
        if False:
            while True:
                i = 10
        'Profile a singe call to the function.'
        fn = self.fn
        timer = self.timer
        self.ncalls += 1
        start = timer()
        try:
            return fn(*args, **kw)
        finally:
            duration = timer() - start
            self.totaltime += duration
            if self.immediate:
                (funcname, filename, lineno) = _identify(fn)
                message = '%s (%s:%s):\n    %.3f seconds\n\n' % (funcname, filename, lineno, duration)
                if self.logger:
                    self.logger.log(self.log_level, message)
                else:
                    sys.stderr.write('\n  ' + message)
                    sys.stderr.flush()

    def atexit(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.ncalls:
            return
        (funcname, filename, lineno) = _identify(self.fn)
        message = '\n  %s (%s:%s):\n    %d calls, %.3f seconds (%.3f seconds per call)\n' % (funcname, filename, lineno, self.ncalls, self.totaltime, self.totaltime / self.ncalls)
        if self.logger:
            self.logger.log(self.log_level, message)
        else:
            print(message)
if __name__ == '__main__':
    local = dict(((name, globals()[name]) for name in __all__))
    message = '********\nInjected `profilehooks`\n--------\n{}\n********\n'.format('\n'.join(local.keys()))

    def interact_():
        if False:
            return 10
        from code import interact
        interact(message, local=local)

    def run_():
        if False:
            for i in range(10):
                print('nop')
        from runpy import run_module
        print(message)
        run_module(sys.argv[1], init_globals=local)
    if len(sys.argv) == 1:
        interact_()
    else:
        run_()