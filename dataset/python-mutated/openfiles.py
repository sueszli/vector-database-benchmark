from __future__ import absolute_import, print_function, division
import os
import unittest
import re
import gc
import functools
from . import sysinfo

def _collects(func):
    if False:
        i = 10
        return i + 15

    @functools.wraps(func)
    def f(**kw):
        if False:
            while True:
                i = 10
        gc.collect()
        gc.collect()
        enabled = gc.isenabled()
        gc.disable()
        try:
            return func(**kw)
        finally:
            if enabled:
                gc.enable()
    return f
if sysinfo.WIN:

    def _run_lsof():
        if False:
            print('Hello World!')
        raise unittest.SkipTest('lsof not expected on Windows')
else:

    @_collects
    def _run_lsof():
        if False:
            i = 10
            return i + 15
        import tempfile
        pid = os.getpid()
        (fd, tmpname) = tempfile.mkstemp('get_open_files')
        os.close(fd)
        lsof_command = 'lsof -p %s > %s' % (pid, tmpname)
        if os.system(lsof_command):
            raise unittest.SkipTest('lsof failed')
        with open(tmpname) as fobj:
            data = fobj.read().strip()
        os.remove(tmpname)
        return data

def default_get_open_files(pipes=False, **_kwargs):
    if False:
        print('Hello World!')
    data = _run_lsof()
    results = {}
    for line in data.split('\n'):
        line = line.strip()
        if not line or line.startswith('COMMAND'):
            continue
        split = re.split('\\s+', line)
        (_command, _pid, _user, fd) = split[:4]
        if fd[:-1].isdigit() or fd.isdigit():
            if not pipes and fd[-1].isdigit():
                continue
            fd = int(fd[:-1]) if not fd[-1].isdigit() else int(fd)
            if fd in results:
                params = (fd, line, split, results.get(fd), data)
                raise AssertionError('error when parsing lsof output: duplicate fd=%r\nline=%r\nsplit=%r\nprevious=%r\ndata:\n%s' % params)
            results[fd] = line
    if not results:
        raise AssertionError('failed to parse lsof:\n%s' % (data,))
    results['data'] = data
    return results

@_collects
def default_get_number_open_files():
    if False:
        print('Hello World!')
    if os.path.exists('/proc/'):
        fd_directory = '/proc/%d/fd' % os.getpid()
        return len(os.listdir(fd_directory))
    try:
        return len(get_open_files(pipes=True)) - 1
    except (OSError, AssertionError, unittest.SkipTest):
        return 0
lsof_get_open_files = default_get_open_files
try:
    import psutil
except ImportError:
    get_open_files = default_get_open_files
    get_number_open_files = default_get_number_open_files
else:

    class _TrivialOpenFile(object):
        __slots__ = ('fd',)

        def __init__(self, fd):
            if False:
                i = 10
                return i + 15
            self.fd = fd

    @_collects
    def get_open_files(count_closing_as_open=True, **_kw):
        if False:
            i = 10
            return i + 15
        '\n        Return a list of popenfile and pconn objects.\n\n        Note that other than `fd`, they have different attributes.\n\n        .. important:: If you want to find open sockets, on Windows\n           and linux, it is important that the socket at least be listening\n           (socket.listen(1)). Unlike the lsof implementation, this will only\n           return sockets in a state like that.\n        '
        results = {}
        for _ in range(3):
            try:
                if count_closing_as_open and os.path.exists('/proc/'):
                    fd_directory = '/proc/%d/fd' % os.getpid()
                    fd_files = os.listdir(fd_directory)
                else:
                    fd_files = []
                process = psutil.Process()
                results['data'] = process.open_files()
                results['data'] += process.connections('all')
                break
            except OSError:
                pass
        else:
            raise unittest.SkipTest('Unable to read open files')
        for x in results['data']:
            results[x.fd] = x
        for fd_str in fd_files:
            if fd_str not in results:
                fd = int(fd_str)
                results[fd] = _TrivialOpenFile(fd)
        results['data'] += [('From psutil', process)]
        results['data'] += [('fd files', fd_files)]
        return results

    @_collects
    def get_number_open_files():
        if False:
            while True:
                i = 10
        process = psutil.Process()
        try:
            return process.num_fds()
        except AttributeError:
            return 0

class DoesNotLeakFilesMixin(object):
    """
    A test case mixin that helps find a method that's leaking an
    open file.

    Only mix this in when needed to debug, it slows tests down.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.__open_files_count = get_number_open_files()
        super(DoesNotLeakFilesMixin, self).setUp()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super(DoesNotLeakFilesMixin, self).tearDown()
        after = get_number_open_files()
        if after > self.__open_files_count:
            raise AssertionError('Too many open files. Before: %s < After: %s.\n%s' % (self.__open_files_count, after, get_open_files()))