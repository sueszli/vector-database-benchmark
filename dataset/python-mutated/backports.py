"""
Backports of fixes for joblib dependencies
"""
import os
import re
import time
from os.path import basename
from multiprocessing import util

class Version:
    """Backport from deprecated distutils

    We maintain this backport to avoid introducing a new dependency on
    `packaging`.

    We might rexplore this choice in the future if all major Python projects
    introduce a dependency on packaging anyway.
    """

    def __init__(self, vstring=None):
        if False:
            while True:
                i = 10
        if vstring:
            self.parse(vstring)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return "%s ('%s')" % (self.__class__.__name__, str(self))

    def __eq__(self, other):
        if False:
            print('Hello World!')
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c == 0

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c < 0

    def __le__(self, other):
        if False:
            while True:
                i = 10
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c <= 0

    def __gt__(self, other):
        if False:
            while True:
                i = 10
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c > 0

    def __ge__(self, other):
        if False:
            i = 10
            return i + 15
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c >= 0

class LooseVersion(Version):
    """Backport from deprecated distutils

    We maintain this backport to avoid introducing a new dependency on
    `packaging`.

    We might rexplore this choice in the future if all major Python projects
    introduce a dependency on packaging anyway.
    """
    component_re = re.compile('(\\d+ | [a-z]+ | \\.)', re.VERBOSE)

    def __init__(self, vstring=None):
        if False:
            return 10
        if vstring:
            self.parse(vstring)

    def parse(self, vstring):
        if False:
            return 10
        self.vstring = vstring
        components = [x for x in self.component_re.split(vstring) if x and x != '.']
        for (i, obj) in enumerate(components):
            try:
                components[i] = int(obj)
            except ValueError:
                pass
        self.version = components

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.vstring

    def __repr__(self):
        if False:
            return 10
        return "LooseVersion ('%s')" % str(self)

    def _cmp(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, str):
            other = LooseVersion(other)
        elif not isinstance(other, LooseVersion):
            return NotImplemented
        if self.version == other.version:
            return 0
        if self.version < other.version:
            return -1
        if self.version > other.version:
            return 1
try:
    import numpy as np

    def make_memmap(filename, dtype='uint8', mode='r+', offset=0, shape=None, order='C', unlink_on_gc_collect=False):
        if False:
            i = 10
            return i + 15
        'Custom memmap constructor compatible with numpy.memmap.\n\n        This function:\n        - is a backport the numpy memmap offset fix (See\n          https://github.com/numpy/numpy/pull/8443 for more details.\n          The numpy fix is available starting numpy 1.13)\n        - adds ``unlink_on_gc_collect``, which specifies  explicitly whether\n          the process re-constructing the memmap owns a reference to the\n          underlying file. If set to True, it adds a finalizer to the\n          newly-created memmap that sends a maybe_unlink request for the\n          memmaped file to resource_tracker.\n        '
        util.debug('[MEMMAP READ] creating a memmap (shape {}, filename {}, pid {})'.format(shape, basename(filename), os.getpid()))
        mm = np.memmap(filename, dtype=dtype, mode=mode, offset=offset, shape=shape, order=order)
        if LooseVersion(np.__version__) < '1.13':
            mm.offset = offset
        if unlink_on_gc_collect:
            from ._memmapping_reducer import add_maybe_unlink_finalizer
            add_maybe_unlink_finalizer(mm)
        return mm
except ImportError:

    def make_memmap(filename, dtype='uint8', mode='r+', offset=0, shape=None, order='C', unlink_on_gc_collect=False):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError("'joblib.backports.make_memmap' should not be used if numpy is not installed.")
if os.name == 'nt':
    access_denied_errors = (5, 13)
    from os import replace

    def concurrency_safe_rename(src, dst):
        if False:
            i = 10
            return i + 15
        'Renames ``src`` into ``dst`` overwriting ``dst`` if it exists.\n\n        On Windows os.replace can yield permission errors if executed by two\n        different processes.\n        '
        max_sleep_time = 1
        total_sleep_time = 0
        sleep_time = 0.001
        while total_sleep_time < max_sleep_time:
            try:
                replace(src, dst)
                break
            except Exception as exc:
                if getattr(exc, 'winerror', None) in access_denied_errors:
                    time.sleep(sleep_time)
                    total_sleep_time += sleep_time
                    sleep_time *= 2
                else:
                    raise
        else:
            raise
else:
    from os import replace as concurrency_safe_rename