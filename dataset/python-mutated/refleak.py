import gc
import os
import re
import sys
import warnings
from inspect import isabstract
from test import cinder_support, support
from test.support import os_helper
from test.libregrtest.utils import clear_caches
from types import CodeType, FunctionType, ModuleType
try:
    from _abc import _get_dump
except ImportError:
    import weakref

    def _get_dump(cls):
        if False:
            return 10
        registry_weakrefs = set((weakref.ref(obj) for obj in cls._abc_registry))
        return (registry_weakrefs, cls._abc_cache, cls._abc_negative_cache, cls._abc_negative_cache_version)
if cinder_support.hasCinderX():
    import cinder
_RUNNING_REF_LEAK_TEST = False

def isRunningRefleakTest():
    if False:
        i = 10
        return i + 15
    return _RUNNING_REF_LEAK_TEST

def dash_R(ns, test_name, test_func):
    if False:
        while True:
            i = 10
    "Run a test multiple times, looking for reference leaks.\n\n    Returns:\n        False if the test didn't leak references; True if we detected refleaks.\n    "
    global _RUNNING_REF_LEAK_TEST
    import copyreg
    import collections.abc
    if not hasattr(sys, 'gettotalrefcount'):
        raise Exception('Tracking reference leaks requires a debug build of Python')
    warm_caches()
    fs = warnings.filters[:]
    ps = copyreg.dispatch_table.copy()
    pic = sys.path_importer_cache.copy()
    try:
        import zipimport
    except ImportError:
        zdc = None
    else:
        zdc = zipimport._zip_directory_cache.copy()
    abcs = {}
    for abc in [getattr(collections.abc, a) for a in collections.abc.__all__]:
        if not isabstract(abc):
            continue
        for obj in abc.__subclasses__() + [abc]:
            abcs[obj] = _get_dump(obj)[0]
    int_pool = {value: value for value in range(-1000, 1000)}

    def get_pooled_int(value):
        if False:
            for i in range(10):
                print('nop')
        return int_pool.setdefault(value, value)
    (nwarmup, ntracked, fname) = ns.huntrleaks
    fname = os.path.join(os_helper.SAVEDCWD, fname)
    repcount = nwarmup + ntracked
    rep_range = list(range(repcount))
    rc_deltas = [0] * repcount
    alloc_deltas = [0] * repcount
    fd_deltas = [0] * repcount
    getallocatedblocks = sys.getallocatedblocks
    gettotalrefcount = sys.gettotalrefcount
    fd_count = os_helper.fd_count
    rc_before = alloc_before = fd_before = 0
    if not ns.quiet:
        print('beginning', repcount, 'repetitions', file=sys.stderr)
        print(('1234567890' * (repcount // 10 + 1))[:repcount], file=sys.stderr, flush=True)
    dash_R_cleanup(fs, ps, pic, zdc, abcs)
    _RUNNING_REF_LEAK_TEST = True
    try:
        for i in rep_range:
            test_func()
            dash_R_cleanup(fs, ps, pic, zdc, abcs)
            alloc_after = getallocatedblocks()
            rc_after = gettotalrefcount()
            fd_after = fd_count()
            if not ns.quiet:
                print('.', end='', file=sys.stderr, flush=True)
            rc_deltas[i] = get_pooled_int(rc_after - rc_before)
            alloc_deltas[i] = get_pooled_int(alloc_after - alloc_before)
            fd_deltas[i] = get_pooled_int(fd_after - fd_before)
            alloc_before = alloc_after
            rc_before = rc_after
            fd_before = fd_after
    finally:
        _RUNNING_REF_LEAK_TEST = False
    if not ns.quiet:
        print(file=sys.stderr)

    def check_rc_deltas(deltas):
        if False:
            for i in range(10):
                print('nop')
        return all((delta >= 1 for delta in deltas))

    def check_fd_deltas(deltas):
        if False:
            while True:
                i = 10
        return any(deltas)
    failed = False
    for (deltas, item_name, checker) in [(rc_deltas, 'references', check_rc_deltas), (alloc_deltas, 'memory blocks', check_rc_deltas), (fd_deltas, 'file descriptors', check_fd_deltas)]:
        deltas = deltas[nwarmup:]
        if checker(deltas):
            msg = '%s leaked %s %s, sum=%s' % (test_name, deltas, item_name, sum(deltas))
            print(msg, file=sys.stderr, flush=True)
            with open(fname, 'a') as refrep:
                print(msg, file=refrep)
                refrep.flush()
            failed = True
    return failed
if cinder_support.hasCinderX():
    CINDER_KNOBS = cinder.getknobs()

def dash_R_cleanup(fs, ps, pic, zdc, abcs):
    if False:
        while True:
            i = 10
    import copyreg
    import collections.abc
    warnings.filters[:] = fs
    copyreg.dispatch_table.clear()
    copyreg.dispatch_table.update(ps)
    sys.path_importer_cache.clear()
    sys.path_importer_cache.update(pic)
    try:
        import zipimport
    except ImportError:
        pass
    else:
        zipimport._zip_directory_cache.clear()
        zipimport._zip_directory_cache.update(zdc)
    if cinder_support.hasCinderX():
        cinder.setknobs({'shadowcode': False})
        cinder.clear_caches()
        cinder.clear_all_shadow_caches()
    sys._clear_type_cache()
    abs_classes = [getattr(collections.abc, a) for a in collections.abc.__all__]
    abs_classes = filter(isabstract, abs_classes)
    for abc in abs_classes:
        for obj in abc.__subclasses__() + [abc]:
            for ref in abcs.get(obj, set()):
                if ref() is not None:
                    obj.register(ref())
            obj._abc_caches_clear()
    clear_caches()
    if cinder_support.hasCinderX():
        cinder.setknobs(CINDER_KNOBS)

def warm_caches():
    if False:
        print('Hello World!')
    s = bytes(range(256))
    for i in range(256):
        s[i:i + 1]
    [chr(i) for i in range(256)]
    list(range(-5, 257))