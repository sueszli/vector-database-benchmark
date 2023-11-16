import contextlib
import functools
import re
import sys
import warnings

def check_syntax_warning(testcase, statement, errtext='', *, lineno=1, offset=None):
    if False:
        for i in range(10):
            print('nop')
    from test.support import check_syntax_error
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter('always', SyntaxWarning)
        compile(statement, '<testcase>', 'exec')
    testcase.assertEqual(len(warns), 1, warns)
    (warn,) = warns
    testcase.assertTrue(issubclass(warn.category, SyntaxWarning), warn.category)
    if errtext:
        testcase.assertRegex(str(warn.message), errtext)
    testcase.assertEqual(warn.filename, '<testcase>')
    testcase.assertIsNotNone(warn.lineno)
    if lineno is not None:
        testcase.assertEqual(warn.lineno, lineno)
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter('error', SyntaxWarning)
        check_syntax_error(testcase, statement, errtext, lineno=lineno, offset=offset)
    testcase.assertEqual(warns, [])

def ignore_warnings(*, category):
    if False:
        i = 10
        return i + 15
    "Decorator to suppress deprecation warnings.\n\n    Use of context managers to hide warnings make diffs\n    more noisy and tools like 'git blame' less useful.\n    "

    def decorator(test):
        if False:
            i = 10
            return i + 15

        @functools.wraps(test)
        def wrapper(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=category)
                return test(self, *args, **kwargs)
        return wrapper
    return decorator

class WarningsRecorder(object):
    """Convenience wrapper for the warnings list returned on
       entry to the warnings.catch_warnings() context manager.
    """

    def __init__(self, warnings_list):
        if False:
            while True:
                i = 10
        self._warnings = warnings_list
        self._last = 0

    def __getattr__(self, attr):
        if False:
            i = 10
            return i + 15
        if len(self._warnings) > self._last:
            return getattr(self._warnings[-1], attr)
        elif attr in warnings.WarningMessage._WARNING_DETAILS:
            return None
        raise AttributeError('%r has no attribute %r' % (self, attr))

    @property
    def warnings(self):
        if False:
            return 10
        return self._warnings[self._last:]

    def reset(self):
        if False:
            while True:
                i = 10
        self._last = len(self._warnings)

@contextlib.contextmanager
def check_warnings(*filters, **kwargs):
    if False:
        return 10
    'Context manager to silence warnings.\n\n    Accept 2-tuples as positional arguments:\n        ("message regexp", WarningCategory)\n\n    Optional argument:\n     - if \'quiet\' is True, it does not fail if a filter catches nothing\n        (default True without argument,\n         default False if some filters are defined)\n\n    Without argument, it defaults to:\n        check_warnings(("", Warning), quiet=True)\n    '
    quiet = kwargs.get('quiet')
    if not filters:
        filters = (('', Warning),)
        if quiet is None:
            quiet = True
    return _filterwarnings(filters, quiet)

@contextlib.contextmanager
def check_no_warnings(testcase, message='', category=Warning, force_gc=False):
    if False:
        return 10
    'Context manager to check that no warnings are emitted.\n\n    This context manager enables a given warning within its scope\n    and checks that no warnings are emitted even with that warning\n    enabled.\n\n    If force_gc is True, a garbage collection is attempted before checking\n    for warnings. This may help to catch warnings emitted when objects\n    are deleted, such as ResourceWarning.\n\n    Other keyword arguments are passed to warnings.filterwarnings().\n    '
    from test.support import gc_collect
    with warnings.catch_warnings(record=True) as warns:
        warnings.filterwarnings('always', message=message, category=category)
        yield
        if force_gc:
            gc_collect()
    testcase.assertEqual(warns, [])

@contextlib.contextmanager
def check_no_resource_warning(testcase):
    if False:
        while True:
            i = 10
    'Context manager to check that no ResourceWarning is emitted.\n\n    Usage:\n\n        with check_no_resource_warning(self):\n            f = open(...)\n            ...\n            del f\n\n    You must remove the object which may emit ResourceWarning before\n    the end of the context manager.\n    '
    with check_no_warnings(testcase, category=ResourceWarning, force_gc=True):
        yield

def _filterwarnings(filters, quiet=False):
    if False:
        print('Hello World!')
    "Catch the warnings, then check if all the expected\n    warnings have been raised and re-raise unexpected warnings.\n    If 'quiet' is True, only re-raise the unexpected warnings.\n    "
    frame = sys._getframe(2)
    registry = frame.f_globals.get('__warningregistry__')
    if registry:
        registry.clear()
    with warnings.catch_warnings(record=True) as w:
        sys.modules['warnings'].simplefilter('always')
        yield WarningsRecorder(w)
    reraise = list(w)
    missing = []
    for (msg, cat) in filters:
        seen = False
        for w in reraise[:]:
            warning = w.message
            if re.match(msg, str(warning), re.I) and issubclass(warning.__class__, cat):
                seen = True
                reraise.remove(w)
        if not seen and (not quiet):
            missing.append((msg, cat.__name__))
    if reraise:
        raise AssertionError('unhandled warning %s' % reraise[0])
    if missing:
        raise AssertionError('filter (%r, %s) did not catch any warning' % missing[0])

@contextlib.contextmanager
def save_restore_warnings_filters():
    if False:
        i = 10
        return i + 15
    old_filters = warnings.filters[:]
    try:
        yield
    finally:
        warnings.filters[:] = old_filters

def _warn_about_deprecation():
    if False:
        i = 10
        return i + 15
    warnings.warn('This is used in test_support test to ensure support.ignore_deprecations_from() works as expected. You should not be seeing this.', DeprecationWarning, stacklevel=0)