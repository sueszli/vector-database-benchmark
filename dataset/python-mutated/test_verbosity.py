from contextlib import contextmanager
from hypothesis import example, find, given
from hypothesis._settings import Verbosity, settings
from hypothesis.reporting import default as default_reporter, with_reporter
from hypothesis.strategies import booleans, integers, lists
from tests.common.debug import minimal
from tests.common.utils import capture_out, fails

@contextmanager
def capture_verbosity():
    if False:
        print('Hello World!')
    with capture_out() as o:
        with with_reporter(default_reporter):
            yield o

def test_prints_intermediate_in_success():
    if False:
        i = 10
        return i + 15
    with capture_verbosity() as o:

        @settings(verbosity=Verbosity.verbose)
        @given(booleans())
        def test_works(x):
            if False:
                print('Hello World!')
            pass
        test_works()
    assert 'Trying example' in o.getvalue()

def test_does_not_log_in_quiet_mode():
    if False:
        while True:
            i = 10
    with capture_verbosity() as o:

        @fails
        @settings(verbosity=Verbosity.quiet, print_blob=False)
        @given(integers())
        def test_foo(x):
            if False:
                return 10
            raise AssertionError
        test_foo()
    assert not o.getvalue()

def test_includes_progress_in_verbose_mode():
    if False:
        for i in range(10):
            print('nop')
    with capture_verbosity() as o:
        minimal(lists(integers(), min_size=1), lambda x: sum(x) >= 100, settings(verbosity=Verbosity.verbose))
    out = o.getvalue()
    assert out
    assert 'Trying example: ' in out

def test_prints_initial_attempts_on_find():
    if False:
        while True:
            i = 10
    with capture_verbosity() as o:

        def foo():
            if False:
                print('Hello World!')
            seen = []

            def not_first(x):
                if False:
                    print('Hello World!')
                if not seen:
                    seen.append(x)
                    return False
                return x not in seen
            find(integers(), not_first, settings=settings(verbosity=Verbosity.verbose, max_examples=1000))
        foo()
    assert 'Trying example' in o.getvalue()

def test_includes_intermediate_results_in_verbose_mode():
    if False:
        for i in range(10):
            print('nop')
    with capture_verbosity() as o:

        @fails
        @settings(verbosity=Verbosity.verbose, database=None, derandomize=True, max_examples=100)
        @given(lists(integers(), min_size=1))
        def test_foo(x):
            if False:
                print('Hello World!')
            assert sum(x) < 10000
        test_foo()
    lines = o.getvalue().splitlines()
    assert len([l for l in lines if 'example' in l]) > 2
    assert [l for l in lines if 'AssertionError' in l]

@example(0)
@settings(verbosity=Verbosity.quiet)
@given(integers())
def test_no_indexerror_in_quiet_mode(x):
    if False:
        return 10
    pass

@fails
@example(0)
@settings(verbosity=Verbosity.quiet, report_multiple_bugs=True)
@given(integers())
def test_no_indexerror_in_quiet_mode_report_multiple(x):
    if False:
        return 10
    assert x

@fails
@example(0)
@settings(verbosity=Verbosity.quiet, report_multiple_bugs=False)
@given(integers())
def test_no_indexerror_in_quiet_mode_report_one(x):
    if False:
        i = 10
        return i + 15
    assert x