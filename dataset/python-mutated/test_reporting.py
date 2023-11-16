import os
import sys
import pytest
from hypothesis import given, reporting
from hypothesis._settings import Verbosity, settings
from hypothesis.reporting import debug_report, report, verbose_report
from hypothesis.strategies import integers
from tests.common.utils import capture_out

def test_can_print_bytes():
    if False:
        while True:
            i = 10
    with capture_out() as o:
        with reporting.with_reporter(reporting.default):
            report(b'hi')
    assert o.getvalue() == 'hi\n'

def test_prints_output_by_default():
    if False:
        return 10

    @given(integers())
    def test_int(x):
        if False:
            return 10
        raise AssertionError
    with pytest.raises(AssertionError) as err:
        test_int()
    assert 'Falsifying example' in '\n'.join(err.value.__notes__)

def test_does_not_print_debug_in_verbose():
    if False:
        return 10

    @given(integers())
    @settings(verbosity=Verbosity.verbose)
    def f(x):
        if False:
            print('Hello World!')
        debug_report('Hi')
    with capture_out() as o:
        f()
    assert 'Hi' not in o.getvalue()

def test_does_print_debug_in_debug():
    if False:
        i = 10
        return i + 15

    @given(integers())
    @settings(verbosity=Verbosity.debug)
    def f(x):
        if False:
            while True:
                i = 10
        debug_report('Hi')
    with capture_out() as o:
        f()
    assert 'Hi' in o.getvalue()

def test_does_print_verbose_in_debug():
    if False:
        return 10

    @given(integers())
    @settings(verbosity=Verbosity.debug)
    def f(x):
        if False:
            print('Hello World!')
        verbose_report('Hi')
    with capture_out() as o:
        f()
    assert 'Hi' in o.getvalue()

def test_can_report_when_system_locale_is_ascii(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    (read, write) = os.pipe()
    with open(read, encoding='ascii') as read:
        with open(write, 'w', encoding='ascii') as write:
            monkeypatch.setattr(sys, 'stdout', write)
            reporting.default('☃')

def test_can_report_functions():
    if False:
        i = 10
        return i + 15
    with capture_out() as out:
        report(lambda : 'foo')
    assert out.getvalue().strip() == 'foo'