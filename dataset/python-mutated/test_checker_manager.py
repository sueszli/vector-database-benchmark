"""Tests for the Manager object for FileCheckers."""
from __future__ import annotations
import errno
import multiprocessing
from unittest import mock
import pytest
from flake8 import checker
from flake8.main.options import JobsArgument
from flake8.plugins import finder

def style_guide_mock():
    if False:
        print('Hello World!')
    'Create a mock StyleGuide object.'
    return mock.MagicMock(**{'options.jobs': JobsArgument('4')})

def _parallel_checker_manager():
    if False:
        i = 10
        return i + 15
    'Call Manager.run() and return the number of calls to `run_serial`.'
    style_guide = style_guide_mock()
    manager = checker.Manager(style_guide, finder.Checkers([], [], []), [])
    manager.filenames = ('file1', 'file2')
    return manager

def test_oserrors_cause_serial_fall_back():
    if False:
        for i in range(10):
            print('nop')
    'Verify that OSErrors will cause the Manager to fallback to serial.'
    err = OSError(errno.ENOSPC, 'Ominous message about spaceeeeee')
    with mock.patch('_multiprocessing.SemLock', side_effect=err):
        manager = _parallel_checker_manager()
        with mock.patch.object(manager, 'run_serial') as serial:
            manager.run()
    assert serial.call_count == 1

def test_oserrors_are_reraised():
    if False:
        return 10
    'Verify that unexpected OSErrors will cause the Manager to reraise.'
    err = OSError(errno.EAGAIN, 'Ominous message')
    with mock.patch('_multiprocessing.SemLock', side_effect=err):
        manager = _parallel_checker_manager()
        with mock.patch.object(manager, 'run_serial') as serial:
            with pytest.raises(OSError):
                manager.run()
    assert serial.call_count == 0

def test_multiprocessing_cpu_count_not_implemented():
    if False:
        i = 10
        return i + 15
    'Verify that jobs is 0 if cpu_count is unavailable.'
    style_guide = style_guide_mock()
    style_guide.options.jobs = JobsArgument('auto')
    with mock.patch.object(multiprocessing, 'cpu_count', side_effect=NotImplementedError):
        manager = checker.Manager(style_guide, finder.Checkers([], [], []), [])
    assert manager.jobs == 0

def test_make_checkers():
    if False:
        for i in range(10):
            print('nop')
    'Verify that we create a list of FileChecker instances.'
    style_guide = style_guide_mock()
    style_guide.options.filenames = ['file1', 'file2']
    manager = checker.Manager(style_guide, finder.Checkers([], [], []), [])
    with mock.patch('flake8.utils.fnmatch', return_value=True):
        with mock.patch('flake8.processor.FileProcessor'):
            manager.start()
    assert manager.filenames == ('file1', 'file2')