from __future__ import annotations
from pre_commit_hooks.tests_should_end_in_test import main

def test_main_all_pass():
    if False:
        while True:
            i = 10
    ret = main(['foo_test.py', 'bar_test.py'])
    assert ret == 0

def test_main_one_fails():
    if False:
        for i in range(10):
            print('nop')
    ret = main(['not_test_ending.py', 'foo_test.py'])
    assert ret == 1

def test_regex():
    if False:
        print('Hello World!')
    assert main(('foo_test_py',)) == 1

def test_main_django_all_pass():
    if False:
        print('Hello World!')
    ret = main(('--django', 'tests.py', 'test_foo.py', 'test_bar.py', 'tests/test_baz.py'))
    assert ret == 0

def test_main_django_one_fails():
    if False:
        return 10
    ret = main(['--django', 'not_test_ending.py', 'test_foo.py'])
    assert ret == 1

def test_validate_nested_files_django_one_fails():
    if False:
        i = 10
        return i + 15
    ret = main(['--django', 'tests/not_test_ending.py', 'test_foo.py'])
    assert ret == 1

def test_main_not_django_fails():
    if False:
        for i in range(10):
            print('nop')
    ret = main(['foo_test.py', 'bar_test.py', 'test_baz.py'])
    assert ret == 1

def test_main_django_fails():
    if False:
        while True:
            i = 10
    ret = main(['--django', 'foo_test.py', 'test_bar.py', 'test_baz.py'])
    assert ret == 1

def test_main_pytest_test_first():
    if False:
        return 10
    assert main(['--pytest-test-first', 'test_foo.py']) == 0
    assert main(['--pytest-test-first', 'foo_test.py']) == 1