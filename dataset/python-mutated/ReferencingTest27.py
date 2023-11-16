""" Reference counting tests for Python2 specific features.

These contain functions that do specific things, where we have a suspect
that references may be lost or corrupted. Executing them repeatedly and
checking the reference count is how they are used.
"""
import os
import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from nuitka.tools.testing.Common import executeReferenceChecked
x = 17

def simpleFunction1():
    if False:
        while True:
            i = 10
    return {i: x for i in range(x)}

def simpleFunction2():
    if False:
        for i in range(10):
            print('nop')
    try:
        return {y: i for i in range(x)}
    except NameError:
        pass

def simpleFunction3():
    if False:
        for i in range(10):
            print('nop')
    return {i for i in range(x)}

def simpleFunction4():
    if False:
        i = 10
        return i + 15
    try:
        return {y for i in range(x)}
    except NameError:
        pass
tests_stderr = ()
tests_skipped = {}
result = executeReferenceChecked(prefix='simpleFunction', names=globals(), tests_skipped=tests_skipped, tests_stderr=tests_stderr)
sys.exit(0 if result else 1)