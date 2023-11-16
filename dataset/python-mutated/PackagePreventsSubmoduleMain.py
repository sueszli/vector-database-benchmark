from __future__ import print_function
import imp
import os
import sys
ORIG = None
attemptImports = None
__warningregistry__ = {}

def diff(dct):
    if False:
        i = 10
        return i + 15
    print('globals diff', ORIG.symmetric_difference(dct))
    mdiff = START.symmetric_difference(sys.modules)
    if str is bytes:
        if 'some_package.os' in mdiff:
            mdiff.remove('some_package.os')
    print('Modules diff', mdiff)
START = set(sys.modules)
ORIG = set(globals())

def attemptImports(prefix):
    if False:
        while True:
            i = 10
    print(prefix, 'GO1:')
    try:
        import some_package
    except BaseException as e:
        print('Exception occurred', e)
    else:
        print('Import success.', some_package.__name__)
    diff(globals())
    print(prefix, 'GO2:')
    try:
        from some_package.some_module import Class4
    except BaseException as e:
        print('Exception occurred', e)
    else:
        print('Import success.', Class4)
    diff(globals())
    print(prefix, 'GO3:')
    try:
        from some_package import some_module
    except BaseException as e:
        print('Exception occurred', e)
    else:
        print('Import success.', some_module.__name__)
    diff(globals())
    print(prefix, 'GO4:')
    try:
        from some_package import raiseError
    except BaseException as e:
        print('Exception occurred', e)
    else:
        print('Import success.', raiseError.__name__)
    diff(globals())
    print(prefix, 'GO5:')
    try:
        from some_package import Class5
    except BaseException as e:
        print('Exception occurred', e)
    else:
        print('Import success.', Class5)
    diff(globals())
    print(prefix, 'GO6:')
    try:
        from some_package import Class3
    except BaseException as e:
        print('Exception occurred', e)
    else:
        print('Import success.', Class3)
    diff(globals().keys())
os.environ['TEST_SHALL_RAISE_ERROR'] = '1'
attemptImports('With expected errors')
os.environ['TEST_SHALL_RAISE_ERROR'] = '0'
attemptImports('With error resolved')
del sys.modules['some_package']
attemptImports('With deleted module')