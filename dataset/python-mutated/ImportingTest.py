from __future__ import print_function

def localImporter1():
    if False:
        for i in range(10):
            print('nop')
    import os
    return os

def localImporter1a():
    if False:
        print('Hello World!')
    import os as my_os_name
    return my_os_name

def localImporter2():
    if False:
        i = 10
        return i + 15
    from os import path
    return path

def localImporter2a():
    if False:
        print('Hello World!')
    from os import path as renamed
    return renamed
print('Direct module import', localImporter1())
print('Direct module import using rename', localImporter1a())
print('From module import', localImporter2())
print('From module import using rename', localImporter2a())
from os import *
print('Star import gave us', path)
import os.path as myname
print('As import gave', myname)

def localImportFailure():
    if False:
        return 10
    try:
        from os import listdir, listdir2, path
    except Exception as e:
        print('gives', type(e), repr(e))
    try:
        print(path)
    except UnboundLocalError:
        print('and path was not imported', end=' ')
    print('but listdir was', listdir)
print('From import that fails in the middle', end=' ')
localImportFailure()

def nonPackageImportFailure():
    if False:
        return 10
    try:
        from . import whatever
    except Exception as e:
        print(type(e), repr(e))
print('Package import fails in non-package:', end=' ')
nonPackageImportFailure()

def importBuiltinTupleFailure():
    if False:
        while True:
            i = 10
    try:
        value = ('something',)
        __import__(value)
    except Exception as e:
        print(type(e), repr(e))
print('The __import__ built-in optimization can handle tuples:', end=' ')
importBuiltinTupleFailure()