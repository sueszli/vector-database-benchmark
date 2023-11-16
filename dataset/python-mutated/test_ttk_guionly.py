import unittest
from test import support
from test.support import import_helper
from test.support import check_sanitizer
if check_sanitizer(address=True, memory=True):
    raise unittest.SkipTest('Tests involvin libX11 can SEGFAULT on ASAN/MSAN builds')
import_helper.import_module('_tkinter')
support.requires('gui')
import tkinter
from _tkinter import TclError
from tkinter import ttk

def setUpModule():
    if False:
        for i in range(10):
            print('nop')
    root = None
    try:
        root = tkinter.Tk()
        button = ttk.Button(root)
        button.destroy()
        del button
    except TclError as msg:
        raise unittest.SkipTest('ttk not available: %s' % msg)
    finally:
        if root is not None:
            root.destroy()
        del root

def load_tests(loader, tests, pattern):
    if False:
        while True:
            i = 10
    return loader.discover('tkinter.test.test_ttk')
if __name__ == '__main__':
    unittest.main()