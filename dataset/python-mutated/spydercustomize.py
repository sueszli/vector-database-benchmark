import logging
import os
import pdb
import sys
import warnings
from spyder_kernels.customize.spyderpdb import SpyderPdb
if not hasattr(sys, 'argv'):
    sys.argv = ['']
IS_EXT_INTERPRETER = os.environ.get('SPY_EXTERNAL_INTERPRETER') == 'True'
HIDE_CMD_WINDOWS = os.environ.get('SPY_HIDE_CMD') == 'True'
if os.name == 'nt' and HIDE_CMD_WINDOWS:
    import subprocess
    creation_flag = 134217728

    class SubprocessPopen(subprocess.Popen):

        def __init__(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            kwargs['creationflags'] = creation_flag
            super(SubprocessPopen, self).__init__(*args, **kwargs)
    subprocess.Popen = SubprocessPopen
try:
    import sitecustomize
except Exception:
    pass
if os.environ.get('QT_API') == 'pyqt':
    try:
        import sip
        for qtype in ('QString', 'QVariant', 'QDate', 'QDateTime', 'QTextStream', 'QTime', 'QUrl'):
            sip.setapi(qtype, 2)
    except Exception:
        pass
else:
    try:
        os.environ.pop('QT_API')
    except KeyError:
        pass
try:
    from PyQt5 import QtWidgets

    class SpyderQApplication(QtWidgets.QApplication):

        def __init__(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            super(SpyderQApplication, self).__init__(*args, **kwargs)
            SpyderQApplication._instance_list.append(self)
    SpyderQApplication._instance_list = []
    QtWidgets.QApplication = SpyderQApplication
except Exception:
    pass
try:
    from PyQt4 import QtGui

    class SpyderQApplication(QtGui.QApplication):

        def __init__(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            super(SpyderQApplication, self).__init__(*args, **kwargs)
            SpyderQApplication._instance_list.append(self)
    SpyderQApplication._instance_list = []
    QtGui.QApplication = SpyderQApplication
except Exception:
    pass
import unittest
from unittest import TestProgram

class IPyTesProgram(TestProgram):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        test_runner = unittest.TextTestRunner(stream=sys.stderr)
        kwargs['testRunner'] = kwargs.pop('testRunner', test_runner)
        kwargs['exit'] = False
        TestProgram.__init__(self, *args, **kwargs)
unittest.main = IPyTesProgram
try:
    warnings.filterwarnings(action='ignore', category=DeprecationWarning, module='ipykernel.ipkernel')
except Exception:
    pass
try:
    import turtle
    from turtle import Screen, Terminator

    def spyder_bye():
        if False:
            while True:
                i = 10
        try:
            Screen().bye()
            turtle.TurtleScreen._RUNNING = True
        except Terminator:
            pass
    turtle.bye = spyder_bye
except Exception:
    pass
try:
    import pandas as pd
    pd.options.display.encoding = 'utf-8'
    warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='pandas.core.format', message='.*invalid value encountered in.*')
    warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='pandas.formats.format', message='.*invalid value encountered in.*')
except Exception:
    pass
try:
    warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='numpy.core._methods', message='.*invalid value encountered in.*')
except Exception:
    pass
try:
    import multiprocessing.spawn
    _old_preparation_data = multiprocessing.spawn.get_preparation_data

    def _patched_preparation_data(name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Patched get_preparation_data to work when all variables are\n        removed before execution.\n        '
        try:
            d = _old_preparation_data(name)
        except AttributeError:
            main_module = sys.modules['__main__']
            main_module.__spec__ = ''
            d = _old_preparation_data(name)
        if os.name == 'nt' and 'init_main_from_path' in d and (not os.path.exists(d['init_main_from_path'])):
            print('Warning: multiprocessing may need the main file to exist. Please save {}'.format(d['init_main_from_path']))
            del d['init_main_from_path']
        return d
    multiprocessing.spawn.get_preparation_data = _patched_preparation_data
except Exception:
    pass

def _patched_get_terminal_size(fd=None):
    if False:
        for i in range(10):
            print('nop')
    return os.terminal_size((80, 30))
os.get_terminal_size = _patched_get_terminal_size
pdb.Pdb = SpyderPdb

def set_spyder_pythonpath():
    if False:
        print('Hello World!')
    pypath = os.environ.get('SPY_PYTHONPATH')
    if pypath:
        sys.path.extend(pypath.split(os.pathsep))
        os.environ.update({'PYTHONPATH': pypath})
set_spyder_pythonpath()