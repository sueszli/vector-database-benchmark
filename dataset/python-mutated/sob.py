"""
Save and load Small OBjects to and from files, using various formats.

Maintainer: Moshe Zadka
"""
import os
import pickle
import sys
from zope.interface import Interface, implementer
from twisted.persisted import styles
from twisted.python import log, runtime

class IPersistable(Interface):
    """An object which can be saved in several formats to a file"""

    def setStyle(style):
        if False:
            i = 10
            return i + 15
        "Set desired format.\n\n        @type style: string (one of 'pickle' or 'source')\n        "

    def save(tag=None, filename=None, passphrase=None):
        if False:
            while True:
                i = 10
        'Save object to file.\n\n        @type tag: string\n        @type filename: string\n        @type passphrase: string\n        '

@implementer(IPersistable)
class Persistent:
    style = 'pickle'

    def __init__(self, original, name):
        if False:
            for i in range(10):
                print('nop')
        self.original = original
        self.name = name

    def setStyle(self, style):
        if False:
            print('Hello World!')
        "Set desired format.\n\n        @type style: string (one of 'pickle' or 'source')\n        "
        self.style = style

    def _getFilename(self, filename, ext, tag):
        if False:
            while True:
                i = 10
        if filename:
            finalname = filename
            filename = finalname + '-2'
        elif tag:
            filename = f'{self.name}-{tag}-2.{ext}'
            finalname = f'{self.name}-{tag}.{ext}'
        else:
            filename = f'{self.name}-2.{ext}'
            finalname = f'{self.name}.{ext}'
        return (finalname, filename)

    def _saveTemp(self, filename, dumpFunc):
        if False:
            for i in range(10):
                print('nop')
        with open(filename, 'wb') as f:
            dumpFunc(self.original, f)

    def _getStyle(self):
        if False:
            for i in range(10):
                print('nop')
        if self.style == 'source':
            from twisted.persisted.aot import jellyToSource as dumpFunc
            ext = 'tas'
        else:

            def dumpFunc(obj, file=None):
                if False:
                    for i in range(10):
                        print('nop')
                pickle.dump(obj, file, 2)
            ext = 'tap'
        return (ext, dumpFunc)

    def save(self, tag=None, filename=None, passphrase=None):
        if False:
            while True:
                i = 10
        'Save object to file.\n\n        @type tag: string\n        @type filename: string\n        @type passphrase: string\n        '
        (ext, dumpFunc) = self._getStyle()
        if passphrase is not None:
            raise TypeError('passphrase must be None')
        (finalname, filename) = self._getFilename(filename, ext, tag)
        log.msg('Saving ' + self.name + ' application to ' + finalname + '...')
        self._saveTemp(filename, dumpFunc)
        if runtime.platformType == 'win32' and os.path.isfile(finalname):
            os.remove(finalname)
        os.rename(filename, finalname)
        log.msg('Saved.')
Persistant = Persistent

class _EverythingEphemeral(styles.Ephemeral):
    initRun = 0

    def __init__(self, mainMod):
        if False:
            print('Hello World!')
        "\n        @param mainMod: The '__main__' module that this class will proxy.\n        "
        self.mainMod = mainMod

    def __getattr__(self, key):
        if False:
            i = 10
            return i + 15
        try:
            return getattr(self.mainMod, key)
        except AttributeError:
            if self.initRun:
                raise
            else:
                log.msg('Warning!  Loading from __main__: %s' % key)
                return styles.Ephemeral()

def load(filename, style):
    if False:
        i = 10
        return i + 15
    "Load an object from a file.\n\n    Deserialize an object from a file. The file can be encrypted.\n\n    @param filename: string\n    @param style: string (one of 'pickle' or 'source')\n    "
    mode = 'r'
    if style == 'source':
        from twisted.persisted.aot import unjellyFromSource as _load
    else:
        (_load, mode) = (pickle.load, 'rb')
    fp = open(filename, mode)
    ee = _EverythingEphemeral(sys.modules['__main__'])
    sys.modules['__main__'] = ee
    ee.initRun = 1
    with fp:
        try:
            value = _load(fp)
        finally:
            sys.modules['__main__'] = ee.mainMod
    styles.doUpgrade()
    ee.initRun = 0
    persistable = IPersistable(value, None)
    if persistable is not None:
        persistable.setStyle(style)
    return value

def loadValueFromFile(filename, variable):
    if False:
        for i in range(10):
            print('nop')
    'Load the value of a variable in a Python file.\n\n    Run the contents of the file in a namespace and return the result of the\n    variable named C{variable}.\n\n    @param filename: string\n    @param variable: string\n    '
    with open(filename) as fileObj:
        data = fileObj.read()
    d = {'__file__': filename}
    codeObj = compile(data, filename, 'exec')
    eval(codeObj, d, d)
    value = d[variable]
    return value

def guessType(filename):
    if False:
        i = 10
        return i + 15
    ext = os.path.splitext(filename)[1]
    return {'.tac': 'python', '.etac': 'python', '.py': 'python', '.tap': 'pickle', '.etap': 'pickle', '.tas': 'source', '.etas': 'source'}[ext]
__all__ = ['loadValueFromFile', 'load', 'Persistent', 'Persistant', 'IPersistable', 'guessType']