"""Contains miscellaneous utility functions and classes."""
from __future__ import annotations
__all__ = ['indent', 'doc', 'adjust', 'difference', 'intersection', 'union', 'sameElements', 'makeList', 'makeTuple', 'list2dict', 'invertDict', 'invertDictLossless', 'uniqueElements', 'disjoint', 'contains', 'replace', 'reduceAngle', 'fitSrcAngle2Dest', 'fitDestAngle2Src', 'closestDestAngle2', 'closestDestAngle', 'getSetterName', 'getSetter', 'Functor', 'Stack', 'Queue', 'bound', 'clamp', 'lerp', 'average', 'addListsByValue', 'boolEqual', 'lineupPos', 'formatElapsedSeconds', 'solveQuadratic', 'findPythonModule', 'mostDerivedLast', 'clampScalar', 'weightedChoice', 'randFloat', 'normalDistrib', 'weightedRand', 'randUint31', 'randInt32', 'SerialNumGen', 'SerialMaskedGen', 'serialNum', 'uniqueName', 'Singleton', 'SingletonError', 'printListEnum', 'safeRepr', 'fastRepr', 'isDefaultValue', 'ScratchPad', 'Sync', 'itype', 'getNumberedTypedString', 'getNumberedTypedSortedString', 'printNumberedTyped', 'DelayedCall', 'DelayedFunctor', 'FrameDelayedCall', 'SubframeCall', 'getBase', 'GoldenRatio', 'GoldenRectangle', 'rad90', 'rad180', 'rad270', 'rad360', 'nullGen', 'loopGen', 'makeFlywheelGen', 'flywheel', 'listToIndex2item', 'listToItem2index', 'formatTimeCompact', 'deeptype', 'StdoutCapture', 'StdoutPassthrough', 'Averager', 'getRepository', 'formatTimeExact', 'typeName', 'safeTypeName', 'histogramDict', 'unescapeHtmlString']
if __debug__:
    __all__ += ['StackTrace', 'traceFunctionCall', 'traceParentCall', 'printThisCall', 'stackEntryInfo', 'lineInfo', 'callerInfo', 'lineTag', 'profileFunc', 'profiled', 'startProfile', 'printProfile', 'getProfileResultString', 'printStack', 'printReverseStack']
import types
import math
import os
import sys
import random
import time
import builtins
import importlib
import functools
from typing import Callable
__report_indent = 3
from panda3d.core import ConfigVariableBool, ConfigVariableString, ConfigFlags
from panda3d.core import ClockObject

class Functor:

    def __init__(self, function, *args, **kargs):
        if False:
            for i in range(10):
                print('nop')
        assert callable(function), 'function should be a callable obj'
        self._function = function
        self._args = args
        self._kargs = kargs
        if hasattr(self._function, '__name__'):
            self.__name__ = self._function.__name__
        else:
            self.__name__ = str(itype(self._function))
        if hasattr(self._function, '__doc__'):
            self.__doc__ = self._function.__doc__
        else:
            self.__doc__ = self.__name__

    def destroy(self):
        if False:
            i = 10
            return i + 15
        del self._function
        del self._args
        del self._kargs
        del self.__name__
        del self.__doc__

    def _do__call__(self, *args, **kargs):
        if False:
            return 10
        _kargs = self._kargs.copy()
        _kargs.update(kargs)
        return self._function(*self._args + args, **_kargs)
    __call__ = _do__call__

    def __repr__(self):
        if False:
            print('Hello World!')
        s = 'Functor(%s' % self._function.__name__
        for arg in self._args:
            try:
                argStr = repr(arg)
            except Exception:
                argStr = 'bad repr: %s' % arg.__class__
            s += ', %s' % argStr
        for (karg, value) in list(self._kargs.items()):
            s += ', %s=%s' % (karg, repr(value))
        s += ')'
        return s

class Stack:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__list = []

    def push(self, item):
        if False:
            i = 10
            return i + 15
        self.__list.append(item)

    def top(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__list[-1]

    def pop(self):
        if False:
            while True:
                i = 10
        return self.__list.pop()

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        self.__list = []

    def isEmpty(self):
        if False:
            print('Hello World!')
        return len(self.__list) == 0

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.__list)

class Queue:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__list = []

    def push(self, item):
        if False:
            print('Hello World!')
        self.__list.append(item)

    def top(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__list[0]

    def front(self):
        if False:
            return 10
        return self.__list[0]

    def back(self):
        if False:
            while True:
                i = 10
        return self.__list[-1]

    def pop(self):
        if False:
            print('Hello World!')
        return self.__list.pop(0)

    def clear(self):
        if False:
            return 10
        self.__list = []

    def isEmpty(self):
        if False:
            i = 10
            return i + 15
        return len(self.__list) == 0

    def __len__(self):
        if False:
            return 10
        return len(self.__list)

def indent(stream, numIndents, str):
    if False:
        print('Hello World!')
    '\n    Write str to stream with numIndents in front of it\n    '
    stream.write('    ' * numIndents + str)
if __debug__:
    import traceback
    import marshal

    class StackTrace:

        def __init__(self, label='', start=0, limit=None):
            if False:
                print('Hello World!')
            '\n            label is a string (or anything that be be a string)\n            that is printed as part of the trace back.\n            This is just to make it easier to tell what the\n            stack trace is referring to.\n            start is an integer number of stack frames back\n            from the most recent.  (This is automatically\n            bumped up by one to skip the __init__ call\n            to the StackTrace).\n            limit is an integer number of stack frames\n            to record (or None for unlimited).\n            '
            self.label = label
            if limit is not None:
                self.trace = traceback.extract_stack(sys._getframe(1 + start), limit=limit)
            else:
                self.trace = traceback.extract_stack(sys._getframe(1 + start))

        def compact(self):
            if False:
                return 10
            r = ''
            comma = ','
            for (filename, lineNum, funcName, text) in self.trace:
                r += '%s.%s:%s%s' % (filename[:filename.rfind('.py')][filename.rfind('\\') + 1:], funcName, lineNum, comma)
            if len(r) > 0:
                r = r[:-len(comma)]
            return r

        def reverseCompact(self):
            if False:
                i = 10
                return i + 15
            r = ''
            comma = ','
            for (filename, lineNum, funcName, text) in self.trace:
                r = '%s.%s:%s%s%s' % (filename[:filename.rfind('.py')][filename.rfind('\\') + 1:], funcName, lineNum, comma, r)
            if len(r) > 0:
                r = r[:-len(comma)]
            return r

        def __str__(self):
            if False:
                for i in range(10):
                    print('nop')
            r = 'Debug stack trace of %s (back %s frames):\n' % (self.label, len(self.trace))
            for i in traceback.format_list(self.trace):
                r += i
            r += '***** NOTE: This is not a crash. This is a debug stack trace. *****'
            return r

    def printStack():
        if False:
            for i in range(10):
                print('nop')
        print(StackTrace(start=1).compact())
        return True

    def printReverseStack():
        if False:
            for i in range(10):
                print('nop')
        print(StackTrace(start=1).reverseCompact())
        return True

    def printVerboseStack():
        if False:
            return 10
        print(StackTrace(start=1))
        return True

    def traceFunctionCall(frame):
        if False:
            i = 10
            return i + 15
        '\n        return a string that shows the call frame with calling arguments.\n        e.g.\n        foo(x=234, y=135)\n        '
        f = frame
        co = f.f_code
        dict = f.f_locals
        n = co.co_argcount
        if co.co_flags & 4:
            n = n + 1
        if co.co_flags & 8:
            n = n + 1
        r = ''
        if 'self' in dict:
            r = '%s.' % (dict['self'].__class__.__name__,)
        r += '%s(' % (f.f_code.co_name,)
        comma = 0
        for i in range(n):
            name = co.co_varnames[i]
            if name == 'self':
                continue
            if comma:
                r += ', '
            else:
                comma = 1
            r += name
            r += '='
            if name in dict:
                v = safeRepr(dict[name])
                if len(v) > 2000:
                    r += v[:2000] + '...'
                else:
                    r += v
            else:
                r += '*** undefined ***'
        return r + ')'

    def traceParentCall():
        if False:
            print('Hello World!')
        return traceFunctionCall(sys._getframe(2))

    def printThisCall():
        if False:
            i = 10
            return i + 15
        print(traceFunctionCall(sys._getframe(1)))
        return 1
_POS_LIST = 4
_KEY_DICT = 8

def doc(obj):
    if False:
        print('Hello World!')
    if isinstance(obj, types.MethodType) or isinstance(obj, types.FunctionType):
        print(obj.__doc__)

def adjust(command=None, dim=1, parent=None, **kw):
    if False:
        print('Hello World!')
    "\n    adjust(command = None, parent = None, **kw)\n    Popup and entry scale to adjust a parameter\n\n    Accepts any Slider keyword argument.  Typical arguments include:\n    command: The one argument command to execute\n    min: The min value of the slider\n    max: The max value of the slider\n    resolution: The resolution of the slider\n    text: The label on the slider\n\n    These values can be accessed and/or changed after the fact\n    >>> vg = adjust()\n    >>> vg['min']\n    0.0\n    >>> vg['min'] = 10.0\n    >>> vg['min']\n    10.0\n    "
    Valuator = importlib.import_module('direct.tkwidgets.Valuator')
    if command:
        kw['command'] = lambda x: command(*x)
        if parent is None:
            kw['title'] = command.__name__
    kw['dim'] = dim
    if not parent:
        vg = Valuator.ValuatorGroupPanel(parent, **kw)
    else:
        vg = Valuator.ValuatorGroup(parent, **kw)
        vg.pack(expand=1, fill='x')
    return vg

def difference(a, b):
    if False:
        for i in range(10):
            print('nop')
    '\n    difference(list, list):\n    '
    if not a:
        return b
    if not b:
        return a
    d = []
    for i in a:
        if i not in b and i not in d:
            d.append(i)
    for i in b:
        if i not in a and i not in d:
            d.append(i)
    return d

def intersection(a, b):
    if False:
        for i in range(10):
            print('nop')
    '\n    intersection(list, list):\n    '
    if not a or not b:
        return []
    d = []
    for i in a:
        if i in b and i not in d:
            d.append(i)
    for i in b:
        if i in a and i not in d:
            d.append(i)
    return d

def union(a, b):
    if False:
        while True:
            i = 10
    '\n    union(list, list):\n    '
    c = a[:]
    for i in b:
        if i not in c:
            c.append(i)
    return c

def sameElements(a, b):
    if False:
        print('Hello World!')
    if len(a) != len(b):
        return 0
    for elem in a:
        if elem not in b:
            return 0
    for elem in b:
        if elem not in a:
            return 0
    return 1

def makeList(x):
    if False:
        return 10
    'returns x, converted to a list'
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]

def makeTuple(x):
    if False:
        print('Hello World!')
    'returns x, converted to a tuple'
    if isinstance(x, list):
        return tuple(x)
    elif isinstance(x, tuple):
        return x
    else:
        return (x,)

def list2dict(L, value=None):
    if False:
        i = 10
        return i + 15
    'creates dict using elements of list, all assigned to same value'
    return dict([(k, value) for k in L])

def listToIndex2item(L):
    if False:
        i = 10
        return i + 15
    'converts list to dict of list index->list item'
    d = {}
    for (i, item) in enumerate(L):
        d[i] = item
    return d
assert listToIndex2item(['a', 'b']) == {0: 'a', 1: 'b'}

def listToItem2index(L):
    if False:
        for i in range(10):
            print('nop')
    'converts list to dict of list item->list index\n    This is lossy if there are duplicate list items'
    d = {}
    for (i, item) in enumerate(L):
        d[item] = i
    return d
assert listToItem2index(['a', 'b']) == {'a': 0, 'b': 1}

def invertDict(D, lossy=False):
    if False:
        print('Hello World!')
    "creates a dictionary by 'inverting' D; keys are placed in the new\n    dictionary under their corresponding value in the old dictionary.\n    It is an error if D contains any duplicate values.\n\n    >>> old = {'key1':1, 'key2':2}\n    >>> invertDict(old)\n    {1: 'key1', 2: 'key2'}\n    "
    n = {}
    for (key, value) in D.items():
        if not lossy and value in n:
            raise Exception('duplicate key in invertDict: %s' % value)
        n[value] = key
    return n

def invertDictLossless(D):
    if False:
        i = 10
        return i + 15
    "similar to invertDict, but values of new dict are lists of keys from\n    old dict. No information is lost.\n\n    >>> old = {'key1':1, 'key2':2, 'keyA':2}\n    >>> invertDictLossless(old)\n    {1: ['key1'], 2: ['key2', 'keyA']}\n    "
    n = {}
    for (key, value) in D.items():
        n.setdefault(value, [])
        n[value].append(key)
    return n

def uniqueElements(L):
    if False:
        for i in range(10):
            print('nop')
    'are all elements of list unique?'
    return len(L) == len(list2dict(L))

def disjoint(L1, L2):
    if False:
        return 10
    'returns non-zero if L1 and L2 have no common elements'
    used = dict([(k, None) for k in L1])
    for k in L2:
        if k in used:
            return 0
    return 1

def contains(whole, sub):
    if False:
        i = 10
        return i + 15
    '\n    Return 1 if whole contains sub, 0 otherwise\n    '
    if whole == sub:
        return 1
    for elem in sub:
        if elem not in whole:
            return 0
    return 1

def replace(list, old, new, all=0):
    if False:
        print('Hello World!')
    "\n    replace 'old' with 'new' in 'list'\n    if all == 0, replace first occurrence\n    otherwise replace all occurrences\n    returns the number of items replaced\n    "
    if old not in list:
        return 0
    if not all:
        i = list.index(old)
        list[i] = new
        return 1
    else:
        numReplaced = 0
        for i in range(len(list)):
            if list[i] == old:
                numReplaced += 1
                list[i] = new
        return numReplaced
rad90 = math.pi / 2.0
rad180 = math.pi
rad270 = 1.5 * math.pi
rad360 = 2.0 * math.pi

def reduceAngle(deg):
    if False:
        i = 10
        return i + 15
    '\n    Reduces an angle (in degrees) to a value in [-180..180)\n    '
    return (deg + 180.0) % 360.0 - 180.0

def fitSrcAngle2Dest(src, dest):
    if False:
        for i in range(10):
            print('nop')
    '\n    given a src and destination angle, returns an equivalent src angle\n    that is within [-180..180) of dest\n    examples:\n    fitSrcAngle2Dest(30, 60) == 30\n    fitSrcAngle2Dest(60, 30) == 60\n    fitSrcAngle2Dest(0, 180) == 0\n    fitSrcAngle2Dest(-1, 180) == 359\n    fitSrcAngle2Dest(-180, 180) == 180\n    '
    return dest + reduceAngle(src - dest)

def fitDestAngle2Src(src, dest):
    if False:
        print('Hello World!')
    '\n    given a src and destination angle, returns an equivalent dest angle\n    that is within [-180..180) of src\n    examples:\n    fitDestAngle2Src(30, 60) == 60\n    fitDestAngle2Src(60, 30) == 30\n    fitDestAngle2Src(0, 180) == -180\n    fitDestAngle2Src(1, 180) == 180\n    '
    return src + reduceAngle(dest - src)

def closestDestAngle2(src, dest):
    if False:
        print('Hello World!')
    diff = src - dest
    if diff > 180:
        return dest - 360
    elif diff < -180:
        return dest + 360
    else:
        return dest

def closestDestAngle(src, dest):
    if False:
        for i in range(10):
            print('nop')
    diff = src - dest
    if diff > 180:
        return src - (diff - 360)
    elif diff < -180:
        return src - (360 + diff)
    else:
        return dest

class StdoutCapture:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._oldStdout = sys.stdout
        sys.stdout = self
        self._string = ''

    def destroy(self):
        if False:
            for i in range(10):
                print('nop')
        sys.stdout = self._oldStdout
        del self._oldStdout

    def getString(self):
        if False:
            while True:
                i = 10
        return self._string

    def write(self, string):
        if False:
            for i in range(10):
                print('nop')
        self._string = ''.join([self._string, string])

class StdoutPassthrough(StdoutCapture):

    def write(self, string):
        if False:
            i = 10
            return i + 15
        self._string = ''.join([self._string, string])
        self._oldStdout.write(string)
if __debug__:
    from io import StringIO
    PyUtilProfileDefaultFilename = 'profiledata'
    PyUtilProfileDefaultLines = 80
    PyUtilProfileDefaultSorts = ['cumulative', 'time', 'calls']
    _ProfileResultStr = ''

    def getProfileResultString():
        if False:
            return 10
        return _ProfileResultStr

    def profileFunc(callback, name, terse, log=True):
        if False:
            print('Hello World!')
        global _ProfileResultStr
        if 'globalProfileFunc' in builtins.__dict__:
            base.notify.warning('PythonUtil.profileStart(%s): aborted, already profiling %s' % (name, builtins.globalProfileFunc))
            return
        builtins.globalProfileFunc = callback
        builtins.globalProfileResult = [None]
        prefix = '***** START PROFILE: %s *****' % name
        if log:
            print(prefix)
        startProfile(cmd='globalProfileResult[0]=globalProfileFunc()', callInfo=not terse, silent=not log)
        suffix = '***** END PROFILE: %s *****' % name
        if log:
            print(suffix)
        else:
            _ProfileResultStr = '%s\n%s\n%s' % (prefix, _ProfileResultStr, suffix)
        result = builtins.globalProfileResult[0]
        del builtins.globalProfileFunc
        del builtins.globalProfileResult
        return result

    def profiled(category=None, terse=False):
        if False:
            print('Hello World!')
        ' decorator for profiling functions\n        turn categories on and off via "want-profile-categoryName 1"\n\n        e.g.::\n\n            @profiled(\'particles\')\n            def loadParticles():\n                ...\n\n        ::\n\n            want-profile-particles 1\n        '
        assert type(category) in (str, type(None)), 'must provide a category name for @profiled'

        def profileDecorator(f):
            if False:
                print('Hello World!')

            def _profiled(*args, **kArgs):
                if False:
                    for i in range(10):
                        print('nop')
                name = '(%s) %s from %s' % (category, f.__name__, f.__module__)
                if category is None or ConfigVariableBool('want-profile-%s' % category, False).value:
                    return profileFunc(Functor(f, *args, **kArgs), name, terse)
                else:
                    return f(*args, **kArgs)
            _profiled.__doc__ = f.__doc__
            return _profiled
        return profileDecorator
    movedOpenFuncs: list[Callable] = []
    movedDumpFuncs: list[Callable] = []
    movedLoadFuncs: list[Callable] = []
    profileFilenames = set()
    profileFilenameList = Stack()
    profileFilename2file = {}
    profileFilename2marshalData = {}

    def _profileOpen(filename, *args, **kArgs):
        if False:
            while True:
                i = 10
        if filename in profileFilenames:
            if filename not in profileFilename2file:
                file = StringIO()
                file._profFilename = filename
                profileFilename2file[filename] = file
            else:
                file = profileFilename2file[filename]
        else:
            file = movedOpenFuncs[-1](filename, *args, **kArgs)
        return file

    def _profileMarshalDump(data, file):
        if False:
            while True:
                i = 10
        if isinstance(file, StringIO) and hasattr(file, '_profFilename'):
            if file._profFilename in profileFilenames:
                profileFilename2marshalData[file._profFilename] = data
                return None
        return movedDumpFuncs[-1](data, file)

    def _profileMarshalLoad(file):
        if False:
            i = 10
            return i + 15
        if isinstance(file, StringIO) and hasattr(file, '_profFilename'):
            if file._profFilename in profileFilenames:
                return profileFilename2marshalData[file._profFilename]
        return movedLoadFuncs[-1](file)

    def _installProfileCustomFuncs(filename):
        if False:
            print('Hello World!')
        assert filename not in profileFilenames
        profileFilenames.add(filename)
        profileFilenameList.push(filename)
        movedOpenFuncs.append(builtins.open)
        builtins.open = _profileOpen
        movedDumpFuncs.append(marshal.dump)
        marshal.dump = _profileMarshalDump
        movedLoadFuncs.append(marshal.load)
        marshal.load = _profileMarshalLoad

    def _getProfileResultFileInfo(filename):
        if False:
            i = 10
            return i + 15
        return (profileFilename2file.get(filename, None), profileFilename2marshalData.get(filename, None))

    def _setProfileResultsFileInfo(filename, info):
        if False:
            for i in range(10):
                print('nop')
        (f, m) = info
        if f:
            profileFilename2file[filename] = f
        if m:
            profileFilename2marshalData[filename] = m

    def _clearProfileResultFileInfo(filename):
        if False:
            while True:
                i = 10
        profileFilename2file.pop(filename, None)
        profileFilename2marshalData.pop(filename, None)

    def _removeProfileCustomFuncs(filename):
        if False:
            return 10
        assert profileFilenameList.top() == filename
        marshal.load = movedLoadFuncs.pop()
        marshal.dump = movedDumpFuncs.pop()
        builtins.open = movedOpenFuncs.pop()
        profileFilenames.remove(filename)
        profileFilenameList.pop()
        profileFilename2file.pop(filename, None)
        profileFilename2marshalData.pop(filename, None)

    def _profileWithoutGarbageLeak(cmd, filename):
        if False:
            i = 10
            return i + 15
        import profile
        Profile = profile.Profile
        statement = cmd
        sort = -1
        prof = Profile()
        try:
            prof = prof.run(statement)
        except SystemExit:
            pass
        if filename is not None:
            prof.dump_stats(filename)
        else:
            prof.print_stats(sort)
        del prof.dispatcher

    def startProfile(filename=PyUtilProfileDefaultFilename, lines=PyUtilProfileDefaultLines, sorts=PyUtilProfileDefaultSorts, silent=0, callInfo=1, useDisk=False, cmd='run()'):
        if False:
            for i in range(10):
                print('nop')
        filename = '%s.%s%s' % (filename, randUint31(), randUint31())
        if not useDisk:
            _installProfileCustomFuncs(filename)
        _profileWithoutGarbageLeak(cmd, filename)
        if silent:
            extractProfile(filename, lines, sorts, callInfo)
        else:
            printProfile(filename, lines, sorts, callInfo)
        if not useDisk:
            _removeProfileCustomFuncs(filename)
        else:
            os.remove(filename)

    def printProfile(filename=PyUtilProfileDefaultFilename, lines=PyUtilProfileDefaultLines, sorts=PyUtilProfileDefaultSorts, callInfo=1):
        if False:
            for i in range(10):
                print('nop')
        import pstats
        s = pstats.Stats(filename)
        s.strip_dirs()
        for sort in sorts:
            s.sort_stats(sort)
            s.print_stats(lines)
            if callInfo:
                s.print_callees(lines)
                s.print_callers(lines)

    def extractProfile(*args, **kArgs):
        if False:
            i = 10
            return i + 15
        global _ProfileResultStr
        sc = StdoutCapture()
        printProfile(*args, **kArgs)
        _ProfileResultStr = sc.getString()
        sc.destroy()

def getSetterName(valueName, prefix='set'):
    if False:
        for i in range(10):
            print('nop')
    return '%s%s%s' % (prefix, valueName[0].upper(), valueName[1:])

def getSetter(targetObj, valueName, prefix='set'):
    if False:
        while True:
            i = 10
    return getattr(targetObj, getSetterName(valueName, prefix))

def mostDerivedLast(classList):
    if False:
        i = 10
        return i + 15
    'pass in list of classes. sorts list in-place, with derived classes\n    appearing after their bases'

    class ClassSortKey(object):
        __slots__ = ('classobj',)

        def __init__(self, classobj):
            if False:
                i = 10
                return i + 15
            self.classobj = classobj

        def __lt__(self, other):
            if False:
                i = 10
                return i + 15
            return issubclass(other.classobj, self.classobj)
    classList.sort(key=ClassSortKey)

def bound(value, bound1, bound2):
    if False:
        while True:
            i = 10
    '\n    returns value if value is between bound1 and bound2\n    otherwise returns bound that is closer to value\n    '
    if bound1 > bound2:
        return min(max(value, bound2), bound1)
    else:
        return min(max(value, bound1), bound2)
clamp = bound

def lerp(v0, v1, t):
    if False:
        i = 10
        return i + 15
    '\n    returns a value lerped between v0 and v1, according to t\n    t == 0 maps to v0, t == 1 maps to v1\n    '
    return v0 + (v1 - v0) * t

def getShortestRotation(start, end):
    if False:
        while True:
            i = 10
    "\n    Given two heading values, return a tuple describing\n    the shortest interval from 'start' to 'end'.  This tuple\n    can be used to lerp a camera between two rotations\n    while avoiding the 'spin' problem.\n    "
    (start, end) = (start % 360, end % 360)
    if abs(end - start) > 180:
        if end < start:
            end += 360
        else:
            start += 360
    return (start, end)

def average(*args):
    if False:
        return 10
    ' returns simple average of list of values '
    val = 0.0
    for arg in args:
        val += arg
    return val / len(args)

class Averager:

    def __init__(self, name):
        if False:
            return 10
        self._name = name
        self.reset()

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        self._total = 0.0
        self._count = 0

    def addValue(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._total += value
        self._count += 1

    def getAverage(self):
        if False:
            i = 10
            return i + 15
        return self._total / self._count

    def getCount(self):
        if False:
            return 10
        return self._count

def addListsByValue(a, b):
    if False:
        i = 10
        return i + 15
    '\n    returns a new array containing the sums of the two array arguments\n    (c[0] = a[0 + b[0], etc.)\n    '
    c = []
    for (x, y) in zip(a, b):
        c.append(x + y)
    return c

def boolEqual(a, b):
    if False:
        while True:
            i = 10
    '\n    returns true if a and b are both true or both false.\n    returns false otherwise\n    (a.k.a. xnor -- eXclusive Not OR).\n    '
    return a and b or not (a or b)

def lineupPos(i, num, spacing):
    if False:
        return 10
    "\n    use to line up a series of 'num' objects, in one dimension,\n    centered around zero\n    'i' is the index of the object in the lineup\n    'spacing' is the amount of space between objects in the lineup\n    "
    assert num >= 1
    assert i >= 0 and i < num
    pos = float(i) * spacing
    return pos - float(spacing) * (num - 1) / 2.0

def formatElapsedSeconds(seconds):
    if False:
        print('Hello World!')
    '\n    Returns a string of the form "mm:ss" or "hh:mm:ss" or "n days",\n    representing the indicated elapsed time in seconds.\n    '
    sign = ''
    if seconds < 0:
        seconds = -seconds
        sign = '-'
    seconds = math.floor(seconds)
    hours = math.floor(seconds / (60 * 60))
    if hours > 36:
        days = math.floor((hours + 12) / 24)
        return '%s%d days' % (sign, days)
    seconds -= hours * (60 * 60)
    minutes = int(seconds / 60)
    seconds -= minutes * 60
    if hours != 0:
        return '%s%d:%02d:%02d' % (sign, hours, minutes, seconds)
    else:
        return '%s%d:%02d' % (sign, minutes, seconds)

def solveQuadratic(a, b, c):
    if False:
        while True:
            i = 10
    if a == 0.0:
        return None
    D = b * b - 4.0 * a * c
    if D < 0:
        return None
    elif D == 0:
        return -b / (2.0 * a)
    else:
        sqrtD = math.sqrt(D)
        twoA = 2.0 * a
        root1 = (-b - sqrtD) / twoA
        root2 = (-b + sqrtD) / twoA
        return [root1, root2]
if __debug__:

    def stackEntryInfo(depth=0, baseFileName=1):
        if False:
            return 10
        "\n        returns the sourcefilename, line number, and function name of\n        an entry in the stack.\n        'depth' is how far back to go in the stack; 0 is the caller of this\n        function, 1 is the function that called the caller of this function, etc.\n        by default, strips off the path of the filename; override with baseFileName\n        returns (fileName, lineNum, funcName) --> (string, int, string)\n        returns (None, None, None) on error\n        "
        import inspect
        try:
            stack = None
            frame = None
            try:
                stack = inspect.stack()
                frame = stack[depth + 1]
                filename = frame[1]
                if baseFileName:
                    filename = os.path.basename(filename)
                lineNum = frame[2]
                funcName = frame[3]
                result = (filename, lineNum, funcName)
            finally:
                del stack
                del frame
        except Exception:
            result = (None, None, None)
        return result

    def lineInfo(baseFileName=1):
        if False:
            print('Hello World!')
        "\n        returns the sourcefilename, line number, and function name of the\n        code that called this function\n        (answers the question: 'hey lineInfo, where am I in the codebase?')\n        see stackEntryInfo, above, for info on 'baseFileName' and return types\n        "
        return stackEntryInfo(1, baseFileName)

    def callerInfo(baseFileName=1, howFarBack=0):
        if False:
            for i in range(10):
                print('nop')
        "\n        returns the sourcefilename, line number, and function name of the\n        caller of the function that called this function\n        (answers the question: 'hey callerInfo, who called me?')\n        see stackEntryInfo, above, for info on 'baseFileName' and return types\n        "
        return stackEntryInfo(2 + howFarBack, baseFileName)

    def lineTag(baseFileName=1, verbose=0, separator=':'):
        if False:
            while True:
                i = 10
        "\n        returns a string containing the sourcefilename and line number\n        of the code that called this function\n        (equivalent to lineInfo, above, with different return type)\n        see stackEntryInfo, above, for info on 'baseFileName'\n\n        if 'verbose' is false, returns a compact string of the form\n        'fileName:lineNum:funcName'\n        if 'verbose' is true, returns a longer string that matches the\n        format of Python stack trace dumps\n\n        returns empty string on error\n        "
        (fileName, lineNum, funcName) = callerInfo(baseFileName)
        if fileName is None:
            return ''
        if verbose:
            return 'File "%s", line %s, in %s' % (fileName, lineNum, funcName)
        else:
            return '%s%s%s%s%s' % (fileName, separator, lineNum, separator, funcName)

def findPythonModule(module):
    if False:
        while True:
            i = 10
    filename = module + '.py'
    for dir in sys.path:
        pathname = os.path.join(dir, filename)
        if os.path.exists(pathname):
            return pathname
    return None

def clampScalar(value, a, b):
    if False:
        i = 10
        return i + 15
    if a < b:
        if value < a:
            return a
        elif value > b:
            return b
        else:
            return value
    elif value < b:
        return b
    elif value > a:
        return a
    else:
        return value

def weightedChoice(choiceList, rng=random.random, sum=None):
    if False:
        print('Hello World!')
    "given a list of (weight, item) pairs, chooses an item based on the\n    weights. rng must return 0..1. if you happen to have the sum of the\n    weights, pass it in 'sum'."
    if not choiceList:
        raise IndexError('Cannot choose from an empty sequence')
    if sum is None:
        sum = 0.0
        for (weight, item) in choiceList:
            sum += weight
    rand = rng()
    accum = rand * sum
    item = None
    for (weight, item) in choiceList:
        accum -= weight
        if accum <= 0.0:
            return item
    return item

def randFloat(a, b=0.0, rng=random.random):
    if False:
        while True:
            i = 10
    'returns a random float in [a, b]\n    call with single argument to generate random float between arg and zero\n    '
    return lerp(a, b, rng())

def normalDistrib(a, b, gauss=random.gauss):
    if False:
        while True:
            i = 10
    '\n    NOTE: assumes a < b\n\n    Returns random number between a and b, using gaussian distribution, with\n    mean=avg(a, b), and a standard deviation that fits ~99.7% of the curve\n    between a and b.\n\n    For ease of use, outlying results are re-computed until result is in [a, b]\n    This should fit the remaining .3% of the curve that lies outside [a, b]\n    uniformly onto the curve inside [a, b]\n\n    ------------------------------------------------------------------------\n    The 68-95-99.7% Rule\n    ====================\n    All normal density curves satisfy the following property which is often\n      referred to as the Empirical Rule:\n    68% of the observations fall within 1 standard deviation of the mean.\n    95% of the observations fall within 2 standard deviations of the mean.\n    99.7% of the observations fall within 3 standard deviations of the mean.\n\n    Thus, for a normal distribution, almost all values lie within 3 standard\n      deviations of the mean.\n    ------------------------------------------------------------------------\n\n    In calculating our standard deviation, we divide (b-a) by 6, since the\n    99.7% figure includes 3 standard deviations _on_either_side_ of the mean.\n    '
    while True:
        r = gauss((a + b) * 0.5, (b - a) / 6.0)
        if r >= a and r <= b:
            return r

def weightedRand(valDict, rng=random.random):
    if False:
        return 10
    '\n    pass in a dictionary with a selection -> weight mapping.  E.g.::\n\n        {"Choice 1": 10,\n         "Choice 2": 30,\n         "bear":     100}\n\n    - Weights need not add up to any particular value.\n    - The actual selection will be returned.\n    '
    selections = list(valDict.keys())
    weights = list(valDict.values())
    totalWeight = 0
    for weight in weights:
        totalWeight += weight
    randomWeight = rng() * totalWeight
    for i in range(len(weights)):
        totalWeight -= weights[i]
        if totalWeight <= randomWeight:
            return selections[i]
    assert True, 'Should never get here'
    return selections[-1]

def randUint31(rng=random.random):
    if False:
        i = 10
        return i + 15
    'returns a random integer in [0..2^31).\n    rng must return float in [0..1]'
    return int(rng() * 2147483647)

def randInt32(rng=random.random):
    if False:
        while True:
            i = 10
    'returns a random integer in [-2147483648..2147483647].\n    rng must return float in [0..1]\n    '
    i = int(rng() * 2147483647)
    if rng() < 0.5:
        i *= -1
    return i

class SerialNumGen:
    """generates serial numbers"""

    def __init__(self, start=None):
        if False:
            i = 10
            return i + 15
        if start is None:
            start = 0
        self.__counter = start - 1

    def next(self):
        if False:
            return 10
        self.__counter += 1
        return self.__counter
    __next__ = next

class SerialMaskedGen(SerialNumGen):

    def __init__(self, mask, start=None):
        if False:
            while True:
                i = 10
        self._mask = mask
        SerialNumGen.__init__(self, start)

    def next(self):
        if False:
            return 10
        v = SerialNumGen.next(self)
        return v & self._mask
    __next__ = next
_serialGen = SerialNumGen()

def serialNum():
    if False:
        i = 10
        return i + 15
    return _serialGen.next()

def uniqueName(name):
    if False:
        while True:
            i = 10
    return f'{name}-{serialNum()}'

class Singleton(type):

    def __init__(cls, name, bases, dic):
        if False:
            print('Hello World!')
        super(Singleton, cls).__init__(name, bases, dic)
        cls.instance = None

    def __call__(cls, *args, **kw):
        if False:
            while True:
                i = 10
        if cls.instance is None:
            cls.instance = super(Singleton, cls).__call__(*args, **kw)
        return cls.instance

class SingletonError(ValueError):
    """ Used to indicate an inappropriate value for a Singleton."""

def printListEnumGen(l):
    if False:
        while True:
            i = 10
    digits = 0
    n = len(l)
    while n > 0:
        digits += 1
        n //= 10
    format = '%0' + '%s' % digits + 'i:%s'
    for i in range(len(l)):
        print(format % (i, l[i]))
        yield None

def printListEnum(l):
    if False:
        print('Hello World!')
    for result in printListEnumGen(l):
        pass
dtoolSuperBase = None

def _getDtoolSuperBase():
    if False:
        i = 10
        return i + 15
    global dtoolSuperBase
    from panda3d.core import TypedObject
    dtoolSuperBase = TypedObject.__bases__[0]
    assert dtoolSuperBase.__name__ == 'DTOOL_SUPER_BASE'
safeReprNotify = None

def _getSafeReprNotify():
    if False:
        while True:
            i = 10
    global safeReprNotify
    from direct.directnotify.DirectNotifyGlobal import directNotify
    safeReprNotify = directNotify.newCategory('safeRepr')
    return safeReprNotify

def safeRepr(obj):
    if False:
        for i in range(10):
            print('nop')
    global dtoolSuperBase
    if dtoolSuperBase is None:
        _getDtoolSuperBase()
    global safeReprNotify
    if safeReprNotify is None:
        _getSafeReprNotify()
    if isinstance(obj, dtoolSuperBase):
        safeReprNotify.info('calling repr on instance of %s.%s' % (obj.__class__.__module__, obj.__class__.__name__))
        sys.stdout.flush()
    try:
        return repr(obj)
    except Exception:
        return '<** FAILED REPR OF %s instance at %s **>' % (obj.__class__.__name__, hex(id(obj)))

def safeReprTypeOnFail(obj):
    if False:
        return 10
    global dtoolSuperBase
    if dtoolSuperBase is None:
        _getDtoolSuperBase()
    global safeReprNotify
    if safeReprNotify is None:
        _getSafeReprNotify()
    if isinstance(obj, dtoolSuperBase):
        return type(obj)
    try:
        return repr(obj)
    except Exception:
        return '<** FAILED REPR OF %s instance at %s **>' % (obj.__class__.__name__, hex(id(obj)))

def fastRepr(obj, maxLen=200, strFactor=10, _visitedIds=None):
    if False:
        return 10
    ' caps the length of iterable types, so very large objects will print faster.\n    also prevents infinite recursion '
    try:
        if _visitedIds is None:
            _visitedIds = set()
        if id(obj) in _visitedIds:
            return '<ALREADY-VISITED %s>' % itype(obj)
        if type(obj) in (tuple, list):
            s = ''
            s += {tuple: '(', list: '['}[type(obj)]
            if maxLen is not None and len(obj) > maxLen:
                o = obj[:maxLen]
                ellips = '...'
            else:
                o = obj
                ellips = ''
            _visitedIds.add(id(obj))
            for item in o:
                s += fastRepr(item, maxLen, _visitedIds=_visitedIds)
                s += ', '
            _visitedIds.remove(id(obj))
            s += ellips
            s += {tuple: ')', list: ']'}[type(obj)]
            return s
        elif type(obj) is dict:
            s = '{'
            if maxLen is not None and len(obj) > maxLen:
                o = list(obj.keys())[:maxLen]
                ellips = '...'
            else:
                o = list(obj.keys())
                ellips = ''
            _visitedIds.add(id(obj))
            for key in o:
                value = obj[key]
                s += '%s: %s, ' % (fastRepr(key, maxLen, _visitedIds=_visitedIds), fastRepr(value, maxLen, _visitedIds=_visitedIds))
            _visitedIds.remove(id(obj))
            s += ellips
            s += '}'
            return s
        elif type(obj) is str:
            if maxLen is not None:
                maxLen *= strFactor
            if maxLen is not None and len(obj) > maxLen:
                return safeRepr(obj[:maxLen])
            else:
                return safeRepr(obj)
        else:
            r = safeRepr(obj)
            maxLen *= strFactor
            if len(r) > maxLen:
                r = r[:maxLen]
            return r
    except Exception:
        return '<** FAILED REPR OF %s **>' % obj.__class__.__name__

def convertTree(objTree, idList):
    if False:
        print('Hello World!')
    newTree = {}
    for key in list(objTree.keys()):
        obj = (idList[key],)
        newTree[obj] = {}
        r_convertTree(objTree[key], newTree[obj], idList)
    return newTree

def r_convertTree(oldTree, newTree, idList):
    if False:
        return 10
    for key in list(oldTree.keys()):
        obj = idList.get(key)
        if not obj:
            continue
        obj = str(obj)[:100]
        newTree[obj] = {}
        r_convertTree(oldTree[key], newTree[obj], idList)

def pretty_print(tree):
    if False:
        return 10
    for name in tree.keys():
        print(name)
        r_pretty_print(tree[name], 0)

def r_pretty_print(tree, num):
    if False:
        print('Hello World!')
    num += 1
    for name in tree.keys():
        print('  ' * num, name)
        r_pretty_print(tree[name], num)

def isDefaultValue(x):
    if False:
        while True:
            i = 10
    return x == type(x)()

def appendStr(obj, st):
    if False:
        print('Hello World!')
    'adds a string onto the __str__ output of an instance'

    def appendedStr(oldStr, st, self):
        if False:
            while True:
                i = 10
        return oldStr() + st
    oldStr = getattr(obj, '__str__', None)
    if oldStr is None:

        def stringer(s):
            if False:
                print('Hello World!')
            return s
        oldStr = Functor(stringer, str(obj))
        stringer = None
    obj.__str__ = types.MethodType(Functor(appendedStr, oldStr, st), obj)
    appendedStr = None
    return obj

class ScratchPad:
    """empty class to stick values onto"""

    def __init__(self, **kArgs):
        if False:
            while True:
                i = 10
        for (key, value) in kArgs.items():
            setattr(self, key, value)
        self._keys = set(kArgs.keys())

    def add(self, **kArgs):
        if False:
            i = 10
            return i + 15
        for (key, value) in kArgs.items():
            setattr(self, key, value)
        self._keys.update(list(kArgs.keys()))

    def destroy(self):
        if False:
            while True:
                i = 10
        for key in self._keys:
            delattr(self, key)

    def __getitem__(self, itemName):
        if False:
            i = 10
            return i + 15
        return getattr(self, itemName)

    def get(self, itemName, default=None):
        if False:
            print('Hello World!')
        return getattr(self, itemName, default)

    def __contains__(self, itemName):
        if False:
            return 10
        return itemName in self._keys

class Sync:
    _SeriesGen = SerialNumGen()

    def __init__(self, name, other=None):
        if False:
            while True:
                i = 10
        self._name = name
        if other is None:
            self._series = self._SeriesGen.next()
            self._value = 0
        else:
            self._series = other._series
            self._value = other._value

    def invalidate(self):
        if False:
            return 10
        self._value = None

    def change(self):
        if False:
            while True:
                i = 10
        self._value += 1

    def sync(self, other):
        if False:
            i = 10
            return i + 15
        if self._series != other._series or self._value != other._value:
            self._series = other._series
            self._value = other._value
            return True
        else:
            return False

    def isSynced(self, other):
        if False:
            while True:
                i = 10
        return self._series == other._series and self._value == other._value

    def __repr__(self):
        if False:
            print('Hello World!')
        return '%s(%s)<family=%s,value=%s>' % (self.__class__.__name__, self._name, self._series, self._value)

def itype(obj):
    if False:
        for i in range(10):
            print('nop')
    global dtoolSuperBase
    t = type(obj)
    if dtoolSuperBase is None:
        _getDtoolSuperBase()
    if isinstance(obj, dtoolSuperBase):
        return "<type 'instance' of %s>" % obj.__class__
    return t

def deeptype(obj, maxLen=100, _visitedIds=None):
    if False:
        while True:
            i = 10
    if _visitedIds is None:
        _visitedIds = set()
    if id(obj) in _visitedIds:
        return '<ALREADY-VISITED %s>' % itype(obj)
    t = type(obj)
    if t in (tuple, list):
        s = ''
        s += {tuple: '(', list: '['}[type(obj)]
        if maxLen is not None and len(obj) > maxLen:
            o = obj[:maxLen]
            ellips = '...'
        else:
            o = obj
            ellips = ''
        _visitedIds.add(id(obj))
        for item in o:
            s += deeptype(item, maxLen, _visitedIds=_visitedIds)
            s += ', '
        _visitedIds.remove(id(obj))
        s += ellips
        s += {tuple: ')', list: ']'}[type(obj)]
        return s
    elif type(obj) is dict:
        s = '{'
        if maxLen is not None and len(obj) > maxLen:
            o = list(obj.keys())[:maxLen]
            ellips = '...'
        else:
            o = list(obj.keys())
            ellips = ''
        _visitedIds.add(id(obj))
        for key in o:
            value = obj[key]
            s += '%s: %s, ' % (deeptype(key, maxLen, _visitedIds=_visitedIds), deeptype(value, maxLen, _visitedIds=_visitedIds))
        _visitedIds.remove(id(obj))
        s += ellips
        s += '}'
        return s
    else:
        return str(itype(obj))

def getNumberedTypedString(items, maxLen=5000, numPrefix=''):
    if False:
        return 10
    'get a string that has each item of the list on its own line,\n    and each item is numbered on the left from zero'
    digits = 0
    n = len(items)
    while n > 0:
        digits += 1
        n //= 10
    format = numPrefix + '%0' + '%s' % digits + 'i:%s \t%s'
    first = True
    s = ''
    snip = '<SNIP>'
    for i in range(len(items)):
        if not first:
            s += '\n'
        first = False
        objStr = fastRepr(items[i])
        if len(objStr) > maxLen:
            objStr = '%s%s' % (objStr[:maxLen - len(snip)], snip)
        s += format % (i, itype(items[i]), objStr)
    return s

def getNumberedTypedSortedString(items, maxLen=5000, numPrefix=''):
    if False:
        return 10
    'get a string that has each item of the list on its own line,\n    the items are stringwise-sorted, and each item is numbered on\n    the left from zero'
    digits = 0
    n = len(items)
    while n > 0:
        digits += 1
        n //= 10
    format = numPrefix + '%0' + '%s' % digits + 'i:%s \t%s'
    snip = '<SNIP>'
    strs = []
    for item in items:
        objStr = fastRepr(item)
        if len(objStr) > maxLen:
            objStr = '%s%s' % (objStr[:maxLen - len(snip)], snip)
        strs.append(objStr)
    first = True
    s = ''
    strs.sort()
    for i in range(len(strs)):
        if not first:
            s += '\n'
        first = False
        objStr = strs[i]
        s += format % (i, itype(items[i]), strs[i])
    return s

def printNumberedTyped(items, maxLen=5000):
    if False:
        for i in range(10):
            print('nop')
    'print out each item of the list on its own line,\n    with each item numbered on the left from zero'
    digits = 0
    n = len(items)
    while n > 0:
        digits += 1
        n //= 10
    format = '%0' + '%s' % digits + 'i:%s \t%s'
    for i in range(len(items)):
        objStr = fastRepr(items[i])
        if len(objStr) > maxLen:
            snip = '<SNIP>'
            objStr = '%s%s' % (objStr[:maxLen - len(snip)], snip)
        print(format % (i, itype(items[i]), objStr))

def printNumberedTypesGen(items, maxLen=5000):
    if False:
        return 10
    digits = 0
    n = len(items)
    while n > 0:
        digits += 1
        n //= 10
    format = '%0' + '%s' % digits + 'i:%s'
    for i in range(len(items)):
        print(format % (i, itype(items[i])))
        yield None

def printNumberedTypes(items, maxLen=5000):
    if False:
        while True:
            i = 10
    'print out the type of each item of the list on its own line,\n    with each item numbered on the left from zero'
    for result in printNumberedTypesGen(items, maxLen):
        yield result

class DelayedCall:
    """ calls a func after a specified delay """

    def __init__(self, func, name=None, delay=None):
        if False:
            i = 10
            return i + 15
        if name is None:
            name = 'anonymous'
        if delay is None:
            delay = 0.01
        self._func = func
        self._taskName = 'DelayedCallback-%s' % name
        self._delay = delay
        self._finished = False
        self._addDoLater()

    def destroy(self):
        if False:
            i = 10
            return i + 15
        self._finished = True
        self._removeDoLater()

    def finish(self):
        if False:
            while True:
                i = 10
        if not self._finished:
            self._doCallback(None)
        self.destroy()

    def _addDoLater(self):
        if False:
            print('Hello World!')
        taskMgr.doMethodLater(self._delay, self._doCallback, self._taskName)

    def _removeDoLater(self):
        if False:
            print('Hello World!')
        taskMgr.remove(self._taskName)

    def _doCallback(self, task):
        if False:
            for i in range(10):
                print('nop')
        self._finished = True
        func = self._func
        del self._func
        func()

class FrameDelayedCall:
    """ calls a func after N frames """

    def __init__(self, name, callback, frames=None, cancelFunc=None):
        if False:
            i = 10
            return i + 15
        if frames is None:
            frames = 1
        self._name = name
        self._frames = frames
        self._callback = callback
        self._cancelFunc = cancelFunc
        self._taskName = uniqueName('%s-%s' % (self.__class__.__name__, self._name))
        self._finished = False
        self._startTask()

    def destroy(self):
        if False:
            while True:
                i = 10
        self._finished = True
        self._stopTask()

    def finish(self):
        if False:
            i = 10
            return i + 15
        if not self._finished:
            self._finished = True
            self._callback()
        self.destroy()

    def _startTask(self):
        if False:
            i = 10
            return i + 15
        taskMgr.add(self._frameTask, self._taskName)
        self._counter = 0

    def _stopTask(self):
        if False:
            print('Hello World!')
        taskMgr.remove(self._taskName)

    def _frameTask(self, task):
        if False:
            print('Hello World!')
        if self._cancelFunc and self._cancelFunc():
            self.destroy()
            return task.done
        self._counter += 1
        if self._counter >= self._frames:
            self.finish()
            return task.done
        return task.cont

class DelayedFunctor:
    """ Waits for this object to be called, then calls supplied functor after a delay.
    Effectively inserts a time delay between the caller and the functor. """

    def __init__(self, functor, name=None, delay=None):
        if False:
            for i in range(10):
                print('nop')
        self._functor = functor
        self._name = name
        self.__name__ = self._name
        self._delay = delay

    def _callFunctor(self):
        if False:
            i = 10
            return i + 15
        cb = Functor(self._functor, *self._args, **self._kwArgs)
        del self._functor
        del self._name
        del self._delay
        del self._args
        del self._kwArgs
        del self._delayedCall
        del self.__name__
        cb()

    def __call__(self, *args, **kwArgs):
        if False:
            return 10
        self._args = args
        self._kwArgs = kwArgs
        self._delayedCall = DelayedCall(self._callFunctor, self._name, self._delay)

class SubframeCall:
    """Calls a callback at a specific time during the frame using the
    task system"""

    def __init__(self, functor, taskPriority, name=None):
        if False:
            print('Hello World!')
        self._functor = functor
        self._name = name
        self._taskName = uniqueName('SubframeCall-%s' % self._name)
        taskMgr.add(self._doCallback, self._taskName, priority=taskPriority)

    def _doCallback(self, task):
        if False:
            i = 10
            return i + 15
        functor = self._functor
        del self._functor
        functor()
        del self._name
        self._taskName = None
        return task.done

    def cleanup(self):
        if False:
            print('Hello World!')
        if self._taskName:
            taskMgr.remove(self._taskName)
            self._taskName = None

class PStatScope:
    collectors: dict = {}

    def __init__(self, level=None):
        if False:
            for i in range(10):
                print('nop')
        self.levels = []
        if level:
            self.levels.append(level)

    def copy(self, push=None):
        if False:
            return 10
        c = PStatScope()
        c.levels = self.levels[:]
        if push:
            c.push(push)
        return c

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return "PStatScope - '%s'" % (self,)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return ':'.join(self.levels)

    def push(self, level):
        if False:
            for i in range(10):
                print('nop')
        self.levels.append(level.replace('_', ''))

    def pop(self):
        if False:
            print('Hello World!')
        return self.levels.pop()

    def start(self, push=None):
        if False:
            while True:
                i = 10
        if push:
            self.push(push)
        self.getCollector().start()

    def stop(self, pop=False):
        if False:
            print('Hello World!')
        self.getCollector().stop()
        if pop:
            self.pop()

    def getCollector(self):
        if False:
            while True:
                i = 10
        label = str(self)
        if label not in self.collectors:
            from panda3d.core import PStatCollector
            self.collectors[label] = PStatCollector(label)
        return self.collectors[label]

def pstatcollect(scope, level=None):
    if False:
        return 10

    def decorator(f):
        if False:
            i = 10
            return i + 15
        return f
    try:
        if not (__dev__ or ConfigVariableBool('force-pstatcollect', False)) or not scope:
            return decorator

        def decorator(f):
            if False:
                return 10

            def wrap(*args, **kw):
                if False:
                    for i in range(10):
                        print('nop')
                scope.start(push=level or f.__name__)
                val = f(*args, **kw)
                scope.stop(pop=True)
                return val
            return wrap
    except Exception:
        pass
    return decorator
__report_indent = 0

def report(types=[], prefix='', xform=None, notifyFunc=None, dConfigParam=[]):
    if False:
        return 10
    "\n    This is a decorator generating function.  Use is similar to\n    a @decorator, except you must be sure to call it as a function.\n    It actually returns the decorator which is then used to transform\n    your decorated function. Confusing at first, I know.\n\n    Decoration occurs at function definition time.\n\n    If __dev__ is not defined, or resolves to False, this function\n    has no effect and no wrapping/transform occurs.  So in production,\n    it's as if the report has been asserted out.\n\n    Parameters:\n        types: A subset list of ['timeStamp', 'frameCount', 'avLocation']\n            This allows you to specify certain useful bits of info:\n\n              - *module*: Prints the module that this report statement\n                can be found in.\n              - *args*: Prints the arguments as they were passed to this\n                function.\n              - *timeStamp*: Adds the current frame time to the output.\n              - *deltaStamp*: Adds the current AI synched frame time to\n                the output\n              - *frameCount*: Adds the current frame count to the output.\n                Usually cleaner than the timeStamp output.\n              - *avLocation*: Adds the localAvatar's network location to\n                the output.  Useful for interest debugging.\n              - *interests*: Prints the current interest state after the\n                report.\n              - *stackTrace*: Prints a stack trace after the report.\n\n        prefix: Optional string to prepend to output, just before the\n            function.  Allows for easy grepping and is useful when\n            merging AI/Client reports into a single file.\n\n        xform:  Optional callback that accepts a single parameter:\n            argument 0 to the decorated function. (assumed to be 'self')\n            It should return a value to be inserted into the report\n            output string.\n\n        notifyFunc: A notify function such as info, debug, warning, etc.\n            By default the report will be printed to stdout. This will\n            allow you send the report to a designated 'notify' output.\n\n        dConfigParam: A list of Config.prc string variables.\n            By default the report will always print.  If you specify\n            this param, it will only print if one of the specified\n            config strings resolve to True.\n    "

    def indent(str):
        if False:
            return 10
        global __report_indent
        return ' ' * __report_indent + str

    def decorator(f):
        if False:
            return 10
        return f
    try:
        if not __dev__ and (not ConfigVariableBool('force-reports', False)):
            return decorator
        dConfigParamList = []
        doPrint = False
        if not dConfigParam:
            doPrint = True
        else:
            if not isinstance(dConfigParam, (list, tuple)):
                dConfigParams = (dConfigParam,)
            else:
                dConfigParams = dConfigParam
            dConfigParamList = [param for param in dConfigParams if ConfigVariableBool('want-%s-report' % (param,), False)]
            doPrint = bool(dConfigParamList)
        if not doPrint:
            return decorator
        if prefix:
            prefixes = set([prefix])
        else:
            prefixes = set()
        for param in dConfigParamList:
            prefix = ConfigVariableString(f'prefix-{param}-report', '', 'DConfig', ConfigFlags.F_dconfig).value
            if prefix:
                prefixes.add(prefix)
    except NameError as e:
        return decorator
    globalClockDelta = importlib.import_module('direct.distributed.ClockDelta').globalClockDelta

    def decorator(f):
        if False:
            print('Hello World!')

        def wrap(*args, **kwargs):
            if False:
                print('Hello World!')
            if args:
                rArgs = [args[0].__class__.__name__ + ', ']
            else:
                rArgs = []
            if 'args' in types:
                rArgs += [repr(x) + ', ' for x in args[1:]] + [x + ' = ' + '%s, ' % repr(y) for (x, y) in kwargs.items()]
            if not rArgs:
                rArgs = '()'
            else:
                rArgs = '(' + functools.reduce(str.__add__, rArgs)[:-2] + ')'
            outStr = '%s%s' % (f.__name__, rArgs)
            if prefixes:
                outStr = '%%s %s' % (outStr,)
            globalClock = ClockObject.getGlobalClock()
            if 'module' in types:
                outStr = '%s {M:%s}' % (outStr, f.__module__.split('.')[-1])
            if 'frameCount' in types:
                outStr = '%-8d : %s' % (globalClock.getFrameCount(), outStr)
            if 'timeStamp' in types:
                outStr = '%-8.3f : %s' % (globalClock.getFrameTime(), outStr)
            if 'deltaStamp' in types:
                outStr = '%-8.2f : %s' % (globalClock.getRealTime() - globalClockDelta.delta, outStr)
            if 'avLocation' in types:
                outStr = '%s : %s' % (outStr, str(localAvatar.getLocation()))
            if xform:
                outStr = '%s : %s' % (outStr, xform(args[0]))
            if prefixes:
                for prefix in prefixes:
                    if notifyFunc:
                        notifyFunc(outStr % (prefix,))
                    else:
                        print(indent(outStr % (prefix,)))
            elif notifyFunc:
                notifyFunc(outStr)
            else:
                print(indent(outStr))
            if 'interests' in types:
                base.cr.printInterestSets()
            if 'stackTrace' in types:
                print(StackTrace())
            global __report_indent
            rVal = None
            try:
                __report_indent += 1
                rVal = f(*args, **kwargs)
            finally:
                __report_indent -= 1
                if rVal is not None:
                    print(indent(' -> ' + repr(rVal)))
            return rVal
        wrap.__name__ = f.__name__
        wrap.__dict__ = f.__dict__
        wrap.__doc__ = f.__doc__
        wrap.__module__ = f.__module__
        return wrap
    return decorator

def getBase():
    if False:
        for i in range(10):
            print('nop')
    try:
        return base
    except Exception:
        return simbase

def getRepository():
    if False:
        while True:
            i = 10
    try:
        return base.cr
    except Exception:
        return simbase.air
exceptionLoggedNotify = None
if __debug__:

    def exceptionLogged(append=True):
        if False:
            i = 10
            return i + 15
        "decorator that outputs the function name and all arguments\n        if an exception passes back through the stack frame\n        if append is true, string is appended to the __str__ output of\n        the exception. if append is false, string is printed to the log\n        directly. If the output will take up many lines, it's recommended\n        to set append to False so that the exception stack is not hidden\n        by the output of this decorator.\n        "
        try:
            null = not __dev__
        except Exception:
            null = not __debug__
        if null:

            def nullDecorator(f):
                if False:
                    print('Hello World!')
                return f
            return nullDecorator

        def _decoratorFunc(f, append=append):
            if False:
                while True:
                    i = 10
            global exceptionLoggedNotify
            if exceptionLoggedNotify is None:
                from direct.directnotify.DirectNotifyGlobal import directNotify
                exceptionLoggedNotify = directNotify.newCategory('ExceptionLogged')

            def _exceptionLogged(*args, **kArgs):
                if False:
                    for i in range(10):
                        print('nop')
                try:
                    return f(*args, **kArgs)
                except Exception as e:
                    try:
                        s = '%s(' % f.__name__
                        for arg in args:
                            s += '%s, ' % arg
                        for (key, value) in list(kArgs.items()):
                            s += '%s=%s, ' % (key, value)
                        if len(args) > 0 or len(kArgs) > 0:
                            s = s[:-2]
                        s += ')'
                        if append:
                            appendStr(e, '\n%s' % s)
                        else:
                            exceptionLoggedNotify.info(s)
                    except Exception:
                        exceptionLoggedNotify.info('%s: ERROR IN PRINTING' % f.__name__)
                    raise
            _exceptionLogged.__doc__ = f.__doc__
            return _exceptionLogged
        return _decoratorFunc
GoldenRatio = (1.0 + math.sqrt(5.0)) / 2.0

class GoldenRectangle:

    @staticmethod
    def getLongerEdge(shorter):
        if False:
            return 10
        return shorter * GoldenRatio

    @staticmethod
    def getShorterEdge(longer):
        if False:
            print('Hello World!')
        return longer / GoldenRatio

def nullGen():
    if False:
        return 10
    if False:
        yield None

def loopGen(l):
    if False:
        for i in range(10):
            print('nop')

    def _gen(l):
        if False:
            return 10
        while True:
            for item in l:
                yield item
    gen = _gen(l)
    _gen = None
    return gen

def makeFlywheelGen(objects, countList=None, countFunc=None, scale=None):
    if False:
        for i in range(10):
            print('nop')

    def flywheel(index2objectAndCount):
        if False:
            while True:
                i = 10
        while len(index2objectAndCount) > 0:
            keyList = list(index2objectAndCount.keys())
            for key in keyList:
                if index2objectAndCount[key][1] > 0:
                    yield index2objectAndCount[key][0]
                    index2objectAndCount[key][1] -= 1
                if index2objectAndCount[key][1] <= 0:
                    del index2objectAndCount[key]
    if countList is None:
        countList = []
        for object in objects:
            yield None
            countList.append(countFunc(object))
    if scale is not None:
        for i in range(len(countList)):
            yield None
            if countList[i] > 0:
                countList[i] = max(1, int(countList[i] * scale))
    index2objectAndCount = {}
    for i in range(len(countList)):
        yield None
        index2objectAndCount[i] = [objects[i], countList[i]]
    yield flywheel(index2objectAndCount)

def flywheel(*args, **kArgs):
    if False:
        while True:
            i = 10
    '\n    >>> for i in flywheel([1,2,3], countList=[10, 5, 1]):\n    ...   print i,\n    ...\n    1 2 3 1 2 1 2 1 2 1 2 1 1 1 1 1\n    '
    for flywheel in makeFlywheelGen(*args, **kArgs):
        pass
    return flywheel
if __debug__:

    def quickProfile(name='unnamed'):
        if False:
            for i in range(10):
                print('nop')
        import pstats

        def profileDecorator(f):
            if False:
                i = 10
                return i + 15
            if not ConfigVariableBool('use-profiler', False):
                return f

            def _profiled(*args, **kArgs):
                if False:
                    i = 10
                    return i + 15
                if not ConfigVariableBool('profile-debug', False):
                    clock = ClockObject.getGlobalClock()
                    st = clock.getRealTime()
                    f(*args, **kArgs)
                    s = clock.getRealTime() - st
                    print('Function %s.%s took %s seconds' % (f.__module__, f.__name__, s))
                else:
                    import profile as prof
                    if not hasattr(base, 'stats'):
                        base.stats = {}
                    if not base.stats.get(name):
                        base.stats[name] = []
                    prof.runctx('f(*args, **kArgs)', {'f': f, 'args': args, 'kArgs': kArgs}, None, 't.prof')
                    s = pstats.Stats('t.prof')
                    s.strip_dirs()
                    s.sort_stats('cumulative')
                    base.stats[name].append(s)
            _profiled.__doc__ = f.__doc__
            return _profiled
        return profileDecorator

def getTotalAnnounceTime():
    if False:
        i = 10
        return i + 15
    td = 0
    for objs in base.stats.values():
        for stat in objs:
            td += getAnnounceGenerateTime(stat)
    return td

def getAnnounceGenerateTime(stat):
    if False:
        print('Hello World!')
    val = 0
    stats = stat.stats
    for i in list(stats.keys()):
        if i[2] == 'announceGenerate':
            newVal = stats[i][3]
            if newVal > val:
                val = newVal
    return val

class MiniLog:

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        self.indent = 1
        self.name = name
        self.lines = []

    def __str__(self):
        if False:
            print('Hello World!')
        return '%s\nMiniLog: %s\n%s\n%s\n%s' % ('*' * 50, self.name, '-' * 50, '\n'.join(self.lines), '*' * 50)

    def enterFunction(self, funcName, *args, **kw):
        if False:
            return 10
        rArgs = [repr(x) + ', ' for x in args] + [x + ' = ' + '%s, ' % repr(y) for (x, y) in kw.items()]
        if not rArgs:
            rArgs = '()'
        else:
            rArgs = '(' + functools.reduce(str.__add__, rArgs)[:-2] + ')'
        line = '%s%s' % (funcName, rArgs)
        self.appendFunctionCall(line)
        self.indent += 1
        return line

    def exitFunction(self):
        if False:
            for i in range(10):
                print('nop')
        self.indent -= 1
        return self.indent

    def appendFunctionCall(self, line):
        if False:
            for i in range(10):
                print('nop')
        self.lines.append(' ' * (self.indent * 2) + line)
        return line

    def appendLine(self, line):
        if False:
            i = 10
            return i + 15
        self.lines.append(' ' * (self.indent * 2) + '<< ' + line + ' >>')
        return line

    def flush(self):
        if False:
            while True:
                i = 10
        outStr = str(self)
        self.indent = 0
        self.lines = []
        return outStr

class MiniLogSentry:

    def __init__(self, log, funcName, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        self.log = log
        if self.log:
            self.log.enterFunction(funcName, *args, **kw)

    def __del__(self):
        if False:
            return 10
        if self.log:
            self.log.exitFunction()
        del self.log

def logBlock(id, msg):
    if False:
        for i in range(10):
            print('nop')
    print('<< LOGBLOCK(%03d)' % id)
    print(str(msg))
    print('/LOGBLOCK(%03d) >>' % id)

class HierarchyException(Exception):
    JOSWILSO = 0

    def __init__(self, owner, description):
        if False:
            while True:
                i = 10
        self.owner = owner
        self.desc = description

    def __str__(self):
        if False:
            while True:
                i = 10
        return '(%s): %s' % (self.owner, self.desc)

    def __repr__(self):
        if False:
            return 10
        return 'HierarchyException(%s)' % (self.owner,)

def formatTimeCompact(seconds):
    if False:
        return 10
    result = ''
    a = int(seconds)
    seconds = a % 60
    a //= 60
    if a > 0:
        minutes = a % 60
        a //= 60
        if a > 0:
            hours = a % 24
            a //= 24
            if a > 0:
                days = a
                result += '%sd' % days
            result += '%sh' % hours
        result += '%sm' % minutes
    result += '%ss' % seconds
    return result

def formatTimeExact(seconds):
    if False:
        return 10
    result = ''
    a = int(seconds)
    seconds = a % 60
    a //= 60
    if a > 0:
        minutes = a % 60
        a //= 60
        if a > 0:
            hours = a % 24
            a //= 24
            if a > 0:
                days = a
                result += '%sd' % days
            if hours or minutes or seconds:
                result += '%sh' % hours
        if minutes or seconds:
            result += '%sm' % minutes
    if seconds or result == '':
        result += '%ss' % seconds
    return result

class AlphabetCounter:

    def __init__(self):
        if False:
            return 10
        self._curCounter = ['A']

    def next(self):
        if False:
            print('Hello World!')
        result = ''.join([c for c in self._curCounter])
        index = -1
        while True:
            curChar = self._curCounter[index]
            if curChar == 'Z':
                nextChar = 'A'
                carry = True
            else:
                nextChar = chr(ord(self._curCounter[index]) + 1)
                carry = False
            self._curCounter[index] = nextChar
            if carry:
                if -index == len(self._curCounter):
                    self._curCounter = ['A'] + self._curCounter
                    break
                else:
                    index -= 1
                carry = False
            else:
                break
        return result
    __next__ = next

class Default:
    pass

def configIsToday(configName):
    if False:
        i = 10
        return i + 15
    today = time.localtime()
    confStr = ConfigVariableString(configName, '', 'DConfig', ConfigFlags.F_dconfig).value
    for format in ('%m/%d/%Y', '%m-%d-%Y', '%m.%d.%Y'):
        try:
            confDate = time.strptime(confStr, format)
        except ValueError:
            pass
        else:
            if confDate.tm_year == today.tm_year and confDate.tm_mon == today.tm_mon and (confDate.tm_mday == today.tm_mday):
                return True
    return False

def typeName(o):
    if False:
        while True:
            i = 10
    if hasattr(o, '__class__'):
        return o.__class__.__name__
    else:
        return o.__name__

def safeTypeName(o):
    if False:
        for i in range(10):
            print('nop')
    try:
        return typeName(o)
    except Exception:
        pass
    try:
        return type(o)
    except Exception:
        pass
    return '<failed safeTypeName()>'

def histogramDict(l):
    if False:
        while True:
            i = 10
    d = {}
    for e in l:
        d.setdefault(e, 0)
        d[e] += 1
    return d

def unescapeHtmlString(s):
    if False:
        return 10
    result = ''
    i = 0
    while i < len(s):
        char = s[i]
        if char == '+':
            char = ' '
        elif char == '%':
            if i < len(s) - 2:
                num = int(s[i + 1:i + 3], 16)
                char = chr(num)
                i += 2
        i += 1
        result += char
    return result

class PriorityCallbacks:
    """ manage a set of prioritized callbacks, and allow them to be invoked in order of priority """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._callbacks = []

    def clear(self):
        if False:
            return 10
        del self._callbacks[:]

    def add(self, callback, priority=None):
        if False:
            for i in range(10):
                print('nop')
        if priority is None:
            priority = 0
        callbacks = self._callbacks
        lo = 0
        hi = len(callbacks)
        while lo < hi:
            mid = (lo + hi) // 2
            if priority < callbacks[mid][0]:
                hi = mid
            else:
                lo = mid + 1
        item = (priority, callback)
        callbacks.insert(lo, item)
        return item

    def remove(self, item):
        if False:
            print('Hello World!')
        self._callbacks.remove(item)

    def __call__(self):
        if False:
            print('Hello World!')
        for (priority, callback) in self._callbacks:
            callback()
builtins.Functor = Functor
builtins.Stack = Stack
builtins.Queue = Queue
builtins.SerialNumGen = SerialNumGen
builtins.SerialMaskedGen = SerialMaskedGen
builtins.ScratchPad = ScratchPad
builtins.uniqueName = uniqueName
builtins.serialNum = serialNum
if __debug__:
    builtins.profiled = profiled
    builtins.exceptionLogged = exceptionLogged
builtins.itype = itype
builtins.appendStr = appendStr
builtins.bound = bound
builtins.clamp = clamp
builtins.lerp = lerp
builtins.makeList = makeList
builtins.makeTuple = makeTuple
if __debug__:
    builtins.printStack = printStack
    builtins.printReverseStack = printReverseStack
    builtins.printVerboseStack = printVerboseStack
builtins.DelayedCall = DelayedCall
builtins.DelayedFunctor = DelayedFunctor
builtins.FrameDelayedCall = FrameDelayedCall
builtins.SubframeCall = SubframeCall
builtins.invertDict = invertDict
builtins.invertDictLossless = invertDictLossless
builtins.getBase = getBase
builtins.getRepository = getRepository
builtins.safeRepr = safeRepr
builtins.fastRepr = fastRepr
builtins.nullGen = nullGen
builtins.flywheel = flywheel
builtins.loopGen = loopGen
if __debug__:
    builtins.StackTrace = StackTrace
builtins.report = report
builtins.pstatcollect = pstatcollect
builtins.MiniLog = MiniLog
builtins.MiniLogSentry = MiniLogSentry
builtins.logBlock = logBlock
builtins.HierarchyException = HierarchyException
builtins.deeptype = deeptype
builtins.Default = Default
builtins.configIsToday = configIsToday
builtins.typeName = typeName
builtins.safeTypeName = safeTypeName
builtins.histogramDict = histogramDict