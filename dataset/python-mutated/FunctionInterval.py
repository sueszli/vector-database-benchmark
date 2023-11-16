"""FunctionInterval module: contains the FunctionInterval class"""
from __future__ import annotations
__all__ = ['FunctionInterval', 'EventInterval', 'AcceptInterval', 'IgnoreInterval', 'ParentInterval', 'WrtParentInterval', 'PosInterval', 'HprInterval', 'ScaleInterval', 'PosHprInterval', 'HprScaleInterval', 'PosHprScaleInterval', 'Func', 'Wait']
from panda3d.direct import WaitInterval
from direct.showbase.MessengerGlobal import messenger
from direct.directnotify.DirectNotifyGlobal import directNotify
from . import Interval

class FunctionInterval(Interval.Interval):
    functionIntervalNum = 1
    if __debug__:
        import weakref
        FunctionIntervals: weakref.WeakKeyDictionary[FunctionInterval, int] = weakref.WeakKeyDictionary()

        @classmethod
        def replaceMethod(cls, oldFunction, newFunction):
            if False:
                print('Hello World!')
            import types
            count = 0
            for ival in cls.FunctionIntervals:
                if isinstance(ival.function, types.MethodType):
                    if ival.function.__func__ == oldFunction:
                        ival.function = types.MethodType(newFunction, ival.function.__self__)
                        count += 1
            return count
    notify = directNotify.newCategory('FunctionInterval')

    def __init__(self, function, **kw):
        if False:
            for i in range(10):
                print('nop')
        '__init__(function, name = None, openEnded = 1, extraArgs = [])\n        '
        name = kw.pop('name', None)
        openEnded = kw.pop('openEnded', 1)
        extraArgs = kw.pop('extraArgs', [])
        self.function = function
        if name is None:
            name = self.makeUniqueName(function)
        assert isinstance(name, str)
        self.extraArgs = extraArgs
        self.kw = kw
        Interval.Interval.__init__(self, name, duration=0.0, openEnded=openEnded)
        if __debug__:
            self.FunctionIntervals[self] = 1

    @staticmethod
    def makeUniqueName(func, suffix=''):
        if False:
            print('Hello World!')
        func_name = getattr(func, '__name__', None)
        if func_name is None:
            func_name = str(func)
        name = 'Func-%s-%d' % (func_name, FunctionInterval.functionIntervalNum)
        FunctionInterval.functionIntervalNum += 1
        if suffix:
            name = '%s-%s' % (name, str(suffix))
        return name

    def privInstant(self):
        if False:
            return 10
        self.function(*self.extraArgs, **self.kw)
        self.notify.debug('updateFunc() - %s: executing Function' % self.name)

class EventInterval(FunctionInterval):

    def __init__(self, event, sentArgs=[]):
        if False:
            while True:
                i = 10
        '__init__(event, sentArgs)\n        '

        def sendFunc(event=event, sentArgs=sentArgs):
            if False:
                while True:
                    i = 10
            messenger.send(event, sentArgs)
        FunctionInterval.__init__(self, sendFunc, name=event)

class AcceptInterval(FunctionInterval):

    def __init__(self, dirObj, event, function, name=None):
        if False:
            i = 10
            return i + 15
        '__init__(dirObj, event, function, name)\n        '

        def acceptFunc(dirObj=dirObj, event=event, function=function):
            if False:
                i = 10
                return i + 15
            dirObj.accept(event, function)
        if name is None:
            name = 'Accept-' + event
        FunctionInterval.__init__(self, acceptFunc, name=name)

class IgnoreInterval(FunctionInterval):

    def __init__(self, dirObj, event, name=None):
        if False:
            print('Hello World!')
        '__init__(dirObj, event, name)\n        '

        def ignoreFunc(dirObj=dirObj, event=event):
            if False:
                while True:
                    i = 10
            dirObj.ignore(event)
        if name is None:
            name = 'Ignore-' + event
        FunctionInterval.__init__(self, ignoreFunc, name=name)

class ParentInterval(FunctionInterval):
    parentIntervalNum = 1

    def __init__(self, nodePath, parent, name=None):
        if False:
            for i in range(10):
                print('nop')
        '__init__(nodePath, parent, name)\n        '

        def reparentFunc(nodePath=nodePath, parent=parent):
            if False:
                print('Hello World!')
            nodePath.reparentTo(parent)
        if name is None:
            name = 'ParentInterval-%d' % ParentInterval.parentIntervalNum
            ParentInterval.parentIntervalNum += 1
        FunctionInterval.__init__(self, reparentFunc, name=name)

class WrtParentInterval(FunctionInterval):
    wrtParentIntervalNum = 1

    def __init__(self, nodePath, parent, name=None):
        if False:
            return 10
        '__init__(nodePath, parent, name)\n        '

        def wrtReparentFunc(nodePath=nodePath, parent=parent):
            if False:
                i = 10
                return i + 15
            nodePath.wrtReparentTo(parent)
        if name is None:
            name = 'WrtParentInterval-%d' % WrtParentInterval.wrtParentIntervalNum
            WrtParentInterval.wrtParentIntervalNum += 1
        FunctionInterval.__init__(self, wrtReparentFunc, name=name)

class PosInterval(FunctionInterval):
    posIntervalNum = 1

    def __init__(self, nodePath, pos, duration=0.0, name=None, other=None):
        if False:
            while True:
                i = 10
        '__init__(nodePath, pos, duration, name)\n        '

        def posFunc(np=nodePath, pos=pos, other=other):
            if False:
                i = 10
                return i + 15
            if other:
                np.setPos(other, pos)
            else:
                np.setPos(pos)
        if name is None:
            name = 'PosInterval-%d' % PosInterval.posIntervalNum
            PosInterval.posIntervalNum += 1
        FunctionInterval.__init__(self, posFunc, name=name)

class HprInterval(FunctionInterval):
    hprIntervalNum = 1

    def __init__(self, nodePath, hpr, duration=0.0, name=None, other=None):
        if False:
            return 10
        '__init__(nodePath, hpr, duration, name)\n        '

        def hprFunc(np=nodePath, hpr=hpr, other=other):
            if False:
                while True:
                    i = 10
            if other:
                np.setHpr(other, hpr)
            else:
                np.setHpr(hpr)
        if name is None:
            name = 'HprInterval-%d' % HprInterval.hprIntervalNum
            HprInterval.hprIntervalNum += 1
        FunctionInterval.__init__(self, hprFunc, name=name)

class ScaleInterval(FunctionInterval):
    scaleIntervalNum = 1

    def __init__(self, nodePath, scale, duration=0.0, name=None, other=None):
        if False:
            i = 10
            return i + 15
        '__init__(nodePath, scale, duration, name)\n        '

        def scaleFunc(np=nodePath, scale=scale, other=other):
            if False:
                i = 10
                return i + 15
            if other:
                np.setScale(other, scale)
            else:
                np.setScale(scale)
        if name is None:
            name = 'ScaleInterval-%d' % ScaleInterval.scaleIntervalNum
            ScaleInterval.scaleIntervalNum += 1
        FunctionInterval.__init__(self, scaleFunc, name=name)

class PosHprInterval(FunctionInterval):
    posHprIntervalNum = 1

    def __init__(self, nodePath, pos, hpr, duration=0.0, name=None, other=None):
        if False:
            print('Hello World!')
        '__init__(nodePath, pos, hpr, duration, name)\n        '

        def posHprFunc(np=nodePath, pos=pos, hpr=hpr, other=other):
            if False:
                i = 10
                return i + 15
            if other:
                np.setPosHpr(other, pos, hpr)
            else:
                np.setPosHpr(pos, hpr)
        if name is None:
            name = 'PosHprInterval-%d' % PosHprInterval.posHprIntervalNum
            PosHprInterval.posHprIntervalNum += 1
        FunctionInterval.__init__(self, posHprFunc, name=name)

class HprScaleInterval(FunctionInterval):
    hprScaleIntervalNum = 1

    def __init__(self, nodePath, hpr, scale, duration=0.0, name=None, other=None):
        if False:
            print('Hello World!')
        '__init__(nodePath, hpr, scale, duration, other, name)\n        '

        def hprScaleFunc(np=nodePath, hpr=hpr, scale=scale, other=other):
            if False:
                while True:
                    i = 10
            if other:
                np.setHprScale(other, hpr, scale)
            else:
                np.setHprScale(hpr, scale)
        if name is None:
            name = 'HprScale-%d' % HprScaleInterval.hprScaleIntervalNum
            HprScaleInterval.hprScaleIntervalNum += 1
        FunctionInterval.__init__(self, hprScaleFunc, name=name)

class PosHprScaleInterval(FunctionInterval):
    posHprScaleIntervalNum = 1

    def __init__(self, nodePath, pos, hpr, scale, duration=0.0, name=None, other=None):
        if False:
            return 10
        '__init__(nodePath, pos, hpr, scale, duration, other, name)\n        '

        def posHprScaleFunc(np=nodePath, pos=pos, hpr=hpr, scale=scale, other=other):
            if False:
                while True:
                    i = 10
            if other:
                np.setPosHprScale(other, pos, hpr, scale)
            else:
                np.setPosHprScale(pos, hpr, scale)
        if name is None:
            name = 'PosHprScale-%d' % PosHprScaleInterval.posHprScaleIntervalNum
            PosHprScaleInterval.posHprScaleIntervalNum += 1
        FunctionInterval.__init__(self, posHprScaleFunc, name=name)

class Func(FunctionInterval):

    def __init__(self, *args, **kw):
        if False:
            print('Hello World!')
        function = args[0]
        assert hasattr(function, '__call__')
        extraArgs = args[1:]
        kw['extraArgs'] = extraArgs
        FunctionInterval.__init__(self, function, **kw)

class Wait(WaitInterval):

    def __init__(self, duration):
        if False:
            print('Hello World!')
        WaitInterval.__init__(self, duration)