"""Contains the Timer class."""
__all__ = ['Timer']
from panda3d.core import ClockObject
from . import Task
from .TaskManagerGlobal import taskMgr

class Timer:
    id = 0

    def __init__(self, name=None):
        if False:
            while True:
                i = 10
        self.finalT = 0.0
        self.currT = 0.0
        if name is None:
            name = 'default-timer-%d' % Timer.id
            Timer.id += 1
        self.name = name
        self.started = 0
        self.callback = None

    def start(self, t, name):
        if False:
            while True:
                i = 10
        if self.started:
            self.stop()
        self.callback = None
        self.finalT = t
        self.name = name
        self.startT = ClockObject.getGlobalClock().getFrameTime()
        self.currT = 0.0
        taskMgr.add(self.__timerTask, self.name + '-run')
        self.started = 1

    def startCallback(self, t, callback):
        if False:
            i = 10
            return i + 15
        if self.started:
            self.stop()
        self.callback = callback
        self.finalT = t
        self.startT = ClockObject.getGlobalClock().getFrameTime()
        self.currT = 0.0
        taskMgr.add(self.__timerTask, self.name + '-run')
        self.started = 1

    def stop(self):
        if False:
            print('Hello World!')
        if not self.started:
            return 0.0
        taskMgr.remove(self.name + '-run')
        self.started = 0
        return self.currT

    def resume(self):
        if False:
            return 10
        assert self.currT <= self.finalT
        assert self.started == 0
        self.start(self.finalT - self.currT, self.name)

    def restart(self):
        if False:
            i = 10
            return i + 15
        if self.callback is not None:
            self.startCallback(self.finalT, self.callback)
        else:
            self.start(self.finalT, self.name)

    def isStarted(self):
        if False:
            for i in range(10):
                print('nop')
        return self.started

    def addT(self, t):
        if False:
            for i in range(10):
                print('nop')
        self.finalT = self.finalT + t

    def setT(self, t):
        if False:
            while True:
                i = 10
        self.finalT = t

    def getT(self):
        if False:
            print('Hello World!')
        return self.finalT - self.currT

    def __timerTask(self, task):
        if False:
            for i in range(10):
                print('nop')
        t = ClockObject.getGlobalClock().getFrameTime()
        te = t - self.startT
        self.currT = te
        if te >= self.finalT:
            if self.callback is not None:
                self.callback()
            else:
                from direct.showbase.MessengerGlobal import messenger
                messenger.send(self.name)
            return Task.done
        return Task.cont