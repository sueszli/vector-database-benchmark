"""Defines the IntervalManager class as well as the global instance of
this class, ivalMgr."""
__all__ = ['IntervalManager', 'ivalMgr']
from panda3d.core import EventQueue
from panda3d.direct import CIntervalManager, Dtool_BorrowThisReference
from direct.showbase import EventManager
import fnmatch

class IntervalManager(CIntervalManager):

    def __init__(self, globalPtr=0):
        if False:
            for i in range(10):
                print('nop')
        if globalPtr:
            self.cObj = CIntervalManager.getGlobalPtr()
            Dtool_BorrowThisReference(self, self.cObj)
            self.dd = self
        else:
            CIntervalManager.__init__(self)
        self.eventQueue = EventQueue()
        self.MyEventmanager = EventManager.EventManager(self.eventQueue)
        self.setEventQueue(self.eventQueue)
        self.ivals = []
        self.removedIvals = {}

    def addInterval(self, interval):
        if False:
            i = 10
            return i + 15
        index = self.addCInterval(interval, 1)
        self.__storeInterval(interval, index)

    def removeInterval(self, interval):
        if False:
            while True:
                i = 10
        index = self.findCInterval(interval.getName())
        if index >= 0:
            self.removeCInterval(index)
            if index < len(self.ivals):
                self.ivals[index] = None
            return 1
        return 0

    def getInterval(self, name):
        if False:
            i = 10
            return i + 15
        index = self.findCInterval(name)
        if index >= 0:
            if index < len(self.ivals) and self.ivals[index]:
                return self.ivals[index]
            return self.getCInterval(index)
        return None

    def getIntervalsMatching(self, pattern):
        if False:
            i = 10
            return i + 15
        ivals = []
        count = 0
        maxIndex = self.getMaxIndex()
        for index in range(maxIndex):
            ival = self.getCInterval(index)
            if ival and fnmatch.fnmatchcase(ival.getName(), pattern):
                count += 1
                if index < len(self.ivals) and self.ivals[index]:
                    ivals.append(self.ivals[index])
                else:
                    ivals.append(ival)
        return ivals

    def finishIntervalsMatching(self, pattern):
        if False:
            for i in range(10):
                print('nop')
        ivals = self.getIntervalsMatching(pattern)
        for ival in ivals:
            ival.finish()
        return len(ivals)

    def pauseIntervalsMatching(self, pattern):
        if False:
            while True:
                i = 10
        ivals = self.getIntervalsMatching(pattern)
        for ival in ivals:
            ival.pause()
        return len(ivals)

    def step(self):
        if False:
            for i in range(10):
                print('nop')
        CIntervalManager.step(self)
        self.__doPythonCallbacks()

    def interrupt(self):
        if False:
            while True:
                i = 10
        CIntervalManager.interrupt(self)
        self.__doPythonCallbacks()

    def __doPythonCallbacks(self):
        if False:
            i = 10
            return i + 15
        index = self.getNextRemoval()
        while index >= 0:
            ival = self.ivals[index]
            self.ivals[index] = None
            ival.privPostEvent()
            index = self.getNextRemoval()
        index = self.getNextEvent()
        while index >= 0:
            self.ivals[index].privPostEvent()
            index = self.getNextEvent()
        self.MyEventmanager.doEvents()

    def __storeInterval(self, interval, index):
        if False:
            for i in range(10):
                print('nop')
        while index >= len(self.ivals):
            self.ivals.append(None)
        assert self.ivals[index] is None or self.ivals[index] == interval
        self.ivals[index] = interval
ivalMgr = IntervalManager(1)