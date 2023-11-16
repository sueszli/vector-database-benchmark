"""Contains the TaskThreaded and TaskThread classes."""
__all__ = ['TaskThreaded', 'TaskThread']
from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.task import Task
from direct.task.TaskManagerGlobal import taskMgr
from panda3d.core import ClockObject
from .PythonUtil import SerialNumGen, Functor

class TaskThreaded:
    """ derive from this if you need to do a bunch of CPU-intensive
    processing and you don't want to hang up the show. Lets you break
    up the processing over multiple frames """
    notify = directNotify.newCategory('TaskThreaded')
    _Serial = SerialNumGen()

    def __init__(self, name, threaded=True, timeslice=None, callback=None):
        if False:
            return 10
        self.__name = name
        self.__threaded = threaded
        if timeslice is None:
            timeslice = 0.01
        self.__timeslice = timeslice
        self.__taskNames = set()
        self._taskStartTime = None
        self.__threads = set()
        self._callback = callback

    def finished(self):
        if False:
            while True:
                i = 10
        if self._callback:
            self._callback()

    def destroy(self):
        if False:
            i = 10
            return i + 15
        for taskName in self.__taskNames:
            taskMgr.remove(taskName)
        del self.__taskNames
        for thread in self.__threads:
            thread.tearDown()
            thread._destroy()
        del self.__threads
        del self._callback
        self.ignoreAll()

    def getTimeslice(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__timeslice

    def setTimeslice(self, timeslice):
        if False:
            return 10
        self.__timeslice = timeslice

    def scheduleCallback(self, callback):
        if False:
            print('Hello World!')
        assert self.notify.debugCall()
        if not self.__threaded:
            callback()
        else:
            taskName = '%s-ThreadedTask-%s' % (self.__name, TaskThreaded._Serial.next())
            assert taskName not in self.__taskNames
            self.__taskNames.add(taskName)
            taskMgr.add(Functor(self.__doCallback, callback, taskName), taskName)

    def scheduleThread(self, thread):
        if False:
            i = 10
            return i + 15
        assert self.notify.debugCall()
        thread._init(self)
        thread.setUp()
        if thread.isFinished():
            thread._destroy()
        elif not self.__threaded:
            while not thread.isFinished():
                thread.run()
            thread._destroy()
        else:
            assert not thread in self.__threads
            self.__threads.add(thread)
            taskName = '%s-ThreadedTask-%s-%s' % (self.__name, thread.__class__.__name__, TaskThreaded._Serial.next())
            assert taskName not in self.__taskNames
            self.__taskNames.add(taskName)
            self.__threads.add(thread)
            taskMgr.add(Functor(self._doThreadCallback, thread, taskName), taskName)

    def _doCallback(self, callback, taskName, task):
        if False:
            i = 10
            return i + 15
        assert self.notify.debugCall()
        self.__taskNames.remove(taskName)
        self._taskStartTime = ClockObject.getGlobalClock().getRealTime()
        callback()
        self._taskStartTime = None
        return Task.done

    def _doThreadCallback(self, thread, taskName, task):
        if False:
            return 10
        assert self.notify.debugCall()
        self._taskStartTime = ClockObject.getGlobalClock().getRealTime()
        thread.run()
        self._taskStartTime = None
        if thread.isFinished():
            thread._destroy()
            self.__taskNames.remove(taskName)
            self.__threads.remove(thread)
            return Task.done
        else:
            return Task.cont

    def taskTimeLeft(self):
        if False:
            i = 10
            return i + 15
        'returns True if there is time left for the current task callback\n        to run without going over the allotted timeslice'
        if self._taskStartTime is None:
            return True
        return ClockObject.getGlobalClock().getRealTime() - self._taskStartTime < self.__timeslice

class TaskThread:

    def setUp(self):
        if False:
            return 10
        pass

    def run(self):
        if False:
            return 10
        pass

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        pass

    def done(self):
        if False:
            while True:
                i = 10
        pass

    def finished(self):
        if False:
            return 10
        self.tearDown()
        self._finished = True
        self.done()

    def isFinished(self):
        if False:
            print('Hello World!')
        return self._finished

    def timeLeft(self):
        if False:
            for i in range(10):
                print('nop')
        return self.parent.taskTimeLeft()

    def _init(self, parent):
        if False:
            print('Hello World!')
        self.parent = parent
        self._finished = False

    def _destroy(self):
        if False:
            while True:
                i = 10
        del self.parent
        del self._finished