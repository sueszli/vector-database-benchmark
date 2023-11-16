from panda3d.core import ConfigVariableInt, ConfigVariableDouble
from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.fsm.StatePush import FunctionCall
from direct.showbase.PythonUtil import Averager
from .TaskManagerGlobal import taskMgr

class TaskTracker:
    notify = directNotify.newCategory('TaskProfiler')
    MinSamples = None
    SpikeThreshold = None

    def __init__(self, namePrefix):
        if False:
            return 10
        self._namePrefix = namePrefix
        self._durationAverager = Averager('%s-durationAverager' % namePrefix)
        self._avgSession = None
        if TaskTracker.MinSamples is None:
            TaskTracker.MinSamples = ConfigVariableInt('profile-task-spike-min-samples', 30).value
            TaskTracker.SpikeThreshold = TaskProfiler.GetDefaultSpikeThreshold()

    def destroy(self):
        if False:
            return 10
        self.flush()
        del self._namePrefix
        del self._durationAverager

    def flush(self):
        if False:
            i = 10
            return i + 15
        self._durationAverager.reset()
        if self._avgSession:
            self._avgSession.release()
        self._avgSession = None

    def getNamePrefix(self, namePrefix):
        if False:
            print('Hello World!')
        return self._namePrefix

    def _checkSpike(self, session):
        if False:
            for i in range(10):
                print('nop')
        duration = session.getDuration()
        isSpike = False
        if self.getNumDurationSamples() > self.MinSamples:
            if duration > self.getAvgDuration() * self.SpikeThreshold:
                isSpike = True
                avgSession = self.getAvgSession()
                s = '\n%s task CPU spike profile (%s) %s\n' % ('=' * 30, self._namePrefix, '=' * 30)
                s += '|' * 80 + '\n'
                for sorts in (['cumulative'], ['time'], ['calls']):
                    s += '-- AVERAGE --\n%s-- SPIKE --\n%s' % (avgSession.getResults(sorts=sorts, totalTime=duration), session.getResults(sorts=sorts))
                self.notify.info(s)
        return isSpike

    def addProfileSession(self, session):
        if False:
            while True:
                i = 10
        duration = session.getDuration()
        if duration == 0.0:
            return
        isSpike = self._checkSpike(session)
        self._durationAverager.addValue(duration)
        storeAvg = True
        if self._avgSession is not None:
            avgDur = self.getAvgDuration()
            if abs(self._avgSession.getDuration() - avgDur) < abs(duration - avgDur):
                storeAvg = False
        if storeAvg:
            if self._avgSession:
                self._avgSession.release()
            self._avgSession = session.getReference()

    def getAvgDuration(self):
        if False:
            return 10
        return self._durationAverager.getAverage()

    def getNumDurationSamples(self):
        if False:
            print('Hello World!')
        return self._durationAverager.getCount()

    def getAvgSession(self):
        if False:
            while True:
                i = 10
        return self._avgSession

    def log(self):
        if False:
            while True:
                i = 10
        if self._avgSession:
            s = 'task CPU profile (%s):\n' % self._namePrefix
            s += '|' * 80 + '\n'
            for sorts in (['cumulative'], ['time'], ['calls']):
                s += self._avgSession.getResults(sorts=sorts)
            self.notify.info(s)
        else:
            self.notify.info('task CPU profile (%s): no data collected' % self._namePrefix)

class TaskProfiler:
    notify = directNotify.newCategory('TaskProfiler')

    def __init__(self):
        if False:
            return 10
        self._enableFC = FunctionCall(self._setEnabled, taskMgr.getProfileTasksSV())
        self._enableFC.pushCurrentState()
        self._namePrefix2tracker = {}
        self._task = None

    def destroy(self):
        if False:
            print('Hello World!')
        if taskMgr.getProfileTasks():
            self._setEnabled(False)
        self._enableFC.destroy()
        for tracker in self._namePrefix2tracker.values():
            tracker.destroy()
        del self._namePrefix2tracker
        del self._task

    @staticmethod
    def GetDefaultSpikeThreshold():
        if False:
            i = 10
            return i + 15
        return ConfigVariableDouble('profile-task-spike-threshold', 5.0).value

    @staticmethod
    def SetSpikeThreshold(spikeThreshold):
        if False:
            print('Hello World!')
        TaskTracker.SpikeThreshold = spikeThreshold

    @staticmethod
    def GetSpikeThreshold():
        if False:
            for i in range(10):
                print('nop')
        return TaskTracker.SpikeThreshold

    def logProfiles(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        if name:
            name = name.lower()
        for (namePrefix, tracker) in self._namePrefix2tracker.items():
            if name and name not in namePrefix.lower():
                continue
            tracker.log()

    def flush(self, name):
        if False:
            i = 10
            return i + 15
        if name:
            name = name.lower()
        for (namePrefix, tracker) in self._namePrefix2tracker.items():
            if name and name not in namePrefix.lower():
                continue
            tracker.flush()

    def _setEnabled(self, enabled):
        if False:
            while True:
                i = 10
        if enabled:
            self.notify.info('task profiler started')
            self._taskName = 'profile-tasks-%s' % id(self)
            taskMgr.add(self._doProfileTasks, self._taskName, priority=-200)
        else:
            taskMgr.remove(self._taskName)
            del self._taskName
            self.notify.info('task profiler stopped')

    def _doProfileTasks(self, task=None):
        if False:
            i = 10
            return i + 15
        if self._task is not None and taskMgr._hasProfiledDesignatedTask():
            session = taskMgr._getLastTaskProfileSession()
            if session.profileSucceeded():
                namePrefix = self._task.getNamePrefix()
                if namePrefix not in self._namePrefix2tracker:
                    self._namePrefix2tracker[namePrefix] = TaskTracker(namePrefix)
                tracker = self._namePrefix2tracker[namePrefix]
                tracker.addProfileSession(session)
        self._task = taskMgr._getRandomTask()
        taskMgr._setProfileTask(self._task)
        return task.cont