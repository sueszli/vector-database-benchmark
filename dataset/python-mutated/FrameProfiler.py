from panda3d.core import ConfigVariableBool, ClockObject
from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.fsm.StatePush import FunctionCall
from direct.showbase.PythonUtil import formatTimeExact, normalDistrib, serialNum
from direct.showbase.PythonUtil import Functor
from .Task import Task
from .TaskManagerGlobal import taskMgr

class FrameProfiler:
    notify = directNotify.newCategory('FrameProfiler')
    Minute = 60
    Hour = 60 * Minute
    Day = 24 * Hour

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        Hour = FrameProfiler.Hour
        frequent_profiles = ConfigVariableBool('frequent-frame-profiles', False)
        self._period = 2 * FrameProfiler.Minute
        if frequent_profiles:
            self._period = 1 * FrameProfiler.Minute
        self._jitterMagnitude = self._period * 0.75
        self._logSchedule = [1 * FrameProfiler.Hour, 4 * FrameProfiler.Hour, 12 * FrameProfiler.Hour, 1 * FrameProfiler.Day]
        if frequent_profiles:
            self._logSchedule = [1 * FrameProfiler.Minute, 4 * FrameProfiler.Minute, 12 * FrameProfiler.Minute, 24 * FrameProfiler.Minute]
        for t in self._logSchedule:
            assert t % self._period == 0
        for i in range(len(self._logSchedule)):
            e = self._logSchedule[i]
            for j in range(i, len(self._logSchedule)):
                assert self._logSchedule[j] % e == 0
        self._enableFC = FunctionCall(self._setEnabled, taskMgr.getProfileFramesSV())
        self._enableFC.pushCurrentState()

    def destroy(self):
        if False:
            while True:
                i = 10
        self._enableFC.set(False)
        self._enableFC.destroy()

    def _setEnabled(self, enabled):
        if False:
            return 10
        if enabled:
            self.notify.info('frame profiler started')
            self._startTime = ClockObject.getGlobalClock().getFrameTime()
            self._profileCounter = 0
            self._jitter = None
            self._period2aggregateProfile = {}
            self._id2session = {}
            self._id2task = {}
            self._task = taskMgr.doMethodLater(self._period, self._scheduleNextProfileDoLater, 'FrameProfilerStart-%s' % serialNum())
        else:
            self._task.remove()
            del self._task
            for session in self._period2aggregateProfile.values():
                session.release()
            del self._period2aggregateProfile
            for task in self._id2task.values():
                task.remove()
            del self._id2task
            for session in self._id2session.values():
                session.release()
            del self._id2session
            self.notify.info('frame profiler stopped')

    def _scheduleNextProfileDoLater(self, task):
        if False:
            print('Hello World!')
        self._scheduleNextProfile()
        return Task.done

    def _scheduleNextProfile(self):
        if False:
            for i in range(10):
                print('nop')
        self._profileCounter += 1
        self._timeElapsed = self._profileCounter * self._period
        time = self._startTime + self._timeElapsed
        jitter = self._jitter
        if jitter is None:
            jitter = normalDistrib(-self._jitterMagnitude, self._jitterMagnitude)
            time += jitter
        else:
            time -= jitter
            jitter = None
        self._jitter = jitter
        sessionId = serialNum()
        session = taskMgr.getProfileSession('FrameProfile-%s' % sessionId)
        self._id2session[sessionId] = session
        taskMgr.profileFrames(num=1, session=session, callback=Functor(self._analyzeResults, sessionId))
        delay = max(time - ClockObject.getGlobalClock().getFrameTime(), 0.0)
        self._task = taskMgr.doMethodLater(delay, self._scheduleNextProfileDoLater, 'FrameProfiler-%s' % serialNum())

    def _analyzeResults(self, sessionId):
        if False:
            return 10
        self._id2task[sessionId] = taskMgr.add(Functor(self._doAnalysis, sessionId), 'FrameProfilerAnalysis-%s' % sessionId)

    def _doAnalysis(self, sessionId, task):
        if False:
            print('Hello World!')
        if hasattr(task, '_generator'):
            gen = task._generator
        else:
            gen = self._doAnalysisGen(sessionId)
            task._generator = gen
        result = next(gen)
        if result == Task.done:
            del task._generator
        return result

    def _doAnalysisGen(self, sessionId):
        if False:
            for i in range(10):
                print('nop')
        p2ap = self._period2aggregateProfile
        self._id2task.pop(sessionId)
        session = self._id2session.pop(sessionId)
        if session.profileSucceeded():
            period = self._logSchedule[0]
            if period not in self._period2aggregateProfile:
                p2ap[period] = session.getReference()
            else:
                p2ap[period].aggregate(session)
        else:
            self.notify.warning('frame profile did not succeed')
        session.release()
        session = None
        counter = 0
        for pi in range(len(self._logSchedule)):
            period = self._logSchedule[pi]
            if self._timeElapsed % period == 0:
                if period in p2ap:
                    if counter >= 3:
                        counter = 0
                        yield Task.cont
                    self.notify.info('aggregate profile of sampled frames over last %s\n%s' % (formatTimeExact(period), p2ap[period].getResults()))
                    counter += 1
                    nextIndex = pi + 1
                    if nextIndex >= len(self._logSchedule):
                        nextPeriod = period * 2
                        self._logSchedule.append(nextPeriod)
                    else:
                        nextPeriod = self._logSchedule[nextIndex]
                    if nextPeriod not in p2ap:
                        p2ap[nextPeriod] = p2ap[period].getReference()
                    else:
                        p2ap[nextPeriod].aggregate(p2ap[period])
                    p2ap[period].release()
                    del p2ap[period]
            else:
                break
        yield Task.done