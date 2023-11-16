from panda3d.core import ConfigVariableBool, ConfigVariableDouble, ClockObject
from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.task.TaskManagerGlobal import taskMgr
from direct.showbase.Job import Job
from direct.showbase.PythonUtil import flywheel
from direct.showbase.MessengerGlobal import messenger

class JobManager:
    """
    Similar to the taskMgr but designed for tasks that are CPU-intensive and/or
    not time-critical. Jobs run in a fixed timeslice that the JobManager is
    allotted each frame.
    """
    notify = directNotify.newCategory('JobManager')
    TaskName = 'jobManager'

    def __init__(self, timeslice=None):
        if False:
            print('Hello World!')
        self._timeslice = timeslice
        self._pri2jobId2job = {}
        self._pri2jobIds = {}
        self._jobId2pri = {}
        self._jobId2timeslices = {}
        self._jobId2overflowTime = {}
        self._useOverflowTime = None
        self._jobIdGenerator = None
        self._highestPriority = Job.Priorities.Normal

    def destroy(self):
        if False:
            i = 10
            return i + 15
        taskMgr.remove(JobManager.TaskName)
        del self._pri2jobId2job

    def add(self, job):
        if False:
            return 10
        pri = job.getPriority()
        jobId = job._getJobId()
        self._pri2jobId2job.setdefault(pri, {})
        self._pri2jobId2job[pri][jobId] = job
        self._jobId2pri[jobId] = pri
        self._pri2jobIds.setdefault(pri, [])
        self._pri2jobIds[pri].append(jobId)
        self._jobId2timeslices[jobId] = pri
        self._jobId2overflowTime[jobId] = 0.0
        self._jobIdGenerator = None
        if len(self._jobId2pri) == 1:
            taskMgr.add(self._process, JobManager.TaskName)
            self._highestPriority = pri
        elif pri > self._highestPriority:
            self._highestPriority = pri
        self.notify.debug('added job: %s' % job.getJobName())

    def remove(self, job):
        if False:
            while True:
                i = 10
        jobId = job._getJobId()
        pri = self._jobId2pri.pop(jobId)
        self._pri2jobIds[pri].remove(jobId)
        del self._pri2jobId2job[pri][jobId]
        job._cleanupGenerator()
        self._jobId2timeslices.pop(jobId)
        self._jobId2overflowTime.pop(jobId)
        if len(self._pri2jobId2job[pri]) == 0:
            del self._pri2jobId2job[pri]
            if pri == self._highestPriority:
                if len(self._jobId2pri) > 0:
                    priorities = self._getSortedPriorities()
                    self._highestPriority = priorities[-1]
                else:
                    taskMgr.remove(JobManager.TaskName)
                    self._highestPriority = 0
        self.notify.debug('removed job: %s' % job.getJobName())

    def finish(self, job):
        if False:
            while True:
                i = 10
        assert self.notify.debugCall()
        jobId = job._getJobId()
        pri = self._jobId2pri[jobId]
        job = self._pri2jobId2job[pri][jobId]
        gen = job._getGenerator()
        if __debug__:
            job._pstats.start()
        job.resume()
        while True:
            try:
                result = next(gen)
            except StopIteration:
                self.notify.warning('job %s never yielded Job.Done' % job)
                result = Job.Done
            if result is Job.Done:
                job.suspend()
                self.remove(job)
                job._setFinished()
                messenger.send(job.getFinishedEvent())
                break
        if __debug__:
            job._pstats.stop()

    @staticmethod
    def getDefaultTimeslice():
        if False:
            i = 10
            return i + 15
        return ConfigVariableDouble('job-manager-timeslice-ms', 0.5).value / 1000.0

    def getTimeslice(self):
        if False:
            for i in range(10):
                print('nop')
        if self._timeslice:
            return self._timeslice
        return self.getDefaultTimeslice()

    def setTimeslice(self, timeslice):
        if False:
            while True:
                i = 10
        self._timeslice = timeslice

    def _getSortedPriorities(self):
        if False:
            for i in range(10):
                print('nop')
        return sorted(self._pri2jobId2job)

    def _process(self, task=None):
        if False:
            return 10
        if self._useOverflowTime is None:
            self._useOverflowTime = ConfigVariableBool('job-use-overflow-time', 1).value
        if len(self._pri2jobId2job) > 0:
            clock = ClockObject.getGlobalClock()
            endT = clock.getRealTime() + self.getTimeslice() * 0.9
            while True:
                if self._jobIdGenerator is None:
                    self._jobIdGenerator = flywheel(list(self._jobId2timeslices.keys()), countFunc=lambda jobId: self._jobId2timeslices[jobId])
                try:
                    jobId = next(self._jobIdGenerator)
                except StopIteration:
                    self._jobIdGenerator = None
                    continue
                pri = self._jobId2pri.get(jobId)
                if pri is None:
                    continue
                if self._useOverflowTime:
                    overflowTime = self._jobId2overflowTime[jobId]
                    timeLeft = endT - clock.getRealTime()
                    if overflowTime >= timeLeft:
                        self._jobId2overflowTime[jobId] = max(0.0, overflowTime - timeLeft)
                        break
                job = self._pri2jobId2job[pri][jobId]
                gen = job._getGenerator()
                if __debug__:
                    job._pstats.start()
                job.resume()
                while clock.getRealTime() < endT:
                    try:
                        result = next(gen)
                    except StopIteration:
                        self.notify.warning('job %s never yielded Job.Done' % job)
                        result = Job.Done
                    if result is Job.Sleep:
                        job.suspend()
                        if __debug__:
                            job._pstats.stop()
                        break
                    elif result is Job.Done:
                        job.suspend()
                        self.remove(job)
                        job._setFinished()
                        if __debug__:
                            job._pstats.stop()
                        messenger.send(job.getFinishedEvent())
                        break
                else:
                    job.suspend()
                    overflowTime = clock.getRealTime() - endT
                    if overflowTime > self.getTimeslice():
                        self._jobId2overflowTime[jobId] += overflowTime
                    if __debug__:
                        job._pstats.stop()
                    break
                if len(self._pri2jobId2job) == 0:
                    break
        return task.cont

    def __repr__(self):
        if False:
            print('Hello World!')
        s = '======================================================='
        s += '\nJobManager: active jobs in descending order of priority'
        s += '\n======================================================='
        pris = self._getSortedPriorities()
        if len(pris) == 0:
            s += '\n    no jobs running'
        else:
            pris.reverse()
            for pri in pris:
                jobId2job = self._pri2jobId2job[pri]
                for jobId in self._pri2jobIds[pri]:
                    job = jobId2job[jobId]
                    s += '\n%5d: %s (jobId %s)' % (pri, job.getJobName(), jobId)
        s += '\n'
        return s