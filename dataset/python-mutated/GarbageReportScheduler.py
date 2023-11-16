from direct.showbase.GarbageReport import GarbageReport
from direct.showbase.PythonUtil import serialNum
from direct.task.TaskManagerGlobal import taskMgr

class GarbageReportScheduler:
    """Runs a garbage report every once in a while and logs the results."""

    def __init__(self, waitBetween=None, waitScale=None):
        if False:
            print('Hello World!')
        if waitBetween is None:
            waitBetween = 30 * 60
        if waitScale is None:
            waitScale = 1.5
        self._waitBetween = waitBetween
        self._waitScale = waitScale
        self._taskName = 'startScheduledGarbageReport-%s' % serialNum()
        self._garbageReport = None
        self._scheduleNextGarbageReport()

    def getTaskName(self):
        if False:
            for i in range(10):
                print('nop')
        return self._taskName

    def _scheduleNextGarbageReport(self, garbageReport=None):
        if False:
            for i in range(10):
                print('nop')
        if garbageReport:
            assert garbageReport is self._garbageReport
            self._garbageReport = None
        taskMgr.doMethodLater(self._waitBetween, self._runGarbageReport, self._taskName)
        self._waitBetween = self._waitBetween * self._waitScale

    def _runGarbageReport(self, task):
        if False:
            while True:
                i = 10
        self._garbageReport = GarbageReport('ScheduledGarbageReport', threaded=True, doneCallback=self._scheduleNextGarbageReport, autoDestroy=True, priority=GarbageReport.Priorities.Normal * 3)
        return task.done