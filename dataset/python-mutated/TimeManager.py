from panda3d.core import ClockObject, ConfigVariableDouble, ConfigVariableInt
from direct.task import Task
from direct.task.TaskManagerGlobal import taskMgr
from direct.distributed import DistributedObject
from direct.directnotify import DirectNotifyGlobal
from direct.distributed.ClockDelta import globalClockDelta
from direct.showbase.MessengerGlobal import messenger

class TimeManager(DistributedObject.DistributedObject):
    """
    This DistributedObject lives on the AI and on the client side, and
    serves to synchronize the time between them so they both agree, to
    within a few hundred milliseconds at least, what time it is.

    It uses a pull model where the client can request a
    synchronization check from time to time.  It also employs a
    round-trip measurement to minimize the effect of latency.
    """
    notify = DirectNotifyGlobal.directNotify.newCategory('TimeManager')
    updateFreq = ConfigVariableDouble('time-manager-freq', 1800).getValue()
    minWait = ConfigVariableDouble('time-manager-min-wait', 10).getValue()
    maxUncertainty = ConfigVariableDouble('time-manager-max-uncertainty', 1).getValue()
    maxAttempts = ConfigVariableInt('time-manager-max-attempts', 5).getValue()
    extraSkew = ConfigVariableInt('time-manager-extra-skew', 0).getValue()
    if extraSkew != 0:
        notify.info('Simulating clock skew of %0.3f s' % extraSkew)
    reportFrameRateInterval = ConfigVariableDouble('report-frame-rate-interval', 300.0).getValue()

    def __init__(self, cr):
        if False:
            print('Hello World!')
        DistributedObject.DistributedObject.__init__(self, cr)
        self.thisContext = -1
        self.nextContext = 0
        self.attemptCount = 0
        self.start = 0
        self.lastAttempt = -self.minWait * 2

    def generate(self):
        if False:
            i = 10
            return i + 15
        '\n        This method is called when the DistributedObject is reintroduced\n        to the world, either for the first time or from the cache.\n        '
        DistributedObject.DistributedObject.generate(self)
        self.accept('clock_error', self.handleClockError)
        if self.updateFreq > 0:
            self.startTask()

    def announceGenerate(self):
        if False:
            while True:
                i = 10
        DistributedObject.DistributedObject.announceGenerate(self)
        self.cr.timeManager = self
        self.synchronize('TimeManager.announceGenerate')

    def disable(self):
        if False:
            i = 10
            return i + 15
        '\n        This method is called when the DistributedObject is removed from\n        active duty and stored in a cache.\n        '
        self.ignore('clock_error')
        self.stopTask()
        taskMgr.remove('frameRateMonitor')
        if self.cr.timeManager is self:
            self.cr.timeManager = None
        DistributedObject.DistributedObject.disable(self)

    def delete(self):
        if False:
            while True:
                i = 10
        '\n        This method is called when the DistributedObject is permanently\n        removed from the world and deleted from the cache.\n        '
        DistributedObject.DistributedObject.delete(self)

    def startTask(self):
        if False:
            while True:
                i = 10
        self.stopTask()
        taskMgr.doMethodLater(self.updateFreq, self.doUpdate, 'timeMgrTask')

    def stopTask(self):
        if False:
            while True:
                i = 10
        taskMgr.remove('timeMgrTask')

    def doUpdate(self, task):
        if False:
            return 10
        self.synchronize('timer')
        taskMgr.doMethodLater(self.updateFreq, self.doUpdate, 'timeMgrTask')
        return Task.done

    def handleClockError(self):
        if False:
            i = 10
            return i + 15
        self.synchronize('clock error')

    def synchronize(self, description):
        if False:
            i = 10
            return i + 15
        'synchronize(self, string description)\n\n        Call this function from time to time to synchronize watches\n        with the server.  This initiates a round-trip transaction;\n        when the transaction completes, the time will be synced.\n\n        The description is the string that will be written to the log\n        file regarding the reason for this synchronization attempt.\n\n        The return value is true if the attempt is made, or false if\n        it is too soon since the last attempt.\n        '
        now = ClockObject.getGlobalClock().getRealTime()
        if now - self.lastAttempt < self.minWait:
            self.notify.debug('Not resyncing (too soon): %s' % description)
            return 0
        self.talkResult = 0
        self.thisContext = self.nextContext
        self.attemptCount = 0
        self.nextContext = self.nextContext + 1 & 255
        self.notify.info('Clock sync: %s' % description)
        self.start = now
        self.lastAttempt = now
        self.sendUpdate('requestServerTime', [self.thisContext])
        return 1

    def serverTime(self, context, timestamp):
        if False:
            while True:
                i = 10
        'serverTime(self, int8 context, int32 timestamp)\n\n        This message is sent from the AI to the client in response to\n        a previous requestServerTime.  It contains the time as\n        observed by the AI.\n\n        The client should use this, in conjunction with the time\n        measurement taken before calling requestServerTime (above), to\n        determine the clock delta between the AI and the client\n        machines.\n        '
        clock = ClockObject.getGlobalClock()
        end = clock.getRealTime()
        if context != self.thisContext:
            self.notify.info('Ignoring TimeManager response for old context %d' % context)
            return
        elapsed = end - self.start
        self.attemptCount += 1
        self.notify.info('Clock sync roundtrip took %0.3f ms' % (elapsed * 1000.0))
        average = (self.start + end) / 2.0 - self.extraSkew
        uncertainty = (end - self.start) / 2.0 + abs(self.extraSkew)
        globalClockDelta.resynchronize(average, timestamp, uncertainty)
        self.notify.info('Local clock uncertainty +/- %.3f s' % globalClockDelta.getUncertainty())
        if globalClockDelta.getUncertainty() > self.maxUncertainty:
            if self.attemptCount < self.maxAttempts:
                self.notify.info('Uncertainty is too high, trying again.')
                self.start = clock.getRealTime()
                self.sendUpdate('requestServerTime', [self.thisContext])
                return
            self.notify.info('Giving up on uncertainty requirement.')
        messenger.send('gotTimeSync', taskChain='default')
        messenger.send(self.cr.uniqueName('gotTimeSync'), taskChain='default')