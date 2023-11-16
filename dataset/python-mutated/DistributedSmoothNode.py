"""DistributedSmoothNode module: contains the DistributedSmoothNode class"""
import math
from panda3d.core import ClockObject, ConfigVariableBool, ConfigVariableDouble, NodePath
from panda3d.direct import SmoothMover
from .ClockDelta import globalClockDelta
from . import DistributedNode
from . import DistributedSmoothNodeBase
from direct.task.Task import cont
from direct.task.TaskManagerGlobal import taskMgr
from direct.showbase.PythonUtil import report
MaxFuture = ConfigVariableDouble('smooth-max-future', 0.2)
MinSuggestResync = ConfigVariableDouble('smooth-min-suggest-resync', 15)
EnableSmoothing = ConfigVariableBool('smooth-enable-smoothing', True)
EnablePrediction = ConfigVariableBool('smooth-enable-prediction', True)
Lag = ConfigVariableDouble('smooth-lag', 0.2)
PredictionLag = ConfigVariableDouble('smooth-prediction-lag', 0.0)
GlobalSmoothing = 0
GlobalPrediction = 0

def globalActivateSmoothing(smoothing, prediction):
    if False:
        print('Hello World!')
    ' Globally activates or deactivates smoothing and prediction on\n    all DistributedSmoothNodes currently in existence, or yet to be\n    generated. '
    global GlobalSmoothing, GlobalPrediction
    GlobalSmoothing = smoothing
    GlobalPrediction = prediction
    for obj in base.cr.getAllOfType(DistributedSmoothNode):
        obj.activateSmoothing(smoothing, prediction)
activateSmoothing = globalActivateSmoothing

class DistributedSmoothNode(DistributedNode.DistributedNode, DistributedSmoothNodeBase.DistributedSmoothNodeBase):
    """
    This specializes DistributedNode to add functionality to smooth
    motion over time, via the SmoothMover C++ object defined in
    DIRECT.
    """

    def __init__(self, cr):
        if False:
            while True:
                i = 10
        if not hasattr(self, 'DistributedSmoothNode_initialized'):
            self.DistributedSmoothNode_initialized = 1
            DistributedNode.DistributedNode.__init__(self, cr)
            DistributedSmoothNodeBase.DistributedSmoothNodeBase.__init__(self)
            self.smoothStarted = 0
            self.localControl = False
            self.stopped = False

    def generate(self):
        if False:
            return 10
        self.smoother = SmoothMover()
        self.smoothStarted = 0
        self.lastSuggestResync = 0
        self._smoothWrtReparents = False
        DistributedNode.DistributedNode.generate(self)
        DistributedSmoothNodeBase.DistributedSmoothNodeBase.generate(self)
        self.cnode.setRepository(self.cr, 0, 0)
        self.activateSmoothing(GlobalSmoothing, GlobalPrediction)
        self.stopped = False

    def disable(self):
        if False:
            for i in range(10):
                print('nop')
        DistributedSmoothNodeBase.DistributedSmoothNodeBase.disable(self)
        DistributedNode.DistributedNode.disable(self)
        del self.smoother

    def delete(self):
        if False:
            return 10
        DistributedSmoothNodeBase.DistributedSmoothNodeBase.delete(self)
        DistributedNode.DistributedNode.delete(self)

    def smoothPosition(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This function updates the position of the node to its computed\n        smoothed position.  This may be overridden by a derived class\n        to specialize the behavior.\n        '
        self.smoother.computeAndApplySmoothPosHpr(self, self)

    def doSmoothTask(self, task):
        if False:
            print('Hello World!')
        self.smoothPosition()
        return cont

    def wantsSmoothing(self):
        if False:
            return 10
        return 1

    def startSmooth(self):
        if False:
            print('Hello World!')
        "\n        This function starts the task that ensures the node is\n        positioned correctly every frame.  However, while the task is\n        running, you won't be able to lerp the node or directly\n        position it.\n        "
        if not self.wantsSmoothing() or self.isDisabled() or self.isLocal():
            return
        if not self.smoothStarted:
            taskName = self.taskName('smooth')
            taskMgr.remove(taskName)
            self.reloadPosition()
            taskMgr.add(self.doSmoothTask, taskName)
            self.smoothStarted = 1

    def stopSmooth(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This function stops the task spawned by startSmooth(), and\n        allows show code to move the node around directly.\n        '
        if self.smoothStarted:
            taskName = self.taskName('smooth')
            taskMgr.remove(taskName)
            self.forceToTruePosition()
            self.smoothStarted = 0

    def setSmoothWrtReparents(self, flag):
        if False:
            for i in range(10):
                print('nop')
        self._smoothWrtReparents = flag

    def getSmoothWrtReparents(self):
        if False:
            for i in range(10):
                print('nop')
        return self._smoothWrtReparents

    def forceToTruePosition(self):
        if False:
            return 10
        '\n        This forces the node to reposition itself to its latest known\n        position.  This may result in a pop as the node skips the last\n        of its lerp points.\n        '
        if not self.isLocal() and self.smoother.getLatestPosition():
            self.smoother.applySmoothPosHpr(self, self)
        self.smoother.clearPositions(1)

    def reloadPosition(self):
        if False:
            print('Hello World!')
        '\n        This function re-reads the position from the node itself and\n        clears any old position reports for the node.  This should be\n        used whenever show code bangs on the node position and expects\n        it to stick.\n        '
        self.smoother.clearPositions(0)
        self.smoother.setPosHpr(self.getPos(), self.getHpr())
        self.smoother.setPhonyTimestamp()
        self.smoother.markPosition()

    def _checkResume(self, timestamp):
        if False:
            print('Hello World!')
        "\n        Determine if we were previously stopped and now need to\n        resume movement by making sure any old stored positions\n        reflect the node's current position\n        "
        if self.stopped:
            currTime = ClockObject.getGlobalClock().getFrameTime()
            now = currTime - self.smoother.getExpectedBroadcastPeriod()
            last = self.smoother.getMostRecentTimestamp()
            if now > last:
                if timestamp is None:
                    local = 0.0
                else:
                    local = globalClockDelta.networkToLocalTime(timestamp, currTime)
                self.smoother.setPhonyTimestamp(local, True)
                self.smoother.markPosition()
        self.stopped = False

    def setSmStop(self, timestamp=None):
        if False:
            while True:
                i = 10
        self.setComponentTLive(timestamp)
        self.stopped = True

    def setSmH(self, h, timestamp=None):
        if False:
            i = 10
            return i + 15
        self._checkResume(timestamp)
        self.setComponentH(h)
        self.setComponentTLive(timestamp)

    def setSmZ(self, z, timestamp=None):
        if False:
            for i in range(10):
                print('nop')
        self._checkResume(timestamp)
        self.setComponentZ(z)
        self.setComponentTLive(timestamp)

    def setSmXY(self, x, y, timestamp=None):
        if False:
            for i in range(10):
                print('nop')
        self._checkResume(timestamp)
        self.setComponentX(x)
        self.setComponentY(y)
        self.setComponentTLive(timestamp)

    def setSmXZ(self, x, z, timestamp=None):
        if False:
            i = 10
            return i + 15
        self._checkResume(timestamp)
        self.setComponentX(x)
        self.setComponentZ(z)
        self.setComponentTLive(timestamp)

    def setSmPos(self, x, y, z, timestamp=None):
        if False:
            i = 10
            return i + 15
        self._checkResume(timestamp)
        self.setComponentX(x)
        self.setComponentY(y)
        self.setComponentZ(z)
        self.setComponentTLive(timestamp)

    def setSmHpr(self, h, p, r, timestamp=None):
        if False:
            for i in range(10):
                print('nop')
        self._checkResume(timestamp)
        self.setComponentH(h)
        self.setComponentP(p)
        self.setComponentR(r)
        self.setComponentTLive(timestamp)

    def setSmXYH(self, x, y, h, timestamp):
        if False:
            print('Hello World!')
        self._checkResume(timestamp)
        self.setComponentX(x)
        self.setComponentY(y)
        self.setComponentH(h)
        self.setComponentTLive(timestamp)

    def setSmXYZH(self, x, y, z, h, timestamp=None):
        if False:
            while True:
                i = 10
        self._checkResume(timestamp)
        self.setComponentX(x)
        self.setComponentY(y)
        self.setComponentZ(z)
        self.setComponentH(h)
        self.setComponentTLive(timestamp)

    def setSmPosHpr(self, x, y, z, h, p, r, timestamp=None):
        if False:
            for i in range(10):
                print('nop')
        self._checkResume(timestamp)
        self.setComponentX(x)
        self.setComponentY(y)
        self.setComponentZ(z)
        self.setComponentH(h)
        self.setComponentP(p)
        self.setComponentR(r)
        self.setComponentTLive(timestamp)

    def setSmPosHprL(self, l, x, y, z, h, p, r, timestamp=None):
        if False:
            print('Hello World!')
        self._checkResume(timestamp)
        self.setComponentL(l)
        self.setComponentX(x)
        self.setComponentY(y)
        self.setComponentZ(z)
        self.setComponentH(h)
        self.setComponentP(p)
        self.setComponentR(r)
        self.setComponentTLive(timestamp)

    @report(types=['args'], dConfigParam='smoothnode')
    def setComponentX(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.smoother.setX(x)

    @report(types=['args'], dConfigParam='smoothnode')
    def setComponentY(self, y):
        if False:
            return 10
        self.smoother.setY(y)

    @report(types=['args'], dConfigParam='smoothnode')
    def setComponentZ(self, z):
        if False:
            print('Hello World!')
        self.smoother.setZ(z)

    @report(types=['args'], dConfigParam='smoothnode')
    def setComponentH(self, h):
        if False:
            i = 10
            return i + 15
        self.smoother.setH(h)

    @report(types=['args'], dConfigParam='smoothnode')
    def setComponentP(self, p):
        if False:
            while True:
                i = 10
        self.smoother.setP(p)

    @report(types=['args'], dConfigParam='smoothnode')
    def setComponentR(self, r):
        if False:
            while True:
                i = 10
        self.smoother.setR(r)

    @report(types=['args'], dConfigParam='smoothnode')
    def setComponentL(self, l):
        if False:
            print('Hello World!')
        if l != self.zoneId:
            self.setLocation(self.parentId, l)

    @report(types=['args'], dConfigParam='smoothnode')
    def setComponentT(self, timestamp):
        if False:
            print('Hello World!')
        self.smoother.setPhonyTimestamp()
        self.smoother.clearPositions(1)
        self.smoother.markPosition()
        self.forceToTruePosition()

    @report(types=['args'], dConfigParam='smoothnode')
    def setComponentTLive(self, timestamp):
        if False:
            while True:
                i = 10
        if timestamp is None:
            if self.smoother.hasMostRecentTimestamp():
                self.smoother.setTimestamp(self.smoother.getMostRecentTimestamp())
            else:
                self.smoother.setPhonyTimestamp()
            self.smoother.markPosition()
        else:
            globalClock = ClockObject.getGlobalClock()
            now = globalClock.getFrameTime()
            local = globalClockDelta.networkToLocalTime(timestamp, now)
            realTime = globalClock.getRealTime()
            chug = realTime - now
            howFarFuture = local - now
            if howFarFuture - chug >= MaxFuture.value:
                if globalClockDelta.getUncertainty() is not None and realTime - self.lastSuggestResync >= MinSuggestResync.value and hasattr(self.cr, 'localAvatarDoId'):
                    self.lastSuggestResync = realTime
                    timestampB = globalClockDelta.localToNetworkTime(realTime)
                    serverTime = realTime - globalClockDelta.getDelta()
                    assert self.notify.info('Suggesting resync for %s, with discrepency %s; local time is %s and server time is %s.' % (self.doId, howFarFuture - chug, realTime, serverTime))
                    self.d_suggestResync(self.cr.localAvatarDoId, timestamp, timestampB, serverTime, globalClockDelta.getUncertainty())
            self.smoother.setTimestamp(local)
            self.smoother.markPosition()
        if not self.localControl and (not self.smoothStarted) and self.smoother.getLatestPosition():
            self.smoother.applySmoothPosHpr(self, self)

    def getComponentL(self):
        if False:
            i = 10
            return i + 15
        return self.zoneId

    def getComponentX(self):
        if False:
            return 10
        return self.getX()

    def getComponentY(self):
        if False:
            for i in range(10):
                print('nop')
        return self.getY()

    def getComponentZ(self):
        if False:
            while True:
                i = 10
        return self.getZ()

    def getComponentH(self):
        if False:
            for i in range(10):
                print('nop')
        return self.getH()

    def getComponentP(self):
        if False:
            return 10
        return self.getP()

    def getComponentR(self):
        if False:
            return 10
        return self.getR()

    def getComponentT(self):
        if False:
            print('Hello World!')
        return 0

    @report(types=['args'], dConfigParam='smoothnode')
    def clearSmoothing(self, bogus=None):
        if False:
            for i in range(10):
                print('nop')
        self.smoother.clearPositions(1)

    @report(types=['args'], dConfigParam='smoothnode')
    def wrtReparentTo(self, parent):
        if False:
            print('Hello World!')
        if self.smoothStarted:
            if self._smoothWrtReparents:
                self.smoother.handleWrtReparent(self.getParent(), parent)
                NodePath.wrtReparentTo(self, parent)
            else:
                self.forceToTruePosition()
                NodePath.wrtReparentTo(self, parent)
                self.reloadPosition()
        else:
            NodePath.wrtReparentTo(self, parent)

    @report(types=['args'], dConfigParam='smoothnode')
    def d_setParent(self, parentToken):
        if False:
            for i in range(10):
                print('nop')
        DistributedNode.DistributedNode.d_setParent(self, parentToken)
        self.forceToTruePosition()
        self.sendCurrentPosition()

    def d_suggestResync(self, avId, timestampA, timestampB, serverTime, uncertainty):
        if False:
            return 10
        serverTimeSec = math.floor(serverTime)
        serverTimeUSec = (serverTime - serverTimeSec) * 10000.0
        self.sendUpdate('suggestResync', [avId, timestampA, timestampB, serverTimeSec, serverTimeUSec, uncertainty])

    def suggestResync(self, avId, timestampA, timestampB, serverTimeSec, serverTimeUSec, uncertainty):
        if False:
            for i in range(10):
                print('nop')
        '\n        This message is sent from one client to another when the other\n        client receives a timestamp from this client that is so far\n        out of date as to suggest that one or both clients needs to\n        resynchronize their clock information.\n        '
        serverTime = float(serverTimeSec) + float(serverTimeUSec) / 10000.0
        result = self.peerToPeerResync(avId, timestampA, serverTime, uncertainty)
        if result >= 0 and globalClockDelta.getUncertainty() is not None:
            other = self.cr.doId2do.get(avId)
            if not other:
                assert self.notify.info("Warning: couldn't find the avatar %d" % avId)
            elif hasattr(other, 'd_returnResync') and hasattr(self.cr, 'localAvatarDoId'):
                globalClock = ClockObject.getGlobalClock()
                realTime = globalClock.getRealTime()
                serverTime = realTime - globalClockDelta.getDelta()
                assert self.notify.info('Returning resync for %s; local time is %s and server time is %s.' % (self.doId, realTime, serverTime))
                other.d_returnResync(self.cr.localAvatarDoId, timestampB, serverTime, globalClockDelta.getUncertainty())

    def d_returnResync(self, avId, timestampB, serverTime, uncertainty):
        if False:
            for i in range(10):
                print('nop')
        serverTimeSec = math.floor(serverTime)
        serverTimeUSec = (serverTime - serverTimeSec) * 10000.0
        self.sendUpdate('returnResync', [avId, timestampB, serverTimeSec, serverTimeUSec, uncertainty])

    def returnResync(self, avId, timestampB, serverTimeSec, serverTimeUSec, uncertainty):
        if False:
            return 10
        "\n        A reply sent by a client whom we recently sent suggestResync\n        to, this reports the client's new delta information so we can\n        adjust our clock as well.\n        "
        serverTime = float(serverTimeSec) + float(serverTimeUSec) / 10000.0
        self.peerToPeerResync(avId, timestampB, serverTime, uncertainty)

    def peerToPeerResync(self, avId, timestamp, serverTime, uncertainty):
        if False:
            print('Hello World!')
        gotSync = globalClockDelta.peerToPeerResync(avId, timestamp, serverTime, uncertainty)
        if not gotSync:
            if self.cr.timeManager is not None:
                self.cr.timeManager.synchronize('suggested by %d' % avId)
        return gotSync

    def activateSmoothing(self, smoothing, prediction):
        if False:
            for i in range(10):
                print('nop')
        "\n        Enables or disables the smoothing of other avatars' motion.\n        This used to be a global flag, but now it is specific to each\n        avatar instance.  However, see globalActivateSmoothing() in\n        this module.\n\n        If smoothing is off, no kind of smoothing will be performed,\n        regardless of the setting of prediction.\n\n        This is not necessarily predictive smoothing; if predictive\n        smoothing is off, avatars will be lagged by a certain factor\n        to achieve smooth motion.  Otherwise, if predictive smoothing\n        is on, avatars will be drawn as nearly as possible in their\n        current position, by extrapolating from old position reports.\n\n        This assumes you have a client repository that knows its\n        localAvatarDoId -- stored in self.cr.localAvatarDoId\n        "
        if smoothing and EnableSmoothing:
            if prediction and EnablePrediction:
                self.smoother.setSmoothMode(SmoothMover.SMOn)
                self.smoother.setPredictionMode(SmoothMover.PMOn)
                self.smoother.setDelay(PredictionLag.value)
            else:
                self.smoother.setSmoothMode(SmoothMover.SMOn)
                self.smoother.setPredictionMode(SmoothMover.PMOff)
                self.smoother.setDelay(Lag.value)
        else:
            self.smoother.setSmoothMode(SmoothMover.SMOff)
            self.smoother.setPredictionMode(SmoothMover.PMOff)
            self.smoother.setDelay(0.0)