from panda3d.core import ClockObject, ConfigVariableBool
from direct.directnotify import DirectNotifyGlobal
from direct.showbase import DirectObject
import math
NetworkTimeBits = 16
NetworkTimePrecision = 100.0
NetworkTimeMask = (1 << NetworkTimeBits) - 1
NetworkTimeSignedMask = NetworkTimeMask >> 1
NetworkTimeTopBits = 32 - NetworkTimeBits
MaxTimeDelta = NetworkTimeSignedMask / NetworkTimePrecision
ClockDriftPerHour = 1.0
ClockDriftPerSecond = ClockDriftPerHour / 3600.0
P2PResyncDelay = 10.0

class ClockDelta(DirectObject.DirectObject):
    """
    The ClockDelta object converts between universal ("network") time,
    which is used for all network traffic, and local time (e.g. as
    returned by getFrameTime() or getRealTime()), which is used for
    everything else.
    """
    notify = DirectNotifyGlobal.directNotify.newCategory('ClockDelta')

    def __init__(self):
        if False:
            while True:
                i = 10
        self.globalClock = ClockObject.getGlobalClock()
        self.delta = 0
        self.uncertainty = None
        self.lastResync = 0.0
        self.accept('resetClock', self.__resetClock)

    def getDelta(self):
        if False:
            return 10
        return self.delta

    def getUncertainty(self):
        if False:
            for i in range(10):
                print('nop')
        if self.uncertainty is None:
            return None
        now = self.globalClock.getRealTime()
        elapsed = now - self.lastResync
        return self.uncertainty + elapsed * ClockDriftPerSecond

    def getLastResync(self):
        if False:
            while True:
                i = 10
        return self.lastResync

    def __resetClock(self, timeDelta):
        if False:
            while True:
                i = 10
        '\n        this is called when the global clock gets adjusted\n        timeDelta is equal to the amount of time, in seconds,\n        that has been added to the global clock\n        '
        assert self.notify.debug('adjusting timebase by %f seconds' % timeDelta)
        self.delta += timeDelta

    def clear(self):
        if False:
            while True:
                i = 10
        '\n        Throws away any previous synchronization information.\n        '
        self.delta = 0
        self.uncertainty = None
        self.lastResync = 0.0

    def resynchronize(self, localTime, networkTime, newUncertainty, trustNew=1):
        if False:
            print('Hello World!')
        'resynchronize(self, float localTime, int32 networkTime,\n                         float newUncertainty)\n\n        Accepts a new networkTime value, which is understood to\n        represent the same moment as localTime, plus or minus\n        uncertainty seconds.  Improves our current notion of the time\n        delta accordingly.\n        '
        newDelta = float(localTime) - float(networkTime) / NetworkTimePrecision
        self.newDelta(localTime, newDelta, newUncertainty, trustNew=trustNew)

    def peerToPeerResync(self, avId, timestamp, serverTime, uncertainty):
        if False:
            return 10
        "\n        Accepts an AI time and uncertainty value from another client,\n        along with a local timestamp value of the message from this\n        client which prompted the other client to send us its delta\n        information.\n\n        The return value is true if the other client's measurement was\n        reasonably close to our own, or false if the other client's\n        time estimate was wildly divergent from our own; the return\n        value is negative if the test was not even considered (because\n        it happened too soon after another recent request).\n        "
        now = self.globalClock.getRealTime()
        if now - self.lastResync < P2PResyncDelay:
            assert self.notify.debug('Ignoring request for resync from %s within %.3f s.' % (avId, now - self.lastResync))
            return -1
        local = self.networkToLocalTime(timestamp, now)
        elapsed = now - local
        delta = (local + now) / 2.0 - serverTime
        gotSync = 0
        if elapsed <= 0 or elapsed > P2PResyncDelay:
            self.notify.info('Ignoring old request for resync from %s.' % avId)
        else:
            self.notify.info('Got sync +/- %.3f s, elapsed %.3f s, from %s.' % (uncertainty, elapsed, avId))
            delta -= elapsed / 2.0
            uncertainty += elapsed / 2.0
            gotSync = self.newDelta(local, delta, uncertainty, trustNew=0)
        return gotSync

    def newDelta(self, localTime, newDelta, newUncertainty, trustNew=1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Accepts a new delta and uncertainty pair, understood to\n        represent time as of localTime.  Improves our current notion\n        of the time delta accordingly.  The return value is true if\n        the new measurement was used, false if it was discarded.\n        '
        oldUncertainty = self.getUncertainty()
        if oldUncertainty is not None:
            self.notify.info('previous delta at %.3f s, +/- %.3f s.' % (self.delta, oldUncertainty))
            self.notify.info('new delta at %.3f s, +/- %.3f s.' % (newDelta, newUncertainty))
            oldLow = self.delta - oldUncertainty
            oldHigh = self.delta + oldUncertainty
            newLow = newDelta - newUncertainty
            newHigh = newDelta + newUncertainty
            low = max(oldLow, newLow)
            high = min(oldHigh, newHigh)
            if low > high:
                if not trustNew:
                    self.notify.info('discarding new delta.')
                    return 0
                self.notify.info('discarding previous delta.')
            else:
                newDelta = (low + high) / 2.0
                newUncertainty = (high - low) / 2.0
                self.notify.info('intersection at %.3f s, +/- %.3f s.' % (newDelta, newUncertainty))
        self.delta = newDelta
        self.uncertainty = newUncertainty
        self.lastResync = localTime
        return 1

    def networkToLocalTime(self, networkTime, now=None, bits=16, ticksPerSec=NetworkTimePrecision):
        if False:
            while True:
                i = 10
        'networkToLocalTime(self, int networkTime)\n\n        Converts the indicated networkTime to the corresponding\n        localTime value.  The time is assumed to be within +/- 5\n        minutes of the current local time given in now, or\n        getRealTime() if now is not specified.\n        '
        if now is None:
            now = self.globalClock.getRealTime()
        if self.globalClock.getMode() == ClockObject.MNonRealTime and ConfigVariableBool('movie-network-time', False):
            return now
        ntime = int(math.floor((now - self.delta) * ticksPerSec + 0.5))
        if bits == 16:
            diff = self.__signExtend(networkTime - ntime)
        else:
            diff = networkTime - ntime
        return now + float(diff) / ticksPerSec

    def localToNetworkTime(self, localTime, bits=16, ticksPerSec=NetworkTimePrecision):
        if False:
            for i in range(10):
                print('nop')
        'localToNetworkTime(self, float localTime)\n\n        Converts the indicated localTime to the corresponding\n        networkTime value.\n        '
        ntime = int(math.floor((localTime - self.delta) * ticksPerSec + 0.5))
        if bits == 16:
            return self.__signExtend(ntime)
        else:
            return ntime

    def getRealNetworkTime(self, bits=16, ticksPerSec=NetworkTimePrecision):
        if False:
            i = 10
            return i + 15
        '\n        Returns the current getRealTime() expressed as a network time.\n        '
        return self.localToNetworkTime(self.globalClock.getRealTime(), bits=bits, ticksPerSec=ticksPerSec)

    def getFrameNetworkTime(self, bits=16, ticksPerSec=NetworkTimePrecision):
        if False:
            i = 10
            return i + 15
        '\n        Returns the current getFrameTime() expressed as a network time.\n        '
        return self.localToNetworkTime(self.globalClock.getFrameTime(), bits=bits, ticksPerSec=ticksPerSec)

    def localElapsedTime(self, networkTime, bits=16, ticksPerSec=NetworkTimePrecision):
        if False:
            for i in range(10):
                print('nop')
        'localElapsedTime(self, int networkTime)\n\n        Returns the amount of time elapsed (in seconds) on the client\n        since the server message was sent.  Negative values are\n        clamped to zero.\n        '
        now = self.globalClock.getFrameTime()
        dt = now - self.networkToLocalTime(networkTime, now, bits=bits, ticksPerSec=ticksPerSec)
        return max(dt, 0.0)

    def __signExtend(self, networkTime):
        if False:
            i = 10
            return i + 15
        '__signExtend(self, int networkTime)\n\n        Preserves the lower NetworkTimeBits of the networkTime value,\n        and extends the sign bit all the way up.\n        '
        r = (networkTime + 32768 & NetworkTimeMask) - 32768
        assert -32768 <= r <= 32767
        return r
globalClockDelta = ClockDelta()