"""Class used to create and control VRPN devices."""
from direct.showbase.DirectObject import DirectObject
from panda3d.core import AnalogNode, ButtonNode, ConfigVariableDouble, ConfigVariableString, DialNode, TrackerNode
from panda3d.vrpn import VrpnClient
ANALOG_MIN = -0.95
ANALOG_MAX = 0.95
ANALOG_DEADBAND = 0.125
ANALOG_CENTER = 0.0

class DirectDeviceManager(VrpnClient, DirectObject):

    def __init__(self, server=None):
        if False:
            for i in range(10):
                print('nop')
        if server is not None:
            self.server = server
        else:
            self.server = ConfigVariableString('vrpn-server', 'spacedyne').getValue()
        VrpnClient.__init__(self, self.server)

    def createButtons(self, device):
        if False:
            print('Hello World!')
        return DirectButtons(self, device)

    def createAnalogs(self, device):
        if False:
            for i in range(10):
                print('nop')
        return DirectAnalogs(self, device)

    def createTracker(self, device):
        if False:
            i = 10
            return i + 15
        return DirectTracker(self, device)

    def createDials(self, device):
        if False:
            print('Hello World!')
        return DirectDials(self, device)

    def createTimecodeReader(self, device):
        if False:
            while True:
                i = 10
        return DirectTimecodeReader(self, device)

class DirectButtons(ButtonNode, DirectObject):
    buttonCount = 0

    def __init__(self, vrpnClient, device):
        if False:
            while True:
                i = 10
        DirectButtons.buttonCount += 1
        ButtonNode.__init__(self, vrpnClient, device)
        self.name = 'DirectButtons-' + repr(DirectButtons.buttonCount)
        try:
            self._base = base
        except NameError:
            self._base = simbase
        self.nodePath = self._base.dataRoot.attachNewNode(self)

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        if index < 0 or index >= self.getNumButtons():
            raise IndexError
        return self.getButtonState(index)

    def __len__(self):
        if False:
            print('Hello World!')
        return self.getNumButtons()

    def enable(self):
        if False:
            for i in range(10):
                print('nop')
        self.nodePath.reparentTo(self._base.dataRoot)

    def disable(self):
        if False:
            while True:
                i = 10
        self.nodePath.reparentTo(self._base.dataUnused)

    def getName(self):
        if False:
            while True:
                i = 10
        return self.name

    def getNodePath(self):
        if False:
            print('Hello World!')
        return self.nodePath

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        string = self.name + ': '
        for val in self:
            string = string + '%d' % val + ' '
        return string

class DirectAnalogs(AnalogNode, DirectObject):
    analogCount = 0
    _analogDeadband = ConfigVariableDouble('vrpn-analog-deadband', ANALOG_DEADBAND)
    _analogMin = ConfigVariableDouble('vrpn-analog-min', ANALOG_MIN)
    _analogMax = ConfigVariableDouble('vrpn-analog-max', ANALOG_MAX)
    _analogCenter = ConfigVariableDouble('vrpn-analog-center', ANALOG_CENTER)

    def __init__(self, vrpnClient, device):
        if False:
            while True:
                i = 10
        DirectAnalogs.analogCount += 1
        AnalogNode.__init__(self, vrpnClient, device)
        self.name = 'DirectAnalogs-' + repr(DirectAnalogs.analogCount)
        try:
            self._base = base
        except NameError:
            self._base = simbase
        self.nodePath = self._base.dataRoot.attachNewNode(self)
        self.analogDeadband = self._analogDeadband.getValue()
        self.analogMin = self._analogMin.getValue()
        self.analogMax = self._analogMax.getValue()
        self.analogCenter = self._analogCenter.getValue()
        self.analogRange = self.analogMax - self.analogMin

    def __getitem__(self, index):
        if False:
            return 10
        if index < 0 or index >= self.getNumControls():
            raise IndexError
        return self.getControlState(index)

    def __len__(self):
        if False:
            print('Hello World!')
        return self.getNumControls()

    def enable(self):
        if False:
            for i in range(10):
                print('nop')
        self.nodePath.reparentTo(self._base.dataRoot)

    def disable(self):
        if False:
            while True:
                i = 10
        self.nodePath.reparentTo(self._base.dataUnused)

    def normalizeWithoutCentering(self, val, minVal=-1, maxVal=1):
        if False:
            print('Hello World!')
        if val < 0:
            sign = -1
        else:
            sign = 1
        val = sign * max(abs(val) - self.analogDeadband, 0.0)
        val = min(max(val, self.analogMin), self.analogMax)
        return (maxVal - minVal) * ((val - self.analogMin) / float(self.analogRange)) + minVal

    def normalize(self, rawValue, minVal=-1, maxVal=1, sf=1.0):
        if False:
            for i in range(10):
                print('nop')
        aMax = self.analogMax
        aMin = self.analogMin
        center = self.analogCenter
        deadband = self.analogDeadband
        if abs(rawValue - center) <= deadband:
            return 0.0
        if rawValue >= center:
            val = min(rawValue * sf, aMax)
            percentVal = (val - (center + deadband)) / float(aMax - (center + deadband))
        else:
            val = max(rawValue * sf, aMin)
            percentVal = -((val - (center - deadband)) / float(aMin - (center - deadband)))
        return (maxVal - minVal) * ((percentVal + 1) / 2.0) + minVal

    def normalizeChannel(self, chan, minVal=-1, maxVal=1, sf=1.0):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self.normalize(self[chan], minVal, maxVal, sf)
        except IndexError:
            return 0.0

    def getName(self):
        if False:
            return 10
        return self.name

    def getNodePath(self):
        if False:
            while True:
                i = 10
        return self.nodePath

    def __repr__(self):
        if False:
            return 10
        string = self.name + ': '
        for val in self:
            string = string + '%.3f' % val + ' '
        return string

class DirectTracker(TrackerNode, DirectObject):
    trackerCount = 0

    def __init__(self, vrpnClient, device):
        if False:
            i = 10
            return i + 15
        DirectTracker.trackerCount += 1
        TrackerNode.__init__(self, vrpnClient, device)
        self.name = 'DirectTracker-' + repr(DirectTracker.trackerCount)
        try:
            self._base = base
        except NameError:
            self._base = simbase
        self.nodePath = self._base.dataRoot.attachNewNode(self)

    def enable(self):
        if False:
            while True:
                i = 10
        self.nodePath.reparentTo(self._base.dataRoot)

    def disable(self):
        if False:
            for i in range(10):
                print('nop')
        self.nodePath.reparentTo(self._base.dataUnused)

    def getName(self):
        if False:
            for i in range(10):
                print('nop')
        return self.name

    def getNodePath(self):
        if False:
            print('Hello World!')
        return self.nodePath

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return self.name

class DirectDials(DialNode, DirectObject):
    dialCount = 0

    def __init__(self, vrpnClient, device):
        if False:
            i = 10
            return i + 15
        DirectDials.dialCount += 1
        DialNode.__init__(self, vrpnClient, device)
        self.name = 'DirectDials-' + repr(DirectDials.dialCount)
        try:
            self._base = base
        except NameError:
            self._base = simbase
        self.nodePath = self._base.dataRoot.attachNewNode(self)

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        '\n        if (index < 0) or (index >= self.getNumDials()):\n            raise IndexError\n        '
        return self.readDial(index)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.getNumDials()

    def enable(self):
        if False:
            print('Hello World!')
        self.nodePath.reparentTo(self._base.dataRoot)

    def disable(self):
        if False:
            for i in range(10):
                print('nop')
        self.nodePath.reparentTo(self._base.dataUnused)

    def getName(self):
        if False:
            return 10
        return self.name

    def getNodePath(self):
        if False:
            i = 10
            return i + 15
        return self.nodePath

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        string = self.name + ': '
        for i in range(self.getNumDials()):
            string = string + '%.3f' % self[i] + ' '
        return string

class DirectTimecodeReader(AnalogNode, DirectObject):
    timecodeReaderCount = 0

    def __init__(self, vrpnClient, device):
        if False:
            i = 10
            return i + 15
        DirectTimecodeReader.timecodeReaderCount += 1
        AnalogNode.__init__(self, vrpnClient, device)
        self.name = 'DirectTimecodeReader-' + repr(DirectTimecodeReader.timecodeReaderCount)
        self.frames = 0
        self.seconds = 0
        self.minutes = 0
        self.hours = 0
        try:
            self._base = base
        except NameError:
            self._base = simbase
        self.nodePath = self._base.dataRoot.attachNewNode(self)

    def enable(self):
        if False:
            print('Hello World!')
        self.nodePath.reparentTo(self._base.dataRoot)

    def disable(self):
        if False:
            for i in range(10):
                print('nop')
        self.nodePath.reparentTo(self._base.dataUnused)

    def getName(self):
        if False:
            for i in range(10):
                print('nop')
        return self.name

    def getNodePath(self):
        if False:
            print('Hello World!')
        return self.nodePath

    def getTime(self):
        if False:
            return 10
        timeBits = int(self.getControlState(0))
        self.frames = (timeBits & 15) + ((timeBits & 240) >> 4) * 10
        self.seconds = ((timeBits & 3840) >> 8) + ((timeBits & 61440) >> 12) * 10
        self.minutes = ((timeBits & 983040) >> 16) + ((timeBits & 15728640) >> 20) * 10
        self.hours = ((timeBits & 251658240) >> 24) + ((timeBits & 4026531840) >> 28) * 10
        self.totalSeconds = self.hours * 3600 + self.minutes * 60 + self.seconds + self.frames / 30.0
        return (self.hours, self.minutes, self.seconds, self.frames, self.totalSeconds)

    def __repr__(self):
        if False:
            print('Hello World!')
        string = '%s: %d:%d:%d:%d' % ((self.name,) + self.getTime()[:-1])
        return string