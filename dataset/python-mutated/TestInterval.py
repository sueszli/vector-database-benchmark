"""
Contains the TestInterval class
"""
__all__ = ['TestInterval']
from panda3d.direct import CInterval
from direct.directnotify.DirectNotifyGlobal import directNotify
from .Interval import Interval

class TestInterval(Interval):
    particleNum = 1
    notify = directNotify.newCategory('TestInterval')

    def __init__(self, particleEffect, duration=0.0, parent=None, renderParent=None, name=None):
        if False:
            return 10
        '\n        particleEffect is ??\n        parent is ??\n        worldRelative is a boolean\n        loop is a boolean\n        duration is a float for the time\n        name is ??\n        '
        id = 'Particle-%d' % TestInterval.particleNum
        TestInterval.particleNum += 1
        if name is None:
            name = id
        self.particleEffect = particleEffect
        self.parent = parent
        self.renderParent = renderParent
        Interval.__init__(self, name, duration)

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def __step(self, dt):
        if False:
            while True:
                i = 10
        self.particleEffect.accelerate(dt, 1, 0.05)

    def start(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.particleEffect.clearToInitial()
        self.currT = 0
        Interval.start(self, *args, **kwargs)

    def privInitialize(self, t):
        if False:
            while True:
                i = 10
        if self.parent is not None:
            self.particleEffect.reparentTo(self.parent)
        if self.renderParent is not None:
            self.setRenderParent(self.renderParent.node())
        self.state = CInterval.SStarted
        for f in self.particleEffect.forceGroupDict.values():
            f.enable()
        self.__step(t - self.currT)
        self.currT = t

    def privStep(self, t):
        if False:
            return 10
        if self.state == CInterval.SPaused:
            self.privInitialize(t)
        else:
            self.state = CInterval.SStarted
            self.__step(t - self.currT)
            self.currT = t

    def privFinalize(self):
        if False:
            while True:
                i = 10
        self.__step(self.getDuration() - self.currT)
        self.currT = self.getDuration()
        self.state = CInterval.SFinal

    def privInstant(self):
        if False:
            print('Hello World!')
        '\n        Full jump from Initial state to Final State\n        '
        self.__step(self.getDuration() - self.currT)
        self.currT = self.getDuration()
        self.state = CInterval.SFinal

    def privInterrupt(self):
        if False:
            print('Hello World!')
        if not self.isStopped():
            self.state = CInterval.SPaused