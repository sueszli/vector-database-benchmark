"""
Contains the ParticleInterval class
"""
__all__ = ['ParticleInterval']
from panda3d.direct import CInterval
from direct.directnotify.DirectNotifyGlobal import directNotify
from .Interval import Interval

class ParticleInterval(Interval):
    """
    Use this interval when you want to have greater control over a
    ParticleEffect.  The interval does not register the effect with
    the global particle and physics managers, but it does call upon
    them to perform its stepping.  You should NOT call
    particleEffect.start() with an effect that is being controlled
    by a ParticleInterval.
    """
    particleNum = 1
    notify = directNotify.newCategory('ParticleInterval')

    def __init__(self, particleEffect, parent, worldRelative=1, renderParent=None, duration=0.0, softStopT=0.0, cleanup=False, name=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Args:\n            particleEffect (ParticleEffect): a particle effect\n            parent (NodePath): this is where the effect will be parented in the\n                scene graph\n            worldRelative (bool): this will override 'renderParent' with render\n            renderParent (NodePath): this is where the particles will be\n                rendered in the scenegraph\n            duration (float): for the time\n            softStopT (float): no effect if 0.0, a positive value will count\n                from the start of the interval, a negative value will count\n                from the end of the interval\n            cleanup (boolean): if True the effect will be destroyed and removed\n                from the scenegraph upon interval completion.  Set to False if\n                planning on reusing the interval.\n            name (string): use this for unique intervals so that they can be\n                easily found in the taskMgr.\n        "
        id = 'Particle-%d' % ParticleInterval.particleNum
        ParticleInterval.particleNum += 1
        if name is None:
            name = id
        self.particleEffect = particleEffect
        self.cleanup = cleanup
        if parent is not None:
            self.particleEffect.reparentTo(parent)
        if worldRelative:
            renderParent = render
        if renderParent:
            for particles in self.particleEffect.getParticlesList():
                particles.setRenderParent(renderParent.node())
        self.__softStopped = False
        if softStopT == 0.0:
            self.softStopT = duration
        elif softStopT < 0.0:
            self.softStopT = duration + softStopT
        else:
            self.softStopT = softStopT
        Interval.__init__(self, name, duration)

    def __step(self, dt):
        if False:
            while True:
                i = 10
        if self.particleEffect:
            self.particleEffect.accelerate(dt, 1, 0.05)

    def __softStart(self):
        if False:
            while True:
                i = 10
        if self.particleEffect:
            self.particleEffect.softStart()
        self.__softStopped = False

    def __softStop(self):
        if False:
            return 10
        if self.particleEffect:
            self.particleEffect.softStop()
        self.__softStopped = True

    def privInitialize(self, t):
        if False:
            while True:
                i = 10
        if self.state != CInterval.SPaused:
            self.__softStart()
            if self.particleEffect:
                self.particleEffect.clearToInitial()
            self.currT = 0
        if self.particleEffect:
            for forceGroup in self.particleEffect.getForceGroupList():
                forceGroup.enable()
        Interval.privInitialize(self, t)

    def privInstant(self):
        if False:
            for i in range(10):
                print('nop')
        self.privInitialize(self.getDuration())
        self.privFinalize()

    def privStep(self, t):
        if False:
            return 10
        if self.state == CInterval.SPaused or t < self.currT:
            self.privInitialize(t)
        else:
            if not self.__softStopped and t > self.softStopT:
                self.__step(self.softStopT - self.currT)
                self.__softStop()
                self.__step(t - self.softStopT)
            else:
                self.__step(t - self.currT)
            Interval.privStep(self, t)

    def privFinalize(self):
        if False:
            for i in range(10):
                print('nop')
        Interval.privFinalize(self)
        if self.cleanup and self.particleEffect:
            self.particleEffect.cleanup()
            self.particleEffect = None