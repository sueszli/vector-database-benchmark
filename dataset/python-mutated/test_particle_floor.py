from panda3d.core import NodePath, Vec3
from panda3d.physics import LinearVectorForce
from direct.particles import ParticleEffect
from direct.particles import Particles
from direct.particles import ForceGroup

class ParticleFloorTest(NodePath):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        NodePath.__init__(self, 'particleFloorTest')
        self.setDepthWrite(0)
        self.f = ParticleEffect.ParticleEffect()
        self.f.reparentTo(self)
        self.p0 = Particles.Particles('particles-1')
        self.p0.setFactory('PointParticleFactory')
        self.p0.setRenderer('PointParticleRenderer')
        self.p0.setEmitter('SphereVolumeEmitter')
        self.p0.setPoolSize(64)
        self.p0.setBirthRate(0.02)
        self.p0.setLitterSize(7)
        self.p0.setLitterSpread(2)
        self.p0.setSystemLifespan(0.0)
        self.p0.setFloorZ(-1.0)
        self.p0.setSystemGrowsOlderFlag(0)
        self.p0.factory.setLifespanBase(10.0)
        self.p0.factory.setLifespanSpread(0.5)
        self.p0.factory.setMassBase(1.8)
        self.p0.factory.setMassSpread(1.0)
        self.p0.factory.setTerminalVelocityBase(400.0)
        self.p0.factory.setTerminalVelocitySpread(0.0)
        self.f.addParticles(self.p0)
        f0 = ForceGroup.ForceGroup('frict')
        force0 = LinearVectorForce(Vec3(0.0, 0.0, -1.0))
        force0.setActive(1)
        f0.addForce(force0)
        self.f.addForceGroup(f0)

    def start(self):
        if False:
            i = 10
            return i + 15
        self.f.enable()

def test_ParticleFloorTest(base):
    if False:
        return 10
    base.enableParticles()
    pt = ParticleFloorTest()
    pt.reparentTo(base.render)
    pt.start()