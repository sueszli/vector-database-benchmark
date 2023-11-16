from panda3d.core import *
import seParticles
import seForceGroup
from direct.directnotify import DirectNotifyGlobal

class ParticleEffect(NodePath):
    notify = DirectNotifyGlobal.directNotify.newCategory('ParticleEffect')
    pid = 1

    def __init__(self, name=None, particles=None):
        if False:
            for i in range(10):
                print('nop')
        '__init__()'
        if name == None:
            name = 'particle-effect-%d' % ParticleEffect.pid
            ParticleEffect.pid += 1
        NodePath.__init__(self, name)
        self.name = name
        self.fEnabled = 0
        self.particlesDict = {}
        self.forceGroupDict = {}
        if particles != None:
            self.addParticles(particles)
        self.renderParent = None

    def start(self, parent=None, renderParent=None):
        if False:
            for i in range(10):
                print('nop')
        assert self.notify.debug('start() - name: %s' % self.name)
        self.renderParent = renderParent
        self.enable()
        if parent != None:
            self.reparentTo(parent)

    def cleanup(self):
        if False:
            print('Hello World!')
        self.removeNode()
        self.disable()
        for f in self.forceGroupDict.values():
            f.cleanup()
        for p in self.particlesDict.values():
            p.cleanup()
        del self.renderParent
        del self.particlesDict
        del self.forceGroupDict

    def reset(self):
        if False:
            return 10
        self.removeAllForces()
        self.removeAllParticles()
        self.forceGroupDict = {}
        self.particlesDict = {}

    def enable(self):
        if False:
            return 10
        'enable()'
        if self.renderParent != None:
            for p in self.particlesDict.values():
                p.setRenderParent(self.renderParent.node())
        for f in self.forceGroupDict.values():
            f.enable()
        for p in self.particlesDict.values():
            p.enable()
        self.fEnabled = 1

    def disable(self):
        if False:
            print('Hello World!')
        'disable()'
        self.detachNode()
        for p in self.particlesDict.values():
            p.setRenderParent(p.node)
        for f in self.forceGroupDict.values():
            f.disable()
        for p in self.particlesDict.values():
            p.disable()
        self.fEnabled = 0

    def isEnabled(self):
        if False:
            print('Hello World!')
        '\n        isEnabled()\n        Note: this may be misleading if enable(),disable() not used\n        '
        return self.fEnabled

    def addForceGroup(self, forceGroup):
        if False:
            return 10
        'addForceGroup(forceGroup)'
        forceGroup.nodePath.reparentTo(self)
        forceGroup.particleEffect = self
        self.forceGroupDict[forceGroup.getName()] = forceGroup
        for i in range(len(forceGroup)):
            self.addForce(forceGroup[i])

    def addForce(self, force):
        if False:
            while True:
                i = 10
        'addForce(force)'
        for p in self.particlesDict.values():
            p.addForce(force)

    def removeForceGroup(self, forceGroup):
        if False:
            i = 10
            return i + 15
        'removeForceGroup(forceGroup)'
        for i in range(len(forceGroup)):
            self.removeForce(forceGroup[i])
        forceGroup.nodePath.removeNode()
        forceGroup.particleEffect = None
        del self.forceGroupDict[forceGroup.getName()]

    def removeForce(self, force):
        if False:
            while True:
                i = 10
        'removeForce(force)'
        for p in self.particlesDict.values():
            p.removeForce(force)

    def removeAllForces(self):
        if False:
            while True:
                i = 10
        for fg in self.forceGroupDict.values():
            self.removeForceGroup(fg)

    def addParticles(self, particles):
        if False:
            i = 10
            return i + 15
        'addParticles(particles)'
        particles.nodePath.reparentTo(self)
        self.particlesDict[particles.getName()] = particles
        for fg in self.forceGroupDict.values():
            for i in range(len(fg)):
                particles.addForce(fg[i])

    def removeParticles(self, particles):
        if False:
            for i in range(10):
                print('nop')
        'removeParticles(particles)'
        if particles == None:
            self.notify.warning('removeParticles() - particles == None!')
            return
        particles.nodePath.detachNode()
        del self.particlesDict[particles.getName()]
        for fg in self.forceGroupDict.values():
            for f in fg.asList():
                particles.removeForce(f)

    def removeAllParticles(self):
        if False:
            while True:
                i = 10
        for p in self.particlesDict.values():
            self.removeParticles(p)

    def getParticlesList(self):
        if False:
            i = 10
            return i + 15
        'getParticles()'
        return self.particlesDict.values()

    def getParticlesNamed(self, name):
        if False:
            return 10
        'getParticlesNamed(name)'
        return self.particlesDict.get(name, None)

    def getParticlesDict(self):
        if False:
            while True:
                i = 10
        'getParticlesDict()'
        return self.particlesDict

    def getForceGroupList(self):
        if False:
            print('Hello World!')
        'getForceGroup()'
        return self.forceGroupDict.values()

    def getForceGroupNamed(self, name):
        if False:
            i = 10
            return i + 15
        'getForceGroupNamed(name)'
        return self.forceGroupDict.get(name, None)

    def getForceGroupDict(self):
        if False:
            for i in range(10):
                print('nop')
        'getForceGroup()'
        return self.forceGroupDict

    def saveConfig(self, filename):
        if False:
            for i in range(10):
                print('nop')
        'saveFileData(filename)'
        f = open(filename.toOsSpecific(), 'wb')
        f.write('\n')
        f.write('self.reset()\n')
        pos = self.getPos()
        hpr = self.getHpr()
        scale = self.getScale()
        f.write('self.setPos(%0.3f, %0.3f, %0.3f)\n' % (pos[0], pos[1], pos[2]))
        f.write('self.setHpr(%0.3f, %0.3f, %0.3f)\n' % (hpr[0], hpr[1], hpr[2]))
        f.write('self.setScale(%0.3f, %0.3f, %0.3f)\n' % (scale[0], scale[1], scale[2]))
        num = 0
        for p in self.particlesDict.values():
            target = 'p%d' % num
            num = num + 1
            f.write(target + " = Particles.Particles('%s')\n" % p.getName())
            p.printParams(f, target)
            f.write('self.addParticles(%s)\n' % target)
        num = 0
        for fg in self.forceGroupDict.values():
            target = 'f%d' % num
            num = num + 1
            f.write(target + " = ForceGroup.ForceGroup('%s')\n" % fg.getName())
            fg.printParams(f, target)
            f.write('self.addForceGroup(%s)\n' % target)
        f.close()

    def loadConfig(self, filename):
        if False:
            return 10
        'loadConfig(filename)'
        print(vfs.readFile(filename))
        exec(vfs.readFile(filename))
        print('Particle Effect Reading using VFS')

    def AppendConfig(self, f):
        if False:
            i = 10
            return i + 15
        f.write('\n')
        i1 = '    '
        i2 = i1 + i1
        f.write(i2 + 'self.effect.reset()\n')
        pos = self.getPos()
        hpr = self.getHpr()
        scale = self.getScale()
        f.write(i2 + 'self.effect.setPos(%0.3f, %0.3f, %0.3f)\n' % (pos[0], pos[1], pos[2]))
        f.write(i2 + 'self.effect.setHpr(%0.3f, %0.3f, %0.3f)\n' % (hpr[0], hpr[1], hpr[2]))
        f.write(i2 + 'self.effect.setScale(%0.3f, %0.3f, %0.3f)\n' % (scale[0], scale[1], scale[2]))
        num = 0
        for p in self.particlesDict.values():
            target = 'p%d' % num
            num = num + 1
            f.write(i2 + 'if(mode==0):\n')
            f.write(i2 + i1 + target + " = seParticles.Particles('%s')\n" % p.getName())
            f.write(i2 + 'else:\n')
            f.write(i2 + i1 + target + " = Particles.Particles('%s')\n" % p.getName())
            p.printParams(f, target)
            f.write(i2 + 'self.effect.addParticles(%s)\n' % target)
        num = 0
        for fg in self.forceGroupDict.values():
            target = 'f%d' % num
            num = num + 1
            f.write(i2 + target + " = ForceGroup.ForceGroup('%s')\n" % fg.getName())
            fg.printParams(f, target)
            f.write(i2 + 'self.effect.addForceGroup(%s)\n' % target)