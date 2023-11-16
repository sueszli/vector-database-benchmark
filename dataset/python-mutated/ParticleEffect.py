from panda3d.core import Filename, NodePath, VirtualFileSystem, getModelPath
from panda3d.core import *
from panda3d.physics import *
from . import Particles
from . import ForceGroup
from direct.directnotify import DirectNotifyGlobal

class ParticleEffect(NodePath):
    notify = DirectNotifyGlobal.directNotify.newCategory('ParticleEffect')
    pid = 1

    def __init__(self, name=None, particles=None):
        if False:
            return 10
        if name is None:
            name = 'particle-effect-%d' % ParticleEffect.pid
            ParticleEffect.pid += 1
        NodePath.__init__(self, name)
        self.name = name
        self.fEnabled = 0
        self.particlesDict = {}
        self.forceGroupDict = {}
        if particles is not None:
            self.addParticles(particles)
        self.renderParent = None

    def birthLitter(self):
        if False:
            while True:
                i = 10
        for p in self.particlesDict.values():
            p.birthLitter()

    def cleanup(self):
        if False:
            i = 10
            return i + 15
        self.removeNode()
        self.disable()
        if self.__isValid():
            for f in self.forceGroupDict.values():
                f.cleanup()
            for p in self.particlesDict.values():
                p.cleanup()
            del self.forceGroupDict
            del self.particlesDict
        del self.renderParent

    def getName(self):
        if False:
            while True:
                i = 10
        return self.name

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        self.removeAllForces()
        self.removeAllParticles()
        self.forceGroupDict = {}
        self.particlesDict = {}

    def start(self, parent=None, renderParent=None):
        if False:
            return 10
        assert self.notify.debug('start() - name: %s' % self.name)
        self.renderParent = renderParent
        self.enable()
        if parent is not None:
            self.reparentTo(parent)

    def enable(self):
        if False:
            while True:
                i = 10
        if self.__isValid():
            if self.renderParent:
                for p in self.particlesDict.values():
                    p.setRenderParent(self.renderParent.node())
            for f in self.forceGroupDict.values():
                f.enable()
            for p in self.particlesDict.values():
                p.enable()
            self.fEnabled = 1

    def disable(self):
        if False:
            for i in range(10):
                print('nop')
        self.detachNode()
        if self.__isValid():
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
        '\n        Note: this may be misleading if enable(), disable() not used\n        '
        return self.fEnabled

    def addForceGroup(self, forceGroup):
        if False:
            return 10
        forceGroup.nodePath.reparentTo(self)
        forceGroup.particleEffect = self
        self.forceGroupDict[forceGroup.name] = forceGroup
        for force in forceGroup:
            self.addForce(force)

    def addForce(self, force):
        if False:
            print('Hello World!')
        for p in list(self.particlesDict.values()):
            p.addForce(force)

    def removeForceGroup(self, forceGroup):
        if False:
            i = 10
            return i + 15
        for force in forceGroup:
            self.removeForce(force)
        forceGroup.nodePath.removeNode()
        forceGroup.particleEffect = None
        self.forceGroupDict.pop(forceGroup.getName(), None)

    def removeForce(self, force):
        if False:
            print('Hello World!')
        for p in list(self.particlesDict.values()):
            p.removeForce(force)

    def removeAllForces(self):
        if False:
            for i in range(10):
                print('nop')
        for fg in list(self.forceGroupDict.values()):
            self.removeForceGroup(fg)

    def addParticles(self, particles):
        if False:
            return 10
        particles.nodePath.reparentTo(self)
        self.particlesDict[particles.getName()] = particles
        for fg in list(self.forceGroupDict.values()):
            for force in fg:
                particles.addForce(force)

    def removeParticles(self, particles):
        if False:
            return 10
        if particles is None:
            self.notify.warning('removeParticles() - particles is None!')
            return
        particles.nodePath.detachNode()
        self.particlesDict.pop(particles.getName(), None)
        for fg in list(self.forceGroupDict.values()):
            for f in fg:
                particles.removeForce(f)

    def removeAllParticles(self):
        if False:
            print('Hello World!')
        for p in list(self.particlesDict.values()):
            self.removeParticles(p)

    def getParticlesList(self):
        if False:
            for i in range(10):
                print('nop')
        return list(self.particlesDict.values())

    def getParticlesNamed(self, name):
        if False:
            print('Hello World!')
        return self.particlesDict.get(name, None)

    def getParticlesDict(self):
        if False:
            return 10
        return self.particlesDict

    def getForceGroupList(self):
        if False:
            print('Hello World!')
        return list(self.forceGroupDict.values())

    def getForceGroupNamed(self, name):
        if False:
            print('Hello World!')
        return self.forceGroupDict.get(name, None)

    def getForceGroupDict(self):
        if False:
            while True:
                i = 10
        return self.forceGroupDict

    def saveConfig(self, filename):
        if False:
            return 10
        filename = Filename(filename)
        with open(filename.toOsSpecific(), 'w') as f:
            f.write('\n')
            f.write('self.reset()\n')
            pos = self.getPos()
            hpr = self.getHpr()
            scale = self.getScale()
            f.write('self.setPos(%0.3f, %0.3f, %0.3f)\n' % (pos[0], pos[1], pos[2]))
            f.write('self.setHpr(%0.3f, %0.3f, %0.3f)\n' % (hpr[0], hpr[1], hpr[2]))
            f.write('self.setScale(%0.3f, %0.3f, %0.3f)\n' % (scale[0], scale[1], scale[2]))
            num = 0
            for p in list(self.particlesDict.values()):
                target = 'p%d' % num
                num = num + 1
                f.write(target + " = Particles.Particles('%s')\n" % p.getName())
                p.printParams(f, target)
                f.write('self.addParticles(%s)\n' % target)
            num = 0
            for fg in list(self.forceGroupDict.values()):
                target = 'f%d' % num
                num = num + 1
                f.write(target + " = ForceGroup.ForceGroup('%s')\n" % fg.getName())
                fg.printParams(f, target)
                f.write('self.addForceGroup(%s)\n' % target)

    def loadConfig(self, filename):
        if False:
            i = 10
            return i + 15
        fn = Filename(filename)
        vfs = VirtualFileSystem.getGlobalPtr()
        try:
            if not vfs.resolveFilename(fn, getModelPath().value) and (not fn.isRegularFile()):
                raise FileNotFoundError('could not find particle file: %s' % filename)
            data = vfs.readFile(fn, True)
            data = data.replace(b'\r', b'')
            exec(data)
        except Exception:
            self.notify.warning('loadConfig: failed to load particle file: ' + repr(filename))
            raise

    def accelerate(self, time, stepCount=1, stepTime=0.0):
        if False:
            print('Hello World!')
        for particles in self.getParticlesList():
            particles.accelerate(time, stepCount, stepTime)

    def clearToInitial(self):
        if False:
            for i in range(10):
                print('nop')
        for particles in self.getParticlesList():
            particles.clearToInitial()

    def softStop(self):
        if False:
            print('Hello World!')
        for particles in self.getParticlesList():
            particles.softStop()

    def softStart(self, firstBirthDelay=None):
        if False:
            print('Hello World!')
        if self.__isValid():
            for particles in self.getParticlesList():
                if firstBirthDelay is not None:
                    particles.softStart(br=-1, first_birth_delay=firstBirthDelay)
                else:
                    particles.softStart()
        else:
            self.notify.error('Trying to start effect(%s) after cleanup.' % (self.getName(),))

    def __isValid(self):
        if False:
            for i in range(10):
                print('nop')
        return hasattr(self, 'forceGroupDict') and hasattr(self, 'particlesDict')
    is_enabled = isEnabled
    add_force_group = addForceGroup
    add_force = addForce
    remove_force_group = removeForceGroup
    remove_force = removeForce
    remove_all_forces = removeAllForces
    add_particles = addParticles
    remove_particles = removeParticles
    remove_all_particles = removeAllParticles
    get_particles_list = getParticlesList
    get_particles_named = getParticlesNamed
    get_particles_dict = getParticlesDict
    get_force_group_list = getForceGroupList
    get_force_group_named = getForceGroupNamed
    get_force_group_dict = getForceGroupDict
    save_config = saveConfig
    load_config = loadConfig
    clear_to_initial = clearToInitial
    soft_stop = softStop
    soft_start = softStart
    birth_litter = birthLitter