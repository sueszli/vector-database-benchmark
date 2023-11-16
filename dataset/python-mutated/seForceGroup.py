from panda3d.core import *
from direct.showbase.DirectObject import DirectObject
from direct.showbase.PhysicsManagerGlobal import *
from direct.directnotify import DirectNotifyGlobal
import sys

class ForceGroup(DirectObject):
    notify = DirectNotifyGlobal.directNotify.newCategory('ForceGroup')
    id = 1

    def __init__(self, name=None):
        if False:
            return 10
        '__init__(self)'
        if name == None:
            self.name = 'ForceGroup-%d' % ForceGroup.id
            ForceGroup.id += 1
        else:
            self.name = name
        self.node = ForceNode.ForceNode(self.name)
        self.nodePath = NodePath(self.node)
        self.fEnabled = 0
        self.particleEffect = None

    def cleanup(self):
        if False:
            print('Hello World!')
        self.node.clear()
        self.nodePath.removeNode()
        del self.nodePath
        del self.node
        del self.particleEffect

    def enable(self):
        if False:
            return 10
        '\n        Convenience function to enable all forces in force group\n        '
        for i in range(self.node.getNumForces()):
            f = self.node.getForce(i)
            f.setActive(1)
        self.fEnabled = 1

    def disable(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convenience function to disable all forces in force group\n        '
        for i in range(self.node.getNumForces()):
            f = self.node.getForce(i)
            f.setActive(0)
        self.fEnabled = 0

    def isEnabled(self):
        if False:
            i = 10
            return i + 15
        return self.fEnabled

    def addForce(self, force):
        if False:
            while True:
                i = 10
        'addForce(self, force)'
        self.node.addForce(force)
        if self.particleEffect:
            self.particleEffect.addForce(force)

    def removeForce(self, force):
        if False:
            return 10
        'removeForce(self, force)'
        self.node.removeForce(force)
        if self.particleEffect != None:
            self.particleEffect.removeForce(force)

    def getName(self):
        if False:
            for i in range(10):
                print('nop')
        return self.name

    def getNode(self):
        if False:
            for i in range(10):
                print('nop')
        return self.node

    def getNodePath(self):
        if False:
            i = 10
            return i + 15
        return self.nodePath

    def __getitem__(self, index):
        if False:
            print('Hello World!')
        numForces = self.node.getNumForces()
        if index < 0 or index >= numForces:
            raise IndexError
        return self.node.getForce(index)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return self.node.getNumForces()

    def asList(self):
        if False:
            i = 10
            return i + 15
        l = []
        for i in range(self.node.getNumForces()):
            l.append(self.node.getForce(i))
        return l

    def printParams(self, file=sys.stdout, targ='self'):
        if False:
            return 10
        i1 = '    '
        i2 = i1 + i1
        file.write(i2 + '# Force parameters\n')
        for i in range(self.node.getNumForces()):
            f = self.node.getForce(i)
            fname = 'force%d' % i
            if isinstance(f, LinearForce):
                amplitude = f.getAmplitude()
                massDependent = f.getMassDependent()
                if isinstance(f, LinearCylinderVortexForce):
                    file.write(i2 + fname + ' = LinearCylinderVortexForce(%.4f, %.4f, %.4f, %.4f, %d)\n' % (f.getRadius(), f.getLength(), f.getCoef(), amplitude, massDependent))
                elif isinstance(f, LinearDistanceForce):
                    radius = f.getRadius()
                    falloffType = f.getFalloffType()
                    ftype = 'FTONEOVERR'
                    if falloffType == LinearDistanceForce.FTONEOVERR:
                        ftype = 'FTONEOVERR'
                    elif falloffType == LinearDistanceForce.FTONEOVERRSQUARED:
                        ftype = 'FTONEOVERRSQUARED'
                    elif falloffType == LinearDistanceForce.FTONEOVERRCUBED:
                        ftype = 'FTONEOVERRCUBED'
                    forceCenter = f.getForceCenter()
                    if isinstance(f, LinearSinkForce):
                        file.write(i2 + fname + ' = LinearSinkForce(Point3(%.4f, %.4f, %.4f), LinearDistanceForce.%s, %.4f, %.4f, %d)\n' % (forceCenter[0], forceCenter[1], forceCenter[2], ftype, radius, amplitude, massDependent))
                    elif isinstance(f, LinearSourceForce):
                        file.write(i2 + fname + ' = LinearSourceForce(Point3(%.4f, %.4f, %.4f), LinearDistanceForce.%s, %.4f, %.4f, %d)\n' % (forceCenter[0], forceCenter[1], forceCenter[2], ftype, radius, amplitude, massDependent))
                elif isinstance(f, LinearFrictionForce):
                    file.write(i2 + fname + ' = LinearFrictionForce(%.4f, %.4f, %d)\n' % (f.getCoef(), amplitude, massDependent))
                elif isinstance(f, LinearJitterForce):
                    file.write(i2 + fname + ' = LinearJitterForce(%.4f, %d)\n' % (amplitude, massDependent))
                elif isinstance(f, LinearNoiseForce):
                    file.write(i2 + fname + ' = LinearNoiseForce(%.4f, %d)\n' % (amplitude, massDependent))
                elif isinstance(f, LinearVectorForce):
                    vec = f.getLocalVector()
                    file.write(i2 + fname + ' = LinearVectorForce(Vec3(%.4f, %.4f, %.4f), %.4f, %d)\n' % (vec[0], vec[1], vec[2], amplitude, massDependent))
            elif isinstance(f, AngularForce):
                if isinstance(f, AngularVectorForce):
                    vec = f.getQuat()
                    file.write(i2 + fname + ' = AngularVectorForce(Quat(%.4f, %.4f, %.4f))\n' % (vec[0], vec[1], vec[2], vec[3]))
            file.write(i2 + fname + '.setActive(%d)\n' % f.getActive())
            file.write(i2 + targ + '.addForce(%s)\n' % fname)