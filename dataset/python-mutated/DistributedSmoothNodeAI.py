from . import DistributedNodeAI
from . import DistributedSmoothNodeBase

class DistributedSmoothNodeAI(DistributedNodeAI.DistributedNodeAI, DistributedSmoothNodeBase.DistributedSmoothNodeBase):

    def __init__(self, air, name=None):
        if False:
            i = 10
            return i + 15
        DistributedNodeAI.DistributedNodeAI.__init__(self, air, name)
        DistributedSmoothNodeBase.DistributedSmoothNodeBase.__init__(self)

    def generate(self):
        if False:
            while True:
                i = 10
        DistributedNodeAI.DistributedNodeAI.generate(self)
        DistributedSmoothNodeBase.DistributedSmoothNodeBase.generate(self)
        self.cnode.setRepository(self.air, 1, self.air.ourChannel)

    def disable(self):
        if False:
            for i in range(10):
                print('nop')
        DistributedSmoothNodeBase.DistributedSmoothNodeBase.disable(self)
        DistributedNodeAI.DistributedNodeAI.disable(self)

    def delete(self):
        if False:
            return 10
        DistributedSmoothNodeBase.DistributedSmoothNodeBase.delete(self)
        DistributedNodeAI.DistributedNodeAI.delete(self)

    def setSmStop(self, t=None):
        if False:
            print('Hello World!')
        pass

    def setSmH(self, h, t=None):
        if False:
            return 10
        self.setH(h)

    def setSmZ(self, z, t=None):
        if False:
            return 10
        self.setZ(z)

    def setSmXY(self, x, y, t=None):
        if False:
            i = 10
            return i + 15
        self.setX(x)
        self.setY(y)

    def setSmXZ(self, x, z, t=None):
        if False:
            for i in range(10):
                print('nop')
        self.setX(x)
        self.setZ(z)

    def setSmPos(self, x, y, z, t=None):
        if False:
            i = 10
            return i + 15
        self.setPos(x, y, z)

    def setSmHpr(self, h, p, r, t=None):
        if False:
            print('Hello World!')
        self.setHpr(h, p, r)

    def setSmXYH(self, x, y, h, t=None):
        if False:
            for i in range(10):
                print('nop')
        self.setX(x)
        self.setY(y)
        self.setH(h)

    def setSmXYZH(self, x, y, z, h, t=None):
        if False:
            return 10
        self.setPos(x, y, z)
        self.setH(h)

    def setSmPosHpr(self, x, y, z, h, p, r, t=None):
        if False:
            print('Hello World!')
        self.setPosHpr(x, y, z, h, p, r)

    def setSmPosHprL(self, l, x, y, z, h, p, r, t=None):
        if False:
            i = 10
            return i + 15
        self.setPosHpr(x, y, z, h, p, r)

    def clearSmoothing(self, bogus=None):
        if False:
            while True:
                i = 10
        pass

    def setComponentX(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.setX(x)

    def setComponentY(self, y):
        if False:
            i = 10
            return i + 15
        self.setY(y)

    def setComponentZ(self, z):
        if False:
            return 10
        self.setZ(z)

    def setComponentH(self, h):
        if False:
            return 10
        self.setH(h)

    def setComponentP(self, p):
        if False:
            return 10
        self.setP(p)

    def setComponentR(self, r):
        if False:
            print('Hello World!')
        self.setR(r)

    def setComponentL(self, l):
        if False:
            i = 10
            return i + 15
        pass

    def setComponentT(self, t):
        if False:
            print('Hello World!')
        pass

    def getComponentX(self):
        if False:
            print('Hello World!')
        return self.getX()

    def getComponentY(self):
        if False:
            while True:
                i = 10
        return self.getY()

    def getComponentZ(self):
        if False:
            return 10
        return self.getZ()

    def getComponentH(self):
        if False:
            i = 10
            return i + 15
        return self.getH()

    def getComponentP(self):
        if False:
            for i in range(10):
                print('nop')
        return self.getP()

    def getComponentR(self):
        if False:
            while True:
                i = 10
        return self.getR()

    def getComponentL(self):
        if False:
            while True:
                i = 10
        if self.zoneId:
            return self.zoneId
        else:
            return 0

    def getComponentT(self):
        if False:
            print('Hello World!')
        return 0