from .DistributedObjectUD import DistributedObjectUD

class DistributedNodeUD(DistributedObjectUD):

    def __init__(self, air, name=None):
        if False:
            i = 10
            return i + 15
        if not hasattr(self, 'DistributedNodeUD_initialized'):
            self.DistributedNodeUD_initialized = 1
            DistributedObjectUD.__init__(self, air)
            if name is None:
                name = self.__class__.__name__

    def b_setParent(self, parentToken):
        if False:
            i = 10
            return i + 15
        if isinstance(parentToken, str):
            self.setParentStr(parentToken)
        else:
            self.setParent(parentToken)
        self.d_setParent(parentToken)

    def d_setParent(self, parentToken):
        if False:
            i = 10
            return i + 15
        if isinstance(parentToken, str):
            self.sendUpdate('setParentStr', [parentToken])
        else:
            self.sendUpdate('setParent', [parentToken])

    def setParentStr(self, parentToken):
        if False:
            print('Hello World!')
        self.notify.debugCall()
        if len(parentToken) > 0:
            self.do_setParent(parentToken)

    def setParent(self, parentToken):
        if False:
            i = 10
            return i + 15
        self.notify.debugCall()
        self.do_setParent(parentToken)

    def do_setParent(self, parentToken):
        if False:
            i = 10
            return i + 15
        self.getParentMgr().requestReparent(self, parentToken)

    def d_setX(self, x):
        if False:
            return 10
        self.sendUpdate('setX', [x])

    def d_setY(self, y):
        if False:
            while True:
                i = 10
        self.sendUpdate('setY', [y])

    def d_setZ(self, z):
        if False:
            return 10
        self.sendUpdate('setZ', [z])

    def d_setH(self, h):
        if False:
            return 10
        self.sendUpdate('setH', [h])

    def d_setP(self, p):
        if False:
            i = 10
            return i + 15
        self.sendUpdate('setP', [p])

    def d_setR(self, r):
        if False:
            print('Hello World!')
        self.sendUpdate('setR', [r])

    def setXY(self, x, y):
        if False:
            i = 10
            return i + 15
        self.setX(x)
        self.setY(y)

    def d_setXY(self, x, y):
        if False:
            while True:
                i = 10
        self.sendUpdate('setXY', [x, y])

    def d_setPos(self, x, y, z):
        if False:
            for i in range(10):
                print('nop')
        self.sendUpdate('setPos', [x, y, z])

    def d_setHpr(self, h, p, r):
        if False:
            print('Hello World!')
        self.sendUpdate('setHpr', [h, p, r])

    def setXYH(self, x, y, h):
        if False:
            while True:
                i = 10
        self.setX(x)
        self.setY(y)
        self.setH(h)

    def d_setXYH(self, x, y, h):
        if False:
            for i in range(10):
                print('nop')
        self.sendUpdate('setXYH', [x, y, h])

    def setXYZH(self, x, y, z, h):
        if False:
            for i in range(10):
                print('nop')
        self.setPos(x, y, z)
        self.setH(h)

    def d_setXYZH(self, x, y, z, h):
        if False:
            return 10
        self.sendUpdate('setXYZH', [x, y, z, h])

    def d_setPosHpr(self, x, y, z, h, p, r):
        if False:
            print('Hello World!')
        self.sendUpdate('setPosHpr', [x, y, z, h, p, r])