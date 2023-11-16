from panda3d.core import NodePath
from . import DistributedObjectAI
from . import GridParent

class DistributedNodeAI(DistributedObjectAI.DistributedObjectAI, NodePath):

    def __init__(self, air, name=None):
        if False:
            while True:
                i = 10
        if not hasattr(self, 'DistributedNodeAI_initialized'):
            self.DistributedNodeAI_initialized = 1
            DistributedObjectAI.DistributedObjectAI.__init__(self, air)
            if name is None:
                name = self.__class__.__name__
            NodePath.__init__(self, name)
            self.gridParent = None

    def delete(self):
        if False:
            while True:
                i = 10
        if self.gridParent:
            self.gridParent.delete()
            self.gridParent = None
        if not self.isEmpty():
            self.removeNode()
        DistributedObjectAI.DistributedObjectAI.delete(self)

    def setLocation(self, parentId, zoneId, teleport=0):
        if False:
            return 10
        DistributedObjectAI.DistributedObjectAI.setLocation(self, parentId, zoneId)
        parentObj = self.air.doId2do.get(parentId)
        if parentObj:
            if parentObj.isGridParent():
                if not self.gridParent:
                    self.gridParent = GridParent.GridParent(self)
                self.gridParent.setGridParent(parentObj, zoneId)
            elif self.gridParent:
                self.gridParent.delete()
                self.gridParent = None

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
            return 10
        if isinstance(parentToken, str):
            self.sendUpdate('setParentStr', [parentToken])
        else:
            self.sendUpdate('setParent', [parentToken])

    def setParentStr(self, parentToken):
        if False:
            return 10
        self.notify.debug('setParentStr(%s): %s' % (self.doId, parentToken))
        if len(parentToken) > 0:
            self.do_setParent(parentToken)

    def setParent(self, parentToken):
        if False:
            return 10
        self.notify.debug('setParent(%s): %s' % (self.doId, parentToken))
        if parentToken == 0:
            senderId = self.air.getAvatarIdFromSender()
            self.air.writeServerEvent('suspicious', senderId, 'setParent(0)')
        else:
            self.do_setParent(parentToken)

    def do_setParent(self, parentToken):
        if False:
            print('Hello World!')
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
            for i in range(10):
                print('nop')
        self.sendUpdate('setZ', [z])

    def d_setH(self, h):
        if False:
            for i in range(10):
                print('nop')
        self.sendUpdate('setH', [h])

    def d_setP(self, p):
        if False:
            print('Hello World!')
        self.sendUpdate('setP', [p])

    def d_setR(self, r):
        if False:
            return 10
        self.sendUpdate('setR', [r])

    def setXY(self, x, y):
        if False:
            while True:
                i = 10
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
            for i in range(10):
                print('nop')
        self.sendUpdate('setHpr', [h, p, r])

    def setXYH(self, x, y, h):
        if False:
            i = 10
            return i + 15
        self.setX(x)
        self.setY(y)
        self.setH(h)

    def d_setXYH(self, x, y, h):
        if False:
            for i in range(10):
                print('nop')
        self.sendUpdate('setXYH', [x, y, h])

    def b_setXYZH(self, x, y, z, h):
        if False:
            while True:
                i = 10
        self.setXYZH(x, y, z, h)
        self.d_setXYZH(x, y, z, h)

    def setXYZH(self, x, y, z, h):
        if False:
            i = 10
            return i + 15
        self.setPos(x, y, z)
        self.setH(h)

    def getXYZH(self):
        if False:
            while True:
                i = 10
        pos = self.getPos()
        h = self.getH()
        return (pos[0], pos[1], pos[2], h)

    def d_setXYZH(self, x, y, z, h):
        if False:
            while True:
                i = 10
        self.sendUpdate('setXYZH', [x, y, z, h])

    def b_setPosHpr(self, x, y, z, h, p, r):
        if False:
            i = 10
            return i + 15
        self.setPosHpr(x, y, z, h, p, r)
        self.d_setPosHpr(x, y, z, h, p, r)

    def d_setPosHpr(self, x, y, z, h, p, r):
        if False:
            for i in range(10):
                print('nop')
        self.sendUpdate('setPosHpr', [x, y, z, h, p, r])