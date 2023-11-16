"""DistributedNode module: contains the DistributedNode class"""
from panda3d.core import NodePath
from . import GridParent
from . import DistributedObject

class DistributedNode(DistributedObject.DistributedObject, NodePath):
    """Distributed Node class:"""

    def __init__(self, cr):
        if False:
            i = 10
            return i + 15
        if not hasattr(self, 'DistributedNode_initialized'):
            self.DistributedNode_initialized = 1
            self.gotStringParentToken = 0
            DistributedObject.DistributedObject.__init__(self, cr)
            if not self.this:
                NodePath.__init__(self, 'DistributedNode')
            self.gridParent = None

    def disable(self):
        if False:
            i = 10
            return i + 15
        if self.activeState != DistributedObject.ESDisabled:
            if not self.isEmpty():
                self.reparentTo(hidden)
            DistributedObject.DistributedObject.disable(self)

    def delete(self):
        if False:
            print('Hello World!')
        if not hasattr(self, 'DistributedNode_deleted'):
            self.DistributedNode_deleted = 1
            if not self.isEmpty():
                self.removeNode()
            if self.gridParent:
                self.gridParent.delete()
            DistributedObject.DistributedObject.delete(self)

    def generate(self):
        if False:
            return 10
        DistributedObject.DistributedObject.generate(self)
        self.gotStringParentToken = 0

    def setLocation(self, parentId, zoneId, teleport=0):
        if False:
            for i in range(10):
                print('nop')
        DistributedObject.DistributedObject.setLocation(self, parentId, zoneId)
        parentObj = self.cr.doId2do.get(parentId)
        if parentObj:
            if parentObj.isGridParent() and zoneId >= parentObj.startingZone:
                if not self.gridParent:
                    self.gridParent = GridParent.GridParent(self)
                self.gridParent.setGridParent(parentObj, zoneId, teleport)
            elif self.gridParent:
                self.gridParent.delete()
                self.gridParent = None
        elif self.gridParent:
            self.gridParent.delete()
            self.gridParent = None

    def __cmp__(self, other):
        if False:
            print('Hello World!')
        if self is other:
            return 0
        else:
            return 1

    def b_setParent(self, parentToken):
        if False:
            for i in range(10):
                print('nop')
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

    def setParentStr(self, parentTokenStr):
        if False:
            for i in range(10):
                print('nop')
        assert self.notify.debug('setParentStr: %s' % parentTokenStr)
        assert self.notify.debug('isGenerated: %s' % self.isGenerated())
        if len(parentTokenStr) > 0:
            self.do_setParent(parentTokenStr)
            self.gotStringParentToken = 1

    def setParent(self, parentToken):
        if False:
            return 10
        assert self.notify.debug('setParent: %s' % parentToken)
        assert self.notify.debug('isGenerated: %s' % self.isGenerated())
        justGotRequiredParentAsStr = not self.isGenerated() and self.gotStringParentToken
        if not justGotRequiredParentAsStr:
            if parentToken != 0:
                self.do_setParent(parentToken)
        self.gotStringParentToken = 0

    def do_setParent(self, parentToken):
        if False:
            print('Hello World!')
        'do_setParent(self, int parentToken)\n\n        This function is defined simply to allow a derived class (like\n        DistributedAvatar) to override the behavior of setParent if\n        desired.\n        '
        if not self.isDisabled():
            self.cr.parentMgr.requestReparent(self, parentToken)

    def d_setX(self, x):
        if False:
            return 10
        self.sendUpdate('setX', [x])

    def d_setY(self, y):
        if False:
            i = 10
            return i + 15
        self.sendUpdate('setY', [y])

    def d_setZ(self, z):
        if False:
            print('Hello World!')
        self.sendUpdate('setZ', [z])

    def d_setH(self, h):
        if False:
            for i in range(10):
                print('nop')
        self.sendUpdate('setH', [h])

    def d_setP(self, p):
        if False:
            i = 10
            return i + 15
        self.sendUpdate('setP', [p])

    def d_setR(self, r):
        if False:
            for i in range(10):
                print('nop')
        self.sendUpdate('setR', [r])

    def setXY(self, x, y):
        if False:
            print('Hello World!')
        self.setX(x)
        self.setY(y)

    def d_setXY(self, x, y):
        if False:
            i = 10
            return i + 15
        self.sendUpdate('setXY', [x, y])

    def setXZ(self, x, z):
        if False:
            while True:
                i = 10
        self.setX(x)
        self.setZ(z)

    def d_setXZ(self, x, z):
        if False:
            for i in range(10):
                print('nop')
        self.sendUpdate('setXZ', [x, z])

    def d_setPos(self, x, y, z):
        if False:
            i = 10
            return i + 15
        self.sendUpdate('setPos', [x, y, z])

    def d_setHpr(self, h, p, r):
        if False:
            i = 10
            return i + 15
        self.sendUpdate('setHpr', [h, p, r])

    def setXYH(self, x, y, h):
        if False:
            print('Hello World!')
        self.setX(x)
        self.setY(y)
        self.setH(h)

    def d_setXYH(self, x, y, h):
        if False:
            i = 10
            return i + 15
        self.sendUpdate('setXYH', [x, y, h])

    def setXYZH(self, x, y, z, h):
        if False:
            return 10
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