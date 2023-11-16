from __future__ import annotations
from panda3d.core import NodePath

class GridParent:
    GridZone2CellOrigin: dict[tuple, NodePath] = {}
    GridZone2count: dict[tuple, int] = {}

    @staticmethod
    def getCellOrigin(grid, zoneId):
        if False:
            return 10
        tup = (grid, zoneId)
        if tup not in GridParent.GridZone2count:
            GridParent.GridZone2count[tup] = 0
            GridParent.GridZone2CellOrigin[tup] = grid.attachNewNode('cellOrigin-%s' % zoneId)
            cellPos = grid.getZoneCellOrigin(zoneId)
            GridParent.GridZone2CellOrigin[tup].setPos(*cellPos)
        GridParent.GridZone2count[tup] += 1
        return GridParent.GridZone2CellOrigin[tup]

    @staticmethod
    def releaseCellOrigin(grid, zoneId):
        if False:
            print('Hello World!')
        tup = (grid, zoneId)
        GridParent.GridZone2count[tup] -= 1
        if GridParent.GridZone2count[tup] == 0:
            del GridParent.GridZone2count[tup]
            GridParent.GridZone2CellOrigin[tup].removeNode()
            del GridParent.GridZone2CellOrigin[tup]

    def __init__(self, av):
        if False:
            for i in range(10):
                print('nop')
        self.av = av
        self.grid = None
        self.ownCellOrigin = NodePath('cellOrigin')
        self.cellOrigin = self.ownCellOrigin

    def delete(self):
        if False:
            for i in range(10):
                print('nop')
        if self.av:
            if self.av.getParent() == self.cellOrigin:
                self.av.detachNode()
            del self.av
            self.av = None
        if self.ownCellOrigin is not None:
            self.ownCellOrigin.removeNode()
            self.ownCellOrigin = None
        if self.grid is not None:
            self.releaseCellOrigin(self.grid, self.zoneId)
            self.grid = None
            self.zoneId = None

    def setGridParent(self, grid, zoneId, teleport=0):
        if False:
            for i in range(10):
                print('nop')
        if self.av.getParent().isEmpty():
            teleport = 1
        if not teleport:
            self.av.wrtReparentTo(hidden)
        if self.grid is not None:
            self.releaseCellOrigin(self.grid, self.zoneId)
        self.grid = grid
        self.zoneId = zoneId
        self.cellOrigin = self.getCellOrigin(self.grid, self.zoneId)
        if not teleport:
            self.av.wrtReparentTo(self.cellOrigin)
        else:
            self.av.reparentTo(self.cellOrigin)