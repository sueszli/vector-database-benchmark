from direct.distributed.DistributedSmoothNodeBase import DistributedSmoothNodeBase
from direct.distributed.GridParent import GridParent
from direct.showbase.PythonUtil import report, getBase

class GridChild:
    """
    Any object that expects to be parented to a grid should inherit from this.
    It works with GridParent to manage its grid cell hierarchy in the scenegraph.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        try:
            self.__initiallized
        except AttributeError:
            self._gridParent = None
            self._gridInterestEnabled = False
            self._gridInterests = {}

    def delete(self):
        if False:
            while True:
                i = 10
        self.__setGridParent(None)
        self.enableGridInterest(False)

    @report(types=['args'], dConfigParam='smoothnode')
    def setGridCell(self, grid, zoneId):
        if False:
            while True:
                i = 10
        if not hasattr(self, 'getParent'):
            return
        if grid is None:
            self.__setGridParent(None)
            self.__clearGridInterest()
        else:
            if not self._gridParent:
                self.__setGridParent(GridParent(self))
            self._gridParent.setGridCell(grid, zoneId)
            self.updateGridInterest(grid, zoneId)

    def updateGridInterest(self, grid, zoneId):
        if False:
            while True:
                i = 10
        self.__setGridInterest(grid, zoneId)

    def enableGridInterest(self, enabled=True):
        if False:
            while True:
                i = 10
        self._gridInterestEnabled = enabled
        if enabled and self.isOnAGrid():
            for (currGridId, interestInfo) in self._gridInterests.items():
                currGrid = getBase().getRepository().doId2do.get(currGridId)
                if currGrid:
                    self.__setGridInterest(currGrid, interestInfo[1])
                else:
                    self.notify.warning('unknown grid interest %s' % currGridId)
        else:
            for (currGridId, interestInfo) in self._gridInterests.items():
                self.cr.removeTaggedInterest(interestInfo[0])

    def isOnAGrid(self):
        if False:
            print('Hello World!')
        return self._gridParent is not None

    def getGrid(self):
        if False:
            for i in range(10):
                print('nop')
        if self._gridParent:
            return self._gridParent.getGrid()
        else:
            return None

    def getGridZone(self):
        if False:
            for i in range(10):
                print('nop')
        if self._gridParent:
            return self._gridParent.getGridZone()
        else:
            return None

    def __setGridParent(self, gridParent):
        if False:
            i = 10
            return i + 15
        if self._gridParent and self._gridParent is not gridParent:
            self._gridParent.delete()
        self._gridParent = gridParent

    def __setGridInterest(self, grid, zoneId):
        if False:
            while True:
                i = 10
        assert not self.cr.noNewInterests()
        if self.cr.noNewInterests():
            self.notify.warning('startProcessVisibility(%s): tried to open a new interest during logout' % self.doId)
            return
        gridDoId = grid.getDoId()
        existingInterest = self._gridInterests.get(gridDoId)
        if self._gridInterestEnabled:
            if existingInterest and existingInterest[0]:
                self.cr.alterInterest(existingInterest[0], grid.getDoId(), zoneId)
                existingInterest[1] = zoneId
            else:
                newInterest = self.cr.addTaggedInterest(gridDoId, zoneId, self.cr.ITAG_GAME, self.uniqueName('gridvis'))
                self._gridInterests[gridDoId] = [newInterest, zoneId]
        elif game.process == 'client':
            self._gridInterests[gridDoId] = [None, zoneId]

    def getGridInterestIds(self):
        if False:
            return 10
        return list(self._gridInterests.keys())

    def getGridInterestZoneId(self, gridDoId):
        if False:
            print('Hello World!')
        return self._gridInterests.get(gridDoId, [None, None])[1]

    def __clearGridInterest(self):
        if False:
            i = 10
            return i + 15
        if self._gridInterestEnabled:
            for (currGridId, interestInfo) in self._gridInterests.items():
                self.cr.removeTaggedInterest(interestInfo[0])
        self._gridInterests = {}

class SmoothGridChild(GridChild):
    """
    SmoothNodes have a special requirement in that they need to send
    their current cell along with their telemetry data stream. This
    allows the distributed receiving objects to update their grid parent
    according to this value, rather than the setLocation() data.

    Use this instead of GridNode when you expect this object to send its
    telemetry data out.
    """

    def __init__(self):
        if False:
            return 10
        GridChild.__init__(self)
        assert isinstance(self, DistributedSmoothNodeBase), 'All GridChild objects must be instances of DistributedSmoothNodeBase'

    @report(types=['args'], dConfigParam='smoothnode')
    def setGridCell(self, grid, zoneId):
        if False:
            print('Hello World!')
        GridChild.setGridCell(self, grid, zoneId)
        if grid and self.isGenerated():
            self.cnode.setEmbeddedVal(zoneId)

    @report(types=['args'], dConfigParam='smoothnode')
    def transformTelemetry(self, x, y, z, h, p, r, e):
        if False:
            while True:
                i = 10
        if self.isOnAGrid():
            self.setGridCell(self.getGrid(), e)
        return (x, y, z, h, p, r)