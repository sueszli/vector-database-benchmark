from direct.distributed.DistributedObjectAI import DistributedObjectAI

class DistributedCameraAI(DistributedObjectAI):

    def __init__(self, air):
        if False:
            print('Hello World!')
        DistributedObjectAI.__init__(self, air)
        self.parent = 0
        self.fixtures = []

    def getCamParent(self):
        if False:
            return 10
        return self.parent

    def getFixtures(self):
        if False:
            for i in range(10):
                print('nop')
        return self.fixtures