from .DistributedObjectAI import DistributedObjectAI
from direct.directnotify.DirectNotifyGlobal import directNotify

class DistributedObjectGlobalAI(DistributedObjectAI):
    notify = directNotify.newCategory('DistributedObjectGlobalAI')
    doNotDeallocateChannel = 1
    isGlobalDistObj = 1

    def __init__(self, air):
        if False:
            while True:
                i = 10
        DistributedObjectAI.__init__(self, air)

    def announceGenerate(self):
        if False:
            for i in range(10):
                print('nop')
        DistributedObjectAI.announceGenerate(self)
        try:
            if not self.doNotListenToChannel:
                self.air.registerForChannel(self.doId)
        except AttributeError:
            self.air.registerForChannel(self.doId)
        return False

    def delete(self):
        if False:
            while True:
                i = 10
        DistributedObjectAI.delete(self)
        try:
            if not self.doNotListenToChannel:
                self.air.unregisterForChannel(self.doId)
        except AttributeError:
            self.air.unregisterForChannel(self.doId)