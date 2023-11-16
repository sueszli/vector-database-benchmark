"""DistributedLargeBlobSenderAI module: contains the DistributedLargeBlobSenderAI class"""
from direct.distributed import DistributedObjectAI
from direct.directnotify import DirectNotifyGlobal
from . import LargeBlobSenderConsts

class DistributedLargeBlobSenderAI(DistributedObjectAI.DistributedObjectAI):
    """DistributedLargeBlobSenderAI: for sending large chunks of data through
    the DC system to a specific avatar"""
    notify = DirectNotifyGlobal.directNotify.newCategory('DistributedLargeBlobSenderAI')

    def __init__(self, air, zoneId, targetAvId, data, useDisk=0):
        if False:
            while True:
                i = 10
        DistributedObjectAI.DistributedObjectAI.__init__(self, air)
        self.targetAvId = targetAvId
        self.mode = 0
        if useDisk:
            self.mode |= LargeBlobSenderConsts.USE_DISK
        self.generateWithRequired(zoneId)
        s = str(data)
        if useDisk:
            import os
            import random
            origDir = os.getcwd()
            bPath = LargeBlobSenderConsts.getLargeBlobPath()
            try:
                os.chdir(bPath)
            except OSError:
                DistributedLargeBlobSenderAI.notify.error('could not access %s' % bPath)
            while 1:
                num = random.randrange((1 << 30) - 1)
                filename = LargeBlobSenderConsts.FilePattern % num
                try:
                    os.stat(filename)
                except OSError:
                    break
            f = open(filename, 'wb')
            f.write(s)
            f.close()
            os.chdir(origDir)
            self.sendUpdateToAvatarId(self.targetAvId, 'setFilename', [filename])
        else:
            chunkSize = LargeBlobSenderConsts.ChunkSize
            while len(s) > 0:
                self.sendUpdateToAvatarId(self.targetAvId, 'setChunk', [s[:chunkSize]])
                s = s[chunkSize:]
            self.sendUpdateToAvatarId(self.targetAvId, 'setChunk', [''])

    def getMode(self):
        if False:
            for i in range(10):
                print('nop')
        return self.mode

    def getTargetAvId(self):
        if False:
            return 10
        return self.targetAvId

    def setAck(self):
        if False:
            print('Hello World!')
        DistributedLargeBlobSenderAI.notify.debug('setAck')
        assert self.air.getAvatarIdFromSender() == self.targetAvId
        self.requestDelete()