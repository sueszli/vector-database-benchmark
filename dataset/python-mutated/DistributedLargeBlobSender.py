"""DistributedLargeBlobSender module: contains the DistributedLargeBlobSender class"""
from direct.distributed import DistributedObject
from direct.directnotify import DirectNotifyGlobal
from direct.showbase.MessengerGlobal import messenger
from . import LargeBlobSenderConsts

class DistributedLargeBlobSender(DistributedObject.DistributedObject):
    """DistributedLargeBlobSender: for sending large chunks of data through
    the DC system"""
    notify = DirectNotifyGlobal.directNotify.newCategory('DistributedLargeBlobSender')

    def __init__(self, cr):
        if False:
            print('Hello World!')
        DistributedObject.DistributedObject.__init__(self, cr)

    def generate(self):
        if False:
            return 10
        DistributedLargeBlobSender.notify.debug('generate')
        DistributedObject.DistributedObject.generate(self)
        self.complete = 0
        self.doneEvent = None

    def setMode(self, mode):
        if False:
            return 10
        self.mode = mode
        self.useDisk = mode & LargeBlobSenderConsts.USE_DISK

    def setTargetAvId(self, avId):
        if False:
            print('Hello World!')
        self.targetAvId = avId

    def announceGenerate(self):
        if False:
            print('Hello World!')
        DistributedLargeBlobSender.notify.debug('announceGenerate')
        DistributedObject.DistributedObject.announceGenerate(self)
        if self.targetAvId != base.localAvatar.doId:
            return
        if not self.useDisk:
            self.blob = ''

    def setChunk(self, chunk):
        if False:
            print('Hello World!')
        DistributedLargeBlobSender.notify.debug('setChunk')
        assert not self.useDisk
        if len(chunk) > 0:
            self.blob += chunk
        else:
            self.privOnBlobComplete()

    def setFilename(self, filename):
        if False:
            print('Hello World!')
        DistributedLargeBlobSender.notify.debug('setFilename: %s' % filename)
        assert self.useDisk
        import os
        origDir = os.getcwd()
        bPath = LargeBlobSenderConsts.getLargeBlobPath()
        try:
            os.chdir(bPath)
        except OSError:
            DistributedLargeBlobSender.notify.error('could not access %s' % bPath)
        f = open(filename, 'rb')
        self.blob = f.read()
        f.close()
        os.unlink(filename)
        os.chdir(origDir)
        self.privOnBlobComplete()

    def isComplete(self):
        if False:
            while True:
                i = 10
        " returns non-zero if we've got the full blob "
        return self.complete

    def setDoneEvent(self, event):
        if False:
            print('Hello World!')
        self.doneEvent = event

    def privOnBlobComplete(self):
        if False:
            i = 10
            return i + 15
        assert not self.isComplete()
        self.complete = 1
        if self.doneEvent is not None:
            messenger.send(self.doneEvent, [self.blob])

    def getBlob(self):
        if False:
            return 10
        ' returns the full blob '
        assert self.isComplete()
        return self.blob

    def sendAck(self):
        if False:
            return 10
        self.sendUpdate('setAck', [])