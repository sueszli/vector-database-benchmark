"""DistributedObjectAI module: contains the DistributedObjectAI class"""
from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.distributed.DistributedObjectBase import DistributedObjectBase
from direct.showbase.MessengerGlobal import messenger
from direct.showbase import PythonUtil

class DistributedObjectAI(DistributedObjectBase):
    notify = directNotify.newCategory('DistributedObjectAI')
    QuietZone = 1

    def __init__(self, air):
        if False:
            print('Hello World!')
        if not hasattr(self, 'DistributedObjectAI_initialized'):
            self.DistributedObjectAI_initialized = 1
            DistributedObjectBase.__init__(self, air)
            self.accountName = ''
            self.air = air
            className = self.__class__.__name__
            self.dclass = self.air.dclassesByName[className]
            self.__preallocDoId = 0
            self.lastNonQuietZone = None
            self._DOAI_requestedDelete = False
            self.__nextBarrierContext = 0
            self.__barriers = {}
            self.__generated = False
            self.__generates = 0
            self._zoneData = None
    if __debug__:

        def status(self, indent=0):
            if False:
                return 10
            '\n            print out doId(parentId, zoneId) className\n                and conditionally show generated or deleted\n            '
            spaces = ' ' * (indent + 2)
            try:
                print('%s%s:' % (' ' * indent, self.__class__.__name__))
                flags = []
                if self.__generated:
                    flags.append('generated')
                if self.air is None:
                    flags.append('deleted')
                flagStr = ''
                if len(flags) > 0:
                    flagStr = ' (%s)' % ' '.join(flags)
                print('%sfrom DistributedObject doId:%s, parent:%s, zone:%s%s' % (spaces, self.doId, self.parentId, self.zoneId, flagStr))
            except Exception as e:
                print('%serror printing status %s' % (spaces, e))

    def getDeleteEvent(self):
        if False:
            print('Hello World!')
        if hasattr(self, 'doId'):
            return 'distObjDelete-%s' % self.doId
        return None

    def sendDeleteEvent(self):
        if False:
            i = 10
            return i + 15
        delEvent = self.getDeleteEvent()
        if delEvent:
            messenger.send(delEvent)

    def getCacheable(self):
        if False:
            i = 10
            return i + 15
        " This method exists only to mirror the similar method on\n        DistributedObject.  AI objects aren't cacheable. "
        return False

    def deleteOrDelay(self):
        if False:
            return 10
        " This method exists only to mirror the similar method on\n        DistributedObject.  AI objects don't have delayDelete, they\n        just get deleted immediately. "
        self.delete()

    def getDelayDeleteCount(self):
        if False:
            return 10
        return 0

    def delete(self):
        if False:
            while True:
                i = 10
        '\n        Inheritors should redefine this to take appropriate action on delete\n        Note that this may be called multiple times if a class inherits\n        from DistributedObjectAI more than once.\n        '
        self.__generates -= 1
        if self.__generates < 0:
            self.notify.debug('DistributedObjectAI: delete() called more times than generate()')
        if self.__generates == 0:
            if self.air is not None:
                assert self.notify.debug('delete(): %s' % self.__dict__.get('doId'))
                self._DOAI_requestedDelete = False
                self.releaseZoneData()
                for barrier in self.__barriers.values():
                    barrier.cleanup()
                self.__barriers = {}
                if not getattr(self, 'doNotDeallocateChannel', False):
                    if self.air:
                        self.air.deallocateChannel(self.doId)
                self.air = None
                self.parentId = None
                self.zoneId = None
                self.__generated = False

    def isDeleted(self):
        if False:
            while True:
                i = 10
        '\n        Returns true if the object has been deleted,\n        or if it is brand new and hasnt yet been generated.\n        '
        return self.air is None

    def isGenerated(self):
        if False:
            return 10
        '\n        Returns true if the object has been generated\n        '
        return self.__generated

    def getDoId(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the distributed object id\n        '
        return self.doId

    def preAllocateDoId(self):
        if False:
            print('Hello World!')
        '\n        objects that need to have a doId before they are generated\n        can call this to pre-allocate a doId for the object\n        '
        assert not self.__preallocDoId
        self.doId = self.air.allocateChannel()
        self.__preallocDoId = 1

    def announceGenerate(self):
        if False:
            return 10
        '\n        Called after the object has been generated and all\n        of its required fields filled in. Overwrite when needed.\n        '

    def b_setLocation(self, parentId, zoneId):
        if False:
            i = 10
            return i + 15
        self.d_setLocation(parentId, zoneId)
        self.setLocation(parentId, zoneId)

    def d_setLocation(self, parentId, zoneId):
        if False:
            i = 10
            return i + 15
        self.air.sendSetLocation(self, parentId, zoneId)

    def setLocation(self, parentId, zoneId):
        if False:
            return 10
        if self.parentId == parentId and self.zoneId == zoneId:
            return
        oldParentId = self.parentId
        oldZoneId = self.zoneId
        self.air.storeObjectLocation(self, parentId, zoneId)
        if oldParentId != parentId or oldZoneId != zoneId:
            self.releaseZoneData()
            messenger.send(self.getZoneChangeEvent(), [zoneId, oldZoneId])
            if zoneId != DistributedObjectAI.QuietZone:
                lastLogicalZone = oldZoneId
                if oldZoneId == DistributedObjectAI.QuietZone:
                    lastLogicalZone = self.lastNonQuietZone
                self.handleLogicalZoneChange(zoneId, lastLogicalZone)
                self.lastNonQuietZone = zoneId

    def getLocation(self):
        if False:
            return 10
        try:
            if self.parentId <= 0 and self.zoneId <= 0:
                return None
            if self.parentId == 4294967295 and self.zoneId == 4294967295:
                return None
            return (self.parentId, self.zoneId)
        except AttributeError:
            return None

    def postGenerateMessage(self):
        if False:
            return 10
        self.__generated = True
        messenger.send(self.uniqueName('generate'), [self])

    def updateRequiredFields(self, dclass, di):
        if False:
            print('Hello World!')
        dclass.receiveUpdateBroadcastRequired(self, di)
        self.announceGenerate()
        self.postGenerateMessage()

    def updateAllRequiredFields(self, dclass, di):
        if False:
            for i in range(10):
                print('nop')
        dclass.receiveUpdateAllRequired(self, di)
        self.announceGenerate()
        self.postGenerateMessage()

    def updateRequiredOtherFields(self, dclass, di):
        if False:
            for i in range(10):
                print('nop')
        dclass.receiveUpdateBroadcastRequired(self, di)
        self.announceGenerate()
        self.postGenerateMessage()
        dclass.receiveUpdateOther(self, di)

    def updateAllRequiredOtherFields(self, dclass, di):
        if False:
            while True:
                i = 10
        dclass.receiveUpdateAllRequired(self, di)
        self.announceGenerate()
        self.postGenerateMessage()
        dclass.receiveUpdateOther(self, di)

    def startMessageBundle(self, name):
        if False:
            print('Hello World!')
        self.air.startMessageBundle(name)

    def sendMessageBundle(self):
        if False:
            for i in range(10):
                print('nop')
        self.air.sendMessageBundle(self.doId)

    def getZoneChangeEvent(self):
        if False:
            for i in range(10):
                print('nop')
        return DistributedObjectAI.staticGetZoneChangeEvent(self.doId)

    def getLogicalZoneChangeEvent(self):
        if False:
            for i in range(10):
                print('nop')
        return DistributedObjectAI.staticGetLogicalZoneChangeEvent(self.doId)

    @staticmethod
    def staticGetZoneChangeEvent(doId):
        if False:
            print('Hello World!')
        return 'DOChangeZone-%s' % doId

    @staticmethod
    def staticGetLogicalZoneChangeEvent(doId):
        if False:
            print('Hello World!')
        return 'DOLogicalChangeZone-%s' % doId

    def handleLogicalZoneChange(self, newZoneId, oldZoneId):
        if False:
            print('Hello World!')
        'this function gets called as if we never go through the\n        quiet zone. Note that it is called once you reach the newZone,\n        and not at the time that you leave the oldZone.'
        messenger.send(self.getLogicalZoneChangeEvent(), [newZoneId, oldZoneId])

    def getZoneData(self):
        if False:
            while True:
                i = 10
        if self._zoneData is None:
            from otp.ai.AIZoneData import AIZoneData
            self._zoneData = AIZoneData(self.air, self.parentId, self.zoneId)
        return self._zoneData

    def releaseZoneData(self):
        if False:
            return 10
        if self._zoneData is not None:
            self._zoneData.destroy()
            self._zoneData = None

    def getRender(self):
        if False:
            for i in range(10):
                print('nop')
        return self.getZoneData().getRender()

    def getNonCollidableParent(self):
        if False:
            for i in range(10):
                print('nop')
        return self.getZoneData().getNonCollidableParent()

    def getParentMgr(self):
        if False:
            for i in range(10):
                print('nop')
        return self.getZoneData().getParentMgr()

    def getCollTrav(self, *args, **kArgs):
        if False:
            i = 10
            return i + 15
        return self.getZoneData().getCollTrav(*args, **kArgs)

    def sendUpdate(self, fieldName, args=[]):
        if False:
            print('Hello World!')
        assert self.notify.debugStateCall(self)
        if self.air:
            self.air.sendUpdate(self, fieldName, args)

    def GetPuppetConnectionChannel(self, doId):
        if False:
            while True:
                i = 10
        return doId + (1001 << 32)

    def GetAccountConnectionChannel(self, doId):
        if False:
            print('Hello World!')
        return doId + (1003 << 32)

    def GetAccountIDFromChannelCode(self, channel):
        if False:
            for i in range(10):
                print('nop')
        return channel >> 32

    def GetAvatarIDFromChannelCode(self, channel):
        if False:
            while True:
                i = 10
        return channel & 4294967295

    def sendUpdateToAvatarId(self, avId, fieldName, args):
        if False:
            i = 10
            return i + 15
        assert self.notify.debugStateCall(self)
        channelId = self.GetPuppetConnectionChannel(avId)
        self.sendUpdateToChannel(channelId, fieldName, args)

    def sendUpdateToAccountId(self, accountId, fieldName, args):
        if False:
            i = 10
            return i + 15
        assert self.notify.debugStateCall(self)
        channelId = self.GetAccountConnectionChannel(accountId)
        self.sendUpdateToChannel(channelId, fieldName, args)

    def sendUpdateToChannel(self, channelId, fieldName, args):
        if False:
            i = 10
            return i + 15
        assert self.notify.debugStateCall(self)
        if self.air:
            self.air.sendUpdateToChannel(self, channelId, fieldName, args)

    def generateWithRequired(self, zoneId, optionalFields=[]):
        if False:
            return 10
        assert self.notify.debugStateCall(self)
        if self.__preallocDoId:
            self.__preallocDoId = 0
            return self.generateWithRequiredAndId(self.doId, zoneId, optionalFields)
        parentId = self.air.districtId
        self.air.generateWithRequired(self, parentId, zoneId, optionalFields)
        self.generate()
        self.announceGenerate()
        self.postGenerateMessage()

    def generateWithRequiredAndId(self, doId, parentId, zoneId, optionalFields=[]):
        if False:
            i = 10
            return i + 15
        assert self.notify.debugStateCall(self)
        if self.__preallocDoId:
            assert doId == self.doId
            self.__preallocDoId = 0
        self.air.generateWithRequiredAndId(self, doId, parentId, zoneId, optionalFields)
        self.generate()
        self.announceGenerate()
        self.postGenerateMessage()

    def generateOtpObject(self, parentId, zoneId, optionalFields=[], doId=None):
        if False:
            return 10
        assert self.notify.debugStateCall(self)
        if self.__preallocDoId:
            assert doId is None or doId == self.doId
            doId = self.doId
            self.__preallocDoId = 0
        if doId is None:
            self.doId = self.air.allocateChannel()
        else:
            self.doId = doId
        self.air.addDOToTables(self, location=(parentId, zoneId))
        self.sendGenerateWithRequired(self.air, parentId, zoneId, optionalFields)
        self.generate()
        self.announceGenerate()
        self.postGenerateMessage()

    def generate(self):
        if False:
            i = 10
            return i + 15
        '\n        Inheritors should put functions that require self.zoneId or\n        other networked info in this function.\n        '
        assert self.notify.debugStateCall(self)
        self.__generates += 1

    def generateInit(self, repository=None):
        if False:
            i = 10
            return i + 15
        '\n        First generate (not from cache).\n        '
        assert self.notify.debugStateCall(self)

    def generateTargetChannel(self, repository):
        if False:
            return 10
        '\n        Who to send this to for generate messages\n        '
        if hasattr(self, 'dbObject'):
            return self.doId
        return repository.serverId

    def sendGenerateWithRequired(self, repository, parentId, zoneId, optionalFields=[]):
        if False:
            return 10
        assert self.notify.debugStateCall(self)
        dg = self.dclass.aiFormatGenerate(self, self.doId, parentId, zoneId, self.generateTargetChannel(repository), repository.ourChannel, optionalFields)
        repository.send(dg)

    def initFromServerResponse(self, valDict):
        if False:
            while True:
                i = 10
        assert self.notify.debugStateCall(self)
        dclass = self.dclass
        for (key, value) in valDict.items():
            dclass.directUpdate(self, key, value)

    def requestDelete(self):
        if False:
            print('Hello World!')
        assert self.notify.debugStateCall(self)
        if not self.air:
            doId = 'none'
            if hasattr(self, 'doId'):
                doId = self.doId
            self.notify.warning('Tried to delete a %s (doId %s) that is already deleted' % (self.__class__, doId))
            return
        self.air.requestDelete(self)
        self._DOAI_requestedDelete = True

    def taskName(self, taskString):
        if False:
            return 10
        return '%s-%s' % (taskString, self.doId)

    def uniqueName(self, idString):
        if False:
            for i in range(10):
                print('nop')
        return '%s-%s' % (idString, self.doId)

    def validate(self, avId, bool, msg):
        if False:
            i = 10
            return i + 15
        if not bool:
            self.air.writeServerEvent('suspicious', avId, msg)
            self.notify.warning('validate error: avId: %s -- %s' % (avId, msg))
        return bool

    def beginBarrier(self, name, avIds, timeout, callback):
        if False:
            i = 10
            return i + 15
        from otp.ai import Barrier
        context = self.__nextBarrierContext
        self.__nextBarrierContext = self.__nextBarrierContext + 1 & 65535
        assert self.notify.debug('beginBarrier(%s, %s, %s, %s)' % (context, name, avIds, timeout))
        if avIds:
            barrier = Barrier.Barrier(name, self.uniqueName(name), avIds, timeout, doneFunc=PythonUtil.Functor(self.__barrierCallback, context, callback))
            self.__barriers[context] = barrier
            self.sendUpdate('setBarrierData', [self.getBarrierData()])
        else:
            callback(avIds)
        return context

    def getBarrierData(self):
        if False:
            while True:
                i = 10
        data = []
        for (context, barrier) in self.__barriers.items():
            avatars = barrier.pendingAvatars
            if avatars:
                data.append((context, barrier.name, avatars))
        return data

    def ignoreBarrier(self, context):
        if False:
            while True:
                i = 10
        barrier = self.__barriers.get(context)
        if barrier:
            barrier.cleanup()
            del self.__barriers[context]

    def setBarrierReady(self, context):
        if False:
            while True:
                i = 10
        avId = self.air.getAvatarIdFromSender()
        assert self.notify.debug('setBarrierReady(%s, %s)' % (context, avId))
        barrier = self.__barriers.get(context)
        if barrier is None:
            return
        barrier.clear(avId)

    def __barrierCallback(self, context, callback, avIds):
        if False:
            return 10
        assert self.notify.debug('barrierCallback(%s, %s)' % (context, avIds))
        barrier = self.__barriers.get(context)
        if barrier:
            barrier.cleanup()
            del self.__barriers[context]
            callback(avIds)
        else:
            self.notify.warning('Unexpected completion from barrier %s' % context)

    def isGridParent(self):
        if False:
            for i in range(10):
                print('nop')
        return 0

    def execCommand(self, string, mwMgrId, avId, zoneId):
        if False:
            while True:
                i = 10
        pass

    def _retrieveCachedData(self):
        if False:
            i = 10
            return i + 15
        ' This is a no-op on the AI. '

    def setAI(self, aiChannel):
        if False:
            i = 10
            return i + 15
        self.air.setAI(self.doId, aiChannel)