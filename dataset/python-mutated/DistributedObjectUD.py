"""DistributedObjectUD module: contains the DistributedObjectUD class"""
from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.distributed.DistributedObjectBase import DistributedObjectBase
from direct.showbase.MessengerGlobal import messenger
from direct.showbase import PythonUtil

class DistributedObjectUD(DistributedObjectBase):
    notify = directNotify.newCategory('DistributedObjectUD')
    QuietZone = 1

    def __init__(self, air):
        if False:
            while True:
                i = 10
        if not hasattr(self, 'DistributedObjectUD_initialized'):
            self.DistributedObjectUD_initialized = 1
            DistributedObjectBase.__init__(self, air)
            self.accountName = ''
            self.air = air
            className = self.__class__.__name__
            self.dclass = self.air.dclassesByName[className]
            self.__preallocDoId = 0
            self.lastNonQuietZone = None
            self._DOUD_requestedDelete = False
            self.__nextBarrierContext = 0
            self.__barriers = {}
            self.__generated = False
            self.__generates = 0
    if __debug__:

        def status(self, indent=0):
            if False:
                for i in range(10):
                    print('nop')
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
            while True:
                i = 10
        if hasattr(self, 'doId'):
            return 'distObjDelete-%s' % self.doId
        return None

    def sendDeleteEvent(self):
        if False:
            for i in range(10):
                print('nop')
        delEvent = self.getDeleteEvent()
        if delEvent:
            messenger.send(delEvent)

    def delete(self):
        if False:
            while True:
                i = 10
        '\n        Inheritors should redefine this to take appropriate action on delete\n        Note that this may be called multiple times if a class inherits\n        from DistributedObjectUD more than once.\n        '
        self.__generates -= 1
        if self.__generates < 0:
            self.notify.debug('DistributedObjectUD: delete() called more times than generate()')
        if self.__generates == 0:
            if self.air is not None:
                assert self.notify.debug('delete(): %s' % self.__dict__.get('doId'))
                self._DOUD_requestedDelete = False
                for barrier in self.__barriers.values():
                    barrier.cleanup()
                self.__barriers = {}
                self.parentId = None
                self.zoneId = None
                self.__generated = False

    def isDeleted(self):
        if False:
            print('Hello World!')
        '\n        Returns true if the object has been deleted,\n        or if it is brand new and hasnt yet been generated.\n        '
        return self.air is None

    def isGenerated(self):
        if False:
            while True:
                i = 10
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
            i = 10
            return i + 15
        '\n        objects that need to have a doId before they are generated\n        can call this to pre-allocate a doId for the object\n        '
        assert not self.__preallocDoId
        self.doId = self.air.allocateChannel()
        self.__preallocDoId = 1

    def announceGenerate(self):
        if False:
            i = 10
            return i + 15
        '\n        Called after the object has been generated and all\n        of its required fields filled in. Overwrite when needed.\n        '
        self.__generated = True

    def postGenerateMessage(self):
        if False:
            print('Hello World!')
        messenger.send(self.uniqueName('generate'), [self])

    def addInterest(self, zoneId, note='', event=None):
        if False:
            return 10
        self.air.addInterest(self.getDoId(), zoneId, note, event)

    def b_setLocation(self, parentId, zoneId):
        if False:
            print('Hello World!')
        self.d_setLocation(parentId, zoneId)
        self.setLocation(parentId, zoneId)

    def d_setLocation(self, parentId, zoneId):
        if False:
            print('Hello World!')
        self.air.sendSetLocation(self, parentId, zoneId)

    def setLocation(self, parentId, zoneId):
        if False:
            i = 10
            return i + 15
        self.air.storeObjectLocation(self, parentId, zoneId)

    def getLocation(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            if self.parentId <= 0 and self.zoneId <= 0:
                return None
            if self.parentId == 4294967295 and self.zoneId == 4294967295:
                return None
            return (self.parentId, self.zoneId)
        except AttributeError:
            return None

    def updateRequiredFields(self, dclass, di):
        if False:
            i = 10
            return i + 15
        dclass.receiveUpdateBroadcastRequired(self, di)
        self.announceGenerate()
        self.postGenerateMessage()

    def updateAllRequiredFields(self, dclass, di):
        if False:
            i = 10
            return i + 15
        dclass.receiveUpdateAllRequired(self, di)
        self.announceGenerate()
        self.postGenerateMessage()

    def updateRequiredOtherFields(self, dclass, di):
        if False:
            print('Hello World!')
        dclass.receiveUpdateBroadcastRequired(self, di)
        self.announceGenerate()
        self.postGenerateMessage()
        dclass.receiveUpdateOther(self, di)

    def updateAllRequiredOtherFields(self, dclass, di):
        if False:
            return 10
        dclass.receiveUpdateAllRequired(self, di)
        self.announceGenerate()
        self.postGenerateMessage()
        dclass.receiveUpdateOther(self, di)

    def sendSetZone(self, zoneId):
        if False:
            print('Hello World!')
        self.air.sendSetZone(self, zoneId)

    def getZoneChangeEvent(self):
        if False:
            print('Hello World!')
        return 'DOChangeZone-%s' % self.doId

    def getLogicalZoneChangeEvent(self):
        if False:
            print('Hello World!')
        return 'DOLogicalChangeZone-%s' % self.doId

    def handleLogicalZoneChange(self, newZoneId, oldZoneId):
        if False:
            while True:
                i = 10
        'this function gets called as if we never go through the\n        quiet zone. Note that it is called once you reach the newZone,\n        and not at the time that you leave the oldZone.'
        messenger.send(self.getLogicalZoneChangeEvent(), [newZoneId, oldZoneId])

    def getRender(self):
        if False:
            while True:
                i = 10
        return self.air.getRender(self.zoneId)

    def getNonCollidableParent(self):
        if False:
            print('Hello World!')
        return self.air.getNonCollidableParent(self.zoneId)

    def getParentMgr(self):
        if False:
            return 10
        return self.air.getParentMgr(self.zoneId)

    def getCollTrav(self, *args, **kArgs):
        if False:
            print('Hello World!')
        return self.air.getCollTrav(self.zoneId, *args, **kArgs)

    def sendUpdate(self, fieldName, args=[]):
        if False:
            return 10
        assert self.notify.debugStateCall(self)
        if self.air:
            self.air.sendUpdate(self, fieldName, args)

    def GetPuppetConnectionChannel(self, doId):
        if False:
            for i in range(10):
                print('nop')
        return doId + (1001 << 32)

    def GetAccountConnectionChannel(self, doId):
        if False:
            for i in range(10):
                print('nop')
        return doId + (1003 << 32)

    def GetAccountIDFromChannelCode(self, channel):
        if False:
            return 10
        return channel >> 32

    def GetAvatarIDFromChannelCode(self, channel):
        if False:
            return 10
        return channel & 4294967295

    def sendUpdateToAvatarId(self, avId, fieldName, args):
        if False:
            for i in range(10):
                print('nop')
        assert self.notify.debugStateCall(self)
        channelId = self.GetPuppetConnectionChannel(avId)
        self.sendUpdateToChannel(channelId, fieldName, args)

    def sendUpdateToAccountId(self, accountId, fieldName, args):
        if False:
            print('Hello World!')
        assert self.notify.debugStateCall(self)
        channelId = self.GetAccountConnectionChannel(accountId)
        self.sendUpdateToChannel(channelId, fieldName, args)

    def sendUpdateToChannel(self, channelId, fieldName, args):
        if False:
            print('Hello World!')
        assert self.notify.debugStateCall(self)
        if self.air:
            self.air.sendUpdateToChannel(self, channelId, fieldName, args)

    def generateWithRequired(self, zoneId, optionalFields=[]):
        if False:
            for i in range(10):
                print('nop')
        assert self.notify.debugStateCall(self)
        if self.__preallocDoId:
            self.__preallocDoId = 0
            return self.generateWithRequiredAndId(self.doId, zoneId, optionalFields)
        parentId = self.air.districtId
        self.parentId = parentId
        self.zoneId = zoneId
        self.air.generateWithRequired(self, parentId, zoneId, optionalFields)
        self.generate()

    def generateWithRequiredAndId(self, doId, parentId, zoneId, optionalFields=[]):
        if False:
            i = 10
            return i + 15
        assert self.notify.debugStateCall(self)
        if self.__preallocDoId:
            assert doId == self.__preallocDoId
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
            assert doId is None or doId == self.__preallocDoId
            doId = self.__preallocDoId
            self.__preallocDoId = 0
        if doId is None:
            self.doId = self.air.allocateChannel()
        else:
            self.doId = doId
        self.air.addDOToTables(self, location=(parentId, zoneId))
        self.sendGenerateWithRequired(self.air, parentId, zoneId, optionalFields)
        self.generate()

    def generate(self):
        if False:
            while True:
                i = 10
        '\n        Inheritors should put functions that require self.zoneId or\n        other networked info in this function.\n        '
        assert self.notify.debugStateCall(self)
        self.__generates += 1
        self.air.storeObjectLocation(self, self.parentId, self.zoneId)

    def generateInit(self, repository=None):
        if False:
            print('Hello World!')
        '\n        First generate (not from cache).\n        '
        assert self.notify.debugStateCall(self)

    def generateTargetChannel(self, repository):
        if False:
            i = 10
            return i + 15
        '\n        Who to send this to for generate messages\n        '
        if hasattr(self, 'dbObject'):
            return self.doId
        return repository.serverId

    def sendGenerateWithRequired(self, repository, parentId, zoneId, optionalFields=[]):
        if False:
            for i in range(10):
                print('nop')
        assert self.notify.debugStateCall(self)
        dg = self.dclass.aiFormatGenerate(self, self.doId, parentId, zoneId, self.generateTargetChannel(repository), repository.ourChannel, optionalFields)
        repository.send(dg)

    def initFromServerResponse(self, valDict):
        if False:
            return 10
        assert self.notify.debugStateCall(self)
        dclass = self.dclass
        for (key, value) in valDict.items():
            dclass.directUpdate(self, key, value)

    def requestDelete(self):
        if False:
            return 10
        assert self.notify.debugStateCall(self)
        if not self.air:
            doId = 'none'
            if hasattr(self, 'doId'):
                doId = self.doId
            self.notify.warning('Tried to delete a %s (doId %s) that is already deleted' % (self.__class__, doId))
            return
        self.air.requestDelete(self)
        self._DOUD_requestedDelete = True

    def taskName(self, taskString):
        if False:
            i = 10
            return i + 15
        return '%s-%s' % (taskString, self.doId)

    def uniqueName(self, idString):
        if False:
            return 10
        return '%s-%s' % (idString, self.doId)

    def validate(self, avId, bool, msg):
        if False:
            print('Hello World!')
        if not bool:
            self.air.writeServerEvent('suspicious', avId, msg)
            self.notify.warning('validate error: avId: %s -- %s' % (avId, msg))
        return bool

    def beginBarrier(self, name, avIds, timeout, callback):
        if False:
            while True:
                i = 10
        from otp.ai import Barrier
        context = self.__nextBarrierContext
        self.__nextBarrierContext = self.__nextBarrierContext + 1 & 65535
        assert self.notify.debug('beginBarrier(%s, %s, %s, %s)' % (context, name, avIds, timeout))
        if avIds:
            barrier = Barrier.Barrier(name, self.uniqueName(name), avIds, timeout, doneFunc=PythonUtil.Functor(self.__barrierCallback, context, callback))
            self.__barriers[context] = barrier
            self.sendUpdate('setBarrierData', [self.__getBarrierData()])
        else:
            callback(avIds)
        return context

    def __getBarrierData(self):
        if False:
            return 10
        data = []
        for (context, barrier) in self.__barriers.items():
            toons = barrier.pendingToons
            if toons:
                data.append((context, barrier.name, toons))
        return data

    def ignoreBarrier(self, context):
        if False:
            for i in range(10):
                print('nop')
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
            while True:
                i = 10
        return 0

    def execCommand(self, string, mwMgrId, avId, zoneId):
        if False:
            print('Hello World!')
        pass