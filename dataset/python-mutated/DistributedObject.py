"""DistributedObject module: contains the DistributedObject class"""
from panda3d.direct import DCPacker
from direct.showbase.MessengerGlobal import messenger
from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.distributed.DistributedObjectBase import DistributedObjectBase
ESNew = 1
ESDeleted = 2
ESDisabling = 3
ESDisabled = 4
ESGenerating = 5
ESGenerated = 6
ESNum2Str = {ESNew: 'ESNew', ESDeleted: 'ESDeleted', ESDisabling: 'ESDisabling', ESDisabled: 'ESDisabled', ESGenerating: 'ESGenerating', ESGenerated: 'ESGenerated'}

class DistributedObject(DistributedObjectBase):
    """
    The Distributed Object class is the base class for all network based
    (i.e. distributed) objects.  These will usually (always?) have a
    dclass entry in a \\*.dc file.
    """
    notify = directNotify.newCategory('DistributedObject')
    neverDisable = 0

    def __init__(self, cr):
        if False:
            print('Hello World!')
        assert self.notify.debugStateCall(self)
        if not hasattr(self, 'DistributedObject_initialized'):
            self.DistributedObject_initialized = 1
            DistributedObjectBase.__init__(self, cr)
            self.setCacheable(0)
            self._token2delayDeleteName = {}
            self._delayDeleteForceAllow = False
            self._delayDeleted = 0
            self.activeState = ESNew
            self.__nextContext = 0
            self.__callbacks = {}
            self.__barrierContext = None
    if __debug__:

        def status(self, indent=0):
            if False:
                return 10
            '\n            print out "doId(parentId, zoneId) className\n                and conditionally show generated, disabled, neverDisable,\n                or cachable"\n            '
            spaces = ' ' * (indent + 2)
            try:
                print('%s%s:' % (' ' * indent, self.__class__.__name__))
                flags = []
                if self.activeState == ESGenerated:
                    flags.append('generated')
                if self.activeState < ESGenerating:
                    flags.append('disabled')
                if self.neverDisable:
                    flags.append('neverDisable')
                if self.cacheable:
                    flags.append('cacheable')
                flagStr = ''
                if len(flags) > 0:
                    flagStr = ' (%s)' % ' '.join(flags)
                print('%sfrom DistributedObject doId:%s, parent:%s, zone:%s%s' % (spaces, self.doId, self.parentId, self.zoneId, flagStr))
            except Exception as e:
                print('%serror printing status %s' % (spaces, e))

    def getAutoInterests(self):
        if False:
            print('Hello World!')

        def _getAutoInterests(cls):
            if False:
                i = 10
                return i + 15
            if 'autoInterests' in cls.__dict__:
                autoInterests = cls.autoInterests
            else:
                autoInterests = set()
                for base in cls.__bases__:
                    autoInterests.update(_getAutoInterests(base))
                if cls.__name__ in self.cr.dclassesByName:
                    dclass = self.cr.dclassesByName[cls.__name__]
                    field = dclass.getFieldByName('AutoInterest')
                    if field is not None:
                        p = DCPacker()
                        p.setUnpackData(field.getDefaultValue())
                        length = p.rawUnpackUint16() // 4
                        for i in range(length):
                            zone = int(p.rawUnpackUint32())
                            autoInterests.add(zone)
                    autoInterests.update(autoInterests)
                    cls.autoInterests = autoInterests
            return set(autoInterests)
        autoInterests = _getAutoInterests(self.__class__)
        if len(autoInterests) > 1:
            self.notify.error('only one auto-interest allowed per DC class, %s has %s autoInterests (%s)' % (self.dclass.getName(), len(autoInterests), list(autoInterests)))
        _getAutoInterests = None
        return list(autoInterests)

    def setNeverDisable(self, boolean):
        if False:
            return 10
        assert boolean == 1 or boolean == 0
        self.neverDisable = boolean

    def getNeverDisable(self):
        if False:
            while True:
                i = 10
        return self.neverDisable

    def _retrieveCachedData(self):
        if False:
            return 10
        if self.cr.doDataCache.hasCachedData(self.doId):
            self._cachedData = self.cr.doDataCache.popCachedData(self.doId)

    def setCachedData(self, name, data):
        if False:
            i = 10
            return i + 15
        assert isinstance(name, str)
        self.cr.doDataCache.setCachedData(self.doId, name, data)

    def hasCachedData(self, name):
        if False:
            return 10
        assert isinstance(name, str)
        if not hasattr(self, '_cachedData'):
            return False
        return name in self._cachedData

    def getCachedData(self, name):
        if False:
            return 10
        assert isinstance(name, str)
        data = self._cachedData[name]
        del self._cachedData[name]
        return data

    def flushCachedData(self, name):
        if False:
            while True:
                i = 10
        assert isinstance(name, str)
        self._cachedData[name].flush()

    def setCacheable(self, boolean):
        if False:
            for i in range(10):
                print('nop')
        assert boolean == 1 or boolean == 0
        self.cacheable = boolean

    def getCacheable(self):
        if False:
            return 10
        return self.cacheable

    def deleteOrDelay(self):
        if False:
            return 10
        if len(self._token2delayDeleteName) > 0:
            if not self._delayDeleted:
                self._delayDeleted = 1
                messenger.send(self.getDelayDeleteEvent())
                if len(self._token2delayDeleteName) > 0:
                    self.delayDelete()
                    if len(self._token2delayDeleteName) > 0:
                        self._deactivateDO()
        else:
            self.disableAnnounceAndDelete()

    def disableAnnounceAndDelete(self):
        if False:
            i = 10
            return i + 15
        self.disableAndAnnounce()
        self.delete()
        self._destroyDO()

    def getDelayDeleteCount(self):
        if False:
            while True:
                i = 10
        return len(self._token2delayDeleteName)

    def getDelayDeleteEvent(self):
        if False:
            i = 10
            return i + 15
        return self.uniqueName('delayDelete')

    def getDisableEvent(self):
        if False:
            i = 10
            return i + 15
        return self.uniqueName('disable')

    def disableAndAnnounce(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Inheritors should *not* redefine this function.\n        '
        if self.activeState != ESDisabled:
            self.activeState = ESDisabling
            messenger.send(self.getDisableEvent())
            self.disable()
            self.activeState = ESDisabled
            if not self._delayDeleted:
                self._deactivateDO()

    def announceGenerate(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sends a message to the world after the object has been\n        generated and all of its required fields filled in.\n        '
        assert self.notify.debug('announceGenerate(): %s' % self.doId)

    def _deactivateDO(self):
        if False:
            i = 10
            return i + 15
        if not self.cr:
            self.notify.warning('self.cr is none in _deactivateDO %d' % self.doId)
            if hasattr(self, 'destroyDoStackTrace'):
                print(self.destroyDoStackTrace)
        self.__callbacks = {}
        self.cr.closeAutoInterests(self)
        self.setLocation(0, 0)
        self.cr.deleteObjectLocation(self, self.parentId, self.zoneId)

    def _destroyDO(self):
        if False:
            return 10
        if __debug__:
            from direct.showbase.PythonUtil import StackTrace
            self.destroyDoStackTrace = StackTrace()
        if hasattr(self, '_cachedData'):
            for (name, cachedData) in self._cachedData.items():
                self.notify.warning('flushing unretrieved cached data: %s' % name)
                cachedData.flush()
            del self._cachedData
        self.cr = None
        self.dclass = None

    def disable(self):
        if False:
            i = 10
            return i + 15
        '\n        Inheritors should redefine this to take appropriate action on disable\n        '
        assert self.notify.debug('disable(): %s' % self.doId)

    def isDisabled(self):
        if False:
            print('Hello World!')
        "\n        Returns true if the object has been disabled and/or deleted,\n        or if it is brand new and hasn't yet been generated.\n        "
        return self.activeState < ESGenerating

    def isGenerated(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns true if the object has been fully generated by now,\n        and not yet disabled.\n        '
        assert self.notify.debugStateCall(self)
        return self.activeState == ESGenerated

    def delete(self):
        if False:
            while True:
                i = 10
        '\n        Inheritors should redefine this to take appropriate action on delete\n        '
        assert self.notify.debug('delete(): %s' % self.doId)
        self.DistributedObject_deleted = 1

    def generate(self):
        if False:
            i = 10
            return i + 15
        '\n        Inheritors should redefine this to take appropriate action on generate\n        '
        assert self.notify.debugStateCall(self)
        self.activeState = ESGenerating
        if not hasattr(self, '_autoInterestHandle'):
            self.cr.openAutoInterests(self)

    def generateInit(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This method is called when the DistributedObject is first introduced\n        to the world... Not when it is pulled from the cache.\n        '
        self.activeState = ESGenerating

    def getDoId(self):
        if False:
            return 10
        '\n        Return the distributed object id\n        '
        return self.doId

    def postGenerateMessage(self):
        if False:
            while True:
                i = 10
        if self.activeState != ESGenerated:
            self.activeState = ESGenerated
            messenger.send(self.uniqueName('generate'), [self])

    def updateRequiredFields(self, dclass, di):
        if False:
            print('Hello World!')
        dclass.receiveUpdateBroadcastRequired(self, di)
        self.announceGenerate()
        self.postGenerateMessage()

    def updateAllRequiredFields(self, dclass, di):
        if False:
            while True:
                i = 10
        dclass.receiveUpdateAllRequired(self, di)
        self.announceGenerate()
        self.postGenerateMessage()

    def updateRequiredOtherFields(self, dclass, di):
        if False:
            while True:
                i = 10
        dclass.receiveUpdateBroadcastRequired(self, di)
        self.announceGenerate()
        self.postGenerateMessage()
        dclass.receiveUpdateOther(self, di)

    def sendUpdate(self, fieldName, args=[], sendToId=None):
        if False:
            while True:
                i = 10
        if self.cr:
            dg = self.dclass.clientFormatUpdate(fieldName, sendToId or self.doId, args)
            self.cr.send(dg)
        else:
            assert self.notify.error('sendUpdate failed, because self.cr is not set')

    def sendDisableMsg(self):
        if False:
            return 10
        self.cr.sendDisableMsg(self.doId)

    def sendDeleteMsg(self):
        if False:
            print('Hello World!')
        self.cr.sendDeleteMsg(self.doId)

    def taskName(self, taskString):
        if False:
            for i in range(10):
                print('nop')
        return '%s-%s' % (taskString, self.doId)

    def uniqueName(self, idString):
        if False:
            for i in range(10):
                print('nop')
        return '%s-%s' % (idString, self.doId)

    def getCallbackContext(self, callback, extraArgs=[]):
        if False:
            for i in range(10):
                print('nop')
        context = self.__nextContext
        self.__callbacks[context] = (callback, extraArgs)
        self.__nextContext = self.__nextContext + 1 & 65535
        return context

    def getCurrentContexts(self):
        if False:
            print('Hello World!')
        return list(self.__callbacks.keys())

    def getCallback(self, context):
        if False:
            i = 10
            return i + 15
        return self.__callbacks[context][0]

    def getCallbackArgs(self, context):
        if False:
            while True:
                i = 10
        return self.__callbacks[context][1]

    def doCallbackContext(self, context, args):
        if False:
            while True:
                i = 10
        tuple = self.__callbacks.get(context)
        if tuple:
            (callback, extraArgs) = tuple
            completeArgs = args + extraArgs
            if callback is not None:
                callback(*completeArgs)
            del self.__callbacks[context]
        else:
            self.notify.warning('Got unexpected context from AI: %s' % context)

    def setBarrierData(self, data):
        if False:
            print('Hello World!')
        for (context, name, avIds) in data:
            for avId in avIds:
                if self.cr.isLocalId(avId):
                    self.__barrierContext = (context, name)
                    assert self.notify.debug('setBarrierData(%s, %s)' % (context, name))
                    return
        assert self.notify.debug('setBarrierData(%s)' % None)
        self.__barrierContext = None

    def getBarrierData(self):
        if False:
            while True:
                i = 10
        return ((0, '', []),)

    def doneBarrier(self, name=None):
        if False:
            while True:
                i = 10
        if self.__barrierContext is not None:
            (context, aiName) = self.__barrierContext
            if name is None or name == aiName:
                assert self.notify.debug('doneBarrier(%s, %s)' % (context, aiName))
                self.sendUpdate('setBarrierReady', [context])
                self.__barrierContext = None
            else:
                assert self.notify.debug('doneBarrier(%s) ignored; current barrier is %s' % (name, aiName))
        else:
            assert self.notify.debug('doneBarrier(%s) ignored; no active barrier.' % name)

    def addInterest(self, zoneId, note='', event=None):
        if False:
            return 10
        return self.cr.addInterest(self.getDoId(), zoneId, note, event)

    def removeInterest(self, handle, event=None):
        if False:
            for i in range(10):
                print('nop')
        return self.cr.removeInterest(handle, event)

    def b_setLocation(self, parentId, zoneId):
        if False:
            i = 10
            return i + 15
        self.d_setLocation(parentId, zoneId)
        self.setLocation(parentId, zoneId)

    def d_setLocation(self, parentId, zoneId):
        if False:
            return 10
        self.cr.sendSetLocation(self.doId, parentId, zoneId)

    def setLocation(self, parentId, zoneId):
        if False:
            i = 10
            return i + 15
        self.cr.storeObjectLocation(self, parentId, zoneId)

    def getLocation(self):
        if False:
            i = 10
            return i + 15
        try:
            if self.parentId == 0 and self.zoneId == 0:
                return None
            if self.parentId == 4294967295 and self.zoneId == 4294967295:
                return None
            return (self.parentId, self.zoneId)
        except AttributeError:
            return None

    def getParentObj(self):
        if False:
            while True:
                i = 10
        if self.parentId is None:
            return None
        return self.cr.doId2do.get(self.parentId)

    def isLocal(self):
        if False:
            for i in range(10):
                print('nop')
        return self.cr and self.cr.isLocalId(self.doId)

    def isGridParent(self):
        if False:
            print('Hello World!')
        return 0

    def execCommand(self, string, mwMgrId, avId, zoneId):
        if False:
            while True:
                i = 10
        pass