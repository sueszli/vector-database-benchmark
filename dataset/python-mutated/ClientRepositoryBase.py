from panda3d.core import ClockObject, ConfigVariableBool, ConfigVariableDouble, Datagram, DatagramIterator
from direct.task import Task
from direct.task.TaskManagerGlobal import taskMgr
from direct.directnotify import DirectNotifyGlobal
from direct.distributed.CRDataCache import CRDataCache
from direct.distributed.ConnectionRepository import ConnectionRepository
from direct.showbase.PythonUtil import safeRepr, itype, makeList
from direct.showbase.MessengerGlobal import messenger
from .MsgTypes import CLIENT_ENTER_OBJECT_REQUIRED_OTHER, MsgId2Names
from . import CRCache
from . import ParentMgr
from . import RelatedObjectMgr
import time

class ClientRepositoryBase(ConnectionRepository):
    """
    This maintains a client-side connection with a Panda server.

    This base class exists to collect the common code between
    ClientRepository, which is the CMU-provided, open-source version
    of the client repository code, and OTPClientRepository, which is
    the VR Studio's implementation of the same.
    """
    notify = DirectNotifyGlobal.directNotify.newCategory('ClientRepositoryBase')

    def __init__(self, dcFileNames=None, dcSuffix='', connectMethod=None, threadedNet=None):
        if False:
            print('Hello World!')
        if connectMethod is None:
            connectMethod = self.CM_HTTP
        ConnectionRepository.__init__(self, connectMethod, base.config, hasOwnerView=True, threadedNet=threadedNet)
        self.dcSuffix = dcSuffix
        if hasattr(self, 'setVerbose'):
            if ConfigVariableBool('verbose-clientrepository', False):
                self.setVerbose(1)
        self.context = 100000
        self.setClientDatagram(1)
        self.deferredGenerates = []
        self.deferredDoIds = {}
        self.lastGenerate = 0
        self.setDeferInterval(ConfigVariableDouble('deferred-generate-interval', 0.2).value)
        self.noDefer = False
        self.recorder = base.recorder
        self.readDCFile(dcFileNames)
        self.cache = CRCache.CRCache()
        self.doDataCache = CRDataCache()
        self.cacheOwner = CRCache.CRCache()
        self.serverDelta = 0
        self.bootedIndex = None
        self.bootedText = None
        self.parentMgr = ParentMgr.ParentMgr()
        self.relatedObjectMgr = RelatedObjectMgr.RelatedObjectMgr(self)
        self.timeManager = None
        self.heartbeatInterval = ConfigVariableDouble('heartbeat-interval', 10).value
        self.heartbeatStarted = 0
        self.lastHeartbeat = 0
        self._delayDeletedDOs = {}
        self.specialNameNumber = 0

    def setDeferInterval(self, deferInterval):
        if False:
            i = 10
            return i + 15
        'Specifies the minimum amount of time, in seconds, that must\n        elapse before generating any two DistributedObjects whose\n        class type is marked "deferrable".  Set this to 0 to indicate\n        no deferring will occur.'
        self.deferInterval = deferInterval
        self.setHandleCUpdates(self.deferInterval == 0)
        if self.deferredGenerates:
            taskMgr.remove('deferredGenerate')
            taskMgr.doMethodLater(self.deferInterval, self.doDeferredGenerate, 'deferredGenerate')

    def specialName(self, label):
        if False:
            for i in range(10):
                print('nop')
        name = f'SpecialName {self.specialNameNumber} {label}'
        self.specialNameNumber += 1
        return name

    def getTables(self, ownerView):
        if False:
            for i in range(10):
                print('nop')
        if ownerView:
            return (self.doId2ownerView, self.cacheOwner)
        else:
            return (self.doId2do, self.cache)

    def _getMsgName(self, msgId):
        if False:
            while True:
                i = 10
        return makeList(MsgId2Names.get(msgId, f'UNKNOWN MESSAGE: {msgId}'))[0]

    def allocateContext(self):
        if False:
            i = 10
            return i + 15
        self.context += 1
        return self.context

    def setServerDelta(self, delta):
        if False:
            while True:
                i = 10
        "\n        Indicates the approximate difference in seconds between the\n        client's clock and the server's clock, in universal time (not\n        including timezone shifts).  This is mainly useful for\n        reporting synchronization information to the logs; don't\n        depend on it for any precise timing requirements.\n\n        Also see Notify.setServerDelta(), which also accounts for a\n        timezone shift.\n        "
        self.serverDelta = delta

    def getServerDelta(self):
        if False:
            i = 10
            return i + 15
        return self.serverDelta

    def getServerTimeOfDay(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns the current time of day (seconds elapsed since the\n        1972 epoch) according to the server's clock.  This is in GMT,\n        and hence is irrespective of timezones.\n\n        The value is computed based on the client's clock and the\n        known delta from the server's clock, which is not terribly\n        precisely measured and may drift slightly after startup, but\n        it should be accurate plus or minus a couple of seconds.\n        "
        return time.time() + self.serverDelta

    def doGenerate(self, parentId, zoneId, classId, doId, di):
        if False:
            for i in range(10):
                print('nop')
        assert parentId == self.GameGlobalsId or parentId in self.doId2do
        dclass = self.dclassesByNumber[classId]
        assert self.notify.debug(f'performing generate for {dclass.getName()} {doId}')
        dclass.startGenerate()
        distObj = self.generateWithRequiredOtherFields(dclass, doId, di, parentId, zoneId)
        dclass.stopGenerate()

    def flushGenerates(self):
        if False:
            print('Hello World!')
        ' Forces all pending generates to be performed immediately. '
        while self.deferredGenerates:
            (msgType, extra) = self.deferredGenerates[0]
            del self.deferredGenerates[0]
            self.replayDeferredGenerate(msgType, extra)
        taskMgr.remove('deferredGenerate')

    def replayDeferredGenerate(self, msgType, extra):
        if False:
            return 10
        ' Override this to do something appropriate with deferred\n        "generate" messages when they are replayed().\n        '
        if msgType == CLIENT_ENTER_OBJECT_REQUIRED_OTHER:
            doId = extra
            if doId in self.deferredDoIds:
                (args, deferrable, dg, updates) = self.deferredDoIds[doId]
                del self.deferredDoIds[doId]
                self.doGenerate(*args)
                if deferrable:
                    self.lastGenerate = ClockObject.getGlobalClock().getFrameTime()
                for (dg, di) in updates:
                    if isinstance(di, tuple):
                        msgType = dg
                        (dg, di) = di
                        self.replayDeferredGenerate(msgType, (dg, di))
                    else:
                        self.__doUpdate(doId, di, True)
        else:
            self.notify.warning('Ignoring deferred message %s' % msgType)

    def doDeferredGenerate(self, task):
        if False:
            return 10
        ' This is the task that generates an object on the deferred\n        queue. '
        now = ClockObject.getGlobalClock().getFrameTime()
        while self.deferredGenerates:
            if now - self.lastGenerate < self.deferInterval:
                return Task.again
            (msgType, extra) = self.deferredGenerates[0]
            del self.deferredGenerates[0]
            self.replayDeferredGenerate(msgType, extra)
        return Task.done

    def generateWithRequiredFields(self, dclass, doId, di, parentId, zoneId):
        if False:
            print('Hello World!')
        if doId in self.doId2do:
            distObj = self.doId2do[doId]
            assert distObj.dclass == dclass
            distObj.generate()
            distObj.setLocation(parentId, zoneId)
            distObj.updateRequiredFields(dclass, di)
        elif self.cache.contains(doId):
            distObj = self.cache.retrieve(doId)
            assert distObj.dclass == dclass
            self.doId2do[doId] = distObj
            distObj.generate()
            distObj.parentId = None
            distObj.zoneId = None
            distObj.setLocation(parentId, zoneId)
            distObj.updateRequiredFields(dclass, di)
        else:
            classDef = dclass.getClassDef()
            if classDef is None:
                self.notify.error('Could not create an undefined %s object.' % dclass.getName())
            distObj = classDef(self)
            distObj.dclass = dclass
            distObj.doId = doId
            self.doId2do[doId] = distObj
            distObj.generateInit()
            distObj._retrieveCachedData()
            distObj.generate()
            distObj.setLocation(parentId, zoneId)
            distObj.updateRequiredFields(dclass, di)
            self.notify.debug('New DO:%s, dclass:%s' % (doId, dclass.getName()))
        return distObj

    def generateWithRequiredOtherFields(self, dclass, doId, di, parentId=None, zoneId=None):
        if False:
            i = 10
            return i + 15
        if doId in self.doId2do:
            distObj = self.doId2do[doId]
            assert distObj.dclass == dclass
            distObj.generate()
            distObj.setLocation(parentId, zoneId)
            distObj.updateRequiredOtherFields(dclass, di)
        elif self.cache.contains(doId):
            distObj = self.cache.retrieve(doId)
            assert distObj.dclass == dclass
            self.doId2do[doId] = distObj
            distObj.generate()
            distObj.parentId = None
            distObj.zoneId = None
            distObj.setLocation(parentId, zoneId)
            distObj.updateRequiredOtherFields(dclass, di)
        else:
            classDef = dclass.getClassDef()
            if classDef is None:
                self.notify.error('Could not create an undefined %s object.' % dclass.getName())
            distObj = classDef(self)
            distObj.dclass = dclass
            distObj.doId = doId
            self.doId2do[doId] = distObj
            distObj.generateInit()
            distObj._retrieveCachedData()
            distObj.generate()
            distObj.setLocation(parentId, zoneId)
            distObj.updateRequiredOtherFields(dclass, di)
        return distObj

    def generateWithRequiredOtherFieldsOwner(self, dclass, doId, di):
        if False:
            while True:
                i = 10
        if doId in self.doId2ownerView:
            self.notify.error('duplicate owner generate for %s (%s)' % (doId, dclass.getName()))
            distObj = self.doId2ownerView[doId]
            assert distObj.dclass == dclass
            distObj.generate()
            distObj.updateRequiredOtherFields(dclass, di)
        elif self.cacheOwner.contains(doId):
            distObj = self.cacheOwner.retrieve(doId)
            assert distObj.dclass == dclass
            self.doId2ownerView[doId] = distObj
            distObj.generate()
            distObj.updateRequiredOtherFields(dclass, di)
        else:
            classDef = dclass.getOwnerClassDef()
            if classDef is None:
                self.notify.error('Could not create an undefined %s object. Have you created an owner view?' % dclass.getName())
            distObj = classDef(self)
            distObj.dclass = dclass
            distObj.doId = doId
            self.doId2ownerView[doId] = distObj
            distObj.generateInit()
            distObj.generate()
            distObj.updateRequiredOtherFields(dclass, di)
        return distObj

    def disableDoId(self, doId, ownerView=False):
        if False:
            return 10
        (table, cache) = self.getTables(ownerView)
        if doId in table:
            distObj = table[doId]
            del table[doId]
            cached = False
            if distObj.getCacheable() and distObj.getDelayDeleteCount() <= 0:
                cached = cache.cache(distObj)
            if not cached:
                distObj.deleteOrDelay()
                if distObj.getDelayDeleteCount() <= 0:
                    distObj.detectLeaks()
        elif doId in self.deferredDoIds:
            del self.deferredDoIds[doId]
            i = self.deferredGenerates.index((CLIENT_ENTER_OBJECT_REQUIRED_OTHER, doId))
            del self.deferredGenerates[i]
            if len(self.deferredGenerates) == 0:
                taskMgr.remove('deferredGenerate')
        else:
            self._logFailedDisable(doId, ownerView)

    def _logFailedDisable(self, doId, ownerView):
        if False:
            print('Hello World!')
        self.notify.warning('Disable failed. DistObj ' + str(doId) + ' is not in dictionary, ownerView=%s' % ownerView)

    def handleDelete(self, di):
        if False:
            return 10
        assert 0

    def handleUpdateField(self, di):
        if False:
            i = 10
            return i + 15
        '\n        This method is called when a CLIENT_OBJECT_UPDATE_FIELD\n        message is received; it decodes the update, unpacks the\n        arguments, and calls the corresponding method on the indicated\n        DistributedObject.\n\n        In fact, this method is exactly duplicated by the C++ method\n        cConnectionRepository::handle_update_field(), which was\n        written to optimize the message loop by handling all of the\n        CLIENT_OBJECT_UPDATE_FIELD messages in C++.  That means that\n        nowadays, this Python method will probably never be called,\n        since UPDATE_FIELD messages will not even be passed to the\n        Python message handlers.  But this method remains for\n        documentation purposes, and also as a "just in case" handler\n        in case we ever do come across a situation in the future in\n        which python might handle the UPDATE_FIELD message.\n        '
        doId = di.getUint32()
        ovUpdated = self.__doUpdateOwner(doId, di)
        if doId in self.deferredDoIds:
            (args, deferrable, dg0, updates) = self.deferredDoIds[doId]
            dg = Datagram(di.getDatagram())
            di = DatagramIterator(dg, di.getCurrentIndex())
            updates.append((dg, di))
        else:
            self.__doUpdate(doId, di, ovUpdated)

    def __doUpdate(self, doId, di, ovUpdated):
        if False:
            for i in range(10):
                print('nop')
        do = self.doId2do.get(doId)
        if do is not None:
            do.dclass.receiveUpdate(do, di)
        elif not ovUpdated:
            try:
                handle = self.identifyAvatar(doId)
                if handle:
                    dclass = self.dclassesByName[handle.dclassName]
                    dclass.receiveUpdate(handle, di)
                else:
                    self.notify.warning(f'Asked to update non-existent DistObj {doId}')
            except Exception:
                self.notify.warning(f'Asked to update non-existent DistObj {doId} and failed to find it')

    def __doUpdateOwner(self, doId, di):
        if False:
            print('Hello World!')
        if not self.hasOwnerView():
            return False
        ovObj = self.doId2ownerView.get(doId)
        if ovObj:
            odg = Datagram(di.getDatagram())
            odi = DatagramIterator(odg, di.getCurrentIndex())
            ovObj.dclass.receiveUpdate(ovObj, odi)
            return True
        return False

    def handleGoGetLost(self, di):
        if False:
            i = 10
            return i + 15
        if di.getRemainingSize() > 0:
            self.bootedIndex = di.getUint16()
            self.bootedText = di.getString()
            self.notify.warning(f'Server is booting us out ({self.bootedIndex}): {self.bootedText}')
        else:
            self.bootedIndex = None
            self.bootedText = None
            self.notify.warning('Server is booting us out with no explanation.')
        self.stopReaderPollTask()
        self.lostConnection()

    def handleServerHeartbeat(self, di):
        if False:
            print('Hello World!')
        if ConfigVariableBool('server-heartbeat-info', True):
            self.notify.info('Server heartbeat.')

    def handleSystemMessage(self, di):
        if False:
            i = 10
            return i + 15
        message = di.getString()
        self.notify.info('Message from server: %s' % message)
        return message

    def handleSystemMessageAknowledge(self, di):
        if False:
            i = 10
            return i + 15
        message = di.getString()
        self.notify.info('Message with aknowledge from server: %s' % message)
        messenger.send('system message aknowledge', [message])
        return message

    def getObjectsOfClass(self, objClass):
        if False:
            return 10
        " returns dict of doId:object, containing all objects\n        that inherit from 'class'. returned dict is safely mutable. "
        doDict = {}
        for (doId, do) in self.doId2do.items():
            if isinstance(do, objClass):
                doDict[doId] = do
        return doDict

    def getObjectsOfExactClass(self, objClass):
        if False:
            for i in range(10):
                print('nop')
        " returns dict of doId:object, containing all objects that\n        are exactly of type 'class' (neglecting inheritance). returned\n        dict is safely mutable. "
        doDict = {}
        for (doId, do) in self.doId2do.items():
            if do.__class__ == objClass:
                doDict[doId] = do
        return doDict

    def considerHeartbeat(self):
        if False:
            for i in range(10):
                print('nop')
        "Send a heartbeat message if we haven't sent one recently."
        if not self.heartbeatStarted:
            self.notify.debug('Heartbeats not started; not sending.')
            return
        elapsed = ClockObject.getGlobalClock().getRealTime() - self.lastHeartbeat
        if elapsed < 0 or elapsed > self.heartbeatInterval:
            self.notify.info('Sending heartbeat mid-frame.')
            self.startHeartbeat()

    def stopHeartbeat(self):
        if False:
            i = 10
            return i + 15
        taskMgr.remove('heartBeat')
        self.heartbeatStarted = 0

    def startHeartbeat(self):
        if False:
            while True:
                i = 10
        self.stopHeartbeat()
        self.heartbeatStarted = 1
        self.sendHeartbeat()
        self.waitForNextHeartBeat()

    def sendHeartbeatTask(self, task):
        if False:
            i = 10
            return i + 15
        self.sendHeartbeat()
        return Task.again

    def waitForNextHeartBeat(self):
        if False:
            while True:
                i = 10
        taskMgr.doMethodLater(self.heartbeatInterval, self.sendHeartbeatTask, 'heartBeat', taskChain='net')

    def replaceMethod(self, oldMethod, newFunction):
        if False:
            return 10
        return 0

    def getWorld(self, doId):
        if False:
            i = 10
            return i + 15
        obj = self.doId2do[doId]
        worldNP = obj.getParent()
        while 1:
            nextNP = worldNP.getParent()
            if nextNP == render:
                break
            elif worldNP.isEmpty():
                return None
        return worldNP

    def isLive(self):
        if False:
            while True:
                i = 10
        if ConfigVariableBool('force-live', False):
            return True
        return not (__dev__ or launcher.isTestServer())

    def isLocalId(self, id):
        if False:
            while True:
                i = 10
        return 0

    def _addDelayDeletedDO(self, do):
        if False:
            return 10
        key = id(do)
        assert key not in self._delayDeletedDOs
        self._delayDeletedDOs[key] = do

    def _removeDelayDeletedDO(self, do):
        if False:
            print('Hello World!')
        key = id(do)
        del self._delayDeletedDOs[key]

    def printDelayDeletes(self):
        if False:
            print('Hello World!')
        print('DelayDeletes:')
        print('=============')
        for obj in self._delayDeletedDOs.values():
            print('%s\t%s (%s)\tdelayDeletes=%s' % (obj.doId, safeRepr(obj), itype(obj), obj.getDelayDeleteNames()))