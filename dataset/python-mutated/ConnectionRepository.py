from panda3d.core import DocumentSpec, Filename, HTTPClient, VirtualFileSystem, getModelPath
from panda3d.direct import CConnectionRepository, DCPacker
from direct.task import Task
from direct.task.TaskManagerGlobal import taskMgr
from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.distributed.DoInterestManager import DoInterestManager
from direct.distributed.DoCollectionManager import DoCollectionManager
from direct.showbase import GarbageReport
from direct.showbase.MessengerGlobal import messenger
from .PyDatagramIterator import PyDatagramIterator
import gc
__all__ = ['ConnectionRepository', 'GCTrigger']

class ConnectionRepository(DoInterestManager, DoCollectionManager, CConnectionRepository):
    """
    This is a base class for things that know how to establish a
    connection (and exchange datagrams) with a gameserver.  This
    includes ClientRepository and AIRepository.
    """
    notify = directNotify.newCategory('ConnectionRepository')
    taskPriority = -30
    taskChain = None
    CM_HTTP = 0
    CM_NET = 1
    CM_NATIVE = 2
    gcNotify = directNotify.newCategory('GarbageCollect')
    GarbageCollectTaskName = 'allowGarbageCollect'
    GarbageThresholdTaskName = 'adjustGarbageCollectThreshold'

    def __init__(self, connectMethod, config, hasOwnerView=False, threadedNet=None):
        if False:
            for i in range(10):
                print('nop')
        assert self.notify.debugCall()
        if threadedNet is None:
            threadedNet = config.GetBool('threaded-net', False)
        CConnectionRepository.__init__(self, hasOwnerView, threadedNet)
        self.setWantMessageBundling(config.GetBool('want-message-bundling', 1))
        DoInterestManager.__init__(self)
        DoCollectionManager.__init__(self)
        self.setPythonRepository(self)
        self.uniqueId = hash(self)
        self.accept(self._getLostConnectionEvent(), self.lostConnection)
        self.config = config
        if self.config.GetBool('verbose-repository'):
            self.setVerbose(1)
        userConnectMethod = self.config.GetString('connect-method', 'default')
        if userConnectMethod == 'http':
            connectMethod = self.CM_HTTP
        elif userConnectMethod == 'net':
            connectMethod = self.CM_NET
        elif userConnectMethod == 'native':
            connectMethod = self.CM_NATIVE
        self.connectMethod = connectMethod
        if self.connectMethod == self.CM_HTTP:
            self.notify.info("Using connect method 'http'")
        elif self.connectMethod == self.CM_NET:
            self.notify.info("Using connect method 'net'")
        elif self.connectMethod == self.CM_NATIVE:
            self.notify.info("Using connect method 'native'")
        self.connectHttp = None
        self.http = None
        self.private__di = PyDatagramIterator()
        self.recorder = None
        self.readerPollTaskObj = None
        self.dcSuffix = ''
        self._serverAddress = ''
        if self.config.GetBool('gc-save-all', 0):
            gc.set_debug(gc.DEBUG_SAVEALL)
        if self.config.GetBool('want-garbage-collect-task', 1):
            taskMgr.add(self._garbageCollect, self.GarbageCollectTaskName, 200)
            taskMgr.doMethodLater(self.config.GetFloat('garbage-threshold-adjust-delay', 5 * 60.0), self._adjustGcThreshold, self.GarbageThresholdTaskName)
        self._gcDefaultThreshold = gc.get_threshold()

    def _getLostConnectionEvent(self):
        if False:
            for i in range(10):
                print('nop')
        return self.uniqueName('lostConnection')

    def _garbageCollect(self, task=None):
        if False:
            while True:
                i = 10
        gc.enable()
        gct = GCTrigger()
        gc.disable()
        return Task.cont

    def _adjustGcThreshold(self, task):
        if False:
            return 10
        numGarbage = GarbageReport.checkForGarbageLeaks()
        if numGarbage == 0:
            self.gcNotify.debug('no garbage found, doubling gc threshold')
            (a, b, c) = gc.get_threshold()
            gc.set_threshold(min(a * 2, 1 << 30), b, c)
            task.delayTime = task.delayTime * 2
            retVal = Task.again
        else:
            self.gcNotify.warning('garbage found, reverting gc threshold')
            gc.set_threshold(*self._gcDefaultThreshold)
            retVal = Task.done
        return retVal

    def generateGlobalObject(self, doId, dcname, values=None):
        if False:
            while True:
                i = 10

        def applyFieldValues(distObj, dclass, values):
            if False:
                for i in range(10):
                    print('nop')
            for i in range(dclass.getNumInheritedFields()):
                field = dclass.getInheritedField(i)
                if field.asMolecularField() is None:
                    value = values.get(field.getName(), None)
                    if value is None and field.isRequired():
                        packer = DCPacker()
                        packer.beginPack(field)
                        packer.packDefaultValue()
                        packer.endPack()
                        unpacker = DCPacker()
                        unpacker.setUnpackData(packer.getString())
                        unpacker.beginUnpack(field)
                        value = unpacker.unpackObject()
                        unpacker.endUnpack()
                    if value is not None:
                        function = getattr(distObj, field.getName())
                        if function is not None:
                            function(*value)
                        else:
                            self.notify.error('\n\n\nNot able to find %s.%s' % (distObj.__class__.__name__, field.getName()))
        dclass = self.dclassesByName.get(dcname + self.dcSuffix)
        if dclass is None:
            self.notify.warning('Need to define %s' % (dcname + self.dcSuffix))
            dclass = self.dclassesByName.get(dcname + 'AI')
        if dclass is None:
            dclass = self.dclassesByName.get(dcname)
        classDef = dclass.getClassDef()
        if classDef is None:
            self.notify.error('Could not create an undefined %s object.' % dclass.getName())
        distObj = classDef(self)
        distObj.dclass = dclass
        distObj.doId = doId
        self.doId2do[doId] = distObj
        distObj.generateInit()
        distObj.generate()
        if values is not None:
            applyFieldValues(distObj, dclass, values)
        distObj.announceGenerate()
        distObj.parentId = 0
        distObj.zoneId = 0
        return distObj

    def readDCFile(self, dcFileNames=None):
        if False:
            print('Hello World!')
        '\n        Reads in the dc files listed in dcFileNames, or if\n        dcFileNames is None, reads in all of the dc files listed in\n        the Config.prc file.\n        '
        dcFile = self.getDcFile()
        dcFile.clear()
        self.dclassesByName = {}
        self.dclassesByNumber = {}
        self.hashVal = 0
        if isinstance(dcFileNames, str):
            dcFileNames = [dcFileNames]
        dcImports = {}
        if dcFileNames is None:
            readResult = dcFile.readAll()
            if not readResult:
                self.notify.error('Could not read dc file.')
        else:
            searchPath = getModelPath().getValue()
            for dcFileName in dcFileNames:
                pathname = Filename(dcFileName)
                vfs = VirtualFileSystem.getGlobalPtr()
                vfs.resolveFilename(pathname, searchPath)
                readResult = dcFile.read(pathname)
                if not readResult:
                    self.notify.error('Could not read dc file: %s' % pathname)
        self.hashVal = dcFile.getHash()
        for n in range(dcFile.getNumImportModules()):
            moduleName = dcFile.getImportModule(n)[:]
            suffix = moduleName.split('/')
            moduleName = suffix[0]
            suffix = suffix[1:]
            if self.dcSuffix in suffix:
                moduleName += self.dcSuffix
            elif self.dcSuffix == 'UD' and 'AI' in suffix:
                moduleName += 'AI'
            importSymbols = []
            for i in range(dcFile.getNumImportSymbols(n)):
                symbolName = dcFile.getImportSymbol(n, i)
                suffix = symbolName.split('/')
                symbolName = suffix[0]
                suffix = suffix[1:]
                if self.dcSuffix in suffix:
                    symbolName += self.dcSuffix
                elif self.dcSuffix == 'UD' and 'AI' in suffix:
                    symbolName += 'AI'
                importSymbols.append(symbolName)
            self.importModule(dcImports, moduleName, importSymbols)
        import inspect
        for i in range(dcFile.getNumClasses()):
            dclass = dcFile.getClass(i)
            number = dclass.getNumber()
            className = dclass.getName() + self.dcSuffix
            classDef = dcImports.get(className)
            if classDef is None and self.dcSuffix == 'UD':
                className = dclass.getName() + 'AI'
                classDef = dcImports.get(className)
            if classDef is None:
                className = dclass.getName()
                classDef = dcImports.get(className)
            if classDef is None:
                self.notify.debug('No class definition for %s.' % className)
            else:
                if inspect.ismodule(classDef):
                    if not hasattr(classDef, className):
                        self.notify.warning('Module %s does not define class %s.' % (className, className))
                        continue
                    classDef = getattr(classDef, className)
                if not inspect.isclass(classDef):
                    self.notify.error('Symbol %s is not a class name.' % className)
                else:
                    dclass.setClassDef(classDef)
            self.dclassesByName[className] = dclass
            if number >= 0:
                self.dclassesByNumber[number] = dclass
        if self.hasOwnerView():
            ownerDcSuffix = self.dcSuffix + 'OV'
            ownerImportSymbols = {}
            for n in range(dcFile.getNumImportModules()):
                moduleName = dcFile.getImportModule(n)
                suffix = moduleName.split('/')
                moduleName = suffix[0]
                suffix = suffix[1:]
                if ownerDcSuffix in suffix:
                    moduleName = moduleName + ownerDcSuffix
                importSymbols = []
                for i in range(dcFile.getNumImportSymbols(n)):
                    symbolName = dcFile.getImportSymbol(n, i)
                    suffix = symbolName.split('/')
                    symbolName = suffix[0]
                    suffix = suffix[1:]
                    if ownerDcSuffix in suffix:
                        symbolName += ownerDcSuffix
                    importSymbols.append(symbolName)
                    ownerImportSymbols[symbolName] = None
                self.importModule(dcImports, moduleName, importSymbols)
            for i in range(dcFile.getNumClasses()):
                dclass = dcFile.getClass(i)
                if dclass.getName() + ownerDcSuffix in ownerImportSymbols:
                    number = dclass.getNumber()
                    className = dclass.getName() + ownerDcSuffix
                    classDef = dcImports.get(className)
                    if classDef is None:
                        self.notify.error('No class definition for %s.' % className)
                    else:
                        if inspect.ismodule(classDef):
                            if not hasattr(classDef, className):
                                self.notify.error('Module %s does not define class %s.' % (className, className))
                            classDef = getattr(classDef, className)
                        dclass.setOwnerClassDef(classDef)
                        self.dclassesByName[className] = dclass

    def importModule(self, dcImports, moduleName, importSymbols):
        if False:
            i = 10
            return i + 15
        '\n        Imports the indicated moduleName and all of its symbols\n        into the current namespace.  This more-or-less reimplements\n        the Python import command.\n        '
        module = __import__(moduleName, globals(), locals(), importSymbols)
        if importSymbols:
            if importSymbols == ['*']:
                if hasattr(module, '__all__'):
                    importSymbols = module.__all__
                else:
                    importSymbols = module.__dict__.keys()
            for symbolName in importSymbols:
                if hasattr(module, symbolName):
                    dcImports[symbolName] = getattr(module, symbolName)
                else:
                    raise Exception('Symbol %s not defined in module %s.' % (symbolName, moduleName))
        else:
            components = moduleName.split('.')
            dcImports[components[0]] = module

    def getServerAddress(self):
        if False:
            for i in range(10):
                print('nop')
        return self._serverAddress

    def connect(self, serverList, successCallback=None, successArgs=[], failureCallback=None, failureArgs=[]):
        if False:
            print('Hello World!')
        '\n        Attempts to establish a connection to the server.  May return\n        before the connection is established.  The two callbacks\n        represent the two functions to call (and their arguments) on\n        success or failure, respectively.  The failure callback also\n        gets one additional parameter, which will be passed in first:\n        the return status code giving reason for failure, if it is\n        known.\n        '
        hasProxy = 0
        if self.checkHttp():
            proxies = self.http.getProxiesForUrl(serverList[0])
            hasProxy = proxies != 'DIRECT'
        if hasProxy:
            self.notify.info('Connecting to gameserver via proxy list: %s' % proxies)
        else:
            self.notify.info('Connecting to gameserver directly (no proxy).')
        self.bootedIndex = None
        self.bootedText = None
        if self.connectMethod == self.CM_HTTP:
            ch = self.http.makeChannel(0)
            self.httpConnectCallback(ch, serverList, 0, successCallback, successArgs, failureCallback, failureArgs)
        elif self.connectMethod == self.CM_NET or not hasattr(self, 'connectNative'):
            for url in serverList:
                self.notify.info('Connecting to %s via NET interface.' % url)
                if self.tryConnectNet(url):
                    self.startReaderPollTask()
                    if successCallback:
                        successCallback(*successArgs)
                    return
            if failureCallback:
                failureCallback(0, '', *failureArgs)
        elif self.connectMethod == self.CM_NATIVE:
            for url in serverList:
                self.notify.info('Connecting to %s via Native interface.' % url)
                if self.connectNative(url):
                    self.startReaderPollTask()
                    if successCallback:
                        successCallback(*successArgs)
                    return
            if failureCallback:
                failureCallback(0, '', *failureArgs)
        else:
            print("uh oh, we aren't using one of the tri-state CM variables")
            failureCallback(0, '', *failureArgs)

    def disconnect(self):
        if False:
            return 10
        '\n        Closes the previously-established connection.\n        '
        self.notify.info('Closing connection to server.')
        self._serverAddress = ''
        CConnectionRepository.disconnect(self)
        self.stopReaderPollTask()

    def shutdown(self):
        if False:
            return 10
        self.ignoreAll()
        CConnectionRepository.shutdown(self)

    def httpConnectCallback(self, ch, serverList, serverIndex, successCallback, successArgs, failureCallback, failureArgs):
        if False:
            for i in range(10):
                print('nop')
        if ch.isConnectionReady():
            self.setConnectionHttp(ch)
            self._serverAddress = serverList[serverIndex - 1]
            self.notify.info('Successfully connected to %s.' % self._serverAddress)
            self.startReaderPollTask()
            if successCallback:
                successCallback(*successArgs)
        elif serverIndex < len(serverList):
            url = serverList[serverIndex]
            self.notify.info('Connecting to %s via HTTP interface.' % url)
            ch.preserveStatus()
            ch.beginConnectTo(DocumentSpec(url))
            ch.spawnTask(name='connect-to-server', callback=self.httpConnectCallback, extraArgs=[ch, serverList, serverIndex + 1, successCallback, successArgs, failureCallback, failureArgs])
        elif failureCallback:
            failureCallback(ch.getStatusCode(), ch.getStatusString(), *failureArgs)

    def checkHttp(self):
        if False:
            for i in range(10):
                print('nop')
        if self.http is None:
            try:
                self.http = HTTPClient()
            except Exception:
                pass
        return self.http

    def startReaderPollTask(self):
        if False:
            for i in range(10):
                print('nop')
        self.stopReaderPollTask()
        self.accept(CConnectionRepository.getOverflowEventName(), self.handleReaderOverflow)
        self.readerPollTaskObj = taskMgr.add(self.readerPollUntilEmpty, self.uniqueName('readerPollTask'), priority=self.taskPriority, taskChain=self.taskChain)

    def stopReaderPollTask(self):
        if False:
            while True:
                i = 10
        if self.readerPollTaskObj:
            taskMgr.remove(self.readerPollTaskObj)
            self.readerPollTaskObj = None
        self.ignore(CConnectionRepository.getOverflowEventName())

    def readerPollUntilEmpty(self, task):
        if False:
            print('Hello World!')
        while self.readerPollOnce():
            pass
        return Task.cont

    def readerPollOnce(self):
        if False:
            while True:
                i = 10
        if self.checkDatagram():
            self.getDatagramIterator(self.private__di)
            self.handleDatagram(self.private__di)
            return 1
        if not self.isConnected():
            self.stopReaderPollTask()
            messenger.send(self.uniqueName('lostConnection'), taskChain='default')
        return 0

    def handleReaderOverflow(self):
        if False:
            return 10
        pass

    def lostConnection(self):
        if False:
            while True:
                i = 10
        self.notify.warning('Lost connection to gameserver.')

    def handleDatagram(self, di):
        if False:
            while True:
                i = 10
        pass

    def send(self, datagram):
        if False:
            for i in range(10):
                print('nop')
        if datagram.getLength() > 0:
            self.sendDatagram(datagram)

    def pullNetworkPlug(self):
        if False:
            i = 10
            return i + 15
        self.notify.warning('*** SIMULATING A NETWORK-PLUG-PULL ***')
        self.setSimulatedDisconnect(1)

    def networkPlugPulled(self):
        if False:
            for i in range(10):
                print('nop')
        return self.getSimulatedDisconnect()

    def restoreNetworkPlug(self):
        if False:
            for i in range(10):
                print('nop')
        if self.networkPlugPulled():
            self.notify.info('*** RESTORING SIMULATED PULLED-NETWORK-PLUG ***')
            self.setSimulatedDisconnect(0)

    def uniqueName(self, idString):
        if False:
            return 10
        return '%s-%s' % (idString, self.uniqueId)

class GCTrigger:
    pass