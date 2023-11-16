from panda3d.core import ClockObject, ConnectionWriter, NetAddress, PointerToConnection, QueuedConnectionListener, QueuedConnectionManager, QueuedConnectionReader, Vec3
from .ClusterMsgs import CLUSTER_CAM_FRUSTUM, CLUSTER_CAM_MOVEMENT, CLUSTER_CAM_OFFSET, CLUSTER_COMMAND_STRING, CLUSTER_DAEMON_PORT, CLUSTER_EXIT, CLUSTER_NAMED_MOVEMENT_DONE, CLUSTER_NAMED_OBJECT_MOVEMENT, CLUSTER_NONE, CLUSTER_SELECTED_MOVEMENT, CLUSTER_SERVER_PORT, CLUSTER_SWAP_NOW, CLUSTER_SWAP_READY, CLUSTER_TIME_DATA, ClusterMsgHandler
from direct.directnotify import DirectNotifyGlobal
from direct.showbase import DirectObject
from direct.task import Task
from direct.task.TaskManagerGlobal import taskMgr
import builtins

class ClusterServer(DirectObject.DirectObject):
    notify = DirectNotifyGlobal.directNotify.newCategory('ClusterServer')
    MSG_NUM = 2000000

    def __init__(self, cameraJig, camera):
        if False:
            return 10
        global clusterServerPort, clusterSyncFlag
        global clusterDaemonClient, clusterDaemonPort
        self.cameraJig = cameraJig
        self.camera = camera
        self.lens = camera.node().getLens()
        self.lastConnection = None
        self.fPosReceived = 0
        self.qcm = QueuedConnectionManager()
        self.qcl = QueuedConnectionListener(self.qcm, 0)
        self.qcr = QueuedConnectionReader(self.qcm, 0)
        self.cw = ConnectionWriter(self.qcm, 0)
        try:
            port = clusterServerPort
        except NameError:
            port = CLUSTER_SERVER_PORT
        self.tcpRendezvous = self.qcm.openTCPServerRendezvous(port, 1)
        self.qcl.addConnection(self.tcpRendezvous)
        self.msgHandler = ClusterMsgHandler(ClusterServer.MSG_NUM, self.notify)
        self.startListenerPollTask()
        self.startReaderPollTask()
        try:
            clusterSyncFlag
        except NameError:
            clusterSyncFlag = 0
        if clusterSyncFlag:
            self.startSwapCoordinator()
            base.graphicsEngine.setAutoFlip(0)
        ClockObject.getGlobalClock().setMode(ClockObject.MSlave)
        self.daemon = DirectD()
        self.objectMappings = {}
        self.objectHasColor = {}
        self.controlMappings = {}
        self.controlPriorities = {}
        self.controlOffsets = {}
        self.messageQueue = []
        self.sortedControlMappings = []
        try:
            clusterDaemonClient
        except NameError:
            clusterDaemonClient = 'localhost'
        try:
            clusterDaemonPort
        except NameError:
            clusterDaemonPort = CLUSTER_DAEMON_PORT
        self.daemon.serverReady(clusterDaemonClient, clusterDaemonPort)

    def startListenerPollTask(self):
        if False:
            while True:
                i = 10
        taskMgr.add(self.listenerPollTask, 'serverListenerPollTask', -40)

    def listenerPollTask(self, task):
        if False:
            print('Hello World!')
        ' Task to listen for a new connection from the client '
        if self.qcl.newConnectionAvailable():
            self.notify.info('New connection is available')
            rendezvous = PointerToConnection()
            netAddress = NetAddress()
            newConnection = PointerToConnection()
            if self.qcl.getNewConnection(rendezvous, netAddress, newConnection):
                newConnection = newConnection.p()
                self.qcr.addConnection(newConnection)
                self.lastConnection = newConnection
                self.notify.info('Got a connection!')
            else:
                self.notify.warning('getNewConnection returned false')
        return Task.cont

    def addNamedObjectMapping(self, object, name, hasColor=True, priority=0):
        if False:
            for i in range(10):
                print('nop')
        if name not in self.objectMappings:
            self.objectMappings[name] = object
            self.objectHasColor[name] = hasColor
        else:
            self.notify.debug('attempt to add duplicate named object: ' + name)

    def removeObjectMapping(self, name):
        if False:
            return 10
        if name in self.objectMappings:
            self.objectMappings.pop(name)

    def redoSortedPriorities(self):
        if False:
            return 10
        self.sortedControlMappings = sorted(([self.controlPriorities[key], key] for key in self.objectMappings))

    def addControlMapping(self, objectName, controlledName, offset=None, priority=0):
        if False:
            print('Hello World!')
        if objectName not in self.controlMappings:
            self.controlMappings[objectName] = controlledName
            if offset is None:
                offset = Vec3(0, 0, 0)
            self.controlOffsets[objectName] = offset
            self.controlPriorities[objectName] = priority
            self.redoSortedPriorities()
        else:
            self.notify.debug('attempt to add duplicate controlled object: ' + objectName)

    def setControlMappingOffset(self, objectName, offset):
        if False:
            return 10
        if objectName in self.controlMappings:
            self.controlOffsets[objectName] = offset

    def removeControlMapping(self, name):
        if False:
            i = 10
            return i + 15
        if name in self.controlMappings:
            self.controlMappings.pop(name)
            self.controlPriorities.pop(name)
        self.redoSortedPriorities()

    def startControlObjectTask(self):
        if False:
            while True:
                i = 10
        self.notify.debug('moving control objects')
        taskMgr.add(self.controlObjectTask, 'controlObjectTask', 50)

    def controlObjectTask(self, task):
        if False:
            return 10
        for pair in self.sortedControlPriorities:
            object = pair[1]
            name = self.controlMappings[object]
            if object in self.objectMappings:
                self.moveObject(self.objectMappings[object], name, self.controlOffsets[object], self.objectHasColor[object])
        self.sendNamedMovementDone()
        return Task.cont

    def sendNamedMovementDone(self):
        if False:
            while True:
                i = 10
        self.notify.debug('named movement done')
        datagram = self.msgHandler.makeNamedMovementDone()
        self.cw.send(datagram, self.lastConnection)

    def moveObject(self, nodePath, object, offset, hasColor):
        if False:
            while True:
                i = 10
        self.notify.debug('moving object ' + object)
        xyz = nodePath.getPos(render) + offset
        hpr = nodePath.getHpr(render)
        scale = nodePath.getScale(render)
        if hasColor:
            color = nodePath.getColor()
        else:
            color = [1, 1, 1, 1]
        hidden = nodePath.isHidden()
        datagram = self.msgHandler.makeNamedObjectMovementDatagram(xyz, hpr, scale, color, hidden, object)
        self.cw.send(datagram, self.lastConnection)

    def startReaderPollTask(self):
        if False:
            for i in range(10):
                print('nop')
        ' Task to handle datagrams from client '
        if clusterSyncFlag:
            taskMgr.add(self._syncReaderPollTask, 'serverReaderPollTask', -39)
        else:
            taskMgr.add(self._readerPollTask, 'serverReaderPollTask', -39)

    def _readerPollTask(self, state):
        if False:
            while True:
                i = 10
        ' Non blocking task to read all available datagrams '
        while 1:
            (datagram, dgi, type) = self.msgHandler.nonBlockingRead(self.qcr)
            if type is CLUSTER_NONE:
                break
            else:
                self.handleDatagram(dgi, type)
        return Task.cont

    def _syncReaderPollTask(self, task):
        if False:
            return 10
        if self.lastConnection is None:
            pass
        elif self.qcr.isConnectionOk(self.lastConnection):
            type = CLUSTER_NONE
            while type != CLUSTER_CAM_MOVEMENT:
                (datagram, dgi, type) = self.msgHandler.blockingRead(self.qcr)
                self.handleDatagram(dgi, type)
        return Task.cont

    def startSwapCoordinator(self):
        if False:
            return 10
        taskMgr.add(self.swapCoordinatorTask, 'serverSwapCoordinator', 51)

    def swapCoordinatorTask(self, task):
        if False:
            for i in range(10):
                print('nop')
        if self.fPosReceived:
            self.fPosReceived = 0
            self.sendSwapReady()
            while 1:
                (datagram, dgi, type) = self.msgHandler.blockingRead(self.qcr)
                self.handleDatagram(dgi, type)
                if type == CLUSTER_SWAP_NOW:
                    break
        return Task.cont

    def sendSwapReady(self):
        if False:
            for i in range(10):
                print('nop')
        self.notify.debug('send swap ready packet %d' % self.msgHandler.packetNumber)
        datagram = self.msgHandler.makeSwapReadyDatagram()
        self.cw.send(datagram, self.lastConnection)

    def handleDatagram(self, dgi, type):
        if False:
            i = 10
            return i + 15
        ' Process a datagram depending upon type flag '
        if type == CLUSTER_NONE:
            pass
        elif type == CLUSTER_EXIT:
            print('GOT EXIT')
            import sys
            sys.exit()
        elif type == CLUSTER_CAM_OFFSET:
            self.handleCamOffset(dgi)
        elif type == CLUSTER_CAM_FRUSTUM:
            self.handleCamFrustum(dgi)
        elif type == CLUSTER_CAM_MOVEMENT:
            self.handleCamMovement(dgi)
        elif type == CLUSTER_SELECTED_MOVEMENT:
            self.handleSelectedMovement(dgi)
        elif type == CLUSTER_COMMAND_STRING:
            self.handleCommandString(dgi)
        elif type == CLUSTER_SWAP_READY:
            pass
        elif type == CLUSTER_SWAP_NOW:
            self.notify.debug('swapping')
            base.graphicsEngine.flipFrame()
        elif type == CLUSTER_TIME_DATA:
            self.notify.debug('time data')
            self.handleTimeData(dgi)
        elif type == CLUSTER_NAMED_OBJECT_MOVEMENT:
            self.messageQueue.append(self.msgHandler.parseNamedMovementDatagram(dgi))
        elif type == CLUSTER_NAMED_MOVEMENT_DONE:
            self.handleMessageQueue()
        else:
            self.notify.warning('Received unknown packet type:' % type)
        return type

    def handleCamOffset(self, dgi):
        if False:
            i = 10
            return i + 15
        ' Set offset of camera from cameraJig '
        (x, y, z, h, p, r) = self.msgHandler.parseCamOffsetDatagram(dgi)
        self.camera.setPos(x, y, z)
        self.lens.setViewHpr(h, p, r)

    def handleCamFrustum(self, dgi):
        if False:
            print('Hello World!')
        ' Adjust camera frustum based on parameters sent by client '
        (fl, fs, fo) = self.msgHandler.parseCamFrustumDatagram(dgi)
        self.lens.setFocalLength(fl)
        self.lens.setFilmSize(fs[0], fs[1])
        self.lens.setFilmOffset(fo[0], fo[1])

    def handleNamedMovement(self, data):
        if False:
            i = 10
            return i + 15
        ' Update cameraJig position to reflect latest position '
        (name, x, y, z, h, p, r, sx, sy, sz, red, g, b, a, hidden) = data
        if name in self.objectMappings:
            self.objectMappings[name].setPosHpr(render, x, y, z, h, p, r)
            self.objectMappings[name].setScale(render, sx, sy, sz)
            self.objectMappings[name].setColor(red, g, b, a)
            if hidden:
                self.objectMappings[name].hide()
            else:
                self.objectMappings[name].show()
        else:
            self.notify.debug('recieved unknown named object command: ' + name)

    def handleMessageQueue(self):
        if False:
            i = 10
            return i + 15
        for data in self.messageQueue:
            self.handleNamedMovement(data)
        self.messageQueue = []

    def handleCamMovement(self, dgi):
        if False:
            return 10
        ' Update cameraJig position to reflect latest position '
        (x, y, z, h, p, r) = self.msgHandler.parseCamMovementDatagram(dgi)
        self.cameraJig.setPosHpr(render, x, y, z, h, p, r)
        self.fPosReceived = 1

    def handleSelectedMovement(self, dgi):
        if False:
            while True:
                i = 10
        ' Update cameraJig position to reflect latest position '
        (x, y, z, h, p, r, sx, sy, sz) = self.msgHandler.parseSelectedMovementDatagram(dgi)
        if getattr(builtins, 'last', None):
            builtins.last.setPosHprScale(x, y, z, h, p, r, sx, sy, sz)

    def handleTimeData(self, dgi):
        if False:
            i = 10
            return i + 15
        ' Update cameraJig position to reflect latest position '
        (frameCount, frameTime, dt) = self.msgHandler.parseTimeDataDatagram(dgi)
        clock = ClockObject.getGlobalClock()
        clock.setFrameCount(frameCount)
        clock.setFrameTime(frameTime)
        clock.dt = dt

    def handleCommandString(self, dgi):
        if False:
            print('Hello World!')
        ' Handle arbitrary command string from client '
        command = self.msgHandler.parseCommandStringDatagram(dgi)
        try:
            exec(command, __builtins__)
        except Exception:
            pass