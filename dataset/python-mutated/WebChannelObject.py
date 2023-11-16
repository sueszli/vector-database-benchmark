"""
Created on 2021/12/15
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: WebChannelObject.py
@description: 交互对象，需要继承QObject并暴露接口
"""
from PyQt5.QtCore import QJsonDocument, QJsonParseError, QObject, pyqtProperty, pyqtSlot
from PyQt5.QtNetwork import QHostAddress
from PyQt5.QtWebChannel import QWebChannel, QWebChannelAbstractTransport
from PyQt5.QtWebSockets import QWebSocketServer

class WebSocketTransport(QWebChannelAbstractTransport):

    def __init__(self, socket, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(WebSocketTransport, self).__init__(*args, **kwargs)
        self.m_socket = socket
        self.m_socket.textMessageReceived.connect(self.textMessageReceived)
        self.m_socket.disconnected.connect(self.deleteLater)

    def sendMessage(self, message):
        if False:
            for i in range(10):
                print('nop')
        print('sendMessage:', message)
        self.m_socket.sendTextMessage(QJsonDocument(message).toJson(QJsonDocument.Compact).data().decode('utf-8', errors='ignore'))

    def textMessageReceived(self, message):
        if False:
            print('Hello World!')
        print('textMessageReceived:', message)
        error = QJsonParseError()
        json = QJsonDocument.fromJson(message.encode('utf-8', errors='ignore'), error)
        if error.error:
            print('Failed to parse message:{}, Error is:{}'.format(message, error.errorString()))
            return
        if not json.isObject():
            print('Received JSON message that is not an object:{}'.format(message))
            return
        self.messageReceived.emit(json.object(), self)

class WebChannelObject(QObject):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(WebChannelObject, self).__init__(*args, **kwargs)
        self._intValue = 0
        self._floatValue = 0.0
        self._boolValue = False
        self._strValue = ''
        self.m_webchannel = QWebChannel(self)
        self.registerObject(self.__class__.__name__, self)
        self.m_clients = {}
        self.m_server = QWebSocketServer(self.__class__.__name__, QWebSocketServer.NonSecureMode, self)

    def registerObject(self, name, obj):
        if False:
            i = 10
            return i + 15
        '注册对象\n        @param name: 名称\n        @type name: str\n        @param obj: 对象\n        @type obj: QObject\n        '
        self.m_webchannel.registerObject(name, obj)

    def registerObjects(self, objects):
        if False:
            print('Hello World!')
        '注册多个对象\n        @param objects: 对象列表\n        @type objects: list\n        '
        for (name, obj) in objects:
            self.registerObject(name, obj)

    def deregisterObject(self, obj):
        if False:
            print('Hello World!')
        '注销对象\n        @param obj: 对象\n        @type obj: QObject\n        '
        self.m_webchannel.deregisterObject(obj)

    def deregisterObjects(self, objects):
        if False:
            while True:
                i = 10
        '注销多个对象\n        @param objects: 对象列表\n        @type objects: list\n        '
        for obj in objects:
            self.deregisterObject(obj)

    def start(self, port=12345):
        if False:
            for i in range(10):
                print('nop')
        '启动服务\n        @param port: 端口\n        @type port: int\n        '
        if not self.m_server.listen(QHostAddress.Any, port):
            raise Exception('Failed to create WebSocket server on port {}'.format(port))
        print('WebSocket server listening on port {}'.format(port))
        self.m_server.newConnection.connect(self._handleNewConnection)

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        '停止服务'
        self.m_server.close()

    def _handleNewConnection(self):
        if False:
            i = 10
            return i + 15
        '新连接'
        socket = self.m_server.nextPendingConnection()
        print('New WebSocket connection from {}'.format(socket.peerAddress().toString()))
        socket.disconnected.connect(self._handleDisconnected)
        transport = WebSocketTransport(socket)
        self.m_clients[socket] = transport
        self.m_webchannel.connectTo(transport)

    def _handleDisconnected(self):
        if False:
            for i in range(10):
                print('nop')
        '连接关闭'
        socket = self.sender()
        print('WebSocket connection from {} closed'.format(socket.peerAddress()))
        if socket in self.m_clients:
            self.m_clients.pop(socket)
        socket.deleteLater()

    @pyqtProperty(int)
    def intValue(self):
        if False:
            i = 10
            return i + 15
        return self._intValue

    @intValue.setter
    def intValue(self, value):
        if False:
            print('Hello World!')
        self._intValue = value

    @pyqtProperty(float)
    def floatValue(self):
        if False:
            i = 10
            return i + 15
        return self._floatValue

    @floatValue.setter
    def floatValue(self, value):
        if False:
            print('Hello World!')
        self._floatValue = value

    @pyqtProperty(bool)
    def boolValue(self):
        if False:
            print('Hello World!')
        return self._boolValue

    @boolValue.setter
    def boolValue(self, value):
        if False:
            print('Hello World!')
        self._boolValue = value

    @pyqtProperty(str)
    def strValue(self):
        if False:
            i = 10
            return i + 15
        return self._strValue

    @strValue.setter
    def strValue(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._strValue = value

    @pyqtSlot(int, int, result=int)
    def testAdd(self, a, b):
        if False:
            return 10
        return a + b