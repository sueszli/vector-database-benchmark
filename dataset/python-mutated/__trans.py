"""
@version: 0.01
@brief: 网络相关模块
"""
import socket
import select
import errno
import threading
from .__logger import tarsLogger
from .__TimeoutQueue import ReqMessage

class EndPointInfo:
    """
    @brief: 保存每个连接端口的信息
    """
    SOCK_TCP = 'TCP'
    SOCK_UDP = 'UDP'

    def __init__(self, ip='', port=0, timeout=-1, weight=0, weightType=0, connType=SOCK_TCP):
        if False:
            print('Hello World!')
        self.__ip = ip
        self.__port = port
        self.__timeout = timeout
        self.__connType = connType
        self.__weightType = weightType
        self.__weight = weight

    def getIp(self):
        if False:
            return 10
        return self.__ip

    def getPort(self):
        if False:
            return 10
        return self.__port

    def getConnType(self):
        if False:
            while True:
                i = 10
        '\n        @return: 传输层连接类型\n        @rtype: EndPointInfo.SOCK_TCP 或 EndPointInfo.SOCK_UDP\n        '
        return self.__connType

    def getWeightType(self):
        if False:
            print('Hello World!')
        return self.__weightType

    def getWeight(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__weight

    def __str__(self):
        if False:
            while True:
                i = 10
        return '%s %s:%s %d:%d' % (self.__connType, self.__ip, self.__port, self.__weightType, self.__weight)

class Transceiver:
    """
    @brief: 网络传输基类，提供网络send/recv接口
    """
    CONNECTED = 0
    CONNECTING = 1
    UNCONNECTED = 2

    def __init__(self, endPointInfo):
        if False:
            i = 10
            return i + 15
        tarsLogger.debug('Transceiver:__init__, %s', endPointInfo)
        self.__epi = endPointInfo
        self.__sock = None
        self.__connStatus = Transceiver.UNCONNECTED
        self.__connFailed = False
        self._sendBuff = ''
        self._recvBuf = ''

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        tarsLogger.debug('Transceiver:__del__')
        self.close()

    def getSock(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        @return: socket对象\n        @rtype: socket.socket\n        '
        return self.__sock

    def getFd(self):
        if False:
            i = 10
            return i + 15
        '\n        @brief: 获取socket的文件描述符\n        @return: 如果self.__sock没有建立返回-1\n        @rtype: int\n        '
        if self.__sock:
            return self.__sock.fileno()
        else:
            return -1

    def getEndPointInfo(self):
        if False:
            return 10
        '\n        @return: 端口信息\n        @rtype: EndPointInfo\n        '
        return self.__epi

    def isValid(self):
        if False:
            while True:
                i = 10
        '\n        @return: 是否创建了socket\n        @rtype: bool\n        '
        return self.__sock is not None

    def hasConnected(self):
        if False:
            print('Hello World!')
        '\n        @return: 是否连接上了\n        @rtype: bool\n        '
        return self.isValid() and self.__connStatus == Transceiver.CONNECTED

    def isConnFailed(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        @return: 是否连接失败\n        @rtype: bool\n        '
        return self.__connFailed

    def isConnecting(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        @return: 是否正在连接\n        @rtype: bool\n        '
        return self.isValid() and self.__connStatus == Transceiver.CONNECTING

    def setConnFailed(self):
        if False:
            return 10
        '\n        @brief: 设置为连接失败\n        @return: None\n        @rtype: None\n        '
        self.__connFailed = True
        self.__connStatus = Transceiver.UNCONNECTED

    def setConnected(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        @brief: 设置为连接完\n        @return: None\n        @rtype: None\n        '
        self.__connFailed = False
        self.__connStatus = Transceiver.CONNECTED

    def close(self):
        if False:
            i = 10
            return i + 15
        '\n        @brief: 关闭连接\n        @return: None\n        @rtype: None\n        @note: 多次调用不会有问题\n        '
        tarsLogger.debug('Transceiver:close')
        if not self.isValid():
            return
        self.__sock.close()
        self.__sock = None
        self.__connStatus = Transceiver.UNCONNECTED
        self.__connFailed = False
        self._sendBuff = ''
        self._recvBuf = ''
        tarsLogger.info('trans close : %s' % self.__epi)

    def writeToSendBuf(self, msg):
        if False:
            i = 10
            return i + 15
        '\n        @brief: 把数据添加到send buffer里\n        @param msg: 发送的数据\n        @type msg: str\n        @return: None\n        @rtype: None\n        @note: 没有加锁，多线程调用会有race conditions\n        '
        self._sendBuff += msg

    def recv(self, bufsize, flag=0):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def send(self, buf, flag=0):
        if False:
            return 10
        raise NotImplementedError()

    def doResponse(self):
        if False:
            return 10
        raise NotImplementedError()

    def doRequest(self):
        if False:
            print('Hello World!')
        '\n        @brief: 将请求数据发送出去\n        @return: 发送的字节数\n        @rtype: int\n        '
        tarsLogger.debug('Transceiver:doRequest')
        if not self.isValid():
            return -1
        nbytes = 0
        buf = buffer(self._sendBuff)
        while True:
            if not buf:
                break
            ret = self.send(buf[nbytes:])
            if ret > 0:
                nbytes += ret
            else:
                break
        self._sendBuff = buf[nbytes:]
        return nbytes

    def reInit(self):
        if False:
            i = 10
            return i + 15
        '\n        @brief: 初始化socket，并连接服务器\n        @return: 成功返回0，失败返回-1\n        @rtype: int\n        '
        tarsLogger.debug('Transceiver:reInit')
        assert self.isValid() is False
        if self.__epi.getConnType() != EndPointInfo.SOCK_TCP:
            return -1
        try:
            self.__sock = socket.socket()
            self.__sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.__sock.setblocking(0)
            self.__sock.connect((self.__epi.getIp(), self.__epi.getPort()))
            self.__connStatus = Transceiver.CONNECTED
        except socket.error as msg:
            if msg.errno == errno.EINPROGRESS:
                self.__connStatus = Transceiver.CONNECTING
            else:
                tarsLogger.info('reInit, %s, faild!, %s', self.__epi, msg)
                self.__sock = None
                return -1
        tarsLogger.info('reInit, connect: %s, fd: %d', self.__epi, self.getFd())
        return 0

class TcpTransceiver(Transceiver):
    """
    @brief: TCP传输实现
    """

    def send(self, buf, flag=0):
        if False:
            print('Hello World!')
        '\n        @brief: 实现tcp的发送\n        @param buf: 发送的数据\n        @type buf: str\n        @param flag: 发送标志\n        @param flag: int\n        @return: 发送字节数\n        @rtype: int\n        '
        tarsLogger.debug('TcpTransceiver:send')
        if not self.isValid():
            return -1
        nbytes = 0
        try:
            nbytes = self.getSock().send(buf, flag)
            tarsLogger.info('tcp send, fd: %d, %s, len: %d', self.getFd(), self.getEndPointInfo(), nbytes)
        except socket.error as msg:
            if msg.errno != errno.EAGAIN:
                tarsLogger.error('tcp send, fd: %d, %s, fail!, %s, close', self.getFd(), self.getEndPointInfo(), msg)
                self.close()
                return 0
        return nbytes

    def recv(self, bufsize, flag=0):
        if False:
            print('Hello World!')
        '\n        @brief: 实现tcp的recv\n        @param bufsize: 接收大小\n        @type bufsize: int\n        @param flag: 接收标志\n        @param flag: int\n        @return: 接收的内容，接收出错返回None\n        @rtype: str\n        '
        tarsLogger.debug('TcpTransceiver:recv')
        assert self.isValid()
        buf = ''
        try:
            buf = self.getSock().recv(bufsize, flag)
            if len(buf) == 0:
                tarsLogger.info('tcp recv, fd: %d, %s, recv 0 bytes, close', self.getFd(), self.getEndPointInfo())
                self.close()
                return None
        except socket.error as msg:
            if msg.errno != errno.EAGAIN:
                tarsLogger.info('tcp recv, fd: %d, %s, faild!, %s, close', self.getFd(), self.getEndPointInfo(), msg)
                self.close()
                return None
        tarsLogger.info('tcp recv, fd: %d, %s, nbytes: %d', self.getFd(), self.getEndPointInfo(), len(buf))
        return buf

    def doResponse(self):
        if False:
            return 10
        '\n        @brief: 处理接收的数据\n        @return: 返回响应报文的列表，如果出错返回None\n        @rtype: list: ResponsePacket\n        '
        tarsLogger.debug('TcpTransceiver:doResponse')
        if not self.isValid():
            return None
        bufs = [self._recvBuf]
        while True:
            buf = self.recv(8292)
            if not buf:
                break
            bufs.append(buf)
        self._recvBuf = ''.join(bufs)
        tarsLogger.info('tcp doResponse, fd: %d, recvbuf: %d', self.getFd(), len(self._recvBuf))
        if not self._recvBuf:
            return None
        rsplist = None
        try:
            (rsplist, bufsize) = ReqMessage.unpackRspList(self._recvBuf)
            self._recvBuf = self._recvBuf[bufsize:]
        except Exception as msg:
            tarsLogger.error('tcp doResponse, fd: %d, %s, tcp recv unpack error: %s', self.getFd(), self.getEndPointInfo(), msg)
            self.close()
        return rsplist

class FDReactor(threading.Thread):
    """
    @brief: 监听FD事件并解发注册的handle
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        tarsLogger.debug('FDReactor:__init__')
        super(FDReactor, self).__init__()
        self.__terminate = False
        self.__ep = None
        self.__shutdown = None
        self.__adapterTab = {}

    def __del__(self):
        if False:
            while True:
                i = 10
        tarsLogger.debug('FDReactor:__del__')
        self.__ep.close()
        self.__shutdown.close()
        self.__ep = None
        self.__shutdown = None

    def initialize(self):
        if False:
            i = 10
            return i + 15
        '\n        @brief: 初始化，使用FDReactor前必须调用\n        @return: None\n        @rtype: None\n        '
        tarsLogger.debug('FDReactor:initialize')
        self.__ep = select.epoll()
        self.__shutdown = socket.socket()
        self.__ep.register(self.__shutdown.fileno(), select.EPOLLET | select.EPOLLIN)
        tarsLogger.debug('FDReactor init, shutdown fd : %d', self.__shutdown.fileno())

    def terminate(self):
        if False:
            print('Hello World!')
        '\n        @brief: 结束FDReactor的线程\n        @return: None\n        @rtype: None\n        '
        tarsLogger.debug('FDReactor:terminate')
        self.__terminate = True
        self.__ep.modify(self.__shutdown.fileno(), select.EPOLLOUT)
        self.__adapterTab = {}

    def handle(self, adapter, events):
        if False:
            return 10
        '\n        @brief: 处理epoll事件\n        @param adapter: 事件对应的adapter\n        @type adapter: AdapterProxy\n        @param events: epoll事件\n        @param events: int\n        @return: None\n        @rtype: None\n        '
        tarsLogger.debug('FDReactor:handle events : %d', events)
        assert adapter
        try:
            if events == 0:
                return
            if events & (select.EPOLLERR | select.EPOLLHUP):
                tarsLogger.debug('FDReactor::handle EPOLLERR or EPOLLHUP: %s', adapter.trans().getEndPointInfo())
                adapter.trans().close()
                return
            if adapter.shouldCloseTrans():
                tarsLogger.debug('FDReactor::handle should close trans: %s', adapter.trans().getEndPointInfo())
                adapter.setCloseTrans(False)
                adapter.trans().close()
                return
            if adapter.trans().isConnecting():
                if not adapter.finishConnect():
                    return
            if events & select.EPOLLIN:
                self.handleInput(adapter)
            if events & select.EPOLLOUT:
                self.handleOutput(adapter)
        except Exception as msg:
            tarsLogger.error('FDReactor handle exception: %s', msg)

    def handleExcept(self):
        if False:
            i = 10
            return i + 15
        pass

    def handleInput(self, adapter):
        if False:
            for i in range(10):
                print('nop')
        '\n        @brief: 处理接收事件\n        @param adapter: 事件对应的adapter\n        @type adapter: AdapterProxy\n        @return: None\n        @rtype: None\n        '
        tarsLogger.debug('FDReactor:handleInput')
        if not adapter.trans().isValid():
            return
        rsplist = adapter.trans().doResponse()
        if not rsplist:
            return
        for rsp in rsplist:
            adapter.finished(rsp)

    def handleOutput(self, adapter):
        if False:
            while True:
                i = 10
        '\n        @brief: 处理发送事件\n        @param adapter: 事件对应的adapter\n        @type adapter: AdapterProxy\n        @return: None\n        @rtype: None\n        '
        tarsLogger.debug('FDReactor:handleOutput')
        if not adapter.trans().isValid():
            return
        while adapter.trans().doRequest() >= 0 and adapter.sendRequest():
            pass

    def notify(self, adapter):
        if False:
            for i in range(10):
                print('nop')
        '\n        @brief: 更新adapter对应的fd的epoll状态\n        @return: None\n        @rtype: None\n        @note: FDReactor使用的epoll是EPOLLET模式，同一事件只通知一次\n               希望某一事件再次通知需调用此函数\n        '
        tarsLogger.debug('FDReactor:notify')
        fd = adapter.trans().getFd()
        if fd != -1:
            self.__ep.modify(fd, select.EPOLLET | select.EPOLLOUT | select.EPOLLIN)

    def registerAdapter(self, adapter, events):
        if False:
            i = 10
            return i + 15
        '\n        @brief: 注册adapter\n        @param adapter: 收发事件处理类\n        @type adapter: AdapterProxy\n        @param events: 注册事件\n        @type events: int\n        @return: None\n        @rtype: None\n        '
        tarsLogger.debug('FDReactor:registerAdapter events : %d', events)
        events |= select.EPOLLET
        try:
            self.__ep.unregister(adapter.trans().getFd())
        except:
            pass
        self.__ep.register(adapter.trans().getFd(), events)
        self.__adapterTab[adapter.trans().getFd()] = adapter

    def unregisterAdapter(self, adapter):
        if False:
            while True:
                i = 10
        '\n        @brief: 注销adapter\n        @param adapter: 收发事件处理类\n        @type adapter: AdapterProxy\n        @return: None\n        @rtype: None\n        '
        tarsLogger.debug('FDReactor:registerAdapter')
        self.__ep.unregister(adapter.trans().getFd())
        self.__adapterTab.pop(adapter.trans().getFd(), None)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        @brief: 线程启动函数，循环监听网络事件\n        '
        tarsLogger.debug('FDReactor:run')
        while not self.__terminate:
            try:
                eplist = self.__ep.poll(1)
                if eplist:
                    tarsLogger.debug('FDReactor run get eplist : %s, terminate : %s', str(eplist), self.__terminate)
                if self.__terminate:
                    tarsLogger.debug('FDReactor terminate')
                    break
                for (fd, events) in eplist:
                    adapter = self.__adapterTab.get(fd, None)
                    if not adapter:
                        continue
                    self.handle(adapter, events)
            except Exception as msg:
                tarsLogger.error('FDReactor run exception: %s', msg)
        tarsLogger.debug('FDReactor:run finished')
if __name__ == '__main__':
    print('hello world')
    epi = EndPointInfo('127.0.0.1', 1313)
    print(epi)
    trans = TcpTransceiver(epi)
    print(trans.getSock())
    print(trans.getFd())
    print(trans.reInit())
    print(trans.isConnecting())
    print(trans.hasConnected())
    buf = 'hello world'
    print(trans.send(buf))
    buf = trans.recv(1024)
    print(buf)
    trans.close()