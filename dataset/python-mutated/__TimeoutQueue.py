"""
@version: 0.01
@brief: 请求响应报文和超时队列
"""
import threading
import time
import struct
from .__logger import tarsLogger
from .__tars import TarsInputStream
from .__tars import TarsOutputStream
from .__packet import RequestPacket
from .__packet import ResponsePacket
from .__util import NewLock, LockGuard

class ReqMessage:
    """
    @brief: 请求响应报文，保存一个请求响应所需要的数据
    """
    SYNC_CALL = 1
    ASYNC_CALL = 2
    ONE_WAY = 3

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.type = ReqMessage.SYNC_CALL
        self.servant = None
        self.lock = None
        self.adapter = None
        self.request = None
        self.response = None
        self.callback = None
        self.begtime = None
        self.endtime = None
        self.isHash = False
        self.isConHash = False
        self.hashCode = 0

    def packReq(self):
        if False:
            print('Hello World!')
        '\n        @brief: 序列化请求报文\n        @return: 序列化后的请求报文\n        @rtype: str\n        '
        if not self.request:
            return ''
        oos = TarsOutputStream()
        RequestPacket.writeTo(oos, self.request)
        reqpkt = oos.getBuffer()
        plen = len(reqpkt) + 4
        reqpkt = struct.pack('!i', plen) + reqpkt
        return reqpkt

    @staticmethod
    def unpackRspList(buf):
        if False:
            print('Hello World!')
        '\n        @brief: 解码响应报文\n        @param buf: 多个序列化后的响应报文数据\n        @type buf: str\n        @return: 解码出来的响应报文和解码的buffer长度\n        @rtype: rsplist: 装有ResponsePacket的list\n                unpacklen: int\n        '
        rsplist = []
        if not buf:
            return rsplist
        unpacklen = 0
        buf = buffer(buf)
        while True:
            if len(buf) - unpacklen < 4:
                break
            packsize = buf[unpacklen:unpacklen + 4]
            (packsize,) = struct.unpack_from('!i', packsize)
            if len(buf) < unpacklen + packsize:
                break
            ios = TarsInputStream(buf[unpacklen + 4:unpacklen + packsize])
            rsp = ResponsePacket.readFrom(ios)
            rsplist.append(rsp)
            unpacklen += packsize
        return (rsplist, unpacklen)

class TimeoutQueue:
    """
    @brief: 超时队列，加锁，线程安全
            可以像队列一样FIFO，也可以像字典一样按key取item
    @todo: 限制队列长度
    """

    def __init__(self, timeout=3):
        if False:
            while True:
                i = 10
        self.__uniqId = 0
        self.__lock = NewLock()
        self.__data = {}
        self.__queue = []
        self.__timeout = timeout

    def getTimeout(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        @brief: 获取超时时间，单位为s\n        @return: 超时时间\n        @rtype: float\n        '
        return self.__timeout

    def setTimeout(self, timeout):
        if False:
            i = 10
            return i + 15
        '\n        @brief: 设置超时时间，单位为s\n        @param timeout: 超时时间\n        @type timeout: float\n        @return: None\n        @rtype: None\n        '
        self.__timeout = timeout

    def size(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        @brief: 获取队列长度\n        @return: 队列长度\n        @rtype: int\n        '
        lock = LockGuard(self.__lock)
        ret = len(self.__data)
        return ret

    def generateId(self):
        if False:
            print('Hello World!')
        '\n        @brief: 生成唯一id，0 < id < 2 ** 32\n        @return: id\n        @rtype: int\n        '
        lock = LockGuard(self.__lock)
        ret = self.__uniqId
        ret = (ret + 1) % 2147483647
        while ret <= 0:
            ret = (ret + 1) % 2147483647
        self.__uniqId = ret
        return ret

    def pop(self, uniqId=0, erase=True):
        if False:
            print('Hello World!')
        '\n        @brief: 弹出item\n        @param uniqId: item的id，如果为0，按FIFO弹出\n        @type uniqId: int\n        @param erase: 弹出后是否从字典里删除item\n        @type erase: bool\n        @return: item\n        @rtype: any type\n        '
        ret = None
        lock = LockGuard(self.__lock)
        if not uniqId:
            if len(self.__queue):
                uniqId = self.__queue.pop(0)
        if uniqId:
            if erase:
                ret = self.__data.pop(uniqId, None)
            else:
                ret = self.__data.get(uniqId, None)
        return ret[0] if ret else None

    def push(self, item, uniqId):
        if False:
            while True:
                i = 10
        '\n        @brief: 数据入队列，如果队列已经有了uniqId，插入失败\n        @param item: 插入的数据\n        @type item: any type\n        @return: 插入是否成功\n        @rtype: bool\n        '
        begtime = time.time()
        ret = True
        lock = LockGuard(self.__lock)
        if uniqId in self.__data:
            ret = False
        else:
            self.__data[uniqId] = [item, begtime]
            self.__queue.append(uniqId)
        return ret

    def peek(self, uniqId):
        if False:
            for i in range(10):
                print('nop')
        '\n        @brief: 根据uniqId获取item，不会删除item\n        @param uniqId: item的id\n        @type uniqId: int\n        @return: item\n        @rtype: any type\n        '
        lock = LockGuard(self.__lock)
        ret = self.__data.get(uniqId, None)
        if not ret:
            return None
        return ret[0]

    def timeout(self):
        if False:
            while True:
                i = 10
        '\n        @brief: 检测是否有item超时，如果有就删除\n        @return: None\n        @rtype: None\n        '
        endtime = time.time()
        lock = LockGuard(self.__lock)
        try:
            new_data = {}
            for (uniqId, item) in self.__data.items():
                if endtime - item[1] < self.__timeout:
                    new_data[uniqId] = item
                else:
                    tarsLogger.debug('TimeoutQueue:timeout remove id : %d' % uniqId)
            self.__data = new_data
        finally:
            pass

class QueueTimeout(threading.Thread):
    """
    超时线程，定时触发超时事件
    """

    def __init__(self, timeout=0.1):
        if False:
            i = 10
            return i + 15
        tarsLogger.debug('QueueTimeout:__init__')
        super(QueueTimeout, self).__init__()
        self.timeout = timeout
        self.__terminate = False
        self.__handler = None
        self.__lock = threading.Condition()

    def terminate(self):
        if False:
            i = 10
            return i + 15
        tarsLogger.debug('QueueTimeout:terminate')
        self.__terminate = True
        self.__lock.acquire()
        self.__lock.notifyAll()
        self.__lock.release()

    def setHandler(self, handler):
        if False:
            for i in range(10):
                print('nop')
        self.__handler = handler

    def run(self):
        if False:
            while True:
                i = 10
        while not self.__terminate:
            try:
                self.__lock.acquire()
                self.__lock.wait(self.timeout)
                self.__lock.release()
                if self.__terminate:
                    break
                self.__handler()
            except Exception as msg:
                tarsLogger.error('QueueTimeout:run exception : %s', msg)
        tarsLogger.debug('QueueTimeout:run finished')
if __name__ == '__main__':
    pass