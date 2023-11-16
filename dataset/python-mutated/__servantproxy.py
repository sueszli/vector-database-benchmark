"""
@version: 0.01
@brief: rpc抽离出servantproxy
"""
import threading
import time
from __logger import tarsLogger
from __util import util
from __packet import RequestPacket
from __TimeoutQueue import ReqMessage
import exception
from exception import TarsException

class ServantProxy(object):
    """
    @brief: 1、远程对象的本地代理
            2、同名servant在一个通信器中最多只有一个实例
            3、防止和用户在Tars中定义的函数名冲突，接口以tars_开头
    """
    TARSSERVERSUCCESS = 0
    TARSSERVERDECODEERR = -1
    TARSSERVERENCODEERR = -2
    TARSSERVERNOFUNCERR = -3
    TARSSERVERNOSERVANTERR = -4
    TARSSERVERRESETGRID = -5
    TARSSERVERQUEUETIMEOUT = -6
    TARSASYNCCALLTIMEOUT = -7
    TARSPROXYCONNECTERR = -8
    TARSSERVERUNKNOWNERR = -99
    TARSVERSION = 1
    TUPVERSION = 2
    TUPVERSION2 = 3
    TARSNORMAL = 0
    TARSONEWAY = 1
    TARSMESSAGETYPENULL = 0
    TARSMESSAGETYPEHASH = 1
    TARSMESSAGETYPEGRID = 2
    TARSMESSAGETYPEDYED = 4
    TARSMESSAGETYPESAMPLE = 8
    TARSMESSAGETYPEASYNC = 16
    mapcls_context = util.mapclass(util.string, util.string)

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        tarsLogger.debug('ServantProxy:__init__')
        self.__reactor = None
        self.__object = None
        self.__initialize = False

    def __del__(self):
        if False:
            while True:
                i = 10
        tarsLogger.debug('ServantProxy:__del__')

    def _initialize(self, reactor, obj):
        if False:
            for i in range(10):
                print('nop')
        '\n        @brief: 初始化函数，需要调用才能使用ServantProxy\n        @param reactor: 网络管理的reactor实例\n        @type reactor: FDReactor\n        @return: None\n        @rtype: None\n        '
        tarsLogger.debug('ServantProxy:_initialize')
        assert reactor and obj
        if self.__initialize:
            return
        self.__reactor = reactor
        self.__object = obj
        self.__initialize = True

    def _terminate(self):
        if False:
            print('Hello World!')
        '\n        @brief: 不再使用ServantProxy时调用，会释放相应资源\n        @return: None\n        @rtype: None\n        '
        tarsLogger.debug('ServantProxy:_terminate')
        self.__object = None
        self.__reactor = None
        self.__initialize = False

    def tars_name(self):
        if False:
            print('Hello World!')
        '\n        @brief: 获取ServantProxy的名字\n        @return: ServantProxy的名字\n        @rtype: str\n        '
        return self.__object.name()

    def tars_timeout(self):
        if False:
            print('Hello World!')
        '\n        @brief: 获取超时时间，单位是ms\n        @return: 超时时间\n        @rtype: int\n        '
        return int(self.__timeout() * 1000)

    def tars_ping(self):
        if False:
            i = 10
            return i + 15
        pass

    def tars_invoke(self, cPacketType, sFuncName, sBuffer, context, status):
        if False:
            print('Hello World!')
        '\n        @brief: TARS协议同步方法调用\n        @param cPacketType: 请求包类型\n        @type cPacketType: int\n        @param sFuncName: 调用函数名\n        @type sFuncName: str\n        @param sBuffer: 序列化后的发送参数\n        @type sBuffer: str\n        @param context: 上下文件信息\n        @type context: ServantProxy.mapcls_context\n        @param status: 状态信息\n        @type status:\n        @return: 响应报文\n        @rtype: ResponsePacket\n        '
        tarsLogger.debug('ServantProxy:tars_invoke, func: %s', sFuncName)
        req = RequestPacket()
        req.iVersion = ServantProxy.TARSVERSION
        req.cPacketType = cPacketType
        req.iMessageType = ServantProxy.TARSMESSAGETYPENULL
        req.iRequestId = 0
        req.sServantName = self.tars_name()
        req.sFuncName = sFuncName
        req.sBuffer = sBuffer
        req.iTimeout = self.tars_timeout()
        reqmsg = ReqMessage()
        reqmsg.type = ReqMessage.SYNC_CALL
        reqmsg.servant = self
        reqmsg.lock = threading.Condition()
        reqmsg.request = req
        reqmsg.begtime = time.time()
        reqmsg.isHash = True
        reqmsg.isConHash = True
        reqmsg.hashCode = 123456
        rsp = None
        try:
            rsp = self.__invoke(reqmsg)
        except exception.TarsSyncCallTimeoutException:
            if reqmsg.adapter:
                reqmsg.adapter.finishInvoke(True)
            raise
        except TarsException:
            raise
        except:
            raise TarsException('ServantProxy::tars_invoke excpetion')
        if reqmsg.adapter:
            reqmsg.adapter.finishInvoke(False)
        return rsp

    def tars_invoke_async(self, cPacketType, sFuncName, sBuffer, context, status, callback):
        if False:
            i = 10
            return i + 15
        '\n        @brief: TARS协议同步方法调用\n        @param cPacketType: 请求包类型\n        @type cPacketType: int\n        @param sFuncName: 调用函数名\n        @type sFuncName: str\n        @param sBuffer: 序列化后的发送参数\n        @type sBuffer: str\n        @param context: 上下文件信息\n        @type context: ServantProxy.mapcls_context\n        @param status: 状态信息\n        @type status:\n        @param callback: 异步调用回调对象\n        @type callback: ServantProxyCallback的子类\n        @return: 响应报文\n        @rtype: ResponsePacket\n        '
        tarsLogger.debug('ServantProxy:tars_invoke')
        req = RequestPacket()
        req.iVersion = ServantProxy.TARSVERSION
        req.cPacketType = cPacketType if callback else ServantProxy.TARSONEWAY
        req.iMessageType = ServantProxy.TARSMESSAGETYPENULL
        req.iRequestId = 0
        req.sServantName = self.tars_name()
        req.sFuncName = sFuncName
        req.sBuffer = sBuffer
        req.iTimeout = self.tars_timeout()
        reqmsg = ReqMessage()
        reqmsg.type = ReqMessage.ASYNC_CALL if callback else ReqMessage.ONE_WAY
        reqmsg.callback = callback
        reqmsg.servant = self
        reqmsg.request = req
        reqmsg.begtime = time.time()
        rsp = None
        try:
            rsp = self.__invoke(reqmsg)
        except TarsException:
            raise
        except Exception:
            raise TarsException('ServantProxy::tars_invoke excpetion')
        if reqmsg.adapter:
            reqmsg.adapter.finishInvoke(False)
        return rsp

    def __timeout(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        @brief: 获取超时时间，单位是s\n        @return: 超时时间\n        @rtype: float\n        '
        return self.__object.timeout()

    def __invoke(self, reqmsg):
        if False:
            while True:
                i = 10
        '\n        @brief: 远程过程调用\n        @param reqmsg: 请求数据\n        @type reqmsg: ReqMessage\n        @return: 调用成功或失败\n        @rtype: bool\n        '
        tarsLogger.debug('ServantProxy:invoke, func: %s', reqmsg.request.sFuncName)
        ret = self.__object.invoke(reqmsg)
        if ret == -2:
            errmsg = 'ServantProxy::invoke fail, no valid servant,' + ' servant name : %s, function name : %s' % (reqmsg.request.sServantName, reqmsg.request.sFuncName)
            raise TarsException(errmsg)
        if ret == -1:
            errmsg = 'ServantProxy::invoke connect fail,' + ' servant name : %s, function name : %s, adapter : %s' % (reqmsg.request.sServantName, reqmsg.request.sFuncName, reqmsg.adapter.getEndPointInfo())
            raise TarsException(errmsg)
        elif ret != 0:
            errmsg = 'ServantProxy::invoke unknown fail, ' + 'Servant name : %s, function name : %s' % (reqmsg.request.sServantName, reqmsg.request.sFuncName)
            raise TarsException(errmsg)
        if reqmsg.type == ReqMessage.SYNC_CALL:
            reqmsg.lock.acquire()
            reqmsg.lock.wait(self.__timeout())
            reqmsg.lock.release()
            if not reqmsg.response:
                errmsg = 'ServantProxy::invoke timeout: %d, servant name: %s, adapter: %s, request id: %d' % (self.tars_timeout(), self.tars_name(), reqmsg.adapter.trans().getEndPointInfo(), reqmsg.request.iRequestId)
                raise exception.TarsSyncCallTimeoutException(errmsg)
            elif reqmsg.response.iRet == ServantProxy.TARSSERVERSUCCESS:
                return reqmsg.response
            else:
                errmsg = 'servant name: %s, function name: %s' % (self.tars_name(), reqmsg.request.sFuncName)
                self.tarsRaiseException(reqmsg.response.iRet, errmsg)

    def _finished(self, reqmsg):
        if False:
            for i in range(10):
                print('nop')
        '\n        @brief: 通知远程过程调用线程响应报文到了\n        @param reqmsg: 请求响应报文\n        @type reqmsg: ReqMessage\n        @return: 函数执行成功或失败\n        @rtype: bool\n        '
        tarsLogger.debug('ServantProxy:finished')
        if not reqmsg.lock:
            return False
        reqmsg.lock.acquire()
        reqmsg.lock.notifyAll()
        reqmsg.lock.release()
        return True

    def tarsRaiseException(self, errno, desc):
        if False:
            while True:
                i = 10
        '\n        @brief: 服务器调用失败，根据服务端给的错误码抛出异常\n        @param errno: 错误码\n        @type errno: int\n        @param desc: 错误描述\n        @type desc: str\n        @return: 没有返回值，函数会抛出异常\n        @rtype:\n        '
        if errno == ServantProxy.TARSSERVERSUCCESS:
            return
        elif errno == ServantProxy.TARSSERVERDECODEERR:
            raise exception.TarsServerDecodeException('server decode exception: errno: %d, msg: %s' % (errno, desc))
        elif errno == ServantProxy.TARSSERVERENCODEERR:
            raise exception.TarsServerEncodeException('server encode exception: errno: %d, msg: %s' % (errno, desc))
        elif errno == ServantProxy.TARSSERVERNOFUNCERR:
            raise exception.TarsServerNoFuncException('server function mismatch exception: errno: %d, msg: %s' % (errno, desc))
        elif errno == ServantProxy.TARSSERVERNOSERVANTERR:
            raise exception.TarsServerNoServantException('server servant mismatch exception: errno: %d, msg: %s' % (errno, desc))
        elif errno == ServantProxy.TARSSERVERRESETGRID:
            raise exception.TarsServerResetGridException('server reset grid exception: errno: %d, msg: %s' % (errno, desc))
        elif errno == ServantProxy.TARSSERVERQUEUETIMEOUT:
            raise exception.TarsServerQueueTimeoutException('server queue timeout exception: errno: %d, msg: %s' % (errno, desc))
        elif errno == ServantProxy.TARSPROXYCONNECTERR:
            raise exception.TarsServerQueueTimeoutException('server connection lost: errno: %d, msg: %s' % (errno, desc))
        else:
            raise exception.TarsServerUnknownException('server unknown exception: errno: %d, msg: %s' % (errno, desc))