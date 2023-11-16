"""
@version: 0.01
@brief: 将rpc部分中的adapterproxymanager抽离出来，实现不同的负载均衡
"""
from enum import Enum
import random
import socket
import select
import os
import time
from .__util import LockGuard, NewLock, ConsistentHashNew
from .__trans import EndPointInfo
from .__logger import tarsLogger
from . import exception
from .__trans import TcpTransceiver
from .__TimeoutQueue import ReqMessage
from .exception import TarsException
from .QueryF import QueryFProxy
from .QueryF import QueryFPrxCallback

class AdapterProxy:
    """
    @brief: 每一个Adapter管理一个服务端端口的连接，数据收发
    """

    def __init__(self):
        if False:
            return 10
        tarsLogger.debug('AdapterProxy:__init__')
        self.__closeTrans = False
        self.__trans = None
        self.__object = None
        self.__reactor = None
        self.__lock = None
        self.__asyncProc = None
        self.__activeStateInReg = True

    @property
    def activatestateinreg(self):
        if False:
            while True:
                i = 10
        return self.__activeStateInReg

    @activatestateinreg.setter
    def activatestateinreg(self, value):
        if False:
            i = 10
            return i + 15
        self.__activeStateInReg = value

    def __del__(self):
        if False:
            i = 10
            return i + 15
        tarsLogger.debug('AdapterProxy:__del__')

    def initialize(self, endPointInfo, objectProxy, reactor, asyncProc):
        if False:
            return 10
        '\n        @brief: 初始化\n        @param endPointInfo: 连接对端信息\n        @type endPointInfo: EndPointInfo\n        @type objectProxy: ObjectProxy\n        @type reactor: FDReactor\n        @type asyncProc: AsyncProcThread\n        '
        tarsLogger.debug('AdapterProxy:initialize')
        self.__closeTrans = False
        self.__trans = TcpTransceiver(endPointInfo)
        self.__object = objectProxy
        self.__reactor = reactor
        self.__lock = NewLock()
        self.__asyncProc = asyncProc

    def terminate(self):
        if False:
            return 10
        '\n        @brief: 关闭\n        '
        tarsLogger.debug('AdapterProxy:terminate')
        self.setCloseTrans(True)

    def trans(self):
        if False:
            i = 10
            return i + 15
        '\n        @brief: 获取传输类\n        @return: 负责网络传输的trans\n        @rtype: Transceiver\n        '
        return self.__trans

    def invoke(self, reqmsg):
        if False:
            return 10
        '\n        @brief: 远程过程调用处理方法\n        @param reqmsg: 请求响应报文\n        @type reqmsg: ReqMessage\n        @return: 错误码：0表示成功，-1表示连接失败\n        @rtype: int\n        '
        tarsLogger.debug('AdapterProxy:invoke')
        assert self.__trans
        if not self.__trans.hasConnected() and (not self.__trans.isConnecting):
            return -1
        reqmsg.request.iRequestId = self.__object.getTimeoutQueue().generateId()
        self.__object.getTimeoutQueue().push(reqmsg, reqmsg.request.iRequestId)
        self.__reactor.notify(self)
        return 0

    def finished(self, rsp):
        if False:
            while True:
                i = 10
        '\n        @brief: 远程过程调用返回处理\n        @param rsp: 响应报文\n        @type rsp: ResponsePacket\n        @return: 函数是否执行成功\n        @rtype: bool\n        '
        tarsLogger.debug('AdapterProxy:finished')
        reqmsg = self.__object.getTimeoutQueue().pop(rsp.iRequestId)
        if not reqmsg:
            tarsLogger.error('finished, can not get ReqMessage, may be timeout, id: %d', rsp.iRequestId)
            return False
        reqmsg.response = rsp
        if reqmsg.type == ReqMessage.SYNC_CALL:
            return reqmsg.servant._finished(reqmsg)
        elif reqmsg.callback:
            self.__asyncProc.put(reqmsg)
            return True
        tarsLogger.error('finished, adapter proxy finish fail, id: %d, ret: %d', rsp.iRequestId, rsp.iRet)
        return False

    def checkActive(self, forceConnect=False):
        if False:
            print('Hello World!')
        '\n        @brief: 检测连接是否失效\n        @param forceConnect: 是否强制发起连接，为true时不对状态进行判断就发起连接\n        @type forceConnect: bool\n        @return: 连接是否有效\n        @rtype: bool\n        '
        tarsLogger.debug('AdapterProxy:checkActive')
        lock = LockGuard(self.__lock)
        tarsLogger.info('checkActive, %s, forceConnect: %s', self.__trans.getEndPointInfo(), forceConnect)
        if not self.__trans.isConnecting() and (not self.__trans.hasConnected()):
            self.doReconnect()
        return self.__trans.isConnecting() or self.__trans.hasConnected()

    def doReconnect(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        @brief: 重新发起连接\n        @return: None\n        @rtype: None\n        '
        tarsLogger.debug('AdapterProxy:doReconnect')
        assert self.__trans
        self.__trans.reInit()
        tarsLogger.info('doReconnect, connect: %s, fd:%d', self.__trans.getEndPointInfo(), self.__trans.getFd())
        self.__reactor.registerAdapter(self, select.EPOLLIN | select.EPOLLOUT)

    def sendRequest(self):
        if False:
            print('Hello World!')
        '\n        @brief: 把队列中的请求放到Transceiver的发送缓存里\n        @return: 放入缓存的数据长度\n        @rtype: int\n        '
        tarsLogger.debug('AdapterProxy:sendRequest')
        if not self.__trans.hasConnected():
            return False
        reqmsg = self.__object.popRequest()
        blen = 0
        while reqmsg:
            reqmsg.adapter = self
            buf = reqmsg.packReq()
            self.__trans.writeToSendBuf(buf)
            tarsLogger.info('sendRequest, id: %d, len: %d', reqmsg.request.iRequestId, len(buf))
            blen += len(buf)
            if self.__trans.getEndPointInfo().getConnType() == EndPointInfo.SOCK_UDP or blen > 8192:
                break
            reqmsg = self.__object.popRequest()
        return blen

    def finishConnect(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        @brief: 使用的非阻塞socket连接不能立刻判断是否连接成功，\n                在epoll响应后调用此函数处理connect结束后的操作\n        @return: 是否连接成功\n        @rtype: bool\n        '
        tarsLogger.debug('AdapterProxy:finishConnect')
        success = True
        errmsg = ''
        try:
            ret = self.__trans.getSock().getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            if ret:
                success = False
                errmsg = os.strerror(ret)
        except Exception as msg:
            errmsg = msg
            success = False
        if not success:
            self.__reactor.unregisterAdapter(self, socket.EPOLLIN | socket.EPOLLOUT)
            self.__trans.close()
            self.__trans.setConnFailed()
            tarsLogger.error('AdapterProxy finishConnect, exception: %s, error: %s', self.__trans.getEndPointInfo(), errmsg)
            return False
        self.__trans.setConnected()
        self.__reactor.notify(self)
        tarsLogger.info('AdapterProxy finishConnect, connect %s success', self.__trans.getEndPointInfo())
        return True

    def finishInvoke(self, isTimeout):
        if False:
            while True:
                i = 10
        pass

    def popRequest(self):
        if False:
            return 10
        pass

    def shouldCloseTrans(self):
        if False:
            i = 10
            return i + 15
        '\n        @brief: 是否设置关闭连接\n        @return: 关闭连接的flag的值\n        @rtype: bool\n        '
        return self.__closeTrans

    def setCloseTrans(self, closeTrans):
        if False:
            for i in range(10):
                print('nop')
        '\n        @brief: 设置关闭连接flag的值\n        @param closeTrans: 是否关闭连接\n        @type closeTrans: bool\n        @return: None\n        @rtype: None\n        '
        self.__closeTrans = closeTrans

class QueryRegisterCallback(QueryFPrxCallback):

    def __init__(self, adpManager):
        if False:
            i = 10
            return i + 15
        self.__adpManager = adpManager
        super(QueryRegisterCallback, self).__init__()

    def callback_findObjectById4All(self, ret, activeEp, inactiveEp):
        if False:
            while True:
                i = 10
        eplist = [EndPointInfo(x.host, x.port, x.timeout, x.weight, x.weightType) for x in activeEp if ret == 0 and x.istcp]
        ieplist = [EndPointInfo(x.host, x.port, x.timeout, x.weight, x.weightType) for x in inactiveEp if ret == 0 and x.istcp]
        self.__adpManager.setEndpoints(eplist, ieplist)

    def callback_findObjectById4All_exception(self, ret):
        if False:
            print('Hello World!')
        tarsLogger.error('callback_findObjectById4All_exception ret: %d', ret)

class EndpointWeightType(Enum):
    E_LOOP = 0
    E_STATIC_WEIGHT = 1

class AdapterProxyManager:
    """
    @brief: 管理Adapter
    """

    def __init__(self):
        if False:
            print('Hello World!')
        tarsLogger.debug('AdapterProxyManager:__init__')
        self.__comm = None
        self.__object = None
        self.__adps = {}
        self.__iadps = {}
        self.__newLock = None
        self.__isDirectProxy = True
        self.__lastFreshTime = 0
        self.__queryRegisterCallback = QueryRegisterCallback(self)
        self.__regAdapterProxyDict = {}
        self.__lastConHashPrxList = []
        self.__consistentHashWeight = None
        self.__weightType = EndpointWeightType.E_LOOP
        self.__update = True
        self.__lastWeightedProxyData = {}

    def initialize(self, comm, objectProxy, eplist):
        if False:
            while True:
                i = 10
        '\n        @brief: 初始化\n        '
        tarsLogger.debug('AdapterProxyManager:initialize')
        self.__comm = comm
        self.__object = objectProxy
        self.__newLock = NewLock()
        self.__isDirectProxy = len(eplist) > 0
        if self.__isDirectProxy:
            self.setEndpoints(eplist, {})
        else:
            self.refreshEndpoints()

    def terminate(self):
        if False:
            i = 10
            return i + 15
        '\n        @brief: 释放资源\n        '
        tarsLogger.debug('AdapterProxyManager:terminate')
        lock = LockGuard(self.__newLock)
        for (ep, epinfo) in self.__adps.items():
            epinfo[1].terminate()
        self.__adps = {}
        self.__lock.release()

    def refreshEndpoints(self):
        if False:
            return 10
        '\n        @brief: 刷新服务器列表\n        @return: 新的服务列表\n        @rtype: EndPointInfo列表\n        '
        tarsLogger.debug('AdapterProxyManager:refreshEndpoints')
        if self.__isDirectProxy:
            return
        interval = self.__comm.getProperty('refresh-endpoint-interval', float) / 1000
        locator = self.__comm.getProperty('locator')
        if '@' not in locator:
            raise exception.TarsRegistryException('locator is not valid: ' + locator)
        now = time.time()
        last = self.__lastFreshTime
        epSize = len(self.__adps)
        if last + interval < now or (epSize <= 0 and last + 2 < now):
            queryFPrx = self.__comm.stringToProxy(QueryFProxy, locator)
            if epSize == 0 or last == 0:
                (ret, activeEps, inactiveEps) = queryFPrx.findObjectById4All(self.__object.name())
                eplist = [EndPointInfo(x.host, x.port, x.timeout, x.weight, x.weightType) for x in activeEps if ret == 0 and x.istcp]
                ieplist = [EndPointInfo(x.host, x.port, x.timeout, x.weight, x.weightType) for x in inactiveEps if ret == 0 and x.istcp]
                self.setEndpoints(eplist, ieplist)
            else:
                queryFPrx.async_findObjectById4All(self.__queryRegisterCallback, self.__object.name())
            self.__lastFreshTime = now

    def getEndpoints(self):
        if False:
            i = 10
            return i + 15
        '\n        @brief: 获取可用服务列表 如果启用分组,只返回同分组的服务端ip\n        @return: 获取节点列表\n        @rtype: EndPointInfo列表\n        '
        tarsLogger.debug('AdapterProxyManager:getEndpoints')
        lock = LockGuard(self.__newLock)
        ret = [x[1][0] for x in list(self.__adps.items())]
        return ret

    def setEndpoints(self, eplist, ieplist):
        if False:
            print('Hello World!')
        '\n        @brief: 设置服务端信息\n        @para eplist: 活跃的被调节点列表\n        @para ieplist: 不活跃的被调节点列表\n        '
        tarsLogger.debug('AdapterProxyManager:setEndpoints')
        adps = {}
        iadps = {}
        comm = self.__comm
        isNeedNotify = False
        lock = LockGuard(self.__newLock)
        isStartStatic = True
        for ep in eplist:
            if ep.getWeightType() == 0:
                isStartStatic = False
            epstr = str(ep)
            if epstr in self.__adps:
                adps[epstr] = self.__adps[epstr]
                continue
            isNeedNotify = True
            self.__update = True
            adapter = AdapterProxy()
            adapter.initialize(ep, self.__object, comm.getReactor(), comm.getAsyncProc())
            adapter.activatestateinreg = True
            adps[epstr] = [ep, adapter, 0]
        (self.__adps, adps) = (adps, self.__adps)
        for iep in ieplist:
            iepstr = str(iep)
            if iepstr in self.__iadps:
                iadps[iepstr] = self.__iadps[iepstr]
                continue
            isNeedNotify = True
            adapter = AdapterProxy()
            adapter.initialize(iep, self.__object, comm.getReactor(), comm.getAsyncProc())
            adapter.activatestateinreg = False
            iadps[iepstr] = [iep, adapter, 0]
        (self.__iadps, iadps) = (iadps, self.__iadps)
        if isStartStatic:
            self.__weightType = EndpointWeightType.E_STATIC_WEIGHT
        else:
            self.__weightType = EndpointWeightType.E_LOOP
        if isNeedNotify:
            self.__notifyEndpoints(self.__adps, self.__iadps)
        for ep in adps:
            if ep not in self.__adps:
                adps[ep][1].terminate()

    def __notifyEndpoints(self, actives, inactives):
        if False:
            for i in range(10):
                print('nop')
        lock = LockGuard(self.__newLock)
        self.__regAdapterProxyDict.clear()
        self.__regAdapterProxyDict.update(actives)
        self.__regAdapterProxyDict.update(inactives)

    def __getNextValidProxy(self):
        if False:
            i = 10
            return i + 15
        '\n        @brief: 刷新本地缓存列表，如果服务下线了，要求删除本地缓存\n        @return:\n        @rtype: EndPointInfo列表\n        @todo: 优化负载均衡算法\n        '
        tarsLogger.debug('AdapterProxyManager:getNextValidProxy')
        lock = LockGuard(self.__newLock)
        if len(self.__adps) == 0:
            raise TarsException('the activate adapter proxy is empty')
        sortedActivateAdp = sorted(list(self.__adps.items()), key=lambda item: item[1][2])
        sortedActivateAdpSize = len(sortedActivateAdp)
        while sortedActivateAdpSize != 0:
            if sortedActivateAdp[0][1][1].checkActive():
                self.__adps[sortedActivateAdp[0][0]][2] += 1
                return self.__adps[sortedActivateAdp[0][0]][1]
            sortedActivateAdp.pop(0)
            sortedActivateAdpSize -= 1
        adpPrx = list(self.__adps.items())[random.randint(0, len(self.__adps))][1][1]
        adpPrx.checkActive()
        return None

    def __getHashProxy(self, reqmsg):
        if False:
            print('Hello World!')
        if self.__weightType == EndpointWeightType.E_LOOP:
            if reqmsg.isConHash:
                return self.__getConHashProxyForNormal(reqmsg.hashCode)
            else:
                return self.__getHashProxyForNormal(reqmsg.hashCode)
        elif reqmsg.isConHash:
            return self.__getConHashProxyForWeight(reqmsg.hashCode)
        else:
            return self.__getHashProxyForWeight(reqmsg.hashCode)

    def __getHashProxyForNormal(self, hashCode):
        if False:
            i = 10
            return i + 15
        tarsLogger.debug('AdapterProxyManager:getHashProxyForNormal')
        lock = LockGuard(self.__newLock)
        regAdapterProxyList = sorted(list(self.__regAdapterProxyDict.items()), key=lambda item: item[0])
        allPrxSize = len(regAdapterProxyList)
        if allPrxSize == 0:
            raise TarsException('the adapter proxy is empty')
        hashNum = hashCode % allPrxSize
        if regAdapterProxyList[hashNum][1][1].activatestateinreg and regAdapterProxyList[hashNum][1][1].checkActive():
            epstr = regAdapterProxyList[hashNum][0]
            self.__regAdapterProxyDict[epstr][2] += 1
            if epstr in self.__adps:
                self.__adps[epstr][2] += 1
            elif epstr in self.__iadps:
                self.__iadps[epstr][2] += 1
            return self.__regAdapterProxyDict[epstr][1]
        else:
            if len(self.__adps) == 0:
                raise TarsException('the activate adapter proxy is empty')
            activeProxyList = list(self.__adps.items())
            actPrxSize = len(activeProxyList)
            while actPrxSize != 0:
                hashNum = hashCode % actPrxSize
                if activeProxyList[hashNum][1][1].checkActive():
                    self.__adps[activeProxyList[hashNum][0]][2] += 1
                    return self.__adps[activeProxyList[hashNum][0]][1]
                activeProxyList.pop(hashNum)
                actPrxSize -= 1
            adpPrx = list(self.__adps.items())[random.randint(0, len(self.__adps))][1][1]
            adpPrx.checkActive()
            return None

    def __getConHashProxyForNormal(self, hashCode):
        if False:
            i = 10
            return i + 15
        tarsLogger.debug('AdapterProxyManager:getConHashProxyForNormal')
        lock = LockGuard(self.__newLock)
        if len(self.__regAdapterProxyDict) == 0:
            raise TarsException('the adapter proxy is empty')
        if self.__consistentHashWeight is None or self.__checkConHashChange(self.__lastConHashPrxList):
            self.__updateConHashProxyWeighted()
        if len(self.__consistentHashWeight.nodes) > 0:
            conHashIndex = self.__consistentHashWeight.getNode(hashCode)
            if conHashIndex in self.__regAdapterProxyDict and self.__regAdapterProxyDict[conHashIndex][1].activatestateinreg and self.__regAdapterProxyDict[conHashIndex][1].checkActive():
                self.__regAdapterProxyDict[conHashIndex][2] += 1
                if conHashIndex in self.__adps:
                    self.__adps[conHashIndex][2] += 1
                elif conHashIndex in self.__iadps:
                    self.__iadps[conHashIndex][2] += 1
                return self.__regAdapterProxyDict[conHashIndex][1]
            else:
                if len(self.__adps) == 0:
                    raise TarsException('the activate adapter proxy is empty')
                activeProxyList = list(self.__adps.items())
                actPrxSize = len(activeProxyList)
                while actPrxSize != 0:
                    hashNum = hashCode % actPrxSize
                    if activeProxyList[hashNum][1][1].checkActive():
                        self.__adps[activeProxyList[hashNum][0]][2] += 1
                        return self.__adps[activeProxyList[hashNum][0]][1]
                    activeProxyList.pop(hashNum)
                    actPrxSize -= 1
                adpPrx = list(self.__adps.items())[random.randint(0, len(self.__adps))][1][1]
                adpPrx.checkActive()
                return None
            pass
        else:
            return self.__getHashProxyForNormal(hashCode)

    def __getHashProxyForWeight(self, hashCode):
        if False:
            for i in range(10):
                print('nop')
        return None
        pass

    def __getConHashProxyForWeight(self, hashCode):
        if False:
            i = 10
            return i + 15
        return None
        pass

    def __checkConHashChange(self, lastConHashPrxList):
        if False:
            return 10
        tarsLogger.debug('AdapterProxyManager:checkConHashChange')
        lock = LockGuard(self.__newLock)
        if len(lastConHashPrxList) != len(self.__regAdapterProxyDict):
            return True
        regAdapterProxyList = sorted(list(self.__regAdapterProxyDict.items()), key=lambda item: item[0])
        regAdapterProxyListSize = len(regAdapterProxyList)
        for index in range(regAdapterProxyListSize):
            if cmp(lastConHashPrxList[index][0], regAdapterProxyList[index][0]) != 0:
                return True
        return False

    def __updateConHashProxyWeighted(self):
        if False:
            return 10
        tarsLogger.debug('AdapterProxyManager:updateConHashProxyWeighted')
        lock = LockGuard(self.__newLock)
        if len(self.__regAdapterProxyDict) == 0:
            raise TarsException('the adapter proxy is empty')
        self.__lastConHashPrxList = sorted(list(self.__regAdapterProxyDict.items()), key=lambda item: item[0])
        nodes = []
        for var in self.__lastConHashPrxList:
            nodes.append(var[0])
        if self.__consistentHashWeight is None:
            self.__consistentHashWeight = ConsistentHashNew(nodes)
        else:
            theOldActiveNodes = [var for var in nodes if var in self.__consistentHashWeight.nodes]
            theOldInactiveNodes = [var for var in self.__consistentHashWeight.nodes if var not in theOldActiveNodes]
            for var in theOldInactiveNodes:
                self.__consistentHashWeight.removeNode(var)
            theNewActiveNodes = [var for var in nodes if var not in theOldActiveNodes]
            for var in theNewActiveNodes:
                self.__consistentHashWeight.addNode(var)
            self.__consistentHashWeight.nodes = nodes
        pass

    def __getWeightedProxy(self):
        if False:
            return 10
        tarsLogger.debug('AdapterProxyManager:getWeightedProxy')
        lock = LockGuard(self.__newLock)
        if len(self.__adps) == 0:
            raise TarsException('the activate adapter proxy is empty')
        if self.__update is True:
            self.__lastWeightedProxyData.clear()
            weightedProxyData = {}
            minWeight = list(self.__adps.items())[0][1][0].getWeight()
            for item in list(self.__adps.items()):
                weight = item[1][0].getWeight()
                weightedProxyData[item[0]] = weight
                if minWeight > weight:
                    minWeight = weight
            if minWeight <= 0:
                addWeight = -minWeight + 1
                for item in list(weightedProxyData.items()):
                    item[1] += addWeight
            self.__update = False
            self.__lastWeightedProxyData = weightedProxyData
        weightedProxyData = self.__lastWeightedProxyData
        while len(weightedProxyData) > 0:
            total = sum(weightedProxyData.values())
            rand = random.randint(1, total)
            temp = 0
            for item in list(weightedProxyData.items()):
                temp += item[1]
                if rand <= temp:
                    if self.__adps[item[0]][1].checkActive():
                        self.__adps[item[0]][2] += 1
                        return self.__adps[item[0]][1]
                    else:
                        weightedProxyData.pop(item[0])
                        break
        adpPrx = list(self.__adps.items())[random.randint(0, len(self.__adps))][1][1]
        adpPrx.checkActive()
        return None

    def selectAdapterProxy(self, reqmsg):
        if False:
            while True:
                i = 10
        '\n        @brief: 刷新本地缓存列表，如果服务下线了，要求删除本地缓存，通过一定算法返回AdapterProxy\n        @param: reqmsg:请求响应报文\n        @type reqmsg: ReqMessage\n        @return:\n        @rtype: EndPointInfo列表\n        @todo: 优化负载均衡算法\n        '
        tarsLogger.debug('AdapterProxyManager:selectAdapterProxy')
        self.refreshEndpoints()
        if reqmsg.isHash:
            return self.__getHashProxy(reqmsg)
        elif self.__weightType == EndpointWeightType.E_STATIC_WEIGHT:
            return self.__getWeightedProxy()
        else:
            return self.__getNextValidProxy()