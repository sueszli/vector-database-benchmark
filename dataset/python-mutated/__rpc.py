"""
@version: 0.01
@brief: rpc调用逻辑实现
"""
import time
import argparse
from .__logger import tarsLogger
from .__logger import initLog
from .__trans import EndPointInfo
from .__TimeoutQueue import TimeoutQueue
from .__TimeoutQueue import QueueTimeout
from .__trans import FDReactor
from .__adapterproxy import AdapterProxyManager
from .__servantproxy import ServantProxy
from .exception import TarsException
from .__async import AsyncProcThread

class Communicator:
    """
    @brief: 通讯器，创建和维护ServantProxy、ObjectProxy、FDReactor线程和超时线程
    """
    default_config = {'tars': {'application': {'client': {'async-invoke-timeout': 20000, 'asyncthread': 0, 'locator': '', 'loglevel': 'error', 'logpath': 'tars.log', 'logsize': 15728640, 'lognum': 0, 'refresh-endpoint-interval': 60000, 'sync-invoke-timeout': 5000}}}}

    def __init__(self, config={}):
        if False:
            print('Hello World!')
        tarsLogger.debug('Communicator:__init__')
        self.__terminate = False
        self.__initialize = False
        self.__objects = {}
        self.__servants = {}
        self.__reactor = None
        self.__qTimeout = None
        self.__asyncProc = None
        self.__config = Communicator.default_config.copy()
        self.__config.update(config)
        self.initialize()

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        tarsLogger.debug('Communicator:__del__')

    def initialize(self):
        if False:
            print('Hello World!')
        '\n        @brief: 使用通讯器前必须先调用此函数\n        '
        tarsLogger.debug('Communicator:initialize')
        if self.__initialize:
            return
        logpath = self.getProperty('logpath')
        logsize = self.getProperty('logsize', int)
        lognum = self.getProperty('lognum', int)
        loglevel = self.getProperty('loglevel')
        initLog(logpath, logsize, lognum, loglevel)
        self.__reactor = FDReactor()
        self.__reactor.initialize()
        self.__reactor.start()
        self.__qTimeout = QueueTimeout()
        self.__qTimeout.setHandler(self.handleTimeout)
        self.__qTimeout.start()
        async_num = self.getProperty('asyncthread', int)
        self.__asyncProc = AsyncProcThread()
        self.__asyncProc.initialize(async_num)
        self.__asyncProc.start()
        self.__initialize = True

    def terminate(self):
        if False:
            return 10
        '\n        @brief: 不再使用通讯器需调用此函数释放资源\n        '
        tarsLogger.debug('Communicator:terminate')
        if not self.__initialize:
            return
        self.__reactor.terminate()
        self.__qTimeout.terminate()
        self.__asyncProc.terminate()
        for objName in self.__servants:
            self.__servants[objName]._terminate()
        for objName in self.__objects:
            self.__objects[objName].terminate()
        self.__objects = {}
        self.__servants = {}
        self.__reactor = None
        self.__initialize = False

    def parseConnAddr(self, connAddr):
        if False:
            for i in range(10):
                print('nop')
        '\n        @brief: 解析connAddr字符串\n        @param connAddr: 连接地址\n        @type connAddr: str\n        @return: 解析结果\n        @rtype: dict, key是str，val里name是str，\n                timeout是float，endpoint是EndPointInfo的list\n        '
        tarsLogger.debug('Communicator:parseConnAddr')
        connAddr = connAddr.strip()
        connInfo = {'name': '', 'timeout': -1, 'endpoint': []}
        if '@' not in connAddr:
            connInfo['name'] = connAddr
            return connInfo
        try:
            tks = connAddr.split('@')
            connInfo['name'] = tks[0]
            tks = tks[1].lower().split(':')
            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument('-h')
            parser.add_argument('-p')
            parser.add_argument('-t')
            for tk in tks:
                argv = tk.split()
                if argv[0] != 'tcp':
                    raise TarsException('unsupport transmission protocal : %s' % connInfo['name'])
                mes = parser.parse_args(argv[1:])
                try:
                    ip = mes.h if mes.h is not None else ''
                    port = int(mes.p) if mes.p is not None else '-1'
                    timeout = int(mes.t) if mes.t is not None else '-1'
                    connInfo['endpoint'].append(EndPointInfo(ip, port, timeout))
                except Exception:
                    raise TarsException('Unrecognized option : %s' % mes)
        except TarsException:
            raise
        except Exception as exp:
            raise TarsException(exp)
        return connInfo

    def getReactor(self):
        if False:
            print('Hello World!')
        '\n        @brief: 获取reactor\n        '
        return self.__reactor

    def getAsyncProc(self):
        if False:
            while True:
                i = 10
        '\n        @brief: 获取asyncProc\n        '
        return self.__asyncProc

    def getProperty(self, name, dt_type=str):
        if False:
            for i in range(10):
                print('nop')
        '\n        @brief: 获取配置\n        @param name: 配置名称\n        @type name: str\n        @param dt_type: 数据类型\n        @type name: type\n        @return: 配置内容\n        @rtype: str\n        '
        try:
            ret = self.__config['tars']['application']['client'][name]
            ret = dt_type(ret)
        except:
            ret = Communicator.default_config['tars']['application']['client'][name]
        return ret

    def setProperty(self, name, value):
        if False:
            i = 10
            return i + 15
        '\n        @brief: 修改配置\n        @param name: 配置名称\n        @type propertys: str\n        @param value: 配置内容\n        @type propertys: str\n        @return: 设置是否成功\n        @rtype: bool\n        '
        try:
            self.__config['tars']['application']['client'][name] = value
            return True
        except:
            return False

    def setPropertys(self, propertys):
        if False:
            i = 10
            return i + 15
        '\n        @brief: 修改配置\n        @param propertys: 配置集合\n        @type propertys: map, key type: str, value type: str\n        @return: 无\n        @rtype: None\n        '
        pass

    def updateConfig(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        @brief: 重新设置配置\n        '

    def stringToProxy(self, servantProxy, connAddr):
        if False:
            i = 10
            return i + 15
        '\n        @brief: 初始化ServantProxy\n        @param connAddr: 服务器地址信息\n        @type connAddr: str\n        @param servant: servant proxy\n        @type servant: ServantProxy子类\n        @return: 无\n        @rtype: None\n        @note: 如果connAddr的ServantObj连接过，返回连接过的ServantProxy\n               如果没有连接过，用参数servant初始化，返回servant\n        '
        tarsLogger.debug('Communicator:stringToProxy')
        connInfo = self.parseConnAddr(connAddr)
        objName = connInfo['name']
        if objName in self.__servants:
            return self.__servants[objName]
        objectPrx = ObjectProxy()
        objectPrx.initialize(self, connInfo)
        servantPrx = servantProxy()
        servantPrx._initialize(self.__reactor, objectPrx)
        self.__objects[objName] = objectPrx
        self.__servants[objName] = servantPrx
        return servantPrx

    def handleTimeout(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        @brief: 处理超时事件\n        @return: 无\n        @rtype: None\n        '
        for obj in self.__objects.values():
            obj.handleQueueTimeout()

class ObjectProxy:
    """
    @brief: 一个object name在一个Communicator里有一个objectproxy
            管理收发的消息队列
    """
    DEFAULT_TIMEOUT = 3.0

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        tarsLogger.debug('ObjectProxy:__init__')
        self.__name = ''
        self.__timeout = ObjectProxy.DEFAULT_TIMEOUT
        self.__comm = None
        self.__epi = None
        self.__adpmanager = None
        self.__timeoutQueue = None
        self.__initialize = False

    def __del__(self):
        if False:
            i = 10
            return i + 15
        tarsLogger.debug('ObjectProxy:__del__')

    def initialize(self, comm, connInfo):
        if False:
            for i in range(10):
                print('nop')
        '\n        @brief: 初始化，使用ObjectProxy前必须调用\n        @param comm: 通讯器\n        @type comm: Communicator\n        @param connInfo: 连接信息\n        @type comm: dict\n        @return: None\n        @rtype: None\n        '
        if self.__initialize:
            return
        tarsLogger.debug('ObjectProxy:initialize')
        self.__comm = comm
        async_timeout = self.__comm.getProperty('async-invoke-timeout', float) / 1000
        self.__timeoutQueue = TimeoutQueue(async_timeout)
        self.__name = connInfo['name']
        self.__timeout = self.__comm.getProperty('sync-invoke-timeout', float) / 1000
        eplist = connInfo['endpoint']
        self.__adpmanager = AdapterProxyManager()
        self.__adpmanager.initialize(comm, self, eplist)
        self.__initialize = True

    def terminate(self):
        if False:
            return 10
        '\n        @brief: 回收资源，不再使用ObjectProxy时调用\n        @return: None\n        @rtype: None\n        '
        tarsLogger.debug('ObjectProxy:terminate')
        self.__timeoutQueue = None
        self.__adpmanager.terminate()
        self.__initialize = False

    def name(self):
        if False:
            print('Hello World!')
        '\n        @brief: 获取object name\n        @return: object name\n        @rtype: str\n        '
        return self.__name

    def timeout(self):
        if False:
            return 10
        '\n        @brief: 获取超时时间\n        @return: 超时时间，单位为s\n        @rtype: float\n        '
        return self.__timeout

    def getTimeoutQueue(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        @brief: 获取超时队列\n        @return: 超时队列\n        @rtype: TimeoutQueue\n        '
        return self.__timeoutQueue

    def handleQueueTimeout(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        @brief: 超时事件发生时处理超时事务\n        @return: None\n        @rtype: None\n        '
        self.__timeoutQueue.timeout()

    def invoke(self, reqmsg):
        if False:
            print('Hello World!')
        '\n        @brief: 远程过程调用\n        @param reqmsg: 请求响应报文\n        @type reqmsg: ReqMessage\n        @return: 错误码\n        @rtype:\n        '
        tarsLogger.debug('ObjectProxy:invoke, objname: %s, func: %s', self.__name, reqmsg.request.sFuncName)
        adapter = self.__adpmanager.selectAdapterProxy(reqmsg)
        if not adapter:
            tarsLogger.error('invoke %s, select adapter proxy return None', self.__name)
            return -2
        adapter.checkActive(True)
        reqmsg.adapter = adapter
        return adapter.invoke(reqmsg)

    def popRequest(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        @brief: 返回消息队列里的请求响应报文，FIFO\n                不删除TimeoutQueue里的数据，响应时要用\n        @return: 请求响应报文\n        @rtype: ReqMessage\n        '
        return self.__timeoutQueue.pop(erase=False)
if __name__ == '__main__':
    connAddr = 'apptest.lightServer.lightServantObj@tcp -h 10.130.64.220 -p 10001 -t 10000'
    connAddr = 'MTT.BookMarksUnifyServer.BookMarksUnifyObj@tcp -h 172.17.149.77 -t 60000 -p 10023'
    comm = Communicator()
    comm.initialize()
    servant = ServantProxy()
    servant = comm.stringToProxy(connAddr, servant)
    print(servant.tars_timeout())
    try:
        rsp = servant.tars_invoke(ServantProxy.TARSNORMAL, 'test', '', ServantProxy.mapcls_context(), None)
        print('Servant invoke success, request id: %d, iRet: %d' % (rsp.iRequestId, rsp.iRet))
    except Exception as msg:
        print(msg)
    finally:
        servant.tars_terminate()
    time.sleep(2)
    print('app closing ...')
    comm.terminate()
    time.sleep(2)
    print('cpp closed')