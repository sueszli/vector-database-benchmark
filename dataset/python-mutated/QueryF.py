from .__init__ import tarscore
from .__servantproxy import ServantProxy
from .__async import ServantProxyCallback
from .EndpointF import EndpointF
import time

class QueryFProxy(ServantProxy):

    def findObjectById(self, id, context=ServantProxy.mapcls_context()):
        if False:
            print('Hello World!')
        oos = tarscore.TarsOutputStream()
        oos.write(tarscore.string, 1, id)
        rsp = self.tars_invoke(ServantProxy.TARSNORMAL, 'findObjectById', oos.getBuffer(), context, None)
        ios = tarscore.TarsInputStream(rsp.sBuffer)
        ret = ios.read(tarscore.vctclass(EndpointF), 0, True)
        return ret

    def async_findObjectById(self, callback, id, context=ServantProxy.mapcls_context()):
        if False:
            return 10
        oos = tarscore.TarsOutputStream()
        oos.write(tarscore.string, 1, id)
        self.tars_invoke_async(ServantProxy.TARSNORMAL, 'findObjectById', oos.getBuffer(), context, None, callback)

    def findObjectById4Any(self, id, context=ServantProxy.mapcls_context()):
        if False:
            i = 10
            return i + 15
        oos = tarscore.TarsOutputStream()
        oos.write(tarscore.string, 1, id)
        rsp = self.tars_invoke(ServantProxy.TARSNORMAL, 'findObjectById4Any', oos.getBuffer(), context, None)
        ios = tarscore.TarsInputStream(rsp.sBuffer)
        ret = ios.read(tarscore.int32, 0, True)
        activeEp = ios.read(tarscore.vctclass(EndpointF), 2, True)
        inactiveEp = ios.read(tarscore.vctclass(EndpointF), 3, True)
        return (ret, activeEp, inactiveEp)

    def async_findObjectById4Any(self, callback, id, context=ServantProxy.mapcls_context()):
        if False:
            for i in range(10):
                print('nop')
        oos = tarscore.TarsOutputStream()
        oos.write(tarscore.string, 1, id)
        self.tars_invoke_async(ServantProxy.TARSNORMAL, 'findObjectById4Any', oos.getBuffer(), context, None, callback)

    def findObjectById4All(self, id, context=ServantProxy.mapcls_context()):
        if False:
            return 10
        oos = tarscore.TarsOutputStream()
        oos.write(tarscore.string, 1, id)
        rsp = self.tars_invoke(ServantProxy.TARSNORMAL, 'findObjectById4All', oos.getBuffer(), context, None)
        ios = tarscore.TarsInputStream(rsp.sBuffer)
        ret = ios.read(tarscore.int32, 0, True)
        activeEp = ios.read(tarscore.vctclass(EndpointF), 2, True)
        inactiveEp = ios.read(tarscore.vctclass(EndpointF), 3, True)
        return (ret, activeEp, inactiveEp)

    def async_findObjectById4All(self, callback, id, context=ServantProxy.mapcls_context()):
        if False:
            print('Hello World!')
        oos = tarscore.TarsOutputStream()
        oos.write(tarscore.string, 1, id)
        self.tars_invoke_async(ServantProxy.TARSNORMAL, 'findObjectById4All', oos.getBuffer(), context, None, callback)

    def findObjectByIdInSameGroup(self, id, context=ServantProxy.mapcls_context()):
        if False:
            for i in range(10):
                print('nop')
        oos = tarscore.TarsOutputStream()
        oos.write(tarscore.string, 1, id)
        rsp = self.tars_invoke(ServantProxy.TARSNORMAL, 'findObjectByIdInSameGroup', oos.getBuffer(), context, None)
        startDecodeTime = time.time()
        ios = tarscore.TarsInputStream(rsp.sBuffer)
        ret = ios.read(tarscore.int32, 0, True)
        activeEp = ios.read(tarscore.vctclass(EndpointF), 2, True)
        inactiveEp = ios.read(tarscore.vctclass(EndpointF), 3, True)
        endDecodeTime = time.time()
        return (ret, activeEp, inactiveEp, endDecodeTime - startDecodeTime)

    def async_findObjectByIdInSameGroup(self, callback, id, context=ServantProxy.mapcls_context()):
        if False:
            print('Hello World!')
        oos = tarscore.TarsOutputStream()
        oos.write(tarscore.string, 1, id)
        self.tars_invoke_async(ServantProxy.TARSNORMAL, 'findObjectByIdInSameGroup', oos.getBuffer(), context, None, callback)

    def findObjectByIdInSameStation(self, id, sStation, context=ServantProxy.mapcls_context()):
        if False:
            while True:
                i = 10
        oos = tarscore.TarsOutputStream()
        oos.write(tarscore.string, 1, id)
        oos.write(tarscore.string, 2, sStation)
        rsp = self.tars_invoke(ServantProxy.TARSNORMAL, 'findObjectByIdInSameStation', oos.getBuffer(), context, None)
        ios = tarscore.TarsInputStream(rsp.sBuffer)
        ret = ios.read(tarscore.int32, 0, True)
        activeEp = ios.read(tarscore.vctclass(EndpointF), 3, True)
        inactiveEp = ios.read(tarscore.vctclass(EndpointF), 4, True)
        return (ret, activeEp, inactiveEp)

    def async_findObjectByIdInSameStation(self, callback, id, sStation, context=ServantProxy.mapcls_context()):
        if False:
            i = 10
            return i + 15
        oos = tarscore.TarsOutputStream()
        oos.write(tarscore.string, 1, id)
        oos.write(tarscore.string, 2, sStation)
        self.tars_invoke_async(ServantProxy.TARSNORMAL, 'findObjectByIdInSameStation', oos.getBuffer(), context, None, callback)

    def findObjectByIdInSameSet(self, id, setId, context=ServantProxy.mapcls_context()):
        if False:
            while True:
                i = 10
        oos = tarscore.TarsOutputStream()
        oos.write(tarscore.string, 1, id)
        oos.write(tarscore.string, 2, setId)
        rsp = self.tars_invoke(ServantProxy.TARSNORMAL, 'findObjectByIdInSameSet', oos.getBuffer(), context, None)
        ios = tarscore.TarsInputStream(rsp.sBuffer)
        ret = ios.read(tarscore.int32, 0, True)
        activeEp = ios.read(tarscore.vctclass(EndpointF), 3, True)
        inactiveEp = ios.read(tarscore.vctclass(EndpointF), 4, True)
        return (ret, activeEp, inactiveEp)

    def async_findObjectByIdInSameSet(self, callback, id, setId, context=ServantProxy.mapcls_context()):
        if False:
            for i in range(10):
                print('nop')
        oos = tarscore.TarsOutputStream()
        oos.write(tarscore.string, 1, id)
        oos.write(tarscore.string, 2, setId)
        self.tars_invoke_async(ServantProxy.TARSNORMAL, 'findObjectByIdInSameSet', oos.getBuffer(), context, None, callback)

class QueryFPrxCallback(ServantProxyCallback):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        ServantProxyCallback.__init__(self)
        self.callback_map = {'findObjectById': self.__invoke_findObjectById, 'findObjectById4Any': self.__invoke_findObjectById4Any, 'findObjectById4All': self.__invoke_findObjectById4All, 'findObjectByIdInSameGroup': self.__invoke_findObjectByIdInSameGroup, 'findObjectByIdInSameStation': self.__invoke_findObjectByIdInSameStation, 'findObjectByIdInSameSet': self.__invoke_findObjectByIdInSameSet}

    def callback_findObjectById(self, ret):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def callback_findObjectById_exception(self, ret):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def callback_findObjectById4Any(self, ret, activeEp, inactiveEp):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def callback_findObjectById4Any_exception(self, ret):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def callback_findObjectById4All(self, ret, activeEp, inactiveEp):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def callback_findObjectById4All_exception(self, ret):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def callback_findObjectByIdInSameGroup(self, ret, activeEp, inactiveEp):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def callback_findObjectByIdInSameGroup_exception(self, ret):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def callback_findObjectByIdInSameStation(self, ret, activeEp, inactiveEp):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def callback_findObjectByIdInSameStation_exception(self, ret):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def callback_findObjectByIdInSameSet(self, ret, activeEp, inactiveEp):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def callback_findObjectByIdInSameSet_exception(self, ret):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def __invoke_findObjectById(self, reqmsg):
        if False:
            while True:
                i = 10
        rsp = reqmsg.response
        if rsp.iRet != ServantProxy.TARSSERVERSUCCESS:
            self.callback_findObjectById_exception(rsp.iRet)
            return rsp.iRet
        ios = tarscore.TarsInputStream(rsp.sBuffer)
        ret = ios.read(tarscore.vctclass(EndpointF), 0, True)
        self.callback_findObjectById(ret)

    def __invoke_findObjectById4Any(self, reqmsg):
        if False:
            for i in range(10):
                print('nop')
        rsp = reqmsg.response
        if rsp.iRet != ServantProxy.TARSSERVERSUCCESS:
            self.callback_findObjectById4Any_exception(rsp.iRet)
            return rsp.iRet
        ios = tarscore.TarsInputStream(rsp.sBuffer)
        ret = ios.read(tarscore.int32, 0, True)
        activeEp = ios.read(tarscore.vctclass(EndpointF), 2, True)
        inactiveEp = ios.read(tarscore.vctclass(EndpointF), 3, True)
        self.callback_findObjectById4Any(ret, activeEp, inactiveEp)

    def __invoke_findObjectById4All(self, reqmsg):
        if False:
            print('Hello World!')
        rsp = reqmsg.response
        if rsp.iRet != ServantProxy.TARSSERVERSUCCESS:
            self.callback_findObjectById4All_exception(rsp.iRet)
            return rsp.iRet
        ios = tarscore.TarsInputStream(rsp.sBuffer)
        ret = ios.read(tarscore.int32, 0, True)
        activeEp = ios.read(tarscore.vctclass(EndpointF), 2, True)
        inactiveEp = ios.read(tarscore.vctclass(EndpointF), 3, True)
        self.callback_findObjectById4All(ret, activeEp, inactiveEp)

    def __invoke_findObjectByIdInSameGroup(self, reqmsg):
        if False:
            while True:
                i = 10
        rsp = reqmsg.response
        if rsp.iRet != ServantProxy.TARSSERVERSUCCESS:
            self.callback_findObjectByIdInSameGroup_exception(rsp.iRet)
            return rsp.iRet
        ios = tarscore.TarsInputStream(rsp.sBuffer)
        ret = ios.read(tarscore.int32, 0, True)
        activeEp = ios.read(tarscore.vctclass(EndpointF), 2, True)
        inactiveEp = ios.read(tarscore.vctclass(EndpointF), 3, True)
        self.callback_findObjectByIdInSameGroup(ret, activeEp, inactiveEp)

    def __invoke_findObjectByIdInSameStation(self, reqmsg):
        if False:
            return 10
        rsp = reqmsg.response
        if rsp.iRet != ServantProxy.TARSSERVERSUCCESS:
            self.callback_findObjectByIdInSameStation_exception(rsp.iRet)
            return rsp.iRet
        ios = tarscore.TarsInputStream(rsp.sBuffer)
        ret = ios.read(tarscore.int32, 0, True)
        activeEp = ios.read(tarscore.vctclass(EndpointF), 3, True)
        inactiveEp = ios.read(tarscore.vctclass(EndpointF), 4, True)
        self.callback_findObjectByIdInSameStation(ret, activeEp, inactiveEp)

    def __invoke_findObjectByIdInSameSet(self, reqmsg):
        if False:
            for i in range(10):
                print('nop')
        rsp = reqmsg.response
        if rsp.iRet != ServantProxy.TARSSERVERSUCCESS:
            self.callback_findObjectByIdInSameSet_exception(rsp.iRet)
            return rsp.iRet
        ios = tarscore.TarsInputStream(rsp.sBuffer)
        ret = ios.read(tarscore.int32, 0, True)
        activeEp = ios.read(tarscore.vctclass(EndpointF), 3, True)
        inactiveEp = ios.read(tarscore.vctclass(EndpointF), 4, True)
        self.callback_findObjectByIdInSameSet(ret, activeEp, inactiveEp)

    def onDispatch(self, reqmsg):
        if False:
            return 10
        self.callback_map[reqmsg.request.sFuncName](reqmsg)