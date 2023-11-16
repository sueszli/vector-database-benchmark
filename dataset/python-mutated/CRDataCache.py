from direct.distributed.CachedDOData import CachedDOData
from panda3d.core import ConfigVariableInt
__all__ = ['CRDataCache']

class CRDataCache:

    def __init__(self):
        if False:
            return 10
        self._doId2name2data = {}
        self._size = ConfigVariableInt('crdatacache-size', 10).getValue()
        assert self._size > 0
        self._junkIndex = 0

    def destroy(self):
        if False:
            return 10
        del self._doId2name2data

    def setCachedData(self, doId, name, data):
        if False:
            print('Hello World!')
        assert isinstance(data, CachedDOData)
        if len(self._doId2name2data) >= self._size:
            if self._junkIndex >= len(self._doId2name2data):
                self._junkIndex = 0
            junkDoId = list(self._doId2name2data.keys())[self._junkIndex]
            self._junkIndex += 1
            for name in self._doId2name2data[junkDoId]:
                self._doId2name2data[junkDoId][name].flush()
            del self._doId2name2data[junkDoId]
        self._doId2name2data.setdefault(doId, {})
        cachedData = self._doId2name2data[doId].get(name)
        if cachedData:
            cachedData.flush()
            cachedData.destroy()
        self._doId2name2data[doId][name] = data

    def hasCachedData(self, doId):
        if False:
            i = 10
            return i + 15
        return doId in self._doId2name2data

    def popCachedData(self, doId):
        if False:
            return 10
        data = self._doId2name2data[doId]
        del self._doId2name2data[doId]
        return data

    def flush(self):
        if False:
            while True:
                i = 10
        for doId in self._doId2name2data:
            for name in self._doId2name2data[doId]:
                self._doId2name2data[doId][name].flush()
        self._doId2name2data = {}
    if __debug__:

        def _startMemLeakCheck(self):
            if False:
                i = 10
                return i + 15
            self._len = len(self._doId2name2data)

        def _stopMemLeakCheck(self):
            if False:
                i = 10
                return i + 15
            del self._len

        def _checkMemLeaks(self):
            if False:
                i = 10
                return i + 15
            assert self._len == len(self._doId2name2data)
if __debug__:

    class TestCachedData(CachedDOData):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            CachedDOData.__init__(self)
            self._destroyed = False
            self._flushed = False

        def destroy(self):
            if False:
                print('Hello World!')
            CachedDOData.destroy(self)
            self._destroyed = True

        def flush(self):
            if False:
                return 10
            CachedDOData.flush(self)
            self._flushed = True