import pickle
import pytest
import cupy
from cupy.cuda import driver
from cupy.cuda import nvrtc
from cupy.cuda import runtime

class TestExceptionPicklable:

    def test(self):
        if False:
            i = 10
            return i + 15
        e1 = runtime.CUDARuntimeError(1)
        e2 = pickle.loads(pickle.dumps(e1))
        assert e1.args == e2.args
        assert str(e1) == str(e2)

class TestMemPool:

    @pytest.mark.skipif(runtime.is_hip, reason='HIP does not support async allocator')
    @pytest.mark.skipif(driver._is_cuda_python() and runtime.runtimeGetVersion() < 11020, reason='cudaMemPool_t is supported since CUDA 11.2')
    @pytest.mark.skipif(not driver._is_cuda_python() and driver.get_build_version() < 11020, reason='cudaMemPool_t is supported since CUDA 11.2')
    @pytest.mark.skipif(runtime.deviceGetAttribute(runtime.cudaDevAttrMemoryPoolsSupported, 0) == 0, reason='cudaMemPool_t is not supported on device 0')
    def test_mallocFromPoolAsync(self):
        if False:
            return 10
        props = runtime.MemPoolProps(runtime.cudaMemAllocationTypePinned, runtime.cudaMemHandleTypeNone, runtime.cudaMemLocationTypeDevice, 0)
        pool = runtime.memPoolCreate(props)
        assert pool > 0
        s = cupy.cuda.Stream()
        ptr = runtime.mallocFromPoolAsync(128, pool, s.ptr)
        assert ptr > 0
        runtime.freeAsync(ptr, s.ptr)
        runtime.memPoolDestroy(pool)

@pytest.mark.skipif(runtime.is_hip, reason='This assumption is correct only in CUDA')
def test_assumed_runtime_version():
    if False:
        return 10
    (major, minor) = nvrtc.getVersion()
    assert runtime.runtimeGetVersion() == major * 1000 + minor * 10