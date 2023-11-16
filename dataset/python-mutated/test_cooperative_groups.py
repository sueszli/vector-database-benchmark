import pytest
import cupy
from cupy.cuda import runtime
from cupyx import jit

@pytest.mark.skipif(runtime.is_hip, reason='not supported on HIP')
class TestCooperativeGroups:

    def test_thread_block_group(self):
        if False:
            return 10

        @jit.rawkernel()
        def test_thread_block(x):
            if False:
                i = 10
                return i + 15
            g = jit.cg.this_thread_block()
            if g.thread_rank() == 0:
                x[0] += 101
            if g.thread_rank() == 1:
                x[1] = g.size()
            if g.thread_rank() == 2:
                g_idx = g.group_index()
                (x[2], x[3], x[4]) = (g_idx.x, g_idx.y, g_idx.z)
            if g.thread_rank() == 3:
                t_idx = g.thread_index()
                (x[5], x[6], x[7]) = (t_idx.x, t_idx.y, t_idx.z)
            if g.thread_rank() == 4:
                g_dim = g.group_dim()
                (x[8], x[9], x[10]) = (g_dim.x, g_dim.y, g_dim.z)
            g.sync()
        x = cupy.empty((16,), dtype=cupy.int64)
        x[:] = -1
        test_thread_block[1, 32](x)
        assert x[0] == 100
        assert x[1] == 32
        assert (x[2], x[3], x[4]) == (0, 0, 0)
        assert (x[5], x[6], x[7]) == (3, 0, 0)
        assert (x[8], x[9], x[10]) == (32, 1, 1)
        assert (x[11:] == -1).all()

    @pytest.mark.skipif(runtime.runtimeGetVersion() < 11060 or (cupy.cuda.driver._is_cuda_python() and cupy.cuda.nvrtc.getVersion() < (11, 6)), reason='not supported until CUDA 11.6')
    def test_thread_block_group_cu116_new_APIs(self):
        if False:
            for i in range(10):
                print('nop')

        @jit.rawkernel()
        def test_thread_block(x):
            if False:
                print('Hello World!')
            g = jit.cg.this_thread_block()
            if g.thread_rank() == 0:
                x[0] = g.num_threads()
            if g.thread_rank() == 1:
                d_th = g.dim_threads()
                (x[1], x[2], x[3]) = (d_th.x, d_th.y, d_th.z)
            g.sync()
        x = cupy.empty((16,), dtype=cupy.int64)
        x[:] = -1
        test_thread_block[1, 32](x)
        assert x[0] == 32
        assert (x[1], x[2], x[3]) == (32, 1, 1)
        assert (x[4:] == -1).all()

    @pytest.mark.skipif(runtime.runtimeGetVersion() < 11000, reason='we do not support it')
    @pytest.mark.skipif(runtime.deviceGetAttribute(runtime.cudaDevAttrCooperativeLaunch, 0) == 0, reason='cooperative launch is not supported on device 0')
    def test_grid_group(self):
        if False:
            return 10

        @jit.rawkernel()
        def test_grid(x):
            if False:
                for i in range(10):
                    print('nop')
            g = jit.cg.this_grid()
            if g.thread_rank() == 0:
                x[0] = g.is_valid()
            if g.thread_rank() == 1:
                x[1] = g.size()
            if g.thread_rank() == 32:
                g_dim = g.group_dim()
                (x[2], x[3], x[4]) = (g_dim.x, g_dim.y, g_dim.z)
            g.sync()
        x = cupy.empty((16,), dtype=cupy.uint64)
        x[:] = -1
        test_grid[2, 32](x)
        assert x[0] == 1
        assert x[1] == 64
        assert (x[2], x[3], x[4]) == (2, 1, 1)
        assert (x[5:] == 2 ** 64 - 1).all()

    @pytest.mark.skipif(runtime.runtimeGetVersion() < 11060 or (cupy.cuda.driver._is_cuda_python() and cupy.cuda.nvrtc.getVersion() < (11, 6)), reason='not supported until CUDA 11.6')
    @pytest.mark.skipif(runtime.deviceGetAttribute(runtime.cudaDevAttrCooperativeLaunch, 0) == 0, reason='cooperative launch is not supported on device 0')
    def test_grid_group_cu116_new_APIs(self):
        if False:
            while True:
                i = 10

        @jit.rawkernel()
        def test_grid(x):
            if False:
                for i in range(10):
                    print('nop')
            g = jit.cg.this_grid()
            if g.thread_rank() == 1:
                x[1] = g.num_threads()
            if g.thread_rank() == 32:
                g_dim = g.dim_blocks()
                (x[2], x[3], x[4]) = (g_dim.x, g_dim.y, g_dim.z)
            if g.thread_rank() == 33:
                x[5] = g.block_rank()
            if g.thread_rank() == 2:
                x[6] = g.num_blocks()
            if g.thread_rank() == 34:
                b_idx = g.block_index()
                (x[7], x[8], x[9]) = (b_idx.x, b_idx.y, b_idx.z)
            g.sync()
        x = cupy.empty((16,), dtype=cupy.uint64)
        x[:] = -1
        test_grid[2, 32](x)
        assert x[1] == 64
        assert (x[2], x[3], x[4]) == (2, 1, 1)
        assert x[5] == 1
        assert x[6] == 2
        assert (x[7], x[8], x[9]) == (1, 0, 0)
        assert (x[10:] == 2 ** 64 - 1).all()

    @pytest.mark.skipif(runtime.runtimeGetVersion() < 11000, reason='we do not support it')
    @pytest.mark.skipif(runtime.deviceGetAttribute(runtime.cudaDevAttrCooperativeLaunch, 0) == 0, reason='cooperative launch is not supported on device 0')
    def test_cg_sync(self):
        if False:
            return 10

        @jit.rawkernel()
        def test_sync():
            if False:
                for i in range(10):
                    print('nop')
            b = jit.cg.this_thread_block()
            g = jit.cg.this_grid()
            jit.cg.sync(b)
            jit.cg.sync(g)
        test_sync[2, 64]()

    @pytest.mark.skipif(runtime.runtimeGetVersion() < 11010, reason='not supported until CUDA 11.0')
    @pytest.mark.parametrize('test_aligned', (True, False))
    def test_cg_memcpy_async_wait_for_wait(self, test_aligned):
        if False:
            for i in range(10):
                print('nop')

        @jit.rawkernel()
        def test_copy(x, y):
            if False:
                print('Hello World!')
            if test_aligned:
                smem = jit.shared_memory(cupy.int32, 32 * 2, alignment=16)
            else:
                smem = jit.shared_memory(cupy.int32, 32 * 2)
            g = jit.cg.this_thread_block()
            tid = g.thread_rank()
            if test_aligned:
                jit.cg.memcpy_async(g, smem, 0, x, 0, 4 * 32, aligned_size=16)
                jit.cg.memcpy_async(g, smem, 32, x, 32, 4 * 32, aligned_size=16)
            else:
                jit.cg.memcpy_async(g, smem, 0, x, 0, 4 * 32)
                jit.cg.memcpy_async(g, smem, 32, x, 32, 4 * 32)
            jit.cg.wait_prior(g, 1)
            if tid < 32:
                y[tid] = smem[tid]
            jit.cg.wait(g)
            if 32 <= tid and tid < 64:
                y[tid] = smem[tid]
        x = cupy.arange(64, dtype=cupy.int32)
        y = cupy.zeros(64, dtype=cupy.int32)
        test_copy[2, 64](x, y)
        assert (x == y).all()