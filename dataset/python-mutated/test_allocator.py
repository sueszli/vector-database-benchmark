import unittest
import jittor as jt
import gc

class TestAllocator(unittest.TestCase):

    def test_stat(self):
        if False:
            print('Hello World!')
        jt.clean()
        with jt.flag_scope(use_stat_allocator=1, use_sfrl_allocator=0):
            a = jt.random([10, 10])
            b = a + a
            c = a * b
            c.data
            del a, b, c
            gc.collect()
        assert jt.flags.stat_allocator_total_alloc_call == 2
        assert jt.flags.stat_allocator_total_alloc_byte == 800
        assert jt.flags.stat_allocator_total_free_call == 2
        assert jt.flags.stat_allocator_total_free_byte == 800

    @unittest.skipIf(not jt.has_cuda, 'Cuda not found')
    @jt.flag_scope(use_cuda=1, use_cuda_managed_allocator=0)
    def test_device_allocator(self):
        if False:
            while True:
                i = 10
        a = jt.array([1, 2, 3, 4, 5])
        b = a + 1
        c = jt.code(a.shape, a.dtype, [b], cpu_src='\n            for (int i=0; i<in0_shape0; i++)\n                @out(i) = @in0(i)*@in0(i)*2;\n        ')
        assert (c.data == [8, 18, 32, 50, 72]).all()
if __name__ == '__main__':
    unittest.main()