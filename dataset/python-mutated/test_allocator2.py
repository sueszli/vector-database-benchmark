import unittest
import jittor as jt
import gc

def test(h, w, total_alloc_call, total_alloc_byte, total_free_call=0, total_free_byte=0):
    if False:
        while True:
            i = 10
    jt.clean()
    jt.gc()
    with jt.flag_scope(use_stat_allocator=1):
        a = jt.random([h, w])
        b = a + a
        c = a * b
        c.data
        del a, b, c
        gc.collect()
        x = (jt.flags.stat_allocator_total_alloc_call, jt.flags.stat_allocator_total_alloc_byte, jt.flags.stat_allocator_total_free_call, jt.flags.stat_allocator_total_free_byte)
        y = (total_alloc_call, total_alloc_byte, total_free_call, total_free_byte)
        assert x == y, (x, y)

class TestAllocator2(unittest.TestCase):

    def test_stat(self):
        if False:
            i = 10
            return i + 15
        test(10, 10, 1, 1048576)
        test(100, 100, 1, 1048576)
        test(1000, 1000, 1, 20971520)
        test(8000, 1000, 2, 67108864)
if __name__ == '__main__':
    unittest.main()