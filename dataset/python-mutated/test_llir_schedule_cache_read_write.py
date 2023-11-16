from test.cinn.utils.testing import assert_llir_equal
from cinn import ir, to_cinn_llir
from cinn.runtime.data_array import DataArray
from cinn.schedule import IRSchedule as sch

def test_cache_read_elementwise():
    if False:
        while True:
            i = 10

    @to_cinn_llir
    def elementwise_add_cache_read(X: DataArray((128, 128)), Y: DataArray((128, 128)), A: DataArray((128, 128))):
        if False:
            for i in range(10):
                print('nop')
        for i in range(128):
            for j in range(128):
                with ir.ScheduleBlockContext('A') as A_block:
                    (i1, j1) = ir.AxisMap('SS', [i, j])
                    A[i1, j1] = X[i1, j1] * 2.0
        for i3 in range(128):
            for j3 in range(128):
                with ir.ScheduleBlockContext('B') as B_block:
                    (i1, j1) = ir.AxisMap('SS', [i3, j3])
                    Y[i1, j1] = -A[i1, j1] + 3.0
        cached_a = sch.cache_read(A_block.block, 0, 'global')
        cached_b = sch.cache_read(B_block.block, 0, 'local')
    assert_llir_equal(elementwise_add_cache_read, elementwise_add_cache_read)

def test_cache_write_elementwise():
    if False:
        i = 10
        return i + 15

    @to_cinn_llir
    def elementwise_add_cache_write(X: DataArray((128, 128)), Y: DataArray((128, 128)), A: DataArray((128, 128))):
        if False:
            for i in range(10):
                print('nop')
        for i in range(128):
            for j in range(128):
                with ir.ScheduleBlockContext('A') as A_block:
                    (i1, j1) = ir.AxisMap('SS', [i, j])
                    A[i1, j1] = X[i1, j1] * 2.0
        for i3 in range(128):
            for j3 in range(128):
                with ir.ScheduleBlockContext('B') as B_block:
                    (i1, j1) = ir.AxisMap('SS', [i3, j3])
                    Y[i1, j1] = -A[i1, j1] + 3.0
        cached_a = sch.cache_write(A_block.block, 0, 'global')
        cached_b = sch.cache_write(B_block.block, 0, 'local')
if __name__ == '__main__':
    test_cache_read_elementwise()
    test_cache_write_elementwise()