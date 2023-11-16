from test.cinn.utils.testing import assert_llir_equal
from cinn import ir, to_cinn_llir
from cinn.runtime.data_array import DataArray
from cinn.schedule import IRSchedule as sch

def test_elementwise_parallel():
    if False:
        return 10

    @to_cinn_llir
    def elementwise_add(X: DataArray((128, 128)), Y: DataArray((128, 128)), A: DataArray((128, 128))):
        if False:
            while True:
                i = 10
        for i in range(128):
            for j in range(128):
                with ir.ScheduleBlockContext('A') as A_block:
                    (i1, j1) = ir.AxisMap('SS', [i, j])
                    A[i1, j1] = X[i1, j1] * 2.0
        for i in range(128):
            for j in range(128):
                with ir.ScheduleBlockContext('Y'):
                    (i1, j1) = ir.AxisMap('SS', [i, j])
                    Y[i1, j1] = A[i1, j1] + 2.0
        sch.parallel(A_block.i)
    assert_llir_equal(elementwise_add, elementwise_add)

def test_elementwise_vectorize():
    if False:
        while True:
            i = 10

    @to_cinn_llir
    def elementwise_add(X: DataArray((128, 128)), Y: DataArray((128, 128)), A: DataArray((128, 128))):
        if False:
            while True:
                i = 10
        for i in range(128):
            for j in range(128):
                with ir.ScheduleBlockContext('A') as A_block:
                    (i1, j1) = ir.AxisMap('SS', [i, j])
                    A[i1, j1] = X[i1, j1] * 2.0
        for i in range(128):
            for j0 in range(32):
                for j1 in range(4):
                    with ir.ScheduleBlockContext('Y') as Y_block:
                        (i1, j1) = ir.AxisMap('SS', [i, j0 * 4 + j1])
                        Y[i1, j1] = A[i1, j1] + 2.0
        sch.vectorize(Y_block.j1, 1)
    assert_llir_equal(elementwise_add, elementwise_add)

def test_elementwise_unroll():
    if False:
        for i in range(10):
            print('nop')

    @to_cinn_llir
    def elementwise_add(X: DataArray((128, 128)), Y: DataArray((128, 128)), A: DataArray((128, 128))):
        if False:
            print('Hello World!')
        for i in range(128):
            for j in range(128):
                with ir.ScheduleBlockContext('A') as A_block:
                    (i1, j1) = ir.AxisMap('SS', [i, j])
                    A[i1, j1] = X[i1, j1] * 2.0
        for i in range(128):
            for j0 in range(32):
                for j1 in range(4):
                    with ir.ScheduleBlockContext('Y') as Y_block:
                        (i1, j1) = ir.AxisMap('SS', [i, j0 * 4 + j1])
                        Y[i1, j1] = A[i1, j1] + 2.0
        sch.unroll(Y_block.j1)
    assert_llir_equal(elementwise_add, elementwise_add)
if __name__ == '__main__':
    test_elementwise_parallel()
    test_elementwise_vectorize()
    test_elementwise_unroll()