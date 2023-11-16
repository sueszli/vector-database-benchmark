from test.cinn.utils.testing import assert_llir_equal
from cinn import ir, to_cinn_llir
from cinn.runtime.data_array import DataArray
from cinn.schedule import IRSchedule as sch

def test_reorder_elementwise():
    if False:
        return 10

    @to_cinn_llir
    def reorder_elementwise(X: DataArray((64, 64, 64, 64)), Y: DataArray((64, 64, 64, 64))):
        if False:
            while True:
                i = 10
        for i in range(64):
            for j in range(64):
                for k in range(64):
                    for l in range(8):
                        with ir.ScheduleBlockContext('Y') as Y_block:
                            (vi, vj, vk, vl) = ir.AxisMap('SSSS', [i, j, k, 8 * l])
                            Y[vi, vj, vk, vl] = X[vi, vj, vk, vl] * 2.0
        sch.reorder([Y_block.k, Y_block.l, Y_block.i])

    @to_cinn_llir
    def reorder_elementwise_gt(X: DataArray((64, 64, 64, 64)), Y: DataArray((64, 64, 64, 64))):
        if False:
            for i in range(10):
                print('nop')
        for k in range(64):
            for j in range(64):
                for l in range(8):
                    for i in range(64):
                        with ir.ScheduleBlockContext('Y'):
                            (vi, vj, vk, vl) = ir.AxisMap('SSSS', [i, j, k, 8 * l])
                            Y[vi, vj, vk, vl] = X[vi, vj, vk, vl] * 2.0
    assert_llir_equal(reorder_elementwise, reorder_elementwise_gt)

def test_reorder_overlapped():
    if False:
        print('Hello World!')

    @to_cinn_llir
    def reorder_overlapped(X: DataArray((28, 8)), Y: DataArray((28, 8))):
        if False:
            for i in range(10):
                print('nop')
        for i in range(12):
            for j in range(4):
                for k in range(4):
                    with ir.ScheduleBlockContext('Y'):
                        (vi, vj) = ir.AxisMap('SS', [i, j])
                        sch.reorder([i, k, j])
                        Y[vi, vj] = X[vi, vj] + 1.0

    @to_cinn_llir
    def reorder_overlapped_gt(X: DataArray((28, 8)), Y: DataArray((28, 8))):
        if False:
            i = 10
            return i + 15
        for i in range(12):
            for k in range(4):
                for j in range(4):
                    with ir.ScheduleBlockContext('Y'):
                        (vi, vj) = ir.AxisMap('SS', [i, j])
                        Y[vi, vj] = X[vi, vj] + 1.0
    assert_llir_equal(reorder_overlapped, reorder_overlapped_gt)
if __name__ == '__main__':
    test_reorder_elementwise()
    test_reorder_overlapped()