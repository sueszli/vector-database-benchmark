from test.cinn.utils.testing import assert_llir_equal
from cinn import ir, to_cinn_llir
from cinn.runtime.data_array import DataArray
from cinn.schedule import IRSchedule as sch

def test_split_reorder_elementwise():
    if False:
        print('Hello World!')

    @to_cinn_llir
    def split_reorder_elementwise(X: DataArray((1024, 1024)), Y: DataArray((1024, 1024)), Z: DataArray((1024, 1024))):
        if False:
            print('Hello World!')
        for i in range(1024):
            for j in range(1024):
                for k in range(1024):
                    with ir.ScheduleBlockContext('Z'):
                        (i_split_0, i_split_1, i_split_2, i_split_3) = sch.split(i, factors=[2, 4, 64, 2])
                        sch.reorder([i_split_2, i_split_0])
                        (i1, j1, k1) = ir.AxisMap('SSS', [i, j, k])
                        Z[i1, j1] = Z[i1, j1] + X[i1, k] * Y[k, j1]

    @to_cinn_llir
    def split_reorder_elementwise_gt(X: DataArray((1024, 1024)), Y: DataArray((1024, 1024)), Z: DataArray((1024, 1024))):
        if False:
            while True:
                i = 10
        for i_1 in range(64):
            for i_0 in range(4):
                for i in range(2):
                    for i_2 in range(2):
                        for j in range(1024):
                            for k in range(1024):
                                with ir.ScheduleBlockContext('Z'):
                                    (i1, j1, k1) = ir.AxisMap('SSS', [512 * i + (128 * i_0 + (2 * i_1 + i_2)), j, k])
                                    Z[i1, j1] = Z[i1, j1] + X[i1, k] * Y[k, j1]
    assert_llir_equal(split_reorder_elementwise, split_reorder_elementwise_gt)
if __name__ == '__main__':
    test_split_reorder_elementwise()