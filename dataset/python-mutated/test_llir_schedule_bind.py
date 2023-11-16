from test.cinn.utils.testing import assert_llir_equal
from cinn import ir, to_cinn_llir
from cinn.runtime.data_array import DataArray
from cinn.schedule import IRSchedule as sch

def test_bind_reduce():
    if False:
        print('Hello World!')

    @to_cinn_llir
    def reduce_sum(A: DataArray((1, 4, 256, 512)), B: DataArray((1, 4, 256))):
        if False:
            for i in range(10):
                print('nop')
        for i1 in range(1):
            for j1 in range(4):
                for k1 in range(256):
                    with ir.ScheduleBlockContext('init') as init:
                        (vi, vj, vk) = ir.AxisMap('SSS', [i1, j1, k1])
                        B[vi, vj, vk] = 0.0
                    for l1 in range(512):
                        with ir.ScheduleBlockContext('B'):
                            sch.bind(i1, 'blockIdx.x')
                            sch.bind(j1, 'threadIdx.y')
                            sch.bind(k1, 'threadIdx.x')
                            (vi1, vj1, vk1, vl1) = ir.AxisMap('SSSR', [i1, j1, k1, l1])
                            B[vi1, vj1, vk1] = B[vi1, vj1, vk1] + A[vi1, vj1, vk1, vl1]

    @to_cinn_llir
    def reduce_sum_expected(A: DataArray((1, 4, 256, 512)), B: DataArray((1, 4, 256))):
        if False:
            print('Hello World!')
        for i1 in range(1):
            for j1 in range(4):
                for k1 in range(256):
                    with ir.ScheduleBlockContext('init') as init:
                        (vi, vj, vk) = ir.AxisMap('SSS', [i1, j1, k1])
                        B[vi, vj, vk] = 0.0
                    for l1 in range(512):
                        with ir.ScheduleBlockContext('B'):
                            (vi1, vj1, vk1, vl1) = ir.AxisMap('SSSR', [i1, j1, k1, l1])
                            B[vi1, vj1, vk1] = B[vi1, vj1, vk1] + A[vi1, vj1, vk1, vl1]
        sch.bind(init.i1, 'blockIdx.x')
        sch.bind(init.j1, 'threadIdx.y')
        sch.bind(init.k1, 'threadIdx.x')
    assert_llir_equal(reduce_sum, reduce_sum_expected)
if __name__ == '__main__':
    test_bind_reduce()