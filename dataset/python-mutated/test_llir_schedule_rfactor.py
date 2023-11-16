from cinn import ir, to_cinn_llir
from cinn.runtime.data_array import DataArray
from cinn.schedule import IRSchedule as sch

def test_matmul():
    if False:
        while True:
            i = 10

    @to_cinn_llir
    def matmul(A: DataArray((128, 128)), B: DataArray((128, 128)), C: DataArray((128, 128))):
        if False:
            while True:
                i = 10
        for i0 in range(128):
            for i1 in range(128):
                with ir.ScheduleBlockContext('init'):
                    (vi, vj) = ir.AxisMap('SS', [i0, i1])
                    C[vi, vj] = 0.0
                for i2_outer in range(4):
                    for i2_inner_outer in range(8):
                        for i2_inner_inner in range(4):
                            with ir.ScheduleBlockContext('compute') as Compute_block:
                                (vi, vj, vk) = ir.AxisMap('SSR', [i0, i1, i2_outer * 32 + i2_inner_outer * 4 + i2_inner_inner])
                                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]
        sch.rfactor(Compute_block.i2_inner_inner, 0)
if __name__ == '__main__':
    test_matmul()