from cinn import ir, lang, to_cinn_llir
from cinn.runtime.data_array import DataArray

def test_call_extern():
    if False:
        i = 10
        return i + 15

    @to_cinn_llir
    def call_sinh(A: DataArray((1, 4, 256, 512)), B: DataArray((1, 4, 256))):
        if False:
            i = 10
            return i + 15
        for i1 in range(1):
            for j1 in range(4):
                for k1 in range(256):
                    with ir.ScheduleBlockContext('init') as init:
                        (vi, vj, vk) = ir.AxisMap('SSS', [i1, j1, k1])
                        B[vi, vj, vk] = lang.call_extern('sinh', [A[vi, vi, vj, vk]], {})
    str(call_sinh)
if __name__ == '__main__':
    test_call_extern()