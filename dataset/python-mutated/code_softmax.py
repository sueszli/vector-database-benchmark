import jittor as jt
from jittor import nn

def can_softmax_v1(a, dim):
    if False:
        for i in range(10):
            print('nop')
    if not jt.flags.use_cuda:
        return False
    if dim != -1 and dim != len(a.shape) - 1:
        return False
    if a.shape[len(a.shape) - 1] > 10000:
        return False
    return True

def softmax_v1(a, log=False):
    if False:
        for i in range(10):
            print('nop')
    assert can_softmax_v1(a, -1)
    length = a.shape[-1]
    tnum = 500 if length % 500 == 0 else 512
    tnum = 125 if length % 125 == 0 else 128
    per_thread = (length - 1) // tnum + 1
    ILP = 1
    for ilp in [8, 4, 2]:
        if length % tnum == 0 and per_thread % ilp == 0:
            ILP = ilp
            per_thread //= ILP
            break
    for_loop = f'\n    #pragma unroll\n    for (int i=0; i<{per_thread}; i++)\n    '
    if length % tnum != 0:
        for_loop += f'if ((i*{tnum}+threadIdx.x)*{ILP} < len)\n'

    class CodeSoftmax(jt.Function):

        def execute(self, x):
            if False:
                i = 10
                return i + 15
            self.save_vars = jt.code(x.shape, x.dtype, [x], cuda_header=f'\n#include <{jt.compile_extern.cub_home}cub/cub.cuh>\n#include <type/fp16_compute.h>\n', cuda_src=f'\n__global__ void kernel(in0_type* x, out0_type* y, int len) {{\n    typedef cub::BlockReduce<float, {tnum}> BlockReduce;\n    constexpr int need_log = {int(log)};\n    __shared__ typename BlockReduce::TempStorage temp_storage;\n\n    int id = blockIdx.x * len;\n    in0_type v[{per_thread}][{ILP}];\n    {for_loop}\n        vload<sizeof(in0_type)*{ILP}>(v[i], &x[id+(i*{tnum}+threadIdx.x)*{ILP}]);\n    // v[i] = x[id+i*{tnum}+threadIdx.x];\n    float v1 = -1e30;\n    {for_loop}\n        #pragma unroll\n        for (int j=0; j<{ILP}; j++) {{\n            v1 = ::max(v1, float(v[i][j]));\n        }}\n    __shared__ float vmax;\n    auto tmp = BlockReduce(temp_storage).Reduce(v1, cub::Max());\n    if (threadIdx.x == 0)\n        vmax = tmp;\n    __syncthreads();\n\n    v1 = 0;\n    {for_loop}\n        #pragma unroll\n        for (int j=0; j<{ILP}; j++) {{\n            if (need_log) {{\n                v[i][j] = float(v[i][j]) - vmax;\n                v1 += expf(float(v[i][j]));\n            }} else {{\n                v[i][j] = expf(float(v[i][j]) - vmax);\n                v1 += float(v[i][j]);\n            }}\n        }}\n\n    tmp = BlockReduce(temp_storage).Sum(v1);\n    __shared__ float vsum;\n    if (threadIdx.x == 0)\n        vsum = tmp;\n    __syncthreads();\n\n    {for_loop}\n        #pragma unroll\n        for (int j=0; j<{ILP}; j++) {{\n            if (need_log)\n                v[i][j] = v[i][j] - @expand_op(log,@in0_type,vsum);\n            else\n                v[i][j] = float(v[i][j])/vsum;\n        }}\n    {for_loop}\n        vload<sizeof(in0_type)*{ILP}>(&y[id+(i*{tnum}+threadIdx.x)*{ILP}], v[i]);\n}}\nint len = in0->shape[in0->shape.size()-1];\nint bnum = in0->numel() / len;\ncudaGetLastError();\nkernel<<<bnum, {tnum}>>>(in0_p, out0_p, len);\nCHECK(0 == cudaGetLastError());\n')
            return self.save_vars

        def grad(self, grad_x):
            if False:
                i = 10
                return i + 15
            x = self.save_vars
            return jt.code(x.shape, x.dtype, [x, grad_x], cuda_header=f'\n#include <{jt.compile_extern.cub_home}cub/cub.cuh>\n#include <type/fp16_compute.h>\n', cuda_src=f"\n__global__ void kernel(in0_type* x, in1_type* y, out0_type* z, int len) {{\n    int id = blockIdx.x * len;\n    in0_type vx[{per_thread}][{ILP}];\n    in0_type vy[{per_thread}][{ILP}];\n    {for_loop} {{\n        vload<sizeof(in0_type)*{ILP}>(vx[i], &x[id+(i*{tnum}+threadIdx.x)*{ILP}]);\n        vload<sizeof(in0_type)*{ILP}>(vy[i], &y[id+(i*{tnum}+threadIdx.x)*{ILP}]);\n    }}\n    float v1 = 0;\n    {for_loop} \n        #pragma unroll\n        for (int j=0; j<{ILP}; j++)\n            v1 += {('float(vy[i][j]);' if log else 'float(vx[i][j]*vy[i][j]);')}\n\n    typedef cub::BlockReduce<float, {tnum}> BlockReduce;\n    __shared__ typename BlockReduce::TempStorage temp_storage;\n    auto tmp = BlockReduce(temp_storage).Sum(v1);\n    __shared__ float reduce_var;\n    if (threadIdx.x == 0)\n        reduce_var = tmp;\n    __syncthreads();\n\n    {for_loop}\n        #pragma unroll\n        for (int j=0; j<{ILP}; j++)\n            vx[i][j] = {('vy[i][j] - in0_type(expf(vx[i][j]) * reduce_var);' if log else 'vx[i][j] * (vy[i][j] - in0_type(reduce_var));')}\n\n    {for_loop}\n        vload<sizeof(in0_type)*{ILP}>(&z[id+(i*{tnum}+threadIdx.x)*{ILP}],\n            vx[i]);\n}}\nint len = in0->shape[in0->shape.size()-1];\nint bnum = in0->numel() / len;\ncudaGetLastError();\nkernel<<<bnum, {tnum}>>>(in0_p, in1_p, out0_p, len);\nCHECK(0 == cudaGetLastError());\n")
    return CodeSoftmax()(a)