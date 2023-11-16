import unittest
import jittor as jt
import numpy as np
import os

def transpose0231(x):
    if False:
        while True:
            i = 10
    (s0, s1, s2, s3) = x.shape
    asize = 16
    bsize = 16
    ILP = 2
    return jt.code([s0, s2, s3, s1], x.dtype, [x], cuda_header='#include <type/fp16_compute.h>\n#include <cassert>', cuda_src=f'\n    __global__ void kernel(in0_type* __restrict__ x, in0_type* __restrict__ y, int s0, int s1, int s2, int s3) {{\n        __shared__ in0_type t[{asize * ILP}*{bsize * ILP + 1}];\n        int t3 = threadIdx.x % {bsize};\n        int t1 = threadIdx.x / {bsize};\n        int b3 = blockIdx.x;\n        int b2 = blockIdx.y;\n        int b0 = blockIdx.z;\n        int x3 = 1;\n        int x2 = s3;\n        int x1 = s2*x2;\n        int x0 = s1*x1;\n        int y3 = 1;\n        int y2 = s1;\n        int y1 = s3*y2;\n        int y0 = s2*y1;\n        in0_type tmp[{ILP}];\n        for (int i=0; i<(s1-1)/{asize * ILP}+1; i++)\n        {{\n            int _b3 = b3 * {bsize * ILP} + t3*{ILP};\n            if (_b3 < s3) {{\n                #pragma unroll\n                for (int j=0; j<{ILP}; j++) {{\n                    vload<sizeof(in0_type)*{ILP}>(\n                        tmp,\n                        &x[b0*x0+(t1*{ILP}+j+i*{asize * ILP})*x1+b2*x2+_b3*x3]\n                    );\n                    #pragma unroll\n                    for (int k=0; k<{ILP}; k++)\n                        t[(t1*{ILP}+j)*{bsize * ILP + 1}+t3*{ILP}+k] = tmp[k];\n                    \n                }}\n            }}\n            __syncthreads();\n            int t3_ = threadIdx.x % {asize};\n            int t1_ = threadIdx.x / {asize};\n            _b3 = b3 * {bsize * ILP} + t1_*{ILP};\n            if (_b3 < s3) {{\n                #pragma unroll\n                for (int j=0; j<{ILP}; j++) {{\n                    #pragma unroll\n                    for (int k=0; k<{ILP}; k++) {{\n                        tmp[k] =\n                            t[(t3*{ILP}+k)*{bsize * ILP + 1}+t1_*{ILP}+j];\n                    }}\n                    vload<sizeof(in0_type)*{ILP}>(\n                        &y[b0*y0+b2*y1+(_b3+j)*y2+((t3*{ILP})+i*{asize * ILP})*y3],\n                        tmp\n                    );\n                }}\n            }}\n            __syncthreads();\n        }}\n    }}\n    int s0, s1, s2, s3;\n    in0->shape.unpack(s0, s1, s2, s3);\n    kernel<<<{{(s3-1)/{bsize * ILP}+1, s2, s0 }}, {bsize * asize}>>>\n        (in0_p, out0_p, s0, s1, s2, s3);\n    ')

def transpose0231_2(x):
    if False:
        print('Hello World!')
    (s0, s1, s2, s3) = x.shape
    asize = 16
    bsize = 8
    ILP = 2
    return jt.code([s0, s2, s3, s1], x.dtype, [x], cuda_header='#include <type/fp16_compute.h>\n#include <cassert>', cuda_src=f'\n    __global__ __launch_bounds__({asize * bsize}) void kernel(in0_type* __restrict__ x, in0_type* __restrict__ y, int s0, int s1, int s2, int s3) {{\n        __shared__ in0_type t[{asize * ILP}*{bsize * ILP + 1}];\n        int t3 = threadIdx.x % {bsize};\n        int t1 = threadIdx.x / {bsize};\n        int b3 = blockIdx.x;\n        int b1 = blockIdx.y;\n        int b2 = 0;\n        int b0 = blockIdx.z;\n        int x3 = 1;\n        int x2 = s3;\n        int x1 = s2*x2;\n        int x0 = s1*x1;\n        int y3 = 1;\n        int y2 = s1;\n        int y1 = s3*y2;\n        int y0 = s2*y1;\n        in0_type tmp[{ILP}];\n        {{\n            int _b3 = b3 * {bsize * ILP} + t3*{ILP};\n            if (_b3 < s3) {{\n                #pragma unroll\n                for (int j=0; j<{ILP}; j++) {{\n                    if (t1*{ILP}+j+b1*{asize * ILP} >= s1)\n                        continue;\n                    vload<sizeof(in0_type)*{ILP}>(\n                        tmp,\n                        &x[b0*x0+(t1*{ILP}+j+b1*{asize * ILP})*x1+b2*x2+_b3*x3]\n                    );\n                    #pragma unroll\n                    for (int k=0; k<{ILP}; k++)\n                        t[(t1*{ILP}+j)*{bsize * ILP + 1}+t3*{ILP}+k] = tmp[k];\n                    \n                }}\n            }}\n            __syncthreads();\n            int t3_ = threadIdx.x % {asize};\n            int t1_ = threadIdx.x / {asize};\n            _b3 = b3 * {bsize * ILP} + t1_*{ILP};\n            int yy3 = (t3_*{ILP})+b1*{asize * ILP};\n            if (_b3 < s3 && yy3 < s1) {{\n                #pragma unroll\n                for (int j=0; j<{ILP}; j++) {{\n                    #pragma unroll\n                    for (int k=0; k<{ILP}; k++) {{\n                        tmp[k] =\n                            t[(t3_*{ILP}+k)*{bsize * ILP + 1}+t1_*{ILP}+j];\n                    }}\n                    vload<sizeof(in0_type)*{ILP}>(\n                        &y[b0*y0+b2*y1+(_b3+j)*y2+yy3*y3],\n                        tmp\n                    );\n                    // printf("%d %d %d %d %d\\n", b0*y0+b2*y1+(_b3+j)*y2+yy3*y3,\n                    //    b0, b2, (_b3+j), yy3);\n                }}\n            }}\n            __syncthreads();\n        }}\n    }}\n    int s0, s1, s2, s3;\n    in0->shape.unpack(s0, s1, s2, s3);\n    kernel<<<{{(s3-1)/{bsize * ILP}+1, (s1-1)/{asize * ILP}+1, s0 }}, {bsize * asize}>>>\n        (in0_p, out0_p, s0, s1, s2, s3);\n    ')

def check_share():
    if False:
        i = 10
        return i + 15
    return
    a = jt.rand((30, 32, 4, 2000)).float32()
    jt.code(a.shape, a.dtype, [a], cuda_header='#include <type/fp16_compute.h>\n#include <cassert>', cuda_src='\n    __global__ void kernel(in0_type* __restrict__ a, in0_type* __restrict__ b) {\n        __shared__ float x[32*33];\n        for (int i=0; i<3; i++) {\n        ((float2*)&x[i])[0] = ((float2*)&a[i])[0];\n        ((float2*)&b[i])[0] = ((float2*)&x[i+1])[0];\n        }\n    }\n    kernel<<<1024,16*16>>>(in0_p, out0_p);\n    ').sync()
    jt.sync_all(True)
    print('pass test')

class TestFP16(unittest.TestCase):

    def test_array(self):
        if False:
            while True:
                i = 10
        a = np.array([1, 2, 3], dtype='float16')
        b = jt.array(a)
        np.testing.assert_allclose(a, b.data)

    def test_add(self):
        if False:
            while True:
                i = 10
        a = np.array([1, 2, 3], dtype='float16')
        b = jt.array(a)
        c = b + b
        np.testing.assert_allclose(c.data, a + a)
        d = c.sum()
        np.testing.assert_allclose(d.data, [12])
        c = c + 1
        print(c)

    def test_matmul(self):
        if False:
            print('Hello World!')
        a = jt.random((100, 100)).float16()
        b = jt.random((100, 100)).float16()
        c = jt.matmul(a, b)
        c.sync()

    def test_bmm(self):
        if False:
            return 10
        a = jt.random((10, 3, 4)).float16()
        b = jt.random((10, 4, 5)).float16()
        c = jt.matmul(a, b)
        c.sync()

    def test_matmul_grad(self):
        if False:
            while True:
                i = 10
        a = jt.random((100, 100)).float16()
        b = jt.random((100, 100)).float16()
        c = jt.matmul(a, b)
        c.sync()
        (da, db) = jt.grad(c, [a, b])
        jt.sync_all()
        assert da.dtype == 'float16'
        assert db.dtype == 'float16'

    def test_array_random_auto_cast(self):
        if False:
            while True:
                i = 10
        a = jt.array([1.0, 2.0])
        assert a.dtype == 'float32'
        with jt.flag_scope(amp_reg=2 + 16):
            a = jt.array([1.0, 2.0])
            assert a.dtype == 'float16', a.dtype
        a = jt.random([10])
        assert a.dtype == 'float32'
        with jt.flag_scope(amp_reg=2 + 16):
            a = jt.random([10])
            assert a.dtype == 'float16', a.dtype

    def test_conv(self):
        if False:
            print('Hello World!')
        a = jt.random((3, 4, 5, 5)).float16()
        b = jt.random((4, 4, 3, 3)).float16()
        c = jt.nn.conv(a, b)
        c.sync()

    def test_max(self):
        if False:
            while True:
                i = 10
        a = jt.random((100,)).float16()
        b = jt.random((100,)).float16()
        c = a.maximum(b)
        c.sync()

    def test_reduce_dtype_infer(self):
        if False:
            return 10
        with jt.flag_scope(amp_reg=1):
            a = jt.random((3, 4, 5, 5)).float16()
            b = a.sum()
            b.sync()
            assert b.dtype == 'float32'
        with jt.flag_scope(amp_reg=2):
            a = jt.random((3, 4, 5, 5)).float16()
            b = a.sum()
            b.sync()
            assert b.dtype == 'float32'
        with jt.flag_scope(amp_reg=0):
            a = jt.random((3, 4, 5, 5)).float16()
            b = a.sum()
            b.sync()
            assert b.dtype == 'float32'
        with jt.flag_scope(amp_reg=2 + 4):
            a = jt.random((3, 4, 5, 5)).float16()
            b = a.sum()
            b.sync()
            assert b.dtype == 'float16', b.dtype

    def test_white_dtype_infer(self):
        if False:
            i = 10
            return i + 15
        with jt.flag_scope(amp_reg=1):
            a = jt.random((3, 4, 5, 5)).float16()
            b = a ** a
            b.sync()
            assert b.dtype == 'float32'
        with jt.flag_scope(amp_reg=2):
            a = jt.random((3, 4, 5, 5)).float16()
            b = a ** a
            b.sync()
            assert b.dtype == 'float32'
        with jt.flag_scope(amp_reg=0):
            a = jt.random((3, 4, 5, 5)).float16()
            b = a ** a
            b.sync()
            assert b.dtype == 'float32'
        with jt.flag_scope(amp_reg=2 + 8):
            a = jt.random((3, 4, 5, 5)).float16()
            b = a ** a
            b.sync()
            assert b.dtype == 'float16', b.dtype

    def test_module_half(self):
        if False:
            return 10
        a = jt.nn.Linear(10, 10)
        assert a.weight.dtype == 'float32'
        a.half()
        assert a.weight.dtype == 'float16'

    def test_scalar(self):
        if False:
            return 10
        a = jt.float16([1, 2, 3])
        assert (a * 1).dtype == 'float16'
        assert (a * jt.float16([1, 2, 3])).dtype == 'float16'
        assert (a * jt.float32([1, 2, 3])).dtype == 'float32'
        assert (a * jt.float32([1, 2, 3]).sum()).dtype == 'float16'
        assert jt.int([0, 1, 0]).ternary(a, jt.float32(1)).dtype == 'float16'

    def test_amp_level3(self):
        if False:
            print('Hello World!')
        with jt.flag_scope(amp_level=3):
            a = jt.float16([1, 2, 3])
            assert a.sum().dtype == 'float16'
            assert a.mean().dtype == 'float16'
            assert a.log().dtype == 'float16'
            assert a.exp().dtype == 'float16'

    def test_safe_clip(self):
        if False:
            print('Hello World!')
        import math
        assert not jt.float16(math.inf).isfinite()
        assert jt.safe_clip(jt.float16(math.inf)).isfinite()

@unittest.skipIf(not jt.compiler.has_cuda, 'No CUDA found')
class TestFP16CUDA(TestFP16):

    def setUp(self):
        if False:
            while True:
                i = 10
        jt.flags.use_cuda = 1

    def tearDown(self):
        if False:
            while True:
                i = 10
        jt.flags.use_cuda = 0

    def test_softmax(self):
        if False:
            i = 10
            return i + 15
        a = jt.rand((120, 2000, 2000)).float16()
        jt.sync_all()
        with jt.profile_scope(10, 100):
            a.log_softmax(-1).sync()

    def test_transpose(self):
        if False:
            while True:
                i = 10
        check_share()
        a = jt.rand((30, 32, 4, 2000)).float32()
        diff = transpose0231(a).data != a.transpose((0, 2, 3, 1)).data
        print(np.where(diff))
        jt.sync_all()
        with jt.profile_scope(100, 11000):
            transpose0231(a).sync()
            a.transpose((0, 2, 3, 1)).sync()
            a.fuse_transpose((0, 2, 1, 3)).sync()
            (a + 1).sync()
        jt.sync_all(True)
        diff = transpose0231(a).data != a.transpose((0, 2, 3, 1)).data
        print(np.where(diff))
        np.testing.assert_allclose(transpose0231(a).data, a.transpose((0, 2, 3, 1)).data)

    def test_transpose2(self):
        if False:
            return 10
        a = jt.rand((1, 10000, 1, 2048)).float32()
        print('transpose')
        transpose0231_2(a).sync()
        print('add')
        (a + 1).sync()
        return
        diff = transpose0231_2(a).data != a.transpose((0, 2, 3, 1)).data
        print(np.where(diff))
        jt.sync_all()
        with jt.profile_scope(100, 1100):
            transpose0231_2(a).sync()
            a.transpose((0, 2, 3, 1)).sync()
            a.fuse_transpose((0, 2, 1, 3)).sync()
            (a + 1).sync()
        jt.sync_all(True)
        diff = transpose0231_2(a).data != a.transpose((0, 2, 3, 1)).data
        print(np.where(diff))
        np.testing.assert_allclose(transpose0231_2(a).data, a.transpose((0, 2, 3, 1)).data)
if __name__ == '__main__':
    unittest.main()