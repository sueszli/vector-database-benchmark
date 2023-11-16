import unittest
import jittor as jt
import numpy as np
from jittor import Function

class TestCodeOp(unittest.TestCase):

    def test(self):
        if False:
            while True:
                i = 10
        a = jt.random([10])
        b = jt.code(a.shape, a.dtype, [a], cpu_src='\n                for (int i=0; i<in0_shape0; i++)\n                    @out(i) = @in0(i)*@in0(i)*2;\n            ', cpu_grad_src=['\n                for (int i=0; i<in0_shape0; i++) {\n                    @out(i) = @dout(i)*@in0(i)*4;\n                }\n            '])
        (na, nb) = jt.fetch_sync([a, b])
        assert np.allclose(na * na * 2, nb)
        c = jt.random([10])
        da = jt.grad(c * b, a)
        assert np.allclose(c.data * na * 4, da.data), (c.data * na * 4, da.data)

    def test_exflags(self):
        if False:
            while True:
                i = 10
        a = jt.random([10])
        b = jt.code(a.shape, a.dtype, [a], cpu_src='\n                LOGir << HAHAHA;\n                @out0(0) = HAHAHA;\n            ')
        b.compile_options = {'FLAGS: -DHAHAHA=233 -I/any/include/path ': 1}
        assert b[0].item() == 233

    def test_global_var(self):
        if False:
            while True:
                i = 10
        header = '\n        namespace jittor {\n            extern int a_global_int_var;\n        }\n        '
        src = '\n        namespace jittor {\n            int a_global_int_var = 123;\n        }\n        '
        with jt.flag_scope(compile_options={'FLAGS:-DGLOBAL_VAR': 1}):
            jt.code([1], 'int', [], cpu_header=header + src, cpu_src=' ').sync()
        assert jt.code([1], 'int', [], cpu_header=header, cpu_src='out0_p[0] = ++a_global_int_var; ').item() == 124
        assert jt.code([1], 'int', [], cpu_header=header, cpu_src='out0_p[0] = ++a_global_int_var; ').item() == 125

    def test_ten_args(self):
        if False:
            i = 10
            return i + 15
        a = jt.random([10])
        b = jt.code([a.shape] * 11, [a.dtype] * 11, [jt.random([10])] * 10 + [a], cpu_src='\n                for (int i=0; i<in10_shape0; i++)\n                    @out10(i) = @in10(i)*@in10(i)*2;\n            ', cpu_grad_src=[''] * 10 + ['\n                for (int i=0; i<in10_shape0; i++) {\n                    @out0(i) = @dout(i)*@in10(i)*4;\n                }\n            '])[-1]
        (na, nb) = jt.fetch_sync([a, b])
        assert np.allclose(na * na * 2, nb)
        c = jt.random([10])
        da = jt.grad(c * b, a)
        assert np.allclose(c.data * na * 4, da.data), (c.data * na * 4, da.data)

    def test_use_func(self):
        if False:
            while True:
                i = 10

        class Func(Function):

            def execute(self, x):
                if False:
                    i = 10
                    return i + 15
                self.save_vars = x
                return jt.code(x.shape, x.dtype, [x], cpu_src='\n                        for (int i=0; i<in0_shape0; i++)\n                            @out(i) = @in0(i)*@in0(i)*2;\n                    ')

            def grad(self, grad_x):
                if False:
                    while True:
                        i = 10
                x = self.save_vars
                return jt.code(x.shape, x.dtype, [x, grad_x], cpu_src='\n                        for (int i=0; i<in0_shape0; i++)\n                            @out(i) = @in1(i)*@in0(i)*4;\n                    ')
        a = jt.random([10])
        func = Func()
        b = func(a)
        (na, nb) = jt.fetch_sync([a, b])
        assert np.allclose(na * na * 2, nb)
        c = jt.random([10])
        da = jt.grad(c * b, a)
        assert np.allclose(c.data * na * 4, da.data), (c.data * na * 4, da.data)

    def test_multi_input(self):
        if False:
            for i in range(10):
                print('nop')
        a = jt.random([10])
        b = jt.random([10])
        c = jt.code(a.shape, a.dtype, [a, b], cpu_src='\n                for (int i=0; i<in0_shape0; i++)\n                    @out(i) = @in0(i)*@in1(i);\n            ', cpu_grad_src=['\n                for (int i=0; i<in0_shape0; i++) {\n                    @out(i) = @dout(i)*@in1(i);\n                }\n            ', '\n                for (int i=0; i<in0_shape0; i++) {\n                    @out(i) = @dout(i)*@in0(i);\n                }\n            '])
        (da, db) = jt.grad(c, [a, b])
        assert np.allclose(c.data, a.data * b.data), (c.data, a.data * b.data)
        assert np.allclose(da.data, b.data)
        assert np.allclose(db.data, a.data)

    def test_header(self):
        if False:
            i = 10
            return i + 15
        a = jt.array([3, 2, 1])
        b = jt.code(a.shape, a.dtype, [a], cpu_header='\n                #include <algorithm>\n                @alias(a, in0)\n                @alias(b, out)\n            ', cpu_src='\n                for (int i=0; i<a_shape0; i++)\n                    @b(i) = @a(i);\n                std::sort(&@b(0), &@b(in0_shape0));\n            ')
        assert (b.data == [1, 2, 3]).all()

    def test_multi_output(self):
        if False:
            print('Hello World!')
        a = jt.array([3, 2, 1])
        (b, c) = jt.code([[2], [4]], ['float32', 'float64'], [a], cpu_src='\n                @alias(a, in0)\n                @alias(b, out0)\n                @alias(c, out1)\n                for (int i=0; i<a_shape0; i++) {\n                    if (i<b_shape0) @b(i) = @a(i);\n                    if (i<c_shape0) @c(i) = @a(i);\n                }\n            ')
        assert b.shape == [2]
        assert c.shape == [4]
        assert b.dtype == 'float32'
        assert c.dtype == 'float64'
        assert (b.data == [3, 2]).all()
        assert (c.data[:3] == [3, 2, 1]).all()

    def test_return_multi_output(self):
        if False:
            while True:
                i = 10
        a = jt.array([3, 2, 1])
        b = jt.array([1, 2])
        c = jt.array([3, 4, 5, 6])
        jt.code([a], [b, c], cpu_src='\n                @alias(a, in0)\n                @alias(b, out0)\n                @alias(c, out1)\n                for (int i=0; i<a_shape0; i++) {\n                    if (i<b_shape0) @b(i) += @a(i);\n                    if (i<c_shape0) @c(i) += @a(i);\n                }\n            ')
        assert b.shape == [2]
        assert c.shape == [4]
        assert (b.data == [4, 4]).all()
        assert (c.data[:3] == [6, 6, 6]).all()

    def test_multi_output2(self):
        if False:
            for i in range(10):
                print('nop')
        a = jt.array([3, 2, 1])
        (b, c) = jt.code([(1,), (1,)], [a.dtype, a.dtype], [a], cpu_header='\n                #include <iostream>\n                using namespace std;\n            ', cpu_src='\n                @alias(a, in0)\n                @alias(b, out0)\n                @alias(c, out1)\n                @b(0) = @c(0) = @a(0);\n                for (int i=0; i<a_shape0; i++) {\n                    @b(0) = std::min(@b(0), @a(i));\n                    @c(0) = std::max(@c(0), @a(i));\n                }\n                cout << "min:" << @b(0) << " max:" << @c(0) << endl;\n            ')
        assert b.data == 1, b
        assert c.data == 3, c

    def test_vary_shape(self):
        if False:
            i = 10
            return i + 15
        a = jt.array([5, -4, 3, -2, 1])
        (b, c) = jt.code([(-5,), (-5,)], [a.dtype, a.dtype], [a], cpu_src='\n                @alias(a, in0)\n                @alias(b, out0)\n                @alias(c, out1)\n                int num_b=0, num_c=0;\n                for (int i=0; i<a_shape0; i++) {\n                    if (@a(i)>0)\n                        @b(num_b++) = @a(i);\n                    else\n                        @c(num_c++) = @a(i);\n                }\n                b->set_shape({num_b});\n                c->set_shape({num_c});\n            ')
        assert (b.data == [5, 3, 1]).all()
        assert (c.data == [-4, -2]).all()

    def test_comment(self):
        if False:
            return 10
        a = jt.array([3, 2, 1])
        b = jt.code(a.shape, a.dtype, [a], cpu_header='\n            #include <algorithm>\n            // asd\n            /* asd\n            */\n            ', cpu_src='\n                // test comment\n                /*\n                multi line\n                */\n                @alias(a, in0)\n                for (int i=0; i<a_shape0; i++)\n                    @out(i) = @a(i);\n                std::sort(&@out(0), &@out(a_shape0));\n            ')
        assert (b.data == [1, 2, 3]).all()

    @unittest.skipIf(not jt.compiler.has_cuda, 'No CUDA found')
    @jt.flag_scope(use_cuda=1)
    def test_cuda(self):
        if False:
            return 10
        a = jt.random([100000])
        b = jt.random([100000])
        c = jt.code(a.shape, a.dtype, [a, b], cuda_src='\n            __global__ static void kernel1(@ARGS_DEF) {\n                @PRECALC\n                int i = threadIdx.x + blockIdx.x * blockDim.x;\n                int stride = blockDim.x * gridDim.x;\n                for (; i<in0_shape0; i+=stride)\n                    @out(i) = @in0(i)*@in1(i);\n            }\n                kernel1<<<(in0_shape0-1)/1024+1, 1024>>>(@ARGS);\n            ', cuda_grad_src=['\n            __global__ static void kernel2(@ARGS_DEF) {\n                @PRECALC\n                int i = threadIdx.x + blockIdx.x * blockDim.x;\n                int stride = blockDim.x * gridDim.x;\n                for (; i<in0_shape0; i+=stride)\n                    @out(i) = @dout(i)*@in1(i);\n            }\n                kernel2<<<(in0_shape0-1)/1024+1, 1024>>>(@ARGS);\n            ', '\n            __global__ static void kernel3(@ARGS_DEF) {\n                @PRECALC\n                int i = threadIdx.x + blockIdx.x * blockDim.x;\n                int stride = blockDim.x * gridDim.x;\n                for (; i<in0_shape0; i+=stride)\n                    @out(i) = @dout(i)*@in0(i);\n            }\n                kernel3<<<(in0_shape0-1)/1024+1, 1024>>>(@ARGS);\n            '])
        (da, db) = jt.grad(c, [a, b])
        assert np.allclose(c.data, a.data * b.data), (c.data, a.data * b.data)
        assert np.allclose(da.data, b.data)
        assert np.allclose(db.data, a.data)

    @unittest.skipIf(not jt.compiler.has_cuda, 'No CUDA found')
    @jt.flag_scope(use_cuda=1)
    def test_cuda2(self):
        if False:
            while True:
                i = 10
        a = jt.random((100, 100))
        b = jt.random((100, 100))
        c = jt.code(a.shape, a.dtype, [a, b], cuda_src='\n                __global__ static void kernel1(@ARGS_DEF) {\n                    @PRECALC\n                    for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)\n                    for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x)\n                        @out(i,j) = @in0(i,j)*@in1(i,j);\n                }\n                kernel1<<<32, 32>>>(@ARGS);\n            ', cuda_grad_src=['\n                __global__ static void kernel(@ARGS_DEF) {\n                    @PRECALC\n                    for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)\n                    for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x)\n                        @out(i,j) = @dout(i,j)*@in1(i,j);\n                }\n                kernel<<<32, 32>>>(@ARGS);\n            ', '\n                __global__ static void kernel(@ARGS_DEF) {\n                    @PRECALC\n                    @pout(0,0);\n                    for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)\n                    for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x)\n                        @out(i,j) = @dout(i,j)*@in0(i,j);\n                }\n                kernel<<<32, 32>>>(@ARGS);\n            '])
        (da, db) = jt.grad(c, [a, b])
        assert np.allclose(c.data, a.data * b.data), (c.data, a.data * b.data)
        assert np.allclose(da.data, b.data)
        assert np.allclose(db.data, a.data)

    @unittest.skipIf(not jt.compiler.has_cuda, 'No CUDA found')
    @jt.flag_scope(use_cuda=1)
    def test_cuda2_use_func(self):
        if False:
            i = 10
            return i + 15

        class Func(Function):

            def execute(self, a, b):
                if False:
                    i = 10
                    return i + 15
                self.save_vars = (a, b)
                return jt.code(a.shape, a.dtype, [a, b], cuda_src='\n                        __global__ static void kernel1(@ARGS_DEF) {\n                            @PRECALC\n                            for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)\n                            for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x)\n                                @out(i,j) = @in0(i,j)*@in1(i,j);\n                        }\n                        kernel1<<<32, 32>>>(@ARGS);\n                    ')

            def grad(self, grad):
                if False:
                    i = 10
                    return i + 15
                (a, b) = self.save_vars
                return jt.code([a.shape, b.shape], [a.dtype, b.dtype], [a, b, grad], cuda_src='\n                        __global__ static void kernel2(@ARGS_DEF) {\n                            @PRECALC\n                            for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)\n                            for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x) {\n                                @out0(i,j) = @in2(i,j)*@in1(i,j);\n                                @out1(i,j) = @in2(i,j)*@in0(i,j);\n                            }\n                        }\n                        kernel2<<<32, 32>>>(@ARGS);\n                    ')
        a = jt.random((100, 100))
        b = jt.random((100, 100))
        func = Func()
        c = func(a, b)
        (da, db) = jt.grad(c, [a, b])
        assert np.allclose(c.data, a.data * b.data), (c.data, a.data * b.data)
        assert np.allclose(da.data, b.data)
        assert np.allclose(db.data, a.data)

    def test_simple_var(self):
        if False:
            while True:
                i = 10
        a = jt.code([1], 'float32', inputs=[], data={'x': 123}, cpu_src='\n                @out0(0) = data["x"];\n            ').sync()
        assert a.item() == 123
if __name__ == '__main__':
    unittest.main()