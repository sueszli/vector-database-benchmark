import sys
import numpy
import cupy
code = '\n__device__ double3 operator+(const double3& lhs, const double3& rhs) {\n    return make_double3(lhs.x + rhs.x,\n                        lhs.y + rhs.y,\n                        lhs.z + rhs.z);\n}\n\nextern "C" __global__ void sum_kernel(const double3* lhs,\n                                            double3  rhs,\n                                            double3* out) {\n  int i = threadIdx.x;\n  out[i] = lhs[i] + rhs;\n}\n'
double3 = numpy.dtype({'names': ['x', 'y', 'z'], 'formats': [numpy.float64] * 3})

def main():
    if False:
        i = 10
        return i + 15
    N = 8
    lhs = cupy.random.rand(3 * N, dtype=numpy.float64).reshape(N, 3)
    rhs = numpy.random.rand(3).astype(numpy.float64)
    out = cupy.empty_like(lhs)
    kernel = cupy.RawKernel(code, 'sum_kernel')
    args = (lhs, rhs.view(double3), out)
    kernel((1,), (N,), args)
    expected = lhs + cupy.asarray(rhs[None, :])
    cupy.testing.assert_array_equal(expected, out)
    print('Kernel output matches expected value.')
if __name__ == '__main__':
    sys.exit(main())