import sys
import numpy
import cupy
code = '\ntemplate<typename T>\nstruct Matrix {\n    T value[4][4];\n\n    __device__ T& operator() (int i, int j) {\n        return this->value[i][j];\n    }\n\n    __device__ const T& operator() (int i, int j) const {\n        return this->value[i][j];\n    }\n};\n\ntemplate<typename T>\n__device__ Matrix<T> operator+ (const Matrix<T>& lhs, const Matrix<T>& rhs) {\n    Matrix<T> res;\n    for (int i = 0; i<4; i++) {\n        for (int j = 0; j<4; j++) {\n            res(i,j) = lhs(i,j) + rhs(i,j);\n        }\n    }\n    return res;\n}\n\ntemplate<typename T>\n__device__ Matrix<T> operator* (const Matrix<T>& lhs, const Matrix<T>& rhs) {\n    Matrix<T> res;\n    for (int i = 0; i<4; i++) {\n        for (int j = 0; j<4; j++) {\n            res(i,j) = T(0);\n            for (int k = 0; k<4; k++) {\n                res(i,j) += lhs(i,k) * rhs(k,j);\n            }\n        }\n    }\n    return res;\n}\n\ntemplate<typename T>\n__global__ void kernel(const Matrix<T>* A,\n                       const Matrix<T>* B,\n                       const Matrix<T>  C,\n                             Matrix<T>* out) {\n  int i = threadIdx.x;\n  out[i] = A[i] * B[i] + C;\n}\n'

def main():
    if False:
        i = 10
        return i + 15
    N = 8
    module = cupy.RawModule(code=code, options=('-std=c++11',), name_expressions=('kernel<float>', 'kernel<double>'))
    for (ctype, dtype) in zip(('float', 'double'), (numpy.float32, numpy.float64)):
        A = cupy.random.rand(16 * N, dtype=dtype).reshape(N, 4, 4)
        B = cupy.random.rand(16 * N, dtype=dtype).reshape(N, 4, 4)
        C = numpy.random.rand(16).astype(dtype).reshape(4, 4)
        out = cupy.empty_like(A)
        Matrix = numpy.dtype({'names': ['value'], 'formats': [(dtype, (4, 4))]})
        kernel = module.get_function('kernel<{}>'.format(ctype))
        args = (A, B, C.ravel().view(Matrix), out)
        kernel((1,), (N,), args)
        expected = cupy.matmul(A, B) + cupy.asarray(C[None, :, :])
        cupy.testing.assert_array_almost_equal(expected, out)
        print("Kernel output matches expected value for type '{}'.".format(ctype))
if __name__ == '__main__':
    sys.exit(main())