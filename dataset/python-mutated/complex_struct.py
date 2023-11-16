import sys
import numpy
import cupy
struct_definition = '\nstruct complex_struct {\n    int4 a;\n    char b;\n    double c[2];\n    short1 d;\n    unsigned long long int e[3];\n};\n'
struct_layout_code = '\n{struct_definition}\n\nextern "C" __global__ void get_struct_layout(\n                                unsigned long long *itemsize,\n                                unsigned long long *sizes,\n                                unsigned long long *offsets) {{\n    const complex_struct* ptr = nullptr;\n\n    itemsize[0] = sizeof(complex_struct);\n\n    sizes[0] = sizeof(ptr->a);\n    sizes[1] = sizeof(ptr->b);\n    sizes[2] = sizeof(ptr->c);\n    sizes[3] = sizeof(ptr->d);\n    sizes[4] = sizeof(ptr->e);\n\n    offsets[0] = (unsigned long long)&ptr->a;\n    offsets[1] = (unsigned long long)&ptr->b;\n    offsets[2] = (unsigned long long)&ptr->c;\n    offsets[3] = (unsigned long long)&ptr->d;\n    offsets[4] = (unsigned long long)&ptr->e;\n}}\n'.format(struct_definition=struct_definition)
kernel_code = '\n{struct_definition}\n\nextern "C" __global__ void test_kernel(const complex_struct s,\n                                       double* out) {{\n    int i = threadIdx.x;\n    double sum = 0.0;\n    sum += s.a.x + s.a.y + s.a.z + s.a.w;\n    sum += s.b;\n    sum += s.c[0] + s.c[1];\n    sum += s.d.x;\n    sum += s.e[0] + s.e[1] + s.e[2];\n    out[i] = i * sum;\n}}\n'.format(struct_definition=struct_definition)

def make_packed(basetype, N, itemsize):
    if False:
        i = 10
        return i + 15
    assert 0 < N <= 4, N
    names = list('xyzw')[:N]
    formats = [basetype] * N
    return numpy.dtype(dict(names=names, formats=formats, itemsize=itemsize))

def main():
    if False:
        i = 10
        return i + 15
    itemsize = cupy.ndarray(shape=(1,), dtype=numpy.uint64)
    sizes = cupy.ndarray(shape=(5,), dtype=numpy.uint64)
    offsets = cupy.ndarray(shape=(5,), dtype=numpy.uint64)
    kernel = cupy.RawKernel(struct_layout_code, 'get_struct_layout', options=('--std=c++11',))
    kernel((1,), (1,), (itemsize, sizes, offsets))
    (itemsize, sizes, offsets) = map(cupy.asnumpy, (itemsize, sizes, offsets))
    print('Overall structure itemsize: {} bytes'.format(itemsize.item()))
    print('Structure members itemsize: {}'.format(sizes))
    print('Structure members offsets: {}'.format(offsets))
    atype = make_packed(numpy.int32, 4, sizes[0])
    btype = make_packed(numpy.int8, 1, sizes[1])
    ctype = make_packed(numpy.float64, 2, sizes[2])
    dtype = make_packed(numpy.int16, 1, sizes[3])
    etype = make_packed(numpy.uint64, 3, sizes[4])
    names = list('abcde')
    formats = [atype, btype, ctype, dtype, etype]
    complex_struct = numpy.dtype(dict(names=names, formats=formats, offsets=offsets, itemsize=itemsize.item()))
    s = numpy.empty(shape=(1,), dtype=complex_struct)
    s['a'] = numpy.arange(0, 4).astype(numpy.int32).view(atype)
    s['b'] = numpy.arange(4, 5).astype(numpy.int8).view(btype)
    s['c'] = numpy.arange(5, 7).astype(numpy.float64).view(ctype)
    s['d'] = numpy.arange(7, 8).astype(numpy.int16).view(dtype)
    s['e'] = numpy.arange(8, 11).astype(numpy.uint64).view(etype)
    print('Complex structure value:\n  {}'.format(s))
    N = 8
    out = cupy.empty(shape=(N,), dtype=numpy.float64)
    kernel = cupy.RawKernel(kernel_code, 'test_kernel')
    kernel((1,), (N,), (s, out))
    expected = cupy.arange(N) * 55.0
    cupy.testing.assert_array_almost_equal(expected, out)
    print('Kernel output matches expected value.')
if __name__ == '__main__':
    sys.exit(main())