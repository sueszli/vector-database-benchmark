"""
Code generator script to make the Cython BLAS and LAPACK wrappers
from the files "cython_blas_signatures.txt" and
"cython_lapack_signatures.txt" which contain the signatures for
all the BLAS/LAPACK routines that should be included in the wrappers.
"""
from collections import defaultdict
from operator import itemgetter
import os
from stat import ST_MTIME
import argparse
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
fortran_types = {'int': 'integer', 'c': 'complex', 'd': 'double precision', 's': 'real', 'z': 'complex*16', 'char': 'character', 'bint': 'logical'}
c_types = {'int': 'int', 'c': 'npy_complex64', 'd': 'double', 's': 'float', 'z': 'npy_complex128', 'char': 'char', 'bint': 'int', 'cselect1': '_cselect1', 'cselect2': '_cselect2', 'dselect2': '_dselect2', 'dselect3': '_dselect3', 'sselect2': '_sselect2', 'sselect3': '_sselect3', 'zselect1': '_zselect1', 'zselect2': '_zselect2'}

def arg_names_and_types(args):
    if False:
        print('Hello World!')
    return zip(*[arg.split(' *') for arg in args.split(', ')])
pyx_func_template = '\ncdef extern from "{header_name}":\n    void _fortran_{name} "F_FUNC({name}wrp, {upname}WRP)"({ret_type} *out, {fort_args}) nogil\ncdef {ret_type} {name}({args}) noexcept nogil:\n    cdef {ret_type} out\n    _fortran_{name}(&out, {argnames})\n    return out\n'
npy_types = {'c': 'npy_complex64', 'z': 'npy_complex128', 'cselect1': '_cselect1', 'cselect2': '_cselect2', 'dselect2': '_dselect2', 'dselect3': '_dselect3', 'sselect2': '_sselect2', 'sselect3': '_sselect3', 'zselect1': '_zselect1', 'zselect2': '_zselect2'}

def arg_casts(arg):
    if False:
        return 10
    if arg in ['npy_complex64', 'npy_complex128', '_cselect1', '_cselect2', '_dselect2', '_dselect3', '_sselect2', '_sselect3', '_zselect1', '_zselect2']:
        return f'<{arg}*>'
    return ''

def pyx_decl_func(name, ret_type, args, header_name):
    if False:
        i = 10
        return i + 15
    (argtypes, argnames) = arg_names_and_types(args)
    if ret_type in argnames:
        argnames = [n if n != ret_type else ret_type + '_' for n in argnames]
        argnames = [n if n not in ['lambda', 'in'] else n + '_' for n in argnames]
        args = ', '.join([' *'.join([n, t]) for (n, t) in zip(argtypes, argnames)])
    argtypes = [npy_types.get(t, t) for t in argtypes]
    fort_args = ', '.join([' *'.join([n, t]) for (n, t) in zip(argtypes, argnames)])
    argnames = [arg_casts(t) + n for (n, t) in zip(argnames, argtypes)]
    argnames = ', '.join(argnames)
    c_ret_type = c_types[ret_type]
    args = args.replace('lambda', 'lambda_')
    return pyx_func_template.format(name=name, upname=name.upper(), args=args, fort_args=fort_args, ret_type=ret_type, c_ret_type=c_ret_type, argnames=argnames, header_name=header_name)
pyx_sub_template = 'cdef extern from "{header_name}":\n    void _fortran_{name} "F_FUNC({name},{upname})"({fort_args}) nogil\ncdef void {name}({args}) noexcept nogil:\n    _fortran_{name}({argnames})\n'

def pyx_decl_sub(name, args, header_name):
    if False:
        while True:
            i = 10
    (argtypes, argnames) = arg_names_and_types(args)
    argtypes = [npy_types.get(t, t) for t in argtypes]
    argnames = [n if n not in ['lambda', 'in'] else n + '_' for n in argnames]
    fort_args = ', '.join([' *'.join([n, t]) for (n, t) in zip(argtypes, argnames)])
    argnames = [arg_casts(t) + n for (n, t) in zip(argnames, argtypes)]
    argnames = ', '.join(argnames)
    args = args.replace('*lambda,', '*lambda_,').replace('*in,', '*in_,')
    return pyx_sub_template.format(name=name, upname=name.upper(), args=args, fort_args=fort_args, argnames=argnames, header_name=header_name)
blas_pyx_preamble = '# cython: boundscheck = False\n# cython: wraparound = False\n# cython: cdivision = True\n\n"""\nBLAS Functions for Cython\n=========================\n\nUsable from Cython via::\n\n    cimport scipy.linalg.cython_blas\n\nThese wrappers do not check for alignment of arrays.\nAlignment should be checked before these wrappers are used.\n\nRaw function pointers (Fortran-style pointer arguments):\n\n- {}\n\n\n"""\n\n# Within SciPy, these wrappers can be used via relative or absolute cimport.\n# Examples:\n# from ..linalg cimport cython_blas\n# from scipy.linalg cimport cython_blas\n# cimport scipy.linalg.cython_blas as cython_blas\n# cimport ..linalg.cython_blas as cython_blas\n\n# Within SciPy, if BLAS functions are needed in C/C++/Fortran,\n# these wrappers should not be used.\n# The original libraries should be linked directly.\n\ncdef extern from "fortran_defs.h":\n    pass\n\nfrom numpy cimport npy_complex64, npy_complex128\n\n'

def make_blas_pyx_preamble(all_sigs):
    if False:
        i = 10
        return i + 15
    names = [sig[0] for sig in all_sigs]
    return blas_pyx_preamble.format('\n- '.join(names))
lapack_pyx_preamble = '"""\nLAPACK functions for Cython\n===========================\n\nUsable from Cython via::\n\n    cimport scipy.linalg.cython_lapack\n\nThis module provides Cython-level wrappers for all primary routines included\nin LAPACK 3.4.0 except for ``zcgesv`` since its interface is not consistent\nfrom LAPACK 3.4.0 to 3.6.0. It also provides some of the\nfixed-api auxiliary routines.\n\nThese wrappers do not check for alignment of arrays.\nAlignment should be checked before these wrappers are used.\n\nRaw function pointers (Fortran-style pointer arguments):\n\n- {}\n\n\n"""\n\n# Within SciPy, these wrappers can be used via relative or absolute cimport.\n# Examples:\n# from ..linalg cimport cython_lapack\n# from scipy.linalg cimport cython_lapack\n# cimport scipy.linalg.cython_lapack as cython_lapack\n# cimport ..linalg.cython_lapack as cython_lapack\n\n# Within SciPy, if LAPACK functions are needed in C/C++/Fortran,\n# these wrappers should not be used.\n# The original libraries should be linked directly.\n\ncdef extern from "fortran_defs.h":\n    pass\n\nfrom numpy cimport npy_complex64, npy_complex128\n\ncdef extern from "_lapack_subroutines.h":\n    # Function pointer type declarations for\n    # gees and gges families of functions.\n    ctypedef bint _cselect1(npy_complex64*)\n    ctypedef bint _cselect2(npy_complex64*, npy_complex64*)\n    ctypedef bint _dselect2(d*, d*)\n    ctypedef bint _dselect3(d*, d*, d*)\n    ctypedef bint _sselect2(s*, s*)\n    ctypedef bint _sselect3(s*, s*, s*)\n    ctypedef bint _zselect1(npy_complex128*)\n    ctypedef bint _zselect2(npy_complex128*, npy_complex128*)\n\n'

def make_lapack_pyx_preamble(all_sigs):
    if False:
        return 10
    names = [sig[0] for sig in all_sigs]
    return lapack_pyx_preamble.format('\n- '.join(names))
blas_py_wrappers = '\n\n# Python-accessible wrappers for testing:\n\ncdef inline bint _is_contiguous(double[:,:] a, int axis) noexcept nogil:\n    return (a.strides[axis] == sizeof(a[0,0]) or a.shape[axis] == 1)\n\ncpdef float complex _test_cdotc(float complex[:] cx, float complex[:] cy) noexcept nogil:\n    cdef:\n        int n = cx.shape[0]\n        int incx = cx.strides[0] // sizeof(cx[0])\n        int incy = cy.strides[0] // sizeof(cy[0])\n    return cdotc(&n, &cx[0], &incx, &cy[0], &incy)\n\ncpdef float complex _test_cdotu(float complex[:] cx, float complex[:] cy) noexcept nogil:\n    cdef:\n        int n = cx.shape[0]\n        int incx = cx.strides[0] // sizeof(cx[0])\n        int incy = cy.strides[0] // sizeof(cy[0])\n    return cdotu(&n, &cx[0], &incx, &cy[0], &incy)\n\ncpdef double _test_dasum(double[:] dx) noexcept nogil:\n    cdef:\n        int n = dx.shape[0]\n        int incx = dx.strides[0] // sizeof(dx[0])\n    return dasum(&n, &dx[0], &incx)\n\ncpdef double _test_ddot(double[:] dx, double[:] dy) noexcept nogil:\n    cdef:\n        int n = dx.shape[0]\n        int incx = dx.strides[0] // sizeof(dx[0])\n        int incy = dy.strides[0] // sizeof(dy[0])\n    return ddot(&n, &dx[0], &incx, &dy[0], &incy)\n\ncpdef int _test_dgemm(double alpha, double[:,:] a, double[:,:] b, double beta,\n                double[:,:] c) except -1 nogil:\n    cdef:\n        char *transa\n        char *transb\n        int m, n, k, lda, ldb, ldc\n        double *a0=&a[0,0]\n        double *b0=&b[0,0]\n        double *c0=&c[0,0]\n    # In the case that c is C contiguous, swap a and b and\n    # swap whether or not each of them is transposed.\n    # This can be done because a.dot(b) = b.T.dot(a.T).T.\n    if _is_contiguous(c, 1):\n        if _is_contiguous(a, 1):\n            transb = \'n\'\n            ldb = (&a[1,0]) - a0 if a.shape[0] > 1 else 1\n        elif _is_contiguous(a, 0):\n            transb = \'t\'\n            ldb = (&a[0,1]) - a0 if a.shape[1] > 1 else 1\n        else:\n            with gil:\n                raise ValueError("Input \'a\' is neither C nor Fortran contiguous.")\n        if _is_contiguous(b, 1):\n            transa = \'n\'\n            lda = (&b[1,0]) - b0 if b.shape[0] > 1 else 1\n        elif _is_contiguous(b, 0):\n            transa = \'t\'\n            lda = (&b[0,1]) - b0 if b.shape[1] > 1 else 1\n        else:\n            with gil:\n                raise ValueError("Input \'b\' is neither C nor Fortran contiguous.")\n        k = b.shape[0]\n        if k != a.shape[1]:\n            with gil:\n                raise ValueError("Shape mismatch in input arrays.")\n        m = b.shape[1]\n        n = a.shape[0]\n        if n != c.shape[0] or m != c.shape[1]:\n            with gil:\n                raise ValueError("Output array does not have the correct shape.")\n        ldc = (&c[1,0]) - c0 if c.shape[0] > 1 else 1\n        dgemm(transa, transb, &m, &n, &k, &alpha, b0, &lda, a0,\n                   &ldb, &beta, c0, &ldc)\n    elif _is_contiguous(c, 0):\n        if _is_contiguous(a, 1):\n            transa = \'t\'\n            lda = (&a[1,0]) - a0 if a.shape[0] > 1 else 1\n        elif _is_contiguous(a, 0):\n            transa = \'n\'\n            lda = (&a[0,1]) - a0 if a.shape[1] > 1 else 1\n        else:\n            with gil:\n                raise ValueError("Input \'a\' is neither C nor Fortran contiguous.")\n        if _is_contiguous(b, 1):\n            transb = \'t\'\n            ldb = (&b[1,0]) - b0 if b.shape[0] > 1 else 1\n        elif _is_contiguous(b, 0):\n            transb = \'n\'\n            ldb = (&b[0,1]) - b0 if b.shape[1] > 1 else 1\n        else:\n            with gil:\n                raise ValueError("Input \'b\' is neither C nor Fortran contiguous.")\n        m = a.shape[0]\n        k = a.shape[1]\n        if k != b.shape[0]:\n            with gil:\n                raise ValueError("Shape mismatch in input arrays.")\n        n = b.shape[1]\n        if m != c.shape[0] or n != c.shape[1]:\n            with gil:\n                raise ValueError("Output array does not have the correct shape.")\n        ldc = (&c[0,1]) - c0 if c.shape[1] > 1 else 1\n        dgemm(transa, transb, &m, &n, &k, &alpha, a0, &lda, b0,\n                   &ldb, &beta, c0, &ldc)\n    else:\n        with gil:\n            raise ValueError("Input \'c\' is neither C nor Fortran contiguous.")\n    return 0\n\ncpdef double _test_dnrm2(double[:] x) noexcept nogil:\n    cdef:\n        int n = x.shape[0]\n        int incx = x.strides[0] // sizeof(x[0])\n    return dnrm2(&n, &x[0], &incx)\n\ncpdef double _test_dzasum(double complex[:] zx) noexcept nogil:\n    cdef:\n        int n = zx.shape[0]\n        int incx = zx.strides[0] // sizeof(zx[0])\n    return dzasum(&n, &zx[0], &incx)\n\ncpdef double _test_dznrm2(double complex[:] x) noexcept nogil:\n    cdef:\n        int n = x.shape[0]\n        int incx = x.strides[0] // sizeof(x[0])\n    return dznrm2(&n, &x[0], &incx)\n\ncpdef int _test_icamax(float complex[:] cx) noexcept nogil:\n    cdef:\n        int n = cx.shape[0]\n        int incx = cx.strides[0] // sizeof(cx[0])\n    return icamax(&n, &cx[0], &incx)\n\ncpdef int _test_idamax(double[:] dx) noexcept nogil:\n    cdef:\n        int n = dx.shape[0]\n        int incx = dx.strides[0] // sizeof(dx[0])\n    return idamax(&n, &dx[0], &incx)\n\ncpdef int _test_isamax(float[:] sx) noexcept nogil:\n    cdef:\n        int n = sx.shape[0]\n        int incx = sx.strides[0] // sizeof(sx[0])\n    return isamax(&n, &sx[0], &incx)\n\ncpdef int _test_izamax(double complex[:] zx) noexcept nogil:\n    cdef:\n        int n = zx.shape[0]\n        int incx = zx.strides[0] // sizeof(zx[0])\n    return izamax(&n, &zx[0], &incx)\n\ncpdef float _test_sasum(float[:] sx) noexcept nogil:\n    cdef:\n        int n = sx.shape[0]\n        int incx = sx.strides[0] // sizeof(sx[0])\n    return sasum(&n, &sx[0], &incx)\n\ncpdef float _test_scasum(float complex[:] cx) noexcept nogil:\n    cdef:\n        int n = cx.shape[0]\n        int incx = cx.strides[0] // sizeof(cx[0])\n    return scasum(&n, &cx[0], &incx)\n\ncpdef float _test_scnrm2(float complex[:] x) noexcept nogil:\n    cdef:\n        int n = x.shape[0]\n        int incx = x.strides[0] // sizeof(x[0])\n    return scnrm2(&n, &x[0], &incx)\n\ncpdef float _test_sdot(float[:] sx, float[:] sy) noexcept nogil:\n    cdef:\n        int n = sx.shape[0]\n        int incx = sx.strides[0] // sizeof(sx[0])\n        int incy = sy.strides[0] // sizeof(sy[0])\n    return sdot(&n, &sx[0], &incx, &sy[0], &incy)\n\ncpdef float _test_snrm2(float[:] x) noexcept nogil:\n    cdef:\n        int n = x.shape[0]\n        int incx = x.strides[0] // sizeof(x[0])\n    return snrm2(&n, &x[0], &incx)\n\ncpdef double complex _test_zdotc(double complex[:] zx, double complex[:] zy) noexcept nogil:\n    cdef:\n        int n = zx.shape[0]\n        int incx = zx.strides[0] // sizeof(zx[0])\n        int incy = zy.strides[0] // sizeof(zy[0])\n    return zdotc(&n, &zx[0], &incx, &zy[0], &incy)\n\ncpdef double complex _test_zdotu(double complex[:] zx, double complex[:] zy) noexcept nogil:\n    cdef:\n        int n = zx.shape[0]\n        int incx = zx.strides[0] // sizeof(zx[0])\n        int incy = zy.strides[0] // sizeof(zy[0])\n    return zdotu(&n, &zx[0], &incx, &zy[0], &incy)\n'

def generate_blas_pyx(func_sigs, sub_sigs, all_sigs, header_name):
    if False:
        for i in range(10):
            print('nop')
    funcs = '\n'.join((pyx_decl_func(*s + (header_name,)) for s in func_sigs))
    subs = '\n' + '\n'.join((pyx_decl_sub(*s[::2] + (header_name,)) for s in sub_sigs))
    return make_blas_pyx_preamble(all_sigs) + funcs + subs + blas_py_wrappers
lapack_py_wrappers = '\n\n# Python accessible wrappers for testing:\n\ndef _test_dlamch(cmach):\n    # This conversion is necessary to handle Python 3 strings.\n    cmach_bytes = bytes(cmach)\n    # Now that it is a bytes representation, a non-temporary variable\n    # must be passed as a part of the function call.\n    cdef char* cmach_char = cmach_bytes\n    return dlamch(cmach_char)\n\ndef _test_slamch(cmach):\n    # This conversion is necessary to handle Python 3 strings.\n    cmach_bytes = bytes(cmach)\n    # Now that it is a bytes representation, a non-temporary variable\n    # must be passed as a part of the function call.\n    cdef char* cmach_char = cmach_bytes\n    return slamch(cmach_char)\n'

def generate_lapack_pyx(func_sigs, sub_sigs, all_sigs, header_name):
    if False:
        while True:
            i = 10
    funcs = '\n'.join((pyx_decl_func(*s + (header_name,)) for s in func_sigs))
    subs = '\n' + '\n'.join((pyx_decl_sub(*s[::2] + (header_name,)) for s in sub_sigs))
    preamble = make_lapack_pyx_preamble(all_sigs)
    return preamble + funcs + subs + lapack_py_wrappers
pxd_template = 'ctypedef {ret_type} {name}_t({args}) noexcept nogil\ncdef {name}_t *{name}_f\n'
pxd_template = 'cdef {ret_type} {name}({args}) noexcept nogil\n'

def pxd_decl(name, ret_type, args):
    if False:
        print('Hello World!')
    args = args.replace('lambda', 'lambda_').replace('*in,', '*in_,')
    return pxd_template.format(name=name, ret_type=ret_type, args=args)
blas_pxd_preamble = '# Within scipy, these wrappers can be used via relative or absolute cimport.\n# Examples:\n# from ..linalg cimport cython_blas\n# from scipy.linalg cimport cython_blas\n# cimport scipy.linalg.cython_blas as cython_blas\n# cimport ..linalg.cython_blas as cython_blas\n\n# Within SciPy, if BLAS functions are needed in C/C++/Fortran,\n# these wrappers should not be used.\n# The original libraries should be linked directly.\n\nctypedef float s\nctypedef double d\nctypedef float complex c\nctypedef double complex z\n\n'

def generate_blas_pxd(all_sigs):
    if False:
        for i in range(10):
            print('nop')
    body = '\n'.join((pxd_decl(*sig) for sig in all_sigs))
    return blas_pxd_preamble + body
lapack_pxd_preamble = '# Within SciPy, these wrappers can be used via relative or absolute cimport.\n# Examples:\n# from ..linalg cimport cython_lapack\n# from scipy.linalg cimport cython_lapack\n# cimport scipy.linalg.cython_lapack as cython_lapack\n# cimport ..linalg.cython_lapack as cython_lapack\n\n# Within SciPy, if LAPACK functions are needed in C/C++/Fortran,\n# these wrappers should not be used.\n# The original libraries should be linked directly.\n\nctypedef float s\nctypedef double d\nctypedef float complex c\nctypedef double complex z\n\n# Function pointer type declarations for\n# gees and gges families of functions.\nctypedef bint cselect1(c*)\nctypedef bint cselect2(c*, c*)\nctypedef bint dselect2(d*, d*)\nctypedef bint dselect3(d*, d*, d*)\nctypedef bint sselect2(s*, s*)\nctypedef bint sselect3(s*, s*, s*)\nctypedef bint zselect1(z*)\nctypedef bint zselect2(z*, z*)\n\n'

def generate_lapack_pxd(all_sigs):
    if False:
        print('Hello World!')
    return lapack_pxd_preamble + '\n'.join((pxd_decl(*sig) for sig in all_sigs))
fortran_template = '      subroutine {name}wrp(\n     +    ret,\n     +    {argnames}\n     +    )\n        external {wrapper}\n        {ret_type} {wrapper}\n        {ret_type} ret\n        {argdecls}\n        ret = {wrapper}(\n     +    {argnames}\n     +    )\n      end\n'
dims = {'work': '(*)', 'ab': '(ldab,*)', 'a': '(lda,*)', 'dl': '(*)', 'd': '(*)', 'du': '(*)', 'ap': '(*)', 'e': '(*)', 'lld': '(*)'}
xy_specialized_dims = {'x': '', 'y': ''}
a_specialized_dims = {'a': '(*)'}
special_cases = defaultdict(dict, ladiv=xy_specialized_dims, lanhf=a_specialized_dims, lansf=a_specialized_dims, lapy2=xy_specialized_dims, lapy3=xy_specialized_dims)

def process_fortran_name(name, funcname):
    if False:
        while True:
            i = 10
    if 'inc' in name:
        return name
    special = special_cases[funcname[1:]]
    if 'x' in name or 'y' in name:
        suffix = special.get(name, '(n)')
    else:
        suffix = special.get(name, '')
    return name + suffix

def called_name(name):
    if False:
        for i in range(10):
            print('nop')
    included = ['cdotc', 'cdotu', 'zdotc', 'zdotu', 'cladiv', 'zladiv']
    if name in included:
        return 'w' + name
    return name

def fort_subroutine_wrapper(name, ret_type, args):
    if False:
        print('Hello World!')
    wrapper = called_name(name)
    (types, names) = arg_names_and_types(args)
    argnames = ',\n     +    '.join(names)
    names = [process_fortran_name(n, name) for n in names]
    argdecls = '\n        '.join((f'{fortran_types[t]} {n}' for (n, t) in zip(names, types)))
    return fortran_template.format(name=name, wrapper=wrapper, argnames=argnames, argdecls=argdecls, ret_type=fortran_types[ret_type])

def generate_fortran(func_sigs):
    if False:
        for i in range(10):
            print('nop')
    return '\n'.join((fort_subroutine_wrapper(*sig) for sig in func_sigs))

def make_c_args(args):
    if False:
        print('Hello World!')
    (types, names) = arg_names_and_types(args)
    types = [c_types[arg] for arg in types]
    return ', '.join((f'{t} *{n}' for (t, n) in zip(types, names)))
c_func_template = 'void F_FUNC({name}wrp, {upname}WRP)({return_type} *ret, {args});\n'

def c_func_decl(name, return_type, args):
    if False:
        print('Hello World!')
    args = make_c_args(args)
    return_type = c_types[return_type]
    return c_func_template.format(name=name, upname=name.upper(), return_type=return_type, args=args)
c_sub_template = 'void F_FUNC({name},{upname})({args});\n'

def c_sub_decl(name, return_type, args):
    if False:
        i = 10
        return i + 15
    args = make_c_args(args)
    return c_sub_template.format(name=name, upname=name.upper(), args=args)
c_preamble = '#ifndef SCIPY_LINALG_{lib}_FORTRAN_WRAPPERS_H\n#define SCIPY_LINALG_{lib}_FORTRAN_WRAPPERS_H\n#include "fortran_defs.h"\n#include "numpy/arrayobject.h"\n'
lapack_decls = '\ntypedef int (*_cselect1)(npy_complex64*);\ntypedef int (*_cselect2)(npy_complex64*, npy_complex64*);\ntypedef int (*_dselect2)(double*, double*);\ntypedef int (*_dselect3)(double*, double*, double*);\ntypedef int (*_sselect2)(float*, float*);\ntypedef int (*_sselect3)(float*, float*, float*);\ntypedef int (*_zselect1)(npy_complex128*);\ntypedef int (*_zselect2)(npy_complex128*, npy_complex128*);\n'
cpp_guard = '\n#ifdef __cplusplus\nextern "C" {\n#endif\n\n'
c_end = '\n#ifdef __cplusplus\n}\n#endif\n#endif\n'

def generate_c_header(func_sigs, sub_sigs, all_sigs, lib_name):
    if False:
        while True:
            i = 10
    funcs = ''.join((c_func_decl(*sig) for sig in func_sigs))
    subs = '\n' + ''.join((c_sub_decl(*sig) for sig in sub_sigs))
    if lib_name == 'LAPACK':
        preamble = c_preamble.format(lib=lib_name) + lapack_decls
    else:
        preamble = c_preamble.format(lib=lib_name)
    return ''.join([preamble, cpp_guard, funcs, subs, c_end])

def split_signature(sig):
    if False:
        i = 10
        return i + 15
    (name_and_type, args) = sig[:-1].split('(')
    (ret_type, name) = name_and_type.split(' ')
    return (name, ret_type, args)

def filter_lines(lines):
    if False:
        return 10
    lines = [line for line in map(str.strip, lines) if line and (not line.startswith('#'))]
    func_sigs = [split_signature(line) for line in lines if line.split(' ')[0] != 'void']
    sub_sigs = [split_signature(line) for line in lines if line.split(' ')[0] == 'void']
    all_sigs = list(sorted(func_sigs + sub_sigs, key=itemgetter(0)))
    return (func_sigs, sub_sigs, all_sigs)

def newer(source, target):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return true if 'source' exists and is more recently modified than\n    'target', or if 'source' exists and 'target' doesn't.  Return false if\n    both exist and 'target' is the same age or younger than 'source'.\n    "
    if not os.path.exists(source):
        raise ValueError("file '%s' does not exist" % os.path.abspath(source))
    if not os.path.exists(target):
        return 1
    mtime1 = os.stat(source)[ST_MTIME]
    mtime2 = os.stat(target)[ST_MTIME]
    return mtime1 > mtime2

def all_newer(src_files, dst_files):
    if False:
        print('Hello World!')
    return all((os.path.exists(dst) and newer(dst, src) for dst in dst_files for src in src_files))

def make_all(outdir, blas_signature_file='cython_blas_signatures.txt', lapack_signature_file='cython_lapack_signatures.txt', blas_name='cython_blas', lapack_name='cython_lapack', blas_fortran_name='_blas_subroutine_wrappers.f', lapack_fortran_name='_lapack_subroutine_wrappers.f', blas_header_name='_blas_subroutines.h', lapack_header_name='_lapack_subroutines.h'):
    if False:
        return 10
    src_files = (os.path.abspath(__file__), blas_signature_file, lapack_signature_file)
    dst_files = (blas_name + '.pyx', blas_name + '.pxd', blas_fortran_name, blas_header_name, lapack_name + '.pyx', lapack_name + '.pxd', lapack_fortran_name, lapack_header_name)
    dst_files = (os.path.join(outdir, f) for f in dst_files)
    os.chdir(BASE_DIR)
    if all_newer(src_files, dst_files):
        print('scipy/linalg/_generate_pyx.py: all files up-to-date')
        return
    comments = ['This file was generated by _generate_pyx.py.\n', 'Do not edit this file directly.\n']
    ccomment = ''.join(['/* ' + line.rstrip() + ' */\n' for line in comments]) + '\n'
    pyxcomment = ''.join(['# ' + line for line in comments]) + '\n'
    fcomment = ''.join(['c     ' + line for line in comments]) + '\n'
    with open(blas_signature_file) as f:
        blas_sigs = f.readlines()
    blas_sigs = filter_lines(blas_sigs)
    blas_pyx = generate_blas_pyx(*blas_sigs + (blas_header_name,))
    with open(os.path.join(outdir, blas_name + '.pyx'), 'w') as f:
        f.write(pyxcomment)
        f.write(blas_pyx)
    blas_pxd = generate_blas_pxd(blas_sigs[2])
    with open(os.path.join(outdir, blas_name + '.pxd'), 'w') as f:
        f.write(pyxcomment)
        f.write(blas_pxd)
    blas_fortran = generate_fortran(blas_sigs[0])
    with open(os.path.join(outdir, blas_fortran_name), 'w') as f:
        f.write(fcomment)
        f.write(blas_fortran)
    blas_c_header = generate_c_header(*blas_sigs + ('BLAS',))
    with open(os.path.join(outdir, blas_header_name), 'w') as f:
        f.write(ccomment)
        f.write(blas_c_header)
    with open(lapack_signature_file) as f:
        lapack_sigs = f.readlines()
    lapack_sigs = filter_lines(lapack_sigs)
    lapack_pyx = generate_lapack_pyx(*lapack_sigs + (lapack_header_name,))
    with open(os.path.join(outdir, lapack_name + '.pyx'), 'w') as f:
        f.write(pyxcomment)
        f.write(lapack_pyx)
    lapack_pxd = generate_lapack_pxd(lapack_sigs[2])
    with open(os.path.join(outdir, lapack_name + '.pxd'), 'w') as f:
        f.write(pyxcomment)
        f.write(lapack_pxd)
    lapack_fortran = generate_fortran(lapack_sigs[0])
    with open(os.path.join(outdir, lapack_fortran_name), 'w') as f:
        f.write(fcomment)
        f.write(lapack_fortran)
    lapack_c_header = generate_c_header(*lapack_sigs + ('LAPACK',))
    with open(os.path.join(outdir, lapack_header_name), 'w') as f:
        f.write(ccomment)
        f.write(lapack_c_header)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', type=str, help='Path to the output directory')
    args = parser.parse_args()
    if not args.outdir:
        raise ValueError('Missing `--outdir` argument to _generate_pyx.py')
    else:
        outdir_abs = os.path.join(os.getcwd(), args.outdir)
    make_all(outdir_abs)