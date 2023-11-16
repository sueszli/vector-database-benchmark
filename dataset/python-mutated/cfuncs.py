"""

C declarations, CPP macros, and C functions for f2py2e.
Only required declarations/macros/functions will be used.

Copyright 1999,2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2005/05/06 11:42:34 $
Pearu Peterson

"""
import sys
import copy
from . import __version__
f2py_version = __version__.version
errmess = sys.stderr.write
outneeds = {'includes0': [], 'includes': [], 'typedefs': [], 'typedefs_generated': [], 'userincludes': [], 'cppmacros': [], 'cfuncs': [], 'callbacks': [], 'f90modhooks': [], 'commonhooks': []}
needs = {}
includes0 = {'includes0': '/*need_includes0*/'}
includes = {'includes': '/*need_includes*/'}
userincludes = {'userincludes': '/*need_userincludes*/'}
typedefs = {'typedefs': '/*need_typedefs*/'}
typedefs_generated = {'typedefs_generated': '/*need_typedefs_generated*/'}
cppmacros = {'cppmacros': '/*need_cppmacros*/'}
cfuncs = {'cfuncs': '/*need_cfuncs*/'}
callbacks = {'callbacks': '/*need_callbacks*/'}
f90modhooks = {'f90modhooks': '/*need_f90modhooks*/', 'initf90modhooksstatic': '/*initf90modhooksstatic*/', 'initf90modhooksdynamic': '/*initf90modhooksdynamic*/'}
commonhooks = {'commonhooks': '/*need_commonhooks*/', 'initcommonhooks': '/*need_initcommonhooks*/'}
includes0['math.h'] = '#include <math.h>'
includes0['string.h'] = '#include <string.h>'
includes0['setjmp.h'] = '#include <setjmp.h>'
includes['arrayobject.h'] = '#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API\n#include "arrayobject.h"'
includes['npy_math.h'] = '#include "numpy/npy_math.h"'
includes['arrayobject.h'] = '#include "fortranobject.h"'
includes['stdarg.h'] = '#include <stdarg.h>'
typedefs['unsigned_char'] = 'typedef unsigned char unsigned_char;'
typedefs['unsigned_short'] = 'typedef unsigned short unsigned_short;'
typedefs['unsigned_long'] = 'typedef unsigned long unsigned_long;'
typedefs['signed_char'] = 'typedef signed char signed_char;'
typedefs['long_long'] = '#if defined(NPY_OS_WIN32)\ntypedef __int64 long_long;\n#else\ntypedef long long long_long;\ntypedef unsigned long long unsigned_long_long;\n#endif\n'
typedefs['unsigned_long_long'] = '#if defined(NPY_OS_WIN32)\ntypedef __uint64 long_long;\n#else\ntypedef unsigned long long unsigned_long_long;\n#endif\n'
typedefs['long_double'] = '#ifndef _LONG_DOUBLE\ntypedef long double long_double;\n#endif\n'
typedefs['complex_long_double'] = 'typedef struct {long double r,i;} complex_long_double;'
typedefs['complex_float'] = 'typedef struct {float r,i;} complex_float;'
typedefs['complex_double'] = 'typedef struct {double r,i;} complex_double;'
typedefs['string'] = 'typedef char * string;'
typedefs['character'] = 'typedef char character;'
cppmacros['CFUNCSMESS'] = '#ifdef DEBUGCFUNCS\n#define CFUNCSMESS(mess) fprintf(stderr,"debug-capi:"mess);\n#define CFUNCSMESSPY(mess,obj) CFUNCSMESS(mess) \\\n    PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\\\n    fprintf(stderr,"\\n");\n#else\n#define CFUNCSMESS(mess)\n#define CFUNCSMESSPY(mess,obj)\n#endif\n'
cppmacros['F_FUNC'] = '#if defined(PREPEND_FORTRAN)\n#if defined(NO_APPEND_FORTRAN)\n#if defined(UPPERCASE_FORTRAN)\n#define F_FUNC(f,F) _##F\n#else\n#define F_FUNC(f,F) _##f\n#endif\n#else\n#if defined(UPPERCASE_FORTRAN)\n#define F_FUNC(f,F) _##F##_\n#else\n#define F_FUNC(f,F) _##f##_\n#endif\n#endif\n#else\n#if defined(NO_APPEND_FORTRAN)\n#if defined(UPPERCASE_FORTRAN)\n#define F_FUNC(f,F) F\n#else\n#define F_FUNC(f,F) f\n#endif\n#else\n#if defined(UPPERCASE_FORTRAN)\n#define F_FUNC(f,F) F##_\n#else\n#define F_FUNC(f,F) f##_\n#endif\n#endif\n#endif\n#if defined(UNDERSCORE_G77)\n#define F_FUNC_US(f,F) F_FUNC(f##_,F##_)\n#else\n#define F_FUNC_US(f,F) F_FUNC(f,F)\n#endif\n'
cppmacros['F_WRAPPEDFUNC'] = '#if defined(PREPEND_FORTRAN)\n#if defined(NO_APPEND_FORTRAN)\n#if defined(UPPERCASE_FORTRAN)\n#define F_WRAPPEDFUNC(f,F) _F2PYWRAP##F\n#else\n#define F_WRAPPEDFUNC(f,F) _f2pywrap##f\n#endif\n#else\n#if defined(UPPERCASE_FORTRAN)\n#define F_WRAPPEDFUNC(f,F) _F2PYWRAP##F##_\n#else\n#define F_WRAPPEDFUNC(f,F) _f2pywrap##f##_\n#endif\n#endif\n#else\n#if defined(NO_APPEND_FORTRAN)\n#if defined(UPPERCASE_FORTRAN)\n#define F_WRAPPEDFUNC(f,F) F2PYWRAP##F\n#else\n#define F_WRAPPEDFUNC(f,F) f2pywrap##f\n#endif\n#else\n#if defined(UPPERCASE_FORTRAN)\n#define F_WRAPPEDFUNC(f,F) F2PYWRAP##F##_\n#else\n#define F_WRAPPEDFUNC(f,F) f2pywrap##f##_\n#endif\n#endif\n#endif\n#if defined(UNDERSCORE_G77)\n#define F_WRAPPEDFUNC_US(f,F) F_WRAPPEDFUNC(f##_,F##_)\n#else\n#define F_WRAPPEDFUNC_US(f,F) F_WRAPPEDFUNC(f,F)\n#endif\n'
cppmacros['F_MODFUNC'] = '#if defined(F90MOD2CCONV1) /*E.g. Compaq Fortran */\n#if defined(NO_APPEND_FORTRAN)\n#define F_MODFUNCNAME(m,f) $ ## m ## $ ## f\n#else\n#define F_MODFUNCNAME(m,f) $ ## m ## $ ## f ## _\n#endif\n#endif\n\n#if defined(F90MOD2CCONV2) /*E.g. IBM XL Fortran, not tested though */\n#if defined(NO_APPEND_FORTRAN)\n#define F_MODFUNCNAME(m,f)  __ ## m ## _MOD_ ## f\n#else\n#define F_MODFUNCNAME(m,f)  __ ## m ## _MOD_ ## f ## _\n#endif\n#endif\n\n#if defined(F90MOD2CCONV3) /*E.g. MIPSPro Compilers */\n#if defined(NO_APPEND_FORTRAN)\n#define F_MODFUNCNAME(m,f)  f ## .in. ## m\n#else\n#define F_MODFUNCNAME(m,f)  f ## .in. ## m ## _\n#endif\n#endif\n/*\n#if defined(UPPERCASE_FORTRAN)\n#define F_MODFUNC(m,M,f,F) F_MODFUNCNAME(M,F)\n#else\n#define F_MODFUNC(m,M,f,F) F_MODFUNCNAME(m,f)\n#endif\n*/\n\n#define F_MODFUNC(m,f) (*(f2pymodstruct##m##.##f))\n'
cppmacros['SWAPUNSAFE'] = '#define SWAP(a,b) (size_t)(a) = ((size_t)(a) ^ (size_t)(b));\\\n (size_t)(b) = ((size_t)(a) ^ (size_t)(b));\\\n (size_t)(a) = ((size_t)(a) ^ (size_t)(b))\n'
cppmacros['SWAP'] = '#define SWAP(a,b,t) {\\\n    t *c;\\\n    c = a;\\\n    a = b;\\\n    b = c;}\n'
cppmacros['PRINTPYOBJERR'] = '#define PRINTPYOBJERR(obj)\\\n    fprintf(stderr,"#modulename#.error is related to ");\\\n    PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\\\n    fprintf(stderr,"\\n");\n'
cppmacros['MINMAX'] = '#ifndef max\n#define max(a,b) ((a > b) ? (a) : (b))\n#endif\n#ifndef min\n#define min(a,b) ((a < b) ? (a) : (b))\n#endif\n#ifndef MAX\n#define MAX(a,b) ((a > b) ? (a) : (b))\n#endif\n#ifndef MIN\n#define MIN(a,b) ((a < b) ? (a) : (b))\n#endif\n'
cppmacros['len..'] = '/* See fortranobject.h for definitions. The macros here are provided for BC. */\n#define rank f2py_rank\n#define shape f2py_shape\n#define fshape f2py_shape\n#define len f2py_len\n#define flen f2py_flen\n#define slen f2py_slen\n#define size f2py_size\n'
cppmacros['pyobj_from_char1'] = '#define pyobj_from_char1(v) (PyLong_FromLong(v))'
cppmacros['pyobj_from_short1'] = '#define pyobj_from_short1(v) (PyLong_FromLong(v))'
needs['pyobj_from_int1'] = ['signed_char']
cppmacros['pyobj_from_int1'] = '#define pyobj_from_int1(v) (PyLong_FromLong(v))'
cppmacros['pyobj_from_long1'] = '#define pyobj_from_long1(v) (PyLong_FromLong(v))'
needs['pyobj_from_long_long1'] = ['long_long']
cppmacros['pyobj_from_long_long1'] = '#ifdef HAVE_LONG_LONG\n#define pyobj_from_long_long1(v) (PyLong_FromLongLong(v))\n#else\n#warning HAVE_LONG_LONG is not available. Redefining pyobj_from_long_long.\n#define pyobj_from_long_long1(v) (PyLong_FromLong(v))\n#endif\n'
needs['pyobj_from_long_double1'] = ['long_double']
cppmacros['pyobj_from_long_double1'] = '#define pyobj_from_long_double1(v) (PyFloat_FromDouble(v))'
cppmacros['pyobj_from_double1'] = '#define pyobj_from_double1(v) (PyFloat_FromDouble(v))'
cppmacros['pyobj_from_float1'] = '#define pyobj_from_float1(v) (PyFloat_FromDouble(v))'
needs['pyobj_from_complex_long_double1'] = ['complex_long_double']
cppmacros['pyobj_from_complex_long_double1'] = '#define pyobj_from_complex_long_double1(v) (PyComplex_FromDoubles(v.r,v.i))'
needs['pyobj_from_complex_double1'] = ['complex_double']
cppmacros['pyobj_from_complex_double1'] = '#define pyobj_from_complex_double1(v) (PyComplex_FromDoubles(v.r,v.i))'
needs['pyobj_from_complex_float1'] = ['complex_float']
cppmacros['pyobj_from_complex_float1'] = '#define pyobj_from_complex_float1(v) (PyComplex_FromDoubles(v.r,v.i))'
needs['pyobj_from_string1'] = ['string']
cppmacros['pyobj_from_string1'] = '#define pyobj_from_string1(v) (PyUnicode_FromString((char *)v))'
needs['pyobj_from_string1size'] = ['string']
cppmacros['pyobj_from_string1size'] = '#define pyobj_from_string1size(v,len) (PyUnicode_FromStringAndSize((char *)v, len))'
needs['TRYPYARRAYTEMPLATE'] = ['PRINTPYOBJERR']
cppmacros['TRYPYARRAYTEMPLATE'] = '/* New SciPy */\n#define TRYPYARRAYTEMPLATECHAR case NPY_STRING: *(char *)(PyArray_DATA(arr))=*v; break;\n#define TRYPYARRAYTEMPLATELONG case NPY_LONG: *(long *)(PyArray_DATA(arr))=*v; break;\n#define TRYPYARRAYTEMPLATEOBJECT case NPY_OBJECT: PyArray_SETITEM(arr,PyArray_DATA(arr),pyobj_from_ ## ctype ## 1(*v)); break;\n\n#define TRYPYARRAYTEMPLATE(ctype,typecode) \\\n        PyArrayObject *arr = NULL;\\\n        if (!obj) return -2;\\\n        if (!PyArray_Check(obj)) return -1;\\\n        if (!(arr=(PyArrayObject *)obj)) {fprintf(stderr,"TRYPYARRAYTEMPLATE:");PRINTPYOBJERR(obj);return 0;}\\\n        if (PyArray_DESCR(arr)->type==typecode)  {*(ctype *)(PyArray_DATA(arr))=*v; return 1;}\\\n        switch (PyArray_TYPE(arr)) {\\\n                case NPY_DOUBLE: *(npy_double *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_INT: *(npy_int *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_LONG: *(npy_long *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_FLOAT: *(npy_float *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_CDOUBLE: *(npy_double *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_CFLOAT: *(npy_float *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_BOOL: *(npy_bool *)(PyArray_DATA(arr))=(*v!=0); break;\\\n                case NPY_UBYTE: *(npy_ubyte *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_BYTE: *(npy_byte *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_SHORT: *(npy_short *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_USHORT: *(npy_ushort *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_UINT: *(npy_uint *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_ULONG: *(npy_ulong *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_LONGLONG: *(npy_longlong *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_ULONGLONG: *(npy_ulonglong *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_LONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_CLONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_OBJECT: PyArray_SETITEM(arr, PyArray_DATA(arr), pyobj_from_ ## ctype ## 1(*v)); break;\\\n        default: return -2;\\\n        };\\\n        return 1\n'
needs['TRYCOMPLEXPYARRAYTEMPLATE'] = ['PRINTPYOBJERR']
cppmacros['TRYCOMPLEXPYARRAYTEMPLATE'] = '#define TRYCOMPLEXPYARRAYTEMPLATEOBJECT case NPY_OBJECT: PyArray_SETITEM(arr, PyArray_DATA(arr), pyobj_from_complex_ ## ctype ## 1((*v))); break;\n#define TRYCOMPLEXPYARRAYTEMPLATE(ctype,typecode)\\\n        PyArrayObject *arr = NULL;\\\n        if (!obj) return -2;\\\n        if (!PyArray_Check(obj)) return -1;\\\n        if (!(arr=(PyArrayObject *)obj)) {fprintf(stderr,"TRYCOMPLEXPYARRAYTEMPLATE:");PRINTPYOBJERR(obj);return 0;}\\\n        if (PyArray_DESCR(arr)->type==typecode) {\\\n            *(ctype *)(PyArray_DATA(arr))=(*v).r;\\\n            *(ctype *)(PyArray_DATA(arr)+sizeof(ctype))=(*v).i;\\\n            return 1;\\\n        }\\\n        switch (PyArray_TYPE(arr)) {\\\n                case NPY_CDOUBLE: *(npy_double *)(PyArray_DATA(arr))=(*v).r;\\\n                                  *(npy_double *)(PyArray_DATA(arr)+sizeof(npy_double))=(*v).i;\\\n                                  break;\\\n                case NPY_CFLOAT: *(npy_float *)(PyArray_DATA(arr))=(*v).r;\\\n                                 *(npy_float *)(PyArray_DATA(arr)+sizeof(npy_float))=(*v).i;\\\n                                 break;\\\n                case NPY_DOUBLE: *(npy_double *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_LONG: *(npy_long *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_FLOAT: *(npy_float *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_INT: *(npy_int *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_SHORT: *(npy_short *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_UBYTE: *(npy_ubyte *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_BYTE: *(npy_byte *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_BOOL: *(npy_bool *)(PyArray_DATA(arr))=((*v).r!=0 && (*v).i!=0); break;\\\n                case NPY_USHORT: *(npy_ushort *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_UINT: *(npy_uint *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_ULONG: *(npy_ulong *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_LONGLONG: *(npy_longlong *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_ULONGLONG: *(npy_ulonglong *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_LONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_CLONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=(*v).r;\\\n                                      *(npy_longdouble *)(PyArray_DATA(arr)+sizeof(npy_longdouble))=(*v).i;\\\n                                      break;\\\n                case NPY_OBJECT: PyArray_SETITEM(arr, PyArray_DATA(arr), pyobj_from_complex_ ## ctype ## 1((*v))); break;\\\n                default: return -2;\\\n        };\\\n        return -1;\n'
needs['GETSTRFROMPYTUPLE'] = ['STRINGCOPYN', 'PRINTPYOBJERR']
cppmacros['GETSTRFROMPYTUPLE'] = '#define GETSTRFROMPYTUPLE(tuple,index,str,len) {\\\n        PyObject *rv_cb_str = PyTuple_GetItem((tuple),(index));\\\n        if (rv_cb_str == NULL)\\\n            goto capi_fail;\\\n        if (PyBytes_Check(rv_cb_str)) {\\\n            str[len-1]=\'\\0\';\\\n            STRINGCOPYN((str),PyBytes_AS_STRING((PyBytesObject*)rv_cb_str),(len));\\\n        } else {\\\n            PRINTPYOBJERR(rv_cb_str);\\\n            PyErr_SetString(#modulename#_error,"string object expected");\\\n            goto capi_fail;\\\n        }\\\n    }\n'
cppmacros['GETSCALARFROMPYTUPLE'] = '#define GETSCALARFROMPYTUPLE(tuple,index,var,ctype,mess) {\\\n        if ((capi_tmp = PyTuple_GetItem((tuple),(index)))==NULL) goto capi_fail;\\\n        if (!(ctype ## _from_pyobj((var),capi_tmp,mess)))\\\n            goto capi_fail;\\\n    }\n'
cppmacros['FAILNULL'] = '\\\n#define FAILNULL(p) do {                                            \\\n    if ((p) == NULL) {                                              \\\n        PyErr_SetString(PyExc_MemoryError, "NULL pointer found");   \\\n        goto capi_fail;                                             \\\n    }                                                               \\\n} while (0)\n'
needs['MEMCOPY'] = ['string.h', 'FAILNULL']
cppmacros['MEMCOPY'] = '#define MEMCOPY(to,from,n)\\\n    do { FAILNULL(to); FAILNULL(from); (void)memcpy(to,from,n); } while (0)\n'
cppmacros['STRINGMALLOC'] = '#define STRINGMALLOC(str,len)\\\n    if ((str = (string)malloc(len+1)) == NULL) {\\\n        PyErr_SetString(PyExc_MemoryError, "out of memory");\\\n        goto capi_fail;\\\n    } else {\\\n        (str)[len] = \'\\0\';\\\n    }\n'
cppmacros['STRINGFREE'] = '#define STRINGFREE(str) do {if (!(str == NULL)) free(str);} while (0)\n'
needs['STRINGPADN'] = ['string.h']
cppmacros['STRINGPADN'] = '/*\nSTRINGPADN replaces null values with padding values from the right.\n\n`to` must have size of at least N bytes.\n\nIf the `to[N-1]` has null value, then replace it and all the\npreceding, nulls with the given padding.\n\nSTRINGPADN(to, N, PADDING, NULLVALUE) is an inverse operation.\n*/\n#define STRINGPADN(to, N, NULLVALUE, PADDING)                   \\\n    do {                                                        \\\n        int _m = (N);                                           \\\n        char *_to = (to);                                       \\\n        for (_m -= 1; _m >= 0 && _to[_m] == NULLVALUE; _m--) {  \\\n             _to[_m] = PADDING;                                 \\\n        }                                                       \\\n    } while (0)\n'
needs['STRINGCOPYN'] = ['string.h', 'FAILNULL']
cppmacros['STRINGCOPYN'] = '/*\nSTRINGCOPYN copies N bytes.\n\n`to` and `from` buffers must have sizes of at least N bytes.\n*/\n#define STRINGCOPYN(to,from,N)                                  \\\n    do {                                                        \\\n        int _m = (N);                                           \\\n        char *_to = (to);                                       \\\n        char *_from = (from);                                   \\\n        FAILNULL(_to); FAILNULL(_from);                         \\\n        (void)strncpy(_to, _from, _m);             \\\n    } while (0)\n'
needs['STRINGCOPY'] = ['string.h', 'FAILNULL']
cppmacros['STRINGCOPY'] = '#define STRINGCOPY(to,from)\\\n    do { FAILNULL(to); FAILNULL(from); (void)strcpy(to,from); } while (0)\n'
cppmacros['CHECKGENERIC'] = '#define CHECKGENERIC(check,tcheck,name) \\\n    if (!(check)) {\\\n        PyErr_SetString(#modulename#_error,"("tcheck") failed for "name);\\\n        /*goto capi_fail;*/\\\n    } else '
cppmacros['CHECKARRAY'] = '#define CHECKARRAY(check,tcheck,name) \\\n    if (!(check)) {\\\n        PyErr_SetString(#modulename#_error,"("tcheck") failed for "name);\\\n        /*goto capi_fail;*/\\\n    } else '
cppmacros['CHECKSTRING'] = '#define CHECKSTRING(check,tcheck,name,show,var)\\\n    if (!(check)) {\\\n        char errstring[256];\\\n        sprintf(errstring, "%s: "show, "("tcheck") failed for "name, slen(var), var);\\\n        PyErr_SetString(#modulename#_error, errstring);\\\n        /*goto capi_fail;*/\\\n    } else '
cppmacros['CHECKSCALAR'] = '#define CHECKSCALAR(check,tcheck,name,show,var)\\\n    if (!(check)) {\\\n        char errstring[256];\\\n        sprintf(errstring, "%s: "show, "("tcheck") failed for "name, var);\\\n        PyErr_SetString(#modulename#_error,errstring);\\\n        /*goto capi_fail;*/\\\n    } else '
cppmacros['ARRSIZE'] = '#define ARRSIZE(dims,rank) (_PyArray_multiply_list(dims,rank))'
cppmacros['OLDPYNUM'] = '#ifdef OLDPYNUM\n#error You need to install NumPy version 0.13 or higher. See https://scipy.org/install.html\n#endif\n'
cppmacros['F2PY_THREAD_LOCAL_DECL'] = '#ifndef F2PY_THREAD_LOCAL_DECL\n#if defined(_MSC_VER)\n#define F2PY_THREAD_LOCAL_DECL __declspec(thread)\n#elif defined(NPY_OS_MINGW)\n#define F2PY_THREAD_LOCAL_DECL __thread\n#elif defined(__STDC_VERSION__) \\\n      && (__STDC_VERSION__ >= 201112L) \\\n      && !defined(__STDC_NO_THREADS__) \\\n      && (!defined(__GLIBC__) || __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 12)) \\\n      && !defined(NPY_OS_OPENBSD) && !defined(NPY_OS_HAIKU)\n/* __STDC_NO_THREADS__ was first defined in a maintenance release of glibc 2.12,\n   see https://lists.gnu.org/archive/html/commit-hurd/2012-07/msg00180.html,\n   so `!defined(__STDC_NO_THREADS__)` may give false positive for the existence\n   of `threads.h` when using an older release of glibc 2.12\n   See gh-19437 for details on OpenBSD */\n#include <threads.h>\n#define F2PY_THREAD_LOCAL_DECL thread_local\n#elif defined(__GNUC__) \\\n      && (__GNUC__ > 4 || (__GNUC__ == 4 && (__GNUC_MINOR__ >= 4)))\n#define F2PY_THREAD_LOCAL_DECL __thread\n#endif\n#endif\n'
cfuncs['calcarrindex'] = 'static int calcarrindex(int *i,PyArrayObject *arr) {\n    int k,ii = i[0];\n    for (k=1; k < PyArray_NDIM(arr); k++)\n        ii += (ii*(PyArray_DIM(arr,k) - 1)+i[k]); /* assuming contiguous arr */\n    return ii;\n}'
cfuncs['calcarrindextr'] = 'static int calcarrindextr(int *i,PyArrayObject *arr) {\n    int k,ii = i[PyArray_NDIM(arr)-1];\n    for (k=1; k < PyArray_NDIM(arr); k++)\n        ii += (ii*(PyArray_DIM(arr,PyArray_NDIM(arr)-k-1) - 1)+i[PyArray_NDIM(arr)-k-1]); /* assuming contiguous arr */\n    return ii;\n}'
cfuncs['forcomb'] = 'static struct { int nd;npy_intp *d;int *i,*i_tr,tr; } forcombcache;\nstatic int initforcomb(npy_intp *dims,int nd,int tr) {\n  int k;\n  if (dims==NULL) return 0;\n  if (nd<0) return 0;\n  forcombcache.nd = nd;\n  forcombcache.d = dims;\n  forcombcache.tr = tr;\n  if ((forcombcache.i = (int *)malloc(sizeof(int)*nd))==NULL) return 0;\n  if ((forcombcache.i_tr = (int *)malloc(sizeof(int)*nd))==NULL) return 0;\n  for (k=1;k<nd;k++) {\n    forcombcache.i[k] = forcombcache.i_tr[nd-k-1] = 0;\n  }\n  forcombcache.i[0] = forcombcache.i_tr[nd-1] = -1;\n  return 1;\n}\nstatic int *nextforcomb(void) {\n  int j,*i,*i_tr,k;\n  int nd=forcombcache.nd;\n  if ((i=forcombcache.i) == NULL) return NULL;\n  if ((i_tr=forcombcache.i_tr) == NULL) return NULL;\n  if (forcombcache.d == NULL) return NULL;\n  i[0]++;\n  if (i[0]==forcombcache.d[0]) {\n    j=1;\n    while ((j<nd) && (i[j]==forcombcache.d[j]-1)) j++;\n    if (j==nd) {\n      free(i);\n      free(i_tr);\n      return NULL;\n    }\n    for (k=0;k<j;k++) i[k] = i_tr[nd-k-1] = 0;\n    i[j]++;\n    i_tr[nd-j-1]++;\n  } else\n    i_tr[nd-1]++;\n  if (forcombcache.tr) return i_tr;\n  return i;\n}'
needs['try_pyarr_from_string'] = ['STRINGCOPYN', 'PRINTPYOBJERR', 'string']
cfuncs['try_pyarr_from_string'] = '/*\n  try_pyarr_from_string copies str[:len(obj)] to the data of an `ndarray`.\n\n  If obj is an `ndarray`, it is assumed to be contiguous.\n\n  If the specified len==-1, str must be null-terminated.\n*/\nstatic int try_pyarr_from_string(PyObject *obj,\n                                 const string str, const int len) {\n#ifdef DEBUGCFUNCS\nfprintf(stderr, "try_pyarr_from_string(str=\'%s\', len=%d, obj=%p)\\n",\n        (char*)str,len, obj);\n#endif\n    if (PyArray_Check(obj)) {\n        PyArrayObject *arr = (PyArrayObject *)obj;\n        assert(ISCONTIGUOUS(arr));\n        string buf = PyArray_DATA(arr);\n        npy_intp n = len;\n        if (n == -1) {\n            /* Assuming null-terminated str. */\n            n = strlen(str);\n        }\n        if (n > PyArray_NBYTES(arr)) {\n            n = PyArray_NBYTES(arr);\n        }\n        STRINGCOPYN(buf, str, n);\n        return 1;\n    }\ncapi_fail:\n    PRINTPYOBJERR(obj);\n    PyErr_SetString(#modulename#_error, "try_pyarr_from_string failed");\n    return 0;\n}\n'
needs['string_from_pyobj'] = ['string', 'STRINGMALLOC', 'STRINGCOPYN']
cfuncs['string_from_pyobj'] = '/*\n  Create a new string buffer `str` of at most length `len` from a\n  Python string-like object `obj`.\n\n  The string buffer has given size (len) or the size of inistr when len==-1.\n\n  The string buffer is padded with blanks: in Fortran, trailing blanks\n  are insignificant contrary to C nulls.\n */\nstatic int\nstring_from_pyobj(string *str, int *len, const string inistr, PyObject *obj,\n                  const char *errmess)\n{\n    PyObject *tmp = NULL;\n    string buf = NULL;\n    npy_intp n = -1;\n#ifdef DEBUGCFUNCS\nfprintf(stderr,"string_from_pyobj(str=\'%s\',len=%d,inistr=\'%s\',obj=%p)\\n",\n               (char*)str, *len, (char *)inistr, obj);\n#endif\n    if (obj == Py_None) {\n        n = strlen(inistr);\n        buf = inistr;\n    }\n    else if (PyArray_Check(obj)) {\n        PyArrayObject *arr = (PyArrayObject *)obj;\n        if (!ISCONTIGUOUS(arr)) {\n            PyErr_SetString(PyExc_ValueError,\n                            "array object is non-contiguous.");\n            goto capi_fail;\n        }\n        n = PyArray_NBYTES(arr);\n        buf = PyArray_DATA(arr);\n        n = strnlen(buf, n);\n    }\n    else {\n        if (PyBytes_Check(obj)) {\n            tmp = obj;\n            Py_INCREF(tmp);\n        }\n        else if (PyUnicode_Check(obj)) {\n            tmp = PyUnicode_AsASCIIString(obj);\n        }\n        else {\n            PyObject *tmp2;\n            tmp2 = PyObject_Str(obj);\n            if (tmp2) {\n                tmp = PyUnicode_AsASCIIString(tmp2);\n                Py_DECREF(tmp2);\n            }\n            else {\n                tmp = NULL;\n            }\n        }\n        if (tmp == NULL) goto capi_fail;\n        n = PyBytes_GET_SIZE(tmp);\n        buf = PyBytes_AS_STRING(tmp);\n    }\n    if (*len == -1) {\n        /* TODO: change the type of `len` so that we can remove this */\n        if (n > NPY_MAX_INT) {\n            PyErr_SetString(PyExc_OverflowError,\n                            "object too large for a 32-bit int");\n            goto capi_fail;\n        }\n        *len = n;\n    }\n    else if (*len < n) {\n        /* discard the last (len-n) bytes of input buf */\n        n = *len;\n    }\n    if (n < 0 || *len < 0 || buf == NULL) {\n        goto capi_fail;\n    }\n    STRINGMALLOC(*str, *len);  // *str is allocated with size (*len + 1)\n    if (n < *len) {\n        /*\n          Pad fixed-width string with nulls. The caller will replace\n          nulls with blanks when the corresponding argument is not\n          intent(c).\n        */\n        memset(*str + n, \'\\0\', *len - n);\n    }\n    STRINGCOPYN(*str, buf, n);\n    Py_XDECREF(tmp);\n    return 1;\ncapi_fail:\n    Py_XDECREF(tmp);\n    {\n        PyObject* err = PyErr_Occurred();\n        if (err == NULL) {\n            err = #modulename#_error;\n        }\n        PyErr_SetString(err, errmess);\n    }\n    return 0;\n}\n'
cfuncs['character_from_pyobj'] = 'static int\ncharacter_from_pyobj(character* v, PyObject *obj, const char *errmess) {\n    if (PyBytes_Check(obj)) {\n        /* empty bytes has trailing null, so dereferencing is always safe */\n        *v = PyBytes_AS_STRING(obj)[0];\n        return 1;\n    } else if (PyUnicode_Check(obj)) {\n        PyObject* tmp = PyUnicode_AsASCIIString(obj);\n        if (tmp != NULL) {\n            *v = PyBytes_AS_STRING(tmp)[0];\n            Py_DECREF(tmp);\n            return 1;\n        }\n    } else if (PyArray_Check(obj)) {\n        PyArrayObject* arr = (PyArrayObject*)obj;\n        if (F2PY_ARRAY_IS_CHARACTER_COMPATIBLE(arr)) {\n            *v = PyArray_BYTES(arr)[0];\n            return 1;\n        } else if (F2PY_IS_UNICODE_ARRAY(arr)) {\n            // TODO: update when numpy will support 1-byte and\n            // 2-byte unicode dtypes\n            PyObject* tmp = PyUnicode_FromKindAndData(\n                              PyUnicode_4BYTE_KIND,\n                              PyArray_BYTES(arr),\n                              (PyArray_NBYTES(arr)>0?1:0));\n            if (tmp != NULL) {\n                if (character_from_pyobj(v, tmp, errmess)) {\n                    Py_DECREF(tmp);\n                    return 1;\n                }\n                Py_DECREF(tmp);\n            }\n        }\n    } else if (PySequence_Check(obj)) {\n        PyObject* tmp = PySequence_GetItem(obj,0);\n        if (tmp != NULL) {\n            if (character_from_pyobj(v, tmp, errmess)) {\n                Py_DECREF(tmp);\n                return 1;\n            }\n            Py_DECREF(tmp);\n        }\n    }\n    {\n        /* TODO: This error (and most other) error handling needs cleaning. */\n        char mess[F2PY_MESSAGE_BUFFER_SIZE];\n        strcpy(mess, errmess);\n        PyObject* err = PyErr_Occurred();\n        if (err == NULL) {\n            err = PyExc_TypeError;\n            Py_INCREF(err);\n        }\n        else {\n            Py_INCREF(err);\n            PyErr_Clear();\n        }\n        sprintf(mess + strlen(mess),\n                " -- expected str|bytes|sequence-of-str-or-bytes, got ");\n        f2py_describe(obj, mess + strlen(mess));\n        PyErr_SetString(err, mess);\n        Py_DECREF(err);\n    }\n    return 0;\n}\n'
needs['char_from_pyobj'] = ['int_from_pyobj']
cfuncs['char_from_pyobj'] = 'static int\nchar_from_pyobj(char* v, PyObject *obj, const char *errmess) {\n    int i = 0;\n    if (int_from_pyobj(&i, obj, errmess)) {\n        *v = (char)i;\n        return 1;\n    }\n    return 0;\n}\n'
needs['signed_char_from_pyobj'] = ['int_from_pyobj', 'signed_char']
cfuncs['signed_char_from_pyobj'] = 'static int\nsigned_char_from_pyobj(signed_char* v, PyObject *obj, const char *errmess) {\n    int i = 0;\n    if (int_from_pyobj(&i, obj, errmess)) {\n        *v = (signed_char)i;\n        return 1;\n    }\n    return 0;\n}\n'
needs['short_from_pyobj'] = ['int_from_pyobj']
cfuncs['short_from_pyobj'] = 'static int\nshort_from_pyobj(short* v, PyObject *obj, const char *errmess) {\n    int i = 0;\n    if (int_from_pyobj(&i, obj, errmess)) {\n        *v = (short)i;\n        return 1;\n    }\n    return 0;\n}\n'
cfuncs['int_from_pyobj'] = 'static int\nint_from_pyobj(int* v, PyObject *obj, const char *errmess)\n{\n    PyObject* tmp = NULL;\n\n    if (PyLong_Check(obj)) {\n        *v = Npy__PyLong_AsInt(obj);\n        return !(*v == -1 && PyErr_Occurred());\n    }\n\n    tmp = PyNumber_Long(obj);\n    if (tmp) {\n        *v = Npy__PyLong_AsInt(tmp);\n        Py_DECREF(tmp);\n        return !(*v == -1 && PyErr_Occurred());\n    }\n\n    if (PyComplex_Check(obj)) {\n        PyErr_Clear();\n        tmp = PyObject_GetAttrString(obj,"real");\n    }\n    else if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {\n        /*pass*/;\n    }\n    else if (PySequence_Check(obj)) {\n        PyErr_Clear();\n        tmp = PySequence_GetItem(obj, 0);\n    }\n\n    if (tmp) {\n        if (int_from_pyobj(v, tmp, errmess)) {\n            Py_DECREF(tmp);\n            return 1;\n        }\n        Py_DECREF(tmp);\n    }\n\n    {\n        PyObject* err = PyErr_Occurred();\n        if (err == NULL) {\n            err = #modulename#_error;\n        }\n        PyErr_SetString(err, errmess);\n    }\n    return 0;\n}\n'
cfuncs['long_from_pyobj'] = 'static int\nlong_from_pyobj(long* v, PyObject *obj, const char *errmess) {\n    PyObject* tmp = NULL;\n\n    if (PyLong_Check(obj)) {\n        *v = PyLong_AsLong(obj);\n        return !(*v == -1 && PyErr_Occurred());\n    }\n\n    tmp = PyNumber_Long(obj);\n    if (tmp) {\n        *v = PyLong_AsLong(tmp);\n        Py_DECREF(tmp);\n        return !(*v == -1 && PyErr_Occurred());\n    }\n\n    if (PyComplex_Check(obj)) {\n        PyErr_Clear();\n        tmp = PyObject_GetAttrString(obj,"real");\n    }\n    else if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {\n        /*pass*/;\n    }\n    else if (PySequence_Check(obj)) {\n        PyErr_Clear();\n        tmp = PySequence_GetItem(obj, 0);\n    }\n\n    if (tmp) {\n        if (long_from_pyobj(v, tmp, errmess)) {\n            Py_DECREF(tmp);\n            return 1;\n        }\n        Py_DECREF(tmp);\n    }\n    {\n        PyObject* err = PyErr_Occurred();\n        if (err == NULL) {\n            err = #modulename#_error;\n        }\n        PyErr_SetString(err, errmess);\n    }\n    return 0;\n}\n'
needs['long_long_from_pyobj'] = ['long_long']
cfuncs['long_long_from_pyobj'] = 'static int\nlong_long_from_pyobj(long_long* v, PyObject *obj, const char *errmess)\n{\n    PyObject* tmp = NULL;\n\n    if (PyLong_Check(obj)) {\n        *v = PyLong_AsLongLong(obj);\n        return !(*v == -1 && PyErr_Occurred());\n    }\n\n    tmp = PyNumber_Long(obj);\n    if (tmp) {\n        *v = PyLong_AsLongLong(tmp);\n        Py_DECREF(tmp);\n        return !(*v == -1 && PyErr_Occurred());\n    }\n\n    if (PyComplex_Check(obj)) {\n        PyErr_Clear();\n        tmp = PyObject_GetAttrString(obj,"real");\n    }\n    else if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {\n        /*pass*/;\n    }\n    else if (PySequence_Check(obj)) {\n        PyErr_Clear();\n        tmp = PySequence_GetItem(obj, 0);\n    }\n\n    if (tmp) {\n        if (long_long_from_pyobj(v, tmp, errmess)) {\n            Py_DECREF(tmp);\n            return 1;\n        }\n        Py_DECREF(tmp);\n    }\n    {\n        PyObject* err = PyErr_Occurred();\n        if (err == NULL) {\n            err = #modulename#_error;\n        }\n        PyErr_SetString(err,errmess);\n    }\n    return 0;\n}\n'
needs['long_double_from_pyobj'] = ['double_from_pyobj', 'long_double']
cfuncs['long_double_from_pyobj'] = 'static int\nlong_double_from_pyobj(long_double* v, PyObject *obj, const char *errmess)\n{\n    double d=0;\n    if (PyArray_CheckScalar(obj)){\n        if PyArray_IsScalar(obj, LongDouble) {\n            PyArray_ScalarAsCtype(obj, v);\n            return 1;\n        }\n        else if (PyArray_Check(obj) && PyArray_TYPE(obj) == NPY_LONGDOUBLE) {\n            (*v) = *((npy_longdouble *)PyArray_DATA(obj));\n            return 1;\n        }\n    }\n    if (double_from_pyobj(&d, obj, errmess)) {\n        *v = (long_double)d;\n        return 1;\n    }\n    return 0;\n}\n'
cfuncs['double_from_pyobj'] = 'static int\ndouble_from_pyobj(double* v, PyObject *obj, const char *errmess)\n{\n    PyObject* tmp = NULL;\n    if (PyFloat_Check(obj)) {\n        *v = PyFloat_AsDouble(obj);\n        return !(*v == -1.0 && PyErr_Occurred());\n    }\n\n    tmp = PyNumber_Float(obj);\n    if (tmp) {\n        *v = PyFloat_AsDouble(tmp);\n        Py_DECREF(tmp);\n        return !(*v == -1.0 && PyErr_Occurred());\n    }\n\n    if (PyComplex_Check(obj)) {\n        PyErr_Clear();\n        tmp = PyObject_GetAttrString(obj,"real");\n    }\n    else if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {\n        /*pass*/;\n    }\n    else if (PySequence_Check(obj)) {\n        PyErr_Clear();\n        tmp = PySequence_GetItem(obj, 0);\n    }\n\n    if (tmp) {\n        if (double_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}\n        Py_DECREF(tmp);\n    }\n    {\n        PyObject* err = PyErr_Occurred();\n        if (err==NULL) err = #modulename#_error;\n        PyErr_SetString(err,errmess);\n    }\n    return 0;\n}\n'
needs['float_from_pyobj'] = ['double_from_pyobj']
cfuncs['float_from_pyobj'] = 'static int\nfloat_from_pyobj(float* v, PyObject *obj, const char *errmess)\n{\n    double d=0.0;\n    if (double_from_pyobj(&d,obj,errmess)) {\n        *v = (float)d;\n        return 1;\n    }\n    return 0;\n}\n'
needs['complex_long_double_from_pyobj'] = ['complex_long_double', 'long_double', 'complex_double_from_pyobj', 'npy_math.h']
cfuncs['complex_long_double_from_pyobj'] = 'static int\ncomplex_long_double_from_pyobj(complex_long_double* v, PyObject *obj, const char *errmess)\n{\n    complex_double cd = {0.0,0.0};\n    if (PyArray_CheckScalar(obj)){\n        if PyArray_IsScalar(obj, CLongDouble) {\n            PyArray_ScalarAsCtype(obj, v);\n            return 1;\n        }\n        else if (PyArray_Check(obj) && PyArray_TYPE(obj)==NPY_CLONGDOUBLE) {\n            (*v).r = npy_creall(*(((npy_clongdouble *)PyArray_DATA(obj))));\n            (*v).i = npy_cimagl(*(((npy_clongdouble *)PyArray_DATA(obj))));\n            return 1;\n        }\n    }\n    if (complex_double_from_pyobj(&cd,obj,errmess)) {\n        (*v).r = (long_double)cd.r;\n        (*v).i = (long_double)cd.i;\n        return 1;\n    }\n    return 0;\n}\n'
needs['complex_double_from_pyobj'] = ['complex_double', 'npy_math.h']
cfuncs['complex_double_from_pyobj'] = 'static int\ncomplex_double_from_pyobj(complex_double* v, PyObject *obj, const char *errmess) {\n    Py_complex c;\n    if (PyComplex_Check(obj)) {\n        c = PyComplex_AsCComplex(obj);\n        (*v).r = c.real;\n        (*v).i = c.imag;\n        return 1;\n    }\n    if (PyArray_IsScalar(obj, ComplexFloating)) {\n        if (PyArray_IsScalar(obj, CFloat)) {\n            npy_cfloat new;\n            PyArray_ScalarAsCtype(obj, &new);\n            (*v).r = (double)npy_crealf(new);\n            (*v).i = (double)npy_cimagf(new);\n        }\n        else if (PyArray_IsScalar(obj, CLongDouble)) {\n            npy_clongdouble new;\n            PyArray_ScalarAsCtype(obj, &new);\n            (*v).r = (double)npy_creall(new);\n            (*v).i = (double)npy_cimagl(new);\n        }\n        else { /* if (PyArray_IsScalar(obj, CDouble)) */\n            PyArray_ScalarAsCtype(obj, v);\n        }\n        return 1;\n    }\n    if (PyArray_CheckScalar(obj)) { /* 0-dim array or still array scalar */\n        PyArrayObject *arr;\n        if (PyArray_Check(obj)) {\n            arr = (PyArrayObject *)PyArray_Cast((PyArrayObject *)obj, NPY_CDOUBLE);\n        }\n        else {\n            arr = (PyArrayObject *)PyArray_FromScalar(obj, PyArray_DescrFromType(NPY_CDOUBLE));\n        }\n        if (arr == NULL) {\n            return 0;\n        }\n        (*v).r = npy_creal(*(((npy_cdouble *)PyArray_DATA(arr))));\n        (*v).i = npy_cimag(*(((npy_cdouble *)PyArray_DATA(arr))));\n        Py_DECREF(arr);\n        return 1;\n    }\n    /* Python does not provide PyNumber_Complex function :-( */\n    (*v).i = 0.0;\n    if (PyFloat_Check(obj)) {\n        (*v).r = PyFloat_AsDouble(obj);\n        return !((*v).r == -1.0 && PyErr_Occurred());\n    }\n    if (PyLong_Check(obj)) {\n        (*v).r = PyLong_AsDouble(obj);\n        return !((*v).r == -1.0 && PyErr_Occurred());\n    }\n    if (PySequence_Check(obj) && !(PyBytes_Check(obj) || PyUnicode_Check(obj))) {\n        PyObject *tmp = PySequence_GetItem(obj,0);\n        if (tmp) {\n            if (complex_double_from_pyobj(v,tmp,errmess)) {\n                Py_DECREF(tmp);\n                return 1;\n            }\n            Py_DECREF(tmp);\n        }\n    }\n    {\n        PyObject* err = PyErr_Occurred();\n        if (err==NULL)\n            err = PyExc_TypeError;\n        PyErr_SetString(err,errmess);\n    }\n    return 0;\n}\n'
needs['complex_float_from_pyobj'] = ['complex_float', 'complex_double_from_pyobj']
cfuncs['complex_float_from_pyobj'] = 'static int\ncomplex_float_from_pyobj(complex_float* v,PyObject *obj,const char *errmess)\n{\n    complex_double cd={0.0,0.0};\n    if (complex_double_from_pyobj(&cd,obj,errmess)) {\n        (*v).r = (float)cd.r;\n        (*v).i = (float)cd.i;\n        return 1;\n    }\n    return 0;\n}\n'
cfuncs['try_pyarr_from_character'] = 'static int try_pyarr_from_character(PyObject* obj, character* v) {\n    PyArrayObject *arr = (PyArrayObject*)obj;\n    if (!obj) return -2;\n    if (PyArray_Check(obj)) {\n        if (F2PY_ARRAY_IS_CHARACTER_COMPATIBLE(arr))  {\n            *(character *)(PyArray_DATA(arr)) = *v;\n            return 1;\n        }\n    }\n    {\n        char mess[F2PY_MESSAGE_BUFFER_SIZE];\n        PyObject* err = PyErr_Occurred();\n        if (err == NULL) {\n            err = PyExc_ValueError;\n            strcpy(mess, "try_pyarr_from_character failed"\n                         " -- expected bytes array-scalar|array, got ");\n            f2py_describe(obj, mess + strlen(mess));\n            PyErr_SetString(err, mess);\n        }\n    }\n    return 0;\n}\n'
needs['try_pyarr_from_char'] = ['pyobj_from_char1', 'TRYPYARRAYTEMPLATE']
cfuncs['try_pyarr_from_char'] = "static int try_pyarr_from_char(PyObject* obj,char* v) {\n    TRYPYARRAYTEMPLATE(char,'c');\n}\n"
needs['try_pyarr_from_signed_char'] = ['TRYPYARRAYTEMPLATE', 'unsigned_char']
cfuncs['try_pyarr_from_unsigned_char'] = "static int try_pyarr_from_unsigned_char(PyObject* obj,unsigned_char* v) {\n    TRYPYARRAYTEMPLATE(unsigned_char,'b');\n}\n"
needs['try_pyarr_from_signed_char'] = ['TRYPYARRAYTEMPLATE', 'signed_char']
cfuncs['try_pyarr_from_signed_char'] = "static int try_pyarr_from_signed_char(PyObject* obj,signed_char* v) {\n    TRYPYARRAYTEMPLATE(signed_char,'1');\n}\n"
needs['try_pyarr_from_short'] = ['pyobj_from_short1', 'TRYPYARRAYTEMPLATE']
cfuncs['try_pyarr_from_short'] = "static int try_pyarr_from_short(PyObject* obj,short* v) {\n    TRYPYARRAYTEMPLATE(short,'s');\n}\n"
needs['try_pyarr_from_int'] = ['pyobj_from_int1', 'TRYPYARRAYTEMPLATE']
cfuncs['try_pyarr_from_int'] = "static int try_pyarr_from_int(PyObject* obj,int* v) {\n    TRYPYARRAYTEMPLATE(int,'i');\n}\n"
needs['try_pyarr_from_long'] = ['pyobj_from_long1', 'TRYPYARRAYTEMPLATE']
cfuncs['try_pyarr_from_long'] = "static int try_pyarr_from_long(PyObject* obj,long* v) {\n    TRYPYARRAYTEMPLATE(long,'l');\n}\n"
needs['try_pyarr_from_long_long'] = ['pyobj_from_long_long1', 'TRYPYARRAYTEMPLATE', 'long_long']
cfuncs['try_pyarr_from_long_long'] = "static int try_pyarr_from_long_long(PyObject* obj,long_long* v) {\n    TRYPYARRAYTEMPLATE(long_long,'L');\n}\n"
needs['try_pyarr_from_float'] = ['pyobj_from_float1', 'TRYPYARRAYTEMPLATE']
cfuncs['try_pyarr_from_float'] = "static int try_pyarr_from_float(PyObject* obj,float* v) {\n    TRYPYARRAYTEMPLATE(float,'f');\n}\n"
needs['try_pyarr_from_double'] = ['pyobj_from_double1', 'TRYPYARRAYTEMPLATE']
cfuncs['try_pyarr_from_double'] = "static int try_pyarr_from_double(PyObject* obj,double* v) {\n    TRYPYARRAYTEMPLATE(double,'d');\n}\n"
needs['try_pyarr_from_complex_float'] = ['pyobj_from_complex_float1', 'TRYCOMPLEXPYARRAYTEMPLATE', 'complex_float']
cfuncs['try_pyarr_from_complex_float'] = "static int try_pyarr_from_complex_float(PyObject* obj,complex_float* v) {\n    TRYCOMPLEXPYARRAYTEMPLATE(float,'F');\n}\n"
needs['try_pyarr_from_complex_double'] = ['pyobj_from_complex_double1', 'TRYCOMPLEXPYARRAYTEMPLATE', 'complex_double']
cfuncs['try_pyarr_from_complex_double'] = "static int try_pyarr_from_complex_double(PyObject* obj,complex_double* v) {\n    TRYCOMPLEXPYARRAYTEMPLATE(double,'D');\n}\n"
needs['create_cb_arglist'] = ['CFUNCSMESS', 'PRINTPYOBJERR', 'MINMAX']
cfuncs['create_cb_arglist'] = 'static int\ncreate_cb_arglist(PyObject* fun, PyTupleObject* xa , const int maxnofargs,\n                  const int nofoptargs, int *nofargs, PyTupleObject **args,\n                  const char *errmess)\n{\n    PyObject *tmp = NULL;\n    PyObject *tmp_fun = NULL;\n    Py_ssize_t tot, opt, ext, siz, i, di = 0;\n    CFUNCSMESS("create_cb_arglist\\n");\n    tot=opt=ext=siz=0;\n    /* Get the total number of arguments */\n    if (PyFunction_Check(fun)) {\n        tmp_fun = fun;\n        Py_INCREF(tmp_fun);\n    }\n    else {\n        di = 1;\n        if (PyObject_HasAttrString(fun,"im_func")) {\n            tmp_fun = PyObject_GetAttrString(fun,"im_func");\n        }\n        else if (PyObject_HasAttrString(fun,"__call__")) {\n            tmp = PyObject_GetAttrString(fun,"__call__");\n            if (PyObject_HasAttrString(tmp,"im_func"))\n                tmp_fun = PyObject_GetAttrString(tmp,"im_func");\n            else {\n                tmp_fun = fun; /* built-in function */\n                Py_INCREF(tmp_fun);\n                tot = maxnofargs;\n                if (PyCFunction_Check(fun)) {\n                    /* In case the function has a co_argcount (like on PyPy) */\n                    di = 0;\n                }\n                if (xa != NULL)\n                    tot += PyTuple_Size((PyObject *)xa);\n            }\n            Py_XDECREF(tmp);\n        }\n        else if (PyFortran_Check(fun) || PyFortran_Check1(fun)) {\n            tot = maxnofargs;\n            if (xa != NULL)\n                tot += PyTuple_Size((PyObject *)xa);\n            tmp_fun = fun;\n            Py_INCREF(tmp_fun);\n        }\n        else if (F2PyCapsule_Check(fun)) {\n            tot = maxnofargs;\n            if (xa != NULL)\n                ext = PyTuple_Size((PyObject *)xa);\n            if(ext>0) {\n                fprintf(stderr,"extra arguments tuple cannot be used with PyCapsule call-back\\n");\n                goto capi_fail;\n            }\n            tmp_fun = fun;\n            Py_INCREF(tmp_fun);\n        }\n    }\n\n    if (tmp_fun == NULL) {\n        fprintf(stderr,\n                "Call-back argument must be function|instance|instance.__call__|f2py-function "\n                "but got %s.\\n",\n                ((fun == NULL) ? "NULL" : Py_TYPE(fun)->tp_name));\n        goto capi_fail;\n    }\n\n    if (PyObject_HasAttrString(tmp_fun,"__code__")) {\n        if (PyObject_HasAttrString(tmp = PyObject_GetAttrString(tmp_fun,"__code__"),"co_argcount")) {\n            PyObject *tmp_argcount = PyObject_GetAttrString(tmp,"co_argcount");\n            Py_DECREF(tmp);\n            if (tmp_argcount == NULL) {\n                goto capi_fail;\n            }\n            tot = PyLong_AsSsize_t(tmp_argcount) - di;\n            Py_DECREF(tmp_argcount);\n        }\n    }\n    /* Get the number of optional arguments */\n    if (PyObject_HasAttrString(tmp_fun,"__defaults__")) {\n        if (PyTuple_Check(tmp = PyObject_GetAttrString(tmp_fun,"__defaults__")))\n            opt = PyTuple_Size(tmp);\n        Py_XDECREF(tmp);\n    }\n    /* Get the number of extra arguments */\n    if (xa != NULL)\n        ext = PyTuple_Size((PyObject *)xa);\n    /* Calculate the size of call-backs argument list */\n    siz = MIN(maxnofargs+ext,tot);\n    *nofargs = MAX(0,siz-ext);\n\n#ifdef DEBUGCFUNCS\n    fprintf(stderr,\n            "debug-capi:create_cb_arglist:maxnofargs(-nofoptargs),"\n            "tot,opt,ext,siz,nofargs = %d(-%d), %zd, %zd, %zd, %zd, %d\\n",\n            maxnofargs, nofoptargs, tot, opt, ext, siz, *nofargs);\n#endif\n\n    if (siz < tot-opt) {\n        fprintf(stderr,\n                "create_cb_arglist: Failed to build argument list "\n                "(siz) with enough arguments (tot-opt) required by "\n                "user-supplied function (siz,tot,opt=%zd, %zd, %zd).\\n",\n                siz, tot, opt);\n        goto capi_fail;\n    }\n\n    /* Initialize argument list */\n    *args = (PyTupleObject *)PyTuple_New(siz);\n    for (i=0;i<*nofargs;i++) {\n        Py_INCREF(Py_None);\n        PyTuple_SET_ITEM((PyObject *)(*args),i,Py_None);\n    }\n    if (xa != NULL)\n        for (i=(*nofargs);i<siz;i++) {\n            tmp = PyTuple_GetItem((PyObject *)xa,i-(*nofargs));\n            Py_INCREF(tmp);\n            PyTuple_SET_ITEM(*args,i,tmp);\n        }\n    CFUNCSMESS("create_cb_arglist-end\\n");\n    Py_DECREF(tmp_fun);\n    return 1;\n\ncapi_fail:\n    if (PyErr_Occurred() == NULL)\n        PyErr_SetString(#modulename#_error, errmess);\n    Py_XDECREF(tmp_fun);\n    return 0;\n}\n'

def buildcfuncs():
    if False:
        return 10
    from .capi_maps import c2capi_map
    for k in c2capi_map.keys():
        m = 'pyarr_from_p_%s1' % k
        cppmacros[m] = '#define %s(v) (PyArray_SimpleNewFromData(0,NULL,%s,(char *)v))' % (m, c2capi_map[k])
    k = 'string'
    m = 'pyarr_from_p_%s1' % k
    cppmacros[m] = '#define %s(v,dims) (PyArray_New(&PyArray_Type, 1, dims, NPY_STRING, NULL, v, 1, NPY_ARRAY_CARRAY, NULL))' % m

def append_needs(need, flag=1):
    if False:
        while True:
            i = 10
    if isinstance(need, list):
        for n in need:
            append_needs(n, flag)
    elif isinstance(need, str):
        if not need:
            return
        if need in includes0:
            n = 'includes0'
        elif need in includes:
            n = 'includes'
        elif need in typedefs:
            n = 'typedefs'
        elif need in typedefs_generated:
            n = 'typedefs_generated'
        elif need in cppmacros:
            n = 'cppmacros'
        elif need in cfuncs:
            n = 'cfuncs'
        elif need in callbacks:
            n = 'callbacks'
        elif need in f90modhooks:
            n = 'f90modhooks'
        elif need in commonhooks:
            n = 'commonhooks'
        else:
            errmess('append_needs: unknown need %s\n' % repr(need))
            return
        if need in outneeds[n]:
            return
        if flag:
            tmp = {}
            if need in needs:
                for nn in needs[need]:
                    t = append_needs(nn, 0)
                    if isinstance(t, dict):
                        for nnn in t.keys():
                            if nnn in tmp:
                                tmp[nnn] = tmp[nnn] + t[nnn]
                            else:
                                tmp[nnn] = t[nnn]
            for nn in tmp.keys():
                for nnn in tmp[nn]:
                    if nnn not in outneeds[nn]:
                        outneeds[nn] = [nnn] + outneeds[nn]
            outneeds[n].append(need)
        else:
            tmp = {}
            if need in needs:
                for nn in needs[need]:
                    t = append_needs(nn, flag)
                    if isinstance(t, dict):
                        for nnn in t.keys():
                            if nnn in tmp:
                                tmp[nnn] = t[nnn] + tmp[nnn]
                            else:
                                tmp[nnn] = t[nnn]
            if n not in tmp:
                tmp[n] = []
            tmp[n].append(need)
            return tmp
    else:
        errmess('append_needs: expected list or string but got :%s\n' % repr(need))

def get_needs():
    if False:
        for i in range(10):
            print('nop')
    res = {}
    for n in outneeds.keys():
        out = []
        saveout = copy.copy(outneeds[n])
        while len(outneeds[n]) > 0:
            if outneeds[n][0] not in needs:
                out.append(outneeds[n][0])
                del outneeds[n][0]
            else:
                flag = 0
                for k in outneeds[n][1:]:
                    if k in needs[outneeds[n][0]]:
                        flag = 1
                        break
                if flag:
                    outneeds[n] = outneeds[n][1:] + [outneeds[n][0]]
                else:
                    out.append(outneeds[n][0])
                    del outneeds[n][0]
            if saveout and 0 not in map(lambda x, y: x == y, saveout, outneeds[n]) and (outneeds[n] != []):
                print(n, saveout)
                errmess('get_needs: no progress in sorting needs, probably circular dependence, skipping.\n')
                out = out + saveout
                break
            saveout = copy.copy(outneeds[n])
        if out == []:
            out = [n]
        res[n] = out
    return res