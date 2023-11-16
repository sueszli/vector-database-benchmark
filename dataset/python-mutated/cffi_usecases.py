import sys
import numpy as np
import numba.core.typing.cffi_utils as cffi_support
from numba.tests.support import import_dynamic, temp_directory
from numba.core.types import complex128

def load_inline_module():
    if False:
        return 10
    '\n    Create an inline module, return the corresponding ffi and dll objects.\n    '
    from cffi import FFI
    defs = '\n    double _numba_test_sin(double x);\n    double _numba_test_cos(double x);\n    double _numba_test_funcptr(double (*func)(double));\n    bool _numba_test_boolean(void);\n    '
    ffi = FFI()
    ffi.cdef(defs)
    from numba import _helperlib
    return (ffi, ffi.dlopen(_helperlib.__file__))

def load_ool_module():
    if False:
        return 10
    '\n    Compile an out-of-line module, return the corresponding ffi and\n    module objects.\n    '
    from cffi import FFI
    numba_complex = '\n    typedef struct _numba_complex {\n        double real;\n        double imag;\n    } numba_complex;\n    '
    bool_define = '\n    #ifdef _MSC_VER\n        #define false 0\n        #define true 1\n        #define bool int\n    #else\n        #include <stdbool.h>\n    #endif\n    '
    defs = numba_complex + '\n    bool boolean(void);\n    double sin(double x);\n    double cos(double x);\n    int foo(int a, int b, int c);\n    void vsSin(int n, float* x, float* y);\n    void vdSin(int n, double* x, double* y);\n    void vector_real(numba_complex *c, double *real, int n);\n    void vector_imag(numba_complex *c, double *imag, int n);\n    '
    source = numba_complex + bool_define + '\n    static bool boolean(void)\n    {\n        return true;\n    }\n\n    static int foo(int a, int b, int c)\n    {\n        return a + b * c;\n    }\n\n    void vsSin(int n, float* x, float* y) {\n        int i;\n        for (i=0; i<n; i++)\n            y[i] = sin(x[i]);\n    }\n\n    void vdSin(int n, double* x, double* y) {\n        int i;\n        for (i=0; i<n; i++)\n            y[i] = sin(x[i]);\n    }\n\n    static void vector_real(numba_complex *c, double *real, int n) {\n        int i;\n        for (i = 0; i < n; i++)\n            real[i] = c[i].real;\n    }\n\n    static void vector_imag(numba_complex *c, double *imag, int n) {\n        int i;\n        for (i = 0; i < n; i++)\n            imag[i] = c[i].imag;\n    }\n    '
    ffi = FFI()
    ffi.set_source('cffi_usecases_ool', source)
    ffi.cdef(defs, override=True)
    tmpdir = temp_directory('test_cffi')
    ffi.compile(tmpdir=tmpdir)
    sys.path.append(tmpdir)
    try:
        mod = import_dynamic('cffi_usecases_ool')
        cffi_support.register_module(mod)
        cffi_support.register_type(mod.ffi.typeof('struct _numba_complex'), complex128)
        return (mod.ffi, mod)
    finally:
        sys.path.remove(tmpdir)

def init():
    if False:
        print('Hello World!')
    '\n    Initialize module globals.  This can invoke external utilities, hence not\n    being executed implicitly at module import.\n    '
    global ffi, cffi_sin, cffi_cos, cffi_bool
    if ffi is None:
        (ffi, dll) = load_inline_module()
        cffi_sin = dll._numba_test_sin
        cffi_cos = dll._numba_test_cos
        cffi_bool = dll._numba_test_boolean
        del dll

def init_ool():
    if False:
        for i in range(10):
            print('nop')
    '\n    Same as init() for OOL mode.\n    '
    global ffi_ool, cffi_sin_ool, cffi_cos_ool, cffi_foo, cffi_bool_ool
    global vsSin, vdSin, vector_real, vector_imag
    if ffi_ool is None:
        (ffi_ool, mod) = load_ool_module()
        cffi_sin_ool = mod.lib.sin
        cffi_cos_ool = mod.lib.cos
        cffi_foo = mod.lib.foo
        cffi_bool_ool = mod.lib.boolean
        vsSin = mod.lib.vsSin
        vdSin = mod.lib.vdSin
        vector_real = mod.lib.vector_real
        vector_imag = mod.lib.vector_imag
        del mod
ffi = ffi_ool = None

def use_cffi_sin(x):
    if False:
        return 10
    return cffi_sin(x) * 2

def use_two_funcs(x):
    if False:
        return 10
    return cffi_sin(x) - cffi_cos(x)

def use_cffi_sin_ool(x):
    if False:
        while True:
            i = 10
    return cffi_sin_ool(x) * 2

def use_cffi_boolean_true():
    if False:
        while True:
            i = 10
    return cffi_bool_ool()

def use_two_funcs_ool(x):
    if False:
        for i in range(10):
            print('nop')
    return cffi_sin_ool(x) - cffi_cos_ool(x)

def use_func_pointer(fa, fb, x):
    if False:
        i = 10
        return i + 15
    if x > 0:
        return fa(x)
    else:
        return fb(x)

def use_user_defined_symbols():
    if False:
        for i in range(10):
            print('nop')
    return cffi_foo(1, 2, 3)

def vector_sin_float32(x, y):
    if False:
        print('Hello World!')
    vsSin(len(x), ffi.from_buffer(x), ffi_ool.from_buffer(y))

def vector_sin_float64(x, y):
    if False:
        while True:
            i = 10
    vdSin(len(x), ffi.from_buffer(x), ffi_ool.from_buffer(y))

def vector_extract_real(x, y):
    if False:
        i = 10
        return i + 15
    vector_real(ffi.from_buffer(x), ffi.from_buffer(y), len(x))

def vector_extract_imag(x, y):
    if False:
        return 10
    vector_imag(ffi.from_buffer(x), ffi.from_buffer(y), len(x))