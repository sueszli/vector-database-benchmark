import numpy as np
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba import cuda
from numba.cuda import libdevice, compile_ptx
from numba.cuda.libdevicefuncs import functions, create_signature

def use_sincos(s, c, x):
    if False:
        for i in range(10):
            print('nop')
    i = cuda.grid(1)
    if i < len(x):
        (sr, cr) = libdevice.sincos(x[i])
        s[i] = sr
        c[i] = cr

def use_frexp(frac, exp, x):
    if False:
        while True:
            i = 10
    i = cuda.grid(1)
    if i < len(x):
        (fracr, expr) = libdevice.frexp(x[i])
        frac[i] = fracr
        exp[i] = expr

def use_sad(r, x, y, z):
    if False:
        print('Hello World!')
    i = cuda.grid(1)
    if i < len(x):
        r[i] = libdevice.sad(x[i], y[i], z[i])

@skip_on_cudasim('Libdevice functions are not supported on cudasim')
class TestLibdevice(CUDATestCase):
    """
    Some tests of libdevice function wrappers that check the returned values.

    These are mainly to check that the generation of the implementations
    results in correct typing and lowering for each type of function return
    (e.g. scalar return, UniTuple return, Tuple return, etc.).
    """

    def test_sincos(self):
        if False:
            print('Hello World!')
        arr = np.arange(100, dtype=np.float64)
        sres = np.zeros_like(arr)
        cres = np.zeros_like(arr)
        cufunc = cuda.jit(use_sincos)
        cufunc[4, 32](sres, cres, arr)
        np.testing.assert_allclose(np.cos(arr), cres)
        np.testing.assert_allclose(np.sin(arr), sres)

    def test_frexp(self):
        if False:
            i = 10
            return i + 15
        arr = np.linspace(start=1.0, stop=10.0, num=100, dtype=np.float64)
        fracres = np.zeros_like(arr)
        expres = np.zeros(shape=arr.shape, dtype=np.int32)
        cufunc = cuda.jit(use_frexp)
        cufunc[4, 32](fracres, expres, arr)
        (frac_expect, exp_expect) = np.frexp(arr)
        np.testing.assert_array_equal(frac_expect, fracres)
        np.testing.assert_array_equal(exp_expect, expres)

    def test_sad(self):
        if False:
            i = 10
            return i + 15
        x = np.arange(0, 200, 2)
        y = np.arange(50, 150)
        z = np.arange(15, 115)
        r = np.zeros_like(x)
        cufunc = cuda.jit(use_sad)
        cufunc[4, 32](r, x, y, z)
        np.testing.assert_array_equal(np.abs(x - y) + z, r)
function_template = 'from numba.cuda import libdevice\n\ndef pyfunc(%(pyargs)s):\n    ret = libdevice.%(func)s(%(funcargs)s)\n    %(retvars)s = ret\n'

def make_test_call(libname):
    if False:
        return 10
    '\n    Generates a test function for each libdevice function.\n    '

    def _test_call_functions(self):
        if False:
            i = 10
            return i + 15
        apiname = libname[5:]
        apifunc = getattr(libdevice, apiname)
        (retty, args) = functions[libname]
        sig = create_signature(retty, args)
        funcargs = ', '.join(['a%d' % i for (i, arg) in enumerate(args) if not arg.is_ptr])
        if isinstance(sig.return_type, (types.Tuple, types.UniTuple)):
            pyargs = ', '.join(['r%d' % i for i in range(len(sig.return_type))])
            pyargs += ', ' + funcargs
            retvars = ', '.join(['r%d[0]' % i for i in range(len(sig.return_type))])
        else:
            pyargs = 'r0, ' + funcargs
            retvars = 'r0[0]'
        d = {'func': apiname, 'pyargs': pyargs, 'funcargs': funcargs, 'retvars': retvars}
        code = function_template % d
        locals = {}
        exec(code, globals(), locals)
        pyfunc = locals['pyfunc']
        pyargs = [arg.ty for arg in args if not arg.is_ptr]
        if isinstance(sig.return_type, (types.Tuple, types.UniTuple)):
            pyreturns = [ret[::1] for ret in sig.return_type]
            pyargs = pyreturns + pyargs
        else:
            pyargs.insert(0, sig.return_type[::1])
        pyargs = tuple(pyargs)
        (ptx, resty) = compile_ptx(pyfunc, pyargs)
        self.assertIn('ld.param', ptx)
        self.assertIn('st.global', ptx)
    return _test_call_functions

@skip_on_cudasim('Compilation to PTX is not supported on cudasim')
class TestLibdeviceCompilation(unittest.TestCase):
    """
    Class for holding all tests of compiling calls to libdevice functions. We
    generate the actual tests in this class (as opposed to using subTest and
    one test within this class) because there are a lot of tests, and it makes
    the test suite appear frozen to test them all as subTests in one test.
    """
for libname in functions:
    setattr(TestLibdeviceCompilation, 'test_%s' % libname, make_test_call(libname))
if __name__ == '__main__':
    unittest.main()