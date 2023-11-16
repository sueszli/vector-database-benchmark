import math
import numpy as np
import subprocess
import numbers
import importlib
import sys
import re
import traceback
import multiprocessing as mp
from itertools import chain, combinations
import numba
from numba.core import config, cpu
from numba import prange, njit
from numba.core.compiler import compile_isolated, Flags
from numba.tests.support import TestCase, tag, override_env_config
import unittest
needs_svml = unittest.skipUnless(config.USING_SVML, 'SVML tests need SVML to be present')
vlen2cpu = {2: 'nehalem', 4: 'haswell', 8: 'skylake-avx512'}
vlen2cpu_features = {2: '', 4: '', 8: '-prefer-256-bit'}
svml_funcs = {'sin': [np.sin, math.sin], 'cos': [np.cos, math.cos], 'pow': [], 'exp': [np.exp, math.exp], 'log': [np.log, math.log], 'acos': [math.acos], 'acosh': [math.acosh], 'asin': [math.asin], 'asinh': [math.asinh], 'atan2': [], 'atan': [math.atan], 'atanh': [math.atanh], 'cbrt': [], 'cdfnorm': [], 'cdfnorminv': [], 'ceil': [], 'cosd': [], 'cosh': [np.cosh, math.cosh], 'erf': [math.erf], 'erfc': [math.erfc], 'erfcinv': [], 'erfinv': [], 'exp10': [], 'exp2': [], 'expm1': [np.expm1, math.expm1], 'floor': [], 'fmod': [], 'hypot': [], 'invsqrt': [], 'log10': [np.log10, math.log10], 'log1p': [np.log1p, math.log1p], 'log2': [], 'logb': [], 'nearbyint': [], 'rint': [], 'round': [], 'sind': [], 'sinh': [np.sinh, math.sinh], 'tan': [np.tan, math.tan], 'tanh': [np.tanh, math.tanh], 'trunc': []}
complex_funcs_exclude = ['tan', 'log10', 'expm1', 'log1p', 'tanh', 'log']
svml_funcs = {k: v for (k, v) in svml_funcs.items() if len(v) > 0}
numpy_funcs = [f for (f, v) in svml_funcs.items() if '<ufunc' in [str(p).split(' ')[0] for p in v]]
other_funcs = [f for (f, v) in svml_funcs.items() if '<built-in' in [str(p).split(' ')[0] for p in v]]

def func_patterns(func, args, res, dtype, mode, vlen, fastmath, pad=' ' * 8):
    if False:
        while True:
            i = 10
    '\n    For a given function and its usage modes,\n    returns python code and assembly patterns it should and should not generate\n    '
    if mode == 'scalar':
        arg_list = ','.join([a + '[0]' for a in args])
        body = '%s%s[0] += math.%s(%s)\n' % (pad, res, func, arg_list)
    elif mode == 'numpy':
        body = '%s%s += np.%s(%s)' % (pad, res, func, ','.join(args))
        body += '.astype(np.%s)\n' % dtype if dtype.startswith('int') else '\n'
    else:
        assert mode == 'range' or mode == 'prange'
        arg_list = ','.join([a + '[i]' for a in args])
        body = '{pad}for i in {mode}({res}.size):\n{pad}{pad}{res}[i] += math.{func}({arg_list})\n'.format(**locals())
    is_f32 = dtype == 'float32' or dtype == 'complex64'
    f = func + 'f' if is_f32 else func
    v = vlen * 2 if is_f32 else vlen
    prec_suff = '' if fastmath else '_ha'
    scalar_func = '$_' + f if config.IS_OSX else '$' + f
    svml_func = '__svml_%s%d%s,' % (f, v, prec_suff)
    if mode == 'scalar':
        contains = [scalar_func]
        avoids = ['__svml_', svml_func]
    else:
        contains = [svml_func]
        avoids = []
        if vlen != 8 and (is_f32 or dtype == 'int32'):
            avoids += ['%zmm', '__svml_%s%d%s,' % (f, v * 2, prec_suff)]
    return (body, contains, avoids)

def usecase_name(dtype, mode, vlen, name):
    if False:
        while True:
            i = 10
    ' Returns pretty name for given set of modes '
    return f'{dtype}_{mode}{vlen}_{name}'

def combo_svml_usecase(dtype, mode, vlen, fastmath, name):
    if False:
        for i in range(10):
            print('nop')
    ' Combine multiple function calls under single umbrella usecase '
    name = usecase_name(dtype, mode, vlen, name)
    body = 'def {name}(n):\n        x   = np.empty(n*8, dtype=np.{dtype})\n        ret = np.empty_like(x)\n'.format(**locals())
    funcs = set(numpy_funcs if mode == 'numpy' else other_funcs)
    if dtype.startswith('complex'):
        funcs = funcs.difference(complex_funcs_exclude)
    contains = set()
    avoids = set()
    for f in funcs:
        (b, c, a) = func_patterns(f, ['x'], 'ret', dtype, mode, vlen, fastmath)
        avoids.update(a)
        body += b
        contains.update(c)
    body += ' ' * 8 + 'return ret'
    ldict = {}
    exec(body, globals(), ldict)
    ldict[name].__doc__ = body
    return (ldict[name], contains, avoids)

@needs_svml
class TestSVMLGeneration(TestCase):
    """ Tests all SVML-generating functions produce desired calls """
    _numba_parallel_test_ = False
    asm_filter = re.compile('|'.join(['\\$[a-z_]\\w+,'] + list(svml_funcs)))

    @classmethod
    def mp_runner(cls, testname, outqueue):
        if False:
            print('Hello World!')
        method = getattr(cls, testname)
        try:
            (ok, msg) = method()
        except Exception:
            msg = traceback.format_exc()
            ok = False
        outqueue.put({'status': ok, 'msg': msg})

    @classmethod
    def _inject_test(cls, dtype, mode, vlen, flags):
        if False:
            return 10
        if dtype.startswith('complex') and mode != 'numpy':
            return
        skipped = dtype.startswith('int') and vlen == 2
        sig = (numba.int64,)

        @staticmethod
        def run_template():
            if False:
                print('Hello World!')
            (fn, contains, avoids) = combo_svml_usecase(dtype, mode, vlen, flags['fastmath'], flags['name'])
            with override_env_config('NUMBA_CPU_NAME', vlen2cpu[vlen]), override_env_config('NUMBA_CPU_FEATURES', vlen2cpu_features[vlen]):
                try:
                    jitted_fn = njit(sig, fastmath=flags['fastmath'], error_model=flags['error_model'])(fn)
                except:
                    raise Exception('raised while compiling ' + fn.__doc__)
            asm = jitted_fn.inspect_asm(sig)
            missed = [pattern for pattern in contains if not pattern in asm]
            found = [pattern for pattern in avoids if pattern in asm]
            ok = not missed and (not found)
            detail = '\n'.join([line for line in asm.split('\n') if cls.asm_filter.search(line) and (not '"' in line)])
            msg = f'While expecting {missed} and not {found},\nit contains:\n{detail}\nwhen compiling {fn.__doc__}'
            return (ok, msg)
        postfix = usecase_name(dtype, mode, vlen, flags['name'])
        testname = f'run_{postfix}'
        setattr(cls, testname, run_template)

        @unittest.skipUnless(not skipped, 'Not implemented')
        def test_runner(self):
            if False:
                return 10
            ctx = mp.get_context('spawn')
            q = ctx.Queue()
            p = ctx.Process(target=type(self).mp_runner, args=[testname, q])
            p.start()
            term_or_timeout = p.join(timeout=30)
            exitcode = p.exitcode
            if term_or_timeout is None:
                if exitcode is None:
                    self.fail('Process timed out.')
                elif exitcode < 0:
                    self.fail(f'Process terminated with signal {-exitcode}.')
            self.assertEqual(exitcode, 0, msg='process ended unexpectedly')
            out = q.get()
            status = out['status']
            msg = out['msg']
            self.assertTrue(status, msg=msg)
        setattr(cls, f'test_{postfix}', test_runner)

    @classmethod
    def autogenerate(cls):
        if False:
            i = 10
            return i + 15
        flag_list = [{'fastmath': False, 'error_model': 'numpy', 'name': 'usecase'}, {'fastmath': True, 'error_model': 'numpy', 'name': 'fastmath_usecase'}]
        for dtype in ('complex64', 'float64', 'float32', 'int32'):
            for vlen in vlen2cpu:
                for flags in flag_list:
                    for mode in ('scalar', 'range', 'prange', 'numpy'):
                        cls._inject_test(dtype, mode, vlen, dict(flags))
        for n in ('test_int32_range4_usecase',):
            setattr(cls, n, tag('important')(getattr(cls, n)))
TestSVMLGeneration.autogenerate()

def math_sin_scalar(x):
    if False:
        print('Hello World!')
    return math.sin(x)

def math_sin_loop(n):
    if False:
        print('Hello World!')
    ret = np.empty(n, dtype=np.float64)
    for x in range(n):
        ret[x] = math.sin(np.float64(x))
    return ret

@needs_svml
class TestSVML(TestCase):
    """ Tests SVML behaves as expected """
    _numba_parallel_test_ = False

    def __init__(self, *args):
        if False:
            return 10
        self.flags = Flags()
        self.flags.nrt = True
        self.fastflags = Flags()
        self.fastflags.nrt = True
        self.fastflags.fastmath = cpu.FastMathOptions(True)
        super(TestSVML, self).__init__(*args)

    def compile(self, func, *args, **kwargs):
        if False:
            return 10
        assert not kwargs
        sig = tuple([numba.typeof(x) for x in args])
        std = compile_isolated(func, sig, flags=self.flags)
        fast = compile_isolated(func, sig, flags=self.fastflags)
        return (std, fast)

    def copy_args(self, *args):
        if False:
            i = 10
            return i + 15
        if not args:
            return tuple()
        new_args = []
        for x in args:
            if isinstance(x, np.ndarray):
                new_args.append(x.copy('k'))
            elif isinstance(x, np.number):
                new_args.append(x.copy())
            elif isinstance(x, numbers.Number):
                new_args.append(x)
            else:
                raise ValueError('Unsupported argument type encountered')
        return tuple(new_args)

    def check(self, pyfunc, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        (jitstd, jitfast) = self.compile(pyfunc, *args)
        std_pattern = kwargs.pop('std_pattern', None)
        fast_pattern = kwargs.pop('fast_pattern', None)
        cpu_name = kwargs.pop('cpu_name', 'skylake-avx512')
        cpu_features = kwargs.pop('cpu_features', '-prefer-256-bit')
        py_expected = pyfunc(*self.copy_args(*args))
        jitstd_result = jitstd.entry_point(*self.copy_args(*args))
        jitfast_result = jitfast.entry_point(*self.copy_args(*args))
        np.testing.assert_almost_equal(jitstd_result, py_expected, **kwargs)
        np.testing.assert_almost_equal(jitfast_result, py_expected, **kwargs)
        with override_env_config('NUMBA_CPU_NAME', cpu_name), override_env_config('NUMBA_CPU_FEATURES', cpu_features):
            (jitstd, jitfast) = self.compile(pyfunc, *args)
            if std_pattern:
                self.check_svml_presence(jitstd, std_pattern)
            if fast_pattern:
                self.check_svml_presence(jitfast, fast_pattern)

    def check_svml_presence(self, func, pattern):
        if False:
            i = 10
            return i + 15
        asm = func.library.get_asm_str()
        self.assertIn(pattern, asm)

    def test_scalar_context(self):
        if False:
            return 10
        pat = '$_sin' if config.IS_OSX else '$sin'
        self.check(math_sin_scalar, 7.0, std_pattern=pat)
        self.check(math_sin_scalar, 7.0, fast_pattern=pat)

    def test_svml(self):
        if False:
            return 10
        std = '__svml_sin8_ha,'
        fast = '__svml_sin8,'
        self.check(math_sin_loop, 10, std_pattern=std, fast_pattern=fast)

    def test_svml_disabled(self):
        if False:
            print('Hello World!')
        code = "if 1:\n            import os\n            import numpy as np\n            import math\n\n            def math_sin_loop(n):\n                ret = np.empty(n, dtype=np.float64)\n                for x in range(n):\n                    ret[x] = math.sin(np.float64(x))\n                return ret\n\n            def check_no_svml():\n                try:\n                    # ban the use of SVML\n                    os.environ['NUMBA_DISABLE_INTEL_SVML'] = '1'\n\n                    # delay numba imports to account for env change as\n                    # numba.__init__ picks up SVML and it is too late by\n                    # then to override using `numba.config`\n                    import numba\n                    from numba import config\n                    from numba.core import cpu\n                    from numba.tests.support import override_env_config\n                    from numba.core.compiler import compile_isolated, Flags\n\n                    # compile for overridden CPU, with and without fastmath\n                    with override_env_config('NUMBA_CPU_NAME', 'skylake-avx512'),                          override_env_config('NUMBA_CPU_FEATURES', ''):\n                        sig = (numba.int32,)\n                        f = Flags()\n                        f.nrt = True\n                        std = compile_isolated(math_sin_loop, sig, flags=f)\n                        f.fastmath = cpu.FastMathOptions(True)\n                        fast = compile_isolated(math_sin_loop, sig, flags=f)\n                        fns = std, fast\n\n                        # assert no SVML call is present in the asm\n                        for fn in fns:\n                            asm = fn.library.get_asm_str()\n                            assert '__svml_sin' not in asm\n                finally:\n                    # not really needed as process is separate\n                    os.environ['NUMBA_DISABLE_INTEL_SVML'] = '0'\n                    config.reload_config()\n            check_no_svml()\n            "
        popen = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out, err) = popen.communicate()
        if popen.returncode != 0:
            raise AssertionError('process failed with code %s: stderr follows\n%s\n' % (popen.returncode, err.decode()))

    def test_svml_working_in_non_isolated_context(self):
        if False:
            i = 10
            return i + 15

        @njit(fastmath={'fast'}, error_model='numpy')
        def impl(n):
            if False:
                for i in range(10):
                    print('nop')
            x = np.empty(n * 8, dtype=np.float64)
            ret = np.empty_like(x)
            for i in range(ret.size):
                ret[i] += math.cosh(x[i])
            return ret
        impl(1)
        self.assertTrue('intel_svmlcc' in impl.inspect_llvm(impl.signatures[0]))
if __name__ == '__main__':
    unittest.main()