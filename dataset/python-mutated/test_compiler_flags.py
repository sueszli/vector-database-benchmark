import re
from numba import njit
from numba.core.extending import overload
from numba.core.targetconfig import ConfigStack
from numba.core.compiler import Flags, DEFAULT_FLAGS
from numba.core import types
from numba.core.funcdesc import default_mangler
from numba.tests.support import TestCase, unittest

class TestCompilerFlagCachedOverload(TestCase):

    def test_fastmath_in_overload(self):
        if False:
            return 10

        def fastmath_status():
            if False:
                i = 10
                return i + 15
            pass

        @overload(fastmath_status)
        def ov_fastmath_status():
            if False:
                while True:
                    i = 10
            flags = ConfigStack().top()
            val = 'Has fastmath' if flags.fastmath else 'No fastmath'

            def codegen():
                if False:
                    for i in range(10):
                        print('nop')
                return val
            return codegen

        @njit(fastmath=True)
        def set_fastmath():
            if False:
                print('Hello World!')
            return fastmath_status()

        @njit()
        def foo():
            if False:
                return 10
            a = fastmath_status()
            b = set_fastmath()
            return (a, b)
        (a, b) = foo()
        self.assertEqual(a, 'No fastmath')
        self.assertEqual(b, 'Has fastmath')

class TestFlagMangling(TestCase):

    def test_demangle(self):
        if False:
            return 10

        def check(flags):
            if False:
                print('Hello World!')
            mangled = flags.get_mangle_string()
            out = flags.demangle(mangled)
            self.assertEqual(out, flags.summary())
        flags = Flags()
        check(flags)
        check(DEFAULT_FLAGS)
        flags = Flags()
        flags.no_cpython_wrapper = True
        flags.nrt = True
        flags.fastmath = True
        check(flags)

    def test_mangled_flags_is_shorter(self):
        if False:
            while True:
                i = 10
        flags = Flags()
        flags.nrt = True
        flags.auto_parallel = True
        self.assertLess(len(flags.get_mangle_string()), len(flags.summary()))

    def test_mangled_flags_with_fastmath_parfors_inline(self):
        if False:
            i = 10
            return i + 15
        flags = Flags()
        flags.nrt = True
        flags.auto_parallel = True
        flags.fastmath = True
        flags.inline = 'always'
        self.assertLess(len(flags.get_mangle_string()), len(flags.summary()))
        demangled = flags.demangle(flags.get_mangle_string())
        self.assertNotIn('0x', demangled)

    def test_demangling_from_mangled_symbols(self):
        if False:
            print('Hello World!')
        'Test demangling of flags from mangled symbol'
        fname = 'foo'
        argtypes = (types.int32,)
        flags = Flags()
        flags.nrt = True
        flags.target_backend = 'myhardware'
        name = default_mangler(fname, argtypes, abi_tags=[flags.get_mangle_string()])
        prefix = '_Z3fooB'
        m = re.match('[0-9]+', name[len(prefix):])
        size = m.group(0)
        base = len(prefix) + len(size)
        abi_mangled = name[base:base + int(size)]
        demangled = Flags.demangle(abi_mangled)
        self.assertEqual(demangled, flags.summary())
if __name__ == '__main__':
    unittest.main()