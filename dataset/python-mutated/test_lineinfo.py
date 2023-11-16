from numba import cuda, float32, int32
from numba.core.errors import NumbaInvalidConfigWarning
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import ignore_internal_warnings
import re
import unittest
import warnings

@skip_on_cudasim('Simulator does not produce lineinfo')
class TestCudaLineInfo(CUDATestCase):

    def _loc_directive_regex(self):
        if False:
            print('Hello World!')
        pat = '\\.loc\\s+[0-9]+\\s+[0-9]+\\s+[0-9]+'
        return re.compile(pat)

    def _check(self, fn, sig, expect):
        if False:
            for i in range(10):
                print('nop')
        fn.compile(sig)
        llvm = fn.inspect_llvm(sig)
        ptx = fn.inspect_asm(sig)
        assertfn = self.assertIsNotNone if expect else self.assertIsNone
        pat = '!DICompileUnit\\(.*emissionKind:\\s+DebugDirectivesOnly'
        match = re.compile(pat).search(llvm)
        assertfn(match, msg=ptx)
        pat = '!DICompileUnit\\(.*emissionKind:\\s+FullDebug'
        match = re.compile(pat).search(llvm)
        self.assertIsNone(match, msg=ptx)
        pat = '\\.file\\s+[0-9]+\\s+".*test_lineinfo.py"'
        match = re.compile(pat).search(ptx)
        assertfn(match, msg=ptx)
        self._loc_directive_regex().search(ptx)
        assertfn(match, msg=ptx)
        pat = '\\.section\\s+\\.debug_info'
        match = re.compile(pat).search(ptx)
        self.assertIsNone(match, msg=ptx)

    def test_no_lineinfo_in_asm(self):
        if False:
            for i in range(10):
                print('nop')

        @cuda.jit(lineinfo=False)
        def foo(x):
            if False:
                while True:
                    i = 10
            x[0] = 1
        self._check(foo, sig=(int32[:],), expect=False)

    def test_lineinfo_in_asm(self):
        if False:
            i = 10
            return i + 15

        @cuda.jit(lineinfo=True)
        def foo(x):
            if False:
                print('Hello World!')
            x[0] = 1
        self._check(foo, sig=(int32[:],), expect=True)

    def test_lineinfo_maintains_error_model(self):
        if False:
            return 10
        sig = (float32[::1], float32[::1])

        @cuda.jit(sig, lineinfo=True)
        def divide_kernel(x, y):
            if False:
                i = 10
                return i + 15
            x[0] /= y[0]
        llvm = divide_kernel.inspect_llvm(sig)
        self.assertNotIn('ret i32 1', llvm)

    def test_no_lineinfo_in_device_function(self):
        if False:
            return 10

        @cuda.jit
        def callee(x):
            if False:
                print('Hello World!')
            x[0] += 1

        @cuda.jit
        def caller(x):
            if False:
                i = 10
                return i + 15
            x[0] = 1
            callee(x)
        sig = (int32[:],)
        self._check(caller, sig=sig, expect=False)

    def test_lineinfo_in_device_function(self):
        if False:
            print('Hello World!')

        @cuda.jit(lineinfo=True)
        def callee(x):
            if False:
                while True:
                    i = 10
            x[0] += 1

        @cuda.jit(lineinfo=True)
        def caller(x):
            if False:
                print('Hello World!')
            x[0] = 1
            callee(x)
        sig = (int32[:],)
        self._check(caller, sig=sig, expect=True)
        ptx = caller.inspect_asm(sig)
        ptxlines = ptx.splitlines()
        devfn_start = re.compile('^\\.weak\\s+\\.func')
        for line in ptxlines:
            if devfn_start.match(line) is not None:
                self.fail(f'Found device function in PTX:\n\n{ptx}')
        loc_directive = self._loc_directive_regex()
        found = False
        for line in ptxlines:
            if loc_directive.search(line) is not None:
                if 'inlined_at' in line:
                    found = True
                    break
        if not found:
            self.fail(f'No .loc directive with inlined_at info foundin:\n\n{ptx}')
        llvm = caller.inspect_llvm(sig)
        subprograms = 0
        for line in llvm.splitlines():
            if 'distinct !DISubprogram' in line:
                subprograms += 1
        expected_subprograms = 3
        self.assertEqual(subprograms, expected_subprograms, f'"Expected {expected_subprograms} DISubprograms; got {subprograms}')

    def test_debug_and_lineinfo_warning(self):
        if False:
            for i in range(10):
                print('nop')
        with warnings.catch_warnings(record=True) as w:
            ignore_internal_warnings()

            @cuda.jit(debug=True, lineinfo=True, opt=False)
            def f():
                if False:
                    i = 10
                    return i + 15
                pass
        self.assertEqual(len(w), 1)
        self.assertEqual(w[0].category, NumbaInvalidConfigWarning)
        self.assertIn('debug and lineinfo are mutually exclusive', str(w[0].message))
if __name__ == '__main__':
    unittest.main()