import unittest
from numba.tests.support import captured_stdout, override_config

class DocsLLVMPassTimings(unittest.TestCase):

    def test_pass_timings(self):
        if False:
            for i in range(10):
                print('nop')
        with override_config('LLVM_PASS_TIMINGS', True):
            with captured_stdout() as stdout:
                import numba

                @numba.njit
                def foo(n):
                    if False:
                        while True:
                            i = 10
                    c = 0
                    for i in range(n):
                        for j in range(i):
                            c += j
                    return c
                foo(10)
                md = foo.get_metadata(foo.signatures[0])
                print(md['llvm_pass_timings'])
            self.assertIn('Finalize object', stdout.getvalue())
if __name__ == '__main__':
    unittest.main()