import sys
import unittest
import warnings
from cinder import StrictModule

def run_static_tests():
    if False:
        while True:
            i = 10
    import test_compiler.test_static as test_static
    from test_compiler.test_static.common import StaticTestBase
    from test_compiler.test_static.compile import init_xxclassloader
    CODE_SAMPLES_IN_MODULE = []
    CODE_SAMPLES_IN_STRICT_MODULE = []
    CODE_SAMPLES_RUN = []

    class CompileCaptureOverrides:

        def _finalize_module(self, name, mod_dict=None):
            if False:
                print('Hello World!')
            pass

        def _in_module(self, *args):
            if False:
                return 10
            (d, m) = super()._in_module(*args)
            args = list(args)
            args[0] = d['__name__']
            CODE_SAMPLES_IN_MODULE.append(args)
            return (d, m)

        def _in_strict_module(self, *args):
            if False:
                return 10
            (d, m) = super()._in_strict_module(*args)
            args = list(args)
            args[0] = d['__name__']
            CODE_SAMPLES_IN_STRICT_MODULE.append(args)
            return (d, m)

        def _run_code(self, *args):
            if False:
                return 10
            (modname, r) = super()._run_code(*args)
            args = list(args)
            args[2] = modname
            CODE_SAMPLES_RUN.append(args)
            return (modname, r)

    class StaticCompilationTests(CompileCaptureOverrides, test_static.StaticCompilationTests):

        @classmethod
        def tearDownClass(cls):
            if False:
                return 10
            pass

    class StaticRuntimeTests(CompileCaptureOverrides, test_static.StaticRuntimeTests):
        pass
    suite = unittest.TestLoader().loadTestsFromTestCase(StaticCompilationTests)
    unittest.TextTestRunner().run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(StaticRuntimeTests)
    unittest.TextTestRunner().run(suite)
    print('Regenerate Static Python tests Python code')

    class StaticTestCodeRegenerator(StaticTestBase):

        def __init__(self):
            if False:
                print('Hello World!')
            init_xxclassloader()
            for args in CODE_SAMPLES_IN_MODULE:
                self._in_module(*args)
            for args in CODE_SAMPLES_IN_STRICT_MODULE:
                self._in_strict_module(*args)
            for args in CODE_SAMPLES_RUN:
                (_, d) = self._run_code(*args)
                sys.modules[args[2]] = d
    StaticTestCodeRegenerator()

def main():
    if False:
        return 10
    import cinderjit
    import test_cinderjit
    run_static_tests()
    cinderjit.multithreaded_compile_test()
if __name__ == '__main__':
    main()