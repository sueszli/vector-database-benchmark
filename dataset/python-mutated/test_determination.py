import os
import run_test
from torch.testing._internal.common_utils import TestCase, run_tests

class DummyOptions:
    verbose = False

class DeterminationTest(TestCase):
    TESTS = ['test_nn', 'test_jit_profiling', 'test_jit', 'test_torch', 'test_cpp_extensions_aot_ninja', 'test_cpp_extensions_aot_no_ninja', 'test_utils', 'test_determination', 'test_quantization']

    @classmethod
    def determined_tests(cls, changed_files):
        if False:
            print('Hello World!')
        changed_files = [os.path.normpath(path) for path in changed_files]
        return [test for test in cls.TESTS if run_test.should_run_test(run_test.TARGET_DET_LIST, test, changed_files, DummyOptions())]

    def test_target_det_list_is_sorted(self):
        if False:
            print('Hello World!')
        self.assertListEqual(run_test.TARGET_DET_LIST, sorted(run_test.TARGET_DET_LIST))

    def test_config_change_only(self):
        if False:
            i = 10
            return i + 15
        'CI configs trigger all tests'
        self.assertEqual(self.determined_tests(['.ci/pytorch/test.sh']), self.TESTS)

    def test_run_test(self):
        if False:
            return 10
        'run_test.py is imported by determination tests'
        self.assertEqual(self.determined_tests(['test/run_test.py']), ['test_determination'])

    def test_non_code_change(self):
        if False:
            print('Hello World!')
        "Non-code changes don't trigger any tests"
        self.assertEqual(self.determined_tests(['CODEOWNERS', 'README.md', 'docs/doc.md']), [])

    def test_cpp_file(self):
        if False:
            return 10
        'CPP files trigger all tests'
        self.assertEqual(self.determined_tests(['aten/src/ATen/native/cpu/Activation.cpp']), self.TESTS)

    def test_test_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test files trigger themselves and dependent tests'
        self.assertEqual(self.determined_tests(['test/test_jit.py']), ['test_jit_profiling', 'test_jit'])
        self.assertEqual(self.determined_tests(['test/jit/test_custom_operators.py']), ['test_jit_profiling', 'test_jit'])
        self.assertEqual(self.determined_tests(['test/quantization/eager/test_quantize_eager_ptq.py']), ['test_quantization'])

    def test_test_internal_file(self):
        if False:
            i = 10
            return i + 15
        'testing/_internal files trigger dependent tests'
        self.assertEqual(self.determined_tests(['torch/testing/_internal/common_quantization.py']), ['test_jit_profiling', 'test_jit', 'test_quantization'])

    def test_torch_file(self):
        if False:
            i = 10
            return i + 15
        'Torch files trigger dependent tests'
        self.assertEqual(self.determined_tests(['torch/onnx/utils.py']), self.TESTS)
        self.assertEqual(self.determined_tests(['torch/autograd/_functions/utils.py', 'torch/autograd/_functions/utils.pyi']), ['test_utils'])
        self.assertEqual(self.determined_tests(['torch/utils/cpp_extension.py']), ['test_cpp_extensions_aot_ninja', 'test_cpp_extensions_aot_no_ninja', 'test_utils', 'test_determination'])

    def test_caffe2_file(self):
        if False:
            return 10
        'Caffe2 files trigger dependent tests'
        self.assertEqual(self.determined_tests(['caffe2/python/brew_test.py']), [])
        self.assertEqual(self.determined_tests(['caffe2/python/context.py']), self.TESTS)

    def test_new_folder(self):
        if False:
            while True:
                i = 10
        'New top-level Python folder triggers all tests'
        self.assertEqual(self.determined_tests(['new_module/file.py']), self.TESTS)

    def test_new_test_script(self):
        if False:
            for i in range(10):
                print('nop')
        "New test script triggers nothing (since it's not in run_tests.py)"
        self.assertEqual(self.determined_tests(['test/test_new_test_script.py']), [])
if __name__ == '__main__':
    run_tests()