"""Tests for tpu_test_wrapper.py."""
import importlib.util
import os
from absl.testing import flagsaver
from tensorflow.python.platform import flags
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_test_wrapper

class TPUTestWrapperTest(test.TestCase):

    @flagsaver.flagsaver()
    def test_flags_undefined(self):
        if False:
            for i in range(10):
                print('nop')
        tpu_test_wrapper.maybe_define_flags()
        self.assertIn('tpu', flags.FLAGS)
        self.assertIn('zone', flags.FLAGS)
        self.assertIn('project', flags.FLAGS)
        self.assertIn('model_dir', flags.FLAGS)

    @flagsaver.flagsaver()
    def test_flags_already_defined_not_overridden(self):
        if False:
            i = 10
            return i + 15
        flags.DEFINE_string('tpu', 'tpuname', 'helpstring')
        tpu_test_wrapper.maybe_define_flags()
        self.assertIn('tpu', flags.FLAGS)
        self.assertIn('zone', flags.FLAGS)
        self.assertIn('project', flags.FLAGS)
        self.assertIn('model_dir', flags.FLAGS)
        self.assertEqual(flags.FLAGS.tpu, 'tpuname')

    @flagsaver.flagsaver(bazel_repo_root='tensorflow/python')
    def test_parent_path(self):
        if False:
            return 10
        filepath = '/filesystem/path/tensorflow/python/tpu/example_test.runfiles/tensorflow/python/tpu/example_test'
        self.assertEqual(tpu_test_wrapper.calculate_parent_python_path(filepath), 'tensorflow.python.tpu')

    @flagsaver.flagsaver(bazel_repo_root='tensorflow/python')
    def test_parent_path_raises(self):
        if False:
            i = 10
            return i + 15
        filepath = '/bad/path'
        with self.assertRaisesWithLiteralMatch(ValueError, 'Filepath "/bad/path" does not contain repo root "tensorflow/python"'):
            tpu_test_wrapper.calculate_parent_python_path(filepath)

    def test_is_test_class_positive(self):
        if False:
            for i in range(10):
                print('nop')

        class A(test.TestCase):
            pass
        self.assertTrue(tpu_test_wrapper._is_test_class(A))

    def test_is_test_class_negative(self):
        if False:
            print('Hello World!')

        class A(object):
            pass
        self.assertFalse(tpu_test_wrapper._is_test_class(A))

    @flagsaver.flagsaver(wrapped_tpu_test_module_relative='.tpu_test_wrapper_test')
    def test_move_test_classes_into_scope(self):
        if False:
            print('Hello World!')
        with test.mock.patch.object(tpu_test_wrapper, 'calculate_parent_python_path') as mock_parent_path:
            mock_parent_path.return_value = tpu_test_wrapper.__name__.rpartition('.')[0]
            module = tpu_test_wrapper.import_user_module()
            tpu_test_wrapper.move_test_classes_into_scope(module)
        self.assertEqual(tpu_test_wrapper.tpu_test_imported_TPUTestWrapperTest.__name__, self.__class__.__name__)

    @flagsaver.flagsaver(test_dir_base='gs://example-bucket/tempfiles')
    def test_set_random_test_dir(self):
        if False:
            return 10
        tpu_test_wrapper.maybe_define_flags()
        tpu_test_wrapper.set_random_test_dir()
        self.assertStartsWith(flags.FLAGS.model_dir, 'gs://example-bucket/tempfiles')
        self.assertGreater(len(flags.FLAGS.model_dir), len('gs://example-bucket/tempfiles'))

    @flagsaver.flagsaver(test_dir_base='gs://example-bucket/tempfiles')
    def test_set_random_test_dir_repeatable(self):
        if False:
            i = 10
            return i + 15
        tpu_test_wrapper.maybe_define_flags()
        tpu_test_wrapper.set_random_test_dir()
        first = flags.FLAGS.model_dir
        tpu_test_wrapper.set_random_test_dir()
        second = flags.FLAGS.model_dir
        self.assertNotEqual(first, second)

    def test_run_user_main(self):
        if False:
            for i in range(10):
                print('nop')
        test_module = _write_and_load_module("\nVARS = 1\n\nif 'unrelated_if' == 'should_be_ignored':\n  VARS = 2\n\nif __name__ == '__main__':\n  VARS = 3\n\nif 'extra_if_at_bottom' == 'should_be_ignored':\n  VARS = 4\n")
        self.assertEqual(test_module.VARS, 1)
        tpu_test_wrapper.run_user_main(test_module)
        self.assertEqual(test_module.VARS, 3)

    def test_run_user_main_missing_if(self):
        if False:
            return 10
        test_module = _write_and_load_module('\nVARS = 1\n')
        self.assertEqual(test_module.VARS, 1)
        with self.assertRaises(NotImplementedError):
            tpu_test_wrapper.run_user_main(test_module)

    def test_run_user_main_double_quotes(self):
        if False:
            for i in range(10):
                print('nop')
        test_module = _write_and_load_module('\nVARS = 1\n\nif "unrelated_if" == "should_be_ignored":\n  VARS = 2\n\nif __name__ == "__main__":\n  VARS = 3\n\nif "extra_if_at_bottom" == "should_be_ignored":\n  VARS = 4\n')
        self.assertEqual(test_module.VARS, 1)
        tpu_test_wrapper.run_user_main(test_module)
        self.assertEqual(test_module.VARS, 3)

    def test_run_user_main_test(self):
        if False:
            return 10
        test_module = _write_and_load_module("\nfrom tensorflow.python.platform import test as unique_name\n\nclass DummyTest(unique_name.TestCase):\n  def test_fail(self):\n    self.fail()\n\nif __name__ == '__main__':\n  unique_name.main()\n")
        with test.mock.patch.object(test, 'main') as mock_main:
            tpu_test_wrapper.run_user_main(test_module)
        mock_main.assert_called_once()

def _write_and_load_module(source):
    if False:
        for i in range(10):
            print('nop')
    fp = os.path.join(test.get_temp_dir(), 'testmod.py')
    with open(fp, 'w') as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location('testmodule', fp)
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    return test_module
if __name__ == '__main__':
    test.main()