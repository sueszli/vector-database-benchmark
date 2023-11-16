"""Tests for dataset options utilities."""
from absl.testing import parameterized
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.util import options
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test

class _TestOptions(options.OptionsBase):
    x = options.create_option(name='x', ty=int, docstring='the answer to everything', default_factory=lambda : 42)
    y = options.create_option(name='y', ty=float, docstring='a tasty pie', default_factory=lambda : 3.14)

class _NestedTestOptions(options.OptionsBase):
    opts = options.create_option(name='opts', ty=_TestOptions, docstring='nested options')

class OptionsTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(test_base.default_test_combinations())
    def testDocumentation(self):
        if False:
            print('Hello World!')
        self.assertEqual(_TestOptions.x.__doc__, 'the answer to everything')
        self.assertEqual(_TestOptions.y.__doc__, 'a tasty pie')

    @combinations.generate(test_base.default_test_combinations())
    def testCreateOption(self):
        if False:
            while True:
                i = 10
        opts = _TestOptions()
        self.assertEqual(opts.x, 42)
        self.assertEqual(opts.y, 3.14)
        self.assertIsInstance(opts.x, int)
        self.assertIsInstance(opts.y, float)
        opts.x = 0
        self.assertEqual(opts.x, 0)
        with self.assertRaises(TypeError):
            opts.x = 3.14
        opts.y = 0.0
        self.assertEqual(opts.y, 0.0)
        with self.assertRaises(TypeError):
            opts.y = 42

    @combinations.generate(test_base.default_test_combinations())
    def testMergeOptions(self):
        if False:
            for i in range(10):
                print('nop')
        (options1, options2) = (_TestOptions(), _TestOptions())
        with self.assertRaises(ValueError):
            options.merge_options()
        merged_options = options.merge_options(options1, options2)
        self.assertEqual(merged_options.x, 42)
        self.assertEqual(merged_options.y, 3.14)
        options1.x = 0
        options2.y = 0.0
        merged_options = options.merge_options(options1, options2)
        self.assertEqual(merged_options.x, 0)
        self.assertEqual(merged_options.y, 0.0)

    @combinations.generate(test_base.default_test_combinations())
    def testMergeNestedOptions(self):
        if False:
            i = 10
            return i + 15
        (options1, options2) = (_NestedTestOptions(), _NestedTestOptions())
        merged_options = options.merge_options(options1, options2)
        self.assertEqual(merged_options.opts, None)
        options1.opts = _TestOptions()
        merged_options = options.merge_options(options1, options2)
        self.assertEqual(merged_options.opts, _TestOptions())
        options2.opts = _TestOptions()
        merged_options = options.merge_options(options1, options2)
        self.assertEqual(merged_options.opts, _TestOptions())
        options1.opts.x = 0
        options2.opts.y = 0.0
        merged_options = options.merge_options(options1, options2)
        self.assertEqual(merged_options.opts.x, 0)
        self.assertEqual(merged_options.opts.y, 0.0)

    @combinations.generate(test_base.default_test_combinations())
    def testImmutable(self):
        if False:
            print('Hello World!')
        test_options = _TestOptions()
        test_options._set_mutable(False)
        with self.assertRaisesRegex(ValueError, 'Mutating `tf.data.Options\\(\\)` returned by `tf.data.Dataset.options\\(\\)` has no effect. Use `tf.data.Dataset.with_options\\(options\\)` to set or update dataset options.'):
            test_options.test = 100

    @combinations.generate(test_base.default_test_combinations())
    def testNoSpuriousAttrs(self):
        if False:
            while True:
                i = 10
        test_options = _TestOptions()
        with self.assertRaisesRegex(AttributeError, 'Cannot set the property wrong_attr on _TestOptions.'):
            test_options.wrong_attr = True
        with self.assertRaises(AttributeError):
            _ = test_options.wrong_attr

    @combinations.generate(test_base.default_test_combinations())
    def testMergeNoOptions(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(ValueError, 'At least one options should be provided'):
            options.merge_options()

    @combinations.generate(test_base.default_test_combinations())
    def testMergeOptionsDifferentType(self):
        if False:
            print('Hello World!')
        (options1, options2) = (_TestOptions(), _NestedTestOptions())
        with self.assertRaisesRegex(TypeError, "Could not merge incompatible options of type \\<class \\'__main__._NestedTestOptions\\'\\> and \\<class \\'__main__._TestOptions\\'\\>."):
            options.merge_options(options1, options2)

    @combinations.generate(test_base.default_test_combinations())
    def testMergeOptionsWrongType(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(TypeError, "All options to be merged should inherit from \\`OptionsBase\\` but found option of type \\<class \\'int\\'\\> which does not."):
            options.merge_options(1, 2, 3)
if __name__ == '__main__':
    test.main()