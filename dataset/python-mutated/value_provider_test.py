"""Unit tests for the ValueProvider class."""
import logging
import unittest
from mock import Mock
from apache_beam.options.pipeline_options import DebugOptions
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.value_provider import NestedValueProvider
from apache_beam.options.value_provider import RuntimeValueProvider
from apache_beam.options.value_provider import StaticValueProvider

class ValueProviderTests(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        RuntimeValueProvider.set_runtime_options(None)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        RuntimeValueProvider.set_runtime_options(None)

    def test_static_value_provider_keyword_argument(self):
        if False:
            print('Hello World!')

        class UserDefinedOptions(PipelineOptions):

            @classmethod
            def _add_argparse_args(cls, parser):
                if False:
                    while True:
                        i = 10
                parser.add_value_provider_argument('--vpt_vp_arg1', help='This keyword argument is a value provider', default='some value')
        options = UserDefinedOptions(['--vpt_vp_arg1', 'abc'])
        self.assertTrue(isinstance(options.vpt_vp_arg1, StaticValueProvider))
        self.assertTrue(options.vpt_vp_arg1.is_accessible())
        self.assertEqual(options.vpt_vp_arg1.get(), 'abc')

    def test_runtime_value_provider_keyword_argument(self):
        if False:
            while True:
                i = 10

        class UserDefinedOptions(PipelineOptions):

            @classmethod
            def _add_argparse_args(cls, parser):
                if False:
                    i = 10
                    return i + 15
                parser.add_value_provider_argument('--vpt_vp_arg2', help='This keyword argument is a value provider')
        options = UserDefinedOptions()
        self.assertTrue(isinstance(options.vpt_vp_arg2, RuntimeValueProvider))
        self.assertFalse(options.vpt_vp_arg2.is_accessible())
        with self.assertRaises(RuntimeError):
            options.vpt_vp_arg2.get()

    def test_static_value_provider_positional_argument(self):
        if False:
            print('Hello World!')

        class UserDefinedOptions(PipelineOptions):

            @classmethod
            def _add_argparse_args(cls, parser):
                if False:
                    while True:
                        i = 10
                parser.add_value_provider_argument('vpt_vp_arg3', help='This positional argument is a value provider', default='some value')
        options = UserDefinedOptions(['abc'])
        self.assertTrue(isinstance(options.vpt_vp_arg3, StaticValueProvider))
        self.assertTrue(options.vpt_vp_arg3.is_accessible())
        self.assertEqual(options.vpt_vp_arg3.get(), 'abc')

    def test_runtime_value_provider_positional_argument(self):
        if False:
            return 10

        class UserDefinedOptions(PipelineOptions):

            @classmethod
            def _add_argparse_args(cls, parser):
                if False:
                    i = 10
                    return i + 15
                parser.add_value_provider_argument('vpt_vp_arg4', help='This positional argument is a value provider')
        options = UserDefinedOptions([])
        self.assertTrue(isinstance(options.vpt_vp_arg4, RuntimeValueProvider))
        self.assertFalse(options.vpt_vp_arg4.is_accessible())
        with self.assertRaises(RuntimeError):
            options.vpt_vp_arg4.get()

    def test_static_value_provider_type_cast(self):
        if False:
            while True:
                i = 10

        class UserDefinedOptions(PipelineOptions):

            @classmethod
            def _add_argparse_args(cls, parser):
                if False:
                    print('Hello World!')
                parser.add_value_provider_argument('--vpt_vp_arg5', type=int, help='This flag is a value provider')
        options = UserDefinedOptions(['--vpt_vp_arg5', '123'])
        self.assertTrue(isinstance(options.vpt_vp_arg5, StaticValueProvider))
        self.assertTrue(options.vpt_vp_arg5.is_accessible())
        self.assertEqual(options.vpt_vp_arg5.get(), 123)

    def test_set_runtime_option(self):
        if False:
            return 10

        class UserDefinedOptions1(PipelineOptions):

            @classmethod
            def _add_argparse_args(cls, parser):
                if False:
                    return 10
                parser.add_value_provider_argument('--vpt_vp_arg6', help='This keyword argument is a value provider')
                parser.add_value_provider_argument('-v', '--vpt_vp_arg7', default=123, type=int)
                parser.add_value_provider_argument('--vpt_vp-arg8', default='123', type=str)
                parser.add_value_provider_argument('--vpt_vp_arg9', type=float)
                parser.add_value_provider_argument('vpt_vp_arg10', help='This positional argument is a value provider', type=float, default=5.4)
        options = UserDefinedOptions1(['1.2'])
        self.assertFalse(options.vpt_vp_arg6.is_accessible())
        self.assertFalse(options.vpt_vp_arg7.is_accessible())
        self.assertFalse(options.vpt_vp_arg8.is_accessible())
        self.assertFalse(options.vpt_vp_arg9.is_accessible())
        self.assertTrue(options.vpt_vp_arg10.is_accessible())
        RuntimeValueProvider.set_runtime_options({'vpt_vp_arg6': 'abc', 'vpt_vp_arg10': '3.2'})
        self.assertTrue(options.vpt_vp_arg6.is_accessible())
        self.assertEqual(options.vpt_vp_arg6.get(), 'abc')
        self.assertTrue(options.vpt_vp_arg7.is_accessible())
        self.assertEqual(options.vpt_vp_arg7.get(), 123)
        self.assertTrue(options.vpt_vp_arg8.is_accessible())
        self.assertEqual(options.vpt_vp_arg8.get(), '123')
        self.assertTrue(options.vpt_vp_arg9.is_accessible())
        self.assertIsNone(options.vpt_vp_arg9.get())
        self.assertTrue(options.vpt_vp_arg10.is_accessible())
        self.assertEqual(options.vpt_vp_arg10.get(), 1.2)

    def test_choices(self):
        if False:
            while True:
                i = 10

        class UserDefinedOptions(PipelineOptions):

            @classmethod
            def _add_argparse_args(cls, parser):
                if False:
                    for i in range(10):
                        print('nop')
                parser.add_argument('--vpt_vp_arg11', choices=['a', 'b'], help='This flag is a value provider with concrete choices')
                parser.add_argument('--vpt_vp_arg12', choices=[1, 2], type=int, help='This flag is a value provider with concrete choices')
        options = UserDefinedOptions(['--vpt_vp_arg11', 'a', '--vpt_vp_arg12', '2'])
        self.assertEqual(options.vpt_vp_arg11, 'a')
        self.assertEqual(options.vpt_vp_arg12, 2)

    def test_static_value_provider_choices(self):
        if False:
            while True:
                i = 10

        class UserDefinedOptions(PipelineOptions):

            @classmethod
            def _add_argparse_args(cls, parser):
                if False:
                    for i in range(10):
                        print('nop')
                parser.add_value_provider_argument('--vpt_vp_arg13', choices=['a', 'b'], help='This flag is a value provider with concrete choices')
                parser.add_value_provider_argument('--vpt_vp_arg14', choices=[1, 2], type=int, help='This flag is a value provider with concrete choices')
        options = UserDefinedOptions(['--vpt_vp_arg13', 'a', '--vpt_vp_arg14', '2'])
        self.assertEqual(options.vpt_vp_arg13.get(), 'a')
        self.assertEqual(options.vpt_vp_arg14.get(), 2)

    def test_experiments_setup(self):
        if False:
            return 10
        self.assertFalse('feature_1' in RuntimeValueProvider.experiments)
        RuntimeValueProvider.set_runtime_options({'experiments': ['feature_1', 'feature_2']})
        self.assertTrue(isinstance(RuntimeValueProvider.experiments, set))
        self.assertTrue('feature_1' in RuntimeValueProvider.experiments)
        self.assertTrue('feature_2' in RuntimeValueProvider.experiments)

    def test_experiments_options_setup(self):
        if False:
            i = 10
            return i + 15
        options = PipelineOptions(['--experiments', 'a', '--experiments', 'b,c'])
        options = options.view_as(DebugOptions)
        self.assertIn('a', options.experiments)
        self.assertIn('b,c', options.experiments)
        self.assertNotIn('c', options.experiments)

    def test_nested_value_provider_wrap_static(self):
        if False:
            for i in range(10):
                print('nop')
        vp = NestedValueProvider(StaticValueProvider(int, 1), lambda x: x + 1)
        self.assertTrue(vp.is_accessible())
        self.assertEqual(vp.get(), 2)

    def test_nested_value_provider_caches_value(self):
        if False:
            for i in range(10):
                print('nop')
        mock_fn = Mock()

        def translator(x):
            if False:
                return 10
            mock_fn()
            return x
        vp = NestedValueProvider(StaticValueProvider(int, 1), translator)
        vp.get()
        self.assertEqual(mock_fn.call_count, 1)
        vp.get()
        self.assertEqual(mock_fn.call_count, 1)

    def test_nested_value_provider_wrap_runtime(self):
        if False:
            while True:
                i = 10

        class UserDefinedOptions(PipelineOptions):

            @classmethod
            def _add_argparse_args(cls, parser):
                if False:
                    return 10
                parser.add_value_provider_argument('--vpt_vp_arg15', help='This keyword argument is a value provider')
        options = UserDefinedOptions([])
        vp = NestedValueProvider(options.vpt_vp_arg15, lambda x: x + x)
        self.assertFalse(vp.is_accessible())
        RuntimeValueProvider.set_runtime_options({'vpt_vp_arg15': 'abc'})
        self.assertTrue(vp.is_accessible())
        self.assertEqual(vp.get(), 'abcabc')
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()