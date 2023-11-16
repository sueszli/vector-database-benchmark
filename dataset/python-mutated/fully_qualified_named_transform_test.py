import unittest
from mock import call
from mock import patch
import apache_beam as beam
from apache_beam.runners.portability import expansion_service
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.transforms.external import ImplicitSchemaPayloadBuilder
from apache_beam.transforms.fully_qualified_named_transform import PYTHON_FULLY_QUALIFIED_NAMED_TRANSFORM_URN
from apache_beam.transforms.fully_qualified_named_transform import FullyQualifiedNamedTransform
from apache_beam.utils import python_callable

class FullyQualifiedNamedTransformTest(unittest.TestCase):

    def test_test_transform(self):
        if False:
            for i in range(10):
                print('nop')
        with beam.Pipeline() as p:
            assert_that(p | beam.Create(['a', 'b', 'c']) | _TestTransform('x', 'y'), equal_to(['xay', 'xby', 'xcy']))

    def test_expand(self):
        if False:
            while True:
                i = 10
        with FullyQualifiedNamedTransform.with_filter('*'):
            with beam.Pipeline() as p:
                assert_that(p | beam.Create(['a', 'b', 'c']) | FullyQualifiedNamedTransform('apache_beam.transforms.fully_qualified_named_transform_test._TestTransform', ('x',), {'suffix': 'y'}), equal_to(['xay', 'xby', 'xcy']))

    def test_static_constructor(self):
        if False:
            while True:
                i = 10
        with FullyQualifiedNamedTransform.with_filter('*'):
            with beam.Pipeline() as p:
                assert_that(p | beam.Create(['a', 'b', 'c']) | FullyQualifiedNamedTransform('apache_beam.transforms.fully_qualified_named_transform_test._TestTransform.create', ('x',), {'suffix': 'y'}), equal_to(['xay', 'xby', 'xcy']))

    def test_as_external_transform(self):
        if False:
            print('Hello World!')
        with FullyQualifiedNamedTransform.with_filter('*'):
            with beam.Pipeline() as p:
                assert_that(p | beam.Create(['a', 'b', 'c']) | beam.ExternalTransform(PYTHON_FULLY_QUALIFIED_NAMED_TRANSFORM_URN, ImplicitSchemaPayloadBuilder({'constructor': 'apache_beam.transforms.fully_qualified_named_transform_test._TestTransform', 'args': beam.Row(arg0='x'), 'kwargs': beam.Row(suffix='y')}), expansion_service.ExpansionServiceServicer()), equal_to(['xay', 'xby', 'xcy']))

    def test_as_external_transform_no_args(self):
        if False:
            return 10
        with FullyQualifiedNamedTransform.with_filter('*'):
            with beam.Pipeline() as p:
                assert_that(p | beam.Create(['a', 'b', 'c']) | beam.ExternalTransform(PYTHON_FULLY_QUALIFIED_NAMED_TRANSFORM_URN, ImplicitSchemaPayloadBuilder({'constructor': 'apache_beam.transforms.fully_qualified_named_transform_test._TestTransform', 'kwargs': beam.Row(prefix='x', suffix='y')}), expansion_service.ExpansionServiceServicer()), equal_to(['xay', 'xby', 'xcy']))

    def test_as_external_transform_no_kwargs(self):
        if False:
            return 10
        with FullyQualifiedNamedTransform.with_filter('*'):
            with beam.Pipeline() as p:
                assert_that(p | beam.Create(['a', 'b', 'c']) | beam.ExternalTransform(PYTHON_FULLY_QUALIFIED_NAMED_TRANSFORM_URN, ImplicitSchemaPayloadBuilder({'constructor': 'apache_beam.transforms.fully_qualified_named_transform_test._TestTransform', 'args': beam.Row(arg0='x', arg1='y')}), expansion_service.ExpansionServiceServicer()), equal_to(['xay', 'xby', 'xcy']))

    def test_callable_transform(self):
        if False:
            while True:
                i = 10
        with FullyQualifiedNamedTransform.with_filter('*'):
            with beam.Pipeline() as p:
                assert_that(p | beam.Create(['a', 'b', 'c']) | FullyQualifiedNamedTransform('__callable__', (python_callable.PythonCallableWithSource('\n                      def func(pcoll, x):\n                        return pcoll | beam.Map(lambda e: e + x)\n                      '), 'x'), {}), equal_to(['ax', 'bx', 'cx']))

    def test_constructor_transform(self):
        if False:
            return 10
        with FullyQualifiedNamedTransform.with_filter('*'):
            with beam.Pipeline() as p:
                assert_that(p | beam.Create(['a', 'b', 'c']) | FullyQualifiedNamedTransform('__constructor__', (), {'source': python_callable.PythonCallableWithSource('\n                    class MyTransform(beam.PTransform):\n                      def __init__(self, x):\n                        self._x = x\n                      def expand(self, pcoll):\n                        return pcoll | beam.Map(lambda e: e + self._x)\n                    '), 'x': 'x'}), equal_to(['ax', 'bx', 'cx']))

    def test_glob_filter(self):
        if False:
            for i in range(10):
                print('nop')
        with FullyQualifiedNamedTransform.with_filter('*'):
            self.assertIs(FullyQualifiedNamedTransform._resolve('apache_beam.Row'), beam.Row)
        with FullyQualifiedNamedTransform.with_filter('apache_beam.*'):
            self.assertIs(FullyQualifiedNamedTransform._resolve('apache_beam.Row'), beam.Row)
        with FullyQualifiedNamedTransform.with_filter('apache_beam.foo.*'):
            with self.assertRaises(ValueError):
                FullyQualifiedNamedTransform._resolve('apache_beam.Row')

    @patch('importlib.import_module')
    def test_resolve_by_path_segment(self, mock_import_module):
        if False:
            for i in range(10):
                print('nop')
        mock_import_module.return_value = None
        with FullyQualifiedNamedTransform.with_filter('*'):
            FullyQualifiedNamedTransform._resolve('a.b.c.d')
        mock_import_module.assert_has_calls([call('a'), call('a.b'), call('a.b.c'), call('a.b.c.d')])

    def test_resolve(self):
        if False:
            return 10
        with FullyQualifiedNamedTransform.with_filter('*'):
            dataframe_transform = FullyQualifiedNamedTransform._resolve('apache_beam.dataframe.transforms.DataframeTransform')
            from apache_beam.dataframe.transforms import DataframeTransform
            self.assertIs(dataframe_transform, DataframeTransform)
        with FullyQualifiedNamedTransform.with_filter('*'):
            argument_placeholder = FullyQualifiedNamedTransform._resolve('apache_beam.internal.util.ArgumentPlaceholder')
            from apache_beam.internal.util import ArgumentPlaceholder
            self.assertIs(argument_placeholder, ArgumentPlaceholder)

class _TestTransform(beam.PTransform):

    @classmethod
    def create(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return cls(*args, **kwargs)

    def __init__(self, prefix, suffix):
        if False:
            for i in range(10):
                print('nop')
        self._prefix = prefix
        self._suffix = suffix

    def expand(self, pcoll):
        if False:
            i = 10
            return i + 15
        return pcoll | beam.Map(lambda s: self._prefix + s + self._suffix)
if __name__ == '__main__':
    unittest.main()