import collections
import glob
import logging
import os
import tempfile
import unittest
import apache_beam as beam
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.utils import python_callable
from apache_beam.yaml import yaml_provider
from apache_beam.yaml.yaml_transform import YamlTransform

class CreateTimestamped(beam.PTransform):
    _yaml_requires_inputs = False

    def __init__(self, elements):
        if False:
            return 10
        self._elements = elements

    def expand(self, p):
        if False:
            for i in range(10):
                print('nop')
        return p | beam.Create(self._elements) | beam.Map(lambda x: beam.transforms.window.TimestampedValue(x, x))

class CreateInts(beam.PTransform):
    _yaml_requires_inputs = False

    def __init__(self, elements):
        if False:
            print('Hello World!')
        self._elements = elements

    def expand(self, p):
        if False:
            for i in range(10):
                print('nop')
        return p | beam.Create(self._elements)

class SumGlobally(beam.PTransform):

    def expand(self, pcoll):
        if False:
            for i in range(10):
                print('nop')
        return pcoll | beam.CombineGlobally(sum).without_defaults()

class SizeLimiter(beam.PTransform):

    def __init__(self, limit, error_handling):
        if False:
            for i in range(10):
                print('nop')
        self._limit = limit
        self._error_handling = error_handling

    def expand(self, pcoll):
        if False:
            for i in range(10):
                print('nop')

        def raise_on_big(row):
            if False:
                return 10
            if len(row.element) > self._limit:
                raise ValueError(row.element)
            else:
                return row.element
        (good, bad) = pcoll | beam.Map(raise_on_big).with_exception_handling()
        return {'small_elements': good, self._error_handling['output']: bad}
TEST_PROVIDERS = {'CreateInts': CreateInts, 'CreateTimestamped': CreateTimestamped, 'SumGlobally': SumGlobally, 'SizeLimiter': SizeLimiter, 'PyMap': lambda fn: beam.Map(python_callable.PythonCallableWithSource(fn))}

class YamlTransformE2ETest(unittest.TestCase):

    def test_composite(self):
        if False:
            i = 10
            return i + 15
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            elements = p | beam.Create([1, 2, 3])
            result = elements | YamlTransform('\n          type: composite\n          input:\n              elements: input\n          transforms:\n            - type: PyMap\n              name: Square\n              input: elements\n              config:\n                  fn: "lambda x: x * x"\n            - type: PyMap\n              name: Cube\n              input: elements\n              config:\n                  fn: "lambda x: x * x * x"\n            - type: Flatten\n              input: [Square, Cube]\n          output:\n              Flatten\n          ', providers=TEST_PROVIDERS)
            assert_that(result, equal_to([1, 4, 9, 1, 8, 27]))

    def test_chain_with_input(self):
        if False:
            for i in range(10):
                print('nop')
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            elements = p | beam.Create(range(10))
            result = elements | YamlTransform('\n          type: chain\n          input:\n              elements: input\n          transforms:\n            - type: PyMap\n              config:\n                  fn: "lambda x: x * x + x"\n            - type: PyMap\n              config:\n                  fn: "lambda x: x + 41"\n          ', providers=TEST_PROVIDERS)
            assert_that(result, equal_to([41, 43, 47, 53, 61, 71, 83, 97, 113, 131]))

    def test_chain_with_source_sink(self):
        if False:
            i = 10
            return i + 15
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            result = p | YamlTransform('\n          type: chain\n          source:\n            type: CreateInts\n            config:\n                elements: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]\n          transforms:\n            - type: PyMap\n              config:\n                  fn: "lambda x: x * x + x"\n          sink:\n            type: PyMap\n            config:\n                fn: "lambda x: x + 41"\n          ', providers=TEST_PROVIDERS)
            assert_that(result, equal_to([41, 43, 47, 53, 61, 71, 83, 97, 113, 131]))

    def test_chain_with_root(self):
        if False:
            print('Hello World!')
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            result = p | YamlTransform('\n          type: chain\n          transforms:\n            - type: CreateInts\n              config:\n                  elements: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]\n            - type: PyMap\n              config:\n                  fn: "lambda x: x * x + x"\n            - type: PyMap\n              config:\n                  fn: "lambda x: x + 41"\n          ', providers=TEST_PROVIDERS)
            assert_that(result, equal_to([41, 43, 47, 53, 61, 71, 83, 97, 113, 131]))

    def create_has_schema(self):
        if False:
            print('Hello World!')
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            result = p | YamlTransform("\n          type: chain\n          transforms:\n            - type: Create\n              config:\n                  elements: [{a: 1, b: 'x'}, {a: 2, b: 'y'}]\n            - type: MapToFields\n              config:\n                  language: python\n                  fields:\n                      repeated: a * b\n          ") | beam.Map(lambda x: x.repeated)
            assert_that(result, equal_to(['x', 'yy']))

    def test_implicit_flatten(self):
        if False:
            while True:
                i = 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            result = p | YamlTransform('\n          type: composite\n          transforms:\n            - type: Create\n              name: CreateSmall\n              config:\n                  elements: [1, 2, 3]\n            - type: Create\n              name: CreateBig\n              config:\n                  elements: [100, 200]\n            - type: PyMap\n              input: [CreateBig, CreateSmall]\n              config:\n                  fn: "lambda x: x.element * x.element"\n          output: PyMap\n          ', providers=TEST_PROVIDERS)
            assert_that(result, equal_to([1, 4, 9, 10000, 40000]))

    def test_csv_to_json(self):
        if False:
            return 10
        try:
            import pandas as pd
        except ImportError:
            raise unittest.SkipTest('Pandas not available.')
        with tempfile.TemporaryDirectory() as tmpdir:
            data = pd.DataFrame([{'label': '11a', 'rank': 0}, {'label': '37a', 'rank': 1}, {'label': '389a', 'rank': 2}])
            input = os.path.join(tmpdir, 'input.csv')
            output = os.path.join(tmpdir, 'output.json')
            data.to_csv(input, index=False)
            with beam.Pipeline() as p:
                result = p | YamlTransform('\n            type: chain\n            transforms:\n              - type: ReadFromCsv\n                config:\n                    path: %s\n              - type: WriteToJson\n                config:\n                    path: %s\n                num_shards: 1\n            ' % (repr(input), repr(output)))
            output_shard = list(glob.glob(output + '*'))[0]
            result = pd.read_json(output_shard, orient='records', lines=True).sort_values('rank').reindex()
            pd.testing.assert_frame_equal(data, result)

    def test_name_is_not_ambiguous(self):
        if False:
            while True:
                i = 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            result = p | YamlTransform('\n            type: composite\n            transforms:\n              - type: Create\n                name: Create\n                config:\n                    elements: [0, 1, 3, 4]\n              - type: PyMap\n                name: PyMap\n                config:\n                    fn: "lambda row: row.element * row.element"\n                input: Create\n            output: PyMap\n            ', providers=TEST_PROVIDERS)
            assert_that(result, equal_to([0, 1, 9, 16]))

    def test_name_is_ambiguous(self):
        if False:
            i = 10
            return i + 15
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            with self.assertRaisesRegex(ValueError, 'Ambiguous.*'):
                p | YamlTransform('\n            type: composite\n            transforms:\n              - type: Create\n                name: CreateData\n                config:\n                    elements: [0, 1, 3, 4]\n              - type: PyMap\n                name: PyMap\n                config:\n                    fn: "lambda elem: elem + 2"\n                input: CreateData\n              - type: PyMap\n                name: AnotherMap\n                config:\n                    fn: "lambda elem: elem + 3"\n                input: PyMap\n            output: AnotherMap\n            ', providers=TEST_PROVIDERS)

    def test_empty_inputs_throws_error(self):
        if False:
            return 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            with self.assertRaisesRegex(ValueError, 'Missing inputs for transform at "EmptyInputOkButYamlDoesntKnow" at line .*'):
                _ = p | YamlTransform('\n            type: composite\n            transforms:\n              - type: PyTransform\n                name: EmptyInputOkButYamlDoesntKnow\n                config:\n                  constructor: apache_beam.Impulse\n            ')

    def test_empty_inputs_ok_in_source(self):
        if False:
            return 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            _ = p | YamlTransform('\n          type: composite\n          source:\n            type: PyTransform\n            name: EmptyInputOkButYamlDoesntKnow\n            config:\n              constructor: apache_beam.Impulse\n          ')

    def test_empty_inputs_ok_if_explicit(self):
        if False:
            for i in range(10):
                print('nop')
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            _ = p | YamlTransform('\n          type: composite\n          transforms:\n            - type: PyTransform\n              name: EmptyInputOkButYamlDoesntKnow\n              input: {}\n              config:\n                constructor: apache_beam.Impulse\n          ')

    def test_annotations(self):
        if False:
            print('Hello World!')
        t = LinearTransform(5, b=100)
        annotations = t.annotations()
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            result = p | YamlTransform('\n          type: chain\n          transforms:\n            - type: Create\n              config:\n                elements: [0, 1, 2, 3]\n            - type: %r\n              config: %s\n          ' % (annotations['yaml_type'], annotations['yaml_args']))
            assert_that(result, equal_to([100, 105, 110, 115]))

class ErrorHandlingTest(unittest.TestCase):

    def test_error_handling_outputs(self):
        if False:
            i = 10
            return i + 15
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            result = p | YamlTransform('\n          type: composite\n          transforms:\n            - type: Create\n              config:\n                  elements: [\'a\', \'b\', \'biiiiig\']\n            - type: SizeLimiter\n              input: Create\n              config:\n                  limit: 5\n                  error_handling:\n                    output: errors\n            - name: TrimErrors\n              type: PyMap\n              input: SizeLimiter.errors\n              config:\n                  fn: "lambda x: x[1][1]"\n          output:\n            good: SizeLimiter\n            bad: TrimErrors\n          ', providers=TEST_PROVIDERS)
            assert_that(result['good'], equal_to(['a', 'b']), label='CheckGood')
            assert_that(result['bad'], equal_to(["ValueError('biiiiig')"]))

    def test_must_handle_error_output(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(Exception, 'Unconsumed error output .*line 7'):
            with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
                _ = p | YamlTransform("\n            type: composite\n            transforms:\n              - type: Create\n                config:\n                    elements: ['a', 'b', 'biiiiig']\n              - type: SizeLimiter\n                input: Create\n                config:\n                    limit: 5\n                    error_handling:\n                      output: errors\n            ", providers=TEST_PROVIDERS)

    def test_mapping_errors(self):
        if False:
            return 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            result = p | YamlTransform('\n          type: composite\n          transforms:\n            - type: Create\n              config:\n                  elements: [0, 1, 2, 4]\n            - type: MapToFields\n              name: ToRow\n              input: Create\n              config:\n                  language: python\n                  fields:\n                      num: element\n                      str: "\'a\' * element or \'bbb\'"\n            - type: Filter\n              input: ToRow\n              config:\n                  language: python\n                  keep:\n                    str[1] >= \'a\'\n                  error_handling:\n                    output: errors\n            - type: MapToFields\n              name: MapWithErrorHandling\n              input: Filter\n              config:\n                  language: python\n                  fields:\n                    num: num\n                    inverse: float(1 / num)\n                  error_handling:\n                    output: errors\n            - type: PyMap\n              name: TrimErrors\n              input: [MapWithErrorHandling.errors, Filter.errors]\n              config:\n                  fn: "lambda x: x.msg"\n            - type: MapToFields\n              name: Sum\n              input: MapWithErrorHandling\n              config:\n                  language: python\n                  append: True\n                  fields:\n                    sum: num + inverse\n          output:\n            good: Sum\n            bad: TrimErrors\n          ', providers=TEST_PROVIDERS)
            assert_that(result['good'], equal_to([beam.Row(num=2, inverse=0.5, sum=2.5), beam.Row(num=4, inverse=0.25, sum=4.25)]), label='CheckGood')
            assert_that(result['bad'], equal_to(["IndexError('string index out of range')", "ZeroDivisionError('division by zero')"]), label='CheckErrors')

class YamlWindowingTest(unittest.TestCase):

    def test_explicit_window_into(self):
        if False:
            while True:
                i = 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            result = p | YamlTransform('\n          type: chain\n          transforms:\n            - type: CreateTimestamped\n              config:\n                  elements: [0, 1, 2, 3, 4, 5]\n            - type: WindowInto\n              windowing:\n                type: fixed\n                size: 4\n            - type: SumGlobally\n          ', providers=TEST_PROVIDERS)
            assert_that(result, equal_to([6, 9]))

    def test_windowing_on_input(self):
        if False:
            return 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            result = p | YamlTransform('\n          type: chain\n          transforms:\n            - type: CreateTimestamped\n              config:\n                  elements: [0, 1, 2, 3, 4, 5]\n            - type: SumGlobally\n              windowing:\n                type: fixed\n                size: 4\n          ', providers=TEST_PROVIDERS)
            assert_that(result, equal_to([6, 9]))

    def test_windowing_multiple_inputs(self):
        if False:
            i = 10
            return i + 15
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            result = p | YamlTransform('\n          type: composite\n          transforms:\n            - type: CreateTimestamped\n              name: Create1\n              config:\n                  elements: [0, 2, 4]\n            - type: CreateTimestamped\n              name: Create2\n              config:\n                  elements: [1, 3, 5]\n            - type: SumGlobally\n              input: [Create1, Create2]\n              windowing:\n                type: fixed\n                size: 4\n          output: SumGlobally\n          ', providers=TEST_PROVIDERS)
            assert_that(result, equal_to([6, 9]))

    def test_windowing_on_output(self):
        if False:
            for i in range(10):
                print('nop')
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            result = p | YamlTransform('\n          type: chain\n          transforms:\n            - type: CreateTimestamped\n              config:\n                  elements: [0, 1, 2, 3, 4, 5]\n              windowing:\n                type: fixed\n                size: 4\n            - type: SumGlobally\n          ', providers=TEST_PROVIDERS)
            assert_that(result, equal_to([6, 9]))

    def test_windowing_on_outer(self):
        if False:
            print('Hello World!')
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            result = p | YamlTransform('\n          type: chain\n          transforms:\n            - type: CreateTimestamped\n              config:\n                  elements: [0, 1, 2, 3, 4, 5]\n            - type: SumGlobally\n          windowing:\n            type: fixed\n            size: 4\n          ', providers=TEST_PROVIDERS)
            assert_that(result, equal_to([6, 9]))

class AnnotatingProvider(yaml_provider.InlineProvider):
    """A provider that vends transforms that do nothing but record that this
  provider (as identified by name) was used, along with any prior history
  of the given element.
  """

    def __init__(self, name, transform_names):
        if False:
            i = 10
            return i + 15
        super().__init__({transform_name: lambda : beam.Map(lambda x: (x if type(x) == tuple else ()) + (name,)) for transform_name in transform_names.strip().split()})
        self._name = name

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'AnnotatingProvider(%r)' % self._name

class AnotherAnnProvider(AnnotatingProvider):
    """A Provider that behaves exactly as AnnotatingProvider, but is not
  of the same type and so is considered "more distant" for matching purposes.
  """
    pass

class ProviderAffinityTest(unittest.TestCase):
    """These tests check that for a sequence of transforms, the "closest"
  proveders are chosen among multiple possible implementations.
  """
    provider1 = AnnotatingProvider('provider1', 'P1 A B C  ')
    provider2 = AnnotatingProvider('provider2', 'P2 A   C D')
    provider3 = AnotherAnnProvider('provider3', 'P3 A B    ')
    provider4 = AnotherAnnProvider('provider4', 'P4 A B   D')
    providers_dict = collections.defaultdict(list)
    for provider in [provider1, provider2, provider3, provider4]:
        for transform_type in provider.provided_transforms():
            providers_dict[transform_type].append(provider)

    def test_prefers_same_provider(self):
        if False:
            i = 10
            return i + 15
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            result1 = p | 'Yaml1' >> YamlTransform('\n          type: chain\n          transforms:\n            - type: Create\n              config:\n                  elements: [0]\n            - type: P1\n            - type: A\n            - type: C\n          ', providers=self.providers_dict)
            assert_that(result1, equal_to([('provider1', 'provider1', 'provider1')]), label='StartWith1')
            result2 = p | 'Yaml2' >> YamlTransform('\n          type: chain\n          transforms:\n            - type: Create\n              config:\n                  elements: [0]\n            - type: P2\n            - type: A\n            - type: C\n          ', providers=self.providers_dict)
            assert_that(result2, equal_to([('provider2', 'provider2', 'provider2')]), label='StartWith2')

    def test_prefers_same_provider_class(self):
        if False:
            for i in range(10):
                print('nop')
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            result1 = p | 'Yaml1' >> YamlTransform('\n          type: chain\n          transforms:\n            - type: Create\n              config:\n                  elements: [0]\n            - type: P1\n            - type: A\n            - type: D\n            - type: A\n          ', providers=self.providers_dict)
            assert_that(result1, equal_to([('provider1', 'provider1', 'provider2', 'provider2')]), label='StartWith1')
            result3 = p | 'Yaml2' >> YamlTransform('\n          type: chain\n          transforms:\n            - type: Create\n              config:\n                  elements: [0]\n            - type: P3\n            - type: A\n            - type: D\n            - type: A\n          ', providers=self.providers_dict)
            assert_that(result3, equal_to([('provider3', 'provider3', 'provider4', 'provider4')]), label='StartWith3')

@beam.transforms.ptransform.annotate_yaml
class LinearTransform(beam.PTransform):
    """A transform used for testing annotate_yaml."""

    def __init__(self, a, b):
        if False:
            i = 10
            return i + 15
        self._a = a
        self._b = b

    def expand(self, pcoll):
        if False:
            i = 10
            return i + 15
        a = self._a
        b = self._b
        return pcoll | beam.Map(lambda x: a * x.element + b)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()