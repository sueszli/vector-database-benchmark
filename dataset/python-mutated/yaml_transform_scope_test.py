import collections
import logging
import unittest
import yaml
import apache_beam as beam
from apache_beam.yaml import yaml_provider
from apache_beam.yaml import yaml_transform
from apache_beam.yaml.yaml_transform import LightweightScope
from apache_beam.yaml.yaml_transform import SafeLineLoader
from apache_beam.yaml.yaml_transform import Scope

class ScopeTest(unittest.TestCase):

    def get_scope_by_spec(self, p, spec):
        if False:
            return 10
        spec = yaml.load(spec, Loader=SafeLineLoader)
        scope = Scope(beam.pvalue.PBegin(p), {}, spec['transforms'], yaml_provider.standard_providers(), {})
        return (scope, spec)

    def test_get_pcollection_input(self):
        if False:
            i = 10
            return i + 15
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            elements = p | beam.Create(range(3))
            scope = Scope(p, {'input': elements}, transforms=[], providers=yaml_provider.standard_providers(), input_providers={})
            result = scope.get_pcollection('input')
            self.assertEqual('PCollection[Create/Map(decode).None]', str(result))

    def test_get_pcollection_output(self):
        if False:
            return 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            spec = '\n        transforms:\n          - type: Create\n            config:\n              elements: [0, 1, 3, 4]\n          - type: LogForTesting\n            name: Square\n            input: Create\n        '
        (scope, spec) = self.get_scope_by_spec(p, spec)
        self.assertEqual('PCollection[Create/Map(decode).None]', str(scope.get_pcollection('Create')))
        self.assertEqual('PCollection[Square.None]', str(scope.get_pcollection('Square')))
        self.assertEqual('PCollection[Square.None]', str(scope.get_pcollection('LogForTesting')))
        self.assertTrue(scope.get_pcollection('Square') == scope.get_pcollection('LogForTesting'))

    def test_create_ptransform(self):
        if False:
            while True:
                i = 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            spec = '\n        transforms:\n          - type: Create\n            config:\n              elements: [1, 2, 3]\n        '
            (scope, spec) = self.get_scope_by_spec(p, spec)
            result = scope.create_ptransform(spec['transforms'][0], [])
            self.assertIsInstance(result, beam.transforms.Create)
            self.assertEqual(result.label, 'Create')
            result_annotations = {**result.annotations()}
            target_annotations = {'yaml_type': 'Create', 'yaml_args': '{"elements": [1, 2, 3]}', 'yaml_provider': '{"type": "InlineProvider"}'}
            self.assertDictEqual(result_annotations, {**result_annotations, **target_annotations})

class TestProvider(yaml_provider.InlineProvider):

    def __init__(self, transform, name):
        if False:
            for i in range(10):
                print('nop')
        super().__init__({name: lambda : beam.Map(lambda x: (x or ()) + (name,)), transform: lambda : beam.Map(lambda x: (x or ()) + (name,))})
        self._transform = transform
        self._name = name

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'TestProvider(%r, %r)' % (self._transform, self._name)

    def _affinity(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, TestProvider):
            affinity = 1
            for (x, y) in zip(self._name, other._name):
                if x != y:
                    break
                affinity *= 10
            return affinity
        else:
            return -1000

class ProviderAffinityTest(unittest.TestCase):

    @staticmethod
    def create_scope(s, providers):
        if False:
            i = 10
            return i + 15
        providers_dict = collections.defaultdict(list)
        for provider in providers:
            for transform_type in provider.provided_transforms():
                providers_dict[transform_type].append(provider)
        spec = yaml_transform.preprocess(yaml.load(s, Loader=SafeLineLoader))
        return Scope(None, {}, transforms=spec['transforms'], providers=providers_dict, input_providers={})

    def test_best_provider_based_on_input(self):
        if False:
            while True:
                i = 10
        provider_Ax = TestProvider('A', 'xxx')
        provider_Ay = TestProvider('A', 'yyy')
        provider_Bx = TestProvider('B', 'xxz')
        provider_By = TestProvider('B', 'yyz')
        scope = self.create_scope('\n        type: chain\n        transforms:\n          - type: A\n          - type: B\n        ', [provider_Ax, provider_Ay, provider_Bx, provider_By])
        self.assertEqual(scope.best_provider('B', [provider_Ax]), provider_Bx)
        self.assertEqual(scope.best_provider('B', [provider_Ay]), provider_By)

    def test_best_provider_based_on_followers(self):
        if False:
            print('Hello World!')
        close_provider = TestProvider('A', 'xxy')
        far_provider = TestProvider('A', 'yyy')
        following_provider = TestProvider('B', 'xxx')
        scope = self.create_scope('\n        type: chain\n        transforms:\n          - type: A\n          - type: B\n        ', [far_provider, close_provider, following_provider])
        self.assertEqual(scope.best_provider('A', []), close_provider)

    def test_best_provider_based_on_multiple_followers(self):
        if False:
            return 10
        close_provider = TestProvider('A', 'xxy')
        provider_B = TestProvider('B', 'xxx')
        far_provider = TestProvider('A', 'yyy')
        provider_C = TestProvider('C', 'yzz')
        scope = self.create_scope('\n        type: composite\n        transforms:\n          - type: A\n          - type: B\n            input: A\n          - type: C\n            input: A\n        ', [far_provider, close_provider, provider_B, provider_C])
        self.assertEqual(scope.best_provider('A', []), close_provider)

    def test_best_provider_based_on_distant_follower(self):
        if False:
            i = 10
            return i + 15
        providers = [TestProvider('A', 'xxx'), TestProvider('A', 'yyy'), TestProvider('B', 'xxx'), TestProvider('B', 'yyy'), TestProvider('C', 'xxx'), TestProvider('C', 'yyy'), TestProvider('D', 'xxx'), TestProvider('E', 'yyy')]
        scope = self.create_scope('\n        type: chain\n        transforms:\n          - type: A\n          - type: B\n          - type: C\n          - type: D\n        ', providers)
        self.assertEqual(scope.best_provider('A', []), providers[0])
        scope = self.create_scope('\n        type: chain\n        transforms:\n          - type: A\n          - type: B\n          - type: C\n          - type: E\n        ', providers)
        self.assertEqual(scope.best_provider('A', []), providers[1])
        scope = self.create_scope('\n        type: chain\n        transforms:\n          - type: A\n          - type: B\n          - type: C\n          - type: D\n          - type: E\n        ', providers)
        self.assertEqual(scope.best_provider('A', []), providers[0])

class LightweightScopeTest(unittest.TestCase):

    @staticmethod
    def get_spec():
        if False:
            print('Hello World!')
        pipeline_yaml = '\n          - type: PyMap\n            name: Square\n            input: elements\n            fn: "lambda x: x * x"\n          - type: PyMap\n            name: PyMap\n            input: elements\n            fn: "lambda x: x * x * x"\n          - type: Filter\n            name: FilterOutBigNumbers\n            input: PyMap\n            keep: "lambda x: x<100"\n          '
        return yaml.load(pipeline_yaml, Loader=SafeLineLoader)

    def test_init(self):
        if False:
            return 10
        spec = self.get_spec()
        scope = LightweightScope(spec)
        self.assertEqual(len(scope._transforms_by_uuid), 3)
        self.assertCountEqual(list(scope._uuid_by_name.keys()), ['PyMap', 'Square', 'Filter', 'FilterOutBigNumbers'])

    def test_get_transform_id_and_output_name(self):
        if False:
            for i in range(10):
                print('nop')
        spec = self.get_spec()
        scope = LightweightScope(spec)
        (transform_id, output) = scope.get_transform_id_and_output_name('Square')
        self.assertEqual(transform_id, spec[0]['__uuid__'])
        self.assertEqual(output, None)

    def test_get_transform_id_and_output_name_with_dot(self):
        if False:
            return 10
        spec = self.get_spec()
        scope = LightweightScope(spec)
        (transform_id, output) = scope.get_transform_id_and_output_name('Square.OutputName')
        self.assertEqual(transform_id, spec[0]['__uuid__'])
        self.assertEqual(output, 'OutputName')

    def test_get_transform_id_by_uuid(self):
        if False:
            while True:
                i = 10
        spec = self.get_spec()
        scope = LightweightScope(spec)
        transform_id = scope.get_transform_id(spec[0]['__uuid__'])
        self.assertEqual(spec[0]['__uuid__'], transform_id)

    def test_get_transform_id_by_unique_name(self):
        if False:
            while True:
                i = 10
        spec = self.get_spec()
        scope = LightweightScope(spec)
        transform_id = scope.get_transform_id('Square')
        self.assertEqual(transform_id, spec[0]['__uuid__'])

    def test_get_transform_id_by_ambiguous_name(self):
        if False:
            return 10
        spec = self.get_spec()
        scope = LightweightScope(spec)
        with self.assertRaisesRegex(ValueError, 'Ambiguous.*PyMap'):
            scope.get_transform_id(scope.get_transform_id(spec[1]['name']))

    def test_get_transform_id_by_unknown_name(self):
        if False:
            print('Hello World!')
        spec = self.get_spec()
        scope = LightweightScope(spec)
        with self.assertRaisesRegex(ValueError, 'Unknown.*NotExistingTransform'):
            scope.get_transform_id('NotExistingTransform')
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()