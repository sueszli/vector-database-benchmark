import logging
import unittest
import yaml
import apache_beam as beam
from apache_beam.yaml import YamlTransform
from apache_beam.yaml import yaml_provider
from apache_beam.yaml.yaml_provider import InlineProvider
from apache_beam.yaml.yaml_transform import SafeLineLoader
from apache_beam.yaml.yaml_transform import Scope
from apache_beam.yaml.yaml_transform import chain_as_composite
from apache_beam.yaml.yaml_transform import ensure_errors_consumed
from apache_beam.yaml.yaml_transform import ensure_transforms_have_types
from apache_beam.yaml.yaml_transform import expand_composite_transform
from apache_beam.yaml.yaml_transform import extract_name
from apache_beam.yaml.yaml_transform import identify_object
from apache_beam.yaml.yaml_transform import normalize_inputs_outputs
from apache_beam.yaml.yaml_transform import normalize_source_sink
from apache_beam.yaml.yaml_transform import only_element
from apache_beam.yaml.yaml_transform import pipeline_as_composite
from apache_beam.yaml.yaml_transform import preprocess
from apache_beam.yaml.yaml_transform import preprocess_flattened_inputs
from apache_beam.yaml.yaml_transform import preprocess_windowing
from apache_beam.yaml.yaml_transform import push_windowing_to_roots

class SafeLineLoaderTest(unittest.TestCase):

    def test_get_line(self):
        if False:
            print('Hello World!')
        pipeline_yaml = '\n          type: composite\n          input:\n              elements: input\n          transforms:\n            - type: PyMap\n              name: Square\n              input: elements\n              config:\n                fn: "lambda x: x * x"\n            - type: PyMap\n              name: Cube\n              input: elements\n              config:\n                fn: "lambda x: x * x * x"\n          output:\n              Flatten\n          '
        spec = yaml.load(pipeline_yaml, Loader=SafeLineLoader)
        self.assertEqual(SafeLineLoader.get_line(spec['type']), 2)
        self.assertEqual(SafeLineLoader.get_line(spec['input']), 4)
        self.assertEqual(SafeLineLoader.get_line(spec['transforms'][0]), 6)
        self.assertEqual(SafeLineLoader.get_line(spec['transforms'][0]['type']), 6)
        self.assertEqual(SafeLineLoader.get_line(spec['transforms'][0]['name']), 7)
        self.assertEqual(SafeLineLoader.get_line(spec['transforms'][1]), 11)
        self.assertEqual(SafeLineLoader.get_line(spec['output']), 17)
        self.assertEqual(SafeLineLoader.get_line(spec['transforms']), 'unknown')

    def test_strip_metadata(self):
        if False:
            for i in range(10):
                print('nop')
        spec_yaml = '\n    transforms:\n      - type: PyMap\n        name: Square\n    '
        spec = yaml.load(spec_yaml, Loader=SafeLineLoader)
        stripped = SafeLineLoader.strip_metadata(spec['transforms'])
        self.assertFalse(hasattr(stripped[0], '__line__'))
        self.assertFalse(hasattr(stripped[0], '__uuid__'))

    def test_strip_metadata_nothing_to_strip(self):
        if False:
            return 10
        spec_yaml = 'prop: 123'
        spec = yaml.load(spec_yaml, Loader=SafeLineLoader)
        stripped = SafeLineLoader.strip_metadata(spec['prop'])
        self.assertFalse(hasattr(stripped, '__line__'))
        self.assertFalse(hasattr(stripped, '__uuid__'))

def new_pipeline():
    if False:
        i = 10
        return i + 15
    return beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle'))

class MainTest(unittest.TestCase):

    def assertYaml(self, expected, result):
        if False:
            return 10
        result = SafeLineLoader.strip_metadata(result)
        expected = yaml.load(expected, Loader=SafeLineLoader)
        expected = SafeLineLoader.strip_metadata(expected)
        self.assertEqual(expected, result)

    def get_scope_by_spec(self, p, spec, inputs=None):
        if False:
            i = 10
            return i + 15
        if inputs is None:
            inputs = {}
        spec = yaml.load(spec, Loader=SafeLineLoader)
        scope = Scope(beam.pvalue.PBegin(p), inputs, spec['transforms'], yaml_provider.standard_providers(), {})
        return (scope, spec)

    def test_pipeline_as_composite_with_type_transforms(self):
        if False:
            print('Hello World!')
        spec = "\n      type: composite\n      transforms:\n      - type: Create\n        config:\n          elements: [0,1,2]\n      - type: PyMap\n        config:\n          fn: 'lambda x: x*x'\n      "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        result = pipeline_as_composite(spec)
        self.assertEqual(result['type'], 'composite')
        self.assertEqual(result['name'], None)

    def test_pipeline_as_composite_with_transforms(self):
        if False:
            for i in range(10):
                print('nop')
        spec = "\n      transforms:\n      - type: Create\n        config:\n          elements: [0,1,2]\n      - type: PyMap\n        config:\n          fn: 'lambda x: x*x'\n      "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        result = pipeline_as_composite(spec)
        self.assertEqual(result['type'], 'composite')
        self.assertEqual(result['name'], None)

    def test_pipeline_as_composite_list(self):
        if False:
            print('Hello World!')
        spec = "\n        - type: Create\n          config:\n            elements: [0,1,2]\n        - type: PyMap\n          config:\n            fn: 'lambda x: x*x'\n      "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        result = pipeline_as_composite(spec)
        expected = "\n      type: composite\n      name: null\n      transforms:\n      - type: Create\n        config:\n          elements: [0,1,2]\n      - type: PyMap\n        config:\n          fn: 'lambda x: x*x'\n    "
        self.assertYaml(expected, result)

    def test_expand_composite_transform_with_name(self):
        if False:
            print('Hello World!')
        with new_pipeline() as p:
            spec = '\n        type: composite\n        name: Custom\n        transforms:\n          - type: Create\n            config:\n              elements: [0,1,2]\n        output:\n          Create\n        '
            (scope, spec) = self.get_scope_by_spec(p, spec)
            self.assertRegex(str(expand_composite_transform(spec, scope)['output']), 'PCollection.*Custom/Create/Map.*')

    def test_expand_composite_transform_with_name_input(self):
        if False:
            return 10
        with new_pipeline() as p:
            spec = '\n        type: composite\n        input: elements\n        transforms:\n          - type: LogForTesting\n            input: input\n        output:\n          LogForTesting\n        '
            elements = p | beam.Create(range(3))
            (scope, spec) = self.get_scope_by_spec(p, spec, inputs={'elements': elements})
            self.assertRegex(str(expand_composite_transform(spec, scope)['output']), 'PCollection.*Composite/Map.*')

    def test_expand_composite_transform_root(self):
        if False:
            while True:
                i = 10
        with new_pipeline() as p:
            spec = '\n        type: composite\n        transforms:\n          - type: Create\n            config:\n              elements: [0,1,2]\n        output:\n          Create\n        '
            (scope, spec) = self.get_scope_by_spec(p, spec)
            self.assertRegex(str(expand_composite_transform(spec, scope)['output']), 'PCollection.*Create/Map.*')

    def test_chain_as_composite(self):
        if False:
            return 10
        spec = "\n        type: chain\n        transforms:\n        - type: Create\n          config:\n            elements: [0,1,2]\n        - type: PyMap\n          config:\n            fn: 'lambda x: x*x'\n      "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        result = chain_as_composite(spec)
        expected = f"\n      type: composite\n      name: Chain\n      transforms:\n      - type: Create\n        config:\n          elements: [0,1,2]\n      - type: PyMap\n        config:\n          fn: 'lambda x: x*x'\n        input: {spec['transforms'][0]['__uuid__']}\n      output: {spec['transforms'][1]['__uuid__']}\n    "
        self.assertYaml(expected, result)

    def test_chain_as_composite_with_wrong_output_type(self):
        if False:
            print('Hello World!')
        spec = "\n        type: chain\n        transforms:\n        - type: Create\n          elements: [0,1,2]\n        - type: PyMap\n          fn: 'lambda x: x*x'\n        output:\n          Create\n      "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        with self.assertRaisesRegex(ValueError, 'Explicit output.*of the chain transform is not an output of the last transform'):
            chain_as_composite(spec)

    def test_chain_as_composite_with_wrong_output_name(self):
        if False:
            return 10
        spec = "\n        type: chain\n        transforms:\n        - type: Create\n          name: elements\n          elements: [0,1,2]\n        - type: PyMap\n          fn: 'lambda x: x*x'\n        output:\n          elements\n      "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        with self.assertRaisesRegex(ValueError, 'Explicit output.*of the chain transform is not an output of the last transform'):
            chain_as_composite(spec)

    def test_chain_as_composite_with_outputs_override(self):
        if False:
            i = 10
            return i + 15
        spec = "\n        type: chain\n        transforms:\n        - type: Create\n          elements: [0,1,2]\n        - type: PyMap\n          fn: 'lambda x: x*x'\n        output:\n          PyMap\n      "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        self.assertEqual(chain_as_composite(spec)['output']['output'], f"{spec['transforms'][1]['__uuid__']}.PyMap")

    def test_chain_as_composite_with_input(self):
        if False:
            print('Hello World!')
        spec = "\n        type: chain\n        input:\n          elements\n        transforms:\n        - type: PyMap\n          fn: 'lambda x: x*x'\n      "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        self.assertEqual(chain_as_composite(spec)['transforms'][0]['input'], {'input': 'input'})

    def test_normalize_source_sink(self):
        if False:
            while True:
                i = 10
        spec = '\n        source:\n          type: Create\n          config:\n            elements: [0,1,2]\n        transforms:\n        - type: PyMap\n          config:\n            fn: \'lambda x: x*x\'\n        sink:\n          type: PyMap\n          config:\n            fn: "lambda x: x + 41"\n      '
        spec = yaml.load(spec, Loader=SafeLineLoader)
        result = normalize_source_sink(spec)
        expected = '\n      transforms:\n      - type: Create\n        input: {\'__explicitly_empty__\': null}\n        config:\n          elements: [0,1,2]\n      - type: PyMap\n        config:\n          fn: \'lambda x: x*x\'\n      - type: PyMap\n        config:\n          fn: "lambda x: x + 41"\n    '
        self.assertYaml(expected, result)

    def test_normalize_source_sink_only_source(self):
        if False:
            return 10
        spec = "\n        source:\n          type: Create\n          config:\n            elements: [0,1,2]\n        transforms:\n        - type: PyMap\n          config:\n            fn: 'lambda x: x*x'\n\n      "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        result = normalize_source_sink(spec)
        expected = "\n      transforms:\n      - type: Create\n        input: {'__explicitly_empty__': null}\n        config:\n          elements: [0,1,2]\n      - type: PyMap\n        config:\n          fn: 'lambda x: x*x'\n    "
        self.assertYaml(expected, result)

    def test_normalize_source_sink_only_sink(self):
        if False:
            while True:
                i = 10
        spec = '\n        transforms:\n        - type: PyMap\n          config:\n            fn: \'lambda x: x*x\'\n        sink:\n          type: PyMap\n          config:\n            fn: "lambda x: x + 41"\n      '
        spec = yaml.load(spec, Loader=SafeLineLoader)
        result = normalize_source_sink(spec)
        expected = '\n      transforms:\n      - type: PyMap\n        config:\n          fn: \'lambda x: x*x\'\n      - type: PyMap\n        config:\n          fn: "lambda x: x + 41"\n    '
        self.assertYaml(expected, result)

    def test_normalize_source_sink_no_source_no_sink(self):
        if False:
            while True:
                i = 10
        spec = "\n        transforms:\n        - type: PyMap\n          config:\n            fn: 'lambda x: x*x'\n      "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        result = normalize_source_sink(spec)
        expected = "\n      transforms:\n      - type: PyMap\n        config:\n          fn: 'lambda x: x*x'\n    "
        self.assertYaml(expected, result)

    def test_preprocess_source_sink_composite(self):
        if False:
            while True:
                i = 10
        spec = "\n      type: composite\n      source:\n        type: Create\n        config:\n          elements: [0,1,2]\n      transforms:\n      - type: PyMap\n        config:\n          fn: 'lambda x: x*x'\n      "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        result = normalize_source_sink(spec)
        expected = "\n      type: composite\n      transforms:\n      - type: Create\n        input: {'__explicitly_empty__': null}\n        config:\n          elements: [0,1,2]\n      - type: PyMap\n        config:\n          fn: 'lambda x: x*x'\n    "
        self.assertYaml(expected, result)

    def test_preprocess_source_sink_chain(self):
        if False:
            i = 10
            return i + 15
        spec = "\n      type: chain\n      source:\n        type: Create\n        config:\n          elements: [0,1,2]\n      transforms:\n      - type: PyMap\n        config:\n          fn: 'lambda x: x*x'\n      "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        result = normalize_source_sink(spec)
        expected = "\n      type: chain\n      transforms:\n      - type: Create\n        input: {'__explicitly_empty__': null}\n        config:\n          elements: [0,1,2]\n      - type: PyMap\n        config:\n          fn: 'lambda x: x*x'\n    "
        self.assertYaml(expected, result)

    def test_preprocess_source_sink_other(self):
        if False:
            while True:
                i = 10
        spec = "\n      - type: PyMap\n        fn: 'lambda x: x*x'\n      "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        self.assertEqual(normalize_source_sink(spec), spec)

    def test_normalize_inputs_outputs(self):
        if False:
            i = 10
            return i + 15
        spec = "\n        type: PyMap\n        input: [Create1, Create2]\n        fn: 'lambda x: x*x'\n        output: Squared\n      "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        result = normalize_inputs_outputs(spec)
        expected = "\n      type: PyMap\n      input:\n        input: [Create1, Create2]\n      fn: 'lambda x: x*x'\n      output:\n        output: Squared\n    "
        self.assertYaml(expected, result)

    def test_normalize_inputs_outputs_dict(self):
        if False:
            for i in range(10):
                print('nop')
        spec = "\n        type: PyMap\n        input: [Create1, Create2]\n        fn: 'lambda x: x*x'\n        output:\n          out1: Squared1\n          out2: Squared2\n      "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        result = normalize_inputs_outputs(spec)
        expected = "\n      type: PyMap\n      input:\n        input: [Create1, Create2]\n      fn: 'lambda x: x*x'\n      output:\n        out1: Squared1\n        out2: Squared2\n    "
        self.assertYaml(expected, result)

    def test_identify_object_with_name(self):
        if False:
            return 10
        spec = "\n      type: PyMap\n      fn: 'lambda x: x*x'\n    "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        self.assertRegex(identify_object(spec), 'PyMap.*[0-9]')

    def test_identify_object(self):
        if False:
            for i in range(10):
                print('nop')
        spec = "\n      argument: PyMap\n      fn: 'lambda x: x*x'\n    "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        self.assertRegex(identify_object(spec), 'at.*[0-9]')

    def test_extract_name_by_type(self):
        if False:
            print('Hello World!')
        spec = "\n      type: PyMap\n      fn: 'lambda x: x*x'\n    "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        self.assertEqual(extract_name(spec), 'PyMap')

    def test_extract_name_by_id(self):
        if False:
            return 10
        spec = "\n      type: PyMap\n      id: PyMapId\n      fn: 'lambda x: x*x'\n    "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        self.assertEqual(extract_name(spec), 'PyMapId')

    def test_extract_name_by_name(self):
        if False:
            while True:
                i = 10
        spec = "\n      type: PyMap\n      name: PyMapName\n      fn: 'lambda x: x*x'\n    "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        self.assertEqual(extract_name(spec), 'PyMapName')

    def test_extract_name_no_name(self):
        if False:
            i = 10
            return i + 15
        spec = "\n      transforms:\n      - arg: PyMap\n        fn: 'lambda x: x*x'\n    "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        self.assertEqual(extract_name(spec), '')

    def test_push_windowing_to_roots(self):
        if False:
            while True:
                i = 10
        spec = "\n      type: composite\n      transforms:\n      - type: Create\n        elements: [0,1,2]\n      - type: PyMap\n        fn: 'lambda x: x*x'\n        input: Create\n      windowing:\n        type: fixed\n        size: 2\n    "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        spec = normalize_inputs_outputs(spec)
        spec['transforms'] = [normalize_inputs_outputs(t) for t in spec['transforms']]
        result = push_windowing_to_roots(spec)
        expected = "\n      type: composite\n      transforms:\n      - type: Create\n        elements: [0,1,2]\n        windowing:\n          type: fixed\n          size: 2\n        __consumed_outputs:\n          - null\n        input: {}\n        output: {}\n      - type: PyMap\n        fn: 'lambda x: x*x'\n        input:\n          input: Create\n        output: {}\n      windowing:\n        type: fixed\n        size: 2\n      input: {}\n      output: {}\n    "
        self.assertYaml(expected, result)

    def test_preprocess_windowing_custom_type(self):
        if False:
            i = 10
            return i + 15
        spec = '\n        type: SumGlobally\n        input: Create\n        windowing:\n          type: fixed\n          size: 4\n    '
        spec = yaml.load(spec, Loader=SafeLineLoader)
        spec = normalize_inputs_outputs(spec)
        result = preprocess_windowing(spec)
        expected = f"\n      type: composite\n      name: SumGlobally\n      input:\n        input: Create\n      transforms:\n        - type: SumGlobally\n          input:\n            input: {result['transforms'][1]['__uuid__']}\n          output: {{}}\n        - type: WindowInto\n          name: WindowInto[input]\n          windowing:\n            type: fixed\n            size: 4\n          input: input\n      output: {result['transforms'][0]['__uuid__']}\n    "
        self.assertYaml(expected, result)

    def test_preprocess_windowing_composite_with_windowing_outer(self):
        if False:
            i = 10
            return i + 15
        spec = '\n      type: composite\n      transforms:\n        - type: CreateTimestamped\n          name: Create\n          elements: [0, 2, 4]\n        - type: SumGlobally\n          input: Create\n      windowing:\n        type: fixed\n        size: 4\n      output: SumGlobally\n    '
        spec = yaml.load(spec, Loader=SafeLineLoader)
        spec = normalize_inputs_outputs(spec)
        spec['transforms'] = [normalize_inputs_outputs(t) for t in spec['transforms']]
        result = preprocess_windowing(spec)
        expected = '\n      type: composite\n      input: {}\n      transforms:\n        - type: CreateTimestamped\n          name: Create\n          elements: [0, 2, 4]\n          windowing:\n            type: fixed\n            size: 4\n          __consumed_outputs:\n            - null\n          input: {}\n          output: {}\n        - type: SumGlobally\n          input:\n            input: Create\n          output: {}\n      output:\n        output: SumGlobally\n    '
        self.assertYaml(expected, result)

    def test_preprocess_windowing_composite_with_windowing_on_input(self):
        if False:
            print('Hello World!')
        spec = '\n      type: composite\n      transforms:\n        - type: CreateTimestamped\n          name: Create\n          elements: [0, 2, 4]\n        - type: SumGlobally\n          input: Create\n          windowing:\n            type: fixed\n            size: 4\n      output: SumGlobally\n    '
        spec = yaml.load(spec, Loader=SafeLineLoader)
        spec = normalize_inputs_outputs(spec)
        spec['transforms'] = [normalize_inputs_outputs(t) for t in spec['transforms']]
        result = preprocess_windowing(spec)
        expected = '\n      type: composite\n      input: {}\n      transforms:\n        - type: CreateTimestamped\n          name: Create\n          elements: [0, 2, 4]\n          input: {}\n          output: {}\n        - type: SumGlobally\n          input:\n            input: Create\n          windowing:\n            type: fixed\n            size: 4\n          output: {}\n      output:\n        output: SumGlobally\n    '
        self.assertYaml(expected, result)

    def test_preprocess_windowing_other_type_with_no_inputs(self):
        if False:
            return 10
        spec = '\n      type: CreateTimestamped\n      name: Create\n      elements: [0, 2, 4]\n      windowing:\n        type: fixed\n        size: 4\n    '
        spec = yaml.load(spec, Loader=SafeLineLoader)
        spec = normalize_inputs_outputs(spec)
        result = preprocess_windowing(spec)
        expected = f"\n      type: composite\n      name: Create\n      transforms:\n        - type: CreateTimestamped\n          name: Create\n          elements: [0, 2, 4]\n          input: {{}}\n          output: {{}}\n        - type: WindowInto\n          name: WindowInto[None]\n          input:\n            input: {result['transforms'][0]['__uuid__']}\n          windowing:\n            type: fixed\n            size: 4\n      output: {result['transforms'][1]['__uuid__']}\n    "
        self.maxDiff = 1000000000.0
        self.assertYaml(expected, result)

    def test_preprocess_flattened_inputs_implicit(self):
        if False:
            i = 10
            return i + 15
        spec = "\n      type: composite\n      transforms:\n        - type: PyMap\n          fn: 'lambda x: x*x'\n          input: [Create1, Create2]\n      output: CreateTimestamped\n    "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        spec['transforms'] = [normalize_inputs_outputs(t) for t in spec['transforms']]
        result = preprocess_flattened_inputs(spec)
        expected = f"\n      type: composite\n      transforms:\n        - type: Flatten\n          name: PyMap-Flatten[input]\n          input:\n            input0: Create1\n            input1: Create2\n        - type: PyMap\n          fn: 'lambda x: x*x'\n          input:\n            input: {result['transforms'][0]['__uuid__']}\n          output: {{}}\n      output: CreateTimestamped\n    "
        self.assertYaml(expected, result)

    def test_preprocess_flattened_inputs_explicit_flatten(self):
        if False:
            i = 10
            return i + 15
        spec = "\n      type: composite\n      transforms:\n        - type: Flatten\n          input: [Create1, Create2]\n        - type: PyMap\n          fn: 'lambda x: x*x'\n          input: Flatten\n      output: CreateTimestamped\n    "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        spec['transforms'] = [normalize_inputs_outputs(t) for t in spec['transforms']]
        result = preprocess_flattened_inputs(spec)
        expected = "\n      type: composite\n      transforms:\n        - type: Flatten\n          input:\n            input0: Create1\n            input1: Create2\n          output: {}\n        - type: PyMap\n          fn: 'lambda x: x*x'\n          input:\n            input: Flatten\n          output: {}\n      output: CreateTimestamped\n    "
        self.assertYaml(expected, result)

    def test_ensure_transforms_have_types(self):
        if False:
            return 10
        spec = "\n      type: PyMap\n      fn: 'lambda x: x*x'\n      input: Flatten\n    "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        result = ensure_transforms_have_types(spec)
        self.assertEqual(result, spec)

    def test_ensure_transforms_have_types_error(self):
        if False:
            i = 10
            return i + 15
        spec = "\n      name: PyMap\n      fn: 'lambda x: x*x'\n      input: Flatten\n    "
        spec = yaml.load(spec, Loader=SafeLineLoader)
        with self.assertRaisesRegex(ValueError, 'Missing type .*'):
            ensure_transforms_have_types(spec)
        with self.assertRaisesRegex(ValueError, 'Missing type .*'):
            preprocess(spec)

    def test_ensure_transforms_have_providers_error(self):
        if False:
            return 10
        spec = '\n      type: UnknownType\n    '
        spec = yaml.load(spec, Loader=SafeLineLoader)
        with self.assertRaisesRegex(ValueError, 'Unknown type or missing provider .*'):
            preprocess(spec, known_transforms=['KnownType'])

    def test_ensure_errors_consumed_unconsumed(self):
        if False:
            while True:
                i = 10
        spec = '\n      type: composite\n      transforms:\n        - type: Create\n          elements: [1,2,3]\n        - type: MyTransform\n          input: Create\n          error_handling:\n            output: errors\n      output:\n        good: MyTransform\n    '
        spec = yaml.load(spec, Loader=SafeLineLoader)
        spec = normalize_inputs_outputs(spec)
        spec['transforms'] = [normalize_inputs_outputs(t) for t in spec['transforms']]
        with self.assertRaisesRegex(ValueError, 'Unconsumed error.*'):
            ensure_errors_consumed(spec)

    def test_ensure_errors_consumed_in_transform(self):
        if False:
            print('Hello World!')
        spec = '\n      type: composite\n      transforms:\n        - type: Create\n          elements: [1,2,3]\n        - type: MyTransform\n          input: Create\n          error_handling:\n            output: errors\n        - name: SaveToFile\n          type: PyMap\n          input: MyTransform.errors\n      output:\n        good: MyTransform\n    '
        spec = yaml.load(spec, Loader=SafeLineLoader)
        spec = normalize_inputs_outputs(spec)
        spec['transforms'] = [normalize_inputs_outputs(t) for t in spec['transforms']]
        result = ensure_errors_consumed(spec)
        self.assertEqual(result, spec)
        self.assertEqual(result['transforms'], spec['transforms'])

    def test_ensure_errors_consumed_no_output_in_error_handling(self):
        if False:
            return 10
        spec = '\n      type: composite\n      transforms:\n        - type: Create\n          elements: [1,2,3]\n        - type: MyTransform\n          input: Create\n          error_handling:\n            arg: errors\n        - name: SaveToFile\n          type: PyMap\n          input: MyTransform.errors\n      output:\n        good: MyTransform\n    '
        spec = yaml.load(spec, Loader=SafeLineLoader)
        spec = normalize_inputs_outputs(spec)
        spec['transforms'] = [normalize_inputs_outputs(t) for t in spec['transforms']]
        with self.assertRaisesRegex(ValueError, 'Missing output.*'):
            ensure_errors_consumed(spec)

    def test_only_element(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(only_element((1,)), 1)

class YamlTransformTest(unittest.TestCase):

    def test_init_with_string(self):
        if False:
            while True:
                i = 10
        provider1 = InlineProvider({'MyTransform1': lambda : beam.Map(lambda x: x)})
        provider2 = InlineProvider({'MyTransform2': lambda : beam.Map(lambda x: x)})
        providers_dict = {'p1': [provider1], 'p2': [provider2]}
        spec = '\n        type: chain\n        transforms:\n          - type: Create\n            elements: [1,2,3]\n          - type: LogForTesting\n      '
        result = YamlTransform(spec, providers_dict)
        self.assertIn('p1', result._providers)
        self.assertIn('p2', result._providers)
        self.assertIn('LogForTesting', result._providers)
        self.assertEqual(result._spec['type'], 'composite')

    def test_init_with_dict(self):
        if False:
            for i in range(10):
                print('nop')
        spec = '\n        type: chain\n        transforms:\n          - type: Create\n            config:\n              elements: [1,2,3]\n          - type: LogForTesting\n      '
        spec = yaml.load(spec, Loader=SafeLineLoader)
        result = YamlTransform(spec)
        self.assertIn('LogForTesting', result._providers)
        self.assertEqual(result._spec['type'], 'composite')
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()