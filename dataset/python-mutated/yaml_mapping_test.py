import logging
import os
import unittest
import apache_beam as beam
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.yaml.readme_test import createTestSuite
from apache_beam.yaml.yaml_transform import YamlTransform
DATA = [beam.Row(label='11a', conductor=11, rank=0), beam.Row(label='37a', conductor=37, rank=1), beam.Row(label='389a', conductor=389, rank=2)]

class YamlMappingTest(unittest.TestCase):

    def test_basic(self):
        if False:
            while True:
                i = 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            elements = p | beam.Create(DATA)
            result = elements | YamlTransform('\n          type: MapToFields\n          config:\n              language: python\n              fields:\n                label: label\n                isogeny: "label[-1]"\n          ')
            assert_that(result, equal_to([beam.Row(label='11a', isogeny='a'), beam.Row(label='37a', isogeny='a'), beam.Row(label='389a', isogeny='a')]))

    def test_drop(self):
        if False:
            i = 10
            return i + 15
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            elements = p | beam.Create(DATA)
            result = elements | YamlTransform('\n          type: MapToFields\n          config:\n              fields: {}\n              append: true\n              drop: [conductor]\n          ')
            assert_that(result, equal_to([beam.Row(label='11a', rank=0), beam.Row(label='37a', rank=1), beam.Row(label='389a', rank=2)]))

    def test_filter(self):
        if False:
            print('Hello World!')
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            elements = p | beam.Create(DATA)
            result = elements | YamlTransform('\n          type: Filter\n          config:\n              language: python\n              keep: "rank > 0"\n          ')
            assert_that(result | beam.Map(lambda named_tuple: beam.Row(**named_tuple._asdict())), equal_to([beam.Row(label='37a', conductor=37, rank=1), beam.Row(label='389a', conductor=389, rank=2)]))

    def test_explode(self):
        if False:
            while True:
                i = 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            elements = p | beam.Create([beam.Row(a=2, b='abc', c=0.25), beam.Row(a=3, b='xy', c=0.125)])
            result = elements | YamlTransform('\n          type: chain\n          transforms:\n            - type: MapToFields\n              config:\n                  language: python\n                  append: true\n                  fields:\n                    range: "range(a)"\n            - type: Explode\n              config:\n                  fields: [range, b]\n                  cross_product: true\n          ')
            assert_that(result, equal_to([beam.Row(a=2, b='a', c=0.25, range=0), beam.Row(a=2, b='a', c=0.25, range=1), beam.Row(a=2, b='b', c=0.25, range=0), beam.Row(a=2, b='b', c=0.25, range=1), beam.Row(a=2, b='c', c=0.25, range=0), beam.Row(a=2, b='c', c=0.25, range=1), beam.Row(a=3, b='x', c=0.125, range=0), beam.Row(a=3, b='x', c=0.125, range=1), beam.Row(a=3, b='x', c=0.125, range=2), beam.Row(a=3, b='y', c=0.125, range=0), beam.Row(a=3, b='y', c=0.125, range=1), beam.Row(a=3, b='y', c=0.125, range=2)]))

    def test_validate_explicit_types(self):
        if False:
            return 10
        with self.assertRaisesRegex(TypeError, '.*violates schema.*'):
            with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
                elements = p | beam.Create([beam.Row(a=2, b='abc', c=0.25), beam.Row(a=3, b='xy', c=0.125)])
                result = elements | YamlTransform('\n            type: MapToFields\n            input: input\n            config:\n              language: python\n              fields:\n                bad:\n                  expression: "a + c"\n                  output_type: string  # This is a lie.\n            ')
                self.assertEqual(result.element_type._fields[0][1], str)
YamlMappingDocTest = createTestSuite('YamlMappingDocTest', os.path.join(os.path.dirname(__file__), 'yaml_mapping.md'))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()