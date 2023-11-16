import logging
import unittest
import apache_beam as beam
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.yaml.yaml_transform import YamlTransform
DATA = [beam.Row(a='x', b=1, c=101), beam.Row(a='x', b=1, c=102), beam.Row(a='y', b=1, c=103), beam.Row(a='y', b=2, c=104)]

class YamlCombineTest(unittest.TestCase):

    def test_multiple_aggregations(self):
        if False:
            i = 10
            return i + 15
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle', yaml_experimental_features=['Combine'])) as p:
            elements = p | beam.Create(DATA)
            result = elements | YamlTransform('\n          type: Combine\n          config:\n            group_by: a\n            combine:\n              b: sum\n              c: max\n          ')
            assert_that(result | beam.Map(lambda x: beam.Row(**x._asdict())), equal_to([beam.Row(a='x', b=2, c=102), beam.Row(a='y', b=3, c=104)]))

    def test_multiple_keys(self):
        if False:
            while True:
                i = 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle', yaml_experimental_features=['Combine'])) as p:
            elements = p | beam.Create(DATA)
            result = elements | YamlTransform('\n          type: Combine\n          config:\n            group_by: [a, b]\n            combine:\n              c: sum\n          ')
            assert_that(result | beam.Map(lambda x: beam.Row(**x._asdict())), equal_to([beam.Row(a='x', b=1, c=203), beam.Row(a='y', b=1, c=103), beam.Row(a='y', b=2, c=104)]))

    def test_no_keys(self):
        if False:
            for i in range(10):
                print('nop')
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle', yaml_experimental_features=['Combine'])) as p:
            elements = p | beam.Create(DATA)
            result = elements | YamlTransform('\n          type: Combine\n          config:\n            group_by: []\n            combine:\n              c: sum\n          ')
            assert_that(result | beam.Map(lambda x: beam.Row(**x._asdict())), equal_to([beam.Row(c=410)]))

    def test_multiple_combines(self):
        if False:
            print('Hello World!')
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle', yaml_experimental_features=['Combine'])) as p:
            elements = p | beam.Create(DATA)
            result = elements | YamlTransform('\n          type: Combine\n          config:\n            group_by: a\n            combine:\n              min_c:\n                fn: min\n                value: c\n              max_c:\n                fn: max\n                value: c\n          ')
            assert_that(result | beam.Map(lambda x: beam.Row(**x._asdict())), equal_to([beam.Row(a='x', min_c=101, max_c=102), beam.Row(a='y', min_c=103, max_c=104)]))

    def test_expression(self):
        if False:
            for i in range(10):
                print('nop')
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle', yaml_experimental_features=['Combine'])) as p:
            elements = p | beam.Create(DATA)
            result = elements | YamlTransform('\n          type: Combine\n          config:\n            language: python\n            group_by: a\n            combine:\n              max:\n                fn: max\n                value: b + c\n          ')
            assert_that(result | beam.Map(lambda x: beam.Row(**x._asdict())), equal_to([beam.Row(a='x', max=103), beam.Row(a='y', max=106)]))

    def test_config(self):
        if False:
            while True:
                i = 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle', yaml_experimental_features=['Combine'])) as p:
            elements = p | beam.Create(DATA)
            result = elements | YamlTransform("\n          type: Combine\n          config:\n            language: python\n            group_by: b\n            combine:\n              biggest:\n                fn:\n                  type: 'apache_beam.transforms.combiners.TopCombineFn'\n                  config:\n                    n: 2\n                value: c\n          ")
            assert_that(result | beam.Map(lambda x: beam.Row(**x._asdict())), equal_to([beam.Row(b=1, biggest=[103, 102]), beam.Row(b=2, biggest=[104])]))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()