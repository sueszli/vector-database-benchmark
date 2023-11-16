import logging
import os
import shutil
import tempfile
import unittest
import apache_beam as beam
from apache_beam.io import localfilesystem
from apache_beam.options import pipeline_options
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.yaml.yaml_mapping import py_value_to_js_dict
from apache_beam.yaml.yaml_provider import dicts_to_rows
from apache_beam.yaml.yaml_transform import YamlTransform

def AsRows():
    if False:
        return 10
    return beam.Map(lambda named_tuple: dicts_to_rows(py_value_to_js_dict(named_tuple)))

class YamlUDFMappingTest(unittest.TestCase):

    def __init__(self, method_name='runYamlMappingTest'):
        if False:
            i = 10
            return i + 15
        super().__init__(method_name)
        self.data = [beam.Row(label='11a', conductor=11, row=beam.Row(rank=0, values=[1, 2, 3])), beam.Row(label='37a', conductor=37, row=beam.Row(rank=1, values=[4, 5, 6])), beam.Row(label='389a', conductor=389, row=beam.Row(rank=2, values=[7, 8, 9]))]

    def setUp(self):
        if False:
            while True:
                i = 10
        self.tmpdir = tempfile.mkdtemp()
        self.fs = localfilesystem.LocalFileSystem(pipeline_options)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        shutil.rmtree(self.tmpdir)

    def test_map_to_fields_filter_inline_js(self):
        if False:
            while True:
                i = 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle', yaml_experimental_features=['javascript'])) as p:
            elements = p | beam.Create(self.data)
            result = elements | YamlTransform("\n      type: MapToFields\n      config:\n        language: javascript\n        fields:\n          label:\n            callable: |\n              function label_map(x) {\n                return x.label + 'x'\n              }\n          conductor:\n            callable: |\n              function conductor_map(x) {\n                return x.conductor + 1\n              }\n          row:\n            callable: |\n              function row_map(x) {\n                x.row.values.push(x.row.rank + 10)\n                return x.row\n              }\n      ")
            assert_that(result, equal_to([beam.Row(label='11ax', conductor=12, row=beam.Row(rank=0, values=[1, 2, 3, 10])), beam.Row(label='37ax', conductor=38, row=beam.Row(rank=1, values=[4, 5, 6, 11])), beam.Row(label='389ax', conductor=390, row=beam.Row(rank=2, values=[7, 8, 9, 12]))]))

    def test_map_to_fields_filter_inline_py(self):
        if False:
            while True:
                i = 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            elements = p | beam.Create(self.data)
            result = elements | YamlTransform('\n      type: MapToFields\n      config:\n        language: python\n        fields:\n          label:\n            callable: "lambda x: x.label + \'x\'"\n          conductor:\n            callable: "lambda x: x.conductor + 1"\n          sum:\n            callable: "lambda x: sum(x.row.values)"\n      ')
            assert_that(result, equal_to([beam.Row(label='11ax', conductor=12, sum=6), beam.Row(label='37ax', conductor=38, sum=15), beam.Row(label='389ax', conductor=390, sum=24)]))

    def test_filter_inline_js(self):
        if False:
            return 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle', yaml_experimental_features=['javascript'])) as p:
            elements = p | beam.Create(self.data)
            result = elements | YamlTransform('\n      type: Filter\n      config:\n        language: javascript\n        keep:\n          callable: |\n            function filter(x) {\n              return x.row.rank > 0\n            }\n      ')
            assert_that(result | AsRows(), equal_to([beam.Row(label='37a', conductor=37, row=beam.Row(rank=1, values=[4, 5, 6])), beam.Row(label='389a', conductor=389, row=beam.Row(rank=2, values=[7, 8, 9]))]))

    def test_filter_inline_py(self):
        if False:
            for i in range(10):
                print('nop')
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            elements = p | beam.Create(self.data)
            result = elements | YamlTransform('\n      type: Filter\n      config:\n        language: python\n        keep:\n          callable: "lambda x: x.row.rank > 0"\n      ')
            assert_that(result | AsRows(), equal_to([beam.Row(label='37a', conductor=37, row=beam.Row(rank=1, values=[4, 5, 6])), beam.Row(label='389a', conductor=389, row=beam.Row(rank=2, values=[7, 8, 9]))]))

    def test_filter_expression_js(self):
        if False:
            while True:
                i = 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle', yaml_experimental_features=['javascript'])) as p:
            elements = p | beam.Create(self.data)
            result = elements | YamlTransform('\n      type: Filter\n      config:\n        language: javascript\n        keep:\n          expression: "label.toUpperCase().indexOf(\'3\') == -1 && row.rank < 1"\n      ')
            assert_that(result | AsRows(), equal_to([beam.Row(label='11a', conductor=11, row=beam.Row(rank=0, values=[1, 2, 3]))]))

    def test_filter_expression_py(self):
        if False:
            return 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            elements = p | beam.Create(self.data)
            result = elements | YamlTransform('\n      type: Filter\n      config:\n        language: python\n        keep:\n          expression: "\'3\' not in label"\n      ')
            assert_that(result | AsRows(), equal_to([beam.Row(label='11a', conductor=11, row=beam.Row(rank=0, values=[1, 2, 3]))]))

    def test_filter_inline_js_file(self):
        if False:
            return 10
        data = '\n    function f(x) {\n      return x.row.rank > 0\n    }\n\n    function g(x) {\n      return x.row.rank > 1\n    }\n    '.replace('    ', '')
        path = os.path.join(self.tmpdir, 'udf.js')
        self.fs.create(path).write(data.encode('utf8'))
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle', yaml_experimental_features=['javascript'])) as p:
            elements = p | beam.Create(self.data)
            result = elements | YamlTransform(f'\n        type: Filter\n        config:\n          language: javascript\n          keep:\n            path: {path}\n            name: "f"\n        ')
            assert_that(result | AsRows(), equal_to([beam.Row(label='37a', conductor=37, row=beam.Row(rank=1, values=[4, 5, 6])), beam.Row(label='389a', conductor=389, row=beam.Row(rank=2, values=[7, 8, 9]))]))

    def test_filter_inline_py_file(self):
        if False:
            return 10
        data = '\n    def f(x):\n      return x.row.rank > 0\n\n    def g(x):\n      return x.row.rank > 1\n    '.replace('    ', '')
        path = os.path.join(self.tmpdir, 'udf.py')
        self.fs.create(path).write(data.encode('utf8'))
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            elements = p | beam.Create(self.data)
            result = elements | YamlTransform(f'\n        type: Filter\n        config:\n          language: python\n          keep:\n            path: {path}\n            name: "f"\n        ')
            assert_that(result | AsRows(), equal_to([beam.Row(label='37a', conductor=37, row=beam.Row(rank=1, values=[4, 5, 6])), beam.Row(label='389a', conductor=389, row=beam.Row(rank=2, values=[7, 8, 9]))]))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()