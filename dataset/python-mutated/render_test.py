import os
import argparse
import logging
import subprocess
import unittest
import tempfile
import pytest
import apache_beam as beam
from apache_beam.runners import render
default_options = render.RenderOptions._add_argparse_args(argparse.ArgumentParser()).parse_args([])

class RenderRunnerTest(unittest.TestCase):

    def test_basic_graph(self):
        if False:
            print('Hello World!')
        p = beam.Pipeline()
        _ = p | beam.Impulse() | beam.Map(lambda _: 2) | 'CustomName' >> beam.Map(lambda x: x * x)
        dot = render.PipelineRenderer(p.to_runner_api(), default_options).to_dot()
        self.assertIn('digraph', dot)
        self.assertIn('CustomName', dot)
        self.assertEqual(dot.count('->'), 2)

    def test_render_config_validation(self):
        if False:
            return 10
        p = beam.Pipeline()
        _ = p | beam.Impulse() | beam.Map(lambda _: 2) | 'CustomName' >> beam.Map(lambda x: x * x)
        pipeline_proto = p.to_runner_api()
        with pytest.raises(ValueError):
            render.RenderRunner().run_portable_pipeline(pipeline_proto, render.RenderOptions())

    def test_side_input(self):
        if False:
            print('Hello World!')
        p = beam.Pipeline()
        pcoll = p | beam.Impulse() | beam.FlatMap(lambda x: [1, 2, 3])
        dot = render.PipelineRenderer(p.to_runner_api(), default_options).to_dot()
        self.assertEqual(dot.count('->'), 1)
        self.assertNotIn('dashed', dot)
        _ = pcoll | beam.Map(lambda x, side: x * side, side=beam.pvalue.AsList(pcoll))
        dot = render.PipelineRenderer(p.to_runner_api(), default_options).to_dot()
        self.assertEqual(dot.count('->'), 3)
        self.assertIn('dashed', dot)

    def test_composite_collapse(self):
        if False:
            print('Hello World!')
        p = beam.Pipeline()
        _ = p | beam.Create([1, 2, 3]) | beam.Map(lambda x: x * x)
        pipeline_proto = p.to_runner_api()
        renderer = render.PipelineRenderer(pipeline_proto, default_options)
        self.assertEqual(renderer.to_dot().count('->'), 8)
        (create_transform_id,) = [id for (id, transform) in pipeline_proto.components.transforms.items() if transform.unique_name == 'Create']
        renderer.update(toggle=[create_transform_id])
        self.assertEqual(renderer.to_dot().count('->'), 1)

class DotRequiringRenderingTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        try:
            subprocess.run(['dot', '-V'], capture_output=True, check=True)
        except FileNotFoundError:
            cls._dot_installed = False
        else:
            cls._dot_installed = True

    def setUp(self) -> None:
        if False:
            return 10
        if not self._dot_installed:
            self.skipTest('dot executable not installed')

    def test_run_portable_pipeline(self):
        if False:
            for i in range(10):
                print('nop')
        p = beam.Pipeline()
        _ = p | beam.Impulse() | beam.Map(lambda _: 2) | 'CustomName' >> beam.Map(lambda x: x * x)
        pipeline_proto = p.to_runner_api()
        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = os.path.join(tmpdir, 'my_output.svg')
            render.RenderRunner().run_portable_pipeline(pipeline_proto, render.RenderOptions(render_output=[svg_path]))
            assert os.path.exists(svg_path)

    def test_dot_well_formed(self):
        if False:
            return 10
        p = beam.Pipeline()
        _ = p | beam.Create([1, 2, 3]) | beam.Map(lambda x: x * x)
        pipeline_proto = p.to_runner_api()
        renderer = render.PipelineRenderer(pipeline_proto, default_options)
        renderer.render_data()
        (create_transform_id,) = [id for (id, transform) in pipeline_proto.components.transforms.items() if transform.unique_name == 'Create']
        renderer.update(toggle=[create_transform_id])
        renderer.render_data()

    def test_leaf_composite_filter(self):
        if False:
            while True:
                i = 10
        p = beam.Pipeline()
        _ = p | beam.Create([1, 2, 3]) | beam.Map(lambda x: x * x)
        dot = render.PipelineRenderer(p.to_runner_api(), render.RenderOptions(['--render_leaf_composite_nodes=Create'])).to_dot()
        self.assertEqual(dot.count('->'), 1)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()