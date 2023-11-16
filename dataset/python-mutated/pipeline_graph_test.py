"""Tests for apache_beam.runners.interactive.display.pipeline_graph."""
import unittest
from unittest.mock import patch
import apache_beam as beam
from apache_beam.runners.interactive import interactive_beam as ib
from apache_beam.runners.interactive import interactive_environment as ie
from apache_beam.runners.interactive import interactive_runner as ir
from apache_beam.runners.interactive.display import pipeline_graph
from apache_beam.runners.interactive.testing.mock_ipython import mock_get_ipython

@unittest.skipIf(not ie.current_env().is_interactive_ready, '[interactive] dependency is not installed.')
class PipelineGraphTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        ie.new_env()

    def test_decoration(self):
        if False:
            return 10
        p = beam.Pipeline(ir.InteractiveRunner())
        pcoll = p | '"[1]": "Create\\"' >> beam.Create(range(1000))
        ib.watch(locals())
        self.assertEqual('digraph G {\nnode [color=blue, fontcolor=blue, shape=box];\n"\\"[1]\\": \\"Create\\\\\\"";\npcoll [shape=circle];\n"\\"[1]\\": \\"Create\\\\\\"" -> pcoll;\n}\n', pipeline_graph.PipelineGraph(p).get_dot())

    def test_get_dot(self):
        if False:
            while True:
                i = 10
        p = beam.Pipeline(ir.InteractiveRunner())
        init_pcoll = p | 'Init' >> beam.Create(range(10))
        squares = init_pcoll | 'Square' >> beam.Map(lambda x: x * x)
        cubes = init_pcoll | 'Cube' >> beam.Map(lambda x: x ** 3)
        ib.watch(locals())
        self.assertEqual('digraph G {\nnode [color=blue, fontcolor=blue, shape=box];\n"Init";\ninit_pcoll [shape=circle];\n"Square";\nsquares [shape=circle];\n"Cube";\ncubes [shape=circle];\n"Init" -> init_pcoll;\ninit_pcoll -> "Square";\ninit_pcoll -> "Cube";\n"Square" -> squares;\n"Cube" -> cubes;\n}\n', pipeline_graph.PipelineGraph(p).get_dot())

    @patch('IPython.get_ipython', new_callable=mock_get_ipython)
    def test_get_dot_within_notebook(self, cell):
        if False:
            i = 10
            return i + 15
        ie.current_env()._is_in_ipython = True
        ie.current_env()._is_in_notebook = True
        with cell:
            p = beam.Pipeline(ir.InteractiveRunner())
            ib.watch(locals())
        with cell:
            init_pcoll = p | 'Init' >> beam.Create(range(10))
        with cell:
            squares = init_pcoll | 'Square' >> beam.Map(lambda x: x * x)
        with cell:
            cubes = init_pcoll | 'Cube' >> beam.Map(lambda x: x ** 3)
        ib.watch(locals())
        self.assertEqual('digraph G {\nnode [color=blue, fontcolor=blue, shape=box];\n"[2]: Init";\ninit_pcoll [shape=circle];\n"[3]: Square";\nsquares [shape=circle];\n"[4]: Cube";\ncubes [shape=circle];\n"[2]: Init" -> init_pcoll;\ninit_pcoll -> "[3]: Square";\ninit_pcoll -> "[4]: Cube";\n"[3]: Square" -> squares;\n"[4]: Cube" -> cubes;\n}\n', pipeline_graph.PipelineGraph(p).get_dot())
if __name__ == '__main__':
    unittest.main()