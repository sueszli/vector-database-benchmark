"""Graphviz unit test."""
import graphviz as graphviz
import streamlit as st
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class GraphvizTest(DeltaGeneratorTestCase):
    """Test ability to marshall graphviz_chart protos."""

    def test_spec(self):
        if False:
            return 10
        'Test that it can be called with spec.'
        graph = graphviz.Graph(comment='The Round Table')
        graph.node('A', 'King Arthur')
        graph.node('B', 'Sir Bedevere the Wise')
        graph.edges(['AB'])
        st.graphviz_chart(graph)
        c = self.get_delta_from_queue().new_element.graphviz_chart
        self.assertEqual(hasattr(c, 'spec'), True)

    def test_dot(self):
        if False:
            while True:
                i = 10
        'Test that it can be called with dot string.'
        graph = graphviz.Graph(comment='The Round Table')
        graph.node('A', 'King Arthur')
        graph.node('B', 'Sir Bedevere the Wise')
        graph.edges(['AB'])
        st.graphviz_chart(graph)
        c = self.get_delta_from_queue().new_element.graphviz_chart
        self.assertEqual(hasattr(c, 'spec'), True)

    def test_use_container_width_true(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that it can be called with use_container_width.'
        graph = graphviz.Graph(comment='The Round Table')
        graph.node('A', 'King Arthur')
        graph.node('B', 'Sir Bedevere the Wise')
        graph.edges(['AB'])
        st.graphviz_chart(graph, use_container_width=True)
        c = self.get_delta_from_queue().new_element.graphviz_chart
        self.assertEqual(c.use_container_width, True)

    def test_engines(self):
        if False:
            i = 10
            return i + 15
        'Test that it can be called with engines.'
        engines = ['dot', 'neato', 'twopi', 'circo', 'fdp', 'osage', 'patchwork']
        for engine in engines:
            graph = graphviz.Graph(comment='The Round Table', engine=engine)
            graph.node('A', 'King Arthur')
            graph.node('B', 'Sir Bedevere the gWise')
            graph.edges(['AB'])
            st.graphviz_chart(graph)
            c = self.get_delta_from_queue().new_element.graphviz_chart
            self.assertEqual(hasattr(c, 'engine'), True)
            self.assertEqual(c.engine, engine)