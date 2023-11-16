import sys
from unittest import TestCase
import plotly.graph_objs as go
if sys.version_info >= (3, 3):
    from unittest.mock import MagicMock
else:
    from mock import MagicMock

class TestRestyleMessage(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.figure = go.Figure(data=[go.Scatter(), go.Bar(), go.Parcoords(dimensions=[{}, {'label': 'dim 2'}, {}])])
        self.figure._send_restyle_msg = MagicMock()

    def test_property_assignment_toplevel(self):
        if False:
            while True:
                i = 10
        self.figure.data[1].marker = {'color': 'green'}
        self.figure._send_restyle_msg.assert_called_once_with({'marker': [{'color': 'green'}]}, trace_indexes=1)

    def test_property_assignment_nested(self):
        if False:
            for i in range(10):
                print('nop')
        self.figure.data[0].marker.color = 'green'
        self.figure._send_restyle_msg.assert_called_once_with({'marker.color': ['green']}, trace_indexes=0)

    def test_property_assignment_nested_array(self):
        if False:
            return 10
        self.figure.data[2].dimensions[0].label = 'dim 1'
        self.figure._send_restyle_msg.assert_called_once_with({'dimensions.0.label': ['dim 1']}, trace_indexes=2)

    def test_plotly_restyle_toplevel(self):
        if False:
            while True:
                i = 10
        self.figure.plotly_restyle({'marker': {'color': 'green'}}, trace_indexes=1)
        self.figure._send_restyle_msg.assert_called_once_with({'marker': {'color': 'green'}}, trace_indexes=[1])

    def test_plotly_restyle_nested(self):
        if False:
            while True:
                i = 10
        self.figure.plotly_restyle({'marker.color': 'green'}, trace_indexes=0)
        self.figure._send_restyle_msg.assert_called_once_with({'marker.color': 'green'}, trace_indexes=[0])

    def test_plotly_restyle_nested_array(self):
        if False:
            i = 10
            return i + 15
        self.figure.plotly_restyle({'dimensions[0].label': 'dim 1'}, trace_indexes=2)
        self.figure._send_restyle_msg.assert_called_once_with({'dimensions[0].label': 'dim 1'}, trace_indexes=[2])

    def test_plotly_restyle_multi_prop(self):
        if False:
            for i in range(10):
                print('nop')
        self.figure.plotly_restyle({'marker': {'color': 'green'}, 'name': 'MARKER 1'}, trace_indexes=1)
        self.figure._send_restyle_msg.assert_called_once_with({'marker': {'color': 'green'}, 'name': 'MARKER 1'}, trace_indexes=[1])

    def test_plotly_restyle_multi_trace(self):
        if False:
            print('Hello World!')
        self.figure.plotly_restyle({'marker': {'color': 'green'}, 'name': 'MARKER 1'}, trace_indexes=[0, 1])
        self.figure._send_restyle_msg.assert_called_once_with({'marker': {'color': 'green'}, 'name': 'MARKER 1'}, trace_indexes=[0, 1])