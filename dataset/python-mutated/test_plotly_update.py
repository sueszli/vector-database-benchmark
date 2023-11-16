import sys
from unittest import TestCase
import plotly.graph_objs as go
from plotly.basedatatypes import Undefined
if sys.version_info >= (3, 3):
    from unittest.mock import MagicMock
else:
    from mock import MagicMock

class TestBatchUpdateMessage(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.figure = go.Figure(data=[go.Scatter(y=[3, 2, 1], marker={'color': 'green'}), go.Bar(y=[3, 2, 1, 0, -1], marker={'opacity': 0.5})], layout={'xaxis': {'range': [-1, 4]}}, frames=[go.Frame(layout={'yaxis': {'title': 'f1'}})])
        self.figure._send_update_msg = MagicMock()

    def test_batch_update(self):
        if False:
            i = 10
            return i + 15
        with self.figure.batch_update():
            self.figure.data[0].marker.color = 'yellow'
            self.figure.data[1].marker.opacity = 0.9
            self.figure.layout.xaxis.range = [10, 20]
            self.figure.frames[0].layout.yaxis.title.text = 'f2'
            self.assertEqual(self.figure.data[0].marker.color, 'green')
            self.assertEqual(self.figure.data[1].marker.opacity, 0.5)
            self.assertEqual(self.figure.layout.xaxis.range, (-1, 4))
            self.assertEqual(self.figure.frames[0].layout.yaxis.title.text, 'f2')
        self.assertEqual(self.figure.data[0].marker.color, 'yellow')
        self.assertEqual(self.figure.data[1].marker.opacity, 0.9)
        self.assertEqual(self.figure.layout.xaxis.range, (10, 20))
        self.figure._send_update_msg.assert_called_once_with(restyle_data={'marker.color': ['yellow', Undefined], 'marker.opacity': [Undefined, 0.9]}, relayout_data={'xaxis.range': [10, 20]}, trace_indexes=[0, 1])

    def test_plotly_update(self):
        if False:
            print('Hello World!')
        self.figure.plotly_update(restyle_data={'marker.color': ['yellow', Undefined], 'marker.opacity': [Undefined, 0.9]}, relayout_data={'xaxis.range': [10, 20]}, trace_indexes=[0, 1])
        self.assertEqual(self.figure.data[0].marker.color, 'yellow')
        self.assertEqual(self.figure.data[1].marker.opacity, 0.9)
        self.assertEqual(self.figure.layout.xaxis.range, (10, 20))
        self.figure._send_update_msg.assert_called_once_with(restyle_data={'marker.color': ['yellow', Undefined], 'marker.opacity': [Undefined, 0.9]}, relayout_data={'xaxis.range': [10, 20]}, trace_indexes=[0, 1])