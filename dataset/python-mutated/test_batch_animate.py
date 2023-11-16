import sys
from unittest import TestCase
import plotly.graph_objs as go
if sys.version_info >= (3, 3):
    from unittest.mock import MagicMock
else:
    from mock import MagicMock

class TestBatchAnimateMessage(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.figure = go.Figure(data=[go.Scatter(y=[3, 2, 1], marker={'color': 'green'}), go.Bar(y=[3, 2, 1, 0, -1], marker={'opacity': 0.5})], layout={'xaxis': {'range': [-1, 4]}}, frames=[go.Frame(layout={'yaxis': {'title': 'f1'}})])
        self.figure._send_animate_msg = MagicMock()

    def test_batch_animate(self):
        if False:
            print('Hello World!')
        with self.figure.batch_animate(easing='elastic', duration=1200):
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
        self.figure._send_animate_msg.assert_called_once_with(styles_data=[{'marker.color': 'yellow'}, {'marker.opacity': 0.9}], relayout_data={'xaxis.range': [10, 20]}, trace_indexes=[0, 1], animation_opts={'transition': {'easing': 'elastic', 'duration': 1200}, 'frame': {'duration': 1200}})