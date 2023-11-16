import sys
from unittest import TestCase
import plotly.graph_objs as go
if sys.version_info >= (3, 3):
    from unittest.mock import MagicMock
else:
    from mock import MagicMock

class TestRelayoutMessage(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.figure = go.Figure(layout={'xaxis': {'range': [-1, 4]}})
        self.figure._send_relayout_msg = MagicMock()

    def test_property_assignment_toplevel(self):
        if False:
            for i in range(10):
                print('nop')
        self.figure.layout.title.text = 'hello'
        self.figure._send_relayout_msg.assert_called_once_with({'title.text': 'hello'})

    def test_property_assignment_nested(self):
        if False:
            while True:
                i = 10
        self.figure.layout.xaxis.title.font.family = 'courier'
        self.figure._send_relayout_msg.assert_called_once_with({'xaxis.title.font.family': 'courier'})

    def test_property_assignment_nested_subplot2(self):
        if False:
            while True:
                i = 10
        self.figure.layout.xaxis2 = {'range': [0, 1]}
        self.figure._send_relayout_msg.assert_called_once_with({'xaxis2': {'range': [0, 1]}})
        self.figure._send_relayout_msg = MagicMock()
        self.figure.layout.xaxis2.title.font.family = 'courier'
        self.figure._send_relayout_msg.assert_called_once_with({'xaxis2.title.font.family': 'courier'})

    def test_property_assignment_nested_array(self):
        if False:
            while True:
                i = 10
        self.figure.layout.updatemenus = [{}, go.layout.Updatemenu(buttons=[{}, {}, go.layout.updatemenu.Button(method='relayout')]), {}]
        self.figure._send_relayout_msg.assert_called_once_with({'updatemenus': [{}, {'buttons': [{}, {}, {'method': 'relayout'}]}, {}]})
        self.figure._send_relayout_msg = MagicMock()
        self.figure.layout.updatemenus[1].buttons[0].method = 'restyle'
        self.figure._send_relayout_msg.assert_called_once_with({'updatemenus.1.buttons.0.method': 'restyle'})

    def test_property_assignment_template(self):
        if False:
            while True:
                i = 10
        self.figure.layout.template = {'layout': {'xaxis': {'title': {'text': 'x-label'}}}}
        self.figure._send_relayout_msg.assert_called_with({'template': {'layout': {'xaxis': {'title': {'text': 'x-label'}}}}})
        self.figure.layout.template.layout.title.text = 'Template Title'
        self.figure._send_relayout_msg.assert_called_with({'template.layout.title.text': 'Template Title'})
        self.figure.layout.template.data = {'bar': [{'marker': {'color': 'blue'}}, {'marker': {'color': 'yellow'}}]}
        self.figure._send_relayout_msg.assert_called_with({'template.data': {'bar': [{'type': 'bar', 'marker': {'color': 'blue'}}, {'type': 'bar', 'marker': {'color': 'yellow'}}]}})
        self.figure.layout.template.data.bar[1].marker.opacity = 0.5
        self.figure._send_relayout_msg.assert_called_with({'template.data.bar.1.marker.opacity': 0.5})
        self.figure.layout.template.layout.imagedefaults.sizex = 300
        self.figure._send_relayout_msg.assert_called_with({'template.layout.imagedefaults.sizex': 300})

    def test_plotly_relayout_toplevel(self):
        if False:
            for i in range(10):
                print('nop')
        self.figure.plotly_relayout({'title': 'hello'})
        self.figure._send_relayout_msg.assert_called_once_with({'title': 'hello'})

    def test_plotly_relayout_nested(self):
        if False:
            return 10
        self.figure.plotly_relayout({'xaxis.title.font.family': 'courier'})
        self.figure._send_relayout_msg.assert_called_once_with({'xaxis.title.font.family': 'courier'})

    def test_plotly_relayout_nested_subplot2(self):
        if False:
            return 10
        self.figure.layout.xaxis2 = {'range': [0, 1]}
        self.figure._send_relayout_msg.assert_called_once_with({'xaxis2': {'range': [0, 1]}})
        self.figure._send_relayout_msg = MagicMock()
        self.figure.plotly_relayout({'xaxis2.title.font.family': 'courier'})
        self.figure._send_relayout_msg.assert_called_once_with({'xaxis2.title.font.family': 'courier'})

    def test_plotly_relayout_nested_array(self):
        if False:
            return 10
        self.figure.layout.updatemenus = [{}, go.layout.Updatemenu(buttons=[{}, {}, go.layout.updatemenu.Button(method='relayout')]), {}]
        self.figure._send_relayout_msg.assert_called_once_with({'updatemenus': [{}, {'buttons': [{}, {}, {'method': 'relayout'}]}, {}]})
        self.figure._send_relayout_msg = MagicMock()
        self.figure.plotly_relayout({'updatemenus[1].buttons.0.method': 'restyle'})
        self.figure._send_relayout_msg.assert_called_once_with({'updatemenus[1].buttons.0.method': 'restyle'})