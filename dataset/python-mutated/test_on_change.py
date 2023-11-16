import sys
from unittest import TestCase
import pytest
import plotly.graph_objs as go
if sys.version_info >= (3, 3):
    from unittest.mock import MagicMock
else:
    from mock import MagicMock

class TestOnChangeCallbacks(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.figure = go.Figure(data=[go.Scatter(y=[3, 2, 1], marker={'color': 'green'}), go.Bar(y=[3, 2, 1, 0, -1], marker={'opacity': 0.5})], layout={'xaxis': {'range': [-1, 4]}, 'width': 1000}, frames=[go.Frame(layout={'yaxis': {'title': 'f1'}})])

    def test_raise_if_no_figure(self):
        if False:
            for i in range(10):
                print('nop')
        scatt = go.Scatter()
        fn = MagicMock()
        with pytest.raises(ValueError):
            scatt.on_change(fn, 'x')

    def test_raise_on_frame_hierarchy(self):
        if False:
            for i in range(10):
                print('nop')
        fn = MagicMock()
        with pytest.raises(ValueError):
            self.figure.frames[0].layout.xaxis.on_change(fn, 'range')

    def test_validate_property_path_nested(self):
        if False:
            while True:
                i = 10
        fn = MagicMock()
        with pytest.raises(ValueError):
            self.figure.layout.xaxis.on_change(fn, 'bogus')

    def test_validate_property_path_nested(self):
        if False:
            return 10
        fn = MagicMock()
        with pytest.raises(ValueError):
            self.figure.layout.on_change(fn, 'xaxis.titlefont.bogus')

    def test_single_prop_callback_on_assignment(self):
        if False:
            print('Hello World!')
        fn_x = MagicMock()
        fn_y = MagicMock()
        self.figure.data[0].on_change(fn_x, 'x')
        self.figure.data[0].on_change(fn_y, 'y')
        self.figure.data[1].x = [1, 2, 3]
        self.figure.data[1].y = [1, 2, 3]
        self.assertFalse(fn_x.called)
        self.assertFalse(fn_y.called)
        self.figure.data[0].x = [10, 20, 30]
        fn_x.assert_called_once_with(self.figure.data[0], (10, 20, 30))
        self.assertFalse(fn_y.called)
        self.figure.data[0].y = [11, 22, 33]
        fn_y.assert_called_once_with(self.figure.data[0], (11, 22, 33))

    def test_multi_prop_callback_on_assignment_trace(self):
        if False:
            i = 10
            return i + 15
        fn = MagicMock()
        self.figure.data[0].on_change(fn, 'x', 'y')
        self.figure.data[0].x = [11, 22, 33]
        fn.assert_called_once_with(self.figure.data[0], (11, 22, 33), (3, 2, 1))

    def test_multi_prop_callback_on_assignment_layout(self):
        if False:
            while True:
                i = 10
        fn_range = MagicMock()
        self.figure.layout.on_change(fn_range, ('xaxis', 'range'), 'yaxis.range')
        self.figure.layout.xaxis.range = [-10, 10]
        fn_range.assert_called_once_with(self.figure.layout, (-10, 10), None)

    def test_multi_prop_callback_on_assignment_layout_nested(self):
        if False:
            return 10
        fn_titlefont = MagicMock()
        fn_xaxis = MagicMock()
        fn_layout = MagicMock()
        self.figure.layout.xaxis.titlefont.on_change(fn_titlefont, 'family')
        self.figure.layout.xaxis.on_change(fn_xaxis, 'range', 'title.font.family')
        self.figure.layout.on_change(fn_layout, 'xaxis')
        self.figure.layout.xaxis.title.font.family = 'courier'
        fn_titlefont.assert_called_once_with(self.figure.layout.xaxis.title.font, 'courier')
        fn_xaxis.assert_called_once_with(self.figure.layout.xaxis, (-1, 4), 'courier')
        fn_layout.assert_called_once_with(self.figure.layout, go.layout.XAxis(range=(-1, 4), title={'font': {'family': 'courier'}}))

    def test_prop_callback_nested_arrays(self):
        if False:
            i = 10
            return i + 15
        self.figure.layout.updatemenus = [{}, {}, {}]
        self.figure.layout.updatemenus[2].buttons = [{}, {}]
        self.figure.layout.updatemenus[2].buttons[1].label = 'button 1'
        self.figure.layout.updatemenus[2].buttons[1].method = 'relayout'
        fn_button = MagicMock()
        fn_layout = MagicMock()
        self.figure.layout.updatemenus[2].buttons[1].on_change(fn_button, 'method')
        self.figure.layout.on_change(fn_layout, 'updatemenus[2].buttons[1].method')
        self.figure.layout.updatemenus[2].buttons[1].method = 'restyle'
        fn_button.assert_called_once_with(self.figure.layout.updatemenus[2].buttons[1], 'restyle')
        fn_layout.assert_called_once_with(self.figure.layout, 'restyle')

    def test_callback_on_update(self):
        if False:
            for i in range(10):
                print('nop')
        fn_range = MagicMock()
        self.figure.layout.on_change(fn_range, 'xaxis.range', 'yaxis.range')
        self.figure.update({'layout': {'yaxis': {'range': [11, 22]}}})
        fn_range.assert_called_once_with(self.figure.layout, (-1, 4), (11, 22))

    def test_callback_on_update_single_call(self):
        if False:
            print('Hello World!')
        fn_range = MagicMock()
        self.figure.layout.on_change(fn_range, 'xaxis.range', 'yaxis.range', 'width')
        self.figure.update({'layout': {'xaxis': {'range': [-10, 10]}, 'yaxis': {'range': [11, 22]}}})
        fn_range.assert_called_once_with(self.figure.layout, (-10, 10), (11, 22), 1000)

    def test_callback_on_batch_update(self):
        if False:
            print('Hello World!')
        fn_range = MagicMock()
        self.figure.layout.on_change(fn_range, 'xaxis.range', 'yaxis.range', 'width')
        with self.figure.batch_update():
            self.figure.layout.xaxis.range = [-10, 10]
            self.figure.layout.width = 500
            self.assertFalse(fn_range.called)
        fn_range.assert_called_once_with(self.figure.layout, (-10, 10), None, 500)

    def test_callback_on_batch_animate(self):
        if False:
            print('Hello World!')
        fn_range = MagicMock()
        self.figure.layout.on_change(fn_range, 'xaxis.range', 'yaxis.range', 'width')
        with self.figure.batch_animate():
            self.figure['layout.xaxis.range'] = [-10, 10]
            self.figure['layout', 'yaxis', 'range'] = (11, 22)
            self.assertFalse(fn_range.called)
        fn_range.assert_called_once_with(self.figure.layout, (-10, 10), (11, 22), 1000)

    def test_callback_on_plotly_relayout(self):
        if False:
            for i in range(10):
                print('nop')
        fn_range = MagicMock()
        self.figure.layout.on_change(fn_range, 'xaxis.range', 'yaxis.range', 'width')
        self.figure.plotly_relayout(relayout_data={'xaxis.range': [-10, 10], 'yaxis.range': [11, 22]})
        fn_range.assert_called_once_with(self.figure.layout, (-10, 10), (11, 22), 1000)

    def test_callback_on_plotly_restyle(self):
        if False:
            while True:
                i = 10
        fn = MagicMock()
        self.figure.data[0].on_change(fn, 'x', 'y')
        self.figure.plotly_restyle({'x': [[11, 22, 33], [1, 11, 111]]}, trace_indexes=[0, 1])
        fn.assert_called_once_with(self.figure.data[0], (11, 22, 33), (3, 2, 1))

    def test_callback_on_plotly_update(self):
        if False:
            while True:
                i = 10
        fn_range = MagicMock()
        self.figure.layout.on_change(fn_range, 'xaxis.range', 'yaxis.range', 'width')
        self.figure.plotly_update(restyle_data={'marker.color': 'blue'}, relayout_data={'xaxis.range': [-10, 10], 'yaxis.range': [11, 22]})
        fn_range.assert_called_once_with(self.figure.layout, (-10, 10), (11, 22), 1000)