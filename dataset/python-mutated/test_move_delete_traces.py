import sys
from unittest import TestCase
import pytest
import plotly.graph_objs as go
if sys.version_info >= (3, 3):
    from unittest.mock import MagicMock
else:
    from mock import MagicMock

class TestMoveDeleteTracesMessages(TestCase):

    def setUp(self):
        if False:
            return 10
        self.figure = go.Figure(data=[go.Scatter(y=[3, 2, 1], marker={'color': 'green'}), go.Bar(y=[3, 2, 1, 0, -1], marker={'opacity': 0.5}), go.Sankey(arrangement='snap')])
        self.figure._send_moveTraces_msg = MagicMock()
        self.figure._send_deleteTraces_msg = MagicMock()

    def test_move_traces_swap(self):
        if False:
            for i in range(10):
                print('nop')
        traces = self.figure.data
        self.figure.data = [traces[2], traces[1], traces[0]]
        self.figure._send_moveTraces_msg.assert_called_once_with([0, 1, 2], [2, 1, 0])
        self.assertFalse(self.figure._send_deleteTraces_msg.called)

    def test_move_traces_cycle(self):
        if False:
            for i in range(10):
                print('nop')
        traces = self.figure.data
        self.figure.data = [traces[2], traces[0], traces[1]]
        self.figure._send_moveTraces_msg.assert_called_once_with([0, 1, 2], [1, 2, 0])
        self.assertFalse(self.figure._send_deleteTraces_msg.called)

    def test_delete_single_traces(self):
        if False:
            print('Hello World!')
        traces = self.figure.data
        self.figure.data = [traces[0], traces[2]]
        self.figure._send_deleteTraces_msg.assert_called_once_with([1])
        self.assertFalse(self.figure._send_moveTraces_msg.called)

    def test_delete_multiple_traces(self):
        if False:
            while True:
                i = 10
        traces = self.figure.data
        self.figure.data = [traces[1]]
        self.figure._send_deleteTraces_msg.assert_called_once_with([0, 2])
        self.assertFalse(self.figure._send_moveTraces_msg.called)

    def test_delete_all_traces(self):
        if False:
            i = 10
            return i + 15
        self.figure.data = []
        self.figure._send_deleteTraces_msg.assert_called_once_with([0, 1, 2])
        self.assertFalse(self.figure._send_moveTraces_msg.called)

    def test_move_and_delete_traces(self):
        if False:
            i = 10
            return i + 15
        traces = self.figure.data
        self.figure.data = [traces[2], traces[0]]
        self.figure._send_deleteTraces_msg.assert_called_once_with([1])
        self.figure._send_moveTraces_msg.assert_called_once_with([0, 1], [1, 0])

    def test_validate_assigned_traces_are_subset(self):
        if False:
            i = 10
            return i + 15
        traces = self.figure.data
        with pytest.raises(ValueError):
            self.figure.data = [traces[2], go.Scatter(y=[3, 2, 1]), traces[1]]

    def test_validate_assigned_traces_are_not_duplicates(self):
        if False:
            while True:
                i = 10
        traces = self.figure.data
        with pytest.raises(ValueError):
            self.figure.data = [traces[2], traces[1], traces[1]]