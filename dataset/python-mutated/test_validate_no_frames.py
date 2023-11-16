from unittest import TestCase
import plotly.graph_objs as go
import pytest
try:
    go.FigureWidget()
    figure_widget_available = True
except ImportError:
    figure_widget_available = False

class TestNoFrames(TestCase):
    if figure_widget_available:

        def test_no_frames_in_constructor_kwarg(self):
            if False:
                print('Hello World!')
            with pytest.raises(ValueError):
                go.FigureWidget(frames=[{}])

        def test_emtpy_frames_ok_as_constructor_kwarg(self):
            if False:
                while True:
                    i = 10
            go.FigureWidget(frames=[])

        def test_no_frames_in_constructor_dict(self):
            if False:
                print('Hello World!')
            with pytest.raises(ValueError):
                go.FigureWidget({'frames': [{}]})

        def test_emtpy_frames_ok_as_constructor_dict_key(self):
            if False:
                while True:
                    i = 10
            go.FigureWidget({'frames': []})

        def test_no_frames_assignment(self):
            if False:
                while True:
                    i = 10
            fig = go.FigureWidget()
            with pytest.raises(ValueError):
                fig.frames = [{}]

        def test_emtpy_frames_assignment_ok(self):
            if False:
                while True:
                    i = 10
            fig = go.FigureWidget()
            fig.frames = []