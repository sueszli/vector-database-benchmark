import pytest
import copy
import pickle
from plotly.tools import make_subplots
import plotly.graph_objs as go
import plotly.io as pio

@pytest.fixture
def fig1(request):
    if False:
        print('Hello World!')
    return go.Figure(data=[{'type': 'scattergl', 'marker': {'color': 'green'}}, {'type': 'parcoords', 'dimensions': [{'values': [1, 2, 3]}, {'values': [3, 2, 1]}], 'line': {'color': 'blue'}}], layout={'title': 'Figure title'})

@pytest.fixture
def fig_subplots(request):
    if False:
        for i in range(10):
            print('nop')
    fig = make_subplots(3, 2)
    fig.add_scatter(y=[2, 1, 3], row=1, col=1)
    fig.add_scatter(y=[1, 3, 3], row=2, col=2)
    return fig

def test_deepcopy_figure(fig1):
    if False:
        return 10
    fig_copied = copy.deepcopy(fig1)
    assert fig_copied.to_dict() == fig1.to_dict()
    assert fig_copied is not fig1
    assert fig_copied.layout is not fig1.layout
    assert fig_copied.data is not fig1.data

def test_deepcopy_figure_subplots(fig_subplots):
    if False:
        return 10
    fig_copied = copy.deepcopy(fig_subplots)
    assert fig_copied.to_dict() == fig_subplots.to_dict()
    assert fig_subplots._grid_ref == fig_copied._grid_ref
    assert fig_subplots._grid_str == fig_copied._grid_str
    assert fig_copied is not fig_subplots
    assert fig_copied.layout is not fig_subplots.layout
    assert fig_copied.data is not fig_subplots.data
    fig_subplots.add_bar(y=[0, 0, 1], row=1, col=2)
    fig_copied.add_bar(y=[0, 0, 1], row=1, col=2)
    assert fig_copied.to_dict() == fig_subplots.to_dict()

def test_deepcopy_layout(fig1):
    if False:
        while True:
            i = 10
    copied_layout = copy.deepcopy(fig1.layout)
    assert copied_layout == fig1.layout
    assert copied_layout is not fig1.layout
    assert fig1.layout.parent is fig1
    assert copied_layout.parent is None

def test_pickle_figure_round_trip(fig1):
    if False:
        print('Hello World!')
    fig_copied = pickle.loads(pickle.dumps(fig1))
    assert fig_copied.to_dict() == fig1.to_dict()

def test_pickle_figure_subplots_round_trip(fig_subplots):
    if False:
        print('Hello World!')
    fig_copied = pickle.loads(pickle.dumps(fig_subplots))
    assert fig_copied.to_dict() == fig_subplots.to_dict()
    fig_subplots.add_bar(y=[0, 0, 1], row=1, col=2)
    fig_copied.add_bar(y=[0, 0, 1], row=1, col=2)
    assert fig_copied.to_dict() == fig_subplots.to_dict()

def test_pickle_layout(fig1):
    if False:
        while True:
            i = 10
    copied_layout = pickle.loads(pickle.dumps(fig1.layout))
    assert copied_layout == fig1.layout
    assert copied_layout is not fig1.layout
    assert fig1.layout.parent is fig1
    assert copied_layout.parent is None