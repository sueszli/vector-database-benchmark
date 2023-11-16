import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.basedatatypes import _indexing_combinations
import pytest
from itertools import product
NROWS = 4
NCOLS = 5

@pytest.fixture
def subplot_fig_fixture():
    if False:
        for i in range(10):
            print('nop')
    fig = make_subplots(NROWS, NCOLS)
    return fig

@pytest.fixture
def non_subplot_fig_fixture():
    if False:
        while True:
            i = 10
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[4, 3, 2]))
    return fig

def test_invalid_validate_get_grid_ref(non_subplot_fig_fixture):
    if False:
        print('Hello World!')
    with pytest.raises(Exception):
        _ = non_subplot_fig_fixture._validate_get_grid_ref()

def test_get_subplot_coordinates(subplot_fig_fixture):
    if False:
        for i in range(10):
            print('nop')
    assert set(subplot_fig_fixture._get_subplot_coordinates()) == set([(r, c) for r in range(1, NROWS + 1) for c in range(1, NCOLS + 1)])

def test_indexing_combinations_edge_cases():
    if False:
        return 10
    assert _indexing_combinations([], []) == []
    with pytest.raises(ValueError):
        _ = _indexing_combinations([[1, 2], [3, 4, 5]], [[1, 2]])
all_rows = [1, 2, 3, 4]
all_cols = [1, 2, 3, 4, 5]

@pytest.mark.parametrize('test_input,expected', [(dict(dims=['all', 'all'], alls=[all_rows, all_cols], product=False), set(zip(all_rows, all_cols))), (dict(dims=['all', 'all'], alls=[all_rows, all_cols], product=True), set([(r, c) for r in all_rows for c in all_cols])), (dict(dims=['all', [2, 4, 5]], alls=[all_rows, all_cols], product=False), set(zip(all_rows, [2, 4, 5]))), (dict(dims=['all', [2, 4, 5]], alls=[all_rows, all_cols], product=True), set([(r, c) for r in all_rows for c in [2, 4, 5]])), (dict(dims=['all', 3], alls=[all_rows, all_cols], product=False), set([(all_rows[0], 3)])), (dict(dims=['all', 3], alls=[all_rows, all_cols], product=True), set([(r, c) for r in all_rows for c in [3]])), (dict(dims=[[1, 3], 'all'], alls=[all_rows, all_cols], product=False), set(zip([1, 3], all_cols))), (dict(dims=[[1, 3], 'all'], alls=[all_rows, all_cols], product=True), set([(r, c) for r in [1, 3] for c in all_cols])), (dict(dims=[[1, 3], [2, 4, 5]], alls=[all_rows, all_cols], product=False), set(zip([1, 3], [2, 4, 5]))), (dict(dims=[[1, 3], [2, 4, 5]], alls=[all_rows, all_cols], product=True), set([(r, c) for r in [1, 3] for c in [2, 4, 5]])), (dict(dims=[[1, 3], 3], alls=[all_rows, all_cols], product=False), set([(1, 3)])), (dict(dims=[[1, 3], 3], alls=[all_rows, all_cols], product=True), set([(r, c) for r in [1, 3] for c in [3]])), (dict(dims=[2, 'all'], alls=[all_rows, all_cols], product=False), set([(2, all_cols[0])])), (dict(dims=[2, 'all'], alls=[all_rows, all_cols], product=True), set([(r, c) for r in [2] for c in all_cols])), (dict(dims=[2, [2, 4, 5]], alls=[all_rows, all_cols], product=False), set([(2, 2)])), (dict(dims=[2, [2, 4, 5]], alls=[all_rows, all_cols], product=True), set([(r, c) for r in [2] for c in [2, 4, 5]])), (dict(dims=[2, 3], alls=[all_rows, all_cols], product=False), set([(2, 3)])), (dict(dims=[2, 3], alls=[all_rows, all_cols], product=True), set([(2, 3)]))])
def test_indexing_combinations(test_input, expected):
    if False:
        for i in range(10):
            print('nop')
    assert set(_indexing_combinations(**test_input)) == expected

def _sort_row_col_lists(rows, cols):
    if False:
        while True:
            i = 10
    si = sorted(range(len(rows)), key=lambda i: rows[i])
    rows = [rows[i] for i in si]
    cols = [cols[i] for i in si]
    return (rows, cols)

@pytest.mark.parametrize('test_input,expected', [(('all', [2, 4, 5], False), zip(*product(range(1, NROWS + 1), [2, 4, 5]))), (([1, 3], 'all', False), zip(*product([1, 3], range(1, NCOLS + 1)))), (([1, 3], 'all', True), zip(*product([1, 3], range(1, NCOLS + 1)))), (([1, 3], [2, 4, 5], False), [(1, 3), (2, 4)]), (([1, 3], [2, 4, 5], True), zip(*product([1, 3], [2, 4, 5])))])
def test_select_subplot_coordinates(subplot_fig_fixture, test_input, expected):
    if False:
        print('Hello World!')
    (rows, cols, product) = test_input
    (er, ec) = _sort_row_col_lists(*expected)
    t = subplot_fig_fixture._select_subplot_coordinates(rows, cols, product=product)
    (r, c) = zip(*t)
    (r, c) = _sort_row_col_lists(r, c)
    assert r == er and c == ec