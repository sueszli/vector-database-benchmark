import re
import sys
import numpy as np
import pytest
from matplotlib import _preprocess_data
from matplotlib.axes import Axes
from matplotlib.testing import subprocess_run_for_testing
from matplotlib.testing.decorators import check_figures_equal

@_preprocess_data(replace_names=['x', 'y'], label_namer='y')
def plot_func(ax, x, y, ls='x', label=None, w='xyz'):
    if False:
        i = 10
        return i + 15
    return f'x: {list(x)}, y: {list(y)}, ls: {ls}, w: {w}, label: {label}'
all_funcs = [plot_func]
all_func_ids = ['plot_func']

def test_compiletime_checks():
    if False:
        return 10
    'Test decorator invocations -> no replacements.'

    def func(ax, x, y):
        if False:
            return 10
        pass

    def func_args(ax, x, y, *args):
        if False:
            return 10
        pass

    def func_kwargs(ax, x, y, **kwargs):
        if False:
            i = 10
            return i + 15
        pass

    def func_no_ax_args(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pass
    _preprocess_data(replace_names=['x', 'y'])(func)
    _preprocess_data(replace_names=['x', 'y'])(func_kwargs)
    _preprocess_data(replace_names=['x', 'y'])(func_args)
    with pytest.raises(AssertionError):
        _preprocess_data(replace_names=['x', 'y', 'z'])(func_args)
    _preprocess_data(replace_names=[], label_namer=None)(func)
    _preprocess_data(replace_names=[], label_namer=None)(func_args)
    _preprocess_data(replace_names=[], label_namer=None)(func_kwargs)
    _preprocess_data(replace_names=[], label_namer=None)(func_no_ax_args)
    with pytest.raises(AssertionError):
        _preprocess_data(label_namer='z')(func)
    with pytest.raises(AssertionError):
        _preprocess_data(label_namer='z')(func_args)

@pytest.mark.parametrize('func', all_funcs, ids=all_func_ids)
def test_function_call_without_data(func):
    if False:
        print('Hello World!')
    'Test without data -> no replacements.'
    assert func(None, 'x', 'y') == "x: ['x'], y: ['y'], ls: x, w: xyz, label: None"
    assert func(None, x='x', y='y') == "x: ['x'], y: ['y'], ls: x, w: xyz, label: None"
    assert func(None, 'x', 'y', label='') == "x: ['x'], y: ['y'], ls: x, w: xyz, label: "
    assert func(None, 'x', 'y', label='text') == "x: ['x'], y: ['y'], ls: x, w: xyz, label: text"
    assert func(None, x='x', y='y', label='') == "x: ['x'], y: ['y'], ls: x, w: xyz, label: "
    assert func(None, x='x', y='y', label='text') == "x: ['x'], y: ['y'], ls: x, w: xyz, label: text"

@pytest.mark.parametrize('func', all_funcs, ids=all_func_ids)
def test_function_call_with_dict_input(func):
    if False:
        print('Hello World!')
    'Tests with dict input, unpacking via preprocess_pipeline'
    data = {'a': 1, 'b': 2}
    assert func(None, data.keys(), data.values()) == "x: ['a', 'b'], y: [1, 2], ls: x, w: xyz, label: None"

@pytest.mark.parametrize('func', all_funcs, ids=all_func_ids)
def test_function_call_with_dict_data(func):
    if False:
        print('Hello World!')
    "Test with dict data -> label comes from the value of 'x' parameter."
    data = {'a': [1, 2], 'b': [8, 9], 'w': 'NOT'}
    assert func(None, 'a', 'b', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b'
    assert func(None, x='a', y='b', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b'
    assert func(None, 'a', 'b', label='', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: '
    assert func(None, 'a', 'b', label='text', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text'
    assert func(None, x='a', y='b', label='', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: '
    assert func(None, x='a', y='b', label='text', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text'

@pytest.mark.parametrize('func', all_funcs, ids=all_func_ids)
def test_function_call_with_dict_data_not_in_data(func):
    if False:
        print('Hello World!')
    'Test the case that one var is not in data -> half replaces, half kept'
    data = {'a': [1, 2], 'w': 'NOT'}
    assert func(None, 'a', 'b', data=data) == "x: [1, 2], y: ['b'], ls: x, w: xyz, label: b"
    assert func(None, x='a', y='b', data=data) == "x: [1, 2], y: ['b'], ls: x, w: xyz, label: b"
    assert func(None, 'a', 'b', label='', data=data) == "x: [1, 2], y: ['b'], ls: x, w: xyz, label: "
    assert func(None, 'a', 'b', label='text', data=data) == "x: [1, 2], y: ['b'], ls: x, w: xyz, label: text"
    assert func(None, x='a', y='b', label='', data=data) == "x: [1, 2], y: ['b'], ls: x, w: xyz, label: "
    assert func(None, x='a', y='b', label='text', data=data) == "x: [1, 2], y: ['b'], ls: x, w: xyz, label: text"

@pytest.mark.parametrize('func', all_funcs, ids=all_func_ids)
def test_function_call_with_pandas_data(func, pd):
    if False:
        print('Hello World!')
    'Test with pandas dataframe -> label comes from ``data["col"].name``.'
    data = pd.DataFrame({'a': np.array([1, 2], dtype=np.int32), 'b': np.array([8, 9], dtype=np.int32), 'w': ['NOT', 'NOT']})
    assert func(None, 'a', 'b', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b'
    assert func(None, x='a', y='b', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b'
    assert func(None, 'a', 'b', label='', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: '
    assert func(None, 'a', 'b', label='text', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text'
    assert func(None, x='a', y='b', label='', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: '
    assert func(None, x='a', y='b', label='text', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text'

def test_function_call_replace_all():
    if False:
        for i in range(10):
            print('nop')
    'Test without a "replace_names" argument, all vars should be replaced.'
    data = {'a': [1, 2], 'b': [8, 9], 'x': 'xyz'}

    @_preprocess_data(label_namer='y')
    def func_replace_all(ax, x, y, ls='x', label=None, w='NOT'):
        if False:
            return 10
        return f'x: {list(x)}, y: {list(y)}, ls: {ls}, w: {w}, label: {label}'
    assert func_replace_all(None, 'a', 'b', w='x', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b'
    assert func_replace_all(None, x='a', y='b', w='x', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b'
    assert func_replace_all(None, 'a', 'b', w='x', label='', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: '
    assert func_replace_all(None, 'a', 'b', w='x', label='text', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text'
    assert func_replace_all(None, x='a', y='b', w='x', label='', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: '
    assert func_replace_all(None, x='a', y='b', w='x', label='text', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text'

def test_no_label_replacements():
    if False:
        i = 10
        return i + 15
    'Test with "label_namer=None" -> no label replacement at all.'

    @_preprocess_data(replace_names=['x', 'y'], label_namer=None)
    def func_no_label(ax, x, y, ls='x', label=None, w='xyz'):
        if False:
            i = 10
            return i + 15
        return f'x: {list(x)}, y: {list(y)}, ls: {ls}, w: {w}, label: {label}'
    data = {'a': [1, 2], 'b': [8, 9], 'w': 'NOT'}
    assert func_no_label(None, 'a', 'b', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: None'
    assert func_no_label(None, x='a', y='b', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: None'
    assert func_no_label(None, 'a', 'b', label='', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: '
    assert func_no_label(None, 'a', 'b', label='text', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text'

def test_more_args_than_pos_parameter():
    if False:
        for i in range(10):
            print('nop')

    @_preprocess_data(replace_names=['x', 'y'], label_namer='y')
    def func(ax, x, y, z=1):
        if False:
            for i in range(10):
                print('nop')
        pass
    data = {'a': [1, 2], 'b': [8, 9], 'w': 'NOT'}
    with pytest.raises(TypeError):
        func(None, 'a', 'b', 'z', 'z', data=data)

def test_docstring_addition():
    if False:
        i = 10
        return i + 15

    @_preprocess_data()
    def funcy(ax, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n        '
    assert re.search('all parameters also accept a string', funcy.__doc__)
    assert not re.search('the following parameters', funcy.__doc__)

    @_preprocess_data(replace_names=[])
    def funcy(ax, x, y, z, bar=None):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n        '
    assert not re.search('all parameters also accept a string', funcy.__doc__)
    assert not re.search('the following parameters', funcy.__doc__)

    @_preprocess_data(replace_names=['bar'])
    def funcy(ax, x, y, z, bar=None):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n        '
    assert not re.search('all parameters also accept a string', funcy.__doc__)
    assert not re.search('the following parameters .*: \\*bar\\*\\.', funcy.__doc__)

    @_preprocess_data(replace_names=['x', 't'])
    def funcy(ax, x, y, z, t=None):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n        '
    assert not re.search('all parameters also accept a string', funcy.__doc__)
    assert not re.search('the following parameters .*: \\*x\\*, \\*t\\*\\.', funcy.__doc__)

def test_data_parameter_replacement():
    if False:
        return 10
    '\n    Test that the docstring contains the correct *data* parameter stub\n    for all methods that we run _preprocess_data() on.\n    '
    program = 'import logging; logging.basicConfig(level=logging.DEBUG); import matplotlib.pyplot as plt'
    cmd = [sys.executable, '-c', program]
    completed_proc = subprocess_run_for_testing(cmd, text=True, capture_output=True)
    assert 'data parameter docstring error' not in completed_proc.stderr

class TestPlotTypes:
    plotters = [Axes.scatter, Axes.bar, Axes.plot]

    @pytest.mark.parametrize('plotter', plotters)
    @check_figures_equal(extensions=['png'])
    def test_dict_unpack(self, plotter, fig_test, fig_ref):
        if False:
            print('Hello World!')
        x = [1, 2, 3]
        y = [4, 5, 6]
        ddict = dict(zip(x, y))
        plotter(fig_test.subplots(), ddict.keys(), ddict.values())
        plotter(fig_ref.subplots(), x, y)

    @pytest.mark.parametrize('plotter', plotters)
    @check_figures_equal(extensions=['png'])
    def test_data_kwarg(self, plotter, fig_test, fig_ref):
        if False:
            while True:
                i = 10
        x = [1, 2, 3]
        y = [4, 5, 6]
        plotter(fig_test.subplots(), 'xval', 'yval', data={'xval': x, 'yval': y})
        plotter(fig_ref.subplots(), x, y)