import numpy as np
import pandas as pd
import pytest
from matplotlib.legend import Legend
from matplotlib.pyplot import Axes, close, rcParams
from ydata_profiling.visualisation.plot import _create_timeseries_heatmap, _plot_pie_chart, _plot_stacked_barh, _prepare_heatmap_data
ids = ['bool', 'cat']
dummy_bool_data = pd.Series({True: 82, False: 36})
dummy_cat_data = pd.Series({'Amadeou_plus': 75, 'Beta_front': 50, 'Calciumus': 20, 'Dimitrius': 1, 'esperagus_anonymoliumus': 75, 'FrigaTTTBrigde_Writap': 50, 'galgarartiy': 30, 'He': 1, 'I': 10, 'JimISGODDOT': 1})

@pytest.fixture
def dataframe() -> pd.DataFrame:
    if False:
        return 10
    size = 100
    return pd.DataFrame({'entity': np.random.randint(3, size=size), 'ints': np.array(size), 'date': pd.date_range('1/1/2022', periods=size), 'floats': np.random.randn(size)})

@pytest.mark.parametrize('data', [dummy_bool_data, dummy_cat_data], ids=ids)
@pytest.mark.parametrize('hide_legend', [False, True])
def test_plot_stacked_barh(data, hide_legend):
    if False:
        i = 10
        return i + 15
    default_colors = rcParams['axes.prop_cycle'].by_key()['color']
    (ax, legend) = _plot_stacked_barh(data=data, colors=default_colors[:len(data)], hide_legend=hide_legend)
    assert issubclass(type(ax), Axes)
    if hide_legend:
        assert legend is None
    else:
        assert issubclass(type(legend), Legend)
    close(ax.get_figure())

@pytest.mark.parametrize('data', [dummy_bool_data, dummy_cat_data], ids=ids)
@pytest.mark.parametrize('hide_legend', [False, True])
def test_plot_pie_chart(data, hide_legend):
    if False:
        while True:
            i = 10
    default_colors = rcParams['axes.prop_cycle'].by_key()['color']
    (ax, legend) = _plot_pie_chart(data=data, colors=default_colors[:len(data)], hide_legend=hide_legend)
    assert issubclass(type(ax), Axes)
    if hide_legend:
        assert legend is None
    else:
        assert issubclass(type(legend), Legend)
    close(ax.get_figure())

def test_timeseries_heatmap(dataframe: pd.DataFrame):
    if False:
        i = 10
        return i + 15
    df = _prepare_heatmap_data(dataframe, 'entity', sortby='ints')
    plot = _create_timeseries_heatmap(df)
    assert isinstance(plot, Axes)
    df = _prepare_heatmap_data(dataframe, 'entity', sortby='date')
    plot = _create_timeseries_heatmap(df)
    assert isinstance(plot, Axes)
    df = _prepare_heatmap_data(dataframe, 'entity', sortby='floats')
    plot = _create_timeseries_heatmap(df)
    assert isinstance(plot, Axes)
    df = _prepare_heatmap_data(dataframe, 'entity')
    plot = _create_timeseries_heatmap(df)
    assert isinstance(plot, Axes)