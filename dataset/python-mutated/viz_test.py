import numpy as np
import pyarrow as pa
import matplotlib
import matplotlib.pyplot as plt
import pytest
import vaex
matplotlib.use('agg')

def test_histogram(df_trimmed):
    if False:
        return 10
    df = df_trimmed
    plt.figure()
    df.x.viz.histogram(show=False)

def test_heatmap(df_trimmed):
    if False:
        i = 10
        return i + 15
    df = df_trimmed
    plt.figure()
    df.x.viz.histogram(show=False)

@pytest.mark.mpl_image_compare(filename='test_histogram_with_what.png')
def test_histogram_with_what(df):
    if False:
        return 10
    fig = plt.figure()
    df.viz.histogram(df.x, what=np.clip(np.log(-vaex.stat.mean(df.z)), 0, 10), limits='99.7%')
    return fig

@pytest.mark.mpl_image_compare(filename='test_heatmap_with_what.png')
def test_heatmap_with_what(df):
    if False:
        print('Hello World!')
    df.viz.heatmap(df.x, df.y, what=np.log(vaex.stat.count() + 1), limits='99.7%', selection=[None, df.x < df.y, df.x < -10])

def test_histogram_with_selection(df):
    if False:
        print('Hello World!')
    x = np.random.normal(size=10000)
    fraction_missing = 0.7
    missing_mask = np.random.binomial(1, fraction_missing, size=x.shape).astype(bool)
    x_numpy = np.ma.array(x, mask=missing_mask)
    x_arrow = pa.array(x, mask=missing_mask)
    df = vaex.from_arrays(x_numpy=x_numpy, x_arrow=x_arrow)
    fig_numpy = df.x_numpy.viz.histogram(selection='x_numpy > 0')[0]
    fig_arrow = df.x_arrow.viz.histogram(selection='x_arrow > 0')[0]
    assert all(fig_numpy.get_xdata() == fig_arrow.get_xdata())
    assert all(fig_numpy.get_ydata() == fig_arrow.get_ydata())
    fig_numpy = df.x_numpy.viz.histogram(selection='x_numpy > 0', limits='90%')[0]
    fig_arrow = df.x_arrow.viz.histogram(selection='x_arrow > 0', limits='90%')[0]
    assert all(fig_numpy.get_xdata() == fig_arrow.get_xdata())
    assert all(fig_numpy.get_ydata() == fig_arrow.get_ydata())