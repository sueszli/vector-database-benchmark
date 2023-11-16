from .context import lux
import pytest
import pandas as pd
from lux.vis.Vis import Vis
from lux.executor.PandasExecutor import PandasExecutor

def test_scatter_code_export(global_var):
    if False:
        print('Hello World!')
    df = pytest.car_df
    vis = Vis([lux.Clause('Horsepower'), lux.Clause('Acceleration')], df)
    PandasExecutor.execute([vis], df)
    code = vis.to_code('python')
    try:
        exec(code, globals())
        create_chart_data(df, vis)
    except:
        assert False

def test_color_scatter_code_export(global_var):
    if False:
        while True:
            i = 10
    df = pytest.car_df
    vis = Vis([lux.Clause('Horsepower'), lux.Clause('Acceleration'), lux.Clause('Origin')], df)
    PandasExecutor.execute([vis], df)
    code = vis.to_code('python')
    try:
        exec(code, globals())
        create_chart_data(df, vis)
    except:
        assert False

def test_histogram_code_export(global_var):
    if False:
        i = 10
        return i + 15
    df = pytest.car_df
    vis = Vis([lux.Clause('Horsepower')], df)
    PandasExecutor.execute([vis], df)
    code = vis.to_code('python')
    try:
        exec(code, globals())
        create_chart_data(df, vis)
    except:
        assert False

def test_heatmap_code_export(global_var):
    if False:
        print('Hello World!')
    df = pd.read_csv('https://raw.githubusercontent.com/lux-org/lux-datasets/master/data/airbnb_nyc.csv')
    lux.config._heatmap_start = 100
    vis = Vis(['price', 'longitude'], df)
    PandasExecutor.execute([vis], df)
    code = vis.to_code('python')
    try:
        exec(code, globals())
        create_chart_data(df, vis)
    except:
        assert False
    lux.config._heatmap_start = 5000