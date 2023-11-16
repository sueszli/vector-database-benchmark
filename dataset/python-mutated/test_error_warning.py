from .context import lux
import pytest
import pandas as pd
from lux.vis.Vis import Vis

def test_intent_str_error(global_var):
    if False:
        while True:
            i = 10
    df = pytest.college_df
    with pytest.raises(TypeError, match='Input intent must be either a list'):
        df.intent = 'bad string input'

def test_export_b4_widget_created(global_var):
    if False:
        print('Hello World!')
    df = pd.read_csv('lux/data/college.csv')
    with pytest.warns(UserWarning, match='No widget attached to the dataframe'):
        df.exported

def test_multi_vis(global_var):
    if False:
        print('Hello World!')
    df = pytest.college_df
    multivis_msg = 'The intent that you specified corresponds to more than one visualization.'
    with pytest.raises(TypeError, match=multivis_msg):
        Vis(['SATAverage', 'AverageCost', 'Geography=?'], df)
    with pytest.raises(TypeError, match=multivis_msg):
        Vis(['SATAverage', '?'], df)
    with pytest.raises(TypeError, match=multivis_msg):
        Vis(['SATAverage', 'AverageCost', 'Region=New England|Southeast'], df)
    with pytest.raises(TypeError, match=multivis_msg):
        Vis(['Region=New England|Southeast'], df)
    with pytest.raises(TypeError, match=multivis_msg):
        Vis(['FundingModel', ['Region', 'ACTMedian']], df)

def test_vis_private_properties(global_var):
    if False:
        for i in range(10):
            print('nop')
    from lux.vis.Vis import Vis
    df = pytest.car_df
    vis = Vis(['Horsepower', 'Weight'], df)
    vis._ipython_display_()
    assert isinstance(vis.data, lux.core.frame.LuxDataFrame)
    with pytest.raises(AttributeError, match="can't set attribute"):
        vis.data = 'some val'
    assert isinstance(vis.code, dict)
    with pytest.raises(AttributeError, match="can't set attribute"):
        vis.code = 'some val'
    assert isinstance(vis.min_max, dict)
    with pytest.raises(AttributeError, match="can't set attribute"):
        vis.min_max = 'some val'
    assert vis.mark == 'scatter'
    with pytest.raises(AttributeError, match="can't set attribute"):
        vis.mark = 'some val'

def test_lux_warnings(global_var):
    if False:
        print('Hello World!')
    df = pd.DataFrame()
    df._ipython_display_()
    assert df._widget.message == f'<ul><li>Lux cannot operate on an empty DataFrame.</li></ul>'
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    df._ipython_display_()
    assert df._widget.message == f'<ul><li>The DataFrame is too small to visualize. To generate visualizations in Lux, the DataFrame must contain at least 5 rows.</li></ul>'
    df = pytest.car_df
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    new_df = df.set_index(['Name', 'Cylinders'])
    new_df._ipython_display_()
    assert new_df._widget.message == f'<ul><li>Lux does not currently support visualizations in a DataFrame with hierarchical indexes.\nPlease convert the DataFrame into a flat table via pandas.DataFrame.reset_index.</li></ul>'