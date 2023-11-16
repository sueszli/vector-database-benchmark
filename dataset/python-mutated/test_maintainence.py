from .context import lux
import pytest
import pandas as pd
from lux.vis.Vis import Vis

def test_metadata_subsequent_display(global_var):
    if False:
        while True:
            i = 10
    df = pytest.car_df
    df._ipython_display_()
    assert df._metadata_fresh == True, 'Failed to maintain metadata after display df'
    df._ipython_display_()
    assert df._metadata_fresh == True, 'Failed to maintain metadata after display df'

def test_metadata_subsequent_vis(global_var):
    if False:
        i = 10
        return i + 15
    df = pytest.car_df
    df._ipython_display_()
    assert df._metadata_fresh == True, 'Failed to maintain metadata after display df'
    vis = Vis(['Acceleration', 'Horsepower'], df)
    assert df._metadata_fresh == True, 'Failed to maintain metadata after display df'

def test_metadata_inplace_operation(global_var):
    if False:
        print('Hello World!')
    df = pytest.car_df
    df._ipython_display_()
    assert df._metadata_fresh == True, 'Failed to maintain metadata after display df'
    df.dropna(inplace=True)
    assert df._metadata_fresh == False, 'Failed to expire metadata after in-place Pandas operation'

def test_metadata_new_df_operation(global_var):
    if False:
        i = 10
        return i + 15
    df = pytest.car_df
    df._ipython_display_()
    assert df._metadata_fresh == True, 'Failed to maintain metadata after display df'
    df[['MilesPerGal', 'Acceleration']]
    assert df._metadata_fresh == True, 'Failed to maintain metadata after display df'
    df2 = df[['MilesPerGal', 'Acceleration']]
    assert not hasattr(df2, '_metadata_fresh')

def test_recs_inplace_operation(global_var):
    if False:
        print('Hello World!')
    df = pytest.college_df
    df._ipython_display_()
    assert df._recs_fresh == True, 'Failed to maintain recommendation after display df'
    assert len(df.recommendation['Occurrence']) == 6
    df.drop(columns=['Name'], inplace=True)
    assert 'Name' not in df.columns, 'Failed to perform `drop` operation in-place'
    assert df._recs_fresh == False, 'Failed to maintain recommendation after in-place Pandas operation'
    df._ipython_display_()
    assert len(df.recommendation['Occurrence']) == 5
    assert df._recs_fresh == True, 'Failed to maintain recommendation after display df'

def test_intent_cleared_after_vis_data():
    if False:
        print('Hello World!')
    df = pd.read_csv('https://github.com/lux-org/lux-datasets/blob/master/data/real_estate_tutorial.csv?raw=true')
    df['Month'] = pd.to_datetime(df['Month'], format='%m')
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    df.intent = [lux.Clause('Year'), lux.Clause('PctForeclosured'), lux.Clause('City=Crofton')]
    df._ipython_display_()
    vis = df.recommendation['Similarity'][0]
    vis.data._ipython_display_()
    all_column_vis = vis.data.current_vis[0]
    assert all_column_vis.get_attr_by_channel('x')[0].attribute == 'Year'
    assert all_column_vis.get_attr_by_channel('y')[0].attribute == 'PctForeclosured'