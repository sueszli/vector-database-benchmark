import os
from dash import Dash, Input, Output, dcc, html
import pytest
import pandas as pd
import numpy as np
from dash.testing.wait import until

@pytest.mark.parametrize('fmt', ('csv', 'json', 'html', 'feather', 'parquet', 'stata', 'pickle'))
def test_dldf001_download_dataframe(fmt, dash_dcc):
    if False:
        while True:
            i = 10
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [2, 1, 5, 6], 'c': ['x', 'x', 'y', 'y']})
    reader = getattr(pd, f'read_{fmt}')
    writer = getattr(df, f'to_{fmt}')
    filename = f'df.{fmt}'
    app = Dash(__name__, prevent_initial_callbacks=True)
    app.layout = html.Div([html.Button('Click me', id='btn'), dcc.Download(id='download')])

    @app.callback(Output('download', 'data'), Input('btn', 'n_clicks'))
    def download(_):
        if False:
            for i in range(10):
                print('nop')
        if fmt in ['csv', 'html', 'excel']:
            return dcc.send_data_frame(writer, filename, index=False)
        if fmt in ['stata']:
            a = dcc.send_data_frame(writer, filename, write_index=False)
            return a
        return dcc.send_data_frame(writer, filename)
    dash_dcc.start_server(app)
    fp = os.path.join(dash_dcc.download_path, filename)
    assert not os.path.isfile(fp)
    dash_dcc.find_element('#btn').click()
    until(lambda : os.path.exists(fp), 10)
    df_download = reader(fp)
    if isinstance(df_download, list):
        df_download = df_download[0]
    assert df.columns.equals(df_download.columns)
    assert df.index.equals(df_download.index)
    np.testing.assert_array_equal(df.values, df_download.values)
    assert dash_dcc.get_logs() == []