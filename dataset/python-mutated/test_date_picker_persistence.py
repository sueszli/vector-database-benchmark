from datetime import datetime
from dash import Dash, Input, Output, html, dcc

def test_rdpr001_persisted_dps(dash_dcc):
    if False:
        return 10
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div([html.Button('fire callback', id='btn', n_clicks=1), html.Div(children=[html.Div(id='container'), html.P('dps', id='dps-p')])])

    @app.callback(Output('container', 'children'), [Input('btn', 'n_clicks')])
    def update_output(value):
        if False:
            i = 10
            return i + 15
        return dcc.DatePickerSingle(id='dps', min_date_allowed=datetime(2020, 1, 1), max_date_allowed=datetime(2020, 1, 7), date=datetime(2020, 1, 3, 1, 1, 1, value), persistence=True, persistence_type='session')

    @app.callback(Output('dps-p', 'children'), [Input('dps', 'date')])
    def display_dps(value):
        if False:
            while True:
                i = 10
        return value
    dash_dcc.start_server(app)
    dash_dcc.select_date_single('dps', day='2')
    dash_dcc.wait_for_text_to_equal('#dps-p', '2020-01-02')
    dash_dcc.find_element('#btn').click()
    dash_dcc.wait_for_text_to_equal('#dps-p', '2020-01-02')
    assert dash_dcc.get_logs() == []

def test_rdpr002_persisted_dpr(dash_dcc):
    if False:
        while True:
            i = 10
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div([html.Button('fire callback', id='btn', n_clicks=1), html.Div(children=[html.Div(id='container'), html.P('dpr', id='dpr-p-start'), html.P('dpr', id='dpr-p-end')])])

    @app.callback(Output('container', 'children'), [Input('btn', 'n_clicks')])
    def update_output(value):
        if False:
            for i in range(10):
                print('nop')
        return dcc.DatePickerRange(id='dpr', min_date_allowed=datetime(2020, 1, 1), max_date_allowed=datetime(2020, 1, 7), start_date=datetime(2020, 1, 3, 1, 1, 1, value), end_date=datetime(2020, 1, 4, 1, 1, 1, value), persistence=True, persistence_type='session')

    @app.callback(Output('dpr-p-start', 'children'), [Input('dpr', 'start_date')])
    def display_dpr_start(value):
        if False:
            print('Hello World!')
        return value

    @app.callback(Output('dpr-p-end', 'children'), [Input('dpr', 'end_date')])
    def display_dpr_end(value):
        if False:
            return 10
        return value
    dash_dcc.start_server(app)
    dash_dcc.select_date_range('dpr', (2, 5))
    dash_dcc.wait_for_text_to_equal('#dpr-p-start', '2020-01-02')
    dash_dcc.wait_for_text_to_equal('#dpr-p-end', '2020-01-05')
    dash_dcc.find_element('#btn').click()
    dash_dcc.wait_for_text_to_equal('#dpr-p-start', '2020-01-02')
    dash_dcc.wait_for_text_to_equal('#dpr-p-end', '2020-01-05')
    assert dash_dcc.get_logs() == []