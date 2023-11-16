from multiprocessing import Lock
import pytest
from dash import Dash, Input, Output, dcc, html
from dash.testing.wait import until

def test_rdls001_multi_loading_components(dash_duo):
    if False:
        print('Hello World!')
    lock = Lock()
    app = Dash(__name__)
    app.layout = html.Div(children=[html.H3('Edit text input to see loading state'), dcc.Input(id='input-3', value='Input triggers the loading states'), dcc.Loading(className='loading-1', children=[html.Div(id='loading-output-1')], type='default'), html.Div([dcc.Loading(className='loading-2', children=[html.Div([html.Div(id='loading-output-2')])], type='circle'), dcc.Loading(className='loading-3', children=dcc.Graph(id='graph'), type='cube')])])

    @app.callback([Output('graph', 'figure'), Output('loading-output-1', 'children'), Output('loading-output-2', 'children')], [Input('input-3', 'value')])
    def input_triggers_nested(value):
        if False:
            print('Hello World!')
        with lock:
            return (dict(data=[dict(y=[1, 4, 2, 3])]), value, value)

    def wait_for_all_spinners():
        if False:
            while True:
                i = 10
        dash_duo.find_element('.loading-1 .dash-spinner.dash-default-spinner')
        dash_duo.find_element('.loading-2 .dash-spinner.dash-sk-circle')
        dash_duo.find_element('.loading-3 .dash-spinner.dash-cube-container')

    def wait_for_no_spinners():
        if False:
            for i in range(10):
                print('nop')
        dash_duo.wait_for_no_elements('.dash-spinner')
    with lock:
        dash_duo.start_server(app)
        wait_for_all_spinners()
    wait_for_no_spinners()
    with lock:
        dash_duo.find_element('#input-3').send_keys('X')
        wait_for_all_spinners()
    wait_for_no_spinners()

def test_rdls002_chained_loading_states(dash_duo):
    if False:
        for i in range(10):
            print('nop')
    (lock1, lock2, lock34) = (Lock(), Lock(), Lock())
    app = Dash(__name__)

    def loading_wrapped_div(_id, color):
        if False:
            return 10
        return html.Div(dcc.Loading(html.Div(id=_id, style={'width': 200, 'height': 200, 'backgroundColor': color}), className=_id), style={'display': 'inline-block'})
    app.layout = html.Div([html.Button(id='button', children='Start', n_clicks=0), loading_wrapped_div('output-1', 'hotpink'), loading_wrapped_div('output-2', 'rebeccapurple'), loading_wrapped_div('output-3', 'green'), loading_wrapped_div('output-4', '#FF851B')])

    @app.callback(Output('output-1', 'children'), [Input('button', 'n_clicks')])
    def update_output_1(n_clicks):
        if False:
            print('Hello World!')
        with lock1:
            return 'Output 1: {}'.format(n_clicks)

    @app.callback(Output('output-2', 'children'), [Input('output-1', 'children')])
    def update_output_2(children):
        if False:
            for i in range(10):
                print('nop')
        with lock2:
            return 'Output 2: {}'.format(children)

    @app.callback([Output('output-3', 'children'), Output('output-4', 'children')], [Input('output-2', 'children')])
    def update_output_34(children):
        if False:
            while True:
                i = 10
        with lock34:
            return ('Output 3: {}'.format(children), 'Output 4: {}'.format(children))
    dash_duo.start_server(app)

    def find_spinners(*nums):
        if False:
            return 10
        if not nums:
            dash_duo.wait_for_no_elements('.dash-spinner')
            return
        for n in nums:
            dash_duo.find_element('.output-{} .dash-spinner'.format(n))
        assert len(dash_duo.find_elements('.dash-spinner')) == len(nums)

    def find_text(spec):
        if False:
            for i in range(10):
                print('nop')
        templates = ['Output 1: {}', 'Output 2: Output 1: {}', 'Output 3: Output 2: Output 1: {}', 'Output 4: Output 2: Output 1: {}']
        for (n, v) in spec.items():
            dash_duo.wait_for_text_to_equal('#output-{}'.format(n), templates[n - 1].format(v))
    find_text({1: 0, 2: 0, 3: 0, 4: 0})
    find_spinners()
    btn = dash_duo.find_element('#button')
    lock1.acquire()
    btn.click()
    find_spinners(1)
    find_text({2: 0, 3: 0, 4: 0})
    lock2.acquire()
    lock1.release()
    find_spinners(2)
    find_text({1: 1, 3: 0, 4: 0})
    lock34.acquire()
    lock2.release()
    find_spinners(3, 4)
    find_text({1: 1, 2: 1})
    lock34.release()
    find_spinners()
    find_text({1: 1, 2: 1, 3: 1, 4: 1})

@pytest.mark.parametrize('kwargs, expected_update_title, clientside_title', [({}, 'Updating...', False), ({'update_title': None}, 'Dash', False), ({'update_title': ''}, 'Dash', False), ({'update_title': 'Hello World'}, 'Hello World', False), ({}, 'Updating...', True), ({'update_title': None}, 'Dash', True), ({'update_title': ''}, 'Dash', True), ({'update_title': 'Hello World'}, 'Hello World', True)])
def test_rdls003_update_title(dash_duo, kwargs, expected_update_title, clientside_title):
    if False:
        return 10
    app = Dash('Dash', **kwargs)
    lock = Lock()
    app.layout = html.Div(children=[html.H3('Press button see document title updating'), html.Div(id='output'), html.Button('Update', id='button', n_clicks=0), html.Button('Update Page', id='page', n_clicks=0), html.Div(id='dummy')])
    if clientside_title:
        app.clientside_callback("\n            function(n_clicks) {\n                document.title = 'Page ' + n_clicks;\n                return 'Page ' + n_clicks;\n            }\n            ", Output('dummy', 'children'), [Input('page', 'n_clicks')])

    @app.callback(Output('output', 'children'), [Input('button', 'n_clicks')])
    def update(n):
        if False:
            i = 10
            return i + 15
        with lock:
            return n
    with lock:
        dash_duo.start_server(app)
        if not clientside_title:
            until(lambda : dash_duo.driver.title == expected_update_title, timeout=1)
    until(lambda : dash_duo.driver.title == 'Page 0' if clientside_title else 'Dash', timeout=1)
    with lock:
        dash_duo.find_element('#button').click()
        if clientside_title and (not kwargs.get('update_title', True)):
            until(lambda : dash_duo.driver.title == 'Page 0', timeout=1)
        else:
            until(lambda : dash_duo.driver.title == expected_update_title, timeout=1)
    if clientside_title:
        dash_duo.find_element('#page').click()
        dash_duo.wait_for_text_to_equal('#dummy', 'Page 1')
        until(lambda : dash_duo.driver.title == 'Page 1', timeout=1)
    dash_duo.find_element('#button').click()
    dash_duo.wait_for_text_to_equal('#output', '2')
    if clientside_title:
        until(lambda : dash_duo.driver.title == 'Page 1', timeout=1)
    else:
        until(lambda : dash_duo.driver.title == 'Dash', timeout=1)

@pytest.mark.parametrize('update_title', [None, 'Custom Update Title'])
def test_rdls004_update_title_chained_callbacks(dash_duo, update_title):
    if False:
        i = 10
        return i + 15
    initial_title = 'Initial Title'
    app = Dash('Dash', title=initial_title, update_title=update_title)
    lock = Lock()
    app.layout = html.Div(children=[html.Button(id='page-title', n_clicks=0, children='Page Title'), html.Div(id='page-output'), html.Div(id='final-output')])
    app.clientside_callback("\n        function(n_clicks) {\n            if (n_clicks > 0) {\n                document.title = 'Page ' + n_clicks;\n            }\n            return n_clicks;\n        }\n        ", Output('page-output', 'children'), [Input('page-title', 'n_clicks')])

    @app.callback(Output('final-output', 'children'), [Input('page-output', 'children')])
    def update(n):
        if False:
            return 10
        with lock:
            return n
    dash_duo.start_server(app)
    dash_duo.wait_for_text_to_equal('#final-output', '0')
    until(lambda : dash_duo.driver.title == initial_title, timeout=1)
    with lock:
        dash_duo.find_element('#page-title').click()
        if update_title:
            until(lambda : dash_duo.driver.title == update_title, timeout=1)
        else:
            until(lambda : dash_duo.driver.title == 'Page 1', timeout=1)
    dash_duo.wait_for_text_to_equal('#final-output', '1')
    until(lambda : dash_duo.driver.title == 'Page 1', timeout=1)