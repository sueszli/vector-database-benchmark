import json
import pytest
from flaky import flaky
from dash import Dash, Input, Output, MATCH, html
from dash.exceptions import PreventUpdate
flavors = [{'clientside': False, 'content': False, 'global': False}, {'clientside': True, 'content': False, 'global': False}, {'clientside': False, 'content': True, 'global': False}, {'clientside': True, 'content': True, 'global': True}, {'clientside': False, 'content': False, 'global': True}]

def make_app(flavor):
    if False:
        i = 10
        return i + 15
    kwargs = {}
    if flavor['global']:
        kwargs['prevent_initial_callbacks'] = True
    return Dash(__name__, **kwargs)

def content_callback(app, flavor, layout):
    if False:
        print('Hello World!')
    kwargs = {}
    if flavor['global']:
        kwargs['prevent_initial_call'] = False
    if flavor['content']:
        app.layout = html.Div(id='content')

        @app.callback(Output('content', 'children'), [Input('content', 'style')], **kwargs)
        def set_content(_):
            if False:
                while True:
                    i = 10
            return layout
    else:
        app.layout = layout

def const_callback(app, flavor, val, outputs, inputs, prevent_initial_call=False):
    if False:
        i = 10
        return i + 15
    kwargs = {}
    if prevent_initial_call != flavor['global']:
        kwargs['prevent_initial_call'] = prevent_initial_call
    if flavor['clientside']:
        vstr = json.dumps(val)
        app.clientside_callback('function() { return ' + vstr + '; }', outputs, inputs, **kwargs)
    else:

        @app.callback(outputs, inputs, **kwargs)
        def f(*args):
            if False:
                i = 10
                return i + 15
            return val

def concat_callback(app, flavor, outputs, inputs, prevent_initial_call=False):
    if False:
        print('Hello World!')
    kwargs = {}
    if prevent_initial_call != flavor['global']:
        kwargs['prevent_initial_call'] = prevent_initial_call
    multi_out = isinstance(outputs, (list, tuple))
    if flavor['clientside']:
        app.clientside_callback("\n            function() {\n                var out = '';\n                for(var i = 0; i < arguments.length; i++) {\n                    out += String(arguments[i]);\n                }\n                return X;\n            }\n            ".replace('X', '[' + ','.join(['out'] * len(outputs)) + ']' if multi_out else 'out'), outputs, inputs, **kwargs)
    else:

        @app.callback(outputs, inputs, **kwargs)
        def f(*args):
            if False:
                print('Hello World!')
            out = ''.join((str(arg) for arg in args))
            return [out] * len(outputs) if multi_out else out

@pytest.mark.parametrize('flavor', flavors)
def test_cbpi001_prevent_initial_call(flavor, dash_duo):
    if False:
        for i in range(10):
            print('nop')
    app = make_app(flavor)
    layout = html.Div([html.Button('click', id='btn'), html.Div('A', id='a'), html.Div('B', id='b'), html.Div('C', id='c'), html.Div('D', id='d'), html.Div('E', id='e'), html.Div('F', id='f')])
    content_callback(app, flavor, layout)
    const_callback(app, flavor, 'Click', Output('a', 'children'), [Input('btn', 'n_clicks')], prevent_initial_call=True)
    concat_callback(app, flavor, Output('b', 'children'), [Input('a', 'children')])
    concat_callback(app, flavor, Output('c', 'children'), [Input('a', 'children')], prevent_initial_call=True)

    @app.callback(Output('d', 'children'), [Input('d', 'style')])
    def d(_):
        if False:
            return 10
        raise PreventUpdate
    concat_callback(app, flavor, Output('e', 'children'), [Input('a', 'children'), Input('d', 'children')])
    concat_callback(app, flavor, Output('f', 'children'), [Input('a', 'children'), Input('b', 'children'), Input('d', 'children')], prevent_initial_call=True)
    dash_duo.start_server(app)
    dash_duo.wait_for_text_to_equal('#f', 'AAD')
    (dash_duo.wait_for_text_to_equal('#e', 'AD'),)
    dash_duo.wait_for_text_to_equal('#d', 'D')
    dash_duo.wait_for_text_to_equal('#c', 'C')
    dash_duo.wait_for_text_to_equal('#b', 'A')
    dash_duo.wait_for_text_to_equal('#a', 'A')
    dash_duo.find_element('#btn').click()
    dash_duo.wait_for_text_to_equal('#f', 'ClickClickD')
    (dash_duo.wait_for_text_to_equal('#e', 'ClickD'),)
    dash_duo.wait_for_text_to_equal('#d', 'D')
    dash_duo.wait_for_text_to_equal('#c', 'Click')
    dash_duo.wait_for_text_to_equal('#b', 'Click')
    dash_duo.wait_for_text_to_equal('#a', 'Click')

@flaky(max_runs=3)
@pytest.mark.parametrize('flavor', flavors)
def test_cbpi002_pattern_matching(flavor, dash_duo):
    if False:
        print('Hello World!')
    app = make_app(flavor)
    layout = html.Div([html.Button('click', id={'i': 0, 'j': 'btn'}, className='btn'), html.Div('A', id={'i': 0, 'j': 'a'}, className='a'), html.Div('B', id={'i': 0, 'j': 'b'}, className='b'), html.Div('C', id={'i': 0, 'j': 'c'}, className='c'), html.Div('D', id={'i': 0, 'j': 'd'}, className='d'), html.Div('E', id={'i': 0, 'j': 'e'}, className='e'), html.Div('F', id={'i': 0, 'j': 'f'}, className='f')])
    content_callback(app, flavor, layout)
    const_callback(app, flavor, 'Click', Output({'i': MATCH, 'j': 'a'}, 'children'), [Input({'i': MATCH, 'j': 'btn'}, 'n_clicks')], prevent_initial_call=True)
    concat_callback(app, flavor, Output({'i': MATCH, 'j': 'b'}, 'children'), [Input({'i': MATCH, 'j': 'a'}, 'children')])
    concat_callback(app, flavor, Output({'i': MATCH, 'j': 'c'}, 'children'), [Input({'i': MATCH, 'j': 'a'}, 'children')], prevent_initial_call=True)

    @app.callback(Output({'i': MATCH, 'j': 'd'}, 'children'), [Input({'i': MATCH, 'j': 'd'}, 'style')])
    def d(_):
        if False:
            i = 10
            return i + 15
        raise PreventUpdate
    concat_callback(app, flavor, Output({'i': MATCH, 'j': 'e'}, 'children'), [Input({'i': MATCH, 'j': 'a'}, 'children'), Input({'i': MATCH, 'j': 'd'}, 'children')])
    concat_callback(app, flavor, Output({'i': MATCH, 'j': 'f'}, 'children'), [Input({'i': MATCH, 'j': 'a'}, 'children'), Input({'i': MATCH, 'j': 'b'}, 'children'), Input({'i': MATCH, 'j': 'd'}, 'children')], prevent_initial_call=True)
    dash_duo.start_server(app)
    dash_duo.wait_for_text_to_equal('.f', 'AAD')
    (dash_duo.wait_for_text_to_equal('.e', 'AD'),)
    dash_duo.wait_for_text_to_equal('.d', 'D')
    dash_duo.wait_for_text_to_equal('.c', 'C')
    dash_duo.wait_for_text_to_equal('.b', 'A')
    dash_duo.wait_for_text_to_equal('.a', 'A')
    dash_duo.find_element('.btn').click()
    dash_duo.wait_for_text_to_equal('.f', 'ClickClickD')
    (dash_duo.wait_for_text_to_equal('.e', 'ClickD'),)
    dash_duo.wait_for_text_to_equal('.d', 'D')
    dash_duo.wait_for_text_to_equal('.c', 'Click')
    dash_duo.wait_for_text_to_equal('.b', 'Click')
    dash_duo.wait_for_text_to_equal('.a', 'Click')

@pytest.mark.parametrize('flavor', flavors)
def test_cbpi003_multi_outputs(flavor, dash_duo):
    if False:
        return 10
    app = make_app(flavor)
    layout = html.Div([html.Button('click', id='btn'), html.Div('A', id='a'), html.Div('B', id='b'), html.Div('C', id='c'), html.Div('D', id='d'), html.Div('E', id='e'), html.Div('F', id='f'), html.Div('G', id='g')])
    content_callback(app, flavor, layout)
    const_callback(app, flavor, ['Blue', 'Cheese'], [Output('a', 'children'), Output('b', 'children')], [Input('btn', 'n_clicks')], prevent_initial_call=True)
    concat_callback(app, flavor, [Output('c', 'children'), Output('d', 'children')], [Input('a', 'children'), Input('b', 'children')], prevent_initial_call=True)
    concat_callback(app, flavor, [Output('e', 'children'), Output('f', 'children')], [Input('a', 'children')], prevent_initial_call=True)
    concat_callback(app, flavor, Output('g', 'children'), [Input('f', 'children')])
    dash_duo.start_server(app)
    dash_duo.wait_for_text_to_equal('#g', 'F')
    dash_duo.wait_for_text_to_equal('#f', 'F')
    dash_duo.wait_for_text_to_equal('#e', 'E')
    dash_duo.wait_for_text_to_equal('#d', 'D')
    dash_duo.wait_for_text_to_equal('#c', 'C')
    dash_duo.wait_for_text_to_equal('#b', 'B')
    dash_duo.wait_for_text_to_equal('#a', 'A')
    dash_duo.find_element('#btn').click()
    dash_duo.wait_for_text_to_equal('#g', 'Blue')
    dash_duo.wait_for_text_to_equal('#f', 'Blue')
    dash_duo.wait_for_text_to_equal('#e', 'Blue')
    dash_duo.wait_for_text_to_equal('#d', 'BlueCheese')
    dash_duo.wait_for_text_to_equal('#c', 'BlueCheese')
    dash_duo.wait_for_text_to_equal('#b', 'Cheese')
    dash_duo.wait_for_text_to_equal('#a', 'Blue')

def test_cbpi004_positional_arg(dash_duo):
    if False:
        for i in range(10):
            print('nop')
    app = Dash(__name__)
    app.layout = html.Div([html.Button('click', id='btn'), html.Div(id='out')])

    @app.callback(Output('out', 'children'), Input('btn', 'n_clicks'), True)
    def f(n):
        if False:
            while True:
                i = 10
        return n
    dash_duo.start_server(app)
    dash_duo._wait_for_callbacks()
    dash_duo.wait_for_text_to_equal('#out', '')
    dash_duo.find_element('#btn').click()
    dash_duo.wait_for_text_to_equal('#out', '1')