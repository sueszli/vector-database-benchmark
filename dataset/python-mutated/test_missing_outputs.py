import pytest
from multiprocessing import Lock, Value
import dash
from dash import Dash, Input, Output, ALL, MATCH, html, dcc
from dash.testing.wait import until
debugging = dict(debug=True, use_reloader=False, use_debugger=True, dev_tools_hot_reload=False)

@pytest.mark.parametrize('with_simple', (False, True))
def test_cbmo001_all_output(with_simple, dash_duo):
    if False:
        while True:
            i = 10
    app = Dash(__name__)
    app.layout = html.Div(children=[html.Button('items', id='items'), html.Button('values', id='values'), html.Div(id='content'), html.Div('Output init', id='output')])

    @app.callback(Output('content', 'children'), [Input('items', 'n_clicks')])
    def set_content(n1):
        if False:
            for i in range(10):
                print('nop')
        return [html.Div(id={'i': i}) for i in range((n1 or 0) % 4)]
    if with_simple:

        @app.callback([Output({'i': ALL}, 'children'), Output('output', 'children')], [Input('values', 'n_clicks'), Input({'i': ALL}, 'id')])
        def content_and_output(n2, content_ids):
            if False:
                return 10
            n1 = len(content_ids)
            content = [n2 or 0] * n1
            return (content, sum(content))
    else:

        @app.callback(Output({'i': ALL}, 'children'), [Input('values', 'n_clicks')])
        def content_inner(n2):
            if False:
                i = 10
                return i + 15
            n1 = len(dash.callback_context.outputs_list)
            if not n1:
                raise ValueError('should not be called with no outputs!')
            return [n2 or 0] * n1

        @app.callback(Output('output', 'children'), [Input({'i': ALL}, 'children')])
        def out2(contents):
            if False:
                while True:
                    i = 10
            return sum(contents)
    dash_duo.start_server(app)
    dash_duo.wait_for_text_to_equal('#content', '')
    dash_duo.wait_for_text_to_equal('#output', '0')
    actions = [['#values', '', '0'], ['#items', '1', '1'], ['#values', '2', '2'], ['#items', '2\n2', '4'], ['#values', '3\n3', '6'], ['#items', '3\n3\n3', '9'], ['#values', '4\n4\n4', '12'], ['#items', '', '0'], ['#values', '', '0'], ['#items', '5', '5']]
    for (selector, content, output) in actions:
        dash_duo.find_element(selector).click()
        dash_duo.wait_for_text_to_equal('#content', content)
        dash_duo.wait_for_text_to_equal('#output', output)
    assert not dash_duo.get_logs()

@pytest.mark.parametrize('with_simple', (False, True))
def test_cbmo002_all_and_match_output(with_simple, dash_duo):
    if False:
        while True:
            i = 10
    app = Dash(__name__)
    app.layout = html.Div(children=[html.Button('items', id='items'), html.Button('values', id='values'), html.Div(id='content')])

    @app.callback(Output('content', 'children'), [Input('items', 'n_clicks')])
    def set_content(n1):
        if False:
            print('Hello World!')
        return [html.Div([html.Div([html.Div(id={'i': i, 'j': j}) for i in range(((n1 or 0) + j) % 4)], className='content{}'.format(j)), html.Div(id={'j': j}, className='output{}'.format(j)), html.Hr()]) for j in range(4)]
    if with_simple:

        @app.callback([Output({'i': ALL, 'j': MATCH}, 'children'), Output({'j': MATCH}, 'children')], [Input('values', 'n_clicks'), Input({'i': ALL, 'j': MATCH}, 'id')])
        def content_and_output(n2, content_ids):
            if False:
                return 10
            n1 = len(content_ids)
            content = [n2 or 0] * n1
            return (content, sum(content))
    else:

        @app.callback(Output({'i': ALL, 'j': MATCH}, 'children'), [Input('values', 'n_clicks')])
        def content_inner(n2):
            if False:
                while True:
                    i = 10
            n1 = len(dash.callback_context.outputs_list)
            if not n1:
                raise ValueError('should not be called with no outputs!')
            return [n2 or 0] * n1

        @app.callback(Output({'j': MATCH}, 'children'), [Input({'i': ALL, 'j': MATCH}, 'children')])
        def out2(contents):
            if False:
                for i in range(10):
                    print('nop')
            return sum(contents)
    dash_duo.start_server(app, **debugging)
    dash_duo.wait_for_text_to_equal('.content0', '')
    dash_duo.wait_for_text_to_equal('.output0', '0')
    actions = [['#values', [['', '0'], ['1', '1'], ['1\n1', '2'], ['1\n1\n1', '3']]], ['#items', [['1', '1'], ['1\n1', '2'], ['1\n1\n1', '3'], ['', '0']]], ['#values', [['2', '2'], ['2\n2', '4'], ['2\n2\n2', '6'], ['', '0']]], ['#items', [['2\n2', '4'], ['2\n2\n2', '6'], ['', '0'], ['2', '2']]], ['#values', [['3\n3', '6'], ['3\n3\n3', '9'], ['', '0'], ['3', '3']]], ['#items', [['3\n3\n3', '9'], ['', '0'], ['3', '3'], ['3\n3', '6']]], ['#values', [['4\n4\n4', '12'], ['', '0'], ['4', '4'], ['4\n4', '8']]], ['#items', [['', '0'], ['4', '4'], ['4\n4', '8'], ['4\n4\n4', '12']]], ['#values', [['', '0'], ['5', '5'], ['5\n5', '10'], ['5\n5\n5', '15']]], ['#items', [['5', '5'], ['5\n5', '10'], ['5\n5\n5', '15'], ['', '0']]]]
    for (selector, output_spec) in actions:
        dash_duo.find_element(selector).click()
        for (j, (content, output)) in enumerate(output_spec):
            dash_duo.wait_for_text_to_equal('.content{}'.format(j), content)
            dash_duo.wait_for_text_to_equal('.output{}'.format(j), output)
    assert not dash_duo.get_logs()

def test_cbmo003_multi_all(dash_duo):
    if False:
        print('Hello World!')
    app = dash.Dash(__name__)
    app.layout = html.Div(children=[html.Button('items', id='items'), html.Button('values', id='values'), html.Div(id='content1'), html.Hr(), html.Div(id='content2'), html.Hr(), html.Div('Output init', id='output')])

    @app.callback([Output('content1', 'children'), Output('content2', 'children')], [Input('items', 'n_clicks')])
    def content(n1):
        if False:
            print('Hello World!')
        c1 = [html.Div(id={'i': i}) for i in range(((n1 or 0) + 2) % 4)]
        c2 = [html.Div(id={'j': j}) for j in range((n1 or 0) % 3)]
        return (c1, c2)

    @app.callback([Output({'i': ALL}, 'children'), Output({'j': ALL}, 'children')], [Input('values', 'n_clicks')])
    def content_inner(n2):
        if False:
            while True:
                i = 10
        n1i = len(dash.callback_context.outputs_list[0])
        n1j = len(dash.callback_context.outputs_list[1])
        if not n1i + n1j:
            raise ValueError('should not be called with no outputs!')
        return ([n2 or 0] * n1i, [(n2 or 0) + 2] * n1j)

    @app.callback(Output('output', 'children'), [Input({'i': ALL}, 'children'), Input({'j': ALL}, 'children')])
    def out2(ci, cj):
        if False:
            while True:
                i = 10
        return sum(ci) + sum(cj)
    dash_duo.start_server(app)
    dash_duo.wait_for_text_to_equal('#content1', '0\n0')
    dash_duo.wait_for_text_to_equal('#content2', '')
    dash_duo.wait_for_text_to_equal('#output', '0')
    actions = [['#values', '1\n1', '', '2'], ['#items', '1\n1\n1', '3', '6'], ['#values', '2\n2\n2', '4', '10'], ['#items', '', '4\n4', '8'], ['#values', '', '5\n5', '10'], ['#items', '3', '', '3'], ['#values', '4', '', '4'], ['#items', '4\n4', '6', '14'], ['#values', '5\n5', '7', '17'], ['#items', '5\n5\n5', '7\n7', '29'], ['#values', '6\n6\n6', '8\n8', '34'], ['#items', '', '', '0'], ['#values', '', '', '0'], ['#items', '7', '9', '16']]
    for (selector, content1, content2, output) in actions:
        dash_duo.find_element(selector).click()
        dash_duo.wait_for_text_to_equal('#content1', content1)
        dash_duo.wait_for_text_to_equal('#content2', content2)
        dash_duo.wait_for_text_to_equal('#output', output)
    assert not dash_duo.get_logs()

def test_cbmo004_removing_element_while_waiting_to_update(dash_duo):
    if False:
        return 10
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div([dcc.RadioItems(id='toc', options=[{'label': i, 'value': i} for i in ['1', '2']], value='1'), html.Div(id='body')])
    call_counts = {'body': Value('i', 0), 'button-output': Value('i', 0)}
    lock = Lock()

    @app.callback(Output('body', 'children'), Input('toc', 'value'))
    def update_body(chapter):
        if False:
            i = 10
            return i + 15
        call_counts['body'].value += 1
        if chapter == '1':
            return [html.Div('Chapter 1', id='ch1-title'), html.Button('clicking this button takes forever', id='button'), html.Div(id='button-output')]
        elif chapter == '2':
            return 'Chapter 2'
        else:
            raise Exception('chapter is {}'.format(chapter))

    @app.callback(Output('button-output', 'children'), Input('button', 'n_clicks'))
    def this_callback_takes_forever(n_clicks):
        if False:
            i = 10
            return i + 15
        if not n_clicks:
            return 'Initial Value'
        with lock:
            call_counts['button-output'].value += 1
            return 'New value!'
    dash_duo.start_server(app)
    dash_duo.wait_for_text_to_equal('#ch1-title', 'Chapter 1')
    assert call_counts['body'].value == 1

    def chapter2_assertions():
        if False:
            while True:
                i = 10
        dash_duo.wait_for_text_to_equal('#body', 'Chapter 2')
        layout = dash_duo.driver.execute_script('return JSON.parse(JSON.stringify(window.store.getState().layout))')
        dcc_radio = layout['props']['children'][0]
        html_body = layout['props']['children'][1]
        assert dcc_radio['props']['id'] == 'toc'
        assert dcc_radio['props']['value'] == '2'
        assert html_body['props']['id'] == 'body'
        assert html_body['props']['children'] == 'Chapter 2'
    with lock:
        dash_duo.find_element('#button').click()
        dash_duo.find_elements('input[type="radio"]')[1].click()
        chapter2_assertions()
        assert call_counts['button-output'].value == 0
    until(lambda : call_counts['button-output'].value == 1, 3)
    dash_duo._wait_for_callbacks()
    chapter2_assertions()
    assert not dash_duo.get_logs()