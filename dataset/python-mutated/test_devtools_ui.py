from time import sleep
import flask
from dash import Dash, Input, Output, dcc, html
import dash.testing.wait as wait
from dash_test_components import WidthComponent
from tests.assets.todo_app import todo_app

def test_dvui001_disable_props_check_config(dash_duo):
    if False:
        return 10
    app = Dash(__name__)
    app.layout = html.Div([html.P(id='tcid', children='Hello Props Check'), dcc.Graph(id='broken', animate=3)])
    dash_duo.start_server(app, debug=True, use_reloader=False, use_debugger=True, dev_tools_hot_reload=False, dev_tools_props_check=False)
    dash_duo.wait_for_text_to_equal('#tcid', 'Hello Props Check')
    assert dash_duo.find_elements('#broken svg.main-svg'), 'graph should be rendered'
    dash_duo.find_element('.dash-debug-menu').click()
    sleep(1)
    dash_duo.percy_snapshot('devtools - disable props check - Graph should render')

def test_dvui002_disable_ui_config(dash_duo):
    if False:
        while True:
            i = 10
    app = Dash(__name__)
    app.layout = html.Div([html.P(id='tcid', children='Hello Disable UI'), dcc.Graph(id='broken', animate=3)])
    dash_duo.start_server(app, debug=True, use_reloader=False, use_debugger=True, dev_tools_hot_reload=False, dev_tools_ui=False)
    dash_duo.wait_for_text_to_equal('#tcid', 'Hello Disable UI')
    logs = str(wait.until(dash_duo.get_logs, timeout=1))
    assert 'Invalid argument `animate` passed into Graph' in logs, 'the error should present in the console without DEV tools UI'
    assert not dash_duo.find_elements('.dash-debug-menu'), 'the debug menu icon should NOT show up'

def test_dvui003_callback_graph(dash_duo):
    if False:
        print('Hello World!')
    app = todo_app()
    dash_duo.start_server(app, debug=True, use_reloader=False, use_debugger=True, dev_tools_hot_reload=False)
    dash_duo.wait_for_text_to_equal('#totals', '0 of 0 items completed')
    dash_duo.driver.execute_script('\n        const cbProfiles = window.store.getState().profile.callbacks;\n        Object.keys(cbProfiles).forEach(k => {\n            cbProfiles[k].compute = 44;\n            cbProfiles[k].network.time = 33;\n            cbProfiles[k].total = 77;\n        });\n        ')
    dash_duo.find_element('.dash-debug-menu').click()
    sleep(1)
    dash_duo.find_element('.dash-debug-menu__button--callbacks').click()
    sleep(3)
    dash_duo.find_element('canvas[data-id="layer2-node"]')
    dash_duo.percy_snapshot('devtools - callback graph', convert_canvases=True)
    pos = dash_duo.driver.execute_script("\n        const pos = store.getState().profile.graphLayout.positions['new-item.Xvalue'];\n        pos.y -= 100;\n        return pos.y;\n        ")
    dash_duo.find_element('.dash-debug-menu__button--callbacks').click()
    dash_duo.find_element('#add').click()
    dash_duo.find_element('.dash-debug-menu__button--callbacks').click()
    dash_duo.wait_for_text_to_equal('#totals', '0 of 1 items completed - 0%')
    sleep(2)
    assert pos == dash_duo.driver.execute_script("\n        const pos = store.getState().profile.graphLayout.positions['new-item.Xvalue'];\n        return pos.y;\n        ")

def test_dvui004_width_props(dash_duo):
    if False:
        return 10
    app = Dash(__name__)
    app.layout = html.Div([html.Button(['Click me!'], id='btn'), WidthComponent(id='width')])

    @app.callback(Output('width', 'width'), Input('btn', 'n_clicks'))
    def get_width(n_clicks):
        if False:
            for i in range(10):
                print('nop')
        n_clicks = n_clicks if n_clicks is not None else 0
        return (n_clicks + 1) * 10
    dash_duo.start_server(app, debug=True, use_reloader=False, use_debugger=True, dev_tools_hot_reload=False)
    dash_duo.find_element('.dash-debug-menu').click()
    sleep(1)
    dash_duo.find_element('.dash-debug-menu__button--callbacks').click()
    sleep(3)
    assert dash_duo.get_logs() == []

def test_dvui005_undo_redo(dash_duo):
    if False:
        while True:
            i = 10

    def click_undo():
        if False:
            return 10
        undo_selector = '._dash-undo-redo span:first-child div:last-child'
        dash_duo.wait_for_text_to_equal(undo_selector, 'undo')
        dash_duo.find_element(undo_selector).click()

    def click_redo():
        if False:
            return 10
        redo_selector = '._dash-undo-redo span:last-child div:last-child'
        dash_duo.wait_for_text_to_equal(redo_selector, 'redo')
        dash_duo.find_element(redo_selector).click()

    def check_undo_redo_exist(has_undo, has_redo):
        if False:
            while True:
                i = 10
        selector = '._dash-undo-redo span div:last-child'
        els = dash_duo.find_elements(selector)
        texts = (['undo'] if has_undo else []) + (['redo'] if has_redo else [])
        assert len(els) == len(texts)
        for (el, text) in zip(els, texts):
            assert el.text == text
    app = Dash(__name__, show_undo_redo=True)
    app.layout = html.Div([dcc.Input(id='a'), html.Div(id='b')])

    @app.callback(Output('b', 'children'), Input('a', 'value'))
    def set_b(a):
        if False:
            for i in range(10):
                print('nop')
        return a
    dash_duo.start_server(app)
    dash_duo.find_element('#a').send_keys('xyz')
    dash_duo.wait_for_text_to_equal('#b', 'xyz')
    check_undo_redo_exist(True, False)
    click_undo()
    dash_duo.wait_for_text_to_equal('#b', 'xy')
    check_undo_redo_exist(True, True)
    click_undo()
    dash_duo.wait_for_text_to_equal('#b', 'x')
    check_undo_redo_exist(True, True)
    click_redo()
    dash_duo.wait_for_text_to_equal('#b', 'xy')
    check_undo_redo_exist(True, True)
    dash_duo.percy_snapshot(name='undo-redo')
    click_undo()
    click_undo()
    dash_duo.wait_for_text_to_equal('#b', '')
    check_undo_redo_exist(False, True)

def test_dvui006_no_undo_redo(dash_duo):
    if False:
        for i in range(10):
            print('nop')
    app = Dash(__name__)
    app.layout = html.Div([dcc.Input(id='a'), html.Div(id='b')])

    @app.callback(Output('b', 'children'), Input('a', 'value'))
    def set_b(a):
        if False:
            return 10
        return a
    dash_duo.start_server(app)
    dash_duo.find_element('#a').send_keys('xyz')
    dash_duo.wait_for_text_to_equal('#b', 'xyz')
    dash_duo.wait_for_no_elements('._dash-undo-redo')

def test_dvui007_other_before_request_func(dash_thread_server, dash_br):
    if False:
        while True:
            i = 10
    driver = dash_br.driver
    app = Dash(__name__)
    app.layout = html.Div([html.P(id='just_an_id', children='You should never see this')])

    @app.server.before_request
    def create_an_alternative_response():
        if False:
            print('Hello World!')
        if flask.request.endpoint == '/':
            return flask.Response('<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">\n<title>Alternative repsonse</title>\n<h1 id="alternative_id">Alternative response header</h1>\n', 200, mimetype='text/html')
    dash_thread_server.start(app, debug=True, use_reloader=False, use_debugger=True, dev_tools_hot_reload=False)
    driver.get(dash_thread_server.url)
    dash_br.find_element('#alternative_id')