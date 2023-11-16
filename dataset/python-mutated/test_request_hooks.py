import json
import functools
import flask
import pytest
from dash import Dash, Output, Input, html, dcc
from dash.types import RendererHooks
from werkzeug.exceptions import HTTPException

def test_rdrh001_request_hooks(dash_duo):
    if False:
        while True:
            i = 10
    app = Dash(__name__)
    app.index_string = '<!DOCTYPE html>\n    <html>\n        <head>\n            {%metas%}\n            <title>{%title%}</title>\n            {%favicon%}\n            {%css%}\n        </head>\n        <body>\n            <div id="top">Testing custom DashRenderer</div>\n            {%app_entry%}\n            <footer>\n                {%config%}\n                {%scripts%}\n                <script id="_dash-renderer" type"application/json">\n                    const renderer = new DashRenderer({\n                        request_pre: (payload) => {\n                            var output = document.getElementById(\'output-pre\')\n                            var outputPayload = document.getElementById(\'output-pre-payload\')\n                            if(output) {\n                                output.innerHTML = \'request_pre changed this text!\';\n                            }\n                            if(outputPayload) {\n                                outputPayload.innerHTML = JSON.stringify(payload);\n                            }\n                        },\n                        request_post: (payload, response) => {\n                            var output = document.getElementById(\'output-post\')\n                            var outputPayload = document.getElementById(\'output-post-payload\')\n                            var outputResponse = document.getElementById(\'output-post-response\')\n                            if(output) {\n                                output.innerHTML = \'request_post changed this text!\';\n                            }\n                            if(outputPayload) {\n                                outputPayload.innerHTML = JSON.stringify(payload);\n                            }\n                            if(outputResponse) {\n                                outputResponse.innerHTML = JSON.stringify(response);\n                            }\n                        }\n                    })\n                </script>\n            </footer>\n            <div id="bottom">With request hooks</div>\n        </body>\n    </html>'
    app.layout = html.Div([dcc.Input(id='input', value='initial value'), html.Div(html.Div([html.Div(id='output-1'), html.Div(id='output-pre'), html.Div(id='output-pre-payload'), html.Div(id='output-post'), html.Div(id='output-post-payload'), html.Div(id='output-post-response')]))])

    @app.callback(Output('output-1', 'children'), [Input('input', 'value')])
    def update_output(value):
        if False:
            for i in range(10):
                print('nop')
        return value
    dash_duo.start_server(app)
    _in = dash_duo.find_element('#input')
    dash_duo.clear_input(_in)
    _in.send_keys('fire request hooks')
    dash_duo.wait_for_text_to_equal('#output-1', 'fire request hooks')
    dash_duo.wait_for_text_to_equal('#output-pre', 'request_pre changed this text!')
    dash_duo.wait_for_text_to_equal('#output-post', 'request_post changed this text!')
    assert json.loads(dash_duo.find_element('#output-pre-payload').text) == {'output': 'output-1.children', 'outputs': {'id': 'output-1', 'property': 'children'}, 'changedPropIds': ['input.value'], 'inputs': [{'id': 'input', 'property': 'value', 'value': 'fire request hooks'}]}
    assert json.loads(dash_duo.find_element('#output-post-payload').text) == {'output': 'output-1.children', 'outputs': {'id': 'output-1', 'property': 'children'}, 'changedPropIds': ['input.value'], 'inputs': [{'id': 'input', 'property': 'value', 'value': 'fire request hooks'}]}
    assert json.loads(dash_duo.find_element('#output-post-response').text) == {'output-1': {'children': 'fire request hooks'}}
    assert dash_duo.find_element('#top').text == 'Testing custom DashRenderer'
    assert dash_duo.find_element('#bottom').text == 'With request hooks'
    assert dash_duo.get_logs() == []

def test_rdrh002_with_custom_renderer_interpolated(dash_duo):
    if False:
        for i in range(10):
            print('nop')
    renderer = '\n        <script id="_dash-renderer" type="application/javascript">\n            console.log(\'firing up a custom renderer!\')\n            const renderer = new DashRenderer({\n                request_pre: () => {\n                    var output = document.getElementById(\'output-pre\')\n                    if(output) {\n                        output.innerHTML = \'request_pre was here!\';\n                    }\n                },\n                request_post: () => {\n                    var output = document.getElementById(\'output-post\')\n                    if(output) {\n                        output.innerHTML = \'request_post!!!\';\n                    }\n                }\n            })\n        </script>\n    '

    class CustomDash(Dash):

        def interpolate_index(self, **kwargs):
            if False:
                while True:
                    i = 10
            return '<!DOCTYPE html>\n            <html>\n                <head>\n                    <title>My App</title>\n                </head>\n                <body>\n\n                    <div id="custom-header">My custom header</div>\n                    {app_entry}\n                    {config}\n                    {scripts}\n                    {renderer}\n                    <div id="custom-footer">My custom footer</div>\n                </body>\n            </html>'.format(app_entry=kwargs['app_entry'], config=kwargs['config'], scripts=kwargs['scripts'], renderer=renderer)
    app = CustomDash()
    app.layout = html.Div([dcc.Input(id='input', value='initial value'), html.Div(html.Div([html.Div(id='output-1'), html.Div(id='output-pre'), html.Div(id='output-post')]))])

    @app.callback(Output('output-1', 'children'), [Input('input', 'value')])
    def update_output(value):
        if False:
            for i in range(10):
                print('nop')
        return value
    dash_duo.start_server(app)
    input1 = dash_duo.find_element('#input')
    dash_duo.clear_input(input1)
    input1.send_keys('fire request hooks')
    dash_duo.wait_for_text_to_equal('#output-1', 'fire request hooks')
    assert dash_duo.find_element('#output-pre').text == 'request_pre was here!'
    assert dash_duo.find_element('#output-post').text == 'request_post!!!'
    assert dash_duo.find_element('#custom-header').text == 'My custom header'
    assert dash_duo.find_element('#custom-footer').text == 'My custom footer'
    assert dash_duo.get_logs() == []

@pytest.mark.parametrize('expiry_code', [401, 400])
def test_rdrh003_refresh_jwt(expiry_code, dash_duo):
    if False:
        while True:
            i = 10
    app = Dash(__name__)
    app.index_string = '<!DOCTYPE html>\n    <html>\n        <head>\n            {%metas%}\n            <title>{%title%}</title>\n            {%favicon%}\n            {%css%}\n        </head>\n        <body>\n            <div>Testing custom DashRenderer</div>\n            {%app_entry%}\n            <footer>\n                {%config%}\n                {%scripts%}\n                <script id="_dash-renderer" type"application/json">\n                    const renderer = new DashRenderer({\n                        request_refresh_jwt: (old_token) => {\n                            console.log("refreshing token", old_token);\n                            var new_token = "." + (old_token || "");\n                            var output = document.getElementById(\'output-token\')\n                            if(output) {\n                                output.innerHTML = new_token;\n                            }\n                            return new_token;\n                        }\n                    })\n                </script>\n            </footer>\n            <div>With request hooks</div>\n        </body>\n    </html>'
    app.layout = html.Div([dcc.Input(id='input', value='initial value'), html.Div(html.Div([html.Div(id='output-1'), html.Div(id='output-token')]))])

    @app.callback(Output('output-1', 'children'), [Input('input', 'value')])
    def update_output(value):
        if False:
            for i in range(10):
                print('nop')
        return value
    required_jwt_len = 0

    def protect_route(func):
        if False:
            i = 10
            return i + 15

        @functools.wraps(func)
        def wrap(*args, **kwargs):
            if False:
                while True:
                    i = 10
            try:
                if flask.request.method == 'OPTIONS':
                    return func(*args, **kwargs)
                token = flask.request.headers.environ.get('HTTP_AUTHORIZATION')
                if required_jwt_len and (not token or len(token) != required_jwt_len + len('Bearer ')):
                    flask.request.get_json(silent=True)
                    flask.abort(expiry_code, description='JWT Expired ' + str(token))
            except HTTPException as e:
                return e
            return func(*args, **kwargs)
        return wrap
    for (name, method) in ((x, app.server.view_functions[x]) for x in app.routes if x in app.server.view_functions):
        app.server.view_functions[name] = protect_route(method)
    dash_duo.start_server(app)
    _in = dash_duo.find_element('#input')
    dash_duo.clear_input(_in)
    required_jwt_len = 1
    _in.send_keys('fired request')
    dash_duo.wait_for_text_to_equal('#output-1', 'fired request')
    dash_duo.wait_for_text_to_equal('#output-token', '.')
    required_jwt_len = 2
    dash_duo.clear_input(_in)
    _in.send_keys('fired request again')
    dash_duo.wait_for_text_to_equal('#output-1', 'fired request again')
    dash_duo.wait_for_text_to_equal('#output-token', '..')
    assert len(dash_duo.get_logs()) == 2

def test_rdrh004_layout_hooks(dash_duo):
    if False:
        i = 10
        return i + 15
    hooks: RendererHooks = {'layout_pre': "\n            () => {\n                var layoutPre = document.createElement('div');\n                layoutPre.setAttribute('id', 'layout-pre');\n                layoutPre.innerHTML = 'layout_pre generated this text';\n                document.body.appendChild(layoutPre);\n            }\n        ", 'layout_post': '\n            (response) => {\n                response.props.children = "layout_post generated this text";\n            }\n        '}
    app = Dash(__name__, hooks=hooks)
    app.layout = html.Div(id='layout')
    dash_duo.start_server(app)
    dash_duo.wait_for_text_to_equal('#layout-pre', 'layout_pre generated this text')
    dash_duo.wait_for_text_to_equal('#layout', 'layout_post generated this text')
    assert dash_duo.get_logs() == []