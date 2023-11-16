import dash
from dash import html, dcc, Output, Input
from dash.exceptions import PreventUpdate
app = dash.Dash(__name__)
app.layout = html.Div([dcc.Input(id='value', placeholder='my-value'), html.Div(['You entered: ', html.Span(id='out')]), html.Button('style-btn', id='style-btn'), html.Div('style-container', id='style-output')])

@app.callback(Output('out', 'children'), Input('value', 'value'))
def on_value(value):
    if False:
        for i in range(10):
            print('nop')
    if value is None:
        raise PreventUpdate
    return value

@app.callback(Output('style-output', 'style'), [Input('style-btn', 'n_clicks')])
def on_style(value):
    if False:
        i = 10
        return i + 15
    if value is None:
        raise PreventUpdate
    return {'padding': '10px'}
if __name__ == '__main__':
    app.run_server(debug=True, port=10850)