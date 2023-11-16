import dash
import dash_daq as daq
import dash_renderjson
from dash import html, Input, Output
from lightning.app import LightningWork, LightningFlow, LightningApp
from lightning.app.utilities.state import AppState

class LitDash(LightningWork):

    def run(self):
        if False:
            print('Hello World!')
        dash_app = dash.Dash(__name__)
        dash_app.layout = html.Div([daq.ToggleSwitch(id='my-toggle-switch', value=False), html.Div(id='output')])

        @dash_app.callback(Output('output', 'children'), [Input('my-toggle-switch', 'value')])
        def display_output(value):
            if False:
                for i in range(10):
                    print('nop')
            if value:
                state = AppState()
                state._request_state()
                return dash_renderjson.DashRenderjson(id='input', data=state._state, max_depth=-1, invert_theme=True)
        dash_app.run_server(host=self.host, port=self.port)

class LitApp(LightningFlow):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.lit_dash = LitDash(parallel=True)

    def run(self):
        if False:
            print('Hello World!')
        self.lit_dash.run()

    def configure_layout(self):
        if False:
            for i in range(10):
                print('nop')
        tab1 = {'name': 'home', 'content': self.lit_dash}
        return tab1
app = LightningApp(LitApp())