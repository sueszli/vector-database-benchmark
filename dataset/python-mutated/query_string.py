import dash
from dash import html
dash.register_page(__name__, id='query_string')

def layout(velocity=None, **other_unknown_query_strings):
    if False:
        print('Hello World!')
    return html.Div([html.Div('text for query_string', id='text_query_string'), dash.dcc.Input(id='velocity', value=velocity)])