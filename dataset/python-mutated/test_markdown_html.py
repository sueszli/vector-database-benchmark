import dash
from dash.dash_table import DataTable

def get_app(markdown_options):
    if False:
        i = 10
        return i + 15
    app = dash.Dash(__name__)
    props = dict(id='table', columns=[dict(name='a', id='a', type='text', presentation='markdown')], data=[dict(a='<h1>html h1 heading</h1>')])
    if markdown_options is not None:
        props['markdown_options'] = markdown_options
    app.layout = DataTable(**props)
    return app

def test_tmdh001_html_not_allowed(test):
    if False:
        while True:
            i = 10
    test.start_server(get_app(None))
    h1_elements = test.find_elements('h1')
    assert len(h1_elements) == 0
    assert test.get_log_errors() == []

def test_tmdh002_html_allowed(test):
    if False:
        for i in range(10):
            print('nop')
    test.start_server(get_app(dict(html=True)))
    h1_elements = test.find_elements('h1')
    assert len(h1_elements) == 1
    assert test.get_log_errors() == []