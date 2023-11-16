import apistar

def test_docs():
    if False:
        i = 10
        return i + 15
    schema = {'openapi': '3.0.0', 'info': {'title': '', 'version': ''}, 'paths': {}}
    index_html = apistar.docs(schema, static_url='/static/')
    assert '<title>API Star</title>' in index_html
    assert 'href="/static/css/base.css"' in index_html

def test_docs_with_static_url_func():
    if False:
        print('Hello World!')
    schema = {'openapi': '3.0.0', 'info': {'title': '', 'version': ''}, 'paths': {}}
    index_html = apistar.docs(schema, static_url=lambda x: '/' + x)
    assert '<title>API Star</title>' in index_html
    assert 'href="/css/base.css"' in index_html