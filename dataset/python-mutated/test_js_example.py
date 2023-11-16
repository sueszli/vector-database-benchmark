import pytest
from flask import template_rendered

@pytest.mark.parametrize(('path', 'template_name'), (('/', 'xhr.html'), ('/plain', 'xhr.html'), ('/fetch', 'fetch.html'), ('/jquery', 'jquery.html')))
def test_index(app, client, path, template_name):
    if False:
        while True:
            i = 10

    def check(sender, template, context):
        if False:
            i = 10
            return i + 15
        assert template.name == template_name
    with template_rendered.connected_to(check, app):
        client.get(path)

@pytest.mark.parametrize(('a', 'b', 'result'), ((2, 3, 5), (2.5, 3, 5.5), (2, None, 2), (2, 'b', 2)))
def test_add(client, a, b, result):
    if False:
        for i in range(10):
            print('nop')
    response = client.post('/add', data={'a': a, 'b': b})
    assert response.get_json()['result'] == result