from dash.resources import Resources

class obj(object):

    def __init__(self, dict):
        if False:
            print('Hello World!')
        self.__dict__ = dict

def test_resources_eager():
    if False:
        for i in range(10):
            print('nop')
    resource = Resources('js_test')
    resource.config = obj({'eager_loading': True, 'serve_locally': False})
    filtered = resource._filter_resources([{'async': 'eager', 'external_url': 'a.js'}, {'async': 'lazy', 'external_url': 'b.js'}, {'async': True, 'external_url': 'c.js'}], False)
    assert len(filtered) == 3
    assert filtered[0].get('external_url') == 'a.js'
    assert filtered[0].get('dynamic') is False
    assert filtered[1].get('external_url') == 'b.js'
    assert filtered[1].get('dynamic') is True
    assert filtered[2].get('external_url') == 'c.js'
    assert filtered[2].get('dynamic') is False

def test_resources_lazy():
    if False:
        for i in range(10):
            print('nop')
    resource = Resources('js_test')
    resource.config = obj({'eager_loading': False, 'serve_locally': False})
    filtered = resource._filter_resources([{'async': 'eager', 'external_url': 'a.js'}, {'async': 'lazy', 'external_url': 'b.js'}, {'async': True, 'external_url': 'c.js'}], False)
    assert len(filtered) == 3
    assert filtered[0].get('external_url') == 'a.js'
    assert filtered[0].get('dynamic') is True
    assert filtered[1].get('external_url') == 'b.js'
    assert filtered[1].get('dynamic') is True
    assert filtered[2].get('external_url') == 'c.js'
    assert filtered[2].get('dynamic') is True