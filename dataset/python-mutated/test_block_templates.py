import pytest
from grc.core.blocks._templates import MakoTemplates
from grc.core.errors import TemplateError

class Block(object):
    namespace_templates = {}
    templates = MakoTemplates(None)

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        self.namespace_templates.update(kwargs)

def test_simple():
    if False:
        for i in range(10):
            print('nop')
    t = MakoTemplates(_bind_to=Block(num='123'), test='abc${num}')
    assert t['test'] == 'abc${num}'
    assert t.render('test') == 'abc123'
    assert 'abc${num}' in t._template_cache

def test_instance():
    if False:
        while True:
            i = 10
    block = Block(num='123')
    block.templates['test'] = 'abc${num}'
    assert block.templates.render('test') == 'abc123'
    assert block.templates is block.__dict__['templates']

def test_list():
    if False:
        while True:
            i = 10
    templates = ['abc${num}', '${2 * num}c']
    t = MakoTemplates(_bind_to=Block(num='123'), test=templates)
    assert t['test'] == templates
    assert t.render('test') == ['abc123', '123123c']
    assert set(templates) == set(t._template_cache.keys())

def test_parse_error():
    if False:
        while True:
            i = 10
    with pytest.raises(TemplateError):
        MakoTemplates(_bind_to=Block(num='123'), test='abc${num NOT CLOSING').render('test')

def test_parse_error2():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(TemplateError):
        MakoTemplates(_bind_to=Block(num='123'), test='abc${ WRONG_VAR }').render('test')