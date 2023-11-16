import pytest
from salt.utils.templates import render_genshi_tmpl
pytest.importorskip('genshi')

def test_render_sanity(render_context):
    if False:
        i = 10
        return i + 15
    tmpl = '<RU>OK</RU>'
    res = render_genshi_tmpl(tmpl, render_context)
    assert res == '<RU>OK</RU>'

def test_render_evaluate(render_context):
    if False:
        print('Hello World!')
    tmpl = '<RU>${ "OK" }</RU>'
    res = render_genshi_tmpl(tmpl, render_context)
    assert res == '<RU>OK</RU>'

def test_render_evaluate_condition(render_context):
    if False:
        while True:
            i = 10
    tmpl = '<RU xmlns:py="http://genshi.edgewall.org/" py:if="1">OK</RU>'
    res = render_genshi_tmpl(tmpl, render_context)
    assert res == '<RU>OK</RU>'

def test_render_variable(render_context):
    if False:
        print('Hello World!')
    tmpl = '<RU>$var</RU>'
    render_context['var'] = 'OK'
    res = render_genshi_tmpl(tmpl, render_context)
    assert res == '<RU>OK</RU>'

def test_render_variable_replace(render_context):
    if False:
        print('Hello World!')
    tmpl = '<RU xmlns:py="http://genshi.edgewall.org/" py:content="var">not ok</RU>'
    render_context['var'] = 'OK'
    res = render_genshi_tmpl(tmpl, render_context)
    assert res == '<RU>OK</RU>'