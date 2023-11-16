import pytest
from salt.utils.templates import render_wempy_tmpl
pytest.importorskip('wemplate')

def test_render_wempy_sanity(render_context):
    if False:
        for i in range(10):
            print('nop')
    tmpl = 'OK'
    res = render_wempy_tmpl(tmpl, render_context)
    assert res == 'OK'

def test_render_wempy_evaluate(render_context):
    if False:
        while True:
            i = 10
    tmpl = '{{="OK"}}'
    res = render_wempy_tmpl(tmpl, render_context)
    assert res == 'OK'

def test_render_wempy_evaluate_multi(render_context):
    if False:
        print('Hello World!')
    tmpl = '{{if 1:}}OK{{pass}}'
    res = render_wempy_tmpl(tmpl, render_context)
    assert res == 'OK'

def test_render_wempy_variable(render_context):
    if False:
        while True:
            i = 10
    tmpl = '{{=var}}'
    render_context['var'] = 'OK'
    res = render_wempy_tmpl(tmpl, render_context)
    assert res == 'OK'