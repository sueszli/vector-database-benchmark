import pytest
from salt.utils.templates import render_cheetah_tmpl
pytest.importorskip('Cheetah')

def test_render_sanity(render_context):
    if False:
        i = 10
        return i + 15
    tmpl = 'OK'
    res = render_cheetah_tmpl(tmpl, render_context)
    assert res == 'OK'

def test_render_evaluate(render_context):
    if False:
        while True:
            i = 10
    tmpl = '<%="OK"%>'
    res = render_cheetah_tmpl(tmpl, render_context)
    assert res == 'OK'

def test_render_evaluate_xml(render_context):
    if False:
        return 10
    tmpl = '\n    <% if 1: %>\n    OK\n    <% pass %>\n    '
    res = render_cheetah_tmpl(tmpl, render_context)
    stripped = res.strip()
    assert stripped == 'OK'

def test_render_evaluate_text(render_context):
    if False:
        print('Hello World!')
    tmpl = '\n    #if 1\n    OK\n    #end if\n    '
    res = render_cheetah_tmpl(tmpl, render_context)
    stripped = res.strip()
    assert stripped == 'OK'

def test_render_variable(render_context):
    if False:
        i = 10
        return i + 15
    tmpl = '$var'
    render_context['var'] = 'OK'
    res = render_cheetah_tmpl(tmpl, render_context)
    assert res.strip() == 'OK'