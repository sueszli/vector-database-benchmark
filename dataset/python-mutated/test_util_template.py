"""Tests sphinx.util.template functions."""
from sphinx.util.template import ReSTRenderer

def test_ReSTRenderer_escape():
    if False:
        while True:
            i = 10
    r = ReSTRenderer()
    template = '{{ "*hello*" | e }}'
    assert r.render_string(template, {}) == '\\*hello\\*'

def test_ReSTRenderer_heading():
    if False:
        return 10
    r = ReSTRenderer()
    template = '{{ "hello" | heading }}'
    assert r.render_string(template, {}) == 'hello\n====='
    template = '{{ "hello" | heading(1) }}'
    assert r.render_string(template, {}) == 'hello\n====='
    template = '{{ "русский язык" | heading(2) }}'
    assert r.render_string(template, {}) == 'русский язык\n------------'
    r.env.language = 'ja'
    template = '{{ "русский язык" | heading }}'
    assert r.render_string(template, {}) == 'русский язык\n======================='