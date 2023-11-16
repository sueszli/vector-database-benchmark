"""Test sphinx.ext.todo extension."""
import re
import pytest

@pytest.mark.sphinx('html', testroot='ext-todo', freshenv=True, confoverrides={'todo_include_todos': True, 'todo_emit_warnings': True})
def test_todo(app, status, warning):
    if False:
        while True:
            i = 10
    todos = []

    def on_todo_defined(app, node):
        if False:
            return 10
        todos.append(node)
    app.connect('todo-defined', on_todo_defined)
    app.builder.build_all()
    content = (app.outdir / 'index.html').read_text(encoding='utf8')
    assert '<p class="admonition-title">Todo</p>\n<p>todo in foo</p>' in content
    assert '<p class="admonition-title">Todo</p>\n<p>todo in bar</p>' in content
    content = (app.outdir / 'foo.html').read_text(encoding='utf8')
    assert '<p class="admonition-title">Todo</p>\n<p>todo in foo</p>' in content
    assert '<p class="admonition-title">Todo</p>\n<p>todo in param field</p>' in content
    assert 'WARNING: TODO entry found: todo in foo' in warning.getvalue()
    assert 'WARNING: TODO entry found: todo in bar' in warning.getvalue()
    assert len(todos) == 3
    assert {todo[1].astext() for todo in todos} == {'todo in foo', 'todo in bar', 'todo in param field'}

@pytest.mark.sphinx('html', testroot='ext-todo', freshenv=True, confoverrides={'todo_include_todos': False, 'todo_emit_warnings': True})
def test_todo_not_included(app, status, warning):
    if False:
        i = 10
        return i + 15
    todos = []

    def on_todo_defined(app, node):
        if False:
            while True:
                i = 10
        todos.append(node)
    app.connect('todo-defined', on_todo_defined)
    app.builder.build_all()
    content = (app.outdir / 'index.html').read_text(encoding='utf8')
    assert '<p class="admonition-title">Todo</p>\n<p>todo in foo</p>' not in content
    assert '<p class="admonition-title">Todo</p>\n<p>todo in bar</p>' not in content
    content = (app.outdir / 'foo.html').read_text(encoding='utf8')
    assert '<p class="admonition-title">Todo</p>\n<p>todo in foo</p>' not in content
    assert 'WARNING: TODO entry found: todo in foo' in warning.getvalue()
    assert 'WARNING: TODO entry found: todo in bar' in warning.getvalue()
    assert len(todos) == 3
    assert {todo[1].astext() for todo in todos} == {'todo in foo', 'todo in bar', 'todo in param field'}

@pytest.mark.sphinx('latex', testroot='ext-todo', freshenv=True, confoverrides={'todo_include_todos': True})
def test_todo_valid_link(app, status, warning):
    if False:
        print('Hello World!')
    '\n    Test that the inserted "original entry" links for todo items have a target\n    that exists in the LaTeX output. The target was previously incorrectly\n    omitted (GitHub issue #1020).\n    '
    app.builder.build_all()
    content = (app.outdir / 'python.tex').read_text(encoding='utf8')
    link = '{\\\\hyperref\\[\\\\detokenize{(.*?foo.*?)}]{\\\\sphinxcrossref{\\\\sphinxstyleemphasis{original entry}}}}'
    m = re.findall(link, content)
    assert len(m) == 4
    target = m[0]
    labels = re.findall('\\\\label{\\\\detokenize{([^}]*)}}', content)
    matched = [l for l in labels if l == target]
    assert len(matched) == 1