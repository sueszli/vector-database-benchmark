"""Test sphinx.ext.autosectionlabel extension."""
import re
import pytest

@pytest.mark.sphinx('html', testroot='ext-autosectionlabel')
def test_autosectionlabel_html(app, status, warning, skipped_labels=False):
    if False:
        print('Hello World!')
    app.builder.build_all()
    content = (app.outdir / 'index.html').read_text(encoding='utf8')
    html = '<li><p><a class="reference internal" href="#introduce-of-sphinx"><span class=".*?">Introduce of Sphinx</span></a></p></li>'
    assert re.search(html, content, re.S)
    html = '<li><p><a class="reference internal" href="#installation"><span class="std std-ref">Installation</span></a></p></li>'
    assert re.search(html, content, re.S)
    html = '<li><p><a class="reference internal" href="#for-windows-users"><span class="std std-ref">For Windows users</span></a></p></li>'
    assert re.search(html, content, re.S)
    html = '<li><p><a class="reference internal" href="#for-unix-users"><span class="std std-ref">For UNIX users</span></a></p></li>'
    assert re.search(html, content, re.S)
    html = '<li><p><a class="reference internal" href="#linux"><span class="std std-ref">Linux</span></a></p></li>'
    assert re.search(html, content, re.S)
    html = '<li><p><a class="reference internal" href="#freebsd"><span class="std std-ref">FreeBSD</span></a></p></li>'
    assert re.search(html, content, re.S)
    html = '<li><p><a class="reference internal" href="#this-one-s-got-an-apostrophe"><span class="std std-ref">This one’s got an apostrophe</span></a></p></li>'
    assert re.search(html, content, re.S)

@pytest.mark.sphinx('html', testroot='ext-autosectionlabel-prefix-document')
def test_autosectionlabel_prefix_document_html(app, status, warning):
    if False:
        print('Hello World!')
    test_autosectionlabel_html(app, status, warning)

@pytest.mark.sphinx('html', testroot='ext-autosectionlabel', confoverrides={'autosectionlabel_maxdepth': 3})
def test_autosectionlabel_maxdepth(app, status, warning):
    if False:
        for i in range(10):
            print('nop')
    app.builder.build_all()
    content = (app.outdir / 'index.html').read_text(encoding='utf8')
    html = '<li><p><a class="reference internal" href="#test-ext-autosectionlabel"><span class=".*?">test-ext-autosectionlabel</span></a></p></li>'
    assert re.search(html, content, re.S)
    html = '<li><p><a class="reference internal" href="#installation"><span class="std std-ref">Installation</span></a></p></li>'
    assert re.search(html, content, re.S)
    html = '<li><p><a class="reference internal" href="#for-windows-users"><span class="std std-ref">For Windows users</span></a></p></li>'
    assert re.search(html, content, re.S)
    html = '<li><p><span class="xref std std-ref">Linux</span></p></li>'
    assert re.search(html, content, re.S)
    assert "WARNING: undefined label: 'linux'" in warning.getvalue()