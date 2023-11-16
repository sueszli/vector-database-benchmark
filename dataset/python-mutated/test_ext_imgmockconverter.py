"""Test image converter with identical basenames"""
import pytest

@pytest.mark.sphinx('latex', testroot='ext-imgmockconverter')
def test_ext_imgmockconverter(app, status, warning):
    if False:
        print('Hello World!')
    app.builder.build_all()
    content = (app.outdir / 'python.tex').read_text(encoding='utf8')
    assert '\\sphinxincludegraphics{{svgimg}.pdf}' in content
    assert '\\sphinxincludegraphics{{svgimg1}.pdf}' in content
    assert not (app.outdir / 'svgimg.svg').exists()
    assert (app.outdir / 'svgimg.pdf').exists()
    assert (app.outdir / 'svgimg1.pdf').exists()