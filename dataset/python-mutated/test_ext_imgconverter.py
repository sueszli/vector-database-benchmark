"""Test sphinx.ext.imgconverter extension."""
import subprocess
import pytest

@pytest.fixture()
def _if_converter_found(app):
    if False:
        print('Hello World!')
    image_converter = getattr(app.config, 'image_converter', '')
    try:
        if image_converter:
            subprocess.run([image_converter, '-version'], capture_output=True)
            return
    except OSError:
        pass
    pytest.skip('image_converter "%s" is not available' % image_converter)

@pytest.mark.usefixtures('_if_converter_found')
@pytest.mark.sphinx('latex', testroot='ext-imgconverter')
def test_ext_imgconverter(app, status, warning):
    if False:
        while True:
            i = 10
    app.builder.build_all()
    content = (app.outdir / 'python.tex').read_text(encoding='utf8')
    assert '\\sphinxincludegraphics{{img}.pdf}' in content
    assert '\\sphinxincludegraphics{{svgimg}.png}' in content
    assert not (app.outdir / 'svgimg.svg').exists()
    assert (app.outdir / 'svgimg.png').exists()