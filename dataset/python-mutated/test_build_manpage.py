"""Test the build process with manpage builder with the test root."""
import docutils
import pytest
from sphinx.builders.manpage import default_man_pages
from sphinx.config import Config

@pytest.mark.sphinx('man')
def test_all(app, status, warning):
    if False:
        while True:
            i = 10
    app.builder.build_all()
    assert (app.outdir / 'sphinxtests.1').exists()
    content = (app.outdir / 'sphinxtests.1').read_text(encoding='utf8')
    assert '\\fBprint \\fP\\fIi\\fP\\fB\\en\\fP' in content
    assert '\\fBmanpage\\en\\fP' in content
    assert 'sphinxtests \\- Sphinx <Tests> 0.6alpha1' in content
    assert '\n.B term1\n' in content
    assert '\nterm2 (\\fBstronged partially\\fP)\n' in content
    assert '\n\\fIvariable_only\\fP\n' in content
    assert '\n\\fIvariable\\fP\\fB and text\\fP\n' in content
    assert '\n\\fBShow \\fP\\fIvariable\\fP\\fB in the middle\\fP\n' in content
    assert 'Footnotes' not in content

@pytest.mark.sphinx('man', testroot='basic', confoverrides={'man_pages': [('index', 'title', None, [], 1)]})
def test_man_pages_empty_description(app, status, warning):
    if False:
        return 10
    app.builder.build_all()
    content = (app.outdir / 'title.1').read_text(encoding='utf8')
    assert 'title \\-' not in content

@pytest.mark.sphinx('man', testroot='basic', confoverrides={'man_make_section_directory': True})
def test_man_make_section_directory(app, status, warning):
    if False:
        while True:
            i = 10
    app.build()
    assert (app.outdir / 'man1' / 'python.1').exists()

@pytest.mark.sphinx('man', testroot='directive-code')
def test_captioned_code_block(app, status, warning):
    if False:
        i = 10
        return i + 15
    app.builder.build_all()
    content = (app.outdir / 'python.1').read_text(encoding='utf8')
    if docutils.__version_info__[:2] < (0, 21):
        expected = '.sp\ncaption \\fItest\\fP rb\n.INDENT 0.0\n.INDENT 3.5\n.sp\n.nf\n.ft C\ndef ruby?\n    false\nend\n.ft P\n.fi\n.UNINDENT\n.UNINDENT\n'
    else:
        expected = '.sp\ncaption \\fItest\\fP rb\n.INDENT 0.0\n.INDENT 3.5\n.sp\n.EX\ndef ruby?\n    false\nend\n.EE\n.UNINDENT\n.UNINDENT\n'
    assert expected in content

def test_default_man_pages():
    if False:
        for i in range(10):
            print('nop')
    config = Config({'project': 'STASI™ Documentation', 'author': "Wolfgang Schäuble & G'Beckstein", 'release': '1.0'})
    config.init_values()
    expected = [('index', 'stasi', 'STASI™ Documentation 1.0', ["Wolfgang Schäuble & G'Beckstein"], 1)]
    assert default_man_pages(config) == expected

@pytest.mark.sphinx('man', testroot='markup-rubric')
def test_rubric(app, status, warning):
    if False:
        print('Hello World!')
    app.build()
    content = (app.outdir / 'python.1').read_text(encoding='utf8')
    assert 'This is a rubric\n' in content