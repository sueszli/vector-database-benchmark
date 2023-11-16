"""Tests for IPython.utils.path.py"""
from contextlib import contextmanager
from unittest.mock import patch
import pytest
from IPython.lib import latextools
from IPython.testing.decorators import onlyif_cmds_exist, skipif_not_matplotlib
from IPython.utils.process import FindCmdError

@pytest.mark.parametrize('command', ['latex', 'dvipng'])
def test_check_latex_to_png_dvipng_fails_when_no_cmd(command):
    if False:
        while True:
            i = 10

    def mock_find_cmd(arg):
        if False:
            return 10
        if arg == command:
            raise FindCmdError
    with patch.object(latextools, 'find_cmd', mock_find_cmd):
        assert latextools.latex_to_png_dvipng('whatever', True) is None

@contextmanager
def no_op(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    yield

@onlyif_cmds_exist('latex', 'dvipng')
@pytest.mark.parametrize('s, wrap', [('$$x^2$$', False), ('x^2', True)])
def test_latex_to_png_dvipng_runs(s, wrap):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that latex_to_png_dvipng just runs without error.\n    '

    def mock_kpsewhich(filename):
        if False:
            i = 10
            return i + 15
        assert filename == 'breqn.sty'
        return None
    latextools.latex_to_png_dvipng(s, wrap)
    with patch_latextool(mock_kpsewhich):
        latextools.latex_to_png_dvipng(s, wrap)

def mock_kpsewhich(filename):
    if False:
        for i in range(10):
            print('nop')
    assert filename == 'breqn.sty'
    return None

@contextmanager
def patch_latextool(mock=mock_kpsewhich):
    if False:
        print('Hello World!')
    with patch.object(latextools, 'kpsewhich', mock):
        yield

@pytest.mark.parametrize('context', [no_op, patch_latextool])
@pytest.mark.parametrize('s_wrap', [('$x^2$', False), ('x^2', True)])
def test_latex_to_png_mpl_runs(s_wrap, context):
    if False:
        print('Hello World!')
    '\n    Test that latex_to_png_mpl just runs without error.\n    '
    try:
        import matplotlib
    except ImportError:
        pytest.skip('This needs matplotlib to be available')
        return
    (s, wrap) = s_wrap
    with context():
        latextools.latex_to_png_mpl(s, wrap)

@skipif_not_matplotlib
def test_latex_to_html():
    if False:
        for i in range(10):
            print('nop')
    img = latextools.latex_to_html('$x^2$')
    assert 'data:image/png;base64,iVBOR' in img

def test_genelatex_no_wrap():
    if False:
        return 10
    '\n    Test genelatex with wrap=False.\n    '

    def mock_kpsewhich(filename):
        if False:
            return 10
        assert False, 'kpsewhich should not be called (called with {0})'.format(filename)
    with patch_latextool(mock_kpsewhich):
        assert '\n'.join(latextools.genelatex('body text', False)) == '\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{amsthm}\n\\usepackage{amssymb}\n\\usepackage{bm}\n\\pagestyle{empty}\n\\begin{document}\nbody text\n\\end{document}'

def test_genelatex_wrap_with_breqn():
    if False:
        while True:
            i = 10
    '\n    Test genelatex with wrap=True for the case breqn.sty is installed.\n    '

    def mock_kpsewhich(filename):
        if False:
            for i in range(10):
                print('nop')
        assert filename == 'breqn.sty'
        return 'path/to/breqn.sty'
    with patch_latextool(mock_kpsewhich):
        assert '\n'.join(latextools.genelatex('x^2', True)) == '\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{amsthm}\n\\usepackage{amssymb}\n\\usepackage{bm}\n\\usepackage{breqn}\n\\pagestyle{empty}\n\\begin{document}\n\\begin{dmath*}\nx^2\n\\end{dmath*}\n\\end{document}'

def test_genelatex_wrap_without_breqn():
    if False:
        return 10
    '\n    Test genelatex with wrap=True for the case breqn.sty is not installed.\n    '

    def mock_kpsewhich(filename):
        if False:
            while True:
                i = 10
        assert filename == 'breqn.sty'
        return None
    with patch_latextool(mock_kpsewhich):
        assert '\n'.join(latextools.genelatex('x^2', True)) == '\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{amsthm}\n\\usepackage{amssymb}\n\\usepackage{bm}\n\\pagestyle{empty}\n\\begin{document}\n$$x^2$$\n\\end{document}'

@skipif_not_matplotlib
@onlyif_cmds_exist('latex', 'dvipng')
def test_latex_to_png_color():
    if False:
        i = 10
        return i + 15
    '\n    Test color settings for latex_to_png.\n    '
    latex_string = '$x^2$'
    default_value = latextools.latex_to_png(latex_string, wrap=False)
    default_hexblack = latextools.latex_to_png(latex_string, wrap=False, color='#000000')
    dvipng_default = latextools.latex_to_png_dvipng(latex_string, False)
    dvipng_black = latextools.latex_to_png_dvipng(latex_string, False, 'Black')
    assert dvipng_default == dvipng_black
    mpl_default = latextools.latex_to_png_mpl(latex_string, False)
    mpl_black = latextools.latex_to_png_mpl(latex_string, False, 'Black')
    assert mpl_default == mpl_black
    assert default_value in [dvipng_black, mpl_black]
    assert default_hexblack in [dvipng_black, mpl_black]
    dvipng_maroon = latextools.latex_to_png_dvipng(latex_string, False, 'Maroon')
    assert dvipng_black != dvipng_maroon
    mpl_maroon = latextools.latex_to_png_mpl(latex_string, False, 'Maroon')
    assert mpl_black != mpl_maroon
    mpl_white = latextools.latex_to_png_mpl(latex_string, False, 'White')
    mpl_hexwhite = latextools.latex_to_png_mpl(latex_string, False, '#FFFFFF')
    assert mpl_white == mpl_hexwhite
    mpl_white_scale = latextools.latex_to_png_mpl(latex_string, False, 'White', 1.2)
    assert mpl_white != mpl_white_scale

def test_latex_to_png_invalid_hex_colors():
    if False:
        return 10
    '\n    Test that invalid hex colors provided to dvipng gives an exception.\n    '
    latex_string = '$x^2$'
    pytest.raises(ValueError, lambda : latextools.latex_to_png(latex_string, backend='dvipng', color='#f00bar'))
    pytest.raises(ValueError, lambda : latextools.latex_to_png(latex_string, backend='dvipng', color='#f00'))