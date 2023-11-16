from __future__ import annotations
import io
from pathlib import Path
import platform
import re
import shlex
from xml.etree import ElementTree as ET
from typing import Any
import numpy as np
from packaging.version import parse as parse_version
import pyparsing
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.pyplot as plt
from matplotlib import mathtext, _mathtext
pyparsing_version = parse_version(pyparsing.__version__)
math_tests = ['$a+b+\\dot s+\\dot{s}+\\ldots$', '$x\\hspace{-0.2}\\doteq\\hspace{-0.2}y$', '\\$100.00 $\\alpha \\_$', '$\\frac{\\$100.00}{y}$', '$x   y$', '$x+y\\ x=y\\ x<y\\ x:y\\ x,y\\ x@y$', '$100\\%y\\ x*y\\ x/y x\\$y$', '$x\\leftarrow y\\ x\\forall y\\ x-y$', '$x \\sf x \\bf x {\\cal X} \\rm x$', '$x\\ x\\,x\\;x\\quad x\\qquad x\\!x\\hspace{ 0.5 }y$', '$\\{ \\rm braces \\}$', '$\\left[\\left\\lfloor\\frac{5}{\\frac{\\left(3\\right)}{4}} y\\right)\\right]$', '$\\left(x\\right)$', '$\\sin(x)$', '$x_2$', '$x^2$', '$x^2_y$', '$x_y^2$', '$\\sum _{\\genfrac{}{}{0}{}{0\\leq i\\leq m}{0<j<n}}f\\left(i,j\\right)\\mathcal{R}\\prod_{i=\\alpha_{i+1}}^\\infty a_i \\sin(2 \\pi f x_i)\\sqrt[2]{\\prod^\\frac{x}{2\\pi^2}_\\infty}$', '$x = \\frac{x+\\frac{5}{2}}{\\frac{y+3}{8}}$', '$dz/dt = \\gamma x^2 + {\\rm sin}(2\\pi y+\\phi)$', 'Foo: $\\alpha_{i+1}^j = {\\rm sin}(2\\pi f_j t_i) e^{-5 t_i/\\tau}$', None, 'Variable $i$ is good', '$\\Delta_i^j$', '$\\Delta^j_{i+1}$', '$\\ddot{o}\\acute{e}\\grave{e}\\hat{O}\\breve{\\imath}\\tilde{n}\\vec{q}$', '$\\arccos((x^i))$', '$\\gamma = \\frac{x=\\frac{6}{8}}{y} \\delta$', '$\\limsup_{x\\to\\infty}$', None, "$f'\\quad f'''(x)\\quad ''/\\mathrm{yr}$", '$\\frac{x_2888}{y}$', '$\\sqrt[3]{\\frac{X_2}{Y}}=5$', None, '$\\sqrt[3]{x}=5$', '$\\frac{X}{\\frac{X}{Y}}$', '$W^{3\\beta}_{\\delta_1 \\rho_1 \\sigma_2} = U^{3\\beta}_{\\delta_1 \\rho_1} + \\frac{1}{8 \\pi 2} \\int^{\\alpha_2}_{\\alpha_2} d \\alpha^\\prime_2 \\left[\\frac{ U^{2\\beta}_{\\delta_1 \\rho_1} - \\alpha^\\prime_2U^{1\\beta}_{\\rho_1 \\sigma_2} }{U^{0\\beta}_{\\rho_1 \\sigma_2}}\\right]$', '$\\mathcal{H} = \\int d \\tau \\left(\\epsilon E^2 + \\mu H^2\\right)$', '$\\widehat{abc}\\widetilde{def}$', '$\\Gamma \\Delta \\Theta \\Lambda \\Xi \\Pi \\Sigma \\Upsilon \\Phi \\Psi \\Omega$', '$\\alpha \\beta \\gamma \\delta \\epsilon \\zeta \\eta \\theta \\iota \\lambda \\mu \\nu \\xi \\pi \\kappa \\rho \\sigma \\tau \\upsilon \\phi \\chi \\psi$', '${x}^{2}{y}^{2}$', '${}_{2}F_{3}$', '$\\frac{x+{y}^{2}}{k+1}$', '$x+{y}^{\\frac{2}{k+1}}$', '$\\frac{a}{b/2}$', '${a}_{0}+\\frac{1}{{a}_{1}+\\frac{1}{{a}_{2}+\\frac{1}{{a}_{3}+\\frac{1}{{a}_{4}}}}}$', '${a}_{0}+\\frac{1}{{a}_{1}+\\frac{1}{{a}_{2}+\\frac{1}{{a}_{3}+\\frac{1}{{a}_{4}}}}}$', '$\\binom{n}{k/2}$', '$\\binom{p}{2}{x}^{2}{y}^{p-2}-\\frac{1}{1-x}\\frac{1}{1-{x}^{2}}$', '${x}^{2y}$', '$\\sum _{i=1}^{p}\\sum _{j=1}^{q}\\sum _{k=1}^{r}{a}_{ij}{b}_{jk}{c}_{ki}$', '$\\sqrt{1+\\sqrt{1+\\sqrt{1+\\sqrt{1+\\sqrt{1+\\sqrt{1+\\sqrt{1+x}}}}}}}$', '$\\left(\\frac{{\\partial }^{2}}{\\partial {x}^{2}}+\\frac{{\\partial }^{2}}{\\partial {y}^{2}}\\right){|\\varphi \\left(x+iy\\right)|}^{2}=0$', '${2}^{{2}^{{2}^{x}}}$', '${\\int }_{1}^{x}\\frac{\\mathrm{dt}}{t}$', '$\\int {\\int }_{D}\\mathrm{dx} \\mathrm{dy}$', '${y}_{{x}^{2}}$', '${y}_{{x}_{2}}$', '${x}_{92}^{31415}+\\pi $', '${x}_{{y}_{b}^{a}}^{{z}_{c}^{d}}$', '${y}_{3}^{\\prime \\prime \\prime }$', '$\\left( \\xi \\left( 1 - \\xi \\right) \\right)$', '$\\left(2 \\, a=b\\right)$', '$? ! &$', None, None, '$\\left\\Vert \\frac{a}{b} \\right\\Vert \\left\\vert \\frac{a}{b} \\right\\vert \\left\\| \\frac{a}{b}\\right\\| \\left| \\frac{a}{b} \\right| \\Vert a \\Vert \\vert b \\vert \\| a \\| | b |$', '$\\mathring{A}  \\AA$', '$M \\, M \\thinspace M \\/ M \\> M \\: M \\; M \\ M \\enspace M \\quad M \\qquad M \\! M$', '$\\Cap$ $\\Cup$ $\\leftharpoonup$ $\\barwedge$ $\\rightharpoonup$', '$\\hspace{-0.2}\\dotplus\\hspace{-0.2}$ $\\hspace{-0.2}\\doteq\\hspace{-0.2}$ $\\hspace{-0.2}\\doteqdot\\hspace{-0.2}$ $\\ddots$', '$xyz^kx_kx^py^{p-2} d_i^jb_jc_kd x^j_i E^0 E^0_u$', '${xyz}^k{x}_{k}{x}^{p}{y}^{p-2} {d}_{i}^{j}{b}_{j}{c}_{k}{d} {x}^{j}_{i}{E}^{0}{E}^0_u$', '${\\int}_x^x x\\oint_x^x x\\int_{X}^{X}x\\int_x x \\int^x x \\int_{x} x\\int^{x}{\\int}_{x} x{\\int}^{x}_{x}x$', 'testing$^{123}$', None, '$6-2$; $-2$; $ -2$; ${-2}$; ${  -2}$; $20^{+3}_{-2}$', '$\\overline{\\omega}^x \\frac{1}{2}_0^x$', '$,$ $.$ $1{,}234{, }567{ , }890$ and $1,234,567,890$', '$\\left(X\\right)_{a}^{b}$', '$\\dfrac{\\$100.00}{y}$']
svgastext_math_tests = ['$-$-']
lightweight_math_tests = ['$\\sqrt[ab]{123}$', '$x \\overset{f}{\\rightarrow} \\overset{f}{x} \\underset{xx}{ff} \\overset{xx}{ff} \\underset{f}{x} \\underset{f}{\\leftarrow} x$', '$\\sum x\\quad\\sum^nx\\quad\\sum_nx\\quad\\sum_n^nx\\quad\\prod x\\quad\\prod^nx\\quad\\prod_nx\\quad\\prod_n^nx$', '$1.$ $2.$ $19680801.$ $a.$ $b.$ $mpl.$', '$\\text{text}_{\\text{sub}}^{\\text{sup}} + \\text{\\$foo\\$} + \\frac{\\text{num}}{\\mathbf{\\text{den}}}\\text{with space, curly brackets \\{\\}, and dash -}$', '$\\boldsymbol{abcde} \\boldsymbol{+} \\boldsymbol{\\Gamma + \\Omega} \\boldsymbol{01234} \\boldsymbol{\\alpha * \\beta}$', '$\\left\\lbrace\\frac{\\left\\lbrack A^b_c\\right\\rbrace}{\\left\\leftbrace D^e_f \\right\\rbrack}\\right\\rightbrace\\ \\left\\leftparen\\max_{x} \\left\\lgroup \\frac{A}{B}\\right\\rgroup \\right\\rightparen$', '$\\left( a\\middle. b \\right)$ $\\left( \\frac{a}{b} \\middle\\vert x_i \\in P^S \\right)$ $\\left[ 1 - \\middle| a\\middle| + \\left( x  - \\left\\lfloor \\dfrac{a}{b}\\right\\rfloor \\right)  \\right]$', '$\\sum_{\\substack{k = 1\\\\ k \\neq \\lfloor n/2\\rfloor}}^{n}P(i,j) \\sum_{\\substack{i \\neq 0\\\\ -1 \\leq i \\leq 3\\\\ 1 \\leq j \\leq 5}} F^i(x,y) \\sum_{\\substack{\\left \\lfloor \\frac{n}{2} \\right\\rfloor}} F(n)$']
digits = '0123456789'
uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
lowercase = 'abcdefghijklmnopqrstuvwxyz'
uppergreek = '\\Gamma \\Delta \\Theta \\Lambda \\Xi \\Pi \\Sigma \\Upsilon \\Phi \\Psi \\Omega'
lowergreek = '\\alpha \\beta \\gamma \\delta \\epsilon \\zeta \\eta \\theta \\iota \\lambda \\mu \\nu \\xi \\pi \\kappa \\rho \\sigma \\tau \\upsilon \\phi \\chi \\psi'
all = [digits, uppercase, lowercase, uppergreek, lowergreek]
font_test_specs: list[tuple[None | list[str], Any]] = [([], all), (['mathrm'], all), (['mathbf'], all), (['mathit'], all), (['mathtt'], [digits, uppercase, lowercase]), (None, 3), (None, 3), (None, 3), (['mathbb'], [digits, uppercase, lowercase, '\\Gamma \\Pi \\Sigma \\gamma \\pi']), (['mathrm', 'mathbb'], [digits, uppercase, lowercase, '\\Gamma \\Pi \\Sigma \\gamma \\pi']), (['mathbf', 'mathbb'], [digits, uppercase, lowercase, '\\Gamma \\Pi \\Sigma \\gamma \\pi']), (['mathcal'], [uppercase]), (['mathfrak'], [uppercase, lowercase]), (['mathbf', 'mathfrak'], [uppercase, lowercase]), (['mathscr'], [uppercase, lowercase]), (['mathsf'], [digits, uppercase, lowercase]), (['mathrm', 'mathsf'], [digits, uppercase, lowercase]), (['mathbf', 'mathsf'], [digits, uppercase, lowercase]), (['mathbfit'], all)]
font_tests: list[None | str] = []
for (fonts, chars) in font_test_specs:
    if fonts is None:
        font_tests.extend([None] * chars)
    else:
        wrapper = ''.join([' '.join(fonts), ' $', *('\\%s{' % font for font in fonts), '%s', *('}' for font in fonts), '$'])
        for set in chars:
            font_tests.append(wrapper % set)

@pytest.fixture
def baseline_images(request, fontset, index, text):
    if False:
        for i in range(10):
            print('nop')
    if text is None:
        pytest.skip('test has been removed')
    return ['%s_%s_%02d' % (request.param, fontset, index)]

@pytest.mark.parametrize('index, text', enumerate(math_tests), ids=range(len(math_tests)))
@pytest.mark.parametrize('fontset', ['cm', 'stix', 'stixsans', 'dejavusans', 'dejavuserif'])
@pytest.mark.parametrize('baseline_images', ['mathtext'], indirect=True)
@image_comparison(baseline_images=None, tol=0.011 if platform.machine() in ('ppc64le', 's390x') else 0)
def test_mathtext_rendering(baseline_images, fontset, index, text):
    if False:
        while True:
            i = 10
    mpl.rcParams['mathtext.fontset'] = fontset
    fig = plt.figure(figsize=(5.25, 0.75))
    fig.text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='center')

@pytest.mark.parametrize('index, text', enumerate(svgastext_math_tests), ids=range(len(svgastext_math_tests)))
@pytest.mark.parametrize('fontset', ['cm', 'dejavusans'])
@pytest.mark.parametrize('baseline_images', ['mathtext0'], indirect=True)
@image_comparison(baseline_images=None, extensions=['svg'], savefig_kwarg={'metadata': {'Creator': None, 'Date': None, 'Format': None, 'Type': None}})
def test_mathtext_rendering_svgastext(baseline_images, fontset, index, text):
    if False:
        i = 10
        return i + 15
    mpl.rcParams['mathtext.fontset'] = fontset
    mpl.rcParams['svg.fonttype'] = 'none'
    fig = plt.figure(figsize=(5.25, 0.75))
    fig.patch.set(visible=False)
    fig.text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='center')

@pytest.mark.parametrize('index, text', enumerate(lightweight_math_tests), ids=range(len(lightweight_math_tests)))
@pytest.mark.parametrize('fontset', ['dejavusans'])
@pytest.mark.parametrize('baseline_images', ['mathtext1'], indirect=True)
@image_comparison(baseline_images=None, extensions=['png'])
def test_mathtext_rendering_lightweight(baseline_images, fontset, index, text):
    if False:
        print('Hello World!')
    fig = plt.figure(figsize=(5.25, 0.75))
    fig.text(0.5, 0.5, text, math_fontfamily=fontset, horizontalalignment='center', verticalalignment='center')

@pytest.mark.parametrize('index, text', enumerate(font_tests), ids=range(len(font_tests)))
@pytest.mark.parametrize('fontset', ['cm', 'stix', 'stixsans', 'dejavusans', 'dejavuserif'])
@pytest.mark.parametrize('baseline_images', ['mathfont'], indirect=True)
@image_comparison(baseline_images=None, extensions=['png'], tol=0.011 if platform.machine() in ('ppc64le', 's390x') else 0)
def test_mathfont_rendering(baseline_images, fontset, index, text):
    if False:
        while True:
            i = 10
    mpl.rcParams['mathtext.fontset'] = fontset
    fig = plt.figure(figsize=(5.25, 0.75))
    fig.text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='center')

@check_figures_equal(extensions=['png'])
def test_short_long_accents(fig_test, fig_ref):
    if False:
        return 10
    acc_map = _mathtext.Parser._accent_map
    short_accs = [s for s in acc_map if len(s) == 1]
    corresponding_long_accs = []
    for s in short_accs:
        (l,) = [l for l in acc_map if len(l) > 1 and acc_map[l] == acc_map[s]]
        corresponding_long_accs.append(l)
    fig_test.text(0, 0.5, '$' + ''.join((f'\\{s}a' for s in short_accs)) + '$')
    fig_ref.text(0, 0.5, '$' + ''.join((f'\\{l} a' for l in corresponding_long_accs)) + '$')

def test_fontinfo():
    if False:
        print('Hello World!')
    fontpath = mpl.font_manager.findfont('DejaVu Sans')
    font = mpl.ft2font.FT2Font(fontpath)
    table = font.get_sfnt_table('head')
    assert table is not None
    assert table['version'] == (1, 0)

@pytest.mark.xfail(pyparsing_version.release == (3, 1, 0), reason='Error messages are incorrect for this version')
@pytest.mark.parametrize('math, msg', [('$\\hspace{}$', 'Expected \\hspace{space}'), ('$\\hspace{foo}$', 'Expected \\hspace{space}'), ('$\\sinx$', 'Unknown symbol: \\sinx'), ('$\\dotx$', 'Unknown symbol: \\dotx'), ('$\\frac$', 'Expected \\frac{num}{den}'), ('$\\frac{}{}$', 'Expected \\frac{num}{den}'), ('$\\binom$', 'Expected \\binom{num}{den}'), ('$\\binom{}{}$', 'Expected \\binom{num}{den}'), ('$\\genfrac$', 'Expected \\genfrac{ldelim}{rdelim}{rulesize}{style}{num}{den}'), ('$\\genfrac{}{}{}{}{}{}$', 'Expected \\genfrac{ldelim}{rdelim}{rulesize}{style}{num}{den}'), ('$\\sqrt$', 'Expected \\sqrt{value}'), ('$\\sqrt f$', 'Expected \\sqrt{value}'), ('$\\overline$', 'Expected \\overline{body}'), ('$\\overline{}$', 'Expected \\overline{body}'), ('$\\leftF$', 'Expected a delimiter'), ('$\\rightF$', 'Unknown symbol: \\rightF'), ('$\\left(\\right$', 'Expected a delimiter'), ('$\\left($', re.compile('Expected ("|\\\'\\\\)\\\\right["\\\']')), ('$\\dfrac$', 'Expected \\dfrac{num}{den}'), ('$\\dfrac{}{}$', 'Expected \\dfrac{num}{den}'), ('$\\overset$', 'Expected \\overset{annotation}{body}'), ('$\\underset$', 'Expected \\underset{annotation}{body}'), ('$\\foo$', 'Unknown symbol: \\foo'), ('$a^2^2$', 'Double superscript'), ('$a_2_2$', 'Double subscript'), ('$a^2_a^2$', 'Double superscript'), ('$a = {b$', "Expected '}'")], ids=['hspace without value', 'hspace with invalid value', 'function without space', 'accent without space', 'frac without parameters', 'frac with empty parameters', 'binom without parameters', 'binom with empty parameters', 'genfrac without parameters', 'genfrac with empty parameters', 'sqrt without parameters', 'sqrt with invalid value', 'overline without parameters', 'overline with empty parameter', 'left with invalid delimiter', 'right with invalid delimiter', 'unclosed parentheses with sizing', 'unclosed parentheses without sizing', 'dfrac without parameters', 'dfrac with empty parameters', 'overset without parameters', 'underset without parameters', 'unknown symbol', 'double superscript', 'double subscript', 'super on sub without braces', 'unclosed group'])
def test_mathtext_exceptions(math, msg):
    if False:
        return 10
    parser = mathtext.MathTextParser('agg')
    match = re.escape(msg) if isinstance(msg, str) else msg
    with pytest.raises(ValueError, match=match):
        parser.parse(math)

def test_get_unicode_index_exception():
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError):
        _mathtext.get_unicode_index('\\foo')

def test_single_minus_sign():
    if False:
        print('Hello World!')
    fig = plt.figure()
    fig.text(0.5, 0.5, '$-$')
    fig.canvas.draw()
    t = np.asarray(fig.canvas.renderer.buffer_rgba())
    assert (t != 255).any()

@check_figures_equal(extensions=['png'])
def test_spaces(fig_test, fig_ref):
    if False:
        print('Hello World!')
    fig_test.text(0.5, 0.5, '$1\\,2\\>3\\ 4$')
    fig_ref.text(0.5, 0.5, '$1\\/2\\:3~4$')

@check_figures_equal(extensions=['png'])
def test_operator_space(fig_test, fig_ref):
    if False:
        return 10
    fig_test.text(0.1, 0.1, '$\\log 6$')
    fig_test.text(0.1, 0.2, '$\\log(6)$')
    fig_test.text(0.1, 0.3, '$\\arcsin 6$')
    fig_test.text(0.1, 0.4, '$\\arcsin|6|$')
    fig_test.text(0.1, 0.5, '$\\operatorname{op} 6$')
    fig_test.text(0.1, 0.6, '$\\operatorname{op}[6]$')
    fig_test.text(0.1, 0.7, '$\\cos^2$')
    fig_test.text(0.1, 0.8, '$\\log_2$')
    fig_test.text(0.1, 0.9, '$\\sin^2 \\cos$')
    fig_ref.text(0.1, 0.1, '$\\mathrm{log\\,}6$')
    fig_ref.text(0.1, 0.2, '$\\mathrm{log}(6)$')
    fig_ref.text(0.1, 0.3, '$\\mathrm{arcsin\\,}6$')
    fig_ref.text(0.1, 0.4, '$\\mathrm{arcsin}|6|$')
    fig_ref.text(0.1, 0.5, '$\\mathrm{op\\,}6$')
    fig_ref.text(0.1, 0.6, '$\\mathrm{op}[6]$')
    fig_ref.text(0.1, 0.7, '$\\mathrm{cos}^2$')
    fig_ref.text(0.1, 0.8, '$\\mathrm{log}_2$')
    fig_ref.text(0.1, 0.9, '$\\mathrm{sin}^2 \\mathrm{\\,cos}$')

@check_figures_equal(extensions=['png'])
def test_inverted_delimiters(fig_test, fig_ref):
    if False:
        return 10
    fig_test.text(0.5, 0.5, '$\\left)\\right($', math_fontfamily='dejavusans')
    fig_ref.text(0.5, 0.5, '$)($', math_fontfamily='dejavusans')

@check_figures_equal(extensions=['png'])
def test_genfrac_displaystyle(fig_test, fig_ref):
    if False:
        print('Hello World!')
    fig_test.text(0.1, 0.1, '$\\dfrac{2x}{3y}$')
    thickness = _mathtext.TruetypeFonts.get_underline_thickness(None, None, fontsize=mpl.rcParams['font.size'], dpi=mpl.rcParams['savefig.dpi'])
    fig_ref.text(0.1, 0.1, '$\\genfrac{}{}{%f}{0}{2x}{3y}$' % thickness)

def test_mathtext_fallback_valid():
    if False:
        while True:
            i = 10
    for fallback in ['cm', 'stix', 'stixsans', 'None']:
        mpl.rcParams['mathtext.fallback'] = fallback

def test_mathtext_fallback_invalid():
    if False:
        return 10
    for fallback in ['abc', '']:
        with pytest.raises(ValueError, match='not a valid fallback font name'):
            mpl.rcParams['mathtext.fallback'] = fallback

@pytest.mark.parametrize('fallback,fontlist', [('cm', ['DejaVu Sans', 'mpltest', 'STIXGeneral', 'cmr10', 'STIXGeneral']), ('stix', ['DejaVu Sans', 'mpltest', 'STIXGeneral'])])
def test_mathtext_fallback(fallback, fontlist):
    if False:
        for i in range(10):
            print('nop')
    mpl.font_manager.fontManager.addfont(str(Path(__file__).resolve().parent / 'mpltest.ttf'))
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'mpltest'
    mpl.rcParams['mathtext.it'] = 'mpltest:italic'
    mpl.rcParams['mathtext.bf'] = 'mpltest:bold'
    mpl.rcParams['mathtext.bfit'] = 'mpltest:italic:bold'
    mpl.rcParams['mathtext.fallback'] = fallback
    test_str = 'a$A\\AA\\breve\\gimel$'
    buff = io.BytesIO()
    (fig, ax) = plt.subplots()
    fig.text(0.5, 0.5, test_str, fontsize=40, ha='center')
    fig.savefig(buff, format='svg')
    tspans = ET.fromstring(buff.getvalue()).findall('.//{http://www.w3.org/2000/svg}tspan[@style]')
    char_fonts = [shlex.split(tspan.attrib['style'])[-1] for tspan in tspans]
    assert char_fonts == fontlist
    mpl.font_manager.fontManager.ttflist.pop()

def test_math_to_image(tmpdir):
    if False:
        return 10
    mathtext.math_to_image('$x^2$', str(tmpdir.join('example.png')))
    mathtext.math_to_image('$x^2$', io.BytesIO())
    mathtext.math_to_image('$x^2$', io.BytesIO(), color='Maroon')

@image_comparison(baseline_images=['math_fontfamily_image.png'], savefig_kwarg={'dpi': 40})
def test_math_fontfamily():
    if False:
        for i in range(10):
            print('nop')
    fig = plt.figure(figsize=(10, 3))
    fig.text(0.2, 0.7, '$This\\ text\\ should\\ have\\ one\\ font$', size=24, math_fontfamily='dejavusans')
    fig.text(0.2, 0.3, '$This\\ text\\ should\\ have\\ another$', size=24, math_fontfamily='stix')

def test_default_math_fontfamily():
    if False:
        return 10
    mpl.rcParams['mathtext.fontset'] = 'cm'
    test_str = 'abc$abc\\alpha$'
    (fig, ax) = plt.subplots()
    text1 = fig.text(0.1, 0.1, test_str, font='Arial')
    prop1 = text1.get_fontproperties()
    assert prop1.get_math_fontfamily() == 'cm'
    text2 = fig.text(0.2, 0.2, test_str, fontproperties='Arial')
    prop2 = text2.get_fontproperties()
    assert prop2.get_math_fontfamily() == 'cm'
    fig.draw_without_rendering()

def test_argument_order():
    if False:
        for i in range(10):
            print('nop')
    mpl.rcParams['mathtext.fontset'] = 'cm'
    test_str = 'abc$abc\\alpha$'
    (fig, ax) = plt.subplots()
    text1 = fig.text(0.1, 0.1, test_str, math_fontfamily='dejavusans', font='Arial')
    prop1 = text1.get_fontproperties()
    assert prop1.get_math_fontfamily() == 'dejavusans'
    text2 = fig.text(0.2, 0.2, test_str, math_fontfamily='dejavusans', fontproperties='Arial')
    prop2 = text2.get_fontproperties()
    assert prop2.get_math_fontfamily() == 'dejavusans'
    text3 = fig.text(0.3, 0.3, test_str, font='Arial', math_fontfamily='dejavusans')
    prop3 = text3.get_fontproperties()
    assert prop3.get_math_fontfamily() == 'dejavusans'
    text4 = fig.text(0.4, 0.4, test_str, fontproperties='Arial', math_fontfamily='dejavusans')
    prop4 = text4.get_fontproperties()
    assert prop4.get_math_fontfamily() == 'dejavusans'
    fig.draw_without_rendering()

def test_mathtext_cmr10_minus_sign():
    if False:
        for i in range(10):
            print('nop')
    mpl.rcParams['font.family'] = 'cmr10'
    mpl.rcParams['axes.formatter.use_mathtext'] = True
    (fig, ax) = plt.subplots()
    ax.plot(range(-1, 1), range(-1, 1))
    fig.canvas.draw()

def test_mathtext_operators():
    if False:
        for i in range(10):
            print('nop')
    test_str = '\n    \\increment \\smallin \\notsmallowns\n    \\smallowns \\QED \\rightangle\n    \\smallintclockwise \\smallvarointclockwise\n    \\smallointctrcclockwise\n    \\ratio \\minuscolon \\dotsminusdots\n    \\sinewave \\simneqq \\nlesssim\n    \\ngtrsim \\nlessgtr \\ngtrless\n    \\cupleftarrow \\oequal \\rightassert\n    \\rightModels \\hermitmatrix \\barvee\n    \\measuredrightangle \\varlrtriangle\n    \\equalparallel \\npreccurlyeq \\nsucccurlyeq\n    \\nsqsubseteq \\nsqsupseteq \\sqsubsetneq\n    \\sqsupsetneq  \\disin \\varisins\n    \\isins \\isindot \\varisinobar\n    \\isinobar \\isinvb \\isinE\n    \\nisd \\varnis \\nis\n    \\varniobar \\niobar \\bagmember\n    \\triangle'.split()
    fig = plt.figure()
    for (x, i) in enumerate(test_str):
        fig.text(0.5, (x + 0.5) / len(test_str), '${%s}$' % i)
    fig.draw_without_rendering()

@check_figures_equal(extensions=['png'])
def test_boldsymbol(fig_test, fig_ref):
    if False:
        print('Hello World!')
    fig_test.text(0.1, 0.2, '$\\boldsymbol{\\mathrm{abc0123\\alpha}}$')
    fig_ref.text(0.1, 0.2, '$\\mathrm{abc0123\\alpha}$')