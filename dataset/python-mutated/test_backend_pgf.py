import datetime
from io import BytesIO
import os
import shutil
import numpy as np
from packaging.version import parse as parse_version
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.testing import _has_tex_package, _check_for_pgf
from matplotlib.testing.exceptions import ImageComparisonFailure
from matplotlib.testing.compare import compare_images
from matplotlib.backends.backend_pgf import PdfPages
from matplotlib.testing.decorators import _image_directories, check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_ghostscript, needs_pgf_lualatex, needs_pgf_pdflatex, needs_pgf_xelatex
(baseline_dir, result_dir) = _image_directories(lambda : 'dummy func')

def compare_figure(fname, savefig_kwargs={}, tol=0):
    if False:
        for i in range(10):
            print('nop')
    actual = os.path.join(result_dir, fname)
    plt.savefig(actual, **savefig_kwargs)
    expected = os.path.join(result_dir, 'expected_%s' % fname)
    shutil.copyfile(os.path.join(baseline_dir, fname), expected)
    err = compare_images(expected, actual, tol=tol)
    if err:
        raise ImageComparisonFailure(err)

@needs_pgf_xelatex
@needs_ghostscript
@pytest.mark.backend('pgf')
def test_tex_special_chars(tmp_path):
    if False:
        while True:
            i = 10
    fig = plt.figure()
    fig.text(0.5, 0.5, '%_^ $a_b^c$')
    buf = BytesIO()
    fig.savefig(buf, format='png', backend='pgf')
    buf.seek(0)
    t = plt.imread(buf)
    assert not (t == 1).all()

def create_figure():
    if False:
        i = 10
        return i + 15
    plt.figure()
    x = np.linspace(0, 1, 15)
    plt.plot(x, x ** 2, 'b-')
    plt.plot(x, 1 - x ** 2, 'g>')
    plt.fill_between([0.0, 0.4], [0.4, 0.0], hatch='//', facecolor='lightgray', edgecolor='red')
    plt.fill([3, 3, 0.8, 0.8, 3], [2, -2, -2, 0, 2], 'b')
    plt.plot([0.9], [0.5], 'ro', markersize=3)
    plt.text(0.9, 0.5, 'unicode (ü, °, §) and math ($\\mu_i = x_i^2$)', ha='right', fontsize=20)
    plt.ylabel('sans-serif, blue, $\\frac{\\sqrt{x}}{y^2}$..', family='sans-serif', color='blue')
    plt.text(1, 1, 'should be clipped as default clip_box is Axes bbox', fontsize=20, clip_on=True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

@needs_pgf_xelatex
@pytest.mark.backend('pgf')
@image_comparison(['pgf_xelatex.pdf'], style='default')
def test_xelatex():
    if False:
        while True:
            i = 10
    rc_xelatex = {'font.family': 'serif', 'pgf.rcfonts': False}
    mpl.rcParams.update(rc_xelatex)
    create_figure()
try:
    _old_gs_version = mpl._get_executable_info('gs').version < parse_version('9.50')
except mpl.ExecutableNotFoundError:
    _old_gs_version = True

@needs_pgf_pdflatex
@pytest.mark.skipif(not _has_tex_package('type1ec'), reason='needs type1ec.sty')
@pytest.mark.skipif(not _has_tex_package('ucs'), reason='needs ucs.sty')
@pytest.mark.backend('pgf')
@image_comparison(['pgf_pdflatex.pdf'], style='default', tol=11.71 if _old_gs_version else 0)
def test_pdflatex():
    if False:
        print('Hello World!')
    rc_pdflatex = {'font.family': 'serif', 'pgf.rcfonts': False, 'pgf.texsystem': 'pdflatex', 'pgf.preamble': '\\usepackage[utf8x]{inputenc}\\usepackage[T1]{fontenc}'}
    mpl.rcParams.update(rc_pdflatex)
    create_figure()

@needs_pgf_xelatex
@needs_pgf_pdflatex
@mpl.style.context('default')
@pytest.mark.backend('pgf')
def test_rcupdate():
    if False:
        return 10
    rc_sets = [{'font.family': 'sans-serif', 'font.size': 30, 'figure.subplot.left': 0.2, 'lines.markersize': 10, 'pgf.rcfonts': False, 'pgf.texsystem': 'xelatex'}, {'font.family': 'monospace', 'font.size': 10, 'figure.subplot.left': 0.1, 'lines.markersize': 20, 'pgf.rcfonts': False, 'pgf.texsystem': 'pdflatex', 'pgf.preamble': '\\usepackage[utf8x]{inputenc}\\usepackage[T1]{fontenc}\\usepackage{sfmath}'}]
    tol = [0, 13.2] if _old_gs_version else [0, 0]
    for (i, rc_set) in enumerate(rc_sets):
        with mpl.rc_context(rc_set):
            for (substring, pkg) in [('sfmath', 'sfmath'), ('utf8x', 'ucs')]:
                if substring in mpl.rcParams['pgf.preamble'] and (not _has_tex_package(pkg)):
                    pytest.skip(f'needs {pkg}.sty')
            create_figure()
            compare_figure(f'pgf_rcupdate{i + 1}.pdf', tol=tol[i])

@needs_pgf_xelatex
@mpl.style.context('default')
@pytest.mark.backend('pgf')
def test_pathclip():
    if False:
        return 10
    np.random.seed(19680801)
    mpl.rcParams.update({'font.family': 'serif', 'pgf.rcfonts': False})
    (fig, axs) = plt.subplots(1, 2)
    axs[0].plot([0.0, 1e+100], [0.0, 1e+100])
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(0, 1)
    axs[1].scatter([0, 1], [1, 1])
    axs[1].hist(np.random.normal(size=1000), bins=20, range=[-10, 10])
    axs[1].set_xscale('log')
    fig.savefig(BytesIO(), format='pdf')

@needs_pgf_xelatex
@pytest.mark.backend('pgf')
@image_comparison(['pgf_mixedmode.pdf'], style='default')
def test_mixedmode():
    if False:
        i = 10
        return i + 15
    mpl.rcParams.update({'font.family': 'serif', 'pgf.rcfonts': False})
    (Y, X) = np.ogrid[-1:1:40j, -1:1:40j]
    plt.pcolor(X ** 2 + Y ** 2).set_rasterized(True)

@needs_pgf_xelatex
@mpl.style.context('default')
@pytest.mark.backend('pgf')
def test_bbox_inches():
    if False:
        print('Hello World!')
    mpl.rcParams.update({'font.family': 'serif', 'pgf.rcfonts': False})
    (fig, (ax1, ax2)) = plt.subplots(1, 2)
    ax1.plot(range(5))
    ax2.plot(range(5))
    plt.tight_layout()
    bbox = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    compare_figure('pgf_bbox_inches.pdf', savefig_kwargs={'bbox_inches': bbox}, tol=0)

@mpl.style.context('default')
@pytest.mark.backend('pgf')
@pytest.mark.parametrize('system', [pytest.param('lualatex', marks=[needs_pgf_lualatex]), pytest.param('pdflatex', marks=[needs_pgf_pdflatex]), pytest.param('xelatex', marks=[needs_pgf_xelatex])])
def test_pdf_pages(system):
    if False:
        i = 10
        return i + 15
    rc_pdflatex = {'font.family': 'serif', 'pgf.rcfonts': False, 'pgf.texsystem': system}
    mpl.rcParams.update(rc_pdflatex)
    (fig1, ax1) = plt.subplots()
    ax1.plot(range(5))
    fig1.tight_layout()
    (fig2, ax2) = plt.subplots(figsize=(3, 2))
    ax2.plot(range(5))
    fig2.tight_layout()
    path = os.path.join(result_dir, f'pdfpages_{system}.pdf')
    md = {'Author': 'me', 'Title': 'Multipage PDF with pgf', 'Subject': 'Test page', 'Keywords': 'test,pdf,multipage', 'ModDate': datetime.datetime(1968, 8, 1, tzinfo=datetime.timezone(datetime.timedelta(0))), 'Trapped': 'Unknown'}
    with PdfPages(path, metadata=md) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.savefig(fig1)
        assert pdf.get_pagecount() == 3

@mpl.style.context('default')
@pytest.mark.backend('pgf')
@pytest.mark.parametrize('system', [pytest.param('lualatex', marks=[needs_pgf_lualatex]), pytest.param('pdflatex', marks=[needs_pgf_pdflatex]), pytest.param('xelatex', marks=[needs_pgf_xelatex])])
def test_pdf_pages_metadata_check(monkeypatch, system):
    if False:
        i = 10
        return i + 15
    pikepdf = pytest.importorskip('pikepdf')
    monkeypatch.setenv('SOURCE_DATE_EPOCH', '0')
    mpl.rcParams.update({'pgf.texsystem': system})
    (fig, ax) = plt.subplots()
    ax.plot(range(5))
    md = {'Author': 'me', 'Title': 'Multipage PDF with pgf', 'Subject': 'Test page', 'Keywords': 'test,pdf,multipage', 'ModDate': datetime.datetime(1968, 8, 1, tzinfo=datetime.timezone(datetime.timedelta(0))), 'Trapped': 'True'}
    path = os.path.join(result_dir, f'pdfpages_meta_check_{system}.pdf')
    with PdfPages(path, metadata=md) as pdf:
        pdf.savefig(fig)
    with pikepdf.Pdf.open(path) as pdf:
        info = {k: str(v) for (k, v) in pdf.docinfo.items()}
    if '/PTEX.FullBanner' in info:
        del info['/PTEX.FullBanner']
    if '/PTEX.Fullbanner' in info:
        del info['/PTEX.Fullbanner']
    producer = info.pop('/Producer')
    assert producer == f'Matplotlib pgf backend v{mpl.__version__}' or (system == 'lualatex' and 'LuaTeX' in producer)
    assert info == {'/Author': 'me', '/CreationDate': 'D:19700101000000Z', '/Creator': f'Matplotlib v{mpl.__version__}, https://matplotlib.org', '/Keywords': 'test,pdf,multipage', '/ModDate': 'D:19680801000000Z', '/Subject': 'Test page', '/Title': 'Multipage PDF with pgf', '/Trapped': '/True'}

@needs_pgf_xelatex
def test_multipage_keep_empty(tmp_path):
    if False:
        return 10
    os.chdir(tmp_path)
    with pytest.warns(mpl.MatplotlibDeprecationWarning), PdfPages('a.pdf') as pdf:
        pass
    assert os.path.exists('a.pdf')
    with pytest.warns(mpl.MatplotlibDeprecationWarning), PdfPages('b.pdf', keep_empty=True) as pdf:
        pass
    assert os.path.exists('b.pdf')
    with PdfPages('c.pdf', keep_empty=False) as pdf:
        pass
    assert not os.path.exists('c.pdf')
    with PdfPages('d.pdf') as pdf:
        pdf.savefig(plt.figure())
    assert os.path.exists('d.pdf')
    with pytest.warns(mpl.MatplotlibDeprecationWarning), PdfPages('e.pdf', keep_empty=True) as pdf:
        pdf.savefig(plt.figure())
    assert os.path.exists('e.pdf')
    with PdfPages('f.pdf', keep_empty=False) as pdf:
        pdf.savefig(plt.figure())
    assert os.path.exists('f.pdf')

@needs_pgf_xelatex
def test_tex_restart_after_error():
    if False:
        while True:
            i = 10
    fig = plt.figure()
    fig.suptitle('\\oops')
    with pytest.raises(ValueError):
        fig.savefig(BytesIO(), format='pgf')
    fig = plt.figure()
    fig.suptitle('this is ok')
    fig.savefig(BytesIO(), format='pgf')

@needs_pgf_xelatex
def test_bbox_inches_tight():
    if False:
        return 10
    (fig, ax) = plt.subplots()
    ax.imshow([[0, 1], [2, 3]])
    fig.savefig(BytesIO(), format='pdf', backend='pgf', bbox_inches='tight')

@needs_pgf_xelatex
@needs_ghostscript
def test_png_transparency():
    if False:
        i = 10
        return i + 15
    buf = BytesIO()
    plt.figure().savefig(buf, format='png', backend='pgf', transparent=True)
    buf.seek(0)
    t = plt.imread(buf)
    assert (t[..., 3] == 0).all()

@needs_pgf_xelatex
def test_unknown_font(caplog):
    if False:
        return 10
    with caplog.at_level('WARNING'):
        mpl.rcParams['font.family'] = 'this-font-does-not-exist'
        plt.figtext(0.5, 0.5, 'hello, world')
        plt.savefig(BytesIO(), format='pgf')
    assert 'Ignoring unknown font: this-font-does-not-exist' in [r.getMessage() for r in caplog.records]

@check_figures_equal(extensions=['pdf'])
@pytest.mark.parametrize('texsystem', ('pdflatex', 'xelatex', 'lualatex'))
@pytest.mark.backend('pgf')
def test_minus_signs_with_tex(fig_test, fig_ref, texsystem):
    if False:
        i = 10
        return i + 15
    if not _check_for_pgf(texsystem):
        pytest.skip(texsystem + ' + pgf is required')
    mpl.rcParams['pgf.texsystem'] = texsystem
    fig_test.text(0.5, 0.5, '$-1$')
    fig_ref.text(0.5, 0.5, '$−1$')

@pytest.mark.backend('pgf')
def test_sketch_params():
    if False:
        i = 10
        return i + 15
    (fig, ax) = plt.subplots(figsize=(3, 3))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    (handle,) = ax.plot([0, 1])
    handle.set_sketch_params(scale=5, length=30, randomness=42)
    with BytesIO() as fd:
        fig.savefig(fd, format='pgf')
        buf = fd.getvalue().decode()
    baseline = '\\pgfpathmoveto{\\pgfqpoint{0.375000in}{0.300000in}}%\n\\pgfpathlineto{\\pgfqpoint{2.700000in}{2.700000in}}%\n\\usepgfmodule{decorations}%\n\\usepgflibrary{decorations.pathmorphing}%\n\\pgfkeys{/pgf/decoration/.cd, segment length = 0.150000in, amplitude = 0.100000in}%\n\\pgfmathsetseed{42}%\n\\pgfdecoratecurrentpath{random steps}%\n\\pgfusepath{stroke}%'
    assert baseline in buf