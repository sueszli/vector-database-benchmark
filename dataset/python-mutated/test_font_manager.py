from io import BytesIO, StringIO
import gc
import multiprocessing
import os
from pathlib import Path
from PIL import Image
import shutil
import subprocess
import sys
import warnings
import numpy as np
import pytest
from matplotlib.font_manager import findfont, findSystemFonts, FontEntry, FontProperties, fontManager, json_dump, json_load, get_font, is_opentype_cff_font, MSUserFontDirectories, _get_fontconfig_fonts, ttfFontProperty
from matplotlib import cbook, ft2font, pyplot as plt, rc_context, figure as mfigure
has_fclist = shutil.which('fc-list') is not None

def test_font_priority():
    if False:
        print('Hello World!')
    with rc_context(rc={'font.sans-serif': ['cmmi10', 'Bitstream Vera Sans']}):
        fontfile = findfont(FontProperties(family=['sans-serif']))
    assert Path(fontfile).name == 'cmmi10.ttf'
    font = get_font(fontfile)
    cmap = font.get_charmap()
    assert len(cmap) == 131
    assert cmap[8729] == 30

def test_score_weight():
    if False:
        i = 10
        return i + 15
    assert 0 == fontManager.score_weight('regular', 'regular')
    assert 0 == fontManager.score_weight('bold', 'bold')
    assert 0 < fontManager.score_weight(400, 400) < fontManager.score_weight('normal', 'bold')
    assert 0 < fontManager.score_weight('normal', 'regular') < fontManager.score_weight('normal', 'bold')
    assert fontManager.score_weight('normal', 'regular') == fontManager.score_weight(400, 400)

def test_json_serialization(tmpdir):
    if False:
        print('Hello World!')
    path = Path(tmpdir, 'fontlist.json')
    json_dump(fontManager, path)
    copy = json_load(path)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'findfont: Font family.*not found')
        for prop in ({'family': 'STIXGeneral'}, {'family': 'Bitstream Vera Sans', 'weight': 700}, {'family': 'no such font family'}):
            fp = FontProperties(**prop)
            assert fontManager.findfont(fp, rebuild_if_missing=False) == copy.findfont(fp, rebuild_if_missing=False)

def test_otf():
    if False:
        i = 10
        return i + 15
    fname = '/usr/share/fonts/opentype/freefont/FreeMono.otf'
    if Path(fname).exists():
        assert is_opentype_cff_font(fname)
    for f in fontManager.ttflist:
        if 'otf' in f.fname:
            with open(f.fname, 'rb') as fd:
                res = fd.read(4) == b'OTTO'
            assert res == is_opentype_cff_font(f.fname)

@pytest.mark.skipif(sys.platform == 'win32' or not has_fclist, reason='no fontconfig installed')
def test_get_fontconfig_fonts():
    if False:
        return 10
    assert len(_get_fontconfig_fonts()) > 1

@pytest.mark.parametrize('factor', [2, 4, 6, 8])
def test_hinting_factor(factor):
    if False:
        for i in range(10):
            print('nop')
    font = findfont(FontProperties(family=['sans-serif']))
    font1 = get_font(font, hinting_factor=1)
    font1.clear()
    font1.set_size(12, 100)
    font1.set_text('abc')
    expected = font1.get_width_height()
    hinted_font = get_font(font, hinting_factor=factor)
    hinted_font.clear()
    hinted_font.set_size(12, 100)
    hinted_font.set_text('abc')
    np.testing.assert_allclose(hinted_font.get_width_height(), expected, rtol=0.1)

def test_utf16m_sfnt():
    if False:
        return 10
    try:
        entry = next((entry for entry in fontManager.ttflist if Path(entry.fname).name == 'seguisbi.ttf'))
    except StopIteration:
        pytest.skip("Couldn't find seguisbi.ttf font to test against.")
    else:
        assert entry.weight == 600

def test_find_ttc():
    if False:
        print('Hello World!')
    fp = FontProperties(family=['WenQuanYi Zen Hei'])
    if Path(findfont(fp)).name != 'wqy-zenhei.ttc':
        pytest.skip('Font wqy-zenhei.ttc may be missing')
    (fig, ax) = plt.subplots()
    ax.text(0.5, 0.5, '⿓', fontproperties=fp)
    for fmt in ['raw', 'svg', 'pdf', 'ps']:
        fig.savefig(BytesIO(), format=fmt)

def test_find_noto():
    if False:
        i = 10
        return i + 15
    fp = FontProperties(family=['Noto Sans CJK SC', 'Noto Sans CJK JP'])
    name = Path(findfont(fp)).name
    if name not in ('NotoSansCJKsc-Regular.otf', 'NotoSansCJK-Regular.ttc'):
        pytest.skip(f'Noto Sans CJK SC font may be missing (found {name})')
    (fig, ax) = plt.subplots()
    ax.text(0.5, 0.5, 'Hello, 你好', fontproperties=fp)
    for fmt in ['raw', 'svg', 'pdf', 'ps']:
        fig.savefig(BytesIO(), format=fmt)

def test_find_invalid(tmpdir):
    if False:
        while True:
            i = 10
    tmp_path = Path(tmpdir)
    with pytest.raises(FileNotFoundError):
        get_font(tmp_path / 'non-existent-font-name.ttf')
    with pytest.raises(FileNotFoundError):
        get_font(str(tmp_path / 'non-existent-font-name.ttf'))
    with pytest.raises(FileNotFoundError):
        get_font(bytes(tmp_path / 'non-existent-font-name.ttf'))
    from matplotlib.ft2font import FT2Font
    with pytest.raises(TypeError, match='font file or a binary-mode file'):
        FT2Font(StringIO())

@pytest.mark.skipif(sys.platform != 'linux' or not has_fclist, reason='only Linux with fontconfig installed')
def test_user_fonts_linux(tmpdir, monkeypatch):
    if False:
        while True:
            i = 10
    font_test_file = 'mpltest.ttf'
    fonts = findSystemFonts()
    if any((font_test_file in font for font in fonts)):
        pytest.skip(f'{font_test_file} already exists in system fonts')
    user_fonts_dir = tmpdir.join('fonts')
    user_fonts_dir.ensure(dir=True)
    shutil.copyfile(Path(__file__).parent / font_test_file, user_fonts_dir.join(font_test_file))
    with monkeypatch.context() as m:
        m.setenv('XDG_DATA_HOME', str(tmpdir))
        _get_fontconfig_fonts.cache_clear()
        fonts = findSystemFonts()
        assert any((font_test_file in font for font in fonts))
    _get_fontconfig_fonts.cache_clear()

def test_addfont_as_path():
    if False:
        i = 10
        return i + 15
    'Smoke test that addfont() accepts pathlib.Path.'
    font_test_file = 'mpltest.ttf'
    path = Path(__file__).parent / font_test_file
    try:
        fontManager.addfont(path)
        (added,) = [font for font in fontManager.ttflist if font.fname.endswith(font_test_file)]
        fontManager.ttflist.remove(added)
    finally:
        to_remove = [font for font in fontManager.ttflist if font.fname.endswith(font_test_file)]
        for font in to_remove:
            fontManager.ttflist.remove(font)

@pytest.mark.skipif(sys.platform != 'win32', reason='Windows only')
def test_user_fonts_win32():
    if False:
        print('Hello World!')
    if not (os.environ.get('APPVEYOR') or os.environ.get('TF_BUILD')):
        pytest.xfail("This test should only run on CI (appveyor or azure) as the developer's font directory should remain unchanged.")
    pytest.xfail('We need to update the registry for this test to work')
    font_test_file = 'mpltest.ttf'
    fonts = findSystemFonts()
    if any((font_test_file in font for font in fonts)):
        pytest.skip(f'{font_test_file} already exists in system fonts')
    user_fonts_dir = MSUserFontDirectories[0]
    os.makedirs(user_fonts_dir)
    shutil.copy(Path(__file__).parent / font_test_file, user_fonts_dir)
    fonts = findSystemFonts()
    assert any((font_test_file in font for font in fonts))

def _model_handler(_):
    if False:
        while True:
            i = 10
    (fig, ax) = plt.subplots()
    fig.savefig(BytesIO(), format='pdf')
    plt.close()

@pytest.mark.skipif(not hasattr(os, 'register_at_fork'), reason='Cannot register at_fork handlers')
def test_fork():
    if False:
        i = 10
        return i + 15
    _model_handler(0)
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(processes=2) as pool:
        pool.map(_model_handler, range(2))

def test_missing_family(caplog):
    if False:
        return 10
    plt.rcParams['font.sans-serif'] = ['this-font-does-not-exist']
    with caplog.at_level('WARNING'):
        findfont('sans')
    assert [rec.getMessage() for rec in caplog.records] == ["findfont: Font family ['sans'] not found. Falling back to DejaVu Sans.", "findfont: Generic family 'sans' not found because none of the following families were found: this-font-does-not-exist"]

def _test_threading():
    if False:
        return 10
    import threading
    from matplotlib.ft2font import LOAD_NO_HINTING
    import matplotlib.font_manager as fm
    N = 10
    b = threading.Barrier(N)

    def bad_idea(n):
        if False:
            for i in range(10):
                print('nop')
        b.wait()
        for j in range(100):
            font = fm.get_font(fm.findfont('DejaVu Sans'))
            font.set_text(str(n), 0.0, flags=LOAD_NO_HINTING)
    threads = [threading.Thread(target=bad_idea, name=f'bad_thread_{j}', args=(j,)) for j in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

def test_fontcache_thread_safe():
    if False:
        return 10
    pytest.importorskip('threading')
    import inspect
    proc = subprocess.run([sys.executable, '-c', inspect.getsource(_test_threading) + '\n_test_threading()'])
    if proc.returncode:
        pytest.fail(f'The subprocess returned with non-zero exit status {proc.returncode}.')

def test_fontentry_dataclass():
    if False:
        print('Hello World!')
    fontent = FontEntry(name='font-name')
    png = fontent._repr_png_()
    img = Image.open(BytesIO(png))
    assert img.width > 0
    assert img.height > 0
    html = fontent._repr_html_()
    assert html.startswith('<img src="data:image/png;base64')

def test_fontentry_dataclass_invalid_path():
    if False:
        return 10
    with pytest.raises(FileNotFoundError):
        fontent = FontEntry(fname='/random', name='font-name')
        fontent._repr_html_()

@pytest.mark.skipif(sys.platform == 'win32', reason='Linux or OS only')
def test_get_font_names():
    if False:
        return 10
    paths_mpl = [cbook._get_data_path('fonts', subdir) for subdir in ['ttf']]
    fonts_mpl = findSystemFonts(paths_mpl, fontext='ttf')
    fonts_system = findSystemFonts(fontext='ttf')
    ttf_fonts = []
    for path in fonts_mpl + fonts_system:
        try:
            font = ft2font.FT2Font(path)
            prop = ttfFontProperty(font)
            ttf_fonts.append(prop.name)
        except Exception:
            pass
    available_fonts = sorted(list(set(ttf_fonts)))
    mpl_font_names = sorted(fontManager.get_font_names())
    assert set(available_fonts) == set(mpl_font_names)
    assert len(available_fonts) == len(mpl_font_names)
    assert available_fonts == mpl_font_names

def test_donot_cache_tracebacks():
    if False:
        print('Hello World!')

    class SomeObject:
        pass

    def inner():
        if False:
            i = 10
            return i + 15
        x = SomeObject()
        fig = mfigure.Figure()
        ax = fig.subplots()
        fig.text(0.5, 0.5, 'aardvark', family='doesnotexist')
        with BytesIO() as out:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                fig.savefig(out, format='raw')
    inner()
    for obj in gc.get_objects():
        if isinstance(obj, SomeObject):
            pytest.fail('object from inner stack still alive')