import copy
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock
from cycler import cycler, Cycler
import pytest
import matplotlib as mpl
from matplotlib import _api, _c_internal_utils
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.rcsetup import validate_bool, validate_color, validate_colorlist, _validate_color_or_linecolor, validate_cycler, validate_float, validate_fontstretch, validate_fontweight, validate_hatch, validate_hist_bins, validate_int, validate_markevery, validate_stringlist, validate_sketch, _validate_linestyle, _listify_validator

def test_rcparams(tmpdir):
    if False:
        print('Hello World!')
    mpl.rc('text', usetex=False)
    mpl.rc('lines', linewidth=22)
    usetex = mpl.rcParams['text.usetex']
    linewidth = mpl.rcParams['lines.linewidth']
    rcpath = Path(tmpdir) / 'test_rcparams.rc'
    rcpath.write_text('lines.linewidth: 33', encoding='utf-8')
    with mpl.rc_context(rc={'text.usetex': not usetex}):
        assert mpl.rcParams['text.usetex'] == (not usetex)
    assert mpl.rcParams['text.usetex'] == usetex
    with mpl.rc_context(fname=rcpath):
        assert mpl.rcParams['lines.linewidth'] == 33
    assert mpl.rcParams['lines.linewidth'] == linewidth
    with mpl.rc_context(fname=rcpath, rc={'lines.linewidth': 44}):
        assert mpl.rcParams['lines.linewidth'] == 44
    assert mpl.rcParams['lines.linewidth'] == linewidth

    @mpl.rc_context({'lines.linewidth': 44})
    def func():
        if False:
            for i in range(10):
                print('nop')
        assert mpl.rcParams['lines.linewidth'] == 44
    func()
    func()
    mpl.rc_file(rcpath)
    assert mpl.rcParams['lines.linewidth'] == 33

def test_RcParams_class():
    if False:
        while True:
            i = 10
    rc = mpl.RcParams({'font.cursive': ['Apple Chancery', 'Textile', 'Zapf Chancery', 'cursive'], 'font.family': 'sans-serif', 'font.weight': 'normal', 'font.size': 12})
    expected_repr = "\nRcParams({'font.cursive': ['Apple Chancery',\n                           'Textile',\n                           'Zapf Chancery',\n                           'cursive'],\n          'font.family': ['sans-serif'],\n          'font.size': 12.0,\n          'font.weight': 'normal'})".lstrip()
    assert expected_repr == repr(rc)
    expected_str = "\nfont.cursive: ['Apple Chancery', 'Textile', 'Zapf Chancery', 'cursive']\nfont.family: ['sans-serif']\nfont.size: 12.0\nfont.weight: normal".lstrip()
    assert expected_str == str(rc)
    assert ['font.cursive', 'font.size'] == sorted(rc.find_all('i[vz]'))
    assert ['font.family'] == list(rc.find_all('family'))

def test_rcparams_update():
    if False:
        while True:
            i = 10
    rc = mpl.RcParams({'figure.figsize': (3.5, 42)})
    bad_dict = {'figure.figsize': (3.5, 42, 1)}
    with pytest.raises(ValueError), pytest.warns(UserWarning, match='validate'):
        rc.update(bad_dict)

def test_rcparams_init():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError), pytest.warns(UserWarning, match='validate'):
        mpl.RcParams({'figure.figsize': (3.5, 42, 1)})

def test_nargs_cycler():
    if False:
        return 10
    from matplotlib.rcsetup import cycler as ccl
    with pytest.raises(TypeError, match='3 were given'):
        ccl(ccl(color=list('rgb')), 2, 3)

def test_Bug_2543():
    if False:
        print('Hello World!')
    with _api.suppress_matplotlib_deprecation_warning():
        with mpl.rc_context():
            _copy = mpl.rcParams.copy()
            for key in _copy:
                mpl.rcParams[key] = _copy[key]
        with mpl.rc_context():
            copy.deepcopy(mpl.rcParams)
    with pytest.raises(ValueError):
        validate_bool(None)
    with pytest.raises(ValueError):
        with mpl.rc_context():
            mpl.rcParams['svg.fonttype'] = True
legend_color_tests = [('face', {'color': 'r'}, mcolors.to_rgba('r')), ('face', {'color': 'inherit', 'axes.facecolor': 'r'}, mcolors.to_rgba('r')), ('face', {'color': 'g', 'axes.facecolor': 'r'}, mcolors.to_rgba('g')), ('edge', {'color': 'r'}, mcolors.to_rgba('r')), ('edge', {'color': 'inherit', 'axes.edgecolor': 'r'}, mcolors.to_rgba('r')), ('edge', {'color': 'g', 'axes.facecolor': 'r'}, mcolors.to_rgba('g'))]
legend_color_test_ids = ['same facecolor', 'inherited facecolor', 'different facecolor', 'same edgecolor', 'inherited edgecolor', 'different facecolor']

@pytest.mark.parametrize('color_type, param_dict, target', legend_color_tests, ids=legend_color_test_ids)
def test_legend_colors(color_type, param_dict, target):
    if False:
        while True:
            i = 10
    param_dict[f'legend.{color_type}color'] = param_dict.pop('color')
    get_func = f'get_{color_type}color'
    with mpl.rc_context(param_dict):
        (_, ax) = plt.subplots()
        ax.plot(range(3), label='test')
        leg = ax.legend()
        assert getattr(leg.legendPatch, get_func)() == target

def test_mfc_rcparams():
    if False:
        for i in range(10):
            print('nop')
    mpl.rcParams['lines.markerfacecolor'] = 'r'
    ln = mpl.lines.Line2D([1, 2], [1, 2])
    assert ln.get_markerfacecolor() == 'r'

def test_mec_rcparams():
    if False:
        print('Hello World!')
    mpl.rcParams['lines.markeredgecolor'] = 'r'
    ln = mpl.lines.Line2D([1, 2], [1, 2])
    assert ln.get_markeredgecolor() == 'r'

def test_axes_titlecolor_rcparams():
    if False:
        for i in range(10):
            print('nop')
    mpl.rcParams['axes.titlecolor'] = 'r'
    (_, ax) = plt.subplots()
    title = ax.set_title('Title')
    assert title.get_color() == 'r'

def test_Issue_1713(tmpdir):
    if False:
        return 10
    rcpath = Path(tmpdir) / 'test_rcparams.rc'
    rcpath.write_text('timezone: UTC', encoding='utf-8')
    with mock.patch('locale.getpreferredencoding', return_value='UTF-32-BE'):
        rc = mpl.rc_params_from_file(rcpath, True, False)
    assert rc.get('timezone') == 'UTC'

def test_animation_frame_formats():
    if False:
        for i in range(10):
            print('nop')
    for fmt in ['png', 'jpeg', 'tiff', 'raw', 'rgba', 'ppm', 'sgi', 'bmp', 'pbm', 'svg']:
        mpl.rcParams['animation.frame_format'] = fmt

def generate_validator_testcases(valid):
    if False:
        print('Hello World!')
    validation_tests = ({'validator': validate_bool, 'success': (*((_, True) for _ in ('t', 'y', 'yes', 'on', 'true', '1', 1, True)), *((_, False) for _ in ('f', 'n', 'no', 'off', 'false', '0', 0, False))), 'fail': ((_, ValueError) for _ in ('aardvark', 2, -1, []))}, {'validator': validate_stringlist, 'success': (('', []), ('a,b', ['a', 'b']), ('aardvark', ['aardvark']), ('aardvark, ', ['aardvark']), ('aardvark, ,', ['aardvark']), (['a', 'b'], ['a', 'b']), (('a', 'b'), ['a', 'b']), (iter(['a', 'b']), ['a', 'b']), (np.array(['a', 'b']), ['a', 'b'])), 'fail': ((set(), ValueError), (1, ValueError))}, {'validator': _listify_validator(validate_int, n=2), 'success': ((_, [1, 2]) for _ in ('1, 2', [1.5, 2.5], [1, 2], (1, 2), np.array((1, 2)))), 'fail': ((_, ValueError) for _ in ('aardvark', ('a', 1), (1, 2, 3)))}, {'validator': _listify_validator(validate_float, n=2), 'success': ((_, [1.5, 2.5]) for _ in ('1.5, 2.5', [1.5, 2.5], [1.5, 2.5], (1.5, 2.5), np.array((1.5, 2.5)))), 'fail': ((_, ValueError) for _ in ('aardvark', ('a', 1), (1, 2, 3), (None,), None))}, {'validator': validate_cycler, 'success': (('cycler("color", "rgb")', cycler('color', 'rgb')), (cycler('linestyle', ['-', '--']), cycler('linestyle', ['-', '--'])), ('(cycler("color", ["r", "g", "b"]) +\n                          cycler("mew", [2, 3, 5]))', cycler('color', 'rgb') + cycler('markeredgewidth', [2, 3, 5])), ("cycler(c='rgb', lw=[1, 2, 3])", cycler('color', 'rgb') + cycler('linewidth', [1, 2, 3])), ("cycler('c', 'rgb') * cycler('linestyle', ['-', '--'])", cycler('color', 'rgb') * cycler('linestyle', ['-', '--'])), (cycler('ls', ['-', '--']), cycler('linestyle', ['-', '--'])), (cycler(mew=[2, 5]), cycler('markeredgewidth', [2, 5]))), 'fail': ((4, ValueError), ('cycler("bleh, [])', ValueError), ('Cycler("linewidth", [1, 2, 3])', ValueError), ("cycler('c', [j.__class__(j) for j in ['r', 'b']])", ValueError), ("cycler('c', [j. __class__(j) for j in ['r', 'b']])", ValueError), ("cycler('c', [j.\t__class__(j) for j in ['r', 'b']])", ValueError), ("cycler('c', [j.\x0c__class__(j) for j in ['r', 'b']])", ValueError), ("cycler('c', [j.__class__(j).lower() for j in ['r', 'b']])", ValueError), ('1 + 2', ValueError), ('os.system("echo Gotcha")', ValueError), ('import os', ValueError), ('def badjuju(a): return a; badjuju(cycler("color", "rgb"))', ValueError), ('cycler("waka", [1, 2, 3])', ValueError), ('cycler(c=[1, 2, 3])', ValueError), ("cycler(lw=['a', 'b', 'c'])", ValueError), (cycler('waka', [1, 3, 5]), ValueError), (cycler('color', ['C1', 'r', 'g']), ValueError))}, {'validator': validate_hatch, 'success': (('--|', '--|'), ('\\oO', '\\oO'), ('/+*/.x', '/+*/.x'), ('', '')), 'fail': (('--_', ValueError), (8, ValueError), ('X', ValueError))}, {'validator': validate_colorlist, 'success': (('r,g,b', ['r', 'g', 'b']), (['r', 'g', 'b'], ['r', 'g', 'b']), ('r, ,', ['r']), (['', 'g', 'blue'], ['g', 'blue']), ([np.array([1, 0, 0]), np.array([0, 1, 0])], np.array([[1, 0, 0], [0, 1, 0]])), (np.array([[1, 0, 0], [0, 1, 0]]), np.array([[1, 0, 0], [0, 1, 0]]))), 'fail': (('fish', ValueError),)}, {'validator': validate_color, 'success': (('None', 'none'), ('none', 'none'), ('AABBCC', '#AABBCC'), ('AABBCC00', '#AABBCC00'), ('tab:blue', 'tab:blue'), ('C12', 'C12'), ('(0, 1, 0)', (0.0, 1.0, 0.0)), ((0, 1, 0), (0, 1, 0)), ('(0, 1, 0, 1)', (0.0, 1.0, 0.0, 1.0)), ((0, 1, 0, 1), (0, 1, 0, 1))), 'fail': (('tab:veryblue', ValueError), ('(0, 1)', ValueError), ('(0, 1, 0, 1, 0)', ValueError), ('(0, 1, none)', ValueError), ('(0, 1, "0.5")', ValueError))}, {'validator': _validate_color_or_linecolor, 'success': (('linecolor', 'linecolor'), ('markerfacecolor', 'markerfacecolor'), ('mfc', 'markerfacecolor'), ('markeredgecolor', 'markeredgecolor'), ('mec', 'markeredgecolor')), 'fail': (('line', ValueError), ('marker', ValueError))}, {'validator': validate_hist_bins, 'success': (('auto', 'auto'), ('fd', 'fd'), ('10', 10), ('1, 2, 3', [1, 2, 3]), ([1, 2, 3], [1, 2, 3]), (np.arange(15), np.arange(15))), 'fail': (('aardvark', ValueError),)}, {'validator': validate_markevery, 'success': ((None, None), (1, 1), (0.1, 0.1), ((1, 1), (1, 1)), ((0.1, 0.1), (0.1, 0.1)), ([1, 2, 3], [1, 2, 3]), (slice(2), slice(None, 2, None)), (slice(1, 2, 3), slice(1, 2, 3))), 'fail': (((1, 2, 3), TypeError), ([1, 2, 0.3], TypeError), (['a', 2, 3], TypeError), ([1, 2, 'a'], TypeError), ((0.1, 0.2, 0.3), TypeError), ((0.1, 2, 3), TypeError), ((1, 0.2, 0.3), TypeError), ((1, 0.1), TypeError), ((0.1, 1), TypeError), ('abc', TypeError), ((1, 'a'), TypeError), ((0.1, 'b'), TypeError), (('a', 1), TypeError), (('a', 0.1), TypeError), ('abc', TypeError), ('a', TypeError), (object(), TypeError))}, {'validator': _validate_linestyle, 'success': (('-', '-'), ('solid', 'solid'), ('--', '--'), ('dashed', 'dashed'), ('-.', '-.'), ('dashdot', 'dashdot'), (':', ':'), ('dotted', 'dotted'), ('', ''), (' ', ' '), ('None', 'none'), ('none', 'none'), ('DoTtEd', 'dotted'), ('1, 3', (0, (1, 3))), ([1.23, 456], (0, [1.23, 456.0])), ([1, 2, 3, 4], (0, [1.0, 2.0, 3.0, 4.0])), ((0, [1, 2]), (0, [1, 2])), ((-1, [1, 2]), (-1, [1, 2]))), 'fail': (('aardvark', ValueError), (b'dotted', ValueError), ('dotted'.encode('utf-16'), ValueError), ([1, 2, 3], ValueError), (1.23, ValueError), (('a', [1, 2]), ValueError), ((None, [1, 2]), ValueError), ((1, [1, 2, 3]), ValueError), (([1, 2], 1), ValueError))})
    for validator_dict in validation_tests:
        validator = validator_dict['validator']
        if valid:
            for (arg, target) in validator_dict['success']:
                yield (validator, arg, target)
        else:
            for (arg, error_type) in validator_dict['fail']:
                yield (validator, arg, error_type)

@pytest.mark.parametrize('validator, arg, target', generate_validator_testcases(True))
def test_validator_valid(validator, arg, target):
    if False:
        i = 10
        return i + 15
    res = validator(arg)
    if isinstance(target, np.ndarray):
        np.testing.assert_equal(res, target)
    elif not isinstance(target, Cycler):
        assert res == target
    else:
        assert list(res) == list(target)

@pytest.mark.parametrize('validator, arg, exception_type', generate_validator_testcases(False))
def test_validator_invalid(validator, arg, exception_type):
    if False:
        while True:
            i = 10
    with pytest.raises(exception_type):
        validator(arg)

@pytest.mark.parametrize('weight, parsed_weight', [('bold', 'bold'), ('BOLD', ValueError), (100, 100), ('100', 100), (np.array(100), 100), (20.6, 20), ('20.6', ValueError), ([100], ValueError)])
def test_validate_fontweight(weight, parsed_weight):
    if False:
        return 10
    if parsed_weight is ValueError:
        with pytest.raises(ValueError):
            validate_fontweight(weight)
    else:
        assert validate_fontweight(weight) == parsed_weight

@pytest.mark.parametrize('stretch, parsed_stretch', [('expanded', 'expanded'), ('EXPANDED', ValueError), (100, 100), ('100', 100), (np.array(100), 100), (20.6, 20), ('20.6', ValueError), ([100], ValueError)])
def test_validate_fontstretch(stretch, parsed_stretch):
    if False:
        return 10
    if parsed_stretch is ValueError:
        with pytest.raises(ValueError):
            validate_fontstretch(stretch)
    else:
        assert validate_fontstretch(stretch) == parsed_stretch

def test_keymaps():
    if False:
        i = 10
        return i + 15
    key_list = [k for k in mpl.rcParams if 'keymap' in k]
    for k in key_list:
        assert isinstance(mpl.rcParams[k], list)

def test_no_backend_reset_rccontext():
    if False:
        print('Hello World!')
    assert mpl.rcParams['backend'] != 'module://aardvark'
    with mpl.rc_context():
        mpl.rcParams['backend'] = 'module://aardvark'
    assert mpl.rcParams['backend'] == 'module://aardvark'

def test_rcparams_reset_after_fail():
    if False:
        while True:
            i = 10
    with mpl.rc_context(rc={'text.usetex': False}):
        assert mpl.rcParams['text.usetex'] is False
        with pytest.raises(KeyError):
            with mpl.rc_context(rc={'text.usetex': True, 'test.blah': True}):
                pass
        assert mpl.rcParams['text.usetex'] is False

@pytest.mark.skipif(sys.platform != 'linux', reason='Linux only')
def test_backend_fallback_headless(tmpdir):
    if False:
        return 10
    env = {**os.environ, 'DISPLAY': '', 'WAYLAND_DISPLAY': '', 'MPLBACKEND': '', 'MPLCONFIGDIR': str(tmpdir)}
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([sys.executable, '-c', "import matplotlib;matplotlib.use('tkagg');import matplotlib.pyplot;matplotlib.pyplot.plot(42);"], env=env, check=True, stderr=subprocess.DEVNULL)

@pytest.mark.skipif(sys.platform == 'linux' and (not _c_internal_utils.display_is_valid()), reason='headless')
def test_backend_fallback_headful(tmpdir):
    if False:
        return 10
    pytest.importorskip('tkinter')
    env = {**os.environ, 'MPLBACKEND': '', 'MPLCONFIGDIR': str(tmpdir)}
    backend = subprocess.check_output([sys.executable, '-c', "import matplotlib as mpl; sentinel = mpl.rcsetup._auto_backend_sentinel; assert mpl.RcParams({'backend': sentinel})['backend'] == sentinel; assert mpl.rcParams._get('backend') == sentinel; import matplotlib.pyplot; print(matplotlib.get_backend())"], env=env, text=True)
    assert backend.strip().lower() != 'agg'

def test_deprecation(monkeypatch):
    if False:
        i = 10
        return i + 15
    monkeypatch.setitem(mpl._deprecated_map, 'patch.linewidth', ('0.0', 'axes.linewidth', lambda old: 2 * old, lambda new: new / 2))
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        assert mpl.rcParams['patch.linewidth'] == mpl.rcParams['axes.linewidth'] / 2
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        mpl.rcParams['patch.linewidth'] = 1
    assert mpl.rcParams['axes.linewidth'] == 2
    monkeypatch.setitem(mpl._deprecated_ignore_map, 'patch.edgecolor', ('0.0', 'axes.edgecolor'))
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        assert mpl.rcParams['patch.edgecolor'] == mpl.rcParams['axes.edgecolor']
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        mpl.rcParams['patch.edgecolor'] = '#abcd'
    assert mpl.rcParams['axes.edgecolor'] != '#abcd'
    monkeypatch.setitem(mpl._deprecated_ignore_map, 'patch.force_edgecolor', ('0.0', None))
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        assert mpl.rcParams['patch.force_edgecolor'] is None
    monkeypatch.setitem(mpl._deprecated_remain_as_none, 'svg.hashsalt', ('0.0',))
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        mpl.rcParams['svg.hashsalt'] = 'foobar'
    assert mpl.rcParams['svg.hashsalt'] == 'foobar'
    mpl.rcParams['svg.hashsalt'] = None
    mpl.rcParams.update(mpl.rcParams.copy())

@pytest.mark.parametrize('value', ['best', 1, '1', (0.9, 0.7), (-0.9, 0.7), '(0.9, .7)'])
def test_rcparams_legend_loc(value):
    if False:
        print('Hello World!')
    mpl.rcParams['legend.loc'] = value

@pytest.mark.parametrize('value', ['best', 1, (0.9, 0.7), (-0.9, 0.7)])
def test_rcparams_legend_loc_from_file(tmpdir, value):
    if False:
        print('Hello World!')
    rc_path = tmpdir.join('matplotlibrc')
    rc_path.write(f'legend.loc: {value}')
    with mpl.rc_context(fname=rc_path):
        assert mpl.rcParams['legend.loc'] == value

@pytest.mark.parametrize('value', [(1, 2, 3), '1, 2, 3', '(1, 2, 3)'])
def test_validate_sketch(value):
    if False:
        for i in range(10):
            print('nop')
    mpl.rcParams['path.sketch'] = value
    assert mpl.rcParams['path.sketch'] == (1, 2, 3)
    assert validate_sketch(value) == (1, 2, 3)

@pytest.mark.parametrize('value', [1, '1', '1 2 3'])
def test_validate_sketch_error(value):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError, match='scale, length, randomness'):
        validate_sketch(value)
    with pytest.raises(ValueError, match='scale, length, randomness'):
        mpl.rcParams['path.sketch'] = value

@pytest.mark.parametrize('value', ['1, 2, 3', '(1,2,3)'])
def test_rcparams_path_sketch_from_file(tmpdir, value):
    if False:
        print('Hello World!')
    rc_path = tmpdir.join('matplotlibrc')
    rc_path.write(f'path.sketch: {value}')
    with mpl.rc_context(fname=rc_path):
        assert mpl.rcParams['path.sketch'] == (1, 2, 3)