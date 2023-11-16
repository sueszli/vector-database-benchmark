import pytest
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.testing as npt
from seaborn import rcmod, palettes, utils

def has_verdana():
    if False:
        while True:
            i = 10
    'Helper to verify if Verdana font is present'
    import matplotlib.font_manager as mplfm
    try:
        verdana_font = mplfm.findfont('Verdana', fallback_to_default=False)
    except:
        return False
    try:
        unlikely_font = mplfm.findfont('very_unlikely_to_exist1234', fallback_to_default=False)
    except:
        return True
    return verdana_font != unlikely_font

class RCParamFixtures:

    @pytest.fixture(autouse=True)
    def reset_params(self):
        if False:
            while True:
                i = 10
        yield
        rcmod.reset_orig()

    def flatten_list(self, orig_list):
        if False:
            i = 10
            return i + 15
        iter_list = map(np.atleast_1d, orig_list)
        flat_list = [item for sublist in iter_list for item in sublist]
        return flat_list

    def assert_rc_params(self, params):
        if False:
            for i in range(10):
                print('nop')
        for (k, v) in params.items():
            if k == 'backend':
                continue
            if isinstance(v, np.ndarray):
                npt.assert_array_equal(mpl.rcParams[k], v)
            else:
                assert mpl.rcParams[k] == v

    def assert_rc_params_equal(self, params1, params2):
        if False:
            for i in range(10):
                print('nop')
        for (key, v1) in params1.items():
            if key == 'backend':
                continue
            v2 = params2[key]
            if isinstance(v1, np.ndarray):
                npt.assert_array_equal(v1, v2)
            else:
                assert v1 == v2

class TestAxesStyle(RCParamFixtures):
    styles = ['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']

    def test_default_return(self):
        if False:
            print('Hello World!')
        current = rcmod.axes_style()
        self.assert_rc_params(current)

    def test_key_usage(self):
        if False:
            for i in range(10):
                print('nop')
        _style_keys = set(rcmod._style_keys)
        for style in self.styles:
            assert not set(rcmod.axes_style(style)) ^ _style_keys

    def test_bad_style(self):
        if False:
            return 10
        with pytest.raises(ValueError):
            rcmod.axes_style('i_am_not_a_style')

    def test_rc_override(self):
        if False:
            i = 10
            return i + 15
        rc = {'axes.facecolor': 'blue', 'foo.notaparam': 'bar'}
        out = rcmod.axes_style('darkgrid', rc)
        assert out['axes.facecolor'] == 'blue'
        assert 'foo.notaparam' not in out

    def test_set_style(self):
        if False:
            i = 10
            return i + 15
        for style in self.styles:
            style_dict = rcmod.axes_style(style)
            rcmod.set_style(style)
            self.assert_rc_params(style_dict)

    def test_style_context_manager(self):
        if False:
            print('Hello World!')
        rcmod.set_style('darkgrid')
        orig_params = rcmod.axes_style()
        context_params = rcmod.axes_style('whitegrid')
        with rcmod.axes_style('whitegrid'):
            self.assert_rc_params(context_params)
        self.assert_rc_params(orig_params)

        @rcmod.axes_style('whitegrid')
        def func():
            if False:
                for i in range(10):
                    print('nop')
            self.assert_rc_params(context_params)
        func()
        self.assert_rc_params(orig_params)

    def test_style_context_independence(self):
        if False:
            i = 10
            return i + 15
        assert set(rcmod._style_keys) ^ set(rcmod._context_keys)

    def test_set_rc(self):
        if False:
            while True:
                i = 10
        rcmod.set_theme(rc={'lines.linewidth': 4})
        assert mpl.rcParams['lines.linewidth'] == 4
        rcmod.set_theme()

    def test_set_with_palette(self):
        if False:
            for i in range(10):
                print('nop')
        rcmod.reset_orig()
        rcmod.set_theme(palette='deep')
        assert utils.get_color_cycle() == palettes.color_palette('deep', 10)
        rcmod.reset_orig()
        rcmod.set_theme(palette='deep', color_codes=False)
        assert utils.get_color_cycle() == palettes.color_palette('deep', 10)
        rcmod.reset_orig()
        pal = palettes.color_palette('deep')
        rcmod.set_theme(palette=pal)
        assert utils.get_color_cycle() == palettes.color_palette('deep', 10)
        rcmod.reset_orig()
        rcmod.set_theme(palette=pal, color_codes=False)
        assert utils.get_color_cycle() == palettes.color_palette('deep', 10)
        rcmod.reset_orig()
        rcmod.set_theme()

    def test_reset_defaults(self):
        if False:
            while True:
                i = 10
        rcmod.reset_defaults()
        self.assert_rc_params(mpl.rcParamsDefault)
        rcmod.set_theme()

    def test_reset_orig(self):
        if False:
            i = 10
            return i + 15
        rcmod.reset_orig()
        self.assert_rc_params(mpl.rcParamsOrig)
        rcmod.set_theme()

    def test_set_is_alias(self):
        if False:
            while True:
                i = 10
        rcmod.set_theme(context='paper', style='white')
        params1 = mpl.rcParams.copy()
        rcmod.reset_orig()
        rcmod.set_theme(context='paper', style='white')
        params2 = mpl.rcParams.copy()
        self.assert_rc_params_equal(params1, params2)
        rcmod.set_theme()

class TestPlottingContext(RCParamFixtures):
    contexts = ['paper', 'notebook', 'talk', 'poster']

    def test_default_return(self):
        if False:
            print('Hello World!')
        current = rcmod.plotting_context()
        self.assert_rc_params(current)

    def test_key_usage(self):
        if False:
            print('Hello World!')
        _context_keys = set(rcmod._context_keys)
        for context in self.contexts:
            missing = set(rcmod.plotting_context(context)) ^ _context_keys
            assert not missing

    def test_bad_context(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError):
            rcmod.plotting_context('i_am_not_a_context')

    def test_font_scale(self):
        if False:
            i = 10
            return i + 15
        notebook_ref = rcmod.plotting_context('notebook')
        notebook_big = rcmod.plotting_context('notebook', 2)
        font_keys = ['font.size', 'axes.labelsize', 'axes.titlesize', 'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize', 'legend.title_fontsize']
        for k in font_keys:
            assert notebook_ref[k] * 2 == notebook_big[k]

    def test_rc_override(self):
        if False:
            while True:
                i = 10
        (key, val) = ('grid.linewidth', 5)
        rc = {key: val, 'foo': 'bar'}
        out = rcmod.plotting_context('talk', rc=rc)
        assert out[key] == val
        assert 'foo' not in out

    def test_set_context(self):
        if False:
            for i in range(10):
                print('nop')
        for context in self.contexts:
            context_dict = rcmod.plotting_context(context)
            rcmod.set_context(context)
            self.assert_rc_params(context_dict)

    def test_context_context_manager(self):
        if False:
            for i in range(10):
                print('nop')
        rcmod.set_context('notebook')
        orig_params = rcmod.plotting_context()
        context_params = rcmod.plotting_context('paper')
        with rcmod.plotting_context('paper'):
            self.assert_rc_params(context_params)
        self.assert_rc_params(orig_params)

        @rcmod.plotting_context('paper')
        def func():
            if False:
                return 10
            self.assert_rc_params(context_params)
        func()
        self.assert_rc_params(orig_params)

class TestPalette(RCParamFixtures):

    def test_set_palette(self):
        if False:
            print('Hello World!')
        rcmod.set_palette('deep')
        assert utils.get_color_cycle() == palettes.color_palette('deep', 10)
        rcmod.set_palette('pastel6')
        assert utils.get_color_cycle() == palettes.color_palette('pastel6', 6)
        rcmod.set_palette('dark', 4)
        assert utils.get_color_cycle() == palettes.color_palette('dark', 4)
        rcmod.set_palette('Set2', color_codes=True)
        assert utils.get_color_cycle() == palettes.color_palette('Set2', 8)
        assert mpl.colors.same_color(mpl.rcParams['patch.facecolor'], palettes.color_palette()[0])

class TestFonts(RCParamFixtures):
    _no_verdana = not has_verdana()

    @pytest.mark.skipif(_no_verdana, reason='Verdana font is not present')
    def test_set_font(self):
        if False:
            for i in range(10):
                print('nop')
        rcmod.set_theme(font='Verdana')
        (_, ax) = plt.subplots()
        ax.set_xlabel('foo')
        assert ax.xaxis.label.get_fontname() == 'Verdana'
        rcmod.set_theme()

    def test_set_serif_font(self):
        if False:
            i = 10
            return i + 15
        rcmod.set_theme(font='serif')
        (_, ax) = plt.subplots()
        ax.set_xlabel('foo')
        assert ax.xaxis.label.get_fontname() in mpl.rcParams['font.serif']
        rcmod.set_theme()

    @pytest.mark.skipif(_no_verdana, reason='Verdana font is not present')
    def test_different_sans_serif(self):
        if False:
            while True:
                i = 10
        rcmod.set_theme()
        rcmod.set_style(rc={'font.sans-serif': ['Verdana']})
        (_, ax) = plt.subplots()
        ax.set_xlabel('foo')
        assert ax.xaxis.label.get_fontname() == 'Verdana'
        rcmod.set_theme()