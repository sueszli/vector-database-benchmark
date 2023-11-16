import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal
from seaborn._core.plot import Plot
from seaborn._core.scales import Nominal, Continuous, Boolean, Temporal, PseudoAxis
from seaborn._core.properties import IntervalProperty, ObjectProperty, Coordinate, Alpha, Color, Fill
from seaborn.palettes import color_palette
from seaborn.utils import _version_predates

class TestContinuous:

    @pytest.fixture
    def x(self):
        if False:
            while True:
                i = 10
        return pd.Series([1, 3, 9], name='x', dtype=float)

    def setup_ticks(self, x, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        s = Continuous().tick(*args, **kwargs)._setup(x, Coordinate())
        a = PseudoAxis(s._matplotlib_scale)
        a.set_view_interval(0, 1)
        return a

    def setup_labels(self, x, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        s = Continuous().label(*args, **kwargs)._setup(x, Coordinate())
        a = PseudoAxis(s._matplotlib_scale)
        a.set_view_interval(0, 1)
        locs = a.major.locator()
        return (a, locs)

    def test_coordinate_defaults(self, x):
        if False:
            print('Hello World!')
        s = Continuous()._setup(x, Coordinate())
        assert_series_equal(s(x), x)

    def test_coordinate_transform(self, x):
        if False:
            i = 10
            return i + 15
        s = Continuous(trans='log')._setup(x, Coordinate())
        assert_series_equal(s(x), np.log10(x))

    def test_coordinate_transform_with_parameter(self, x):
        if False:
            print('Hello World!')
        s = Continuous(trans='pow3')._setup(x, Coordinate())
        assert_series_equal(s(x), np.power(x, 3))

    def test_coordinate_transform_error(self, x):
        if False:
            for i in range(10):
                print('nop')
        s = Continuous(trans='bad')
        with pytest.raises(ValueError, match='Unknown value provided'):
            s._setup(x, Coordinate())

    def test_interval_defaults(self, x):
        if False:
            while True:
                i = 10
        s = Continuous()._setup(x, IntervalProperty())
        assert_array_equal(s(x), [0, 0.25, 1])

    def test_interval_with_range(self, x):
        if False:
            while True:
                i = 10
        s = Continuous((1, 3))._setup(x, IntervalProperty())
        assert_array_equal(s(x), [1, 1.5, 3])

    def test_interval_with_norm(self, x):
        if False:
            return 10
        s = Continuous(norm=(3, 7))._setup(x, IntervalProperty())
        assert_array_equal(s(x), [-0.5, 0, 1.5])

    def test_interval_with_range_norm_and_transform(self, x):
        if False:
            i = 10
            return i + 15
        x = pd.Series([1, 10, 100])
        s = Continuous((2, 3), (10, 100), 'log')._setup(x, IntervalProperty())
        assert_array_equal(s(x), [1, 2, 3])

    def test_interval_with_bools(self):
        if False:
            return 10
        x = pd.Series([True, False, False])
        s = Continuous()._setup(x, IntervalProperty())
        assert_array_equal(s(x), [1, 0, 0])

    def test_color_defaults(self, x):
        if False:
            print('Hello World!')
        cmap = color_palette('ch:', as_cmap=True)
        s = Continuous()._setup(x, Color())
        assert_array_equal(s(x), cmap([0, 0.25, 1])[:, :3])

    def test_color_named_values(self, x):
        if False:
            return 10
        cmap = color_palette('viridis', as_cmap=True)
        s = Continuous('viridis')._setup(x, Color())
        assert_array_equal(s(x), cmap([0, 0.25, 1])[:, :3])

    def test_color_tuple_values(self, x):
        if False:
            while True:
                i = 10
        cmap = color_palette('blend:b,g', as_cmap=True)
        s = Continuous(('b', 'g'))._setup(x, Color())
        assert_array_equal(s(x), cmap([0, 0.25, 1])[:, :3])

    def test_color_callable_values(self, x):
        if False:
            for i in range(10):
                print('nop')
        cmap = color_palette('light:r', as_cmap=True)
        s = Continuous(cmap)._setup(x, Color())
        assert_array_equal(s(x), cmap([0, 0.25, 1])[:, :3])

    def test_color_with_norm(self, x):
        if False:
            while True:
                i = 10
        cmap = color_palette('ch:', as_cmap=True)
        s = Continuous(norm=(3, 7))._setup(x, Color())
        assert_array_equal(s(x), cmap([-0.5, 0, 1.5])[:, :3])

    def test_color_with_transform(self, x):
        if False:
            while True:
                i = 10
        x = pd.Series([1, 10, 100], name='x', dtype=float)
        cmap = color_palette('ch:', as_cmap=True)
        s = Continuous(trans='log')._setup(x, Color())
        assert_array_equal(s(x), cmap([0, 0.5, 1])[:, :3])

    def test_tick_locator(self, x):
        if False:
            for i in range(10):
                print('nop')
        locs = [0.2, 0.5, 0.8]
        locator = mpl.ticker.FixedLocator(locs)
        a = self.setup_ticks(x, locator)
        assert_array_equal(a.major.locator(), locs)

    def test_tick_locator_input_check(self, x):
        if False:
            print('Hello World!')
        err = "Tick locator must be an instance of .*?, not <class 'tuple'>."
        with pytest.raises(TypeError, match=err):
            Continuous().tick((1, 2))

    def test_tick_upto(self, x):
        if False:
            while True:
                i = 10
        for n in [2, 5, 10]:
            a = self.setup_ticks(x, upto=n)
            assert len(a.major.locator()) <= n + 1

    def test_tick_every(self, x):
        if False:
            return 10
        for d in [0.05, 0.2, 0.5]:
            a = self.setup_ticks(x, every=d)
            assert np.allclose(np.diff(a.major.locator()), d)

    def test_tick_every_between(self, x):
        if False:
            for i in range(10):
                print('nop')
        (lo, hi) = (0.2, 0.8)
        for d in [0.05, 0.2, 0.5]:
            a = self.setup_ticks(x, every=d, between=(lo, hi))
            expected = np.arange(lo, hi + d, d)
            assert_array_equal(a.major.locator(), expected)

    def test_tick_at(self, x):
        if False:
            i = 10
            return i + 15
        locs = [0.2, 0.5, 0.9]
        a = self.setup_ticks(x, at=locs)
        assert_array_equal(a.major.locator(), locs)

    def test_tick_count(self, x):
        if False:
            for i in range(10):
                print('nop')
        n = 8
        a = self.setup_ticks(x, count=n)
        assert_array_equal(a.major.locator(), np.linspace(0, 1, n))

    def test_tick_count_between(self, x):
        if False:
            i = 10
            return i + 15
        n = 5
        (lo, hi) = (0.2, 0.7)
        a = self.setup_ticks(x, count=n, between=(lo, hi))
        assert_array_equal(a.major.locator(), np.linspace(lo, hi, n))

    def test_tick_minor(self, x):
        if False:
            for i in range(10):
                print('nop')
        n = 3
        a = self.setup_ticks(x, count=2, minor=n)
        expected = np.linspace(0, 1, n + 2)
        if _version_predates(mpl, '3.8.0rc1'):
            expected = expected[1:]
        assert_array_equal(a.minor.locator(), expected)

    def test_log_tick_default(self, x):
        if False:
            i = 10
            return i + 15
        s = Continuous(trans='log')._setup(x, Coordinate())
        a = PseudoAxis(s._matplotlib_scale)
        a.set_view_interval(0.5, 1050)
        ticks = a.major.locator()
        assert np.allclose(np.diff(np.log10(ticks)), 1)

    def test_log_tick_upto(self, x):
        if False:
            for i in range(10):
                print('nop')
        n = 3
        s = Continuous(trans='log').tick(upto=n)._setup(x, Coordinate())
        a = PseudoAxis(s._matplotlib_scale)
        assert a.major.locator.numticks == n

    def test_log_tick_count(self, x):
        if False:
            print('Hello World!')
        with pytest.raises(RuntimeError, match='`count` requires'):
            Continuous(trans='log').tick(count=4)
        s = Continuous(trans='log').tick(count=4, between=(1, 1000))
        a = PseudoAxis(s._setup(x, Coordinate())._matplotlib_scale)
        a.set_view_interval(0.5, 1050)
        assert_array_equal(a.major.locator(), [1, 10, 100, 1000])

    def test_log_tick_format_disabled(self, x):
        if False:
            while True:
                i = 10
        s = Continuous(trans='log').label(base=None)._setup(x, Coordinate())
        a = PseudoAxis(s._matplotlib_scale)
        a.set_view_interval(20, 20000)
        labels = a.major.formatter.format_ticks(a.major.locator())
        for text in labels:
            assert re.match('^\\d+$', text)

    def test_log_tick_every(self, x):
        if False:
            while True:
                i = 10
        with pytest.raises(RuntimeError, match='`every` not supported'):
            Continuous(trans='log').tick(every=2)

    def test_symlog_tick_default(self, x):
        if False:
            for i in range(10):
                print('nop')
        s = Continuous(trans='symlog')._setup(x, Coordinate())
        a = PseudoAxis(s._matplotlib_scale)
        a.set_view_interval(-1050, 1050)
        ticks = a.major.locator()
        assert ticks[0] == -ticks[-1]
        pos_ticks = np.sort(np.unique(np.abs(ticks)))
        assert np.allclose(np.diff(np.log10(pos_ticks[1:])), 1)
        assert pos_ticks[0] == 0

    def test_label_formatter(self, x):
        if False:
            i = 10
            return i + 15
        fmt = mpl.ticker.FormatStrFormatter('%.3f')
        (a, locs) = self.setup_labels(x, fmt)
        labels = a.major.formatter.format_ticks(locs)
        for text in labels:
            assert re.match('^\\d\\.\\d{3}$', text)

    def test_label_like_pattern(self, x):
        if False:
            for i in range(10):
                print('nop')
        (a, locs) = self.setup_labels(x, like='.4f')
        labels = a.major.formatter.format_ticks(locs)
        for text in labels:
            assert re.match('^\\d\\.\\d{4}$', text)

    def test_label_like_string(self, x):
        if False:
            i = 10
            return i + 15
        (a, locs) = self.setup_labels(x, like='x = {x:.1f}')
        labels = a.major.formatter.format_ticks(locs)
        for text in labels:
            assert re.match('^x = \\d\\.\\d$', text)

    def test_label_like_function(self, x):
        if False:
            return 10
        (a, locs) = self.setup_labels(x, like='{:^5.1f}'.format)
        labels = a.major.formatter.format_ticks(locs)
        for text in labels:
            assert re.match('^ \\d\\.\\d $', text)

    def test_label_base(self, x):
        if False:
            return 10
        (a, locs) = self.setup_labels(100 * x, base=2)
        labels = a.major.formatter.format_ticks(locs)
        for text in labels[1:]:
            assert not text or '2^' in text

    def test_label_unit(self, x):
        if False:
            return 10
        (a, locs) = self.setup_labels(1000 * x, unit='g')
        labels = a.major.formatter.format_ticks(locs)
        for text in labels[1:-1]:
            assert re.match('^\\d+ mg$', text)

    def test_label_unit_with_sep(self, x):
        if False:
            i = 10
            return i + 15
        (a, locs) = self.setup_labels(1000 * x, unit=('', 'g'))
        labels = a.major.formatter.format_ticks(locs)
        for text in labels[1:-1]:
            assert re.match('^\\d+mg$', text)

    def test_label_empty_unit(self, x):
        if False:
            while True:
                i = 10
        (a, locs) = self.setup_labels(1000 * x, unit='')
        labels = a.major.formatter.format_ticks(locs)
        for text in labels[1:-1]:
            assert re.match('^\\d+m$', text)

    def test_label_base_from_transform(self, x):
        if False:
            print('Hello World!')
        s = Continuous(trans='log')
        a = PseudoAxis(s._setup(x, Coordinate())._matplotlib_scale)
        a.set_view_interval(10, 1000)
        (label,) = a.major.formatter.format_ticks([100])
        assert '10^{2}' in label

    def test_label_type_checks(self):
        if False:
            while True:
                i = 10
        s = Continuous()
        with pytest.raises(TypeError, match='Label formatter must be'):
            s.label('{x}')
        with pytest.raises(TypeError, match='`like` must be'):
            s.label(like=2)

class TestNominal:

    @pytest.fixture
    def x(self):
        if False:
            while True:
                i = 10
        return pd.Series(['a', 'c', 'b', 'c'], name='x')

    @pytest.fixture
    def y(self):
        if False:
            print('Hello World!')
        return pd.Series([1, -1.5, 3, -1.5], name='y')

    def test_coordinate_defaults(self, x):
        if False:
            i = 10
            return i + 15
        s = Nominal()._setup(x, Coordinate())
        assert_array_equal(s(x), np.array([0, 1, 2, 1], float))

    def test_coordinate_with_order(self, x):
        if False:
            print('Hello World!')
        s = Nominal(order=['a', 'b', 'c'])._setup(x, Coordinate())
        assert_array_equal(s(x), np.array([0, 2, 1, 2], float))

    def test_coordinate_with_subset_order(self, x):
        if False:
            print('Hello World!')
        s = Nominal(order=['c', 'a'])._setup(x, Coordinate())
        assert_array_equal(s(x), np.array([1, 0, np.nan, 0], float))

    def test_coordinate_axis(self, x):
        if False:
            return 10
        ax = mpl.figure.Figure().subplots()
        s = Nominal()._setup(x, Coordinate(), ax.xaxis)
        assert_array_equal(s(x), np.array([0, 1, 2, 1], float))
        f = ax.xaxis.get_major_formatter()
        assert f.format_ticks([0, 1, 2]) == ['a', 'c', 'b']

    def test_coordinate_axis_with_order(self, x):
        if False:
            for i in range(10):
                print('nop')
        order = ['a', 'b', 'c']
        ax = mpl.figure.Figure().subplots()
        s = Nominal(order=order)._setup(x, Coordinate(), ax.xaxis)
        assert_array_equal(s(x), np.array([0, 2, 1, 2], float))
        f = ax.xaxis.get_major_formatter()
        assert f.format_ticks([0, 1, 2]) == order

    def test_coordinate_axis_with_subset_order(self, x):
        if False:
            i = 10
            return i + 15
        order = ['c', 'a']
        ax = mpl.figure.Figure().subplots()
        s = Nominal(order=order)._setup(x, Coordinate(), ax.xaxis)
        assert_array_equal(s(x), np.array([1, 0, np.nan, 0], float))
        f = ax.xaxis.get_major_formatter()
        assert f.format_ticks([0, 1, 2]) == [*order, '']

    def test_coordinate_axis_with_category_dtype(self, x):
        if False:
            for i in range(10):
                print('nop')
        order = ['b', 'a', 'd', 'c']
        x = x.astype(pd.CategoricalDtype(order))
        ax = mpl.figure.Figure().subplots()
        s = Nominal()._setup(x, Coordinate(), ax.xaxis)
        assert_array_equal(s(x), np.array([1, 3, 0, 3], float))
        f = ax.xaxis.get_major_formatter()
        assert f.format_ticks([0, 1, 2, 3]) == order

    def test_coordinate_numeric_data(self, y):
        if False:
            i = 10
            return i + 15
        ax = mpl.figure.Figure().subplots()
        s = Nominal()._setup(y, Coordinate(), ax.yaxis)
        assert_array_equal(s(y), np.array([1, 0, 2, 0], float))
        f = ax.yaxis.get_major_formatter()
        assert f.format_ticks([0, 1, 2]) == ['-1.5', '1.0', '3.0']

    def test_coordinate_numeric_data_with_order(self, y):
        if False:
            i = 10
            return i + 15
        order = [1, 4, -1.5]
        ax = mpl.figure.Figure().subplots()
        s = Nominal(order=order)._setup(y, Coordinate(), ax.yaxis)
        assert_array_equal(s(y), np.array([0, 2, np.nan, 2], float))
        f = ax.yaxis.get_major_formatter()
        assert f.format_ticks([0, 1, 2]) == ['1.0', '4.0', '-1.5']

    def test_color_defaults(self, x):
        if False:
            while True:
                i = 10
        s = Nominal()._setup(x, Color())
        cs = color_palette()
        assert_array_equal(s(x), [cs[0], cs[1], cs[2], cs[1]])

    def test_color_named_palette(self, x):
        if False:
            print('Hello World!')
        pal = 'flare'
        s = Nominal(pal)._setup(x, Color())
        cs = color_palette(pal, 3)
        assert_array_equal(s(x), [cs[0], cs[1], cs[2], cs[1]])

    def test_color_list_palette(self, x):
        if False:
            for i in range(10):
                print('nop')
        cs = color_palette('crest', 3)
        s = Nominal(cs)._setup(x, Color())
        assert_array_equal(s(x), [cs[0], cs[1], cs[2], cs[1]])

    def test_color_dict_palette(self, x):
        if False:
            return 10
        cs = color_palette('crest', 3)
        pal = dict(zip('bac', cs))
        s = Nominal(pal)._setup(x, Color())
        assert_array_equal(s(x), [cs[1], cs[2], cs[0], cs[2]])

    def test_color_numeric_data(self, y):
        if False:
            i = 10
            return i + 15
        s = Nominal()._setup(y, Color())
        cs = color_palette()
        assert_array_equal(s(y), [cs[1], cs[0], cs[2], cs[0]])

    def test_color_numeric_with_order_subset(self, y):
        if False:
            while True:
                i = 10
        s = Nominal(order=[-1.5, 1])._setup(y, Color())
        (c1, c2) = color_palette(n_colors=2)
        null = (np.nan, np.nan, np.nan)
        assert_array_equal(s(y), [c2, c1, null, c1])

    @pytest.mark.xfail(reason='Need to sort out float/int order')
    def test_color_numeric_int_float_mix(self):
        if False:
            while True:
                i = 10
        z = pd.Series([1, 2], name='z')
        s = Nominal(order=[1.0, 2])._setup(z, Color())
        (c1, c2) = color_palette(n_colors=2)
        null = (np.nan, np.nan, np.nan)
        assert_array_equal(s(z), [c1, null, c2])

    def test_color_alpha_in_palette(self, x):
        if False:
            i = 10
            return i + 15
        cs = [(0.2, 0.2, 0.3, 0.5), (0.1, 0.2, 0.3, 1), (0.5, 0.6, 0.2, 0)]
        s = Nominal(cs)._setup(x, Color())
        assert_array_equal(s(x), [cs[0], cs[1], cs[2], cs[1]])

    def test_color_unknown_palette(self, x):
        if False:
            return 10
        pal = 'not_a_palette'
        err = f"'{pal}' is not a valid palette name"
        with pytest.raises(ValueError, match=err):
            Nominal(pal)._setup(x, Color())

    def test_object_defaults(self, x):
        if False:
            for i in range(10):
                print('nop')

        class MockProperty(ObjectProperty):

            def _default_values(self, n):
                if False:
                    return 10
                return list('xyz'[:n])
        s = Nominal()._setup(x, MockProperty())
        assert s(x) == ['x', 'y', 'z', 'y']

    def test_object_list(self, x):
        if False:
            for i in range(10):
                print('nop')
        vs = ['x', 'y', 'z']
        s = Nominal(vs)._setup(x, ObjectProperty())
        assert s(x) == ['x', 'y', 'z', 'y']

    def test_object_dict(self, x):
        if False:
            print('Hello World!')
        vs = {'a': 'x', 'b': 'y', 'c': 'z'}
        s = Nominal(vs)._setup(x, ObjectProperty())
        assert s(x) == ['x', 'z', 'y', 'z']

    def test_object_order(self, x):
        if False:
            return 10
        vs = ['x', 'y', 'z']
        s = Nominal(vs, order=['c', 'a', 'b'])._setup(x, ObjectProperty())
        assert s(x) == ['y', 'x', 'z', 'x']

    def test_object_order_subset(self, x):
        if False:
            i = 10
            return i + 15
        vs = ['x', 'y']
        s = Nominal(vs, order=['a', 'c'])._setup(x, ObjectProperty())
        assert s(x) == ['x', 'y', None, 'y']

    def test_objects_that_are_weird(self, x):
        if False:
            i = 10
            return i + 15
        vs = [('x', 1), (None, None, 0), {}]
        s = Nominal(vs)._setup(x, ObjectProperty())
        assert s(x) == [vs[0], vs[1], vs[2], vs[1]]

    def test_alpha_default(self, x):
        if False:
            i = 10
            return i + 15
        s = Nominal()._setup(x, Alpha())
        assert_array_equal(s(x), [0.95, 0.625, 0.3, 0.625])

    def test_fill(self):
        if False:
            while True:
                i = 10
        x = pd.Series(['a', 'a', 'b', 'a'], name='x')
        s = Nominal()._setup(x, Fill())
        assert_array_equal(s(x), [True, True, False, True])

    def test_fill_dict(self):
        if False:
            while True:
                i = 10
        x = pd.Series(['a', 'a', 'b', 'a'], name='x')
        vs = {'a': False, 'b': True}
        s = Nominal(vs)._setup(x, Fill())
        assert_array_equal(s(x), [False, False, True, False])

    def test_fill_nunique_warning(self):
        if False:
            for i in range(10):
                print('nop')
        x = pd.Series(['a', 'b', 'c', 'a', 'b'], name='x')
        with pytest.warns(UserWarning, match='The variable assigned to fill'):
            s = Nominal()._setup(x, Fill())
        assert_array_equal(s(x), [True, False, True, True, False])

    def test_interval_defaults(self, x):
        if False:
            print('Hello World!')

        class MockProperty(IntervalProperty):
            _default_range = (1, 2)
        s = Nominal()._setup(x, MockProperty())
        assert_array_equal(s(x), [2, 1.5, 1, 1.5])

    def test_interval_tuple(self, x):
        if False:
            print('Hello World!')
        s = Nominal((1, 2))._setup(x, IntervalProperty())
        assert_array_equal(s(x), [2, 1.5, 1, 1.5])

    def test_interval_tuple_numeric(self, y):
        if False:
            while True:
                i = 10
        s = Nominal((1, 2))._setup(y, IntervalProperty())
        assert_array_equal(s(y), [1.5, 2, 1, 2])

    def test_interval_list(self, x):
        if False:
            i = 10
            return i + 15
        vs = [2, 5, 4]
        s = Nominal(vs)._setup(x, IntervalProperty())
        assert_array_equal(s(x), [2, 5, 4, 5])

    def test_interval_dict(self, x):
        if False:
            for i in range(10):
                print('nop')
        vs = {'a': 3, 'b': 4, 'c': 6}
        s = Nominal(vs)._setup(x, IntervalProperty())
        assert_array_equal(s(x), [3, 6, 4, 6])

    def test_interval_with_transform(self, x):
        if False:
            for i in range(10):
                print('nop')

        class MockProperty(IntervalProperty):
            _forward = np.square
            _inverse = np.sqrt
        s = Nominal((2, 4))._setup(x, MockProperty())
        assert_array_equal(s(x), [4, np.sqrt(10), 2, np.sqrt(10)])

    def test_empty_data(self):
        if False:
            while True:
                i = 10
        x = pd.Series([], dtype=object, name='x')
        s = Nominal()._setup(x, Coordinate())
        assert_array_equal(s(x), [])

    @pytest.mark.skipif(_version_predates(mpl, '3.4.0'), reason='Test failing on older matplotlib for unclear reasons')
    def test_finalize(self, x):
        if False:
            for i in range(10):
                print('nop')
        ax = mpl.figure.Figure().subplots()
        s = Nominal()._setup(x, Coordinate(), ax.yaxis)
        s._finalize(Plot(), ax.yaxis)
        levels = x.unique()
        assert ax.get_ylim() == (len(levels) - 0.5, -0.5)
        assert_array_equal(ax.get_yticks(), list(range(len(levels))))
        for (i, expected) in enumerate(levels):
            assert ax.yaxis.major.formatter(i) == expected

class TestTemporal:

    @pytest.fixture
    def t(self):
        if False:
            print('Hello World!')
        dates = pd.to_datetime(['1972-09-27', '1975-06-24', '1980-12-14'])
        return pd.Series(dates, name='x')

    @pytest.fixture
    def x(self, t):
        if False:
            i = 10
            return i + 15
        return pd.Series(mpl.dates.date2num(t), name=t.name)

    def test_coordinate_defaults(self, t, x):
        if False:
            i = 10
            return i + 15
        s = Temporal()._setup(t, Coordinate())
        assert_array_equal(s(t), x)

    def test_interval_defaults(self, t, x):
        if False:
            for i in range(10):
                print('nop')
        s = Temporal()._setup(t, IntervalProperty())
        normed = (x - x.min()) / (x.max() - x.min())
        assert_array_equal(s(t), normed)

    def test_interval_with_range(self, t, x):
        if False:
            i = 10
            return i + 15
        values = (1, 3)
        s = Temporal((1, 3))._setup(t, IntervalProperty())
        normed = (x - x.min()) / (x.max() - x.min())
        expected = normed * (values[1] - values[0]) + values[0]
        assert_array_equal(s(t), expected)

    def test_interval_with_norm(self, t, x):
        if False:
            while True:
                i = 10
        norm = (t[1], t[2])
        s = Temporal(norm=norm)._setup(t, IntervalProperty())
        n = mpl.dates.date2num(norm)
        normed = (x - n[0]) / (n[1] - n[0])
        assert_array_equal(s(t), normed)

    def test_color_defaults(self, t, x):
        if False:
            while True:
                i = 10
        cmap = color_palette('ch:', as_cmap=True)
        s = Temporal()._setup(t, Color())
        normed = (x - x.min()) / (x.max() - x.min())
        assert_array_equal(s(t), cmap(normed)[:, :3])

    def test_color_named_values(self, t, x):
        if False:
            print('Hello World!')
        name = 'viridis'
        cmap = color_palette(name, as_cmap=True)
        s = Temporal(name)._setup(t, Color())
        normed = (x - x.min()) / (x.max() - x.min())
        assert_array_equal(s(t), cmap(normed)[:, :3])

    def test_coordinate_axis(self, t, x):
        if False:
            while True:
                i = 10
        ax = mpl.figure.Figure().subplots()
        s = Temporal()._setup(t, Coordinate(), ax.xaxis)
        assert_array_equal(s(t), x)
        locator = ax.xaxis.get_major_locator()
        formatter = ax.xaxis.get_major_formatter()
        assert isinstance(locator, mpl.dates.AutoDateLocator)
        assert isinstance(formatter, mpl.dates.AutoDateFormatter)

    def test_tick_locator(self, t):
        if False:
            i = 10
            return i + 15
        locator = mpl.dates.YearLocator(month=3, day=15)
        s = Temporal().tick(locator)
        a = PseudoAxis(s._setup(t, Coordinate())._matplotlib_scale)
        a.set_view_interval(0, 365)
        assert 73 in a.major.locator()

    def test_tick_upto(self, t, x):
        if False:
            i = 10
            return i + 15
        n = 8
        ax = mpl.figure.Figure().subplots()
        Temporal().tick(upto=n)._setup(t, Coordinate(), ax.xaxis)
        locator = ax.xaxis.get_major_locator()
        assert set(locator.maxticks.values()) == {n}

    def test_label_formatter(self, t):
        if False:
            while True:
                i = 10
        formatter = mpl.dates.DateFormatter('%Y')
        s = Temporal().label(formatter)
        a = PseudoAxis(s._setup(t, Coordinate())._matplotlib_scale)
        a.set_view_interval(10, 1000)
        (label,) = a.major.formatter.format_ticks([100])
        assert label == '1970'

    def test_label_concise(self, t, x):
        if False:
            i = 10
            return i + 15
        ax = mpl.figure.Figure().subplots()
        Temporal().label(concise=True)._setup(t, Coordinate(), ax.xaxis)
        formatter = ax.xaxis.get_major_formatter()
        assert isinstance(formatter, mpl.dates.ConciseDateFormatter)

class TestBoolean:

    @pytest.fixture
    def x(self):
        if False:
            for i in range(10):
                print('nop')
        return pd.Series([True, False, False, True], name='x', dtype=bool)

    def test_coordinate(self, x):
        if False:
            print('Hello World!')
        s = Boolean()._setup(x, Coordinate())
        assert_array_equal(s(x), x.astype(float))

    def test_coordinate_axis(self, x):
        if False:
            while True:
                i = 10
        ax = mpl.figure.Figure().subplots()
        s = Boolean()._setup(x, Coordinate(), ax.xaxis)
        assert_array_equal(s(x), x.astype(float))
        f = ax.xaxis.get_major_formatter()
        assert f.format_ticks([0, 1]) == ['False', 'True']

    @pytest.mark.parametrize('dtype,value', [(object, np.nan), (object, None), ('boolean', pd.NA)])
    def test_coordinate_missing(self, x, dtype, value):
        if False:
            print('Hello World!')
        x = x.astype(dtype)
        x[2] = value
        s = Boolean()._setup(x, Coordinate())
        assert_array_equal(s(x), x.astype(float))

    def test_color_defaults(self, x):
        if False:
            return 10
        s = Boolean()._setup(x, Color())
        cs = color_palette()
        expected = [cs[int(x_i)] for x_i in ~x]
        assert_array_equal(s(x), expected)

    def test_color_list_palette(self, x):
        if False:
            while True:
                i = 10
        cs = color_palette('crest', 2)
        s = Boolean(cs)._setup(x, Color())
        expected = [cs[int(x_i)] for x_i in ~x]
        assert_array_equal(s(x), expected)

    def test_color_tuple_palette(self, x):
        if False:
            print('Hello World!')
        cs = tuple(color_palette('crest', 2))
        s = Boolean(cs)._setup(x, Color())
        expected = [cs[int(x_i)] for x_i in ~x]
        assert_array_equal(s(x), expected)

    def test_color_dict_palette(self, x):
        if False:
            return 10
        cs = color_palette('crest', 2)
        pal = {True: cs[0], False: cs[1]}
        s = Boolean(pal)._setup(x, Color())
        expected = [pal[x_i] for x_i in x]
        assert_array_equal(s(x), expected)

    def test_object_defaults(self, x):
        if False:
            print('Hello World!')
        vs = ['x', 'y', 'z']

        class MockProperty(ObjectProperty):

            def _default_values(self, n):
                if False:
                    return 10
                return vs[:n]
        s = Boolean()._setup(x, MockProperty())
        expected = [vs[int(x_i)] for x_i in ~x]
        assert s(x) == expected

    def test_object_list(self, x):
        if False:
            for i in range(10):
                print('nop')
        vs = ['x', 'y']
        s = Boolean(vs)._setup(x, ObjectProperty())
        expected = [vs[int(x_i)] for x_i in ~x]
        assert s(x) == expected

    def test_object_dict(self, x):
        if False:
            print('Hello World!')
        vs = {True: 'x', False: 'y'}
        s = Boolean(vs)._setup(x, ObjectProperty())
        expected = [vs[x_i] for x_i in x]
        assert s(x) == expected

    def test_fill(self, x):
        if False:
            i = 10
            return i + 15
        s = Boolean()._setup(x, Fill())
        assert_array_equal(s(x), x)

    def test_interval_defaults(self, x):
        if False:
            for i in range(10):
                print('nop')
        vs = (1, 2)

        class MockProperty(IntervalProperty):
            _default_range = vs
        s = Boolean()._setup(x, MockProperty())
        expected = [vs[int(x_i)] for x_i in x]
        assert_array_equal(s(x), expected)

    def test_interval_tuple(self, x):
        if False:
            return 10
        vs = (3, 5)
        s = Boolean(vs)._setup(x, IntervalProperty())
        expected = [vs[int(x_i)] for x_i in x]
        assert_array_equal(s(x), expected)

    def test_finalize(self, x):
        if False:
            i = 10
            return i + 15
        ax = mpl.figure.Figure().subplots()
        s = Boolean()._setup(x, Coordinate(), ax.xaxis)
        s._finalize(Plot(), ax.xaxis)
        assert ax.get_xlim() == (1.5, -0.5)
        assert_array_equal(ax.get_xticks(), [0, 1])
        assert ax.xaxis.major.formatter(0) == 'False'
        assert ax.xaxis.major.formatter(1) == 'True'