"""
Tests specific to the lines module.
"""
import itertools
import platform
import timeit
from types import SimpleNamespace
from cycler import cycler
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib
import matplotlib as mpl
from matplotlib import _path
import matplotlib.lines as mlines
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.testing.decorators import image_comparison, check_figures_equal

def test_segment_hits():
    if False:
        return 10
    'Test a problematic case.'
    (cx, cy) = (553, 902)
    (x, y) = (np.array([553.0, 553.0]), np.array([95.0, 947.0]))
    radius = 6.94
    assert_array_equal(mlines.segment_hits(cx, cy, x, y, radius), [0])

@pytest.mark.flaky(reruns=3)
def test_invisible_Line_rendering():
    if False:
        i = 10
        return i + 15
    '\n    GitHub issue #1256 identified a bug in Line.draw method\n\n    Despite visibility attribute set to False, the draw method was not\n    returning early enough and some pre-rendering code was executed\n    though not necessary.\n\n    Consequence was an excessive draw time for invisible Line instances\n    holding a large number of points (Npts> 10**6)\n    '
    N = 10 ** 7
    x = np.linspace(0, 1, N)
    y = np.random.normal(size=N)
    fig = plt.figure()
    ax = plt.subplot()
    l = mlines.Line2D(x, y)
    l.set_visible(False)
    t_no_line = min(timeit.repeat(fig.canvas.draw, number=1, repeat=3))
    ax.add_line(l)
    t_invisible_line = min(timeit.repeat(fig.canvas.draw, number=1, repeat=3))
    slowdown_factor = t_invisible_line / t_no_line
    slowdown_threshold = 2
    assert slowdown_factor < slowdown_threshold

def test_set_line_coll_dash():
    if False:
        for i in range(10):
            print('nop')
    (fig, ax) = plt.subplots()
    np.random.seed(0)
    ax.contour(np.random.randn(20, 30), linestyles=[(0, (3, 3))])

def test_invalid_line_data():
    if False:
        print('Hello World!')
    with pytest.raises(RuntimeError, match='xdata must be'):
        mlines.Line2D(0, [])
    with pytest.raises(RuntimeError, match='ydata must be'):
        mlines.Line2D([], 1)
    line = mlines.Line2D([], [])
    with pytest.raises(RuntimeError, match='x must be'):
        line.set_xdata(0)
    with pytest.raises(RuntimeError, match='y must be'):
        line.set_ydata(0)

@image_comparison(['line_dashes'], remove_text=True, tol=0.002)
def test_line_dashes():
    if False:
        print('Hello World!')
    (fig, ax) = plt.subplots()
    ax.plot(range(10), linestyle=(0, (3, 3)), lw=5)

def test_line_colors():
    if False:
        for i in range(10):
            print('nop')
    (fig, ax) = plt.subplots()
    ax.plot(range(10), color='none')
    ax.plot(range(10), color='r')
    ax.plot(range(10), color='.3')
    ax.plot(range(10), color=(1, 0, 0, 1))
    ax.plot(range(10), color=(1, 0, 0))
    fig.canvas.draw()

def test_valid_colors():
    if False:
        print('Hello World!')
    line = mlines.Line2D([], [])
    with pytest.raises(ValueError):
        line.set_color('foobar')

def test_linestyle_variants():
    if False:
        i = 10
        return i + 15
    (fig, ax) = plt.subplots()
    for ls in ['-', 'solid', '--', 'dashed', '-.', 'dashdot', ':', 'dotted', (0, None), (0, ()), (0, [])]:
        ax.plot(range(10), linestyle=ls)
    fig.canvas.draw()

def test_valid_linestyles():
    if False:
        print('Hello World!')
    line = mlines.Line2D([], [])
    with pytest.raises(ValueError):
        line.set_linestyle('aardvark')

@image_comparison(['drawstyle_variants.png'], remove_text=True)
def test_drawstyle_variants():
    if False:
        return 10
    (fig, axs) = plt.subplots(6)
    dss = ['default', 'steps-mid', 'steps-pre', 'steps-post', 'steps', None]
    for (ax, ds) in zip(axs.flat, dss):
        ax.plot(range(2000), drawstyle=ds)
        ax.set(xlim=(0, 2), ylim=(0, 2))

@check_figures_equal(extensions=('png',))
def test_no_subslice_with_transform(fig_ref, fig_test):
    if False:
        return 10
    ax = fig_ref.add_subplot()
    x = np.arange(2000)
    ax.plot(x + 2000, x)
    ax = fig_test.add_subplot()
    t = mtransforms.Affine2D().translate(2000.0, 0.0)
    ax.plot(x, x, transform=t + ax.transData)

def test_valid_drawstyles():
    if False:
        return 10
    line = mlines.Line2D([], [])
    with pytest.raises(ValueError):
        line.set_drawstyle('foobar')

def test_set_drawstyle():
    if False:
        i = 10
        return i + 15
    x = np.linspace(0, 2 * np.pi, 10)
    y = np.sin(x)
    (fig, ax) = plt.subplots()
    (line,) = ax.plot(x, y)
    line.set_drawstyle('steps-pre')
    assert len(line.get_path().vertices) == 2 * len(x) - 1
    line.set_drawstyle('default')
    assert len(line.get_path().vertices) == len(x)

@image_comparison(['line_collection_dashes'], remove_text=True, style='mpl20', tol=0.65 if platform.machine() in ('aarch64', 'ppc64le', 's390x') else 0)
def test_set_line_coll_dash_image():
    if False:
        return 10
    (fig, ax) = plt.subplots()
    np.random.seed(0)
    ax.contour(np.random.randn(20, 30), linestyles=[(0, (3, 3))])

@image_comparison(['marker_fill_styles.png'], remove_text=True)
def test_marker_fill_styles():
    if False:
        for i in range(10):
            print('nop')
    colors = itertools.cycle([[0, 0, 1], 'g', '#ff0000', 'c', 'm', 'y', np.array([0, 0, 0])])
    altcolor = 'lightgreen'
    y = np.array([1, 1])
    x = np.array([0, 9])
    (fig, ax) = plt.subplots()
    for (j, marker) in enumerate('ov^<>8sp*hHDdPX'):
        for (i, fs) in enumerate(mlines.Line2D.fillStyles):
            color = next(colors)
            ax.plot(j * 10 + x, y + i + 0.5 * (j % 2), marker=marker, markersize=20, markerfacecoloralt=altcolor, fillstyle=fs, label=fs, linewidth=5, color=color, markeredgecolor=color, markeredgewidth=2)
    ax.set_ylim([0, 7.5])
    ax.set_xlim([-5, 155])

def test_markerfacecolor_fillstyle():
    if False:
        while True:
            i = 10
    "Test that markerfacecolor does not override fillstyle='none'."
    (l,) = plt.plot([1, 3, 2], marker=MarkerStyle('o', fillstyle='none'), markerfacecolor='red')
    assert l.get_fillstyle() == 'none'
    assert l.get_markerfacecolor() == 'none'

@image_comparison(['scaled_lines'], style='default')
def test_lw_scaling():
    if False:
        for i in range(10):
            print('nop')
    th = np.linspace(0, 32)
    (fig, ax) = plt.subplots()
    lins_styles = ['dashed', 'dotted', 'dashdot']
    cy = cycler(matplotlib.rcParams['axes.prop_cycle'])
    for (j, (ls, sty)) in enumerate(zip(lins_styles, cy)):
        for lw in np.linspace(0.5, 10, 10):
            ax.plot(th, j * np.ones(50) + 0.1 * lw, linestyle=ls, lw=lw, **sty)

def test_is_sorted_and_has_non_nan():
    if False:
        for i in range(10):
            print('nop')
    assert _path.is_sorted_and_has_non_nan(np.array([1, 2, 3]))
    assert _path.is_sorted_and_has_non_nan(np.array([1, np.nan, 3]))
    assert not _path.is_sorted_and_has_non_nan([3, 5] + [np.nan] * 100 + [0, 2])
    n = 2 * mlines.Line2D._subslice_optim_min_size
    plt.plot([np.nan] * n, range(n))

@check_figures_equal()
def test_step_markers(fig_test, fig_ref):
    if False:
        while True:
            i = 10
    fig_test.subplots().step([0, 1], '-o')
    fig_ref.subplots().plot([0, 0, 1], [0, 1, 1], '-o', markevery=[0, 2])

@pytest.mark.parametrize('parent', ['figure', 'axes'])
@check_figures_equal(extensions=('png',))
def test_markevery(fig_test, fig_ref, parent):
    if False:
        i = 10
        return i + 15
    np.random.seed(42)
    x = np.linspace(0, 1, 14)
    y = np.random.rand(len(x))
    cases_test = [None, 4, (2, 5), [1, 5, 11], [0, -1], slice(5, 10, 2), np.arange(len(x))[y > 0.5], 0.3, (0.3, 0.4)]
    cases_ref = ['11111111111111', '10001000100010', '00100001000010', '01000100000100', '10000000000001', '00000101010000', '01110001110110', '11011011011110', '01010011011101']
    if parent == 'figure':
        cases_test = cases_test[:-2]
        cases_ref = cases_ref[:-2]

        def add_test(x, y, *, markevery):
            if False:
                print('Hello World!')
            fig_test.add_artist(mlines.Line2D(x, y, marker='o', markevery=markevery))

        def add_ref(x, y, *, markevery):
            if False:
                for i in range(10):
                    print('nop')
            fig_ref.add_artist(mlines.Line2D(x, y, marker='o', markevery=markevery))
    elif parent == 'axes':
        axs_test = iter(fig_test.subplots(3, 3).flat)
        axs_ref = iter(fig_ref.subplots(3, 3).flat)

        def add_test(x, y, *, markevery):
            if False:
                i = 10
                return i + 15
            next(axs_test).plot(x, y, '-gD', markevery=markevery)

        def add_ref(x, y, *, markevery):
            if False:
                while True:
                    i = 10
            next(axs_ref).plot(x, y, '-gD', markevery=markevery)
    for case in cases_test:
        add_test(x, y, markevery=case)
    for case in cases_ref:
        me = np.array(list(case)).astype(int).astype(bool)
        add_ref(x, y, markevery=me)

def test_markevery_figure_line_unsupported_relsize():
    if False:
        print('Hello World!')
    fig = plt.figure()
    fig.add_artist(mlines.Line2D([0, 1], [0, 1], marker='o', markevery=0.5))
    with pytest.raises(ValueError):
        fig.canvas.draw()

def test_marker_as_markerstyle():
    if False:
        i = 10
        return i + 15
    (fig, ax) = plt.subplots()
    (line,) = ax.plot([2, 4, 3], marker=MarkerStyle('D'))
    fig.canvas.draw()
    assert line.get_marker() == 'D'
    line.set_marker('s')
    fig.canvas.draw()
    line.set_marker(MarkerStyle('o'))
    fig.canvas.draw()
    triangle1 = Path._create_closed([[-1, -1], [1, -1], [0, 2]])
    (line2,) = ax.plot([1, 3, 2], marker=MarkerStyle(triangle1), ms=22)
    (line3,) = ax.plot([0, 2, 1], marker=triangle1, ms=22)
    assert_array_equal(line2.get_marker().vertices, triangle1.vertices)
    assert_array_equal(line3.get_marker().vertices, triangle1.vertices)

@image_comparison(['striped_line.png'], remove_text=True, style='mpl20')
def test_striped_lines():
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.default_rng(19680801)
    (_, ax) = plt.subplots()
    ax.plot(rng.uniform(size=12), color='orange', gapcolor='blue', linestyle='--', lw=5, label=' ')
    ax.plot(rng.uniform(size=12), color='red', gapcolor='black', linestyle=(0, (2, 5, 4, 2)), lw=5, label=' ', alpha=0.5)
    ax.legend(handlelength=5)

@check_figures_equal()
def test_odd_dashes(fig_test, fig_ref):
    if False:
        print('Hello World!')
    fig_test.add_subplot().plot([1, 2], dashes=[1, 2, 3])
    fig_ref.add_subplot().plot([1, 2], dashes=[1, 2, 3, 1, 2, 3])

def test_picking():
    if False:
        return 10
    (fig, ax) = plt.subplots()
    mouse_event = SimpleNamespace(x=fig.bbox.width // 2, y=fig.bbox.height // 2 + 15)
    (l0,) = ax.plot([0, 1], [0, 1], picker=True)
    (found, indices) = l0.contains(mouse_event)
    assert not found
    (l1,) = ax.plot([0, 1], [0, 1], picker=True, pickradius=20)
    (found, indices) = l1.contains(mouse_event)
    assert found
    assert_array_equal(indices['ind'], [0])
    (l2,) = ax.plot([0, 1], [0, 1], picker=True)
    (found, indices) = l2.contains(mouse_event)
    assert not found
    l2.set_pickradius(20)
    (found, indices) = l2.contains(mouse_event)
    assert found
    assert_array_equal(indices['ind'], [0])

@check_figures_equal()
def test_input_copy(fig_test, fig_ref):
    if False:
        return 10
    t = np.arange(0, 6, 2)
    (l,) = fig_test.add_subplot().plot(t, t, '.-')
    t[:] = range(3)
    l.set_drawstyle('steps')
    fig_ref.add_subplot().plot([0, 2, 4], [0, 2, 4], '.-', drawstyle='steps')

@check_figures_equal(extensions=['png'])
def test_markevery_prop_cycle(fig_test, fig_ref):
    if False:
        print('Hello World!')
    'Test that we can set markevery prop_cycle.'
    cases = [None, 8, (30, 8), [16, 24, 30], [0, -1], slice(100, 200, 3), 0.1, 0.3, 1.5, (0.0, 0.1), (0.45, 0.1)]
    cmap = mpl.colormaps['jet']
    colors = cmap(np.linspace(0.2, 0.8, len(cases)))
    x = np.linspace(-1, 1)
    y = 5 * x ** 2
    axs = fig_ref.add_subplot()
    for (i, markevery) in enumerate(cases):
        axs.plot(y - i, 'o-', markevery=markevery, color=colors[i])
    matplotlib.rcParams['axes.prop_cycle'] = cycler(markevery=cases, color=colors)
    ax = fig_test.add_subplot()
    for (i, _) in enumerate(cases):
        ax.plot(y - i, 'o-')

def test_axline_setters():
    if False:
        return 10
    (fig, ax) = plt.subplots()
    line1 = ax.axline((0.1, 0.1), slope=0.6)
    line2 = ax.axline((0.1, 0.1), (0.8, 0.4))
    line1.set_xy1(0.2, 0.3)
    line1.set_slope(2.4)
    line2.set_xy1(0.3, 0.2)
    line2.set_xy2(0.6, 0.8)
    assert line1.get_xy1() == (0.2, 0.3)
    assert line1.get_slope() == 2.4
    assert line2.get_xy1() == (0.3, 0.2)
    assert line2.get_xy2() == (0.6, 0.8)
    with pytest.raises(ValueError, match="Cannot set an 'xy2' value while 'slope' is set"):
        line1.set_xy2(0.2, 0.3)
    with pytest.raises(ValueError, match="Cannot set a 'slope' value while 'xy2' is set"):
        line2.set_slope(3)