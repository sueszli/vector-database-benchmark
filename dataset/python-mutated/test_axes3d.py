import functools
import itertools
import platform
import pytest
from mpl_toolkits.mplot3d import Axes3D, axes3d, proj3d, art3d
import matplotlib as mpl
from matplotlib.backend_bases import MouseButton, MouseEvent, NavigationToolbar2
from matplotlib import cm
from matplotlib import colors as mcolors, patches as mpatch
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.testing.widgets import mock_event
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np
mpl3d_image_comparison = functools.partial(image_comparison, remove_text=True, style='default')

def plot_cuboid(ax, scale):
    if False:
        print('Hello World!')
    r = [0, 1]
    pts = itertools.combinations(np.array(list(itertools.product(r, r, r))), 2)
    for (start, end) in pts:
        if np.sum(np.abs(start - end)) == r[1] - r[0]:
            ax.plot3D(*zip(start * np.array(scale), end * np.array(scale)))

@check_figures_equal(extensions=['png'])
def test_invisible_axes(fig_test, fig_ref):
    if False:
        for i in range(10):
            print('nop')
    ax = fig_test.subplots(subplot_kw=dict(projection='3d'))
    ax.set_visible(False)

@mpl3d_image_comparison(['grid_off.png'], style='mpl20')
def test_grid_off():
    if False:
        print('Hello World!')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.grid(False)

@mpl3d_image_comparison(['invisible_ticks_axis.png'], style='mpl20')
def test_invisible_ticks_axis():
    if False:
        print('Hello World!')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.line.set_visible(False)

@mpl3d_image_comparison(['axis_positions.png'], remove_text=False, style='mpl20')
def test_axis_positions():
    if False:
        for i in range(10):
            print('nop')
    positions = ['upper', 'lower', 'both', 'none']
    (fig, axs) = plt.subplots(2, 2, subplot_kw={'projection': '3d'})
    for (ax, pos) in zip(axs.flatten(), positions):
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_label_position(pos)
            axis.set_ticks_position(pos)
        title = f'{pos}'
        ax.set(xlabel='x', ylabel='y', zlabel='z', title=title)

@mpl3d_image_comparison(['aspects.png'], remove_text=False, style='mpl20')
def test_aspects():
    if False:
        print('Hello World!')
    aspects = ('auto', 'equal', 'equalxy', 'equalyz', 'equalxz', 'equal')
    (_, axs) = plt.subplots(2, 3, subplot_kw={'projection': '3d'})
    for ax in axs.flatten()[0:-1]:
        plot_cuboid(ax, scale=[1, 1, 5])
    plot_cuboid(axs[1][2], scale=[1, 1, 1])
    for (i, ax) in enumerate(axs.flatten()):
        ax.set_title(aspects[i])
        ax.set_box_aspect((3, 4, 5))
        ax.set_aspect(aspects[i], adjustable='datalim')
    axs[1][2].set_title('equal (cube)')

@mpl3d_image_comparison(['aspects_adjust_box.png'], remove_text=False, style='mpl20')
def test_aspects_adjust_box():
    if False:
        print('Hello World!')
    aspects = ('auto', 'equal', 'equalxy', 'equalyz', 'equalxz')
    (fig, axs) = plt.subplots(1, len(aspects), subplot_kw={'projection': '3d'}, figsize=(11, 3))
    for (i, ax) in enumerate(axs):
        plot_cuboid(ax, scale=[4, 3, 5])
        ax.set_title(aspects[i])
        ax.set_aspect(aspects[i], adjustable='box')

def test_axes3d_repr():
    if False:
        i = 10
        return i + 15
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_label('label')
    ax.set_title('title')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    assert repr(ax) == "<Axes3D: label='label', title={'center': 'title'}, xlabel='x', ylabel='y', zlabel='z'>"

@mpl3d_image_comparison(['axes3d_primary_views.png'], style='mpl20')
def test_axes3d_primary_views():
    if False:
        return 10
    views = [(90, -90, 0), (0, -90, 0), (0, 0, 0), (-90, 90, 0), (0, 90, 0), (0, 180, 0)]
    (fig, axs) = plt.subplots(2, 3, subplot_kw={'projection': '3d'})
    for (i, ax) in enumerate(axs.flat):
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_proj_type('ortho')
        ax.view_init(elev=views[i][0], azim=views[i][1], roll=views[i][2])
    plt.tight_layout()

@mpl3d_image_comparison(['bar3d.png'], style='mpl20')
def test_bar3d():
    if False:
        for i in range(10):
            print('nop')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for (c, z) in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):
        xs = np.arange(20)
        ys = np.arange(20)
        cs = [c] * len(xs)
        cs[0] = 'c'
        ax.bar(xs, ys, zs=z, zdir='y', align='edge', color=cs, alpha=0.8)

def test_bar3d_colors():
    if False:
        while True:
            i = 10
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for c in ['red', 'green', 'blue', 'yellow']:
        xs = np.arange(len(c))
        ys = np.zeros_like(xs)
        zs = np.zeros_like(ys)
        ax.bar3d(xs, ys, zs, 1, 1, 1, color=c)

@mpl3d_image_comparison(['bar3d_shaded.png'], style='mpl20')
def test_bar3d_shaded():
    if False:
        for i in range(10):
            print('nop')
    x = np.arange(4)
    y = np.arange(5)
    (x2d, y2d) = np.meshgrid(x, y)
    (x2d, y2d) = (x2d.ravel(), y2d.ravel())
    z = x2d + y2d + 1
    views = [(30, -60, 0), (30, 30, 30), (-30, 30, -90), (300, -30, 0)]
    fig = plt.figure(figsize=plt.figaspect(1 / len(views)))
    axs = fig.subplots(1, len(views), subplot_kw=dict(projection='3d'))
    for (ax, (elev, azim, roll)) in zip(axs, views):
        ax.bar3d(x2d, y2d, x2d * 0, 1, 1, z, shade=True)
        ax.view_init(elev=elev, azim=azim, roll=roll)
    fig.canvas.draw()

@mpl3d_image_comparison(['bar3d_notshaded.png'], style='mpl20')
def test_bar3d_notshaded():
    if False:
        for i in range(10):
            print('nop')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = np.arange(4)
    y = np.arange(5)
    (x2d, y2d) = np.meshgrid(x, y)
    (x2d, y2d) = (x2d.ravel(), y2d.ravel())
    z = x2d + y2d
    ax.bar3d(x2d, y2d, x2d * 0, 1, 1, z, shade=False)
    fig.canvas.draw()

def test_bar3d_lightsource():
    if False:
        for i in range(10):
            print('nop')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ls = mcolors.LightSource(azdeg=0, altdeg=90)
    (length, width) = (3, 4)
    area = length * width
    (x, y) = np.meshgrid(np.arange(length), np.arange(width))
    x = x.ravel()
    y = y.ravel()
    dz = x + y
    color = [cm.coolwarm(i / area) for i in range(area)]
    collection = ax.bar3d(x=x, y=y, z=0, dx=1, dy=1, dz=dz, color=color, shade=True, lightsource=ls)
    np.testing.assert_array_max_ulp(color, collection._facecolor3d[1::6], 4)

@mpl3d_image_comparison(['contour3d.png'], style='mpl20', tol=0.002 if platform.machine() in ('aarch64', 'ppc64le', 's390x') else 0)
def test_contour3d():
    if False:
        for i in range(10):
            print('nop')
    plt.rcParams['axes3d.automargin'] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    (X, Y, Z) = axes3d.get_test_data(0.05)
    ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
    ax.axis(xmin=-40, xmax=40, ymin=-40, ymax=40, zmin=-100, zmax=100)

@mpl3d_image_comparison(['contour3d_extend3d.png'], style='mpl20')
def test_contour3d_extend3d():
    if False:
        for i in range(10):
            print('nop')
    plt.rcParams['axes3d.automargin'] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    (X, Y, Z) = axes3d.get_test_data(0.05)
    ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm, extend3d=True)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-20, 40)
    ax.set_zlim(-80, 80)

@mpl3d_image_comparison(['contourf3d.png'], style='mpl20')
def test_contourf3d():
    if False:
        i = 10
        return i + 15
    plt.rcParams['axes3d.automargin'] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    (X, Y, Z) = axes3d.get_test_data(0.05)
    ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_zlim(-100, 100)

@mpl3d_image_comparison(['contourf3d_fill.png'], style='mpl20')
def test_contourf3d_fill():
    if False:
        print('Hello World!')
    plt.rcParams['axes3d.automargin'] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    (X, Y) = np.meshgrid(np.arange(-2, 2, 0.25), np.arange(-2, 2, 0.25))
    Z = X.clip(0, 0)
    Z[::5, ::5] = 0.1
    ax.contourf(X, Y, Z, offset=0, levels=[-0.1, 0], cmap=cm.coolwarm)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-1, 1)

@pytest.mark.parametrize('extend, levels', [['both', [2, 4, 6]], ['min', [2, 4, 6, 8]], ['max', [0, 2, 4, 6]]])
@check_figures_equal(extensions=['png'])
def test_contourf3d_extend(fig_test, fig_ref, extend, levels):
    if False:
        i = 10
        return i + 15
    (X, Y) = np.meshgrid(np.arange(-2, 2, 0.25), np.arange(-2, 2, 0.25))
    Z = X ** 2 + Y ** 2
    cmap = mpl.colormaps['viridis'].copy()
    cmap.set_under(cmap(0))
    cmap.set_over(cmap(255))
    kwargs = {'vmin': 1, 'vmax': 7, 'cmap': cmap}
    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.contourf(X, Y, Z, levels=[0, 2, 4, 6, 8], **kwargs)
    ax_test = fig_test.add_subplot(projection='3d')
    ax_test.contourf(X, Y, Z, levels, extend=extend, **kwargs)
    for ax in [ax_ref, ax_test]:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-10, 10)

@mpl3d_image_comparison(['tricontour.png'], tol=0.02, style='mpl20')
def test_tricontour():
    if False:
        i = 10
        return i + 15
    plt.rcParams['axes3d.automargin'] = True
    fig = plt.figure()
    np.random.seed(19680801)
    x = np.random.rand(1000) - 0.5
    y = np.random.rand(1000) - 0.5
    z = -(x ** 2 + y ** 2)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.tricontour(x, y, z)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.tricontourf(x, y, z)

def test_contour3d_1d_input():
    if False:
        for i in range(10):
            print('nop')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    (nx, ny) = (30, 20)
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-10, 10, ny)
    z = np.random.randint(0, 2, [ny, nx])
    ax.contour(x, y, z, [0.5])

@mpl3d_image_comparison(['lines3d.png'], style='mpl20')
def test_lines3d():
    if False:
        print('Hello World!')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z ** 2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    ax.plot(x, y, z)

@check_figures_equal(extensions=['png'])
def test_plot_scalar(fig_test, fig_ref):
    if False:
        for i in range(10):
            print('nop')
    ax1 = fig_test.add_subplot(projection='3d')
    ax1.plot([1], [1], 'o')
    ax2 = fig_ref.add_subplot(projection='3d')
    ax2.plot(1, 1, 'o')

def test_invalid_line_data():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(RuntimeError, match='x must be'):
        art3d.Line3D(0, [], [])
    with pytest.raises(RuntimeError, match='y must be'):
        art3d.Line3D([], 0, [])
    with pytest.raises(RuntimeError, match='z must be'):
        art3d.Line3D([], [], 0)
    line = art3d.Line3D([], [], [])
    with pytest.raises(RuntimeError, match='x must be'):
        line.set_data_3d(0, [], [])
    with pytest.raises(RuntimeError, match='y must be'):
        line.set_data_3d([], 0, [])
    with pytest.raises(RuntimeError, match='z must be'):
        line.set_data_3d([], [], 0)

@mpl3d_image_comparison(['mixedsubplot.png'], style='mpl20')
def test_mixedsubplots():
    if False:
        return 10

    def f(t):
        if False:
            for i in range(10):
                print('nop')
        return np.cos(2 * np.pi * t) * np.exp(-t)
    t1 = np.arange(0.0, 5.0, 0.1)
    t2 = np.arange(0.0, 5.0, 0.02)
    plt.rcParams['axes3d.automargin'] = True
    fig = plt.figure(figsize=plt.figaspect(2.0))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(t1, f(t1), 'bo', t2, f(t2), 'k--', markerfacecolor='green')
    ax.grid(True)
    ax = fig.add_subplot(2, 1, 2, projection='3d')
    (X, Y) = np.meshgrid(np.arange(-5, 5, 0.25), np.arange(-5, 5, 0.25))
    R = np.hypot(X, Y)
    Z = np.sin(R)
    ax.plot_surface(X, Y, Z, rcount=40, ccount=40, linewidth=0, antialiased=False)
    ax.set_zlim3d(-1, 1)

@check_figures_equal(extensions=['png'])
def test_tight_layout_text(fig_test, fig_ref):
    if False:
        for i in range(10):
            print('nop')
    ax1 = fig_test.add_subplot(projection='3d')
    ax1.text(0.5, 0.5, 0.5, s='some string')
    fig_test.tight_layout()
    ax2 = fig_ref.add_subplot(projection='3d')
    fig_ref.tight_layout()
    ax2.text(0.5, 0.5, 0.5, s='some string')

@mpl3d_image_comparison(['scatter3d.png'], style='mpl20')
def test_scatter3d():
    if False:
        print('Hello World!')
    plt.rcParams['axes3d.automargin'] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(np.arange(10), np.arange(10), np.arange(10), c='r', marker='o')
    x = y = z = np.arange(10, 20)
    ax.scatter(x, y, z, c='b', marker='^')
    z[-1] = 0
    ax.scatter([], [], [], c='r', marker='X')

@mpl3d_image_comparison(['scatter3d_color.png'], style='mpl20')
def test_scatter3d_color():
    if False:
        while True:
            i = 10
    plt.rcParams['axes3d.automargin'] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(np.arange(10), np.arange(10), np.arange(10), facecolor='r', edgecolor='none', marker='o')
    ax.scatter(np.arange(10), np.arange(10), np.arange(10), facecolor='none', edgecolor='r', marker='o')
    ax.scatter(np.arange(10, 20), np.arange(10, 20), np.arange(10, 20), color='b', marker='s')

@mpl3d_image_comparison(['scatter3d_linewidth.png'], style='mpl20')
def test_scatter3d_linewidth():
    if False:
        print('Hello World!')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(np.arange(10), np.arange(10), np.arange(10), marker='o', linewidth=np.arange(10))

@check_figures_equal(extensions=['png'])
def test_scatter3d_linewidth_modification(fig_ref, fig_test):
    if False:
        print('Hello World!')
    ax_test = fig_test.add_subplot(projection='3d')
    c = ax_test.scatter(np.arange(10), np.arange(10), np.arange(10), marker='o')
    c.set_linewidths(np.arange(10))
    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.scatter(np.arange(10), np.arange(10), np.arange(10), marker='o', linewidths=np.arange(10))

@check_figures_equal(extensions=['png'])
def test_scatter3d_modification(fig_ref, fig_test):
    if False:
        for i in range(10):
            print('nop')
    ax_test = fig_test.add_subplot(projection='3d')
    c = ax_test.scatter(np.arange(10), np.arange(10), np.arange(10), marker='o')
    c.set_facecolor('C1')
    c.set_edgecolor('C2')
    c.set_alpha([0.3, 0.7] * 5)
    assert c.get_depthshade()
    c.set_depthshade(False)
    assert not c.get_depthshade()
    c.set_sizes(np.full(10, 75))
    c.set_linewidths(3)
    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.scatter(np.arange(10), np.arange(10), np.arange(10), marker='o', facecolor='C1', edgecolor='C2', alpha=[0.3, 0.7] * 5, depthshade=False, s=75, linewidths=3)

@pytest.mark.parametrize('depthshade', [True, False])
@check_figures_equal(extensions=['png'])
def test_scatter3d_sorting(fig_ref, fig_test, depthshade):
    if False:
        print('Hello World!')
    'Test that marker properties are correctly sorted.'
    (y, x) = np.mgrid[:10, :10]
    z = np.arange(x.size).reshape(x.shape)
    sizes = np.full(z.shape, 25)
    sizes[0::2, 0::2] = 100
    sizes[1::2, 1::2] = 100
    facecolors = np.full(z.shape, 'C0')
    facecolors[:5, :5] = 'C1'
    facecolors[6:, :4] = 'C2'
    facecolors[6:, 6:] = 'C3'
    edgecolors = np.full(z.shape, 'C4')
    edgecolors[1:5, 1:5] = 'C5'
    edgecolors[5:9, 1:5] = 'C6'
    edgecolors[5:9, 5:9] = 'C7'
    linewidths = np.full(z.shape, 2)
    linewidths[0::2, 0::2] = 5
    linewidths[1::2, 1::2] = 5
    (x, y, z, sizes, facecolors, edgecolors, linewidths) = [a.flatten() for a in [x, y, z, sizes, facecolors, edgecolors, linewidths]]
    ax_ref = fig_ref.add_subplot(projection='3d')
    sets = (np.unique(a) for a in [sizes, facecolors, edgecolors, linewidths])
    for (s, fc, ec, lw) in itertools.product(*sets):
        subset = (sizes != s) | (facecolors != fc) | (edgecolors != ec) | (linewidths != lw)
        subset = np.ma.masked_array(z, subset, dtype=float)
        fc = np.repeat(fc, sum(~subset.mask))
        ax_ref.scatter(x, y, subset, s=s, fc=fc, ec=ec, lw=lw, alpha=1, depthshade=depthshade)
    ax_test = fig_test.add_subplot(projection='3d')
    ax_test.scatter(x, y, z, s=sizes, fc=facecolors, ec=edgecolors, lw=linewidths, alpha=1, depthshade=depthshade)

@pytest.mark.parametrize('azim', [-50, 130])
@check_figures_equal(extensions=['png'])
def test_marker_draw_order_data_reversed(fig_test, fig_ref, azim):
    if False:
        while True:
            i = 10
    '\n    Test that the draw order does not depend on the data point order.\n\n    For the given viewing angle at azim=-50, the yellow marker should be in\n    front. For azim=130, the blue marker should be in front.\n    '
    x = [-1, 1]
    y = [1, -1]
    z = [0, 0]
    color = ['b', 'y']
    ax = fig_test.add_subplot(projection='3d')
    ax.scatter(x, y, z, s=3500, c=color)
    ax.view_init(elev=0, azim=azim, roll=0)
    ax = fig_ref.add_subplot(projection='3d')
    ax.scatter(x[::-1], y[::-1], z[::-1], s=3500, c=color[::-1])
    ax.view_init(elev=0, azim=azim, roll=0)

@check_figures_equal(extensions=['png'])
def test_marker_draw_order_view_rotated(fig_test, fig_ref):
    if False:
        i = 10
        return i + 15
    '\n    Test that the draw order changes with the direction.\n\n    If we rotate *azim* by 180 degrees and exchange the colors, the plot\n    plot should look the same again.\n    '
    azim = 130
    x = [-1, 1]
    y = [1, -1]
    z = [0, 0]
    color = ['b', 'y']
    ax = fig_test.add_subplot(projection='3d')
    ax.set_axis_off()
    ax.scatter(x, y, z, s=3500, c=color)
    ax.view_init(elev=0, azim=azim, roll=0)
    ax = fig_ref.add_subplot(projection='3d')
    ax.set_axis_off()
    ax.scatter(x, y, z, s=3500, c=color[::-1])
    ax.view_init(elev=0, azim=azim - 180, roll=0)

@mpl3d_image_comparison(['plot_3d_from_2d.png'], tol=0.015, style='mpl20')
def test_plot_3d_from_2d():
    if False:
        i = 10
        return i + 15
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = np.arange(0, 5)
    ys = np.arange(5, 10)
    ax.plot(xs, ys, zs=0, zdir='x')
    ax.plot(xs, ys, zs=0, zdir='y')

@mpl3d_image_comparison(['surface3d.png'], style='mpl20')
def test_surface3d():
    if False:
        i = 10
        return i + 15
    plt.rcParams['pcolormesh.snap'] = False
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    (X, Y) = np.meshgrid(X, Y)
    R = np.hypot(X, Y)
    Z = np.sin(R)
    surf = ax.plot_surface(X, Y, Z, rcount=40, ccount=40, cmap=cm.coolwarm, lw=0, antialiased=False)
    plt.rcParams['axes3d.automargin'] = True
    ax.set_zlim(-1.01, 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=5)

@image_comparison(['surface3d_label_offset_tick_position.png'], style='mpl20')
def test_surface3d_label_offset_tick_position():
    if False:
        print('Hello World!')
    plt.rcParams['axes3d.automargin'] = True
    ax = plt.figure().add_subplot(projection='3d')
    (x, y) = np.mgrid[0:6 * np.pi:0.25, 0:4 * np.pi:0.25]
    z = np.sqrt(np.abs(np.cos(x) + np.cos(y)))
    ax.plot_surface(x * 100000.0, y * 1000000.0, z * 100000000.0, cmap='autumn', cstride=2, rstride=2)
    ax.set_xlabel('X label')
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')
    ax.figure.canvas.draw()

@mpl3d_image_comparison(['surface3d_shaded.png'], style='mpl20')
def test_surface3d_shaded():
    if False:
        print('Hello World!')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    (X, Y) = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)
    ax.plot_surface(X, Y, Z, rstride=5, cstride=5, color=[0.25, 1, 0.25], lw=1, antialiased=False)
    plt.rcParams['axes3d.automargin'] = True
    ax.set_zlim(-1.01, 1.01)

@mpl3d_image_comparison(['surface3d_masked.png'], style='mpl20')
def test_surface3d_masked():
    if False:
        print('Hello World!')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    y = [1, 2, 3, 4, 5, 6, 7, 8]
    (x, y) = np.meshgrid(x, y)
    matrix = np.array([[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [-1, 1, 2, 3, 4, 4, 4, 3, 2, 1, 1], [-1, -1.0, 4, 5, 6, 8, 6, 5, 4, 3, -1.0], [-1, -1.0, 7, 8, 11, 12, 11, 8, 7, -1.0, -1.0], [-1, -1.0, 8, 9, 10, 16, 10, 9, 10, 7, -1.0], [-1, -1.0, -1.0, 12, 16, 20, 16, 12, 11, -1.0, -1.0], [-1, -1.0, -1.0, -1.0, 22, 24, 22, 20, 18, -1.0, -1.0], [-1, -1.0, -1.0, -1.0, -1.0, 28, 26, 25, -1.0, -1.0, -1.0]])
    z = np.ma.masked_less(matrix, 0)
    norm = mcolors.Normalize(vmax=z.max(), vmin=z.min())
    colors = mpl.colormaps['plasma'](norm(z))
    ax.plot_surface(x, y, z, facecolors=colors)
    ax.view_init(30, -80, 0)

@check_figures_equal(extensions=['png'])
def test_plot_scatter_masks(fig_test, fig_ref):
    if False:
        return 10
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    z = np.sin(x) * np.cos(y)
    mask = z > 0
    z_masked = np.ma.array(z, mask=mask)
    ax_test = fig_test.add_subplot(projection='3d')
    ax_test.scatter(x, y, z_masked)
    ax_test.plot(x, y, z_masked)
    x[mask] = y[mask] = z[mask] = np.nan
    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.scatter(x, y, z)
    ax_ref.plot(x, y, z)

@check_figures_equal(extensions=['png'])
def test_plot_surface_None_arg(fig_test, fig_ref):
    if False:
        return 10
    (x, y) = np.meshgrid(np.arange(5), np.arange(5))
    z = x + y
    ax_test = fig_test.add_subplot(projection='3d')
    ax_test.plot_surface(x, y, z, facecolors=None)
    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.plot_surface(x, y, z)

@mpl3d_image_comparison(['surface3d_masked_strides.png'], style='mpl20')
def test_surface3d_masked_strides():
    if False:
        while True:
            i = 10
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    (x, y) = np.mgrid[-6:6.1:1, -6:6.1:1]
    z = np.ma.masked_less(x * y, 2)
    ax.plot_surface(x, y, z, rstride=4, cstride=4)
    ax.view_init(60, -45, 0)

@mpl3d_image_comparison(['text3d.png'], remove_text=False, style='mpl20')
def test_text3d():
    if False:
        return 10
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    zdirs = (None, 'x', 'y', 'z', (1, 1, 0), (1, 1, 1))
    xs = (2, 6, 4, 9, 7, 2)
    ys = (6, 4, 8, 7, 2, 2)
    zs = (4, 2, 5, 6, 1, 7)
    for (zdir, x, y, z) in zip(zdirs, xs, ys, zs):
        label = '(%d, %d, %d), dir=%s' % (x, y, z, zdir)
        ax.text(x, y, z, label, zdir)
    ax.text(1, 1, 1, 'red', color='red')
    ax.text2D(0.05, 0.95, '2D Text', transform=ax.transAxes)
    plt.rcParams['axes3d.automargin'] = True
    ax.set_xlim3d(0, 10)
    ax.set_ylim3d(0, 10)
    ax.set_zlim3d(0, 10)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

@check_figures_equal(extensions=['png'])
def test_text3d_modification(fig_ref, fig_test):
    if False:
        print('Hello World!')
    zdirs = (None, 'x', 'y', 'z', (1, 1, 0), (1, 1, 1))
    xs = (2, 6, 4, 9, 7, 2)
    ys = (6, 4, 8, 7, 2, 2)
    zs = (4, 2, 5, 6, 1, 7)
    ax_test = fig_test.add_subplot(projection='3d')
    ax_test.set_xlim3d(0, 10)
    ax_test.set_ylim3d(0, 10)
    ax_test.set_zlim3d(0, 10)
    for (zdir, x, y, z) in zip(zdirs, xs, ys, zs):
        t = ax_test.text(0, 0, 0, f'({x}, {y}, {z}), dir={zdir}')
        t.set_position_3d((x, y, z), zdir=zdir)
    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.set_xlim3d(0, 10)
    ax_ref.set_ylim3d(0, 10)
    ax_ref.set_zlim3d(0, 10)
    for (zdir, x, y, z) in zip(zdirs, xs, ys, zs):
        ax_ref.text(x, y, z, f'({x}, {y}, {z}), dir={zdir}', zdir=zdir)

@mpl3d_image_comparison(['trisurf3d.png'], tol=0.061, style='mpl20')
def test_trisurf3d():
    if False:
        for i in range(10):
            print('nop')
    n_angles = 36
    n_radii = 8
    radii = np.linspace(0.125, 1.0, n_radii)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += np.pi / n_angles
    x = np.append(0, (radii * np.cos(angles)).flatten())
    y = np.append(0, (radii * np.sin(angles)).flatten())
    z = np.sin(-x * y)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)

@mpl3d_image_comparison(['trisurf3d_shaded.png'], tol=0.03, style='mpl20')
def test_trisurf3d_shaded():
    if False:
        while True:
            i = 10
    n_angles = 36
    n_radii = 8
    radii = np.linspace(0.125, 1.0, n_radii)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += np.pi / n_angles
    x = np.append(0, (radii * np.cos(angles)).flatten())
    y = np.append(0, (radii * np.sin(angles)).flatten())
    z = np.sin(-x * y)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(x, y, z, color=[1, 0.5, 0], linewidth=0.2)

@mpl3d_image_comparison(['wireframe3d.png'], style='mpl20')
def test_wireframe3d():
    if False:
        return 10
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    (X, Y, Z) = axes3d.get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rcount=13, ccount=13)

@mpl3d_image_comparison(['wireframe3dzerocstride.png'], style='mpl20')
def test_wireframe3dzerocstride():
    if False:
        for i in range(10):
            print('nop')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    (X, Y, Z) = axes3d.get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rcount=13, ccount=0)

@mpl3d_image_comparison(['wireframe3dzerorstride.png'], style='mpl20')
def test_wireframe3dzerorstride():
    if False:
        for i in range(10):
            print('nop')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    (X, Y, Z) = axes3d.get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rstride=0, cstride=10)

def test_wireframe3dzerostrideraises():
    if False:
        for i in range(10):
            print('nop')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    (X, Y, Z) = axes3d.get_test_data(0.05)
    with pytest.raises(ValueError):
        ax.plot_wireframe(X, Y, Z, rstride=0, cstride=0)

def test_mixedsamplesraises():
    if False:
        i = 10
        return i + 15
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    (X, Y, Z) = axes3d.get_test_data(0.05)
    with pytest.raises(ValueError):
        ax.plot_wireframe(X, Y, Z, rstride=10, ccount=50)
    with pytest.raises(ValueError):
        ax.plot_surface(X, Y, Z, cstride=50, rcount=10)

@mpl3d_image_comparison(['quiver3d.png'], style='mpl20')
def test_quiver3d():
    if False:
        for i in range(10):
            print('nop')
    plt.rcParams['axes3d.automargin'] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    pivots = ['tip', 'middle', 'tail']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for (i, (pivot, color)) in enumerate(zip(pivots, colors)):
        (x, y, z) = np.meshgrid([-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5])
        u = -x
        v = -y
        w = -z
        z += 2 * i
        ax.quiver(x, y, z, u, v, w, length=1, pivot=pivot, color=color)
        ax.scatter(x, y, z, color=color)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-1, 5)

@check_figures_equal(extensions=['png'])
def test_quiver3d_empty(fig_test, fig_ref):
    if False:
        for i in range(10):
            print('nop')
    fig_ref.add_subplot(projection='3d')
    x = y = z = u = v = w = []
    ax = fig_test.add_subplot(projection='3d')
    ax.quiver(x, y, z, u, v, w, length=0.1, pivot='tip', normalize=True)

@mpl3d_image_comparison(['quiver3d_masked.png'], style='mpl20')
def test_quiver3d_masked():
    if False:
        i = 10
        return i + 15
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    (x, y, z) = np.mgrid[-1:0.8:10j, -1:0.8:10j, -1:0.6:3j]
    u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    w = (2 / 3) ** 0.5 * np.cos(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z)
    u = np.ma.masked_where((-0.4 < x) & (x < 0.1), u, copy=False)
    v = np.ma.masked_where((0.1 < y) & (y < 0.7), v, copy=False)
    ax.quiver(x, y, z, u, v, w, length=0.1, pivot='tip', normalize=True)

def test_patch_modification():
    if False:
        while True:
            i = 10
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    circle = Circle((0, 0))
    ax.add_patch(circle)
    art3d.patch_2d_to_3d(circle)
    circle.set_facecolor((1.0, 0.0, 0.0, 1))
    assert mcolors.same_color(circle.get_facecolor(), (1, 0, 0, 1))
    fig.canvas.draw()
    assert mcolors.same_color(circle.get_facecolor(), (1, 0, 0, 1))

@check_figures_equal(extensions=['png'])
def test_patch_collection_modification(fig_test, fig_ref):
    if False:
        print('Hello World!')
    patch1 = Circle((0, 0), 0.05)
    patch2 = Circle((0.1, 0.1), 0.03)
    facecolors = np.array([[0.0, 0.5, 0.0, 1.0], [0.5, 0.0, 0.0, 0.5]])
    c = art3d.Patch3DCollection([patch1, patch2], linewidths=3)
    ax_test = fig_test.add_subplot(projection='3d')
    ax_test.add_collection3d(c)
    c.set_edgecolor('C2')
    c.set_facecolor(facecolors)
    c.set_alpha(0.7)
    assert c.get_depthshade()
    c.set_depthshade(False)
    assert not c.get_depthshade()
    patch1 = Circle((0, 0), 0.05)
    patch2 = Circle((0.1, 0.1), 0.03)
    facecolors = np.array([[0.0, 0.5, 0.0, 1.0], [0.5, 0.0, 0.0, 0.5]])
    c = art3d.Patch3DCollection([patch1, patch2], linewidths=3, edgecolor='C2', facecolor=facecolors, alpha=0.7, depthshade=False)
    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.add_collection3d(c)

def test_poly3dcollection_verts_validation():
    if False:
        print('Hello World!')
    poly = [[0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]]
    with pytest.raises(ValueError, match='list of \\(N, 3\\) array-like'):
        art3d.Poly3DCollection(poly)
    poly = np.array(poly, dtype=float)
    with pytest.raises(ValueError, match='list of \\(N, 3\\) array-like'):
        art3d.Poly3DCollection(poly)

@mpl3d_image_comparison(['poly3dcollection_closed.png'], style='mpl20')
def test_poly3dcollection_closed():
    if False:
        i = 10
        return i + 15
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    poly1 = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]], float)
    poly2 = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], float)
    c1 = art3d.Poly3DCollection([poly1], linewidths=3, edgecolor='k', facecolor=(0.5, 0.5, 1, 0.5), closed=True)
    c2 = art3d.Poly3DCollection([poly2], linewidths=3, edgecolor='k', facecolor=(1, 0.5, 0.5, 0.5), closed=False)
    ax.add_collection3d(c1)
    ax.add_collection3d(c2)

def test_poly_collection_2d_to_3d_empty():
    if False:
        while True:
            i = 10
    poly = PolyCollection([])
    art3d.poly_collection_2d_to_3d(poly)
    assert isinstance(poly, art3d.Poly3DCollection)
    assert poly.get_paths() == []
    (fig, ax) = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.add_artist(poly)
    minz = poly.do_3d_projection()
    assert np.isnan(minz)
    fig.canvas.draw()

@mpl3d_image_comparison(['poly3dcollection_alpha.png'], style='mpl20')
def test_poly3dcollection_alpha():
    if False:
        for i in range(10):
            print('nop')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    poly1 = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]], float)
    poly2 = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], float)
    c1 = art3d.Poly3DCollection([poly1], linewidths=3, edgecolor='k', facecolor=(0.5, 0.5, 1), closed=True)
    c1.set_alpha(0.5)
    c2 = art3d.Poly3DCollection([poly2], linewidths=3, closed=False)
    c2.set_facecolor((1, 0.5, 0.5))
    c2.set_edgecolor('k')
    c2.set_alpha(0.5)
    ax.add_collection3d(c1)
    ax.add_collection3d(c2)

@mpl3d_image_comparison(['add_collection3d_zs_array.png'], style='mpl20')
def test_add_collection3d_zs_array():
    if False:
        for i in range(10):
            print('nop')
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z ** 2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    points = np.column_stack([x, y, z]).reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    norm = plt.Normalize(0, 2 * np.pi)
    lc = LineCollection(segments[:, :, :2], cmap='twilight', norm=norm)
    lc.set_array(np.mod(theta, 2 * np.pi))
    line = ax.add_collection3d(lc, zs=segments[:, :, 2])
    assert line is not None
    plt.rcParams['axes3d.automargin'] = True
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 6)
    ax.set_zlim(-2, 2)

@mpl3d_image_comparison(['add_collection3d_zs_scalar.png'], style='mpl20')
def test_add_collection3d_zs_scalar():
    if False:
        for i in range(10):
            print('nop')
    theta = np.linspace(0, 2 * np.pi, 100)
    z = 1
    r = z ** 2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    norm = plt.Normalize(0, 2 * np.pi)
    lc = LineCollection(segments, cmap='twilight', norm=norm)
    lc.set_array(theta)
    line = ax.add_collection3d(lc, zs=z)
    assert line is not None
    plt.rcParams['axes3d.automargin'] = True
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 6)
    ax.set_zlim(0, 2)

@mpl3d_image_comparison(['axes3d_labelpad.png'], remove_text=False, style='mpl20')
def test_axes3d_labelpad():
    if False:
        i = 10
        return i + 15
    fig = plt.figure()
    ax = fig.add_axes(Axes3D(fig))
    assert ax.xaxis.labelpad == mpl.rcParams['axes.labelpad']
    ax.set_xlabel('X LABEL', labelpad=10)
    assert ax.xaxis.labelpad == 10
    ax.set_ylabel('Y LABEL')
    ax.set_zlabel('Z LABEL', labelpad=20)
    assert ax.zaxis.labelpad == 20
    assert ax.get_zlabel() == 'Z LABEL'
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = -40
    for (i, tick) in enumerate(ax.yaxis.get_major_ticks()):
        tick.set_pad(tick.get_pad() + 5 - i * 5)

@mpl3d_image_comparison(['axes3d_cla.png'], remove_text=False, style='mpl20')
def test_axes3d_cla():
    if False:
        while True:
            i = 10
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_axis_off()
    ax.cla()

@mpl3d_image_comparison(['axes3d_rotated.png'], remove_text=False, style='mpl20')
def test_axes3d_rotated():
    if False:
        return 10
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(90, 45, 0)

def test_plotsurface_1d_raises():
    if False:
        return 10
    x = np.linspace(0.5, 10, num=100)
    y = np.linspace(0.5, 10, num=100)
    (X, Y) = np.meshgrid(x, y)
    z = np.random.randn(100)
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    with pytest.raises(ValueError):
        ax.plot_surface(X, Y, z)

def _test_proj_make_M():
    if False:
        print('Hello World!')
    E = np.array([1000, -1000, 2000])
    R = np.array([100, 100, 100])
    V = np.array([0, 0, 1])
    roll = 0
    (u, v, w) = proj3d._view_axes(E, R, V, roll)
    viewM = proj3d._view_transformation_uvw(u, v, w, E)
    perspM = proj3d._persp_transformation(100, -100, 1)
    M = np.dot(perspM, viewM)
    return M

def test_proj_transform():
    if False:
        while True:
            i = 10
    M = _test_proj_make_M()
    invM = np.linalg.inv(M)
    xs = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0]) * 300.0
    ys = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0]) * 300.0
    zs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) * 300.0
    (txs, tys, tzs) = proj3d.proj_transform(xs, ys, zs, M)
    (ixs, iys, izs) = proj3d.inv_transform(txs, tys, tzs, invM)
    np.testing.assert_almost_equal(ixs, xs)
    np.testing.assert_almost_equal(iys, ys)
    np.testing.assert_almost_equal(izs, zs)

def _test_proj_draw_axes(M, s=1, *args, **kwargs):
    if False:
        print('Hello World!')
    xs = [0, s, 0, 0]
    ys = [0, 0, s, 0]
    zs = [0, 0, 0, s]
    (txs, tys, tzs) = proj3d.proj_transform(xs, ys, zs, M)
    (o, ax, ay, az) = zip(txs, tys)
    lines = [(o, ax), (o, ay), (o, az)]
    (fig, ax) = plt.subplots(*args, **kwargs)
    linec = LineCollection(lines)
    ax.add_collection(linec)
    for (x, y, t) in zip(txs, tys, ['o', 'x', 'y', 'z']):
        ax.text(x, y, t)
    return (fig, ax)

@mpl3d_image_comparison(['proj3d_axes_cube.png'], style='mpl20')
def test_proj_axes_cube():
    if False:
        return 10
    M = _test_proj_make_M()
    ts = '0 1 2 3 0 4 5 6 7 4'.split()
    xs = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0]) * 300.0
    ys = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0]) * 300.0
    zs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) * 300.0
    (txs, tys, tzs) = proj3d.proj_transform(xs, ys, zs, M)
    (fig, ax) = _test_proj_draw_axes(M, s=400)
    ax.scatter(txs, tys, c=tzs)
    ax.plot(txs, tys, c='r')
    for (x, y, t) in zip(txs, tys, ts):
        ax.text(x, y, t)
    plt.rcParams['axes3d.automargin'] = True
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)

@mpl3d_image_comparison(['proj3d_axes_cube_ortho.png'], style='mpl20')
def test_proj_axes_cube_ortho():
    if False:
        print('Hello World!')
    E = np.array([200, 100, 100])
    R = np.array([0, 0, 0])
    V = np.array([0, 0, 1])
    roll = 0
    (u, v, w) = proj3d._view_axes(E, R, V, roll)
    viewM = proj3d._view_transformation_uvw(u, v, w, E)
    orthoM = proj3d._ortho_transformation(-1, 1)
    M = np.dot(orthoM, viewM)
    ts = '0 1 2 3 0 4 5 6 7 4'.split()
    xs = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0]) * 100
    ys = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0]) * 100
    zs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) * 100
    (txs, tys, tzs) = proj3d.proj_transform(xs, ys, zs, M)
    (fig, ax) = _test_proj_draw_axes(M, s=150)
    ax.scatter(txs, tys, s=300 - tzs)
    ax.plot(txs, tys, c='r')
    for (x, y, t) in zip(txs, tys, ts):
        ax.text(x, y, t)
    plt.rcParams['axes3d.automargin'] = True
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)

def test_world():
    if False:
        while True:
            i = 10
    (xmin, xmax) = (100, 120)
    (ymin, ymax) = (-100, 100)
    (zmin, zmax) = (0.1, 0.2)
    M = proj3d.world_transformation(xmin, xmax, ymin, ymax, zmin, zmax)
    np.testing.assert_allclose(M, [[0.05, 0, 0, -5], [0, 0.005, 0, 0.5], [0, 0, 10.0, -1], [0, 0, 0, 1]])

def test_autoscale():
    if False:
        i = 10
        return i + 15
    (fig, ax) = plt.subplots(subplot_kw={'projection': '3d'})
    assert ax.get_zscale() == 'linear'
    ax._view_margin = 0
    ax.margins(x=0, y=0.1, z=0.2)
    ax.plot([0, 1], [0, 1], [0, 1])
    assert ax.get_w_lims() == (0, 1, -0.1, 1.1, -0.2, 1.2)
    ax.autoscale(False)
    ax.set_autoscalez_on(True)
    ax.plot([0, 2], [0, 2], [0, 2])
    assert ax.get_w_lims() == (0, 1, -0.1, 1.1, -0.4, 2.4)
    ax.autoscale(axis='x')
    ax.plot([0, 2], [0, 2], [0, 2])
    assert ax.get_w_lims() == (0, 2, -0.1, 1.1, -0.4, 2.4)

@pytest.mark.parametrize('axis', ('x', 'y', 'z'))
@pytest.mark.parametrize('auto', (True, False, None))
def test_unautoscale(axis, auto):
    if False:
        i = 10
        return i + 15
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = np.arange(100)
    y = np.linspace(-0.1, 0.1, 100)
    ax.scatter(x, y)
    get_autoscale_on = getattr(ax, f'get_autoscale{axis}_on')
    set_lim = getattr(ax, f'set_{axis}lim')
    get_lim = getattr(ax, f'get_{axis}lim')
    post_auto = get_autoscale_on() if auto is None else auto
    set_lim((-0.5, 0.5), auto=auto)
    assert post_auto == get_autoscale_on()
    fig.canvas.draw()
    np.testing.assert_array_equal(get_lim(), (-0.5, 0.5))

def test_axes3d_focal_length_checks():
    if False:
        i = 10
        return i + 15
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    with pytest.raises(ValueError):
        ax.set_proj_type('persp', focal_length=0)
    with pytest.raises(ValueError):
        ax.set_proj_type('ortho', focal_length=1)

@mpl3d_image_comparison(['axes3d_focal_length.png'], remove_text=False, style='mpl20')
def test_axes3d_focal_length():
    if False:
        return 10
    (fig, axs) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    axs[0].set_proj_type('persp', focal_length=np.inf)
    axs[1].set_proj_type('persp', focal_length=0.15)

@mpl3d_image_comparison(['axes3d_ortho.png'], remove_text=False, style='mpl20')
def test_axes3d_ortho():
    if False:
        return 10
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_proj_type('ortho')

@mpl3d_image_comparison(['axes3d_isometric.png'], style='mpl20')
def test_axes3d_isometric():
    if False:
        for i in range(10):
            print('nop')
    from itertools import combinations, product
    (fig, ax) = plt.subplots(subplot_kw=dict(projection='3d', proj_type='ortho', box_aspect=(4, 4, 4)))
    r = (-1, 1)
    for (s, e) in combinations(np.array(list(product(r, r, r))), 2):
        if abs(s - e).sum() == r[1] - r[0]:
            ax.plot3D(*zip(s, e), c='k')
    ax.view_init(elev=np.degrees(np.arctan(1.0 / np.sqrt(2))), azim=-45, roll=0)
    ax.grid(True)

@pytest.mark.parametrize('value', [np.inf, np.nan])
@pytest.mark.parametrize(('setter', 'side'), [('set_xlim3d', 'left'), ('set_xlim3d', 'right'), ('set_ylim3d', 'bottom'), ('set_ylim3d', 'top'), ('set_zlim3d', 'bottom'), ('set_zlim3d', 'top')])
def test_invalid_axes_limits(setter, side, value):
    if False:
        while True:
            i = 10
    limit = {side: value}
    fig = plt.figure()
    obj = fig.add_subplot(projection='3d')
    with pytest.raises(ValueError):
        getattr(obj, setter)(**limit)

class TestVoxels:

    @mpl3d_image_comparison(['voxels-simple.png'], style='mpl20')
    def test_simple(self):
        if False:
            while True:
                i = 10
        (fig, ax) = plt.subplots(subplot_kw={'projection': '3d'})
        (x, y, z) = np.indices((5, 4, 3))
        voxels = (x == y) | (y == z)
        ax.voxels(voxels)

    @mpl3d_image_comparison(['voxels-edge-style.png'], style='mpl20')
    def test_edge_style(self):
        if False:
            for i in range(10):
                print('nop')
        (fig, ax) = plt.subplots(subplot_kw={'projection': '3d'})
        (x, y, z) = np.indices((5, 5, 4))
        voxels = (x - 2) ** 2 + (y - 2) ** 2 + (z - 1.5) ** 2 < 2.2 ** 2
        v = ax.voxels(voxels, linewidths=3, edgecolor='C1')
        v[max(v.keys())].set_edgecolor('C2')

    @mpl3d_image_comparison(['voxels-named-colors.png'], style='mpl20')
    def test_named_colors(self):
        if False:
            while True:
                i = 10
        'Test with colors set to a 3D object array of strings.'
        (fig, ax) = plt.subplots(subplot_kw={'projection': '3d'})
        (x, y, z) = np.indices((10, 10, 10))
        voxels = (x == y) | (y == z)
        voxels = voxels & ~(x * y * z < 1)
        colors = np.full((10, 10, 10), 'C0', dtype=np.object_)
        colors[(x < 5) & (y < 5)] = '0.25'
        colors[x + z < 10] = 'cyan'
        ax.voxels(voxels, facecolors=colors)

    @mpl3d_image_comparison(['voxels-rgb-data.png'], style='mpl20')
    def test_rgb_data(self):
        if False:
            return 10
        'Test with colors set to a 4d float array of rgb data.'
        (fig, ax) = plt.subplots(subplot_kw={'projection': '3d'})
        (x, y, z) = np.indices((10, 10, 10))
        voxels = (x == y) | (y == z)
        colors = np.zeros((10, 10, 10, 3))
        colors[..., 0] = x / 9
        colors[..., 1] = y / 9
        colors[..., 2] = z / 9
        ax.voxels(voxels, facecolors=colors)

    @mpl3d_image_comparison(['voxels-alpha.png'], style='mpl20')
    def test_alpha(self):
        if False:
            print('Hello World!')
        (fig, ax) = plt.subplots(subplot_kw={'projection': '3d'})
        (x, y, z) = np.indices((10, 10, 10))
        v1 = x == y
        v2 = np.abs(x - y) < 2
        voxels = v1 | v2
        colors = np.zeros((10, 10, 10, 4))
        colors[v2] = [1, 0, 0, 0.5]
        colors[v1] = [0, 1, 0, 0.5]
        v = ax.voxels(voxels, facecolors=colors)
        assert type(v) is dict
        for (coord, poly) in v.items():
            assert voxels[coord], 'faces returned for absent voxel'
            assert isinstance(poly, art3d.Poly3DCollection)

    @mpl3d_image_comparison(['voxels-xyz.png'], tol=0.01, remove_text=False, style='mpl20')
    def test_xyz(self):
        if False:
            print('Hello World!')
        (fig, ax) = plt.subplots(subplot_kw={'projection': '3d'})

        def midpoints(x):
            if False:
                return 10
            sl = ()
            for i in range(x.ndim):
                x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
                sl += np.index_exp[:]
            return x
        (r, g, b) = np.indices((17, 17, 17)) / 16.0
        rc = midpoints(r)
        gc = midpoints(g)
        bc = midpoints(b)
        sphere = (rc - 0.5) ** 2 + (gc - 0.5) ** 2 + (bc - 0.5) ** 2 < 0.5 ** 2
        colors = np.zeros(sphere.shape + (3,))
        colors[..., 0] = rc
        colors[..., 1] = gc
        colors[..., 2] = bc
        ax.voxels(r, g, b, sphere, facecolors=colors, edgecolors=np.clip(2 * colors - 0.5, 0, 1), linewidth=0.5)

    def test_calling_conventions(self):
        if False:
            for i in range(10):
                print('nop')
        (x, y, z) = np.indices((3, 4, 5))
        filled = np.ones((2, 3, 4))
        (fig, ax) = plt.subplots(subplot_kw={'projection': '3d'})
        for kw in (dict(), dict(edgecolor='k')):
            ax.voxels(filled, **kw)
            ax.voxels(filled=filled, **kw)
            ax.voxels(x, y, z, filled, **kw)
            ax.voxels(x, y, z, filled=filled, **kw)
        with pytest.raises(TypeError, match='voxels'):
            ax.voxels(x, y, z, filled, filled=filled)
        with pytest.raises(TypeError, match='voxels'):
            ax.voxels(x, y)
        with pytest.raises(AttributeError):
            ax.voxels(filled=filled, x=x, y=y, z=z)

def test_line3d_set_get_data_3d():
    if False:
        print('Hello World!')
    (x, y, z) = ([0, 1], [2, 3], [4, 5])
    (x2, y2, z2) = ([6, 7], [8, 9], [10, 11])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    lines = ax.plot(x, y, z)
    line = lines[0]
    np.testing.assert_array_equal((x, y, z), line.get_data_3d())
    line.set_data_3d(x2, y2, z2)
    np.testing.assert_array_equal((x2, y2, z2), line.get_data_3d())
    line.set_xdata(x)
    line.set_ydata(y)
    line.set_3d_properties(zs=z, zdir='z')
    np.testing.assert_array_equal((x, y, z), line.get_data_3d())
    line.set_3d_properties(zs=0, zdir='z')
    np.testing.assert_array_equal((x, y, np.zeros_like(z)), line.get_data_3d())

@check_figures_equal(extensions=['png'])
def test_inverted(fig_test, fig_ref):
    if False:
        for i in range(10):
            print('nop')
    ax = fig_test.add_subplot(projection='3d')
    ax.plot([1, 1, 10, 10], [1, 10, 10, 10], [1, 1, 1, 10])
    ax.invert_yaxis()
    ax = fig_ref.add_subplot(projection='3d')
    ax.invert_yaxis()
    ax.plot([1, 1, 10, 10], [1, 10, 10, 10], [1, 1, 1, 10])

def test_inverted_cla():
    if False:
        print('Hello World!')
    (fig, ax) = plt.subplots(subplot_kw={'projection': '3d'})
    assert not ax.xaxis_inverted()
    assert not ax.yaxis_inverted()
    assert not ax.zaxis_inverted()
    ax.set_xlim(1, 0)
    ax.set_ylim(1, 0)
    ax.set_zlim(1, 0)
    assert ax.xaxis_inverted()
    assert ax.yaxis_inverted()
    assert ax.zaxis_inverted()
    ax.cla()
    assert not ax.xaxis_inverted()
    assert not ax.yaxis_inverted()
    assert not ax.zaxis_inverted()

def test_ax3d_tickcolour():
    if False:
        for i in range(10):
            print('nop')
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='red')
    ax.tick_params(axis='z', colors='red')
    fig.canvas.draw()
    for tick in ax.xaxis.get_major_ticks():
        assert tick.tick1line._color == 'red'
    for tick in ax.yaxis.get_major_ticks():
        assert tick.tick1line._color == 'red'
    for tick in ax.zaxis.get_major_ticks():
        assert tick.tick1line._color == 'red'

@check_figures_equal(extensions=['png'])
def test_ticklabel_format(fig_test, fig_ref):
    if False:
        i = 10
        return i + 15
    axs = fig_test.subplots(4, 5, subplot_kw={'projection': '3d'})
    for ax in axs.flat:
        ax.set_xlim(10000000.0, 10000000.0 + 10)
    for (row, name) in zip(axs, ['x', 'y', 'z', 'both']):
        row[0].ticklabel_format(axis=name, style='plain')
        row[1].ticklabel_format(axis=name, scilimits=(-2, 2))
        row[2].ticklabel_format(axis=name, useOffset=not mpl.rcParams['axes.formatter.useoffset'])
        row[3].ticklabel_format(axis=name, useLocale=not mpl.rcParams['axes.formatter.use_locale'])
        row[4].ticklabel_format(axis=name, useMathText=not mpl.rcParams['axes.formatter.use_mathtext'])

    def get_formatters(ax, names):
        if False:
            i = 10
            return i + 15
        return [getattr(ax, name).get_major_formatter() for name in names]
    axs = fig_ref.subplots(4, 5, subplot_kw={'projection': '3d'})
    for ax in axs.flat:
        ax.set_xlim(10000000.0, 10000000.0 + 10)
    for (row, names) in zip(axs, [['xaxis'], ['yaxis'], ['zaxis'], ['xaxis', 'yaxis', 'zaxis']]):
        for fmt in get_formatters(row[0], names):
            fmt.set_scientific(False)
        for fmt in get_formatters(row[1], names):
            fmt.set_powerlimits((-2, 2))
        for fmt in get_formatters(row[2], names):
            fmt.set_useOffset(not mpl.rcParams['axes.formatter.useoffset'])
        for fmt in get_formatters(row[3], names):
            fmt.set_useLocale(not mpl.rcParams['axes.formatter.use_locale'])
        for fmt in get_formatters(row[4], names):
            fmt.set_useMathText(not mpl.rcParams['axes.formatter.use_mathtext'])

@check_figures_equal(extensions=['png'])
def test_quiver3D_smoke(fig_test, fig_ref):
    if False:
        while True:
            i = 10
    pivot = 'middle'
    (x, y, z) = np.meshgrid(np.arange(-0.8, 1, 0.2), np.arange(-0.8, 1, 0.2), np.arange(-0.8, 1, 0.8))
    u = v = w = np.ones_like(x)
    for (fig, length) in zip((fig_ref, fig_test), (1, 1.0)):
        ax = fig.add_subplot(projection='3d')
        ax.quiver(x, y, z, u, v, w, length=length, pivot=pivot)

@image_comparison(['minor_ticks.png'], style='mpl20')
def test_minor_ticks():
    if False:
        for i in range(10):
            print('nop')
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xticks([0.25], minor=True)
    ax.set_xticklabels(['quarter'], minor=True)
    ax.set_yticks([0.33], minor=True)
    ax.set_yticklabels(['third'], minor=True)
    ax.set_zticks([0.5], minor=True)
    ax.set_zticklabels(['half'], minor=True)

@mpl3d_image_comparison(['errorbar3d_errorevery.png'], style='mpl20')
def test_errorbar3d_errorevery():
    if False:
        for i in range(10):
            print('nop')
    'Tests errorevery functionality for 3D errorbars.'
    t = np.arange(0, 2 * np.pi + 0.1, 0.01)
    (x, y, z) = (np.sin(t), np.cos(3 * t), np.sin(5 * t))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    estep = 15
    i = np.arange(t.size)
    zuplims = (i % estep == 0) & (i // estep % 3 == 0)
    zlolims = (i % estep == 0) & (i // estep % 3 == 2)
    ax.errorbar(x, y, z, 0.2, zuplims=zuplims, zlolims=zlolims, errorevery=estep)

@mpl3d_image_comparison(['errorbar3d.png'], style='mpl20')
def test_errorbar3d():
    if False:
        while True:
            i = 10
    'Tests limits, color styling, and legend for 3D errorbars.'
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    d = [1, 2, 3, 4, 5]
    e = [0.5, 0.5, 0.5, 0.5, 0.5]
    ax.errorbar(x=d, y=d, z=d, xerr=e, yerr=e, zerr=e, capsize=3, zuplims=[False, True, False, True, True], zlolims=[True, False, False, True, False], yuplims=True, ecolor='purple', label='Error lines')
    ax.legend()

@image_comparison(['stem3d.png'], style='mpl20', tol=0.003)
def test_stem3d():
    if False:
        while True:
            i = 10
    plt.rcParams['axes3d.automargin'] = True
    (fig, axs) = plt.subplots(2, 3, figsize=(8, 6), constrained_layout=True, subplot_kw={'projection': '3d'})
    theta = np.linspace(0, 2 * np.pi)
    x = np.cos(theta - np.pi / 2)
    y = np.sin(theta - np.pi / 2)
    z = theta
    for (ax, zdir) in zip(axs[0], ['x', 'y', 'z']):
        ax.stem(x, y, z, orientation=zdir)
        ax.set_title(f'orientation={zdir}')
    x = np.linspace(-np.pi / 2, np.pi / 2, 20)
    y = np.ones_like(x)
    z = np.cos(x)
    for (ax, zdir) in zip(axs[1], ['x', 'y', 'z']):
        (markerline, stemlines, baseline) = ax.stem(x, y, z, linefmt='C4-.', markerfmt='C1D', basefmt='C2', orientation=zdir)
        ax.set_title(f'orientation={zdir}')
        markerline.set(markerfacecolor='none', markeredgewidth=2)
        baseline.set_linewidth(3)

@image_comparison(['equal_box_aspect.png'], style='mpl20')
def test_equal_box_aspect():
    if False:
        i = 10
        return i + 15
    from itertools import product, combinations
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z)
    r = [-1, 1]
    for (s, e) in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            ax.plot3D(*zip(s, e), color='b')
    xyzlim = np.column_stack([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    XYZlim = [min(xyzlim[0]), max(xyzlim[1])]
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim)
    ax.axis('off')
    ax.set_box_aspect((1, 1, 1))
    with pytest.raises(ValueError, match='Argument zoom ='):
        ax.set_box_aspect((1, 1, 1), zoom=-1)

def test_colorbar_pos():
    if False:
        while True:
            i = 10
    num_plots = 2
    (fig, axs) = plt.subplots(1, num_plots, figsize=(4, 5), constrained_layout=True, subplot_kw={'projection': '3d'})
    for ax in axs:
        p_tri = ax.plot_trisurf(np.random.randn(5), np.random.randn(5), np.random.randn(5))
    cbar = plt.colorbar(p_tri, ax=axs, orientation='horizontal')
    fig.canvas.draw()
    assert cbar.ax.get_position().extents[1] < 0.2

def test_inverted_zaxis():
    if False:
        return 10
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_zlim(0, 1)
    assert not ax.zaxis_inverted()
    assert ax.get_zlim() == (0, 1)
    assert ax.get_zbound() == (0, 1)
    ax.set_zbound((0, 2))
    assert not ax.zaxis_inverted()
    assert ax.get_zlim() == (0, 2)
    assert ax.get_zbound() == (0, 2)
    ax.invert_zaxis()
    assert ax.zaxis_inverted()
    assert ax.get_zlim() == (2, 0)
    assert ax.get_zbound() == (0, 2)
    ax.set_zbound(upper=1)
    assert ax.zaxis_inverted()
    assert ax.get_zlim() == (1, 0)
    assert ax.get_zbound() == (0, 1)
    ax.set_zbound(lower=2)
    assert ax.zaxis_inverted()
    assert ax.get_zlim() == (2, 1)
    assert ax.get_zbound() == (1, 2)

def test_set_zlim():
    if False:
        for i in range(10):
            print('nop')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    assert np.allclose(ax.get_zlim(), (-1 / 48, 49 / 48))
    ax.set_zlim(zmax=2)
    assert np.allclose(ax.get_zlim(), (-1 / 48, 2))
    ax.set_zlim(zmin=1)
    assert ax.get_zlim() == (1, 2)
    with pytest.raises(TypeError, match="Cannot pass both 'lower' and 'min'"):
        ax.set_zlim(bottom=0, zmin=1)
    with pytest.raises(TypeError, match="Cannot pass both 'upper' and 'max'"):
        ax.set_zlim(top=0, zmax=1)

@check_figures_equal(extensions=['png'])
def test_shared_view(fig_test, fig_ref):
    if False:
        while True:
            i = 10
    (elev, azim, roll) = (5, 20, 30)
    ax1 = fig_test.add_subplot(131, projection='3d')
    ax2 = fig_test.add_subplot(132, projection='3d', shareview=ax1)
    ax3 = fig_test.add_subplot(133, projection='3d')
    ax3.shareview(ax1)
    ax2.view_init(elev=elev, azim=azim, roll=roll, share=True)
    for subplot_num in (131, 132, 133):
        ax = fig_ref.add_subplot(subplot_num, projection='3d')
        ax.view_init(elev=elev, azim=azim, roll=roll)

def test_shared_axes_retick():
    if False:
        return 10
    fig = plt.figure()
    ax1 = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212, projection='3d', sharez=ax1)
    ax1.plot([0, 1], [0, 1], [0, 2])
    ax2.plot([0, 1], [0, 1], [0, 2])
    ax1.set_zticks([-0.5, 0, 2, 2.5])
    assert ax1.get_zlim() == (-0.5, 2.5)
    assert ax2.get_zlim() == (-0.5, 2.5)

def test_pan():
    if False:
        while True:
            i = 10
    'Test mouse panning using the middle mouse button.'

    def convert_lim(dmin, dmax):
        if False:
            print('Hello World!')
        'Convert min/max limits to center and range.'
        center = (dmin + dmax) / 2
        range_ = dmax - dmin
        return (center, range_)
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(0, 0, 0)
    ax.figure.canvas.draw()
    (x_center0, x_range0) = convert_lim(*ax.get_xlim3d())
    (y_center0, y_range0) = convert_lim(*ax.get_ylim3d())
    (z_center0, z_range0) = convert_lim(*ax.get_zlim3d())
    ax._button_press(mock_event(ax, button=MouseButton.MIDDLE, xdata=0, ydata=0))
    ax._on_move(mock_event(ax, button=MouseButton.MIDDLE, xdata=1, ydata=1))
    (x_center, x_range) = convert_lim(*ax.get_xlim3d())
    (y_center, y_range) = convert_lim(*ax.get_ylim3d())
    (z_center, z_range) = convert_lim(*ax.get_zlim3d())
    assert x_range == pytest.approx(x_range0)
    assert y_range == pytest.approx(y_range0)
    assert z_range == pytest.approx(z_range0)
    assert x_center != pytest.approx(x_center0)
    assert y_center != pytest.approx(y_center0)
    assert z_center != pytest.approx(z_center0)

@pytest.mark.parametrize('tool,button,key,expected', [('zoom', MouseButton.LEFT, None, ((0.0, 0.06), (0.01, 0.07), (0.02, 0.08))), ('zoom', MouseButton.LEFT, 'x', ((-0.01, 0.1), (-0.03, 0.08), (-0.06, 0.06))), ('zoom', MouseButton.LEFT, 'y', ((-0.07, 0.05), (-0.04, 0.08), (0.0, 0.12))), ('zoom', MouseButton.RIGHT, None, ((-0.09, 0.15), (-0.08, 0.17), (-0.07, 0.18))), ('pan', MouseButton.LEFT, None, ((-0.7, -0.58), (-1.04, -0.91), (-1.27, -1.15))), ('pan', MouseButton.LEFT, 'x', ((-0.97, -0.84), (-0.58, -0.46), (-0.06, 0.06))), ('pan', MouseButton.LEFT, 'y', ((0.2, 0.32), (-0.51, -0.39), (-1.27, -1.15)))])
def test_toolbar_zoom_pan(tool, button, key, expected):
    if False:
        while True:
            i = 10
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(0, 0, 0)
    fig.canvas.draw()
    (xlim0, ylim0, zlim0) = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())
    d0 = (0, 0)
    d1 = (1, 1)
    s0 = ax.transData.transform(d0).astype(int)
    s1 = ax.transData.transform(d1).astype(int)
    start_event = MouseEvent('button_press_event', fig.canvas, *s0, button, key=key)
    stop_event = MouseEvent('button_release_event', fig.canvas, *s1, button, key=key)
    tb = NavigationToolbar2(fig.canvas)
    if tool == 'zoom':
        tb.zoom()
        tb.press_zoom(start_event)
        tb.drag_zoom(stop_event)
        tb.release_zoom(stop_event)
    else:
        tb.pan()
        tb.press_pan(start_event)
        tb.drag_pan(stop_event)
        tb.release_pan(stop_event)
    (xlim, ylim, zlim) = expected
    assert ax.get_xlim3d() == pytest.approx(xlim, abs=0.01)
    assert ax.get_ylim3d() == pytest.approx(ylim, abs=0.01)
    assert ax.get_zlim3d() == pytest.approx(zlim, abs=0.01)
    tb.back()
    assert ax.get_xlim3d() == pytest.approx(xlim0)
    assert ax.get_ylim3d() == pytest.approx(ylim0)
    assert ax.get_zlim3d() == pytest.approx(zlim0)
    tb.forward()
    assert ax.get_xlim3d() == pytest.approx(xlim, abs=0.01)
    assert ax.get_ylim3d() == pytest.approx(ylim, abs=0.01)
    assert ax.get_zlim3d() == pytest.approx(zlim, abs=0.01)
    tb.home()
    assert ax.get_xlim3d() == pytest.approx(xlim0)
    assert ax.get_ylim3d() == pytest.approx(ylim0)
    assert ax.get_zlim3d() == pytest.approx(zlim0)

@mpl.style.context('default')
@check_figures_equal(extensions=['png'])
def test_scalarmap_update(fig_test, fig_ref):
    if False:
        print('Hello World!')
    (x, y, z) = np.array(list(itertools.product(*[np.arange(0, 5, 1), np.arange(0, 5, 1), np.arange(0, 5, 1)]))).T
    c = x + y
    ax_test = fig_test.add_subplot(111, projection='3d')
    sc_test = ax_test.scatter(x, y, z, c=c, s=40, cmap='viridis')
    fig_test.canvas.draw()
    sc_test.changed()
    ax_ref = fig_ref.add_subplot(111, projection='3d')
    sc_ref = ax_ref.scatter(x, y, z, c=c, s=40, cmap='viridis')

def test_subfigure_simple():
    if False:
        return 10
    fig = plt.figure()
    sf = fig.subfigures(1, 2)
    ax = sf[0].add_subplot(1, 1, 1, projection='3d')
    ax = sf[1].add_subplot(1, 1, 1, projection='3d', label='other')

@image_comparison(baseline_images=['computed_zorder'], remove_text=True, extensions=['png'], style='mpl20')
def test_computed_zorder():
    if False:
        i = 10
        return i + 15
    plt.rcParams['axes3d.automargin'] = True
    fig = plt.figure()
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.computed_zorder = False
    corners = ((0, 0, 0), (0, 5, 0), (5, 5, 0), (5, 0, 0))
    for ax in (ax1, ax2):
        tri = art3d.Poly3DCollection([corners], facecolors='white', edgecolors='black', zorder=1)
        ax.add_collection3d(tri)
        ax.plot((2, 2), (2, 2), (0, 4), c='red', zorder=2)
        ax.scatter((3, 3), (1, 3), (1, 3), c='red', zorder=10)
        ax.set_xlim((0, 5.0))
        ax.set_ylim((0, 5.0))
        ax.set_zlim((0, 2.5))
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.computed_zorder = False
    dim = 10
    (X, Y) = np.meshgrid((-dim, dim), (-dim, dim))
    Z = np.zeros((2, 2))
    angle = 0.5
    (X2, Y2) = np.meshgrid((-dim, dim), (0, dim))
    Z2 = Y2 * angle
    (X3, Y3) = np.meshgrid((-dim, dim), (-dim, 0))
    Z3 = Y3 * angle
    r = 7
    M = 1000
    th = np.linspace(0, 2 * np.pi, M)
    (x, y, z) = (r * np.cos(th), r * np.sin(th), angle * r * np.sin(th))
    for ax in (ax3, ax4):
        ax.plot_surface(X2, Y3, Z3, color='blue', alpha=0.5, linewidth=0, zorder=-1)
        ax.plot(x[y < 0], y[y < 0], z[y < 0], lw=5, linestyle='--', color='green', zorder=0)
        ax.plot_surface(X, Y, Z, color='red', alpha=0.5, linewidth=0, zorder=1)
        ax.plot(r * np.sin(th), r * np.cos(th), np.zeros(M), lw=5, linestyle='--', color='black', zorder=2)
        ax.plot_surface(X2, Y2, Z2, color='blue', alpha=0.5, linewidth=0, zorder=3)
        ax.plot(x[y > 0], y[y > 0], z[y > 0], lw=5, linestyle='--', color='green', zorder=4)
        ax.view_init(elev=20, azim=-20, roll=0)
        ax.axis('off')

def test_format_coord():
    if False:
        for i in range(10):
            print('nop')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = np.arange(10)
    ax.plot(x, np.sin(x))
    xv = 0.1
    yv = 0.1
    fig.canvas.draw()
    assert ax.format_coord(xv, yv) == 'x=10.5227, y pane=1.0417, z=0.1444'
    ax.view_init(roll=30, vertical_axis='y')
    fig.canvas.draw()
    assert ax.format_coord(xv, yv) == 'x pane=9.1875, y=0.9761, z=0.1291'
    ax.view_init()
    fig.canvas.draw()
    assert ax.format_coord(xv, yv) == 'x=10.5227, y pane=1.0417, z=0.1444'
    ax.set_proj_type('ortho')
    fig.canvas.draw()
    assert ax.format_coord(xv, yv) == 'x=10.8869, y pane=1.0417, z=0.1528'
    ax.set_proj_type('persp', focal_length=0.1)
    fig.canvas.draw()
    assert ax.format_coord(xv, yv) == 'x=9.0620, y pane=1.0417, z=0.1110'

def test_get_axis_position():
    if False:
        i = 10
        return i + 15
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = np.arange(10)
    ax.plot(x, np.sin(x))
    fig.canvas.draw()
    assert ax.get_axis_position() == (False, True, False)

def test_margins():
    if False:
        print('Hello World!')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.margins(0.2)
    assert ax.margins() == (0.2, 0.2, 0.2)
    ax.margins(0.1, 0.2, 0.3)
    assert ax.margins() == (0.1, 0.2, 0.3)
    ax.margins(x=0)
    assert ax.margins() == (0, 0.2, 0.3)
    ax.margins(y=0.1)
    assert ax.margins() == (0, 0.1, 0.3)
    ax.margins(z=0)
    assert ax.margins() == (0, 0.1, 0)

def test_margin_getters():
    if False:
        i = 10
        return i + 15
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.margins(0.1, 0.2, 0.3)
    assert ax.get_xmargin() == 0.1
    assert ax.get_ymargin() == 0.2
    assert ax.get_zmargin() == 0.3

@pytest.mark.parametrize('err, args, kwargs, match', ((ValueError, (-1,), {}, 'margin must be greater than -0\\.5'), (ValueError, (1, -1, 1), {}, 'margin must be greater than -0\\.5'), (ValueError, (1, 1, -1), {}, 'margin must be greater than -0\\.5'), (ValueError, tuple(), {'x': -1}, 'margin must be greater than -0\\.5'), (ValueError, tuple(), {'y': -1}, 'margin must be greater than -0\\.5'), (ValueError, tuple(), {'z': -1}, 'margin must be greater than -0\\.5'), (TypeError, (1,), {'x': 1}, 'Cannot pass both positional and keyword'), (TypeError, (1,), {'x': 1, 'y': 1, 'z': 1}, 'Cannot pass both positional and keyword'), (TypeError, (1,), {'x': 1, 'y': 1}, 'Cannot pass both positional and keyword'), (TypeError, (1, 1), {}, 'Must pass a single positional argument for')))
def test_margins_errors(err, args, kwargs, match):
    if False:
        i = 10
        return i + 15
    with pytest.raises(err, match=match):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.margins(*args, **kwargs)

@check_figures_equal(extensions=['png'])
def test_text_3d(fig_test, fig_ref):
    if False:
        i = 10
        return i + 15
    ax = fig_ref.add_subplot(projection='3d')
    txt = Text(0.5, 0.5, 'Foo bar $\\int$')
    art3d.text_2d_to_3d(txt, z=1)
    ax.add_artist(txt)
    assert txt.get_position_3d() == (0.5, 0.5, 1)
    ax = fig_test.add_subplot(projection='3d')
    t3d = art3d.Text3D(0.5, 0.5, 1, 'Foo bar $\\int$')
    ax.add_artist(t3d)
    assert t3d.get_position_3d() == (0.5, 0.5, 1)

def test_draw_single_lines_from_Nx1():
    if False:
        return 10
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot([[0], [1]], [[0], [1]], [[0], [1]])

@check_figures_equal(extensions=['png'])
def test_pathpatch_3d(fig_test, fig_ref):
    if False:
        for i in range(10):
            print('nop')
    ax = fig_ref.add_subplot(projection='3d')
    path = Path.unit_rectangle()
    patch = PathPatch(path)
    art3d.pathpatch_2d_to_3d(patch, z=(0, 0.5, 0.7, 1, 0), zdir='y')
    ax.add_artist(patch)
    ax = fig_test.add_subplot(projection='3d')
    pp3d = art3d.PathPatch3D(path, zs=(0, 0.5, 0.7, 1, 0), zdir='y')
    ax.add_artist(pp3d)

@image_comparison(baseline_images=['scatter_spiral.png'], remove_text=True, style='mpl20')
def test_scatter_spiral():
    if False:
        return 10
    plt.rcParams['axes3d.automargin'] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    th = np.linspace(0, 2 * np.pi * 6, 256)
    sc = ax.scatter(np.sin(th), np.cos(th), th, s=1 + th * 5, c=th ** 2)
    fig.canvas.draw()

def test_Poly3DCollection_get_facecolor():
    if False:
        for i in range(10):
            print('nop')
    (y, x) = np.ogrid[1:10:100j, 1:10:100j]
    z2 = np.cos(x) ** 3 - np.sin(y) ** 2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    r = ax.plot_surface(x, y, z2, cmap='hot')
    r.get_facecolor()

def test_Poly3DCollection_get_edgecolor():
    if False:
        while True:
            i = 10
    (y, x) = np.ogrid[1:10:100j, 1:10:100j]
    z2 = np.cos(x) ** 3 - np.sin(y) ** 2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    r = ax.plot_surface(x, y, z2, cmap='hot')
    r.get_edgecolor()

@pytest.mark.parametrize('vertical_axis, proj_expected, axis_lines_expected, tickdirs_expected', [('z', [[0.0, 1.142857, 0.0, -0.571429], [0.0, 0.0, 0.857143, -0.428571], [0.0, 0.0, 0.0, -10.0], [-1.142857, 0.0, 0.0, 10.571429]], [([0.05617978, 0.06329114], [-0.04213483, -0.04746835]), ([-0.06329114, 0.06329114], [-0.04746835, -0.04746835]), ([-0.06329114, -0.06329114], [-0.04746835, 0.04746835])], [1, 0, 0]), ('y', [[1.142857, 0.0, 0.0, -0.571429], [0.0, 0.857143, 0.0, -0.428571], [0.0, 0.0, 0.0, -10.0], [0.0, 0.0, -1.142857, 10.571429]], [([-0.06329114, 0.06329114], [0.04746835, 0.04746835]), ([0.06329114, 0.06329114], [-0.04746835, 0.04746835]), ([-0.05617978, -0.06329114], [0.04213483, 0.04746835])], [2, 2, 0]), ('x', [[0.0, 0.0, 1.142857, -0.571429], [0.857143, 0.0, 0.0, -0.428571], [0.0, 0.0, 0.0, -10.0], [0.0, -1.142857, 0.0, 10.571429]], [([-0.06329114, -0.06329114], [0.04746835, -0.04746835]), ([0.06329114, 0.05617978], [0.04746835, 0.04213483]), ([0.06329114, -0.06329114], [0.04746835, 0.04746835])], [1, 2, 1])])
def test_view_init_vertical_axis(vertical_axis, proj_expected, axis_lines_expected, tickdirs_expected):
    if False:
        print('Hello World!')
    '\n    Test the actual projection, axis lines and ticks matches expected values.\n\n    Parameters\n    ----------\n    vertical_axis : str\n        Axis to align vertically.\n    proj_expected : ndarray\n        Expected values from ax.get_proj().\n    axis_lines_expected : tuple of arrays\n        Edgepoints of the axis line. Expected values retrieved according\n        to ``ax.get_[xyz]axis().line.get_data()``.\n    tickdirs_expected : list of int\n        indexes indicating which axis to create a tick line along.\n    '
    rtol = 2e-06
    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=0, azim=0, roll=0, vertical_axis=vertical_axis)
    ax.figure.canvas.draw()
    proj_actual = ax.get_proj()
    np.testing.assert_allclose(proj_expected, proj_actual, rtol=rtol)
    for (i, axis) in enumerate([ax.get_xaxis(), ax.get_yaxis(), ax.get_zaxis()]):
        axis_line_expected = axis_lines_expected[i]
        axis_line_actual = axis.line.get_data()
        np.testing.assert_allclose(axis_line_expected, axis_line_actual, rtol=rtol)
        tickdir_expected = tickdirs_expected[i]
        tickdir_actual = axis._get_tickdir('default')
        np.testing.assert_array_equal(tickdir_expected, tickdir_actual)

@image_comparison(baseline_images=['arc_pathpatch.png'], remove_text=True, style='mpl20')
def test_arc_pathpatch():
    if False:
        i = 10
        return i + 15
    ax = plt.subplot(1, 1, 1, projection='3d')
    a = mpatch.Arc((0.5, 0.5), width=0.5, height=0.9, angle=20, theta1=10, theta2=130)
    ax.add_patch(a)
    art3d.pathpatch_2d_to_3d(a, z=0, zdir='z')

@image_comparison(baseline_images=['panecolor_rcparams.png'], remove_text=True, style='mpl20')
def test_panecolor_rcparams():
    if False:
        for i in range(10):
            print('nop')
    with plt.rc_context({'axes3d.xaxis.panecolor': 'r', 'axes3d.yaxis.panecolor': 'g', 'axes3d.zaxis.panecolor': 'b'}):
        fig = plt.figure(figsize=(1, 1))
        fig.add_subplot(projection='3d')

@check_figures_equal(extensions=['png'])
def test_mutating_input_arrays_y_and_z(fig_test, fig_ref):
    if False:
        return 10
    '\n    Test to see if the `z` axis does not get mutated\n    after a call to `Axes3D.plot`\n\n    test cases came from GH#8990\n    '
    ax1 = fig_test.add_subplot(111, projection='3d')
    x = [1, 2, 3]
    y = [0.0, 0.0, 0.0]
    z = [0.0, 0.0, 0.0]
    ax1.plot(x, y, z, 'o-')
    y[:] = [1, 2, 3]
    z[:] = [1, 2, 3]
    ax2 = fig_ref.add_subplot(111, projection='3d')
    x = [1, 2, 3]
    y = [0.0, 0.0, 0.0]
    z = [0.0, 0.0, 0.0]
    ax2.plot(x, y, z, 'o-')

def test_scatter_masked_color():
    if False:
        print('Hello World!')
    '\n    Test color parameter usage with non-finite coordinate arrays.\n\n    GH#26236\n    '
    x = [np.nan, 1, 2, 1]
    y = [0, np.inf, 2, 1]
    z = [0, 1, -np.inf, 1]
    colors = [[0.0, 0.0, 0.0, 1], [0.0, 0.0, 0.0, 1], [0.0, 0.0, 0.0, 1], [0.0, 0.0, 0.0, 1]]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    path3d = ax.scatter(x, y, z, color=colors)
    assert len(path3d.get_offsets()) == len(super(type(path3d), path3d).get_facecolors())

@mpl3d_image_comparison(['surface3d_zsort_inf.png'], style='mpl20')
def test_surface3d_zsort_inf():
    if False:
        i = 10
        return i + 15
    plt.rcParams['axes3d.automargin'] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    (x, y) = np.mgrid[-2:2:0.1, -2:2:0.1]
    z = np.sin(x) ** 2 + np.cos(y) ** 2
    z[x.shape[0] // 2:, x.shape[1] // 2:] = np.inf
    ax.plot_surface(x, y, z, cmap='jet')
    ax.view_init(elev=45, azim=145)

def test_Poly3DCollection_init_value_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError, match='You must provide facecolors, edgecolors, or both for shade to work.'):
        poly = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]], float)
        c = art3d.Poly3DCollection([poly], shade=True)

def test_ndarray_color_kwargs_value_error():
    if False:
        while True:
            i = 10
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(1, 0, 0, color=np.array([0, 0, 0, 1]))
    fig.canvas.draw()