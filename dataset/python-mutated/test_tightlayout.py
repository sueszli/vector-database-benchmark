import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle

def example_plot(ax, fontsize=12):
    if False:
        return 10
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

@image_comparison(['tight_layout1'], tol=1.9)
def test_tight_layout1():
    if False:
        i = 10
        return i + 15
    'Test tight_layout for a single subplot.'
    (fig, ax) = plt.subplots()
    example_plot(ax, fontsize=24)
    plt.tight_layout()

@image_comparison(['tight_layout2'])
def test_tight_layout2():
    if False:
        i = 10
        return i + 15
    'Test tight_layout for multiple subplots.'
    (fig, ((ax1, ax2), (ax3, ax4))) = plt.subplots(nrows=2, ncols=2)
    example_plot(ax1)
    example_plot(ax2)
    example_plot(ax3)
    example_plot(ax4)
    plt.tight_layout()

@image_comparison(['tight_layout3'])
def test_tight_layout3():
    if False:
        while True:
            i = 10
    'Test tight_layout for multiple subplots.'
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(122)
    example_plot(ax1)
    example_plot(ax2)
    example_plot(ax3)
    plt.tight_layout()

@image_comparison(['tight_layout4'], freetype_version=('2.5.5', '2.6.1'), tol=0.015)
def test_tight_layout4():
    if False:
        while True:
            i = 10
    'Test tight_layout for subplot2grid.'
    ax1 = plt.subplot2grid((3, 3), (0, 0))
    ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2)
    ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
    ax4 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
    example_plot(ax1)
    example_plot(ax2)
    example_plot(ax3)
    example_plot(ax4)
    plt.tight_layout()

@image_comparison(['tight_layout5'])
def test_tight_layout5():
    if False:
        for i in range(10):
            print('nop')
    'Test tight_layout for image.'
    ax = plt.subplot()
    arr = np.arange(100).reshape((10, 10))
    ax.imshow(arr, interpolation='none')
    plt.tight_layout()

@image_comparison(['tight_layout6'])
def test_tight_layout6():
    if False:
        i = 10
        return i + 15
    'Test tight_layout for gridspec.'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        fig = plt.figure()
        gs1 = mpl.gridspec.GridSpec(2, 1)
        ax1 = fig.add_subplot(gs1[0])
        ax2 = fig.add_subplot(gs1[1])
        example_plot(ax1)
        example_plot(ax2)
        gs1.tight_layout(fig, rect=[0, 0, 0.5, 1])
        gs2 = mpl.gridspec.GridSpec(3, 1)
        for ss in gs2:
            ax = fig.add_subplot(ss)
            example_plot(ax)
            ax.set_title('')
            ax.set_xlabel('')
        ax.set_xlabel('x-label', fontsize=12)
        gs2.tight_layout(fig, rect=[0.5, 0, 1, 1], h_pad=0.45)
        top = min(gs1.top, gs2.top)
        bottom = max(gs1.bottom, gs2.bottom)
        gs1.tight_layout(fig, rect=[None, 0 + (bottom - gs1.bottom), 0.5, 1 - (gs1.top - top)])
        gs2.tight_layout(fig, rect=[0.5, 0 + (bottom - gs2.bottom), None, 1 - (gs2.top - top)], h_pad=0.45)

@image_comparison(['tight_layout7'], tol=1.9)
def test_tight_layout7():
    if False:
        print('Hello World!')
    fontsize = 24
    (fig, ax) = plt.subplots()
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Left Title', loc='left', fontsize=fontsize)
    ax.set_title('Right Title', loc='right', fontsize=fontsize)
    plt.tight_layout()

@image_comparison(['tight_layout8'])
def test_tight_layout8():
    if False:
        for i in range(10):
            print('nop')
    'Test automatic use of tight_layout.'
    fig = plt.figure()
    fig.set_layout_engine(layout='tight', pad=0.1)
    ax = fig.add_subplot()
    example_plot(ax, fontsize=24)
    fig.draw_without_rendering()

@image_comparison(['tight_layout9'])
def test_tight_layout9():
    if False:
        return 10
    (f, axarr) = plt.subplots(2, 2)
    axarr[1][1].set_visible(False)
    plt.tight_layout()

def test_outward_ticks():
    if False:
        for i in range(10):
            print('nop')
    'Test automatic use of tight_layout.'
    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.xaxis.set_tick_params(tickdir='out', length=16, width=3)
    ax.yaxis.set_tick_params(tickdir='out', length=16, width=3)
    ax.xaxis.set_tick_params(tickdir='out', length=32, width=3, tick1On=True, which='minor')
    ax.yaxis.set_tick_params(tickdir='out', length=32, width=3, tick1On=True, which='minor')
    ax.xaxis.set_ticks([0], minor=True)
    ax.yaxis.set_ticks([0], minor=True)
    ax = fig.add_subplot(222)
    ax.xaxis.set_tick_params(tickdir='in', length=32, width=3)
    ax.yaxis.set_tick_params(tickdir='in', length=32, width=3)
    ax = fig.add_subplot(223)
    ax.xaxis.set_tick_params(tickdir='inout', length=32, width=3)
    ax.yaxis.set_tick_params(tickdir='inout', length=32, width=3)
    ax = fig.add_subplot(224)
    ax.xaxis.set_tick_params(tickdir='out', length=32, width=3)
    ax.yaxis.set_tick_params(tickdir='out', length=32, width=3)
    plt.tight_layout()
    ans = [[[0.091, 0.607], [0.433, 0.933]], [[0.579, 0.607], [0.922, 0.933]], [[0.091, 0.14], [0.433, 0.466]], [[0.579, 0.14], [0.922, 0.466]]]
    for (nn, ax) in enumerate(fig.axes):
        assert_array_equal(np.round(ax.get_position().get_points(), 3), ans[nn])

def add_offsetboxes(ax, size=10, margin=0.1, color='black'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Surround ax with OffsetBoxes\n    '
    (m, mp) = (margin, 1 + margin)
    anchor_points = [(-m, -m), (-m, 0.5), (-m, mp), (mp, 0.5), (0.5, mp), (mp, mp), (0.5, -m), (mp, -m), (0.5, -m)]
    for point in anchor_points:
        da = DrawingArea(size, size)
        background = Rectangle((0, 0), width=size, height=size, facecolor=color, edgecolor='None', linewidth=0, antialiased=False)
        da.add_artist(background)
        anchored_box = AnchoredOffsetbox(loc='center', child=da, pad=0.0, frameon=False, bbox_to_anchor=point, bbox_transform=ax.transAxes, borderpad=0.0)
        ax.add_artist(anchored_box)
    return anchored_box

@image_comparison(['tight_layout_offsetboxes1', 'tight_layout_offsetboxes2'])
def test_tight_layout_offsetboxes():
    if False:
        i = 10
        return i + 15
    rows = cols = 2
    colors = ['red', 'blue', 'green', 'yellow']
    x = y = [0, 1]

    def _subplots():
        if False:
            for i in range(10):
                print('nop')
        (_, axs) = plt.subplots(rows, cols)
        axs = axs.flat
        for (ax, color) in zip(axs, colors):
            ax.plot(x, y, color=color)
            add_offsetboxes(ax, 20, color=color)
        return axs
    axs = _subplots()
    plt.tight_layout()
    axs = _subplots()
    for ax in axs[cols - 1::rows]:
        for child in ax.get_children():
            if isinstance(child, AnchoredOffsetbox):
                child.set_visible(False)
    plt.tight_layout()

def test_empty_layout():
    if False:
        while True:
            i = 10
    "Test that tight layout doesn't cause an error when there are no axes."
    fig = plt.gcf()
    fig.tight_layout()

@pytest.mark.parametrize('label', ['xlabel', 'ylabel'])
def test_verybig_decorators(label):
    if False:
        return 10
    'Test that no warning emitted when xlabel/ylabel too big.'
    (fig, ax) = plt.subplots(figsize=(3, 2))
    ax.set(**{label: 'a' * 100})

def test_big_decorators_horizontal():
    if False:
        i = 10
        return i + 15
    "Test that doesn't warn when xlabel too big."
    (fig, axs) = plt.subplots(1, 2, figsize=(3, 2))
    axs[0].set_xlabel('a' * 30)
    axs[1].set_xlabel('b' * 30)

def test_big_decorators_vertical():
    if False:
        i = 10
        return i + 15
    "Test that doesn't warn when ylabel too big."
    (fig, axs) = plt.subplots(2, 1, figsize=(3, 2))
    axs[0].set_ylabel('a' * 20)
    axs[1].set_ylabel('b' * 20)

def test_badsubplotgrid():
    if False:
        while True:
            i = 10
    plt.subplot2grid((4, 5), (0, 0))
    plt.subplot2grid((5, 5), (0, 3), colspan=3, rowspan=5)
    with pytest.warns(UserWarning):
        plt.tight_layout()

def test_collapsed():
    if False:
        while True:
            i = 10
    (fig, ax) = plt.subplots(tight_layout=True)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.annotate('BIG LONG STRING', xy=(1.25, 2), xytext=(10.5, 1.75), annotation_clip=False)
    p1 = ax.get_position()
    with pytest.warns(UserWarning):
        plt.tight_layout()
        p2 = ax.get_position()
        assert p1.width == p2.width
    with pytest.warns(UserWarning):
        plt.tight_layout(rect=[0, 0, 0.8, 0.8])

def test_suptitle():
    if False:
        return 10
    (fig, ax) = plt.subplots(tight_layout=True)
    st = fig.suptitle('foo')
    t = ax.set_title('bar')
    fig.canvas.draw()
    assert st.get_window_extent().y0 > t.get_window_extent().y1

@pytest.mark.backend('pdf')
def test_non_agg_renderer(monkeypatch, recwarn):
    if False:
        print('Hello World!')
    unpatched_init = mpl.backend_bases.RendererBase.__init__

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        assert isinstance(self, mpl.backends.backend_pdf.RendererPdf)
        unpatched_init(self, *args, **kwargs)
    monkeypatch.setattr(mpl.backend_bases.RendererBase, '__init__', __init__)
    (fig, ax) = plt.subplots()
    fig.tight_layout()

def test_manual_colorbar():
    if False:
        print('Hello World!')
    (fig, axes) = plt.subplots(1, 2)
    pts = axes[1].scatter([0, 1], [0, 1], c=[1, 5])
    ax_rect = axes[1].get_position()
    cax = fig.add_axes([ax_rect.x1 + 0.005, ax_rect.y0, 0.015, ax_rect.height])
    fig.colorbar(pts, cax=cax)
    with pytest.warns(UserWarning, match='This figure includes Axes'):
        fig.tight_layout()

def test_clipped_to_axes():
    if False:
        i = 10
        return i + 15
    arr = np.arange(100).reshape((10, 10))
    fig = plt.figure(figsize=(6, 2))
    ax1 = fig.add_subplot(131, projection='rectilinear')
    ax2 = fig.add_subplot(132, projection='mollweide')
    ax3 = fig.add_subplot(133, projection='polar')
    for ax in (ax1, ax2, ax3):
        ax.grid(False)
        (h,) = ax.plot(arr[:, 0])
        m = ax.pcolor(arr)
        assert h._fully_clipped_to_axes()
        assert m._fully_clipped_to_axes()
        rect = Rectangle((0, 0), 0.5, 0.5, transform=ax.transAxes)
        h.set_clip_path(rect)
        m.set_clip_path(rect.get_path(), rect.get_transform())
        assert not h._fully_clipped_to_axes()
        assert not m._fully_clipped_to_axes()

def test_tight_pads():
    if False:
        while True:
            i = 10
    (fig, ax) = plt.subplots()
    with pytest.warns(PendingDeprecationWarning, match='will be deprecated'):
        fig.set_tight_layout({'pad': 0.15})
    fig.draw_without_rendering()

def test_tight_kwargs():
    if False:
        i = 10
        return i + 15
    (fig, ax) = plt.subplots(tight_layout={'pad': 0.15})
    fig.draw_without_rendering()

def test_tight_toggle():
    if False:
        while True:
            i = 10
    (fig, ax) = plt.subplots()
    with pytest.warns(PendingDeprecationWarning):
        fig.set_tight_layout(True)
        assert fig.get_tight_layout()
        fig.set_tight_layout(False)
        assert not fig.get_tight_layout()
        fig.set_tight_layout(True)
        assert fig.get_tight_layout()