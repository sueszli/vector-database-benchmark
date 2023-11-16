import re
from matplotlib import path, transforms
from matplotlib.backend_bases import FigureCanvasBase, KeyEvent, LocationEvent, MouseButton, MouseEvent, NavigationToolbar2, RendererBase
from matplotlib.backend_tools import RubberbandBase
from matplotlib.figure import Figure
from matplotlib.testing._markers import needs_pgf_xelatex
import matplotlib.pyplot as plt
import numpy as np
import pytest
_EXPECTED_WARNING_TOOLMANAGER = 'Treat the new Tool classes introduced in v[0-9]*.[0-9]* as experimental for now; the API and rcParam may change in future versions.'

def test_uses_per_path():
    if False:
        print('Hello World!')
    id = transforms.Affine2D()
    paths = [path.Path.unit_regular_polygon(i) for i in range(3, 7)]
    tforms_matrices = [id.rotate(i).get_matrix().copy() for i in range(1, 5)]
    offsets = np.arange(20).reshape((10, 2))
    facecolors = ['red', 'green']
    edgecolors = ['red', 'green']

    def check(master_transform, paths, all_transforms, offsets, facecolors, edgecolors):
        if False:
            for i in range(10):
                print('nop')
        rb = RendererBase()
        raw_paths = list(rb._iter_collection_raw_paths(master_transform, paths, all_transforms))
        gc = rb.new_gc()
        ids = [path_id for (xo, yo, path_id, gc0, rgbFace) in rb._iter_collection(gc, range(len(raw_paths)), offsets, transforms.AffineDeltaTransform(master_transform), facecolors, edgecolors, [], [], [False], [], 'screen')]
        uses = rb._iter_collection_uses_per_path(paths, all_transforms, offsets, facecolors, edgecolors)
        if raw_paths:
            seen = np.bincount(ids, minlength=len(raw_paths))
            assert set(seen).issubset([uses - 1, uses])
    check(id, paths, tforms_matrices, offsets, facecolors, edgecolors)
    check(id, paths[0:1], tforms_matrices, offsets, facecolors, edgecolors)
    check(id, [], tforms_matrices, offsets, facecolors, edgecolors)
    check(id, paths, tforms_matrices[0:1], offsets, facecolors, edgecolors)
    check(id, paths, [], offsets, facecolors, edgecolors)
    for n in range(0, offsets.shape[0]):
        check(id, paths, tforms_matrices, offsets[0:n, :], facecolors, edgecolors)
    check(id, paths, tforms_matrices, offsets, [], edgecolors)
    check(id, paths, tforms_matrices, offsets, facecolors, [])
    check(id, paths, tforms_matrices, offsets, [], [])
    check(id, paths, tforms_matrices, offsets, facecolors[0:1], edgecolors)

def test_canvas_ctor():
    if False:
        return 10
    assert isinstance(FigureCanvasBase().figure, Figure)

def test_get_default_filename():
    if False:
        print('Hello World!')
    assert plt.figure().canvas.get_default_filename() == 'image.png'

def test_canvas_change():
    if False:
        print('Hello World!')
    fig = plt.figure()
    canvas = FigureCanvasBase(fig)
    plt.close(fig)
    assert not plt.fignum_exists(fig.number)

@pytest.mark.backend('pdf')
def test_non_gui_warning(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    plt.subplots()
    monkeypatch.setenv('DISPLAY', ':999')
    with pytest.warns(UserWarning) as rec:
        plt.show()
        assert len(rec) == 1
        assert 'FigureCanvasPdf is non-interactive, and thus cannot be shown' in str(rec[0].message)
    with pytest.warns(UserWarning) as rec:
        plt.gcf().show()
        assert len(rec) == 1
        assert 'FigureCanvasPdf is non-interactive, and thus cannot be shown' in str(rec[0].message)

def test_grab_clear():
    if False:
        i = 10
        return i + 15
    (fig, ax) = plt.subplots()
    fig.canvas.grab_mouse(ax)
    assert fig.canvas.mouse_grabber == ax
    fig.clear()
    assert fig.canvas.mouse_grabber is None

@pytest.mark.parametrize('x, y', [(42, 24), (None, 42), (None, None), (200, 100.01), (205.75, 2.0)])
def test_location_event_position(x, y):
    if False:
        for i in range(10):
            print('nop')
    (fig, ax) = plt.subplots()
    canvas = FigureCanvasBase(fig)
    event = LocationEvent('test_event', canvas, x, y)
    if x is None:
        assert event.x is None
    else:
        assert event.x == int(x)
        assert isinstance(event.x, int)
    if y is None:
        assert event.y is None
    else:
        assert event.y == int(y)
        assert isinstance(event.y, int)
    if x is not None and y is not None:
        assert re.match(f'x={ax.format_xdata(x)} +y={ax.format_ydata(y)}', ax.format_coord(x, y))
        ax.fmt_xdata = ax.fmt_ydata = lambda x: 'foo'
        assert re.match('x=foo +y=foo', ax.format_coord(x, y))

def test_pick():
    if False:
        return 10
    fig = plt.figure()
    fig.text(0.5, 0.5, 'hello', ha='center', va='center', picker=True)
    fig.canvas.draw()
    picks = []

    def handle_pick(event):
        if False:
            return 10
        assert event.mouseevent.key == 'a'
        picks.append(event)
    fig.canvas.mpl_connect('pick_event', handle_pick)
    KeyEvent('key_press_event', fig.canvas, 'a')._process()
    MouseEvent('button_press_event', fig.canvas, *fig.transFigure.transform((0.5, 0.5)), MouseButton.LEFT)._process()
    KeyEvent('key_release_event', fig.canvas, 'a')._process()
    assert len(picks) == 1

def test_interactive_zoom():
    if False:
        for i in range(10):
            print('nop')
    (fig, ax) = plt.subplots()
    ax.set(xscale='logit')
    assert ax.get_navigate_mode() is None
    tb = NavigationToolbar2(fig.canvas)
    tb.zoom()
    assert ax.get_navigate_mode() == 'ZOOM'
    xlim0 = ax.get_xlim()
    ylim0 = ax.get_ylim()
    d0 = (1e-06, 0.1)
    d1 = (1 - 1e-05, 0.8)
    s0 = ax.transData.transform(d0).astype(int)
    s1 = ax.transData.transform(d1).astype(int)
    start_event = MouseEvent('button_press_event', fig.canvas, *s0, MouseButton.LEFT)
    fig.canvas.callbacks.process(start_event.name, start_event)
    stop_event = MouseEvent('button_release_event', fig.canvas, *s1, MouseButton.LEFT)
    fig.canvas.callbacks.process(stop_event.name, stop_event)
    assert ax.get_xlim() == (start_event.xdata, stop_event.xdata)
    assert ax.get_ylim() == (start_event.ydata, stop_event.ydata)
    start_event = MouseEvent('button_press_event', fig.canvas, *s1, MouseButton.RIGHT)
    fig.canvas.callbacks.process(start_event.name, start_event)
    stop_event = MouseEvent('button_release_event', fig.canvas, *s0, MouseButton.RIGHT)
    fig.canvas.callbacks.process(stop_event.name, stop_event)
    assert ax.get_xlim() == pytest.approx(xlim0, rel=0, abs=1e-10)
    assert ax.get_ylim() == pytest.approx(ylim0, rel=0, abs=1e-10)
    tb.zoom()
    assert ax.get_navigate_mode() is None
    assert not ax.get_autoscalex_on() and (not ax.get_autoscaley_on())

def test_widgetlock_zoompan():
    if False:
        i = 10
        return i + 15
    (fig, ax) = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.canvas.widgetlock(ax)
    tb = NavigationToolbar2(fig.canvas)
    tb.zoom()
    assert ax.get_navigate_mode() is None
    tb.pan()
    assert ax.get_navigate_mode() is None

@pytest.mark.parametrize('plot_func', ['imshow', 'contourf'])
@pytest.mark.parametrize('orientation', ['vertical', 'horizontal'])
@pytest.mark.parametrize('tool,button,expected', [('zoom', MouseButton.LEFT, (4, 6)), ('zoom', MouseButton.RIGHT, (-20, 30)), ('pan', MouseButton.LEFT, (-2, 8)), ('pan', MouseButton.RIGHT, (1.47, 7.78))])
def test_interactive_colorbar(plot_func, orientation, tool, button, expected):
    if False:
        i = 10
        return i + 15
    (fig, ax) = plt.subplots()
    data = np.arange(12).reshape((4, 3))
    (vmin0, vmax0) = (0, 10)
    coll = getattr(ax, plot_func)(data, vmin=vmin0, vmax=vmax0)
    cb = fig.colorbar(coll, ax=ax, orientation=orientation)
    if plot_func == 'contourf':
        assert not cb.ax.get_navigate()
        return
    assert cb.ax.get_navigate()
    (vmin, vmax) = (4, 6)
    d0 = (vmin, 0.5)
    d1 = (vmax, 0.5)
    if orientation == 'vertical':
        d0 = d0[::-1]
        d1 = d1[::-1]
    s0 = cb.ax.transData.transform(d0).astype(int)
    s1 = cb.ax.transData.transform(d1).astype(int)
    start_event = MouseEvent('button_press_event', fig.canvas, *s0, button)
    stop_event = MouseEvent('button_release_event', fig.canvas, *s1, button)
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
    assert (cb.vmin, cb.vmax) == pytest.approx(expected, abs=0.15)

def test_toolbar_zoompan():
    if False:
        while True:
            i = 10
    with pytest.warns(UserWarning, match=_EXPECTED_WARNING_TOOLMANAGER):
        plt.rcParams['toolbar'] = 'toolmanager'
    ax = plt.gca()
    assert ax.get_navigate_mode() is None
    ax.figure.canvas.manager.toolmanager.trigger_tool('zoom')
    assert ax.get_navigate_mode() == 'ZOOM'
    ax.figure.canvas.manager.toolmanager.trigger_tool('pan')
    assert ax.get_navigate_mode() == 'PAN'

def test_toolbar_home_restores_autoscale():
    if False:
        while True:
            i = 10
    (fig, ax) = plt.subplots()
    ax.plot(range(11), range(11))
    tb = NavigationToolbar2(fig.canvas)
    tb.zoom()
    KeyEvent('key_press_event', fig.canvas, 'k', 100, 100)._process()
    KeyEvent('key_press_event', fig.canvas, 'l', 100, 100)._process()
    assert ax.get_xlim() == ax.get_ylim() == (1, 10)
    KeyEvent('key_press_event', fig.canvas, 'k', 100, 100)._process()
    KeyEvent('key_press_event', fig.canvas, 'l', 100, 100)._process()
    assert ax.get_xlim() == ax.get_ylim() == (0, 10)
    (start, stop) = ax.transData.transform([(2, 2), (5, 5)])
    MouseEvent('button_press_event', fig.canvas, *start, MouseButton.LEFT)._process()
    MouseEvent('button_release_event', fig.canvas, *stop, MouseButton.LEFT)._process()
    KeyEvent('key_press_event', fig.canvas, 'h')._process()
    assert ax.get_xlim() == ax.get_ylim() == (0, 10)
    KeyEvent('key_press_event', fig.canvas, 'k', 100, 100)._process()
    KeyEvent('key_press_event', fig.canvas, 'l', 100, 100)._process()
    assert ax.get_xlim() == ax.get_ylim() == (1, 10)

@pytest.mark.parametrize('backend', ['svg', 'ps', 'pdf', pytest.param('pgf', marks=needs_pgf_xelatex)])
def test_draw(backend):
    if False:
        print('Hello World!')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvas
    test_backend = pytest.importorskip(f'matplotlib.backends.backend_{backend}')
    TestCanvas = test_backend.FigureCanvas
    fig_test = Figure(constrained_layout=True)
    TestCanvas(fig_test)
    axes_test = fig_test.subplots(2, 2)
    fig_agg = Figure(constrained_layout=True)
    FigureCanvas(fig_agg)
    axes_agg = fig_agg.subplots(2, 2)
    init_pos = [ax.get_position() for ax in axes_test.ravel()]
    fig_test.canvas.draw()
    fig_agg.canvas.draw()
    layed_out_pos_test = [ax.get_position() for ax in axes_test.ravel()]
    layed_out_pos_agg = [ax.get_position() for ax in axes_agg.ravel()]
    for (init, placed) in zip(init_pos, layed_out_pos_test):
        assert not np.allclose(init, placed, atol=0.005)
    for (ref, test) in zip(layed_out_pos_agg, layed_out_pos_test):
        np.testing.assert_allclose(ref, test, atol=0.005)

@pytest.mark.parametrize('key,mouseend,expectedxlim,expectedylim', [(None, (0.2, 0.2), (3.49, 12.49), (2.7, 11.7)), (None, (0.2, 0.5), (3.49, 12.49), (0, 9)), (None, (0.5, 0.2), (0, 9), (2.7, 11.7)), (None, (0.5, 0.5), (0, 9), (0, 9)), (None, (0.8, 0.25), (-3.47, 5.53), (2.25, 11.25)), (None, (0.2, 0.25), (3.49, 12.49), (2.25, 11.25)), (None, (0.8, 0.85), (-3.47, 5.53), (-3.14, 5.86)), (None, (0.2, 0.85), (3.49, 12.49), (-3.14, 5.86)), ('shift', (0.2, 0.4), (3.49, 12.49), (0, 9)), ('shift', (0.4, 0.2), (0, 9), (2.7, 11.7)), ('shift', (0.2, 0.25), (3.49, 12.49), (3.49, 12.49)), ('shift', (0.8, 0.25), (-3.47, 5.53), (3.47, 12.47)), ('shift', (0.8, 0.9), (-3.58, 5.41), (-3.58, 5.41)), ('shift', (0.2, 0.85), (3.49, 12.49), (-3.49, 5.51)), ('x', (0.2, 0.1), (3.49, 12.49), (0, 9)), ('y', (0.1, 0.2), (0, 9), (2.7, 11.7)), ('control', (0.2, 0.2), (3.49, 12.49), (3.49, 12.49)), ('control', (0.4, 0.2), (2.72, 11.72), (2.72, 11.72))])
def test_interactive_pan(key, mouseend, expectedxlim, expectedylim):
    if False:
        print('Hello World!')
    (fig, ax) = plt.subplots()
    ax.plot(np.arange(10))
    assert ax.get_navigate()
    ax.set_aspect('equal')
    mousestart = (0.5, 0.5)
    sstart = ax.transData.transform(mousestart).astype(int)
    send = ax.transData.transform(mouseend).astype(int)
    start_event = MouseEvent('button_press_event', fig.canvas, *sstart, button=MouseButton.LEFT, key=key)
    stop_event = MouseEvent('button_release_event', fig.canvas, *send, button=MouseButton.LEFT, key=key)
    tb = NavigationToolbar2(fig.canvas)
    tb.pan()
    tb.press_pan(start_event)
    tb.drag_pan(stop_event)
    tb.release_pan(stop_event)
    assert tuple(ax.get_xlim()) == pytest.approx(expectedxlim, abs=0.02)
    assert tuple(ax.get_ylim()) == pytest.approx(expectedylim, abs=0.02)

def test_toolmanager_remove():
    if False:
        return 10
    with pytest.warns(UserWarning, match=_EXPECTED_WARNING_TOOLMANAGER):
        plt.rcParams['toolbar'] = 'toolmanager'
    fig = plt.gcf()
    initial_len = len(fig.canvas.manager.toolmanager.tools)
    assert 'forward' in fig.canvas.manager.toolmanager.tools
    fig.canvas.manager.toolmanager.remove_tool('forward')
    assert len(fig.canvas.manager.toolmanager.tools) == initial_len - 1
    assert 'forward' not in fig.canvas.manager.toolmanager.tools

def test_toolmanager_get_tool():
    if False:
        return 10
    with pytest.warns(UserWarning, match=_EXPECTED_WARNING_TOOLMANAGER):
        plt.rcParams['toolbar'] = 'toolmanager'
    fig = plt.gcf()
    rubberband = fig.canvas.manager.toolmanager.get_tool('rubberband')
    assert isinstance(rubberband, RubberbandBase)
    assert fig.canvas.manager.toolmanager.get_tool(rubberband) is rubberband
    with pytest.warns(UserWarning, match="ToolManager does not control tool 'foo'"):
        assert fig.canvas.manager.toolmanager.get_tool('foo') is None
    assert fig.canvas.manager.toolmanager.get_tool('foo', warn=False) is None
    with pytest.warns(UserWarning, match="ToolManager does not control tool 'foo'"):
        assert fig.canvas.manager.toolmanager.trigger_tool('foo') is None

def test_toolmanager_update_keymap():
    if False:
        return 10
    with pytest.warns(UserWarning, match=_EXPECTED_WARNING_TOOLMANAGER):
        plt.rcParams['toolbar'] = 'toolmanager'
    fig = plt.gcf()
    assert 'v' in fig.canvas.manager.toolmanager.get_tool_keymap('forward')
    with pytest.warns(UserWarning, match='Key c changed from back to forward'):
        fig.canvas.manager.toolmanager.update_keymap('forward', 'c')
    assert fig.canvas.manager.toolmanager.get_tool_keymap('forward') == ['c']
    with pytest.raises(KeyError, match="'foo' not in Tools"):
        fig.canvas.manager.toolmanager.update_keymap('foo', 'c')