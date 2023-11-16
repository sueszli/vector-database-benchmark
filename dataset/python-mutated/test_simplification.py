import base64
import io
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
from matplotlib.testing.decorators import check_figures_equal, image_comparison, remove_ticks_and_titles
import matplotlib.pyplot as plt
from matplotlib import patches, transforms
from matplotlib.path import Path

@image_comparison(['clipping'], remove_text=True)
def test_clipping():
    if False:
        i = 10
        return i + 15
    t = np.arange(0.0, 2.0, 0.01)
    s = np.sin(2 * np.pi * t)
    (fig, ax) = plt.subplots()
    ax.plot(t, s, linewidth=1.0)
    ax.set_ylim((-0.2, -0.28))

@image_comparison(['overflow'], remove_text=True)
def test_overflow():
    if False:
        print('Hello World!')
    x = np.array([1.0, 2.0, 3.0, 200000.0])
    y = np.arange(len(x))
    (fig, ax) = plt.subplots()
    ax.plot(x, y)
    ax.set_xlim(2, 6)

@image_comparison(['clipping_diamond'], remove_text=True)
def test_diamond():
    if False:
        print('Hello World!')
    x = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
    y = np.array([1.0, 0.0, -1.0, 0.0, 1.0])
    (fig, ax) = plt.subplots()
    ax.plot(x, y)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)

def test_clipping_out_of_bounds():
    if False:
        print('Hello World!')
    path = Path([(0, 0), (1, 2), (2, 1)])
    simplified = path.cleaned(clip=(10, 10, 20, 20))
    assert_array_equal(simplified.vertices, [(0, 0)])
    assert simplified.codes == [Path.STOP]
    path = Path([(0, 0), (1, 2), (2, 1)], [Path.MOVETO, Path.LINETO, Path.LINETO])
    simplified = path.cleaned(clip=(10, 10, 20, 20))
    assert_array_equal(simplified.vertices, [(0, 0)])
    assert simplified.codes == [Path.STOP]
    path = Path([(0, 0), (1, 2), (2, 3)], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    simplified = path.cleaned()
    simplified_clipped = path.cleaned(clip=(10, 10, 20, 20))
    assert_array_equal(simplified.vertices, simplified_clipped.vertices)
    assert_array_equal(simplified.codes, simplified_clipped.codes)

def test_noise():
    if False:
        i = 10
        return i + 15
    np.random.seed(0)
    x = np.random.uniform(size=50000) * 50
    (fig, ax) = plt.subplots()
    p1 = ax.plot(x, solid_joinstyle='round', linewidth=2.0)
    fig.canvas.draw()
    path = p1[0].get_path()
    transform = p1[0].get_transform()
    path = transform.transform_path(path)
    simplified = path.cleaned(simplify=True)
    assert simplified.vertices.size == 25512

def test_antiparallel_simplification():
    if False:
        for i in range(10):
            print('nop')

    def _get_simplified(x, y):
        if False:
            i = 10
            return i + 15
        (fig, ax) = plt.subplots()
        p1 = ax.plot(x, y)
        path = p1[0].get_path()
        transform = p1[0].get_transform()
        path = transform.transform_path(path)
        simplified = path.cleaned(simplify=True)
        simplified = transform.inverted().transform_path(simplified)
        return simplified
    x = [0, 0, 0, 0, 0, 1]
    y = [0.5, 1, -1, 1, 2, 0.5]
    simplified = _get_simplified(x, y)
    assert_array_almost_equal([[0.0, 0.5], [0.0, -1.0], [0.0, 2.0], [1.0, 0.5]], simplified.vertices[:-2, :])
    x = [0, 0, 0, 0, 0, 1]
    y = [0.5, 1, -1, 1, -2, 0.5]
    simplified = _get_simplified(x, y)
    assert_array_almost_equal([[0.0, 0.5], [0.0, 1.0], [0.0, -2.0], [1.0, 0.5]], simplified.vertices[:-2, :])
    x = [0, 0, 0, 0, 0, 1]
    y = [0.5, 1, -1, 1, 0, 0.5]
    simplified = _get_simplified(x, y)
    assert_array_almost_equal([[0.0, 0.5], [0.0, 1.0], [0.0, -1.0], [0.0, 0.0], [1.0, 0.5]], simplified.vertices[:-2, :])
    x = [0, 0, 0, 0, 0, 1]
    y = [0.5, 1, 2, 1, 3, 0.5]
    simplified = _get_simplified(x, y)
    assert_array_almost_equal([[0.0, 0.5], [0.0, 3.0], [1.0, 0.5]], simplified.vertices[:-2, :])
    x = [0, 0, 0, 0, 0, 1]
    y = [0.5, 1, 2, 1, 1, 0.5]
    simplified = _get_simplified(x, y)
    assert_array_almost_equal([[0.0, 0.5], [0.0, 2.0], [0.0, 1.0], [1.0, 0.5]], simplified.vertices[:-2, :])

@pytest.mark.parametrize('angle', [0, np.pi / 4, np.pi / 3, np.pi / 2])
@pytest.mark.parametrize('offset', [0, 0.5])
def test_angled_antiparallel(angle, offset):
    if False:
        while True:
            i = 10
    scale = 5
    np.random.seed(19680801)
    vert_offsets = (np.random.rand(15) - offset) * scale
    vert_offsets[0] = 0
    vert_offsets[1] = 1
    x = np.sin(angle) * vert_offsets
    y = np.cos(angle) * vert_offsets
    x_max = x[1:].max()
    x_min = x[1:].min()
    y_max = y[1:].max()
    y_min = y[1:].min()
    if offset > 0:
        p_expected = Path([[0, 0], [x_max, y_max], [x_min, y_min], [x[-1], y[-1]], [0, 0]], codes=[1, 2, 2, 2, 0])
    else:
        p_expected = Path([[0, 0], [x_max, y_max], [x[-1], y[-1]], [0, 0]], codes=[1, 2, 2, 0])
    p = Path(np.vstack([x, y]).T)
    p2 = p.cleaned(simplify=True)
    assert_array_almost_equal(p_expected.vertices, p2.vertices)
    assert_array_equal(p_expected.codes, p2.codes)

def test_sine_plus_noise():
    if False:
        for i in range(10):
            print('nop')
    np.random.seed(0)
    x = np.sin(np.linspace(0, np.pi * 2.0, 50000)) + np.random.uniform(size=50000) * 0.01
    (fig, ax) = plt.subplots()
    p1 = ax.plot(x, solid_joinstyle='round', linewidth=2.0)
    fig.canvas.draw()
    path = p1[0].get_path()
    transform = p1[0].get_transform()
    path = transform.transform_path(path)
    simplified = path.cleaned(simplify=True)
    assert simplified.vertices.size == 25240

@image_comparison(['simplify_curve'], remove_text=True, tol=0.017)
def test_simplify_curve():
    if False:
        return 10
    pp1 = patches.PathPatch(Path([(0, 0), (1, 0), (1, 1), (np.nan, 1), (0, 0), (2, 0), (2, 2), (0, 0)], [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]), fc='none')
    (fig, ax) = plt.subplots()
    ax.add_patch(pp1)
    ax.set_xlim((0, 2))
    ax.set_ylim((0, 2))

@check_figures_equal()
def test_closed_path_nan_removal(fig_test, fig_ref):
    if False:
        i = 10
        return i + 15
    ax_test = fig_test.subplots(2, 2).flatten()
    ax_ref = fig_ref.subplots(2, 2).flatten()
    path = Path([[-3, np.nan], [3, -3], [3, 3], [-3, 3], [-3, -3]], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    ax_test[0].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-3, np.nan], [3, -3], [3, 3], [-3, 3], [-3, np.nan]], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO])
    ax_ref[0].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-2, -2], [2, -2], [2, 2], [-2, np.nan], [-2, -2]], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    ax_test[0].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-2, -2], [2, -2], [2, 2], [-2, np.nan], [-2, -2]], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO])
    ax_ref[0].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-3, np.nan], [3, -3], [3, 3], [-3, 3], [-3, -3], [-2, -2], [2, -2], [2, 2], [-2, np.nan], [-2, -2]], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY, Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    ax_test[1].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-3, np.nan], [3, -3], [3, 3], [-3, 3], [-3, np.nan], [-2, -2], [2, -2], [2, 2], [-2, np.nan], [-2, -2]], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO])
    ax_ref[1].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-1, -1], [1, -1], [1, np.nan], [0, 1], [-1, 1], [-1, -1]], [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY])
    ax_test[2].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-1, -1], [1, -1], [1, np.nan], [0, 1], [-1, 1], [-1, -1]], [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY])
    ax_ref[2].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-3, -3], [3, -3], [3, 0], [0, np.nan], [-3, 3], [-3, -3]], [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.LINETO])
    ax_test[2].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-3, -3], [3, -3], [3, 0], [0, np.nan], [-3, 3], [-3, -3]], [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.LINETO])
    ax_ref[2].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-1, -1], [1, -1], [1, np.nan], [0, 0], [0, 1], [-1, 1], [-1, -1]], [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CLOSEPOLY])
    ax_test[3].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-1, -1], [1, -1], [1, np.nan], [0, 0], [0, 1], [-1, 1], [-1, -1]], [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CLOSEPOLY])
    ax_ref[3].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-2, -2], [2, -2], [2, 0], [0, np.nan], [0, 2], [-2, 2], [-2, -2]], [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.LINETO])
    ax_test[3].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-2, -2], [2, -2], [2, 0], [0, np.nan], [0, 2], [-2, 2], [-2, -2]], [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.LINETO])
    ax_ref[3].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-3, -3], [3, -3], [3, 0], [0, 0], [0, np.nan], [-3, 3], [-3, -3]], [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.LINETO])
    ax_test[3].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-3, -3], [3, -3], [3, 0], [0, 0], [0, np.nan], [-3, 3], [-3, -3]], [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.LINETO])
    ax_ref[3].add_patch(patches.PathPatch(path, facecolor='none'))
    for ax in [*ax_test.flat, *ax_ref.flat]:
        ax.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5))
    remove_ticks_and_titles(fig_test)
    remove_ticks_and_titles(fig_ref)

@check_figures_equal()
def test_closed_path_clipping(fig_test, fig_ref):
    if False:
        i = 10
        return i + 15
    vertices = []
    for roll in range(8):
        offset = 0.1 * roll + 0.1
        pattern = [[-0.5, 1.5], [-0.5, -0.5], [1.5, -0.5], [1.5, 1.5], [1 - offset / 2, 1.5], [1 - offset / 2, offset], [offset / 2, offset], [offset / 2, 1.5]]
        pattern = np.roll(pattern, roll, axis=0)
        pattern = np.concatenate((pattern, pattern[:1, :]))
        vertices.append(pattern)
    codes = np.full(len(vertices[0]), Path.LINETO)
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    codes = np.tile(codes, len(vertices))
    vertices = np.concatenate(vertices)
    fig_test.set_size_inches((5, 5))
    path = Path(vertices, codes)
    fig_test.add_artist(patches.PathPatch(path, facecolor='none'))
    fig_ref.set_size_inches((5, 5))
    codes = codes.copy()
    codes[codes == Path.CLOSEPOLY] = Path.LINETO
    path = Path(vertices, codes)
    fig_ref.add_artist(patches.PathPatch(path, facecolor='none'))

@image_comparison(['hatch_simplify'], remove_text=True)
def test_hatch():
    if False:
        for i in range(10):
            print('nop')
    (fig, ax) = plt.subplots()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, hatch='/'))
    ax.set_xlim((0.45, 0.55))
    ax.set_ylim((0.45, 0.55))

@image_comparison(['fft_peaks'], remove_text=True)
def test_fft_peaks():
    if False:
        while True:
            i = 10
    (fig, ax) = plt.subplots()
    t = np.arange(65536)
    p1 = ax.plot(abs(np.fft.fft(np.sin(2 * np.pi * 0.01 * t) * np.blackman(len(t)))))
    fig.canvas.draw()
    path = p1[0].get_path()
    transform = p1[0].get_transform()
    path = transform.transform_path(path)
    simplified = path.cleaned(simplify=True)
    assert simplified.vertices.size == 36

def test_start_with_moveto():
    if False:
        i = 10
        return i + 15
    data = b'\nZwAAAAku+v9UAQAA+Tj6/z8CAADpQ/r/KAMAANlO+v8QBAAAyVn6//UEAAC6ZPr/2gUAAKpv+v+8\nBgAAm3r6/50HAACLhfr/ewgAAHyQ+v9ZCQAAbZv6/zQKAABepvr/DgsAAE+x+v/lCwAAQLz6/7wM\nAAAxx/r/kA0AACPS+v9jDgAAFN36/zQPAAAF6Pr/AxAAAPfy+v/QEAAA6f36/5wRAADbCPv/ZhIA\nAMwT+/8uEwAAvh77//UTAACwKfv/uRQAAKM0+/98FQAAlT/7/z0WAACHSvv//RYAAHlV+/+7FwAA\nbGD7/3cYAABea/v/MRkAAFF2+//pGQAARIH7/6AaAAA3jPv/VRsAACmX+/8JHAAAHKL7/7ocAAAP\nrfv/ah0AAAO4+/8YHgAA9sL7/8QeAADpzfv/bx8AANzY+/8YIAAA0OP7/78gAADD7vv/ZCEAALf5\n+/8IIgAAqwT8/6kiAACeD/z/SiMAAJIa/P/oIwAAhiX8/4QkAAB6MPz/HyUAAG47/P+4JQAAYkb8\n/1AmAABWUfz/5SYAAEpc/P95JwAAPmf8/wsoAAAzcvz/nCgAACd9/P8qKQAAHIj8/7cpAAAQk/z/\nQyoAAAWe/P/MKgAA+aj8/1QrAADus/z/2isAAOO+/P9eLAAA2Mn8/+AsAADM1Pz/YS0AAMHf/P/g\nLQAAtur8/10uAACr9fz/2C4AAKEA/f9SLwAAlgv9/8ovAACLFv3/QDAAAIAh/f+1MAAAdSz9/ycx\nAABrN/3/mDEAAGBC/f8IMgAAVk39/3UyAABLWP3/4TIAAEFj/f9LMwAANm79/7MzAAAsef3/GjQA\nACKE/f9+NAAAF4/9/+E0AAANmv3/QzUAAAOl/f+iNQAA+a/9/wA2AADvuv3/XDYAAOXF/f+2NgAA\n29D9/w83AADR2/3/ZjcAAMfm/f+7NwAAvfH9/w44AACz/P3/XzgAAKkH/v+vOAAAnxL+//04AACW\nHf7/SjkAAIwo/v+UOQAAgjP+/905AAB5Pv7/JDoAAG9J/v9pOgAAZVT+/606AABcX/7/7zoAAFJq\n/v8vOwAASXX+/207AAA/gP7/qjsAADaL/v/lOwAALZb+/x48AAAjof7/VTwAABqs/v+LPAAAELf+\n/788AAAHwv7/8TwAAP7M/v8hPQAA9df+/1A9AADr4v7/fT0AAOLt/v+oPQAA2fj+/9E9AADQA///\n+T0AAMYO//8fPgAAvRn//0M+AAC0JP//ZT4AAKsv//+GPgAAojr//6U+AACZRf//wj4AAJBQ///d\nPgAAh1v///c+AAB+Zv//Dz8AAHRx//8lPwAAa3z//zk/AABih///TD8AAFmS//9dPwAAUJ3//2w/\nAABHqP//ej8AAD6z//+FPwAANb7//48/AAAsyf//lz8AACPU//+ePwAAGt///6M/AAAR6v//pj8A\nAAj1//+nPwAA/////w=='
    verts = np.frombuffer(base64.decodebytes(data), dtype='<i4')
    verts = verts.reshape((len(verts) // 2, 2))
    path = Path(verts)
    segs = path.iter_segments(transforms.IdentityTransform(), clip=(0.0, 0.0, 100.0, 100.0))
    segs = list(segs)
    assert len(segs) == 1
    assert segs[0][1] == Path.MOVETO

def test_throw_rendering_complexity_exceeded():
    if False:
        for i in range(10):
            print('nop')
    plt.rcParams['path.simplify'] = False
    xx = np.arange(2000000)
    yy = np.random.rand(2000000)
    yy[1000] = np.nan
    (fig, ax) = plt.subplots()
    ax.plot(xx, yy)
    with pytest.raises(OverflowError):
        fig.savefig(io.BytesIO())

@image_comparison(['clipper_edge'], remove_text=True)
def test_clipper():
    if False:
        return 10
    dat = (0, 1, 0, 2, 0, 3, 0, 4, 0, 5)
    fig = plt.figure(figsize=(2, 1))
    fig.subplots_adjust(left=0, bottom=0, wspace=0, hspace=0)
    ax = fig.add_axes((0, 0, 1.0, 1.0), ylim=(0, 5), autoscale_on=False)
    ax.plot(dat)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlim(5, 9)

@image_comparison(['para_equal_perp'], remove_text=True)
def test_para_equal_perp():
    if False:
        i = 10
        return i + 15
    x = np.array([0, 1, 2, 1, 0, -1, 0, 1] + [1] * 128)
    y = np.array([1, 1, 2, 1, 0, -1, 0, 0] + [0] * 128)
    (fig, ax) = plt.subplots()
    ax.plot(x + 1, y + 1)
    ax.plot(x + 1, y + 1, 'ro')

@image_comparison(['clipping_with_nans'])
def test_clipping_with_nans():
    if False:
        while True:
            i = 10
    x = np.linspace(0, 3.14 * 2, 3000)
    y = np.sin(x)
    x[::100] = np.nan
    (fig, ax) = plt.subplots()
    ax.plot(x, y)
    ax.set_ylim(-0.25, 0.25)

def test_clipping_full():
    if False:
        i = 10
        return i + 15
    p = Path([[1e+30, 1e+30]] * 5)
    simplified = list(p.iter_segments(clip=[0, 0, 100, 100]))
    assert simplified == []
    p = Path([[50, 40], [75, 65]], [1, 2])
    simplified = list(p.iter_segments(clip=[0, 0, 100, 100]))
    assert [(list(x), y) for (x, y) in simplified] == [([50, 40], 1), ([75, 65], 2)]
    p = Path([[50, 40]], [1])
    simplified = list(p.iter_segments(clip=[0, 0, 100, 100]))
    assert [(list(x), y) for (x, y) in simplified] == [([50, 40], 1)]