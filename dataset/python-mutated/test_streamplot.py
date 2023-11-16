import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.transforms as mtransforms

def velocity_field():
    if False:
        while True:
            i = 10
    (Y, X) = np.mgrid[-3:3:100j, -3:3:200j]
    U = -1 - X ** 2 + Y
    V = 1 + X - Y ** 2
    return (X, Y, U, V)

def swirl_velocity_field():
    if False:
        return 10
    x = np.linspace(-3.0, 3.0, 200)
    y = np.linspace(-3.0, 3.0, 100)
    (X, Y) = np.meshgrid(x, y)
    a = 0.1
    U = np.cos(a) * -Y - np.sin(a) * X
    V = np.sin(a) * -Y + np.cos(a) * X
    return (x, y, U, V)

@image_comparison(['streamplot_startpoints'], remove_text=True, style='mpl20', extensions=['png'])
def test_startpoints():
    if False:
        while True:
            i = 10
    (X, Y, U, V) = velocity_field()
    (start_x, start_y) = np.meshgrid(np.linspace(X.min(), X.max(), 5), np.linspace(Y.min(), Y.max(), 5))
    start_points = np.column_stack([start_x.ravel(), start_y.ravel()])
    plt.streamplot(X, Y, U, V, start_points=start_points)
    plt.plot(start_x, start_y, 'ok')

@image_comparison(['streamplot_colormap'], remove_text=True, style='mpl20', tol=0.022)
def test_colormap():
    if False:
        for i in range(10):
            print('nop')
    (X, Y, U, V) = velocity_field()
    plt.streamplot(X, Y, U, V, color=U, density=0.6, linewidth=2, cmap=plt.cm.autumn)
    plt.colorbar()

@image_comparison(['streamplot_linewidth'], remove_text=True, style='mpl20', tol=0.002)
def test_linewidth():
    if False:
        print('Hello World!')
    (X, Y, U, V) = velocity_field()
    speed = np.hypot(U, V)
    lw = 5 * speed / speed.max()
    ax = plt.figure().subplots()
    ax.streamplot(X, Y, U, V, density=[0.5, 1], color='k', linewidth=lw)

@image_comparison(['streamplot_masks_and_nans'], remove_text=True, style='mpl20')
def test_masks_and_nans():
    if False:
        for i in range(10):
            print('nop')
    (X, Y, U, V) = velocity_field()
    mask = np.zeros(U.shape, dtype=bool)
    mask[40:60, 80:120] = 1
    U[:20, :40] = np.nan
    U = np.ma.array(U, mask=mask)
    ax = plt.figure().subplots()
    with np.errstate(invalid='ignore'):
        ax.streamplot(X, Y, U, V, color=U, cmap=plt.cm.Blues)

@image_comparison(['streamplot_maxlength.png'], remove_text=True, style='mpl20', tol=0.302)
def test_maxlength():
    if False:
        print('Hello World!')
    (x, y, U, V) = swirl_velocity_field()
    ax = plt.figure().subplots()
    ax.streamplot(x, y, U, V, maxlength=10.0, start_points=[[0.0, 1.5]], linewidth=2, density=2)
    assert ax.get_xlim()[-1] == ax.get_ylim()[-1] == 3
    ax.set(xlim=(None, 3.2555988021882305), ylim=(None, 3.078326760195413))

@image_comparison(['streamplot_maxlength_no_broken.png'], remove_text=True, style='mpl20', tol=0.302)
def test_maxlength_no_broken():
    if False:
        print('Hello World!')
    (x, y, U, V) = swirl_velocity_field()
    ax = plt.figure().subplots()
    ax.streamplot(x, y, U, V, maxlength=10.0, start_points=[[0.0, 1.5]], linewidth=2, density=2, broken_streamlines=False)
    assert ax.get_xlim()[-1] == ax.get_ylim()[-1] == 3
    ax.set(xlim=(None, 3.2555988021882305), ylim=(None, 3.078326760195413))

@image_comparison(['streamplot_direction.png'], remove_text=True, style='mpl20', tol=0.073)
def test_direction():
    if False:
        while True:
            i = 10
    (x, y, U, V) = swirl_velocity_field()
    plt.streamplot(x, y, U, V, integration_direction='backward', maxlength=1.5, start_points=[[1.5, 0.0]], linewidth=2, density=2)

def test_streamplot_limits():
    if False:
        print('Hello World!')
    ax = plt.axes()
    x = np.linspace(-5, 10, 20)
    y = np.linspace(-2, 4, 10)
    (y, x) = np.meshgrid(y, x)
    trans = mtransforms.Affine2D().translate(25, 32) + ax.transData
    plt.barbs(x, y, np.sin(x), np.cos(y), transform=trans)
    assert_array_almost_equal(ax.dataLim.bounds, (20, 30, 15, 6), decimal=1)

def test_streamplot_grid():
    if False:
        for i in range(10):
            print('nop')
    u = np.ones((2, 2))
    v = np.zeros((2, 2))
    x = np.array([[10, 20], [10, 30]])
    y = np.array([[10, 10], [20, 20]])
    with pytest.raises(ValueError, match="The rows of 'x' must be equal"):
        plt.streamplot(x, y, u, v)
    x = np.array([[10, 20], [10, 20]])
    y = np.array([[10, 10], [20, 30]])
    with pytest.raises(ValueError, match="The columns of 'y' must be equal"):
        plt.streamplot(x, y, u, v)
    x = np.array([[10, 20], [10, 20]])
    y = np.array([[10, 10], [20, 20]])
    plt.streamplot(x, y, u, v)
    x = np.array([0, 10])
    y = np.array([[[0, 10]]])
    with pytest.raises(ValueError, match="'y' can have at maximum 2 dimensions"):
        plt.streamplot(x, y, u, v)
    u = np.ones((3, 3))
    v = np.zeros((3, 3))
    x = np.array([0, 10, 20])
    y = np.array([0, 10, 30])
    with pytest.raises(ValueError, match="'y' values must be equally spaced"):
        plt.streamplot(x, y, u, v)
    x = np.array([0, 20, 40])
    y = np.array([0, 20, 10])
    with pytest.raises(ValueError, match="'y' must be strictly increasing"):
        plt.streamplot(x, y, u, v)

def test_streamplot_inputs():
    if False:
        for i in range(10):
            print('nop')
    plt.streamplot(np.arange(3), np.arange(3), np.full((3, 3), np.nan), np.full((3, 3), np.nan), color=np.random.rand(3, 3))
    plt.streamplot(range(3), range(3), np.random.rand(3, 3), np.random.rand(3, 3))