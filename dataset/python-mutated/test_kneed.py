"""
This package contains a port of the tests for knee-point detection package, kneed, by
Kevin Arvai and hosted at https://github.com/arvkevi/kneed. This port is maintained
with permission by the Yellowbrick contributors.
"""
import pytest
import matplotlib.pyplot as plt
import numpy as np
from yellowbrick.utils.kneed import KneeLocator
x = np.arange(0, 10)
y_convex_inc = np.array([1, 2, 3, 4, 5, 10, 15, 20, 40, 100])
y_convex_dec = np.array(y_convex_inc[::-1])
y_concave_dec = np.array(100 - y_convex_inc)
y_concave_inc = np.array(100 - y_convex_dec)

def test_concave_increasing():
    if False:
        i = 10
        return i + 15
    'Tests that a correct knee point is detected in\n    curve having concave and increasing nature.'
    kn = KneeLocator(x, y_concave_inc, curve_nature='concave', curve_direction='increasing')
    assert kn.knee == 2

def test_concave_decreasing():
    if False:
        while True:
            i = 10
    'Tests that a correct knee point is detected in\n    curve having concave and decreasing nature.'
    kn = KneeLocator(x, y_concave_dec, curve_nature='concave', curve_direction='decreasing')
    assert kn.knee == 7

def test_convex_increasing():
    if False:
        i = 10
        return i + 15
    'Tests that a correct knee point is detected in\n    curve having convex and increasing nature.'
    kn = KneeLocator(x, y_convex_inc, curve_nature='convex', curve_direction='increasing')
    assert kn.knee == 7

def test_convex_decreasing():
    if False:
        print('Hello World!')
    'Tests that a correct knee point is detected in\n    curve having convex and decreasing nature.'
    kn = KneeLocator(x, y_convex_dec, curve_nature='convex', curve_direction='decreasing')
    assert kn.knee == 2

def test_concave_increasing_truncated():
    if False:
        print('Hello World!')
    'Tests that a correct knee point is detected in\n    curve having truncated concave increasing nature'
    kn = KneeLocator(x[:-3] / 10, y_concave_inc[:-3] / 10, curve_nature='concave', curve_direction='increasing')
    assert kn.knee == 0.2

def test_concave_decreasing_truncated():
    if False:
        print('Hello World!')
    'Tests that a correct knee point is detected in\n    curve having truncated concave decreasing nature'
    kn = KneeLocator(x[:-3] / 10, y_concave_dec[:-3] / 10, curve_nature='concave', curve_direction='decreasing')
    assert kn.knee == 0.4

def test_convex_increasing_truncated():
    if False:
        while True:
            i = 10
    'Tests that a correct knee point is detected in\n    curve having truncated convex increasing nature'
    kn = KneeLocator(x[:-3] / 10, y_convex_inc[:-3] / 10, curve_nature='convex', curve_direction='increasing')
    assert kn.knee == 0.4

def test_convex_decreasing_truncated():
    if False:
        print('Hello World!')
    'Tests that a correct knee point is detected in\n    curve having truncated convex decreasing nature'
    kn = KneeLocator(x[:-3] / 10, y_convex_dec[:-3] / 10, curve_nature='convex', curve_direction='decreasing')
    assert kn.knee == 0.2

def test_x_equals_y():
    if False:
        while True:
            i = 10
    'Test that a runtime warning is raised when no maxima are found'
    x = range(10)
    y = [1] * len(x)
    with pytest.warns(RuntimeWarning):
        KneeLocator(x, y)

@pytest.mark.parametrize('online, expected', [(True, 482), (False, 22)])
def test_gamma_online_offline(online, expected):
    if False:
        while True:
            i = 10
    'Tests online and offline knee detection.\n    Notable that a large number of samples are highly sensitive to S parameter\n    '
    np.random.seed(23)
    n = 1000
    x = range(1, n + 1)
    y = sorted(np.random.gamma(0.5, 1.0, n), reverse=True)
    kl = KneeLocator(x, y, curve_nature='convex', curve_direction='decreasing', online=online)
    assert kl.knee == expected

def test_properties():
    if False:
        i = 10
        return i + 15
    'Tests that elbow and knee can be used interchangeably.'
    kn = KneeLocator(x, y_concave_inc, curve_nature='concave', curve_direction='increasing')
    assert kn.knee == kn.elbow
    assert kn.norm_knee == kn.norm_elbow
    assert kn.all_knees == kn.all_elbows
    assert kn.all_norm_knees == kn.all_norm_elbows

def test_plot_knee_normalized():
    if False:
        print('Hello World!')
    'Test that plotting is functional'
    with np.errstate(divide='ignore'):
        x = np.linspace(0.0, 1, 10)
        y = np.true_divide(-1, x + 0.1) + 5
    kl = KneeLocator(x, y, S=1.0, curve_nature='concave')
    num_figures_before = plt.gcf().number
    kl.plot_knee_normalized()
    num_figures_after = plt.gcf().number
    assert num_figures_before < num_figures_after

def test_plot_knee():
    if False:
        print('Hello World!')
    'Test that plotting is functional'
    with np.errstate(divide='ignore'):
        x = np.linspace(0.0, 1, 10)
        y = np.true_divide(-1, x + 0.1) + 5
    kl = KneeLocator(x, y, S=1.0, curve_nature='concave')
    num_figures_before = plt.gcf().number
    kl.plot_knee()
    num_figures_after = plt.gcf().number
    assert num_figures_before < num_figures_after

def test_y():
    if False:
        print('Hello World!')
    'Test the y value'
    with np.errstate(divide='ignore'):
        x = np.linspace(0.0, 1, 10)
        y = np.true_divide(-1, x + 0.1) + 5
    kl = KneeLocator(x, y, S=1.0, curve_nature='concave')
    assert kl.knee_y == pytest.approx(1.897, 0.03)
    assert kl.all_knees_y[0] == pytest.approx(1.897, 0.03)
    assert kl.norm_knee_y == pytest.approx(0.758, 0.03)
    assert kl.all_norm_knees_y[0] == pytest.approx(0.758, 0.03)
    assert kl.elbow_y == pytest.approx(1.897, 0.03)
    assert kl.all_elbows_y[0] == pytest.approx(1.897, 0.03)
    assert kl.norm_elbow_y == pytest.approx(0.758, 0.03)
    assert kl.all_norm_elbows_y[0] == pytest.approx(0.758, 0.03)