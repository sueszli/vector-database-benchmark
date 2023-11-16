"""
==========================
``floating_axes`` features
==========================

Demonstration of features of the :mod:`.floating_axes` module:

* Using `~.axes.Axes.scatter` and `~.axes.Axes.bar` with changing the shape of
  the plot.
* Using `~.floating_axes.GridHelperCurveLinear` to rotate the plot and set the
  plot boundary.
* Using `~.Figure.add_subplot` to create a subplot using the return value from
  `~.floating_axes.GridHelperCurveLinear`.
* Making a sector plot by adding more features to
  `~.floating_axes.GridHelperCurveLinear`.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.angle_helper as angle_helper
import mpl_toolkits.axisartist.floating_axes as floating_axes
from mpl_toolkits.axisartist.grid_finder import DictFormatter, FixedLocator, MaxNLocator
np.random.seed(19680801)

def setup_axes1(fig, rect):
    if False:
        i = 10
        return i + 15
    '\n    A simple one.\n    '
    tr = Affine2D().scale(2, 1).rotate_deg(30)
    grid_helper = floating_axes.GridHelperCurveLinear(tr, extremes=(-0.5, 3.5, 0, 4), grid_locator1=MaxNLocator(nbins=4), grid_locator2=MaxNLocator(nbins=4))
    ax1 = fig.add_subplot(rect, axes_class=floating_axes.FloatingAxes, grid_helper=grid_helper)
    ax1.grid()
    aux_ax = ax1.get_aux_axes(tr)
    return (ax1, aux_ax)

def setup_axes2(fig, rect):
    if False:
        i = 10
        return i + 15
    '\n    With custom locator and formatter.\n    Note that the extreme values are swapped.\n    '
    tr = PolarAxes.PolarTransform()
    pi = np.pi
    angle_ticks = [(0, '$0$'), (0.25 * pi, '$\\frac{1}{4}\\pi$'), (0.5 * pi, '$\\frac{1}{2}\\pi$')]
    grid_locator1 = FixedLocator([v for (v, s) in angle_ticks])
    tick_formatter1 = DictFormatter(dict(angle_ticks))
    grid_locator2 = MaxNLocator(2)
    grid_helper = floating_axes.GridHelperCurveLinear(tr, extremes=(0.5 * pi, 0, 2, 1), grid_locator1=grid_locator1, grid_locator2=grid_locator2, tick_formatter1=tick_formatter1, tick_formatter2=None)
    ax1 = fig.add_subplot(rect, axes_class=floating_axes.FloatingAxes, grid_helper=grid_helper)
    ax1.grid()
    aux_ax = ax1.get_aux_axes(tr)
    aux_ax.patch = ax1.patch
    ax1.patch.zorder = 0.9
    return (ax1, aux_ax)

def setup_axes3(fig, rect):
    if False:
        print('Hello World!')
    '\n    Sometimes, things like axis_direction need to be adjusted.\n    '
    tr_rotate = Affine2D().translate(-95, 0)
    tr_scale = Affine2D().scale(np.pi / 180.0, 1.0)
    tr = tr_rotate + tr_scale + PolarAxes.PolarTransform()
    grid_locator1 = angle_helper.LocatorHMS(4)
    tick_formatter1 = angle_helper.FormatterHMS()
    grid_locator2 = MaxNLocator(3)
    (ra0, ra1) = (8.0 * 15, 14.0 * 15)
    (cz0, cz1) = (0, 14000)
    grid_helper = floating_axes.GridHelperCurveLinear(tr, extremes=(ra0, ra1, cz0, cz1), grid_locator1=grid_locator1, grid_locator2=grid_locator2, tick_formatter1=tick_formatter1, tick_formatter2=None)
    ax1 = fig.add_subplot(rect, axes_class=floating_axes.FloatingAxes, grid_helper=grid_helper)
    ax1.axis['left'].set_axis_direction('bottom')
    ax1.axis['right'].set_axis_direction('top')
    ax1.axis['bottom'].set_visible(False)
    ax1.axis['top'].set_axis_direction('bottom')
    ax1.axis['top'].toggle(ticklabels=True, label=True)
    ax1.axis['top'].major_ticklabels.set_axis_direction('top')
    ax1.axis['top'].label.set_axis_direction('top')
    ax1.axis['left'].label.set_text('cz [km$^{-1}$]')
    ax1.axis['top'].label.set_text('$\\alpha_{1950}$')
    ax1.grid()
    aux_ax = ax1.get_aux_axes(tr)
    aux_ax.patch = ax1.patch
    ax1.patch.zorder = 0.9
    return (ax1, aux_ax)
fig = plt.figure(figsize=(8, 4))
fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95)
(ax1, aux_ax1) = setup_axes1(fig, 131)
aux_ax1.bar([0, 1, 2, 3], [3, 2, 1, 3])
(ax2, aux_ax2) = setup_axes2(fig, 132)
theta = np.random.rand(10) * 0.5 * np.pi
radius = np.random.rand(10) + 1.0
aux_ax2.scatter(theta, radius)
(ax3, aux_ax3) = setup_axes3(fig, 133)
theta = (8 + np.random.rand(10) * (14 - 8)) * 15.0
radius = np.random.rand(10) * 14000.0
aux_ax3.scatter(theta, radius)
plt.show()