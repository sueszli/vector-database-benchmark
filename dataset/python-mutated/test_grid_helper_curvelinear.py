import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.projections import PolarAxes
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Affine2D, Transform
from matplotlib.testing.decorators import image_comparison
from mpl_toolkits.axisartist import SubplotHost
from mpl_toolkits.axes_grid1.parasite_axes import host_axes_class_factory
from mpl_toolkits.axisartist import angle_helper
from mpl_toolkits.axisartist.axislines import Axes
from mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear

@image_comparison(['custom_transform.png'], style='default', tol=0.2)
def test_custom_transform():
    if False:
        while True:
            i = 10

    class MyTransform(Transform):
        input_dims = output_dims = 2

        def __init__(self, resolution):
            if False:
                print('Hello World!')
            '\n            Resolution is the number of steps to interpolate between each input\n            line segment to approximate its path in transformed space.\n            '
            Transform.__init__(self)
            self._resolution = resolution

        def transform(self, ll):
            if False:
                return 10
            (x, y) = ll.T
            return np.column_stack([x, y - x])
        transform_non_affine = transform

        def transform_path(self, path):
            if False:
                for i in range(10):
                    print('nop')
            ipath = path.interpolated(self._resolution)
            return Path(self.transform(ipath.vertices), ipath.codes)
        transform_path_non_affine = transform_path

        def inverted(self):
            if False:
                print('Hello World!')
            return MyTransformInv(self._resolution)

    class MyTransformInv(Transform):
        input_dims = output_dims = 2

        def __init__(self, resolution):
            if False:
                while True:
                    i = 10
            Transform.__init__(self)
            self._resolution = resolution

        def transform(self, ll):
            if False:
                i = 10
                return i + 15
            (x, y) = ll.T
            return np.column_stack([x, y + x])

        def inverted(self):
            if False:
                for i in range(10):
                    print('nop')
            return MyTransform(self._resolution)
    fig = plt.figure()
    SubplotHost = host_axes_class_factory(Axes)
    tr = MyTransform(1)
    grid_helper = GridHelperCurveLinear(tr)
    ax1 = SubplotHost(fig, 1, 1, 1, grid_helper=grid_helper)
    fig.add_subplot(ax1)
    ax2 = ax1.get_aux_axes(tr, viewlim_mode='equal')
    ax2.plot([3, 6], [5.0, 10.0])
    ax1.set_aspect(1.0)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.grid(True)

@image_comparison(['polar_box.png'], style='default', tol=0.02)
def test_polar_box():
    if False:
        while True:
            i = 10
    fig = plt.figure(figsize=(5, 5))
    tr = Affine2D().scale(np.pi / 180.0, 1.0) + PolarAxes.PolarTransform()
    extreme_finder = angle_helper.ExtremeFinderCycle(20, 20, lon_cycle=360, lat_cycle=None, lon_minmax=None, lat_minmax=(0, np.inf))
    grid_helper = GridHelperCurveLinear(tr, extreme_finder=extreme_finder, grid_locator1=angle_helper.LocatorDMS(12), tick_formatter1=angle_helper.FormatterDMS(), tick_formatter2=FuncFormatter(lambda x, p: 'eight' if x == 8 else f'{int(x)}'))
    ax1 = SubplotHost(fig, 1, 1, 1, grid_helper=grid_helper)
    ax1.axis['right'].major_ticklabels.set_visible(True)
    ax1.axis['top'].major_ticklabels.set_visible(True)
    ax1.axis['right'].get_helper().nth_coord_ticks = 0
    ax1.axis['bottom'].get_helper().nth_coord_ticks = 1
    fig.add_subplot(ax1)
    ax1.axis['lat'] = axis = grid_helper.new_floating_axis(0, 45, axes=ax1)
    axis.label.set_text('Test')
    axis.label.set_visible(True)
    axis.get_helper().set_extremes(2, 12)
    ax1.axis['lon'] = axis = grid_helper.new_floating_axis(1, 6, axes=ax1)
    axis.label.set_text('Test 2')
    axis.get_helper().set_extremes(-180, 90)
    ax2 = ax1.get_aux_axes(tr, viewlim_mode='equal')
    assert ax2.transData == tr + ax1.transData
    ax2.plot(np.linspace(0, 30, 50), np.linspace(10, 10, 50))
    ax1.set_aspect(1.0)
    ax1.set_xlim(-5, 12)
    ax1.set_ylim(-5, 10)
    ax1.grid(True)

@image_comparison(['axis_direction.png'], style='default', tol=0.12)
def test_axis_direction():
    if False:
        return 10
    plt.rcParams['text.kerning_factor'] = 6
    fig = plt.figure(figsize=(5, 5))
    tr = Affine2D().scale(np.pi / 180.0, 1.0) + PolarAxes.PolarTransform()
    extreme_finder = angle_helper.ExtremeFinderCycle(20, 20, lon_cycle=360, lat_cycle=None, lon_minmax=None, lat_minmax=(0, np.inf))
    grid_locator1 = angle_helper.LocatorDMS(12)
    tick_formatter1 = angle_helper.FormatterDMS()
    grid_helper = GridHelperCurveLinear(tr, extreme_finder=extreme_finder, grid_locator1=grid_locator1, tick_formatter1=tick_formatter1)
    ax1 = SubplotHost(fig, 1, 1, 1, grid_helper=grid_helper)
    for axis in ax1.axis.values():
        axis.set_visible(False)
    fig.add_subplot(ax1)
    ax1.axis['lat1'] = axis = grid_helper.new_floating_axis(0, 130, axes=ax1, axis_direction='left')
    axis.label.set_text('Test')
    axis.label.set_visible(True)
    axis.get_helper().set_extremes(0.001, 10)
    ax1.axis['lat2'] = axis = grid_helper.new_floating_axis(0, 50, axes=ax1, axis_direction='right')
    axis.label.set_text('Test')
    axis.label.set_visible(True)
    axis.get_helper().set_extremes(0.001, 10)
    ax1.axis['lon'] = axis = grid_helper.new_floating_axis(1, 10, axes=ax1, axis_direction='bottom')
    axis.label.set_text('Test 2')
    axis.get_helper().set_extremes(50, 130)
    axis.major_ticklabels.set_axis_direction('top')
    axis.label.set_axis_direction('top')
    grid_helper.grid_finder.grid_locator1.set_params(nbins=5)
    grid_helper.grid_finder.grid_locator2.set_params(nbins=5)
    ax1.set_aspect(1.0)
    ax1.set_xlim(-8, 8)
    ax1.set_ylim(-4, 12)
    ax1.grid(True)