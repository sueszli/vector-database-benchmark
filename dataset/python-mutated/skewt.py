"""
===========================================================
SkewT-logP diagram: using transforms and custom projections
===========================================================

This serves as an intensive exercise of Matplotlib's transforms and custom
projection API. This example produces a so-called SkewT-logP diagram, which is
a common plot in meteorology for displaying vertical profiles of temperature.
As far as Matplotlib is concerned, the complexity comes from having X and Y
axes that are not orthogonal. This is handled by including a skew component to
the basic Axes transforms. Additional complexity comes in handling the fact
that the upper and lower X-axes have different data ranges, which necessitates
a bunch of custom classes for ticks, spines, and axis to handle this.
"""
from contextlib import ExitStack
from matplotlib.axes import Axes
import matplotlib.axis as maxis
from matplotlib.projections import register_projection
import matplotlib.spines as mspines
import matplotlib.transforms as transforms

class SkewXTick(maxis.XTick):

    def draw(self, renderer):
        if False:
            while True:
                i = 10
        with ExitStack() as stack:
            for artist in [self.gridline, self.tick1line, self.tick2line, self.label1, self.label2]:
                stack.callback(artist.set_visible, artist.get_visible())
            needs_lower = transforms.interval_contains(self.axes.lower_xlim, self.get_loc())
            needs_upper = transforms.interval_contains(self.axes.upper_xlim, self.get_loc())
            self.tick1line.set_visible(self.tick1line.get_visible() and needs_lower)
            self.label1.set_visible(self.label1.get_visible() and needs_lower)
            self.tick2line.set_visible(self.tick2line.get_visible() and needs_upper)
            self.label2.set_visible(self.label2.get_visible() and needs_upper)
            super().draw(renderer)

    def get_view_interval(self):
        if False:
            return 10
        return self.axes.xaxis.get_view_interval()

class SkewXAxis(maxis.XAxis):

    def _get_tick(self, major):
        if False:
            while True:
                i = 10
        return SkewXTick(self.axes, None, major=major)

    def get_view_interval(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.axes.upper_xlim[0], self.axes.lower_xlim[1])

class SkewSpine(mspines.Spine):

    def _adjust_location(self):
        if False:
            print('Hello World!')
        pts = self._path.vertices
        if self.spine_type == 'top':
            pts[:, 0] = self.axes.upper_xlim
        else:
            pts[:, 0] = self.axes.lower_xlim

class SkewXAxes(Axes):
    name = 'skewx'

    def _init_axis(self):
        if False:
            while True:
                i = 10
        self.xaxis = SkewXAxis(self)
        self.spines.top.register_axis(self.xaxis)
        self.spines.bottom.register_axis(self.xaxis)
        self.yaxis = maxis.YAxis(self)
        self.spines.left.register_axis(self.yaxis)
        self.spines.right.register_axis(self.yaxis)

    def _gen_axes_spines(self):
        if False:
            i = 10
            return i + 15
        spines = {'top': SkewSpine.linear_spine(self, 'top'), 'bottom': mspines.Spine.linear_spine(self, 'bottom'), 'left': mspines.Spine.linear_spine(self, 'left'), 'right': mspines.Spine.linear_spine(self, 'right')}
        return spines

    def _set_lim_and_transforms(self):
        if False:
            return 10
        '\n        This is called once when the plot is created to set up all the\n        transforms for the data, text and grids.\n        '
        rot = 30
        super()._set_lim_and_transforms()
        self.transDataToAxes = self.transScale + self.transLimits + transforms.Affine2D().skew_deg(rot, 0)
        self.transData = self.transDataToAxes + self.transAxes
        self._xaxis_transform = transforms.blended_transform_factory(self.transScale + self.transLimits, transforms.IdentityTransform()) + transforms.Affine2D().skew_deg(rot, 0) + self.transAxes

    @property
    def lower_xlim(self):
        if False:
            for i in range(10):
                print('nop')
        return self.axes.viewLim.intervalx

    @property
    def upper_xlim(self):
        if False:
            for i in range(10):
                print('nop')
        pts = [[0.0, 1.0], [1.0, 1.0]]
        return self.transDataToAxes.inverted().transform(pts)[:, 0]
register_projection(SkewXAxes)
if __name__ == '__main__':
    from io import StringIO
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MultipleLocator, NullFormatter, ScalarFormatter
    data_txt = '\n        978.0    345    7.8    0.8\n        971.0    404    7.2    0.2\n        946.7    610    5.2   -1.8\n        944.0    634    5.0   -2.0\n        925.0    798    3.4   -2.6\n        911.8    914    2.4   -2.7\n        906.0    966    2.0   -2.7\n        877.9   1219    0.4   -3.2\n        850.0   1478   -1.3   -3.7\n        841.0   1563   -1.9   -3.8\n        823.0   1736    1.4   -0.7\n        813.6   1829    4.5    1.2\n        809.0   1875    6.0    2.2\n        798.0   1988    7.4   -0.6\n        791.0   2061    7.6   -1.4\n        783.9   2134    7.0   -1.7\n        755.1   2438    4.8   -3.1\n        727.3   2743    2.5   -4.4\n        700.5   3048    0.2   -5.8\n        700.0   3054    0.2   -5.8\n        698.0   3077    0.0   -6.0\n        687.0   3204   -0.1   -7.1\n        648.9   3658   -3.2  -10.9\n        631.0   3881   -4.7  -12.7\n        600.7   4267   -6.4  -16.7\n        592.0   4381   -6.9  -17.9\n        577.6   4572   -8.1  -19.6\n        555.3   4877  -10.0  -22.3\n        536.0   5151  -11.7  -24.7\n        533.8   5182  -11.9  -25.0\n        500.0   5680  -15.9  -29.9\n        472.3   6096  -19.7  -33.4\n        453.0   6401  -22.4  -36.0\n        400.0   7310  -30.7  -43.7\n        399.7   7315  -30.8  -43.8\n        387.0   7543  -33.1  -46.1\n        382.7   7620  -33.8  -46.8\n        342.0   8398  -40.5  -53.5\n        320.4   8839  -43.7  -56.7\n        318.0   8890  -44.1  -57.1\n        310.0   9060  -44.7  -58.7\n        306.1   9144  -43.9  -57.9\n        305.0   9169  -43.7  -57.7\n        300.0   9280  -43.5  -57.5\n        292.0   9462  -43.7  -58.7\n        276.0   9838  -47.1  -62.1\n        264.0  10132  -47.5  -62.5\n        251.0  10464  -49.7  -64.7\n        250.0  10490  -49.7  -64.7\n        247.0  10569  -48.7  -63.7\n        244.0  10649  -48.9  -63.9\n        243.3  10668  -48.9  -63.9\n        220.0  11327  -50.3  -65.3\n        212.0  11569  -50.5  -65.5\n        210.0  11631  -49.7  -64.7\n        200.0  11950  -49.9  -64.9\n        194.0  12149  -49.9  -64.9\n        183.0  12529  -51.3  -66.3\n        164.0  13233  -55.3  -68.3\n        152.0  13716  -56.5  -69.5\n        150.0  13800  -57.1  -70.1\n        136.0  14414  -60.5  -72.5\n        132.0  14600  -60.1  -72.1\n        131.4  14630  -60.2  -72.2\n        128.0  14792  -60.9  -72.9\n        125.0  14939  -60.1  -72.1\n        119.0  15240  -62.2  -73.8\n        112.0  15616  -64.9  -75.9\n        108.0  15838  -64.1  -75.1\n        107.8  15850  -64.1  -75.1\n        105.0  16010  -64.7  -75.7\n        103.0  16128  -62.9  -73.9\n        100.0  16310  -62.5  -73.5\n    '
    sound_data = StringIO(data_txt)
    (p, h, T, Td) = np.loadtxt(sound_data, unpack=True)
    fig = plt.figure(figsize=(6.5875, 6.2125))
    ax = fig.add_subplot(projection='skewx')
    plt.grid(True)
    ax.semilogy(T, p, color='C3')
    ax.semilogy(Td, p, color='C2')
    l = ax.axvline(0, color='C0')
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(np.linspace(100, 1000, 10))
    ax.set_ylim(1050, 100)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlim(-50, 50)
    plt.show()