"""
=================
Custom projection
=================

Showcase Hammer projection by alleviating many features of Matplotlib.
"""
import numpy as np
import matplotlib
from matplotlib.axes import Axes
import matplotlib.axis as maxis
from matplotlib.patches import Circle
from matplotlib.path import Path
from matplotlib.projections import register_projection
import matplotlib.spines as mspines
from matplotlib.ticker import FixedLocator, Formatter, NullLocator
from matplotlib.transforms import Affine2D, BboxTransformTo, Transform
rcParams = matplotlib.rcParams

class GeoAxes(Axes):
    """
    An abstract base class for geographic projections
    """

    class ThetaFormatter(Formatter):
        """
        Used to format the theta tick labels.  Converts the native
        unit of radians into degrees and adds a degree symbol.
        """

        def __init__(self, round_to=1.0):
            if False:
                for i in range(10):
                    print('nop')
            self._round_to = round_to

        def __call__(self, x, pos=None):
            if False:
                i = 10
                return i + 15
            degrees = round(np.rad2deg(x) / self._round_to) * self._round_to
            return f'{degrees:0.0f}°'
    RESOLUTION = 75

    def _init_axis(self):
        if False:
            while True:
                i = 10
        self.xaxis = maxis.XAxis(self)
        self.yaxis = maxis.YAxis(self)

    def clear(self):
        if False:
            i = 10
            return i + 15
        super().clear()
        self.set_longitude_grid(30)
        self.set_latitude_grid(15)
        self.set_longitude_grid_ends(75)
        self.xaxis.set_minor_locator(NullLocator())
        self.yaxis.set_minor_locator(NullLocator())
        self.xaxis.set_ticks_position('none')
        self.yaxis.set_ticks_position('none')
        self.yaxis.set_tick_params(label1On=True)
        self.grid(rcParams['axes.grid'])
        Axes.set_xlim(self, -np.pi, np.pi)
        Axes.set_ylim(self, -np.pi / 2.0, np.pi / 2.0)

    def _set_lim_and_transforms(self):
        if False:
            for i in range(10):
                print('nop')
        self.transProjection = self._get_core_transform(self.RESOLUTION)
        self.transAffine = self._get_affine_transform()
        self.transAxes = BboxTransformTo(self.bbox)
        self.transData = self.transProjection + self.transAffine + self.transAxes
        self._xaxis_pretransform = Affine2D().scale(1.0, self._longitude_cap * 2.0).translate(0.0, -self._longitude_cap)
        self._xaxis_transform = self._xaxis_pretransform + self.transData
        self._xaxis_text1_transform = Affine2D().scale(1.0, 0.0) + self.transData + Affine2D().translate(0.0, 4.0)
        self._xaxis_text2_transform = Affine2D().scale(1.0, 0.0) + self.transData + Affine2D().translate(0.0, -4.0)
        yaxis_stretch = Affine2D().scale(np.pi * 2, 1).translate(-np.pi, 0)
        yaxis_space = Affine2D().scale(1.0, 1.1)
        self._yaxis_transform = yaxis_stretch + self.transData
        yaxis_text_base = yaxis_stretch + self.transProjection + (yaxis_space + self.transAffine + self.transAxes)
        self._yaxis_text1_transform = yaxis_text_base + Affine2D().translate(-8.0, 0.0)
        self._yaxis_text2_transform = yaxis_text_base + Affine2D().translate(8.0, 0.0)

    def _get_affine_transform(self):
        if False:
            print('Hello World!')
        transform = self._get_core_transform(1)
        (xscale, _) = transform.transform((np.pi, 0))
        (_, yscale) = transform.transform((0, np.pi / 2))
        return Affine2D().scale(0.5 / xscale, 0.5 / yscale).translate(0.5, 0.5)

    def get_xaxis_transform(self, which='grid'):
        if False:
            i = 10
            return i + 15
        '\n        Override this method to provide a transformation for the\n        x-axis tick labels.\n\n        Returns a tuple of the form (transform, valign, halign)\n        '
        if which not in ['tick1', 'tick2', 'grid']:
            raise ValueError("'which' must be one of 'tick1', 'tick2', or 'grid'")
        return self._xaxis_transform

    def get_xaxis_text1_transform(self, pad):
        if False:
            i = 10
            return i + 15
        return (self._xaxis_text1_transform, 'bottom', 'center')

    def get_xaxis_text2_transform(self, pad):
        if False:
            while True:
                i = 10
        '\n        Override this method to provide a transformation for the\n        secondary x-axis tick labels.\n\n        Returns a tuple of the form (transform, valign, halign)\n        '
        return (self._xaxis_text2_transform, 'top', 'center')

    def get_yaxis_transform(self, which='grid'):
        if False:
            while True:
                i = 10
        '\n        Override this method to provide a transformation for the\n        y-axis grid and ticks.\n        '
        if which not in ['tick1', 'tick2', 'grid']:
            raise ValueError("'which' must be one of 'tick1', 'tick2', or 'grid'")
        return self._yaxis_transform

    def get_yaxis_text1_transform(self, pad):
        if False:
            return 10
        '\n        Override this method to provide a transformation for the\n        y-axis tick labels.\n\n        Returns a tuple of the form (transform, valign, halign)\n        '
        return (self._yaxis_text1_transform, 'center', 'right')

    def get_yaxis_text2_transform(self, pad):
        if False:
            return 10
        '\n        Override this method to provide a transformation for the\n        secondary y-axis tick labels.\n\n        Returns a tuple of the form (transform, valign, halign)\n        '
        return (self._yaxis_text2_transform, 'center', 'left')

    def _gen_axes_patch(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Override this method to define the shape that is used for the\n        background of the plot.  It should be a subclass of Patch.\n\n        In this case, it is a Circle (that may be warped by the axes\n        transform into an ellipse).  Any data and gridlines will be\n        clipped to this shape.\n        '
        return Circle((0.5, 0.5), 0.5)

    def _gen_axes_spines(self):
        if False:
            while True:
                i = 10
        return {'geo': mspines.Spine.circular_spine(self, (0.5, 0.5), 0.5)}

    def set_yscale(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if args[0] != 'linear':
            raise NotImplementedError
    set_xscale = set_yscale

    def set_xlim(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        raise TypeError('Changing axes limits of a geographic projection is not supported.  Please consider using Cartopy.')
    set_ylim = set_xlim

    def format_coord(self, lon, lat):
        if False:
            for i in range(10):
                print('nop')
        '\n        Override this method to change how the values are displayed in\n        the status bar.\n\n        In this case, we want them to be displayed in degrees N/S/E/W.\n        '
        (lon, lat) = np.rad2deg([lon, lat])
        ns = 'N' if lat >= 0.0 else 'S'
        ew = 'E' if lon >= 0.0 else 'W'
        return '%f°%s, %f°%s' % (abs(lat), ns, abs(lon), ew)

    def set_longitude_grid(self, degrees):
        if False:
            return 10
        '\n        Set the number of degrees between each longitude grid.\n\n        This is an example method that is specific to this projection\n        class -- it provides a more convenient interface to set the\n        ticking than set_xticks would.\n        '
        grid = np.arange(-180 + degrees, 180, degrees)
        self.xaxis.set_major_locator(FixedLocator(np.deg2rad(grid)))
        self.xaxis.set_major_formatter(self.ThetaFormatter(degrees))

    def set_latitude_grid(self, degrees):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the number of degrees between each longitude grid.\n\n        This is an example method that is specific to this projection\n        class -- it provides a more convenient interface than\n        set_yticks would.\n        '
        grid = np.arange(-90 + degrees, 90, degrees)
        self.yaxis.set_major_locator(FixedLocator(np.deg2rad(grid)))
        self.yaxis.set_major_formatter(self.ThetaFormatter(degrees))

    def set_longitude_grid_ends(self, degrees):
        if False:
            i = 10
            return i + 15
        "\n        Set the latitude(s) at which to stop drawing the longitude grids.\n\n        Often, in geographic projections, you wouldn't want to draw\n        longitude gridlines near the poles.  This allows the user to\n        specify the degree at which to stop drawing longitude grids.\n\n        This is an example method that is specific to this projection\n        class -- it provides an interface to something that has no\n        analogy in the base Axes class.\n        "
        self._longitude_cap = np.deg2rad(degrees)
        self._xaxis_pretransform.clear().scale(1.0, self._longitude_cap * 2.0).translate(0.0, -self._longitude_cap)

    def get_data_ratio(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the aspect ratio of the data itself.\n\n        This method should be overridden by any Axes that have a\n        fixed data ratio.\n        '
        return 1.0

    def can_zoom(self):
        if False:
            return 10
        '\n        Return whether this Axes supports the zoom box button functionality.\n\n        This Axes object does not support interactive zoom box.\n        '
        return False

    def can_pan(self):
        if False:
            i = 10
            return i + 15
        '\n        Return whether this Axes supports the pan/zoom button functionality.\n\n        This Axes object does not support interactive pan/zoom.\n        '
        return False

    def start_pan(self, x, y, button):
        if False:
            i = 10
            return i + 15
        pass

    def end_pan(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def drag_pan(self, button, key, x, y):
        if False:
            for i in range(10):
                print('nop')
        pass

class HammerAxes(GeoAxes):
    """
    A custom class for the Aitoff-Hammer projection, an equal-area map
    projection.

    https://en.wikipedia.org/wiki/Hammer_projection
    """
    name = 'custom_hammer'

    class HammerTransform(Transform):
        """The base Hammer transform."""
        input_dims = output_dims = 2

        def __init__(self, resolution):
            if False:
                i = 10
                return i + 15
            '\n            Create a new Hammer transform.  Resolution is the number of steps\n            to interpolate between each input line segment to approximate its\n            path in curved Hammer space.\n            '
            Transform.__init__(self)
            self._resolution = resolution

        def transform_non_affine(self, ll):
            if False:
                i = 10
                return i + 15
            (longitude, latitude) = ll.T
            half_long = longitude / 2
            cos_latitude = np.cos(latitude)
            sqrt2 = np.sqrt(2)
            alpha = np.sqrt(1 + cos_latitude * np.cos(half_long))
            x = 2 * sqrt2 * (cos_latitude * np.sin(half_long)) / alpha
            y = sqrt2 * np.sin(latitude) / alpha
            return np.column_stack([x, y])

        def transform_path_non_affine(self, path):
            if False:
                while True:
                    i = 10
            ipath = path.interpolated(self._resolution)
            return Path(self.transform(ipath.vertices), ipath.codes)

        def inverted(self):
            if False:
                print('Hello World!')
            return HammerAxes.InvertedHammerTransform(self._resolution)

    class InvertedHammerTransform(Transform):
        input_dims = output_dims = 2

        def __init__(self, resolution):
            if False:
                for i in range(10):
                    print('nop')
            Transform.__init__(self)
            self._resolution = resolution

        def transform_non_affine(self, xy):
            if False:
                while True:
                    i = 10
            (x, y) = xy.T
            z = np.sqrt(1 - (x / 4) ** 2 - (y / 2) ** 2)
            longitude = 2 * np.arctan(z * x / (2 * (2 * z ** 2 - 1)))
            latitude = np.arcsin(y * z)
            return np.column_stack([longitude, latitude])

        def inverted(self):
            if False:
                while True:
                    i = 10
            return HammerAxes.HammerTransform(self._resolution)

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self._longitude_cap = np.pi / 2.0
        super().__init__(*args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.clear()

    def _get_core_transform(self, resolution):
        if False:
            while True:
                i = 10
        return self.HammerTransform(resolution)
register_projection(HammerAxes)
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    (fig, ax) = plt.subplots(subplot_kw={'projection': 'custom_hammer'})
    ax.plot([-1, 1, 1], [-1, -1, 1], 'o-')
    ax.grid()
    plt.show()