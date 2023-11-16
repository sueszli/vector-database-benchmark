"""
.. _custom_scale:

============
Custom scale
============

Create a custom scale, by implementing the scaling use for latitude data in a
Mercator Projection.

Unless you are making special use of the `.Transform` class, you probably
don't need to use this verbose method, and instead can use `~.scale.FuncScale`
and the ``'function'`` option of `~.Axes.set_xscale` and `~.Axes.set_yscale`.
See the last example in :doc:`/gallery/scales/scales`.
"""
import numpy as np
from numpy import ma
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FixedLocator, FuncFormatter

class MercatorLatitudeScale(mscale.ScaleBase):
    """
    Scales data in range -pi/2 to pi/2 (-90 to 90 degrees) using
    the system used to scale latitudes in a Mercator__ projection.

    The scale function:
      ln(tan(y) + sec(y))

    The inverse scale function:
      atan(sinh(y))

    Since the Mercator scale tends to infinity at +/- 90 degrees,
    there is user-defined threshold, above and below which nothing
    will be plotted.  This defaults to +/- 85 degrees.

    __ https://en.wikipedia.org/wiki/Mercator_projection
    """
    name = 'mercator'

    def __init__(self, axis, *, thresh=np.deg2rad(85), **kwargs):
        if False:
            print('Hello World!')
        "\n        Any keyword arguments passed to ``set_xscale`` and ``set_yscale`` will\n        be passed along to the scale's constructor.\n\n        thresh: The degree above which to crop the data.\n        "
        super().__init__(axis)
        if thresh >= np.pi / 2:
            raise ValueError('thresh must be less than pi/2')
        self.thresh = thresh

    def get_transform(self):
        if False:
            i = 10
            return i + 15
        '\n        Override this method to return a new instance that does the\n        actual transformation of the data.\n\n        The MercatorLatitudeTransform class is defined below as a\n        nested class of this one.\n        '
        return self.MercatorLatitudeTransform(self.thresh)

    def set_default_locators_and_formatters(self, axis):
        if False:
            i = 10
            return i + 15
        '\n        Override to set up the locators and formatters to use with the\n        scale.  This is only required if the scale requires custom\n        locators and formatters.  Writing custom locators and\n        formatters is rather outside the scope of this example, but\n        there are many helpful examples in :mod:`.ticker`.\n\n        In our case, the Mercator example uses a fixed locator from -90 to 90\n        degrees and a custom formatter to convert the radians to degrees and\n        put a degree symbol after the value.\n        '
        fmt = FuncFormatter(lambda x, pos=None: f'{np.degrees(x):.0f}°')
        axis.set(major_locator=FixedLocator(np.radians(range(-90, 90, 10))), major_formatter=fmt, minor_formatter=fmt)

    def limit_range_for_scale(self, vmin, vmax, minpos):
        if False:
            for i in range(10):
                print('nop')
        '\n        Override to limit the bounds of the axis to the domain of the\n        transform.  In the case of Mercator, the bounds should be\n        limited to the threshold that was passed in.  Unlike the\n        autoscaling provided by the tick locators, this range limiting\n        will always be adhered to, whether the axis range is set\n        manually, determined automatically or changed through panning\n        and zooming.\n        '
        return (max(vmin, -self.thresh), min(vmax, self.thresh))

    class MercatorLatitudeTransform(mtransforms.Transform):
        input_dims = output_dims = 1

        def __init__(self, thresh):
            if False:
                print('Hello World!')
            mtransforms.Transform.__init__(self)
            self.thresh = thresh

        def transform_non_affine(self, a):
            if False:
                for i in range(10):
                    print('nop')
            '\n            This transform takes a numpy array and returns a transformed copy.\n            Since the range of the Mercator scale is limited by the\n            user-specified threshold, the input array must be masked to\n            contain only valid values.  Matplotlib will handle masked arrays\n            and remove the out-of-range data from the plot.  However, the\n            returned array *must* have the same shape as the input array, since\n            these values need to remain synchronized with values in the other\n            dimension.\n            '
            masked = ma.masked_where((a < -self.thresh) | (a > self.thresh), a)
            if masked.mask.any():
                return ma.log(np.abs(ma.tan(masked) + 1 / ma.cos(masked)))
            else:
                return np.log(np.abs(np.tan(a) + 1 / np.cos(a)))

        def inverted(self):
            if False:
                return 10
            '\n            Override this method so Matplotlib knows how to get the\n            inverse transform for this transform.\n            '
            return MercatorLatitudeScale.InvertedMercatorLatitudeTransform(self.thresh)

    class InvertedMercatorLatitudeTransform(mtransforms.Transform):
        input_dims = output_dims = 1

        def __init__(self, thresh):
            if False:
                i = 10
                return i + 15
            mtransforms.Transform.__init__(self)
            self.thresh = thresh

        def transform_non_affine(self, a):
            if False:
                while True:
                    i = 10
            return np.arctan(np.sinh(a))

        def inverted(self):
            if False:
                i = 10
                return i + 15
            return MercatorLatitudeScale.MercatorLatitudeTransform(self.thresh)
mscale.register_scale(MercatorLatitudeScale)
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    t = np.arange(-180.0, 180.0, 0.1)
    s = np.radians(t) / 2.0
    plt.plot(t, s, '-', lw=2)
    plt.yscale('mercator')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Mercator projection')
    plt.grid(True)
    plt.show()