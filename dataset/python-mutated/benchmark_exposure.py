import math
import numpy as np
from skimage import data, img_as_float
from skimage.transform import rescale
from skimage import exposure

class ExposureSuite:
    """Benchmark for exposure routines in scikit-image."""

    def setup(self):
        if False:
            return 10
        self.image_u8 = data.moon()
        self.image = img_as_float(self.image_u8)
        self.image = rescale(self.image, 2.0, anti_aliasing=False)
        (self.p2, self.p98) = np.percentile(self.image, (2, 98))

    def time_equalize_hist(self):
        if False:
            while True:
                i = 10
        for i in range(10):
            exposure.equalize_hist(self.image)

    def time_equalize_adapthist(self):
        if False:
            return 10
        exposure.equalize_adapthist(self.image, clip_limit=0.03)

    def time_rescale_intensity(self):
        if False:
            while True:
                i = 10
        exposure.rescale_intensity(self.image, in_range=(self.p2, self.p98))

    def time_histogram(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(10):
            exposure.histogram(self.image)

    def time_gamma_adjust_u8(self):
        if False:
            i = 10
            return i + 15
        for i in range(10):
            _ = exposure.adjust_gamma(self.image_u8)

class MatchHistogramsSuite:
    param_names = ['shape', 'dtype', 'multichannel']
    params = [((64, 64), (256, 256), (1024, 1024)), (np.uint8, np.uint32, np.float32, np.float64), (False, True)]

    def _tile_to_shape(self, image, shape, multichannel):
        if False:
            while True:
                i = 10
        n_tile = tuple((math.ceil(s / n) for (s, n) in zip(shape, image.shape)))
        if multichannel:
            image = image[..., np.newaxis]
            n_tile = n_tile + (3,)
        image = np.tile(image, n_tile)
        sl = tuple((slice(s) for s in shape))
        return image[sl]
    'Benchmark for exposure routines in scikit-image.'

    def setup(self, shape, dtype, multichannel):
        if False:
            while True:
                i = 10
        self.image = data.moon().astype(dtype, copy=False)
        self.reference = data.camera().astype(dtype, copy=False)
        self.image = self._tile_to_shape(self.image, shape, multichannel)
        self.reference = self._tile_to_shape(self.reference, shape, multichannel)
        channel_axis = -1 if multichannel else None
        self.kwargs = {'channel_axis': channel_axis}

    def time_match_histogram(self, *args):
        if False:
            i = 10
            return i + 15
        exposure.match_histograms(self.image, self.reference, **self.kwargs)

    def peakmem_reference(self, *args):
        if False:
            i = 10
            return i + 15
        'Provide reference for memory measurement with empty benchmark.\n\n        Peakmem benchmarks measure the maximum amount of RAM used by a\n        function. However, this maximum also includes the memory used\n        during the setup routine (as of asv 0.2.1; see [1]_).\n        Measuring an empty peakmem function might allow us to disambiguate\n        between the memory used by setup and the memory used by target (see\n        other ``peakmem_`` functions below).\n\n        References\n        ----------\n        .. [1]: https://asv.readthedocs.io/en/stable/writing_benchmarks.html#peak-memory  # noqa\n        '
        pass

    def peakmem_match_histogram(self, *args):
        if False:
            i = 10
            return i + 15
        exposure.match_histograms(self.image, self.reference, **self.kwargs)