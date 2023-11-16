"""Benchmarks for `skimage.morphology`.

See "Writing benchmarks" in the asv docs for more information.
"""
import numpy as np
from numpy.lib import NumpyVersion as Version
import skimage
from skimage import data, morphology, util

class Skeletonize3d:

    def setup(self, *args):
        if False:
            for i in range(10):
                print('nop')
        try:
            if Version(skimage.__version__) < Version('0.16.0'):
                self.skeletonize = morphology.skeletonize_3d
            else:
                self.skeletonize = morphology.skeletonize
        except AttributeError:
            raise NotImplementedError('3d skeletonize unavailable')
        self.image = np.stack(5 * [util.invert(data.horse())])

    def time_skeletonize_3d(self):
        if False:
            return 10
        self.skeletonize(self.image)

    def peakmem_reference(self, *args):
        if False:
            return 10
        'Provide reference for memory measurement with empty benchmark.\n\n        Peakmem benchmarks measure the maximum amount of RAM used by a\n        function. However, this maximum also includes the memory used\n        during the setup routine (as of asv 0.2.1; see [1]_).\n        Measuring an empty peakmem function might allow us to disambiguate\n        between the memory used by setup and the memory used by target (see\n        other ``peakmem_`` functions below).\n\n        References\n        ----------\n        .. [1]: https://asv.readthedocs.io/en/stable/writing_benchmarks.html#peak-memory\n        '
        pass

    def peakmem_skeletonize_3d(self):
        if False:
            while True:
                i = 10
        self.skeletonize(self.image)

class BinaryMorphology2D:
    param_names = ['shape', 'footprint', 'radius', 'decomposition']
    params = [((512, 512),), ('square', 'diamond', 'octagon', 'disk', 'ellipse', 'star'), (1, 3, 5, 15, 25, 40), (None, 'sequence', 'separable', 'crosses')]

    def setup(self, shape, footprint, radius, decomposition):
        if False:
            return 10
        rng = np.random.default_rng(123)
        self.image = rng.standard_normal(shape) < 3.5
        fp_func = getattr(morphology, footprint)
        allow_sequence = ('rectangle', 'square', 'diamond', 'octagon', 'disk')
        allow_separable = ('rectangle', 'square')
        allow_crosses = ('disk', 'ellipse')
        allow_decomp = tuple(set(allow_sequence) | set(allow_separable) | set(allow_crosses))
        footprint_kwargs = {}
        if decomposition == 'sequence' and footprint not in allow_sequence:
            raise NotImplementedError('decomposition unimplemented')
        elif decomposition == 'separable' and footprint not in allow_separable:
            raise NotImplementedError('separable decomposition unavailable')
        elif decomposition == 'crosses' and footprint not in allow_crosses:
            raise NotImplementedError('separable decomposition unavailable')
        if footprint in allow_decomp:
            footprint_kwargs['decomposition'] = decomposition
        if footprint in ['rectangle', 'square']:
            size = 2 * radius + 1
            self.footprint = fp_func(size, **footprint_kwargs)
        elif footprint in ['diamond', 'disk']:
            self.footprint = fp_func(radius, **footprint_kwargs)
        elif footprint == 'star':
            a = max(2 * radius // 3, 1)
            self.footprint = fp_func(a, **footprint_kwargs)
        elif footprint == 'octagon':
            m = n = max(2 * radius // 3, 1)
            self.footprint = fp_func(m, n, **footprint_kwargs)
        elif footprint == 'ellipse':
            if radius > 1:
                self.footprint = fp_func(radius - 1, radius + 1, **footprint_kwargs)
            else:
                self.footprint = fp_func(radius, radius, **footprint_kwargs)

    def time_erosion(self, shape, footprint, radius, *args):
        if False:
            for i in range(10):
                print('nop')
        morphology.binary_erosion(self.image, self.footprint)

class BinaryMorphology3D:
    param_names = ['shape', 'footprint', 'radius', 'decomposition']
    params = [((128, 128, 128),), ('ball', 'cube', 'octahedron'), (1, 3, 5, 10), (None, 'sequence', 'separable')]

    def setup(self, shape, footprint, radius, decomposition):
        if False:
            print('Hello World!')
        rng = np.random.default_rng(123)
        self.image = rng.standard_normal(shape) > -3
        fp_func = getattr(morphology, footprint)
        allow_decomp = ('cube', 'octahedron', 'ball')
        allow_separable = ('cube',)
        if decomposition == 'separable' and footprint != 'cube':
            raise NotImplementedError('separable unavailable')
        footprint_kwargs = {}
        if decomposition is not None and footprint not in allow_decomp:
            raise NotImplementedError('decomposition unimplemented')
        elif decomposition == 'separable' and footprint not in allow_separable:
            raise NotImplementedError('separable decomposition unavailable')
        if footprint in allow_decomp:
            footprint_kwargs['decomposition'] = decomposition
        if footprint == 'cube':
            size = 2 * radius + 1
            self.footprint = fp_func(size, **footprint_kwargs)
        elif footprint in ['ball', 'octahedron']:
            self.footprint = fp_func(radius, **footprint_kwargs)

    def time_erosion(self, shape, footprint, radius, *args):
        if False:
            print('Hello World!')
        morphology.binary_erosion(self.image, self.footprint)

class IsotropicMorphology2D:
    param_names = ['shape', 'radius']
    params = [((512, 512),), (1, 3, 5, 15, 25, 40)]

    def setup(self, shape, radius):
        if False:
            i = 10
            return i + 15
        rng = np.random.default_rng(123)
        self.image = rng.standard_normal(shape) < 3.5

    def time_erosion(self, shape, radius, *args):
        if False:
            return 10
        morphology.isotropic_erosion(self.image, radius)

class GrayMorphology2D(BinaryMorphology2D):

    def time_erosion(self, shape, footprint, radius, *args):
        if False:
            return 10
        morphology.erosion(self.image, self.footprint)

class GrayMorphology3D(BinaryMorphology3D):

    def time_erosion(self, shape, footprint, radius, *args):
        if False:
            i = 10
            return i + 15
        morphology.erosion(self.image, self.footprint)

class GrayReconstruction:
    param_names = ['shape', 'dtype']
    params = [((10, 10), (64, 64), (1200, 1200), (96, 96, 96)), (np.uint8, np.float32, np.float64)]

    def setup(self, shape, dtype):
        if False:
            for i in range(10):
                print('nop')
        rng = np.random.default_rng(123)
        rvals = rng.integers(1, 255, size=shape).astype(dtype=dtype)
        roi1 = tuple((slice(s // 4, s // 2) for s in rvals.shape))
        roi2 = tuple((slice(s // 2 + 1, 3 * s // 4) for s in rvals.shape))
        seed = np.full(rvals.shape, 1, dtype=dtype)
        seed[roi1] = rvals[roi1]
        seed[roi2] = rvals[roi2]
        mask = np.full(seed.shape, 1, dtype=dtype)
        mask[roi1] = 255
        mask[roi2] = 255
        self.seed = seed
        self.mask = mask

    def time_reconstruction(self, shape, dtype):
        if False:
            print('Hello World!')
        morphology.reconstruction(self.seed, self.mask)

    def peakmem_reference(self, *args):
        if False:
            while True:
                i = 10
        'Provide reference for memory measurement with empty benchmark.\n\n        Peakmem benchmarks measure the maximum amount of RAM used by a\n        function. However, this maximum also includes the memory used\n        during the setup routine (as of asv 0.2.1; see [1]_).\n        Measuring an empty peakmem function might allow us to disambiguate\n        between the memory used by setup and the memory used by target (see\n        other ``peakmem_`` functions below).\n\n        References\n        ----------\n        .. [1]: https://asv.readthedocs.io/en/stable/writing_benchmarks.html#peak-memory  # noqa\n        '
        pass

    def peakmem_reconstruction(self, shape, dtype):
        if False:
            return 10
        morphology.reconstruction(self.seed, self.mask)

class LocalMaxima:
    param_names = ['connectivity', 'allow_borders']
    params = [(1, 2), (False, True)]

    def setup(self, *args):
        if False:
            return 10
        self.image = data.moon()

    def time_2d(self, connectivity, allow_borders):
        if False:
            i = 10
            return i + 15
        morphology.local_maxima(self.image, connectivity=connectivity, allow_borders=allow_borders)

    def peakmem_reference(self, *args):
        if False:
            print('Hello World!')
        'Provide reference for memory measurement with empty benchmark.\n\n        .. [1] https://asv.readthedocs.io/en/stable/writing_benchmarks.html#peak-memory\n        '
        pass

    def peakmem_2d(self, connectivity, allow_borders):
        if False:
            return 10
        morphology.local_maxima(self.image, connectivity=connectivity, allow_borders=allow_borders)