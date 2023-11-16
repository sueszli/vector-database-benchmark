import numpy as np
from skimage.transform import SimilarityTransform, warp, resize_local_mean
import warnings
import functools
import inspect
try:
    from skimage.util.dtype import _convert as convert
except ImportError:
    from skimage.util.dtype import convert

class WarpSuite:
    params = ([np.uint8, np.uint16, np.float32, np.float64], [128, 1024, 4096], [0, 1, 3])
    param_names = ['dtype_in', 'N', 'order']

    def setup(self, dtype_in, N, order):
        if False:
            print('Hello World!')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Possible precision loss')
            self.image = convert(np.random.random((N, N)), dtype=dtype_in)
        self.tform = SimilarityTransform(scale=1, rotation=np.pi / 10, translation=(0, 4))
        self.tform.params = self.tform.params.astype('float32')
        self.order = order
        if 'dtype' in inspect.signature(warp).parameters:
            self.warp = functools.partial(warp, dtype=self.image.dtype)
        else:
            self.warp = functools.partial(warp)

    def time_same_type(self, dtype_in, N, order):
        if False:
            print('Hello World!')
        'Test the case where the users wants to preserve their same low\n        precision data type.'
        result = self.warp(self.image, self.tform, order=self.order, preserve_range=True)
        result = result.astype(dtype_in, copy=False)

    def time_to_float64(self, dtype_in, N, order):
        if False:
            while True:
                i = 10
        'Test the case where want to upvert to float64 for continued\n        transformations.'
        warp(self.image, self.tform, order=self.order, preserve_range=True)

class ResizeLocalMeanSuite:
    params = ([np.float32, np.float64], [(512, 512), (2048, 2048), (48, 48, 48), (192, 192, 192)], [(512, 512), (2048, 2048), (48, 48, 48), (192, 192, 192)])
    param_names = ['dtype', 'shape_in', 'shape_out']
    timeout = 180

    def setup(self, dtype, shape_in, shape_out):
        if False:
            return 10
        if len(shape_in) != len(shape_out):
            raise NotImplementedError('shape_in, shape_out must have same dimension')
        self.image = np.zeros(shape_in, dtype=dtype)

    def time_resize_local_mean(self, dtype, shape_in, shape_out):
        if False:
            for i in range(10):
                print('nop')
        resize_local_mean(self.image, shape_out)