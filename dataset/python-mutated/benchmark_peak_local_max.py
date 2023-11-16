import inspect
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
peak_kwargs = {}
parameters = inspect.signature(peak_local_max).parameters
if 'indices' in parameters and parameters['indices'].default:
    peak_kwargs = {'indices': False}

class PeakLocalMaxSuite:

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        mask = np.zeros([500, 500], dtype=bool)
        (x, y) = np.indices((500, 500))
        x_c = x // 20 * 20 + 10
        y_c = y // 20 * 20 + 10
        mask[(x - x_c) ** 2 + (y - y_c) ** 2 < 8 ** 2] = True
        (self.labels, num_objs) = ndi.label(mask)
        self.dist = ndi.distance_transform_edt(mask)

    def time_peak_local_max(self):
        if False:
            print('Hello World!')
        peak_local_max(self.dist, labels=self.labels, min_distance=20, exclude_border=False, **peak_kwargs)