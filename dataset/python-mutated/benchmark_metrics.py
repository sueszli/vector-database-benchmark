import numpy as np
try:
    from skimage import metrics
except ImportError:
    pass

class SetMetricsSuite:
    shape = (6, 6)
    coords_a = np.zeros(shape, dtype=bool)
    coords_b = np.zeros(shape, dtype=bool)

    def setup(self):
        if False:
            print('Hello World!')
        points_a = (1, 0)
        points_b = (5, 2)
        self.coords_a[points_a] = True
        self.coords_b[points_b] = True

    def time_hausdorff_distance(self):
        if False:
            for i in range(10):
                print('nop')
        metrics.hausdorff_distance(self.coords_a, self.coords_b)

    def time_modified_hausdorff_distance(self):
        if False:
            for i in range(10):
                print('nop')
        metrics.hausdorff_distance(self.coords_a, self.coords_b, method='modified')

    def time_hausdorff_pair(self):
        if False:
            while True:
                i = 10
        metrics.hausdorff_pair(self.coords_a, self.coords_b)