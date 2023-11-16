import os, sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.normpath(os.path.join(current_path, '../')))
import numpy as np

class BasePlane(object):

    def __init__(self, _points=None, idx=None):
        if False:
            return 10
        self.points = None
        self.idx = idx
        self.bbox_x = None
        self.bbox_y = None
        self.bbox_z = None
        self.set_points(_points)

    def set_points(self, _points):
        if False:
            for i in range(10):
                print('nop')
        if _points is None:
            return
        if type(_points) is np.ndarray:
            self.points = _points
        elif type(_points) is list:
            self.points = np.array(_points)
        else:
            raise BaseException('[ERROR] @ BasePlane.set_points: _points must be an instance of numpy.ndarray of list. Type of your input = {}'.format(type(_points)))
        x = self.points[:, 0]
        y = self.points[:, 1]
        z = self.points[:, 2]
        self.set_bbox(x.min(), x.max(), y.min(), y.max(), z.min(), z.max())

    def set_bbox(self, xmin, xmax, ymin, ymax, zmin, zmax):
        if False:
            for i in range(10):
                print('nop')
        self.bbox_x = [xmin, xmax]
        self.bbox_y = [ymin, ymax]
        self.bbox_z = [zmin, zmax]

    def is_out_of_xy_range(self, xlim, ylim):
        if False:
            print('Hello World!')
        'line이 완전히 벗어났을 때만 True. 즉, 살짝 겹쳤을 때는 False이다.'
        if self.bbox_x is None or self.bbox_y is None:
            raise BaseException('[ERROR] bbox is not set')
        x_min = self.bbox_x[0]
        x_max = self.bbox_x[1]
        y_min = self.bbox_y[0]
        y_max = self.bbox_y[1]
        if x_max < xlim[0] or xlim[1] < x_min:
            x_out = True
        else:
            x_out = False
        if y_max < ylim[0] or ylim[1] < y_min:
            y_out = True
        else:
            y_out = False
        return x_out or y_out

    def is_completely_included_in_xy_range(self, xlim, ylim):
        if False:
            print('Hello World!')
        'line이 완전히 포함될 때만 True. 즉, 살짝 겹쳤을 때는 False이다.'
        if self.bbox_x is None or self.bbox_y is None:
            raise BaseException('[ERROR] bbox is not set')
        x_min = self.bbox_x[0]
        x_max = self.bbox_x[1]
        y_min = self.bbox_y[0]
        y_max = self.bbox_y[1]
        if xlim[0] <= x_min and x_max <= xlim[1]:
            x_in = True
        else:
            x_in = False
        if ylim[0] <= y_min and y_max <= ylim[1]:
            y_in = True
        else:
            y_in = False
        return x_in and y_in

    def calculate_centroid(self):
        if False:
            while True:
                i = 10
        sx = sy = sz = sL = 0
        for i in range(len(self.points)):
            (x0, y0, z0) = self.points[i - 1]
            (x1, y1, z1) = self.points[i]
            L = ((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) ** 0.5
            sx += (x0 + x1) / 2 * L
            sy += (y0 + y1) / 2 * L
            sz += (z0 + z1) / 2 * L
            sL += L
        centroid_x = sx / sL
        centroid_y = sy / sL
        centroid_z = sz / sL
        return np.array([centroid_x, centroid_y, centroid_z])