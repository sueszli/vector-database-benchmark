import os, sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.normpath(os.path.join(current_path, '../')))
import numpy as np

class BaseLine(object):

    def __init__(self, _points=None, idx=None):
        if False:
            i = 10
            return i + 15
        self.points = None
        self.idx = idx
        self.bbox_x = None
        self.bbox_y = None
        self.bbox_z = None
        self.set_points(_points)

    def set_points(self, _points):
        if False:
            return 10
        if _points is None:
            return
        if type(_points) is np.ndarray:
            self.points = _points
        elif type(_points) is list:
            self.points = np.array(_points)
        else:
            raise BaseException('[ERROR] @ BaseLine.set_points: _points must be an instance of numpy.ndarray of list. Type of your input = {}'.format(type(_points)))
        x = _points[:, 0]
        y = _points[:, 1]
        z = _points[:, 2]
        self.set_bbox(x.min(), x.max(), y.min(), y.max(), z.min(), z.max())

    def set_bbox(self, xmin, xmax, ymin, ymax, zmin, zmax):
        if False:
            i = 10
            return i + 15
        self.bbox_x = [xmin, xmax]
        self.bbox_y = [ymin, ymax]
        self.bbox_z = [zmin, zmax]

    def is_out_of_xy_range(self, xlim, ylim):
        if False:
            for i in range(10):
                print('nop')
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
            while True:
                i = 10
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

    def decimate_points(self, decimation):
        if False:
            for i in range(10):
                print('nop')
        _indx_del = list()
        for i in range(len(self.points)):
            if i % decimation != 0:
                _indx_del.append(i)
            if i == len(self.points) - 1:
                _indx_del.pop()
        _decimated_array = np.delete(self.points, _indx_del, 0)
        self.points = _decimated_array

    def get_num_points(self):
        if False:
            while True:
                i = 10
        return self.points.shape[0]

    def get_total_distance(self):
        if False:
            for i in range(10):
                print('nop')
        total_distance = 0
        for i in range(len(self.points) - 1):
            vect = self.points[i + 1] - self.points[i]
            dist_between_each_point_pair = np.linalg.norm(vect, ord=2)
            total_distance += dist_between_each_point_pair
        return total_distance

    def add_new_points(self, points_to_add):
        if False:
            return 10
        '\n        현재 있는 points에 점을 추가한다\n        '
        self.set_points(np.vstack((self.points, points_to_add)))