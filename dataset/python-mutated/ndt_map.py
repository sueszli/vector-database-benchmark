"""
Normal Distribution Transform (NDTGrid) mapping sample
"""
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from Mapping.grid_map_lib.grid_map_lib import GridMap
from utils.plot import plot_covariance_ellipse

class NDTMap:
    """
    Normal Distribution Transform (NDT) map class

    :param ox: obstacle x position list
    :param oy: obstacle y position list
    :param resolution: grid resolution [m]
    """

    class NDTGrid:
        """
        NDT grid
        """

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.n_points = 0
            self.mean_x = None
            self.mean_y = None
            self.center_grid_x = None
            self.center_grid_y = None
            self.covariance = None
            self.eig_vec = None
            self.eig_values = None

    def __init__(self, ox, oy, resolution):
        if False:
            i = 10
            return i + 15
        self.min_n_points = 3
        self.resolution = resolution
        width = int((max(ox) - min(ox)) / resolution) + 3
        height = int((max(oy) - min(oy)) / resolution) + 3
        center_x = np.mean(ox)
        center_y = np.mean(oy)
        self.ox = ox
        self.oy = oy
        self.grid_index_map = self._create_grid_index_map(ox, oy)
        self._construct_grid_map(center_x, center_y, height, ox, oy, resolution, width)

    def _construct_grid_map(self, center_x, center_y, height, ox, oy, resolution, width):
        if False:
            print('Hello World!')
        self.grid_map = GridMap(width, height, resolution, center_x, center_y, self.NDTGrid())
        for (grid_index, inds) in self.grid_index_map.items():
            ndt = self.NDTGrid()
            ndt.n_points = len(inds)
            if ndt.n_points >= self.min_n_points:
                ndt.mean_x = np.mean(ox[inds])
                ndt.mean_y = np.mean(oy[inds])
                (ndt.center_grid_x, ndt.center_grid_y) = self.grid_map.calc_grid_central_xy_position_from_grid_index(grid_index)
                ndt.covariance = np.cov(ox[inds], oy[inds])
                (ndt.eig_values, ndt.eig_vec) = np.linalg.eig(ndt.covariance)
                self.grid_map.data[grid_index] = ndt

    def _create_grid_index_map(self, ox, oy):
        if False:
            i = 10
            return i + 15
        grid_index_map = defaultdict(list)
        for i in range(len(ox)):
            grid_index = self.grid_map.calc_grid_index_from_xy_pos(ox[i], oy[i])
            grid_index_map[grid_index].append(i)
        return grid_index_map

def create_dummy_observation_data():
    if False:
        while True:
            i = 10
    ox = []
    oy = []
    for y in range(-50, 50):
        ox.append(-20.0)
        oy.append(y)
    for y in range(-50, 0):
        ox.append(20.0)
        oy.append(y)
    for x in range(20, 50):
        ox.append(x)
        oy.append(0)
    for x in range(20, 50):
        ox.append(x)
        oy.append(x / 2.0 + 10)
    for y in range(20, 50):
        ox.append(20)
        oy.append(y)
    ox = np.array(ox)
    oy = np.array(oy)
    ox += np.random.rand(len(ox)) * 1.0
    oy += np.random.rand(len(ox)) * 1.0
    return (ox, oy)

def main():
    if False:
        return 10
    print(__file__ + ' start!!')
    (ox, oy) = create_dummy_observation_data()
    grid_resolution = 10.0
    ndt_map = NDTMap(ox, oy, grid_resolution)
    plt.plot(ox, oy, '.r')
    [plt.plot(ox[inds], oy[inds], 'x') for inds in ndt_map.grid_index_map.values()]
    [plot_covariance_ellipse(ndt.mean_x, ndt.mean_y, ndt.covariance, color='-k') for ndt in ndt_map.grid_map.data if ndt.n_points > 0]
    plt.axis('equal')
    plt.show()
if __name__ == '__main__':
    main()