"""

Voronoi Road Map Planner

author: Atsushi Sakai (@Atsushi_twi)

"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, Voronoi
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from VoronoiRoadMap.dijkstra_search import DijkstraSearch
show_animation = True

class VoronoiRoadMapPlanner:

    def __init__(self):
        if False:
            print('Hello World!')
        self.N_KNN = 10
        self.MAX_EDGE_LEN = 30.0

    def planning(self, sx, sy, gx, gy, ox, oy, robot_radius):
        if False:
            i = 10
            return i + 15
        obstacle_tree = cKDTree(np.vstack((ox, oy)).T)
        (sample_x, sample_y) = self.voronoi_sampling(sx, sy, gx, gy, ox, oy)
        if show_animation:
            plt.plot(sample_x, sample_y, '.b')
        road_map_info = self.generate_road_map_info(sample_x, sample_y, robot_radius, obstacle_tree)
        (rx, ry) = DijkstraSearch(show_animation).search(sx, sy, gx, gy, sample_x, sample_y, road_map_info)
        return (rx, ry)

    def is_collision(self, sx, sy, gx, gy, rr, obstacle_kd_tree):
        if False:
            return 10
        x = sx
        y = sy
        dx = gx - sx
        dy = gy - sy
        yaw = math.atan2(gy - sy, gx - sx)
        d = math.hypot(dx, dy)
        if d >= self.MAX_EDGE_LEN:
            return True
        D = rr
        n_step = round(d / D)
        for i in range(n_step):
            (dist, _) = obstacle_kd_tree.query([x, y])
            if dist <= rr:
                return True
            x += D * math.cos(yaw)
            y += D * math.sin(yaw)
        (dist, _) = obstacle_kd_tree.query([gx, gy])
        if dist <= rr:
            return True
        return False

    def generate_road_map_info(self, node_x, node_y, rr, obstacle_tree):
        if False:
            print('Hello World!')
        '\n        Road map generation\n\n        node_x: [m] x positions of sampled points\n        node_y: [m] y positions of sampled points\n        rr: Robot Radius[m]\n        obstacle_tree: KDTree object of obstacles\n        '
        road_map = []
        n_sample = len(node_x)
        node_tree = cKDTree(np.vstack((node_x, node_y)).T)
        for (i, ix, iy) in zip(range(n_sample), node_x, node_y):
            (dists, indexes) = node_tree.query([ix, iy], k=n_sample)
            edge_id = []
            for ii in range(1, len(indexes)):
                nx = node_x[indexes[ii]]
                ny = node_y[indexes[ii]]
                if not self.is_collision(ix, iy, nx, ny, rr, obstacle_tree):
                    edge_id.append(indexes[ii])
                if len(edge_id) >= self.N_KNN:
                    break
            road_map.append(edge_id)
        return road_map

    @staticmethod
    def plot_road_map(road_map, sample_x, sample_y):
        if False:
            print('Hello World!')
        for (i, _) in enumerate(road_map):
            for ii in range(len(road_map[i])):
                ind = road_map[i][ii]
                plt.plot([sample_x[i], sample_x[ind]], [sample_y[i], sample_y[ind]], '-k')

    @staticmethod
    def voronoi_sampling(sx, sy, gx, gy, ox, oy):
        if False:
            while True:
                i = 10
        oxy = np.vstack((ox, oy)).T
        vor = Voronoi(oxy)
        sample_x = [ix for [ix, _] in vor.vertices]
        sample_y = [iy for [_, iy] in vor.vertices]
        sample_x.append(sx)
        sample_y.append(sy)
        sample_x.append(gx)
        sample_y.append(gy)
        return (sample_x, sample_y)

def main():
    if False:
        while True:
            i = 10
    print(__file__ + ' start!!')
    sx = 10.0
    sy = 10.0
    gx = 50.0
    gy = 50.0
    robot_size = 5.0
    ox = []
    oy = []
    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - i)
    if show_animation:
        plt.plot(ox, oy, '.k')
        plt.plot(sx, sy, '^r')
        plt.plot(gx, gy, '^c')
        plt.grid(True)
        plt.axis('equal')
    (rx, ry) = VoronoiRoadMapPlanner().planning(sx, sy, gx, gy, ox, oy, robot_size)
    assert rx, 'Cannot found path'
    if show_animation:
        plt.plot(rx, ry, '-r')
        plt.pause(0.1)
        plt.show()
if __name__ == '__main__':
    main()