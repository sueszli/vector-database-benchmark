"""

Probabilistic Road Map (PRM) Planner

author: Atsushi Sakai (@Atsushi_twi)

"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
N_SAMPLE = 500
N_KNN = 10
MAX_EDGE_LEN = 30.0
show_animation = True

class Node:
    """
    Node class for dijkstra search
    """

    def __init__(self, x, y, cost, parent_index):
        if False:
            return 10
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self.x) + ',' + str(self.y) + ',' + str(self.cost) + ',' + str(self.parent_index)

def prm_planning(start_x, start_y, goal_x, goal_y, obstacle_x_list, obstacle_y_list, robot_radius, *, rng=None):
    if False:
        i = 10
        return i + 15
    '\n    Run probabilistic road map planning\n\n    :param start_x: start x position\n    :param start_y: start y position\n    :param goal_x: goal x position\n    :param goal_y: goal y position\n    :param obstacle_x_list: obstacle x positions\n    :param obstacle_y_list: obstacle y positions\n    :param robot_radius: robot radius\n    :param rng: (Optional) Random generator\n    :return:\n    '
    obstacle_kd_tree = KDTree(np.vstack((obstacle_x_list, obstacle_y_list)).T)
    (sample_x, sample_y) = sample_points(start_x, start_y, goal_x, goal_y, robot_radius, obstacle_x_list, obstacle_y_list, obstacle_kd_tree, rng)
    if show_animation:
        plt.plot(sample_x, sample_y, '.b')
    road_map = generate_road_map(sample_x, sample_y, robot_radius, obstacle_kd_tree)
    (rx, ry) = dijkstra_planning(start_x, start_y, goal_x, goal_y, road_map, sample_x, sample_y)
    return (rx, ry)

def is_collision(sx, sy, gx, gy, rr, obstacle_kd_tree):
    if False:
        while True:
            i = 10
    x = sx
    y = sy
    dx = gx - sx
    dy = gy - sy
    yaw = math.atan2(gy - sy, gx - sx)
    d = math.hypot(dx, dy)
    if d >= MAX_EDGE_LEN:
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

def generate_road_map(sample_x, sample_y, rr, obstacle_kd_tree):
    if False:
        print('Hello World!')
    '\n    Road map generation\n\n    sample_x: [m] x positions of sampled points\n    sample_y: [m] y positions of sampled points\n    robot_radius: Robot Radius[m]\n    obstacle_kd_tree: KDTree object of obstacles\n    '
    road_map = []
    n_sample = len(sample_x)
    sample_kd_tree = KDTree(np.vstack((sample_x, sample_y)).T)
    for (i, ix, iy) in zip(range(n_sample), sample_x, sample_y):
        (dists, indexes) = sample_kd_tree.query([ix, iy], k=n_sample)
        edge_id = []
        for ii in range(1, len(indexes)):
            nx = sample_x[indexes[ii]]
            ny = sample_y[indexes[ii]]
            if not is_collision(ix, iy, nx, ny, rr, obstacle_kd_tree):
                edge_id.append(indexes[ii])
            if len(edge_id) >= N_KNN:
                break
        road_map.append(edge_id)
    return road_map

def dijkstra_planning(sx, sy, gx, gy, road_map, sample_x, sample_y):
    if False:
        i = 10
        return i + 15
    '\n    s_x: start x position [m]\n    s_y: start y position [m]\n    goal_x: goal x position [m]\n    goal_y: goal y position [m]\n    obstacle_x_list: x position list of Obstacles [m]\n    obstacle_y_list: y position list of Obstacles [m]\n    robot_radius: robot radius [m]\n    road_map: ??? [m]\n    sample_x: ??? [m]\n    sample_y: ??? [m]\n\n    @return: Two lists of path coordinates ([x1, x2, ...], [y1, y2, ...]), empty list when no path was found\n    '
    start_node = Node(sx, sy, 0.0, -1)
    goal_node = Node(gx, gy, 0.0, -1)
    (open_set, closed_set) = (dict(), dict())
    open_set[len(road_map) - 2] = start_node
    path_found = True
    while True:
        if not open_set:
            print('Cannot find path')
            path_found = False
            break
        c_id = min(open_set, key=lambda o: open_set[o].cost)
        current = open_set[c_id]
        if show_animation and len(closed_set.keys()) % 2 == 0:
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(current.x, current.y, 'xg')
            plt.pause(0.001)
        if c_id == len(road_map) - 1:
            print('goal is found!')
            goal_node.parent_index = current.parent_index
            goal_node.cost = current.cost
            break
        del open_set[c_id]
        closed_set[c_id] = current
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.hypot(dx, dy)
            node = Node(sample_x[n_id], sample_y[n_id], current.cost + d, c_id)
            if n_id in closed_set:
                continue
            if n_id in open_set:
                if open_set[n_id].cost > node.cost:
                    open_set[n_id].cost = node.cost
                    open_set[n_id].parent_index = c_id
            else:
                open_set[n_id] = node
    if path_found is False:
        return ([], [])
    (rx, ry) = ([goal_node.x], [goal_node.y])
    parent_index = goal_node.parent_index
    while parent_index != -1:
        n = closed_set[parent_index]
        rx.append(n.x)
        ry.append(n.y)
        parent_index = n.parent_index
    return (rx, ry)

def plot_road_map(road_map, sample_x, sample_y):
    if False:
        while True:
            i = 10
    for (i, _) in enumerate(road_map):
        for ii in range(len(road_map[i])):
            ind = road_map[i][ii]
            plt.plot([sample_x[i], sample_x[ind]], [sample_y[i], sample_y[ind]], '-k')

def sample_points(sx, sy, gx, gy, rr, ox, oy, obstacle_kd_tree, rng):
    if False:
        for i in range(10):
            print('nop')
    max_x = max(ox)
    max_y = max(oy)
    min_x = min(ox)
    min_y = min(oy)
    (sample_x, sample_y) = ([], [])
    if rng is None:
        rng = np.random.default_rng()
    while len(sample_x) <= N_SAMPLE:
        tx = rng.random() * (max_x - min_x) + min_x
        ty = rng.random() * (max_y - min_y) + min_y
        (dist, index) = obstacle_kd_tree.query([tx, ty])
        if dist >= rr:
            sample_x.append(tx)
            sample_y.append(ty)
    sample_x.append(sx)
    sample_y.append(sy)
    sample_x.append(gx)
    sample_y.append(gy)
    return (sample_x, sample_y)

def main(rng=None):
    if False:
        return 10
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
    (rx, ry) = prm_planning(sx, sy, gx, gy, ox, oy, robot_size, rng=rng)
    assert rx, 'Cannot found path'
    if show_animation:
        plt.plot(rx, ry, '-r')
        plt.pause(0.001)
        plt.show()
if __name__ == '__main__':
    main()