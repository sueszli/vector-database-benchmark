"""

A* grid based planning

author: Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""
import heapq
import math
import matplotlib.pyplot as plt
show_animation = False

class Node:

    def __init__(self, x, y, cost, parent_index):
        if False:
            i = 10
            return i + 15
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return str(self.x) + ',' + str(self.y) + ',' + str(self.cost) + ',' + str(self.parent_index)

def calc_final_path(goal_node, closed_node_set, resolution):
    if False:
        i = 10
        return i + 15
    (rx, ry) = ([goal_node.x * resolution], [goal_node.y * resolution])
    parent_index = goal_node.parent_index
    while parent_index != -1:
        n = closed_node_set[parent_index]
        rx.append(n.x * resolution)
        ry.append(n.y * resolution)
        parent_index = n.parent_index
    return (rx, ry)

def calc_distance_heuristic(gx, gy, ox, oy, resolution, rr):
    if False:
        while True:
            i = 10
    '\n    gx: goal x position [m]\n    gx: goal x position [m]\n    ox: x position list of Obstacles [m]\n    oy: y position list of Obstacles [m]\n    resolution: grid resolution [m]\n    rr: robot radius[m]\n    '
    goal_node = Node(round(gx / resolution), round(gy / resolution), 0.0, -1)
    ox = [iox / resolution for iox in ox]
    oy = [ioy / resolution for ioy in oy]
    (obstacle_map, min_x, min_y, max_x, max_y, x_w, y_w) = calc_obstacle_map(ox, oy, resolution, rr)
    motion = get_motion_model()
    (open_set, closed_set) = (dict(), dict())
    open_set[calc_index(goal_node, x_w, min_x, min_y)] = goal_node
    priority_queue = [(0, calc_index(goal_node, x_w, min_x, min_y))]
    while True:
        if not priority_queue:
            break
        (cost, c_id) = heapq.heappop(priority_queue)
        if c_id in open_set:
            current = open_set[c_id]
            closed_set[c_id] = current
            open_set.pop(c_id)
        else:
            continue
        if show_animation:
            plt.plot(current.x * resolution, current.y * resolution, 'xc')
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            if len(closed_set.keys()) % 10 == 0:
                plt.pause(0.001)
        for (i, _) in enumerate(motion):
            node = Node(current.x + motion[i][0], current.y + motion[i][1], current.cost + motion[i][2], c_id)
            n_id = calc_index(node, x_w, min_x, min_y)
            if n_id in closed_set:
                continue
            if not verify_node(node, obstacle_map, min_x, min_y, max_x, max_y):
                continue
            if n_id not in open_set:
                open_set[n_id] = node
                heapq.heappush(priority_queue, (node.cost, calc_index(node, x_w, min_x, min_y)))
            elif open_set[n_id].cost >= node.cost:
                open_set[n_id] = node
                heapq.heappush(priority_queue, (node.cost, calc_index(node, x_w, min_x, min_y)))
    return closed_set

def verify_node(node, obstacle_map, min_x, min_y, max_x, max_y):
    if False:
        while True:
            i = 10
    if node.x < min_x:
        return False
    elif node.y < min_y:
        return False
    elif node.x >= max_x:
        return False
    elif node.y >= max_y:
        return False
    if obstacle_map[node.x][node.y]:
        return False
    return True

def calc_obstacle_map(ox, oy, resolution, vr):
    if False:
        for i in range(10):
            print('nop')
    min_x = round(min(ox))
    min_y = round(min(oy))
    max_x = round(max(ox))
    max_y = round(max(oy))
    x_width = round(max_x - min_x)
    y_width = round(max_y - min_y)
    obstacle_map = [[False for _ in range(y_width)] for _ in range(x_width)]
    for ix in range(x_width):
        x = ix + min_x
        for iy in range(y_width):
            y = iy + min_y
            for (iox, ioy) in zip(ox, oy):
                d = math.hypot(iox - x, ioy - y)
                if d <= vr / resolution:
                    obstacle_map[ix][iy] = True
                    break
    return (obstacle_map, min_x, min_y, max_x, max_y, x_width, y_width)

def calc_index(node, x_width, x_min, y_min):
    if False:
        i = 10
        return i + 15
    return (node.y - y_min) * x_width + (node.x - x_min)

def get_motion_model():
    if False:
        return 10
    motion = [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1], [-1, -1, math.sqrt(2)], [-1, 1, math.sqrt(2)], [1, -1, math.sqrt(2)], [1, 1, math.sqrt(2)]]
    return motion