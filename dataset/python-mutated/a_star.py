"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""
import math
import matplotlib.pyplot as plt
show_animation = True

class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        if False:
            print('Hello World!')
        '\n        Initialize grid map for a star planning\n\n        ox: x position list of Obstacles [m]\n        oy: y position list of Obstacles [m]\n        resolution: grid resolution [m]\n        rr: robot radius[m]\n        '
        self.resolution = resolution
        self.rr = rr
        (self.min_x, self.min_y) = (0, 0)
        (self.max_x, self.max_y) = (0, 0)
        self.obstacle_map = None
        (self.x_width, self.y_width) = (0, 0)
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:

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

    def planning(self, sx, sy, gx, gy):
        if False:
            i = 10
            return i + 15
        '\n        A star path search\n\n        input:\n            s_x: start x position [m]\n            s_y: start y position [m]\n            gx: goal x position [m]\n            gy: goal y position [m]\n\n        output:\n            rx: x position list of the final path\n            ry: y position list of the final path\n        '
        start_node = self.Node(self.calc_xy_index(sx, self.min_x), self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x), self.calc_xy_index(gy, self.min_y), 0.0, -1)
        (open_set, closed_set) = (dict(), dict())
        open_set[self.calc_grid_index(start_node)] = start_node
        while True:
            if len(open_set) == 0:
                print('Open set is empty..')
                break
            c_id = min(open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))
            current = open_set[c_id]
            if show_animation:
                plt.plot(self.calc_grid_position(current.x, self.min_x), self.calc_grid_position(current.y, self.min_y), 'xc')
                plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)
            if current.x == goal_node.x and current.y == goal_node.y:
                print('Find goal')
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break
            del open_set[c_id]
            closed_set[c_id] = current
            for (i, _) in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0], current.y + self.motion[i][1], current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)
                if not self.verify_node(node):
                    continue
                if n_id in closed_set:
                    continue
                if n_id not in open_set:
                    open_set[n_id] = node
                elif open_set[n_id].cost > node.cost:
                    open_set[n_id] = node
        (rx, ry) = self.calc_final_path(goal_node, closed_set)
        return (rx, ry)

    def calc_final_path(self, goal_node, closed_set):
        if False:
            while True:
                i = 10
        (rx, ry) = ([self.calc_grid_position(goal_node.x, self.min_x)], [self.calc_grid_position(goal_node.y, self.min_y)])
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index
        return (rx, ry)

    @staticmethod
    def calc_heuristic(n1, n2):
        if False:
            i = 10
            return i + 15
        w = 1.0
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        if False:
            return 10
        '\n        calc grid position\n\n        :param index:\n        :param min_position:\n        :return:\n        '
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        if False:
            print('Hello World!')
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        if False:
            print('Hello World!')
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        if False:
            i = 10
            return i + 15
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)
        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False
        if self.obstacle_map[node.x][node.y]:
            return False
        return True

    def calc_obstacle_map(self, ox, oy):
        if False:
            i = 10
            return i + 15
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print('min_x:', self.min_x)
        print('min_y:', self.min_y)
        print('max_x:', self.max_x)
        print('max_y:', self.max_y)
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print('x_width:', self.x_width)
        print('y_width:', self.y_width)
        self.obstacle_map = [[False for _ in range(self.y_width)] for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for (iox, ioy) in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        if False:
            i = 10
            return i + 15
        motion = [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1], [-1, -1, math.sqrt(2)], [-1, 1, math.sqrt(2)], [1, -1, math.sqrt(2)], [1, 1, math.sqrt(2)]]
        return motion

def main():
    if False:
        while True:
            i = 10
    print(__file__ + ' start!!')
    sx = 10.0
    sy = 10.0
    gx = 50.0
    gy = 50.0
    grid_size = 2.0
    robot_radius = 1.0
    (ox, oy) = ([], [])
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)
    if show_animation:
        plt.plot(ox, oy, '.k')
        plt.plot(sx, sy, 'og')
        plt.plot(gx, gy, 'xb')
        plt.grid(True)
        plt.axis('equal')
    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    (rx, ry) = a_star.planning(sx, sy, gx, gy)
    if show_animation:
        plt.plot(rx, ry, '-r')
        plt.pause(0.001)
        plt.show()
if __name__ == '__main__':
    main()