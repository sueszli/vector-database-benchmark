"""

Bidirectional Breadth-First grid planning

author: Erwin Lejeune (@spida_rwin)

See Wikipedia article (https://en.wikipedia.org/wiki/Breadth-first_search)

"""
import math
import matplotlib.pyplot as plt
show_animation = True

class BidirectionalBreadthFirstSearchPlanner:

    def __init__(self, ox, oy, resolution, rr):
        if False:
            i = 10
            return i + 15
        '\n        Initialize grid map for bfs planning\n\n        ox: x position list of Obstacles [m]\n        oy: y position list of Obstacles [m]\n        resolution: grid resolution [m]\n        rr: robot radius[m]\n        '
        (self.min_x, self.min_y) = (None, None)
        (self.max_x, self.max_y) = (None, None)
        (self.x_width, self.y_width, self.obstacle_map) = (None, None, None)
        self.resolution = resolution
        self.rr = rr
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()

    class Node:

        def __init__(self, x, y, cost, parent_index, parent):
            if False:
                return 10
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_index = parent_index
            self.parent = parent

        def __str__(self):
            if False:
                for i in range(10):
                    print('nop')
            return str(self.x) + ',' + str(self.y) + ',' + str(self.cost) + ',' + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        if False:
            return 10
        '\n        Bidirectional Breadth First search based planning\n\n        input:\n            s_x: start x position [m]\n            s_y: start y position [m]\n            gx: goal x position [m]\n            gy: goal y position [m]\n\n        output:\n            rx: x position list of the final path\n            ry: y position list of the final path\n        '
        start_node = self.Node(self.calc_xy_index(sx, self.min_x), self.calc_xy_index(sy, self.min_y), 0.0, -1, None)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x), self.calc_xy_index(gy, self.min_y), 0.0, -1, None)
        (open_set_A, closed_set_A) = (dict(), dict())
        (open_set_B, closed_set_B) = (dict(), dict())
        open_set_B[self.calc_grid_index(goal_node)] = goal_node
        open_set_A[self.calc_grid_index(start_node)] = start_node
        (meet_point_A, meet_point_B) = (None, None)
        while True:
            if len(open_set_A) == 0:
                print('Open set A is empty..')
                break
            if len(open_set_B) == 0:
                print('Open set B is empty')
                break
            current_A = open_set_A.pop(list(open_set_A.keys())[0])
            current_B = open_set_B.pop(list(open_set_B.keys())[0])
            c_id_A = self.calc_grid_index(current_A)
            c_id_B = self.calc_grid_index(current_B)
            closed_set_A[c_id_A] = current_A
            closed_set_B[c_id_B] = current_B
            if show_animation:
                plt.plot(self.calc_grid_position(current_A.x, self.min_x), self.calc_grid_position(current_A.y, self.min_y), 'xc')
                plt.plot(self.calc_grid_position(current_B.x, self.min_x), self.calc_grid_position(current_B.y, self.min_y), 'xc')
                plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
                if len(closed_set_A.keys()) % 10 == 0:
                    plt.pause(0.001)
            if c_id_A in closed_set_B:
                print('Find goal')
                meet_point_A = closed_set_A[c_id_A]
                meet_point_B = closed_set_B[c_id_A]
                break
            elif c_id_B in closed_set_A:
                print('Find goal')
                meet_point_A = closed_set_A[c_id_B]
                meet_point_B = closed_set_B[c_id_B]
                break
            for (i, _) in enumerate(self.motion):
                breakA = False
                breakB = False
                node_A = self.Node(current_A.x + self.motion[i][0], current_A.y + self.motion[i][1], current_A.cost + self.motion[i][2], c_id_A, None)
                node_B = self.Node(current_B.x + self.motion[i][0], current_B.y + self.motion[i][1], current_B.cost + self.motion[i][2], c_id_B, None)
                n_id_A = self.calc_grid_index(node_A)
                n_id_B = self.calc_grid_index(node_B)
                if not self.verify_node(node_A):
                    breakA = True
                if not self.verify_node(node_B):
                    breakB = True
                if n_id_A not in closed_set_A and n_id_A not in open_set_A and (not breakA):
                    node_A.parent = current_A
                    open_set_A[n_id_A] = node_A
                if n_id_B not in closed_set_B and n_id_B not in open_set_B and (not breakB):
                    node_B.parent = current_B
                    open_set_B[n_id_B] = node_B
        (rx, ry) = self.calc_final_path_bidir(meet_point_A, meet_point_B, closed_set_A, closed_set_B)
        return (rx, ry)

    def calc_final_path_bidir(self, n1, n2, setA, setB):
        if False:
            i = 10
            return i + 15
        (rxA, ryA) = self.calc_final_path(n1, setA)
        (rxB, ryB) = self.calc_final_path(n2, setB)
        rxA.reverse()
        ryA.reverse()
        rx = rxA + rxB
        ry = ryA + ryB
        return (rx, ry)

    def calc_final_path(self, goal_node, closed_set):
        if False:
            i = 10
            return i + 15
        (rx, ry) = ([self.calc_grid_position(goal_node.x, self.min_x)], [self.calc_grid_position(goal_node.y, self.min_y)])
        n = closed_set[goal_node.parent_index]
        while n is not None:
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            n = n.parent
        return (rx, ry)

    def calc_grid_position(self, index, min_position):
        if False:
            print('Hello World!')
        '\n        calc grid position\n\n        :param index:\n        :param min_position:\n        :return:\n        '
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        if False:
            i = 10
            return i + 15
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        if False:
            for i in range(10):
                print('nop')
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        if False:
            print('Hello World!')
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
            print('Hello World!')
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
            for i in range(10):
                print('nop')
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
        plt.plot(gx, gy, 'ob')
        plt.grid(True)
        plt.axis('equal')
    bi_bfs = BidirectionalBreadthFirstSearchPlanner(ox, oy, grid_size, robot_radius)
    (rx, ry) = bi_bfs.planning(sx, sy, gx, gy)
    if show_animation:
        plt.plot(rx, ry, '-r')
        plt.pause(0.01)
        plt.show()
if __name__ == '__main__':
    main()