"""

Bidirectional A* grid planning

author: Erwin Lejeune (@spida_rwin)

See Wikipedia article (https://en.wikipedia.org/wiki/Bidirectional_search)

"""
import math
import matplotlib.pyplot as plt
show_animation = True

class BidirectionalAStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        if False:
            i = 10
            return i + 15
        '\n        Initialize grid map for a star planning\n\n        ox: x position list of Obstacles [m]\n        oy: y position list of Obstacles [m]\n        resolution: grid resolution [m]\n        rr: robot radius[m]\n        '
        (self.min_x, self.min_y) = (None, None)
        (self.max_x, self.max_y) = (None, None)
        (self.x_width, self.y_width, self.obstacle_map) = (None, None, None)
        self.resolution = resolution
        self.rr = rr
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()

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
                while True:
                    i = 10
            return str(self.x) + ',' + str(self.y) + ',' + str(self.cost) + ',' + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        if False:
            return 10
        '\n        Bidirectional A star path search\n\n        input:\n            s_x: start x position [m]\n            s_y: start y position [m]\n            gx: goal x position [m]\n            gy: goal y position [m]\n\n        output:\n            rx: x position list of the final path\n            ry: y position list of the final path\n        '
        start_node = self.Node(self.calc_xy_index(sx, self.min_x), self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x), self.calc_xy_index(gy, self.min_y), 0.0, -1)
        (open_set_A, closed_set_A) = (dict(), dict())
        (open_set_B, closed_set_B) = (dict(), dict())
        open_set_A[self.calc_grid_index(start_node)] = start_node
        open_set_B[self.calc_grid_index(goal_node)] = goal_node
        current_A = start_node
        current_B = goal_node
        (meet_point_A, meet_point_B) = (None, None)
        while True:
            if len(open_set_A) == 0:
                print('Open set A is empty..')
                break
            if len(open_set_B) == 0:
                print('Open set B is empty..')
                break
            c_id_A = min(open_set_A, key=lambda o: self.find_total_cost(open_set_A, o, current_B))
            current_A = open_set_A[c_id_A]
            c_id_B = min(open_set_B, key=lambda o: self.find_total_cost(open_set_B, o, current_A))
            current_B = open_set_B[c_id_B]
            if show_animation:
                plt.plot(self.calc_grid_position(current_A.x, self.min_x), self.calc_grid_position(current_A.y, self.min_y), 'xc')
                plt.plot(self.calc_grid_position(current_B.x, self.min_x), self.calc_grid_position(current_B.y, self.min_y), 'xc')
                plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
                if len(closed_set_A.keys()) % 10 == 0:
                    plt.pause(0.001)
            if current_A.x == current_B.x and current_A.y == current_B.y:
                print('Found goal')
                meet_point_A = current_A
                meet_point_B = current_B
                break
            del open_set_A[c_id_A]
            del open_set_B[c_id_B]
            closed_set_A[c_id_A] = current_A
            closed_set_B[c_id_B] = current_B
            for (i, _) in enumerate(self.motion):
                c_nodes = [self.Node(current_A.x + self.motion[i][0], current_A.y + self.motion[i][1], current_A.cost + self.motion[i][2], c_id_A), self.Node(current_B.x + self.motion[i][0], current_B.y + self.motion[i][1], current_B.cost + self.motion[i][2], c_id_B)]
                n_ids = [self.calc_grid_index(c_nodes[0]), self.calc_grid_index(c_nodes[1])]
                continue_ = self.check_nodes_and_sets(c_nodes, closed_set_A, closed_set_B, n_ids)
                if not continue_[0]:
                    if n_ids[0] not in open_set_A:
                        open_set_A[n_ids[0]] = c_nodes[0]
                    elif open_set_A[n_ids[0]].cost > c_nodes[0].cost:
                        open_set_A[n_ids[0]] = c_nodes[0]
                if not continue_[1]:
                    if n_ids[1] not in open_set_B:
                        open_set_B[n_ids[1]] = c_nodes[1]
                    elif open_set_B[n_ids[1]].cost > c_nodes[1].cost:
                        open_set_B[n_ids[1]] = c_nodes[1]
        (rx, ry) = self.calc_final_bidirectional_path(meet_point_A, meet_point_B, closed_set_A, closed_set_B)
        return (rx, ry)

    def calc_final_bidirectional_path(self, n1, n2, setA, setB):
        if False:
            i = 10
            return i + 15
        (rx_A, ry_A) = self.calc_final_path(n1, setA)
        (rx_B, ry_B) = self.calc_final_path(n2, setB)
        rx_A.reverse()
        ry_A.reverse()
        rx = rx_A + rx_B
        ry = ry_A + ry_B
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

    def check_nodes_and_sets(self, c_nodes, closedSet_A, closedSet_B, n_ids):
        if False:
            return 10
        continue_ = [False, False]
        if not self.verify_node(c_nodes[0]) or n_ids[0] in closedSet_A:
            continue_[0] = True
        if not self.verify_node(c_nodes[1]) or n_ids[1] in closedSet_B:
            continue_[1] = True
        return continue_

    @staticmethod
    def calc_heuristic(n1, n2):
        if False:
            i = 10
            return i + 15
        w = 1.0
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def find_total_cost(self, open_set, lambda_, n1):
        if False:
            print('Hello World!')
        g_cost = open_set[lambda_].cost
        h_cost = self.calc_heuristic(n1, open_set[lambda_])
        f_cost = g_cost + h_cost
        return f_cost

    def calc_grid_position(self, index, min_position):
        if False:
            return 10
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
            i = 10
            return i + 15
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
            return 10
        motion = [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1], [-1, -1, math.sqrt(2)], [-1, 1, math.sqrt(2)], [1, -1, math.sqrt(2)], [1, 1, math.sqrt(2)]]
        return motion

def main():
    if False:
        for i in range(10):
            print('nop')
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
    bidir_a_star = BidirectionalAStarPlanner(ox, oy, grid_size, robot_radius)
    (rx, ry) = bidir_a_star.planning(sx, sy, gx, gy)
    if show_animation:
        plt.plot(rx, ry, '-r')
        plt.pause(0.0001)
        plt.show()
if __name__ == '__main__':
    main()