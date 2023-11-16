"""
flowfield pathfinding
author: Sarim Mehdi (muhammadsarim.mehdi@studio.unibo.it)
Source: https://leifnode.com/2013/12/flow-field-pathfinding/
"""
import numpy as np
import matplotlib.pyplot as plt
show_animation = True

def draw_horizontal_line(start_x, start_y, length, o_x, o_y, o_dict, path):
    if False:
        return 10
    for i in range(start_x, start_x + length):
        for j in range(start_y, start_y + 2):
            o_x.append(i)
            o_y.append(j)
            o_dict[i, j] = path

def draw_vertical_line(start_x, start_y, length, o_x, o_y, o_dict, path):
    if False:
        return 10
    for i in range(start_x, start_x + 2):
        for j in range(start_y, start_y + length):
            o_x.append(i)
            o_y.append(j)
            o_dict[i, j] = path

class FlowField:

    def __init__(self, obs_grid, goal_x, goal_y, start_x, start_y, limit_x, limit_y):
        if False:
            while True:
                i = 10
        self.start_pt = [start_x, start_y]
        self.goal_pt = [goal_x, goal_y]
        self.obs_grid = obs_grid
        (self.limit_x, self.limit_y) = (limit_x, limit_y)
        self.cost_field = {}
        self.integration_field = {}
        self.vector_field = {}

    def find_path(self):
        if False:
            print('Hello World!')
        self.create_cost_field()
        self.create_integration_field()
        self.assign_vectors()
        self.follow_vectors()

    def create_cost_field(self):
        if False:
            return 10
        'Assign cost to each grid which defines the energy\n        it would take to get there.'
        for i in range(self.limit_x):
            for j in range(self.limit_y):
                if self.obs_grid[i, j] == 'free':
                    self.cost_field[i, j] = 1
                elif self.obs_grid[i, j] == 'medium':
                    self.cost_field[i, j] = 7
                elif self.obs_grid[i, j] == 'hard':
                    self.cost_field[i, j] = 20
                elif self.obs_grid[i, j] == 'obs':
                    continue
                if [i, j] == self.goal_pt:
                    self.cost_field[i, j] = 0

    def create_integration_field(self):
        if False:
            i = 10
            return i + 15
        'Start from the goal node and calculate the value\n        of the integration field at each node. Start by\n        assigning a value of infinity to every node except\n        the goal node which is assigned a value of 0. Put the\n        goal node in the open list and then get its neighbors\n        (must not be obstacles). For each neighbor, the new\n        cost is equal to the cost of the current node in the\n        integration field (in the beginning, this will simply\n        be the goal node) + the cost of the neighbor in the\n        cost field + the extra cost (optional). The new cost\n        is only assigned if it is less than the previously\n        assigned cost of the node in the integration field and,\n        when that happens, the neighbor is put on the open list.\n        This process continues until the open list is empty.'
        for i in range(self.limit_x):
            for j in range(self.limit_y):
                if self.obs_grid[i, j] == 'obs':
                    continue
                self.integration_field[i, j] = np.inf
                if [i, j] == self.goal_pt:
                    self.integration_field[i, j] = 0
        open_list = [(self.goal_pt, 0)]
        while open_list:
            (curr_pos, curr_cost) = open_list[0]
            (curr_x, curr_y) = curr_pos
            for i in range(-1, 2):
                for j in range(-1, 2):
                    (x, y) = (curr_x + i, curr_y + j)
                    if self.obs_grid[x, y] == 'obs':
                        continue
                    if (i, j) in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                        e_cost = 10
                    else:
                        e_cost = 14
                    neighbor_energy = self.cost_field[x, y]
                    neighbor_old_cost = self.integration_field[x, y]
                    neighbor_new_cost = curr_cost + neighbor_energy + e_cost
                    if neighbor_new_cost < neighbor_old_cost:
                        self.integration_field[x, y] = neighbor_new_cost
                        open_list.append(([x, y], neighbor_new_cost))
            del open_list[0]

    def assign_vectors(self):
        if False:
            print('Hello World!')
        'For each node, assign a vector from itself to the node with\n        the lowest cost in the integration field. An agent will simply\n        follow this vector field to the goal'
        for i in range(self.limit_x):
            for j in range(self.limit_y):
                if self.obs_grid[i, j] == 'obs':
                    continue
                if [i, j] == self.goal_pt:
                    self.vector_field[i, j] = (None, None)
                    continue
                offset_list = [(i + a, j + b) for a in range(-1, 2) for b in range(-1, 2)]
                neighbor_list = [{'loc': pt, 'cost': self.integration_field[pt]} for pt in offset_list if self.obs_grid[pt] != 'obs']
                neighbor_list = sorted(neighbor_list, key=lambda x: x['cost'])
                best_neighbor = neighbor_list[0]['loc']
                self.vector_field[i, j] = best_neighbor

    def follow_vectors(self):
        if False:
            for i in range(10):
                print('nop')
        (curr_x, curr_y) = self.start_pt
        while curr_x is not None and curr_y is not None:
            (curr_x, curr_y) = self.vector_field[curr_x, curr_y]
            if show_animation:
                plt.plot(curr_x, curr_y, 'b*')
                plt.pause(0.001)
        if show_animation:
            plt.show()

def main():
    if False:
        i = 10
        return i + 15
    obs_dict = {}
    for i in range(51):
        for j in range(51):
            obs_dict[i, j] = 'free'
    (o_x, o_y, m_x, m_y, h_x, h_y) = ([], [], [], [], [], [])
    s_x = 5.0
    s_y = 5.0
    g_x = 35.0
    g_y = 45.0
    draw_vertical_line(0, 0, 50, o_x, o_y, obs_dict, 'obs')
    draw_vertical_line(48, 0, 50, o_x, o_y, obs_dict, 'obs')
    draw_horizontal_line(0, 0, 50, o_x, o_y, obs_dict, 'obs')
    draw_horizontal_line(0, 48, 50, o_x, o_y, obs_dict, 'obs')
    all_x = [10, 10, 10, 15, 20, 20, 30, 30, 35, 30, 40, 45]
    all_y = [10, 30, 45, 20, 5, 40, 10, 40, 5, 40, 10, 25]
    all_len = [10, 10, 5, 10, 10, 5, 20, 10, 25, 10, 35, 15]
    for (x, y, l) in zip(all_x, all_y, all_len):
        draw_vertical_line(x, y, l, o_x, o_y, obs_dict, 'obs')
    (all_x[:], all_y[:], all_len[:]) = ([], [], [])
    all_x = [35, 40, 15, 10, 45, 20, 10, 15, 25, 45, 10, 30, 10, 40]
    all_y = [5, 10, 15, 20, 20, 25, 30, 35, 35, 35, 40, 40, 45, 45]
    all_len = [10, 5, 10, 10, 5, 5, 10, 5, 10, 5, 10, 5, 5, 5]
    for (x, y, l) in zip(all_x, all_y, all_len):
        draw_horizontal_line(x, y, l, o_x, o_y, obs_dict, 'obs')
    (all_x[:], all_y[:], all_len[:]) = ([], [], [])
    all_x = [10, 45]
    all_y = [22, 20]
    all_len = [8, 5]
    for (x, y, l) in zip(all_x, all_y, all_len):
        draw_vertical_line(x, y, l, m_x, m_y, obs_dict, 'medium')
    (all_x[:], all_y[:], all_len[:]) = ([], [], [])
    all_x = [20, 30, 42] + [47] * 5
    all_y = [35, 30, 38] + [37 + i for i in range(2)]
    all_len = [5, 7, 3] + [1] * 3
    for (x, y, l) in zip(all_x, all_y, all_len):
        draw_horizontal_line(x, y, l, m_x, m_y, obs_dict, 'medium')
    (all_x[:], all_y[:], all_len[:]) = ([], [], [])
    all_x = [15, 20, 35]
    all_y = [45, 20, 35]
    all_len = [3, 5, 7]
    for (x, y, l) in zip(all_x, all_y, all_len):
        draw_vertical_line(x, y, l, h_x, h_y, obs_dict, 'hard')
    (all_x[:], all_y[:], all_len[:]) = ([], [], [])
    all_x = [30] + [47] * 5
    all_y = [10] + [37 + i for i in range(2)]
    all_len = [5] + [1] * 3
    for (x, y, l) in zip(all_x, all_y, all_len):
        draw_horizontal_line(x, y, l, h_x, h_y, obs_dict, 'hard')
    if show_animation:
        plt.plot(o_x, o_y, 'sr')
        plt.plot(m_x, m_y, 'sg')
        plt.plot(h_x, h_y, 'sy')
        plt.plot(s_x, s_y, 'og')
        plt.plot(g_x, g_y, 'o')
        plt.grid(True)
    flow_obj = FlowField(obs_dict, g_x, g_y, s_x, s_y, 50, 50)
    flow_obj.find_path()
if __name__ == '__main__':
    main()