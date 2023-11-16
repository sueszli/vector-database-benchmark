"""
Spiral Spanning Tree Coverage Path Planner

author: Todd Tang
paper: Spiral-STC: An On-Line Coverage Algorithm of Grid Environments
         by a Mobile Robot - Gabriely et.al.
link: https://ieeexplore.ieee.org/abstract/document/1013479
"""
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
do_animation = True

class SpiralSpanningTreeCoveragePlanner:

    def __init__(self, occ_map):
        if False:
            return 10
        self.origin_map_height = occ_map.shape[0]
        self.origin_map_width = occ_map.shape[1]
        if self.origin_map_height % 2 == 1 or self.origin_map_width % 2 == 1:
            sys.exit('original map width/height must be even                 in grayscale .png format')
        self.occ_map = occ_map
        self.merged_map_height = self.origin_map_height // 2
        self.merged_map_width = self.origin_map_width // 2
        self.edge = []

    def plan(self, start):
        if False:
            while True:
                i = 10
        'plan\n\n        performing Spiral Spanning Tree Coverage path planning\n\n        :param start: the start node of Spiral Spanning Tree Coverage\n        '
        visit_times = np.zeros((self.merged_map_height, self.merged_map_width), dtype=int)
        visit_times[start[0]][start[1]] = 1
        route = []
        self.perform_spanning_tree_coverage(start, visit_times, route)
        path = []
        for idx in range(len(route) - 1):
            dp = abs(route[idx][0] - route[idx + 1][0]) + abs(route[idx][1] - route[idx + 1][1])
            if dp == 0:
                path.append(self.get_round_trip_path(route[idx - 1], route[idx]))
            elif dp == 1:
                path.append(self.move(route[idx], route[idx + 1]))
            elif dp == 2:
                mid_node = self.get_intermediate_node(route[idx], route[idx + 1])
                path.append(self.move(route[idx], mid_node))
                path.append(self.move(mid_node, route[idx + 1]))
            else:
                sys.exit('adjacent path node distance larger than 2')
        return (self.edge, route, path)

    def perform_spanning_tree_coverage(self, current_node, visit_times, route):
        if False:
            for i in range(10):
                print('nop')
        'perform_spanning_tree_coverage\n\n        recursive function for function <plan>\n\n        :param current_node: current node\n        '

        def is_valid_node(i, j):
            if False:
                while True:
                    i = 10
            is_i_valid_bounded = 0 <= i < self.merged_map_height
            is_j_valid_bounded = 0 <= j < self.merged_map_width
            if is_i_valid_bounded and is_j_valid_bounded:
                return bool(self.occ_map[2 * i][2 * j] and self.occ_map[2 * i + 1][2 * j] and self.occ_map[2 * i][2 * j + 1] and self.occ_map[2 * i + 1][2 * j + 1])
            return False
        order = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        found = False
        route.append(current_node)
        for inc in order:
            (ni, nj) = (current_node[0] + inc[0], current_node[1] + inc[1])
            if is_valid_node(ni, nj) and visit_times[ni][nj] == 0:
                neighbor_node = (ni, nj)
                self.edge.append((current_node, neighbor_node))
                found = True
                visit_times[ni][nj] += 1
                self.perform_spanning_tree_coverage(neighbor_node, visit_times, route)
        if not found:
            has_node_with_unvisited_ngb = False
            for node in reversed(route):
                if visit_times[node[0]][node[1]] == 2:
                    continue
                visit_times[node[0]][node[1]] += 1
                route.append(node)
                for inc in order:
                    (ni, nj) = (node[0] + inc[0], node[1] + inc[1])
                    if is_valid_node(ni, nj) and visit_times[ni][nj] == 0:
                        has_node_with_unvisited_ngb = True
                        break
                if has_node_with_unvisited_ngb:
                    break
        return route

    def move(self, p, q):
        if False:
            print('Hello World!')
        direction = self.get_vector_direction(p, q)
        if direction == 'E':
            p = self.get_sub_node(p, 'SE')
            q = self.get_sub_node(q, 'SW')
        elif direction == 'W':
            p = self.get_sub_node(p, 'NW')
            q = self.get_sub_node(q, 'NE')
        elif direction == 'S':
            p = self.get_sub_node(p, 'SW')
            q = self.get_sub_node(q, 'NW')
        elif direction == 'N':
            p = self.get_sub_node(p, 'NE')
            q = self.get_sub_node(q, 'SE')
        else:
            sys.exit('move direction error...')
        return [p, q]

    def get_round_trip_path(self, last, pivot):
        if False:
            for i in range(10):
                print('nop')
        direction = self.get_vector_direction(last, pivot)
        if direction == 'E':
            return [self.get_sub_node(pivot, 'SE'), self.get_sub_node(pivot, 'NE')]
        elif direction == 'S':
            return [self.get_sub_node(pivot, 'SW'), self.get_sub_node(pivot, 'SE')]
        elif direction == 'W':
            return [self.get_sub_node(pivot, 'NW'), self.get_sub_node(pivot, 'SW')]
        elif direction == 'N':
            return [self.get_sub_node(pivot, 'NE'), self.get_sub_node(pivot, 'NW')]
        else:
            sys.exit('get_round_trip_path: last->pivot direction error.')

    def get_vector_direction(self, p, q):
        if False:
            i = 10
            return i + 15
        if p[0] == q[0] and p[1] < q[1]:
            return 'E'
        elif p[0] == q[0] and p[1] > q[1]:
            return 'W'
        elif p[0] < q[0] and p[1] == q[1]:
            return 'S'
        elif p[0] > q[0] and p[1] == q[1]:
            return 'N'
        else:
            sys.exit('get_vector_direction: Only E/W/S/N direction supported.')

    def get_sub_node(self, node, direction):
        if False:
            return 10
        if direction == 'SE':
            return [2 * node[0] + 1, 2 * node[1] + 1]
        elif direction == 'SW':
            return [2 * node[0] + 1, 2 * node[1]]
        elif direction == 'NE':
            return [2 * node[0], 2 * node[1] + 1]
        elif direction == 'NW':
            return [2 * node[0], 2 * node[1]]
        else:
            sys.exit('get_sub_node: sub-node direction error.')

    def get_interpolated_path(self, p, q):
        if False:
            return 10
        if (p[0] < q[0]) ^ (p[1] < q[1]):
            ipx = [p[0], p[0], q[0]]
            ipy = [p[1], q[1], q[1]]
        else:
            ipx = [p[0], q[0], q[0]]
            ipy = [p[1], p[1], q[1]]
        return (ipx, ipy)

    def get_intermediate_node(self, p, q):
        if False:
            print('Hello World!')
        (p_ngb, q_ngb) = (set(), set())
        for (m, n) in self.edge:
            if m == p:
                p_ngb.add(n)
            if n == p:
                p_ngb.add(m)
            if m == q:
                q_ngb.add(n)
            if n == q:
                q_ngb.add(m)
        itsc = p_ngb.intersection(q_ngb)
        if len(itsc) == 0:
            sys.exit('get_intermediate_node:                  no intermediate node between', p, q)
        elif len(itsc) == 1:
            return list(itsc)[0]
        else:
            sys.exit('get_intermediate_node:                 more than 1 intermediate node between', p, q)

    def visualize_path(self, edge, path, start):
        if False:
            i = 10
            return i + 15

        def coord_transform(p):
            if False:
                for i in range(10):
                    print('nop')
            return [2 * p[1] + 0.5, 2 * p[0] + 0.5]
        if do_animation:
            last = path[0][0]
            trajectory = [[last[1]], [last[0]]]
            for (p, q) in path:
                distance = math.hypot(p[0] - last[0], p[1] - last[1])
                if distance <= 1.0:
                    trajectory[0].append(p[1])
                    trajectory[1].append(p[0])
                else:
                    (ipx, ipy) = self.get_interpolated_path(last, p)
                    trajectory[0].extend(ipy)
                    trajectory[1].extend(ipx)
                last = q
            trajectory[0].append(last[1])
            trajectory[1].append(last[0])
            for (idx, state) in enumerate(np.transpose(trajectory)):
                plt.cla()
                plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
                plt.imshow(self.occ_map, 'gray')
                for (p, q) in edge:
                    p = coord_transform(p)
                    q = coord_transform(q)
                    plt.plot([p[0], q[0]], [p[1], q[1]], '-oc')
                (sx, sy) = coord_transform(start)
                plt.plot([sx], [sy], 'pr', markersize=10)
                plt.plot(trajectory[0][:idx + 1], trajectory[1][:idx + 1], '-k')
                plt.plot(state[0], state[1], 'or')
                plt.axis('equal')
                plt.grid(True)
                plt.pause(0.01)
        else:
            plt.imshow(self.occ_map, 'gray')
            for (p, q) in edge:
                p = coord_transform(p)
                q = coord_transform(q)
                plt.plot([p[0], q[0]], [p[1], q[1]], '-oc')
            (sx, sy) = coord_transform(start)
            plt.plot([sx], [sy], 'pr', markersize=10)
            last = path[0][0]
            for (p, q) in path:
                distance = math.hypot(p[0] - last[0], p[1] - last[1])
                if distance == 1.0:
                    plt.plot([last[1], p[1]], [last[0], p[0]], '-k')
                else:
                    (ipx, ipy) = self.get_interpolated_path(last, p)
                    plt.plot(ipy, ipx, '-k')
                plt.arrow(p[1], p[0], q[1] - p[1], q[0] - p[0], head_width=0.2)
                last = q
            plt.show()

def main():
    if False:
        while True:
            i = 10
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img = plt.imread(os.path.join(dir_path, 'map', 'test_2.png'))
    STC_planner = SpiralSpanningTreeCoveragePlanner(img)
    start = (10, 0)
    (edge, route, path) = STC_planner.plan(start)
    STC_planner.visualize_path(edge, path, start)
if __name__ == '__main__':
    main()