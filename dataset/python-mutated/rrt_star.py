"""

Path planning Sample Code with RRT*

author: Atsushi Sakai(@Atsushi_twi)

"""
import math
import sys
import matplotlib.pyplot as plt
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from RRT.rrt import RRT
show_animation = True

class RRTStar(RRT):
    """
    Class for RRT Star planning
    """

    class Node(RRT.Node):

        def __init__(self, x, y):
            if False:
                return 10
            super().__init__(x, y)
            self.cost = 0.0

    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=30.0, path_resolution=1.0, goal_sample_rate=20, max_iter=300, connect_circle_dist=50.0, search_until_max_iter=False, robot_radius=0.0):
        if False:
            return 10
        '\n        Setting Parameter\n\n        start:Start Position [x,y]\n        goal:Goal Position [x,y]\n        obstacleList:obstacle Positions [[x,y,size],...]\n        randArea:Random Sampling Area [min,max]\n\n        '
        super().__init__(start, goal, obstacle_list, rand_area, expand_dis, path_resolution, goal_sample_rate, max_iter, robot_radius=robot_radius)
        self.connect_circle_dist = connect_circle_dist
        self.goal_node = self.Node(goal[0], goal[1])
        self.search_until_max_iter = search_until_max_iter
        self.node_list = []

    def planning(self, animation=True):
        if False:
            i = 10
            return i + 15
        '\n        rrt star path planning\n\n        animation: flag for animation on or off .\n        '
        self.node_list = [self.start]
        for i in range(self.max_iter):
            print('Iter:', i, ', number of nodes:', len(self.node_list))
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd, self.expand_dis)
            near_node = self.node_list[nearest_ind]
            new_node.cost = near_node.cost + math.hypot(new_node.x - near_node.x, new_node.y - near_node.y)
            if self.check_collision(new_node, self.obstacle_list, self.robot_radius):
                near_inds = self.find_near_nodes(new_node)
                node_with_updated_parent = self.choose_parent(new_node, near_inds)
                if node_with_updated_parent:
                    self.rewire(node_with_updated_parent, near_inds)
                    self.node_list.append(node_with_updated_parent)
                else:
                    self.node_list.append(new_node)
            if animation:
                self.draw_graph(rnd)
            if not self.search_until_max_iter and new_node:
                last_index = self.search_best_goal_node()
                if last_index is not None:
                    return self.generate_final_course(last_index)
        print('reached max iteration')
        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.generate_final_course(last_index)
        return None

    def choose_parent(self, new_node, near_inds):
        if False:
            for i in range(10):
                print('nop')
        '\n        Computes the cheapest point to new_node contained in the list\n        near_inds and set such a node as the parent of new_node.\n            Arguments:\n            --------\n                new_node, Node\n                    randomly generated node with a path from its neared point\n                    There are not coalitions between this node and th tree.\n                near_inds: list\n                    Indices of indices of the nodes what are near to new_node\n\n            Returns.\n            ------\n                Node, a copy of new_node\n        '
        if not near_inds:
            return None
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(t_node, self.obstacle_list, self.robot_radius):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float('inf'))
        min_cost = min(costs)
        if min_cost == float('inf'):
            print('There is no good path.(min_cost is inf)')
            return None
        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost
        return new_node

    def search_best_goal_node(self):
        if False:
            print('Hello World!')
        dist_to_goal_list = [self.calc_dist_to_goal(n.x, n.y) for n in self.node_list]
        goal_inds = [dist_to_goal_list.index(i) for i in dist_to_goal_list if i <= self.expand_dis]
        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node)
            if self.check_collision(t_node, self.obstacle_list, self.robot_radius):
                safe_goal_inds.append(goal_ind)
        if not safe_goal_inds:
            return None
        safe_goal_costs = [self.node_list[i].cost + self.calc_dist_to_goal(self.node_list[i].x, self.node_list[i].y) for i in safe_goal_inds]
        min_cost = min(safe_goal_costs)
        for (i, cost) in zip(safe_goal_inds, safe_goal_costs):
            if cost == min_cost:
                return i
        return None

    def find_near_nodes(self, new_node):
        if False:
            for i in range(10):
                print('nop')
        '\n        1) defines a ball centered on new_node\n        2) Returns all nodes of the three that are inside this ball\n            Arguments:\n            ---------\n                new_node: Node\n                    new randomly generated node, without collisions between\n                    its nearest node\n            Returns:\n            -------\n                list\n                    List with the indices of the nodes inside the ball of\n                    radius r\n        '
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt(math.log(nnode) / nnode)
        if hasattr(self, 'expand_dis'):
            r = min(r, self.expand_dis)
        dist_list = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r ** 2]
        return near_inds

    def rewire(self, new_node, near_inds):
        if False:
            for i in range(10):
                print('nop')
        '\n            For each node in near_inds, this will check if it is cheaper to\n            arrive to them from new_node.\n            In such a case, this will re-assign the parent of the nodes in\n            near_inds to new_node.\n            Parameters:\n            ----------\n                new_node, Node\n                    Node randomly added which can be joined to the tree\n\n                near_inds, list of uints\n                    A list of indices of the self.new_node which contains\n                    nodes within a circle of a given radius.\n            Remark: parent is designated in choose_parent.\n\n        '
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)
            no_collision = self.check_collision(edge_node, self.obstacle_list, self.robot_radius)
            improved_cost = near_node.cost > edge_node.cost
            if no_collision and improved_cost:
                for node in self.node_list:
                    if node.parent == self.node_list[i]:
                        node.parent = edge_node
                self.node_list[i] = edge_node
                self.propagate_cost_to_leaves(self.node_list[i])

    def calc_new_cost(self, from_node, to_node):
        if False:
            print('Hello World!')
        (d, _) = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):
        if False:
            for i in range(10):
                print('nop')
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

def main():
    if False:
        i = 10
        return i + 15
    print('Start ' + __file__)
    obstacle_list = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2), (9, 5, 2), (8, 10, 1), (6, 12, 1)]
    rrt_star = RRTStar(start=[0, 0], goal=[6, 10], rand_area=[-2, 15], obstacle_list=obstacle_list, expand_dis=1, robot_radius=0.8)
    path = rrt_star.planning(animation=show_animation)
    if path is None:
        print('Cannot find path')
    else:
        print('found path!!')
        if show_animation:
            rrt_star.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], 'r--')
            plt.grid(True)
            plt.show()
if __name__ == '__main__':
    main()