"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import pickle
from create_graph import EDGES_KEY, FILE, NODE_COUNT_KEY
import cvxpy as cp

class Edge:
    """ An undirected, capacity limited edge. """

    def __init__(self, capacity) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.capacity = capacity
        self.flow = cp.Variable()

    def connect(self, in_node, out_node):
        if False:
            print('Hello World!')
        in_node.edge_flows.append(-self.flow)
        out_node.edge_flows.append(self.flow)

    def constraints(self):
        if False:
            while True:
                i = 10
        return [cp.abs(self.flow) <= self.capacity]

class Node:
    """ A node with accumulation. """

    def __init__(self, accumulation: float=0.0) -> None:
        if False:
            i = 10
            return i + 15
        self.accumulation = accumulation
        self.edge_flows = []

    def constraints(self):
        if False:
            while True:
                i = 10
        return [cp.sum([f for f in self.edge_flows]) == self.accumulation]
if __name__ == '__main__':
    f = open(FILE, 'rb')
    data = pickle.load(f)
    f.close()
    node_count = data[NODE_COUNT_KEY]
    nodes = [Node() for i in range(node_count)]
    nodes[0].accumulation = cp.Variable()
    nodes[-1].accumulation = cp.Variable()
    edges = []
    for (n1, n2, capacity) in data[EDGES_KEY]:
        edges.append(Edge(capacity))
        edges[-1].connect(nodes[n1], nodes[n2])
    constraints = []
    for o in nodes + edges:
        constraints += o.constraints()
    p = cp.Problem(cp.Maximize(nodes[-1].accumulation), constraints)
    result = p.solve()
    print(result)