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
from max_flow import Edge, Node
import cvxpy as cp

class Directed(Edge):
    """ A directed, capacity limited edge """

    def constraints(self):
        if False:
            print('Hello World!')
        return [self.flow >= 0, self.flow <= self.capacity]

class LeakyDirected(Directed):
    """ A directed edge that leaks flow. """
    EFFICIENCY = 0.95

    def connect(self, in_node, out_node):
        if False:
            return 10
        in_node.edge_flows.append(-self.flow)
        out_node.edge_flows.append(self.EFFICIENCY * self.flow)

class LeakyUndirected(Edge):
    """ An undirected edge that leaks flow. """

    def __init__(self, capacity) -> None:
        if False:
            while True:
                i = 10
        self.forward = LeakyDirected(capacity)
        self.backward = LeakyDirected(capacity)

    def connect(self, in_node, out_node):
        if False:
            while True:
                i = 10
        self.forward.connect(in_node, out_node)
        self.backward.connect(out_node, in_node)

    def constraints(self):
        if False:
            print('Hello World!')
        return self.forward.constraints() + self.backward.constraints()
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
        edges.append(LeakyUndirected(capacity))
        edges[-1].connect(nodes[n1], nodes[n2])
    constraints = []
    for o in nodes + edges:
        constraints += o.constraints()
    p = cp.Problem(cp.Maximize(nodes[-1].accumulation), constraints)
    result = p.solve()
    print(result)