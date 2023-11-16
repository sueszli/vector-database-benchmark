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
import random as r
from create_graph import FILE, NODE_COUNT_KEY, EDGES_KEY
import numpy as np
from max_flow import Edge, Node
import cvxpy as cp
COMMODITIES = 5
r.seed(1)

class MultiEdge(Edge):
    """ An undirected, capacity limited edge with multiple commodities. """

    def __init__(self, capacity) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.capacity = capacity
        self.in_flow = cp.Variable(COMMODITIES)
        self.out_flow = cp.Variable(COMMODITIES)

    def connect(self, in_node, out_node):
        if False:
            for i in range(10):
                print('nop')
        in_node.edge_flows.append(-self.in_flow)
        out_node.edge_flows.append(self.out_flow)

    def constraints(self):
        if False:
            for i in range(10):
                print('nop')
        return [self.in_flow + self.out_flow == 0, cp.sum(cp.abs(self.in_flow)) <= self.capacity]

class MultiNode(Node):
    """ A node with a target flow accumulation and a capacity. """

    def __init__(self, capacity: float=0.0) -> None:
        if False:
            return 10
        self.capacity = np.array(capacity)
        self.edge_flows = []

    def accumulation(self):
        if False:
            while True:
                i = 10
        return cp.sum([f for f in self.edge_flows])

    def constraints(self):
        if False:
            return 10
        return [cp.abs(self.accumulation()) <= self.capacity.flatten()]
if __name__ == '__main__':
    f = open(FILE, 'rb')
    data = pickle.load(f)
    f.close()
    node_count = data[NODE_COUNT_KEY]
    nodes = [MultiNode() for i in range(node_count)]
    sources = []
    sinks = []
    for i in range(COMMODITIES):
        (source, sink) = r.sample(nodes, 2)
        commodity_vec = cp.Parameter((COMMODITIES, 1))
        commodity_vec.value = np.zeros((COMMODITIES, 1))
        commodity_vec.value[i] = 1
        source.capacity = commodity_vec * cp.Variable()
        sources.append(source)
        sink.capacity = commodity_vec * cp.Variable()
        sinks.append(sink)
    edges = []
    for (n1, n2, capacity) in data[EDGES_KEY]:
        edges.append(MultiEdge(capacity))
        edges[-1].connect(nodes[n1], nodes[n2])
    objective = cp.Maximize(cp.sum(sum((s.accumulation() for s in sinks))))
    constraints = []
    for o in nodes + edges:
        constraints += o.constraints()
    p = cp.Problem(objective, constraints)
    result = p.solve()
    print('Objective value = %s' % result)
    for (i, s) in enumerate(sinks):
        accumulation = sum((f.value[i] for f in s.edge_flows))
        print('Accumulation of commodity %s = %s' % (i, accumulation))