from __future__ import annotations
import heapq
import itertools
import math
import random
from river import base

class Vertex(base.Base):
    _isolated: set[Vertex] = set()

    def __init__(self, item, uuid: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.item = item
        self.uuid = uuid
        self.edges: dict[Vertex, float] = {}
        self.r_edges: dict[Vertex, float] = {}
        self.flags: set[Vertex] = set()
        self.worst_edge: Vertex | None = None

    def __hash__(self) -> int:
        if False:
            while True:
                i = 10
        return self.uuid

    def __eq__(self, other) -> bool:
        if False:
            while True:
                i = 10
        if not isinstance(other, Vertex):
            raise NotImplementedError
        return self.uuid == other.uuid

    def __lt__(self, other) -> bool:
        if False:
            i = 10
            return i + 15
        if not isinstance(other, Vertex):
            raise NotImplementedError
        return self.uuid < other.uuid

    def farewell(self):
        if False:
            for i in range(10):
                print('nop')
        for rn in list(self.r_edges):
            rn.rem_edge(self)
        for n in list(self.edges):
            self.rem_edge(n)
        self.flags = None
        self.worst_edge = None
        Vertex._isolated.discard(self)

    def fill(self, neighbors: list[Vertex], dists: list[float]):
        if False:
            while True:
                i = 10
        for (n, dist) in zip(neighbors, dists):
            self.edges[n] = dist
            self.flags.add(n)
            n.r_edges[self] = dist
        self.worst_edge = n

    def add_edge(self, vertex: Vertex, dist):
        if False:
            for i in range(10):
                print('nop')
        self.edges[vertex] = dist
        self.flags.add(vertex)
        vertex.r_edges[self] = dist
        if self.worst_edge is None or self.edges[self.worst_edge] < dist:
            self.worst_edge = vertex

    def rem_edge(self, vertex: Vertex):
        if False:
            return 10
        self.edges.pop(vertex)
        vertex.r_edges.pop(self)
        self.flags.discard(vertex)
        if self.has_neighbors():
            if vertex == self.worst_edge:
                self.worst_edge = max(self.edges, key=self.edges.get)
        else:
            self.worst_edge = None
            if not self.has_rneighbors():
                Vertex._isolated.add(self)

    def push_edge(self, node: Vertex, dist: float, max_edges: int) -> int:
        if False:
            while True:
                i = 10
        if self.is_neighbor(node) or node == self:
            return 0
        if len(self.edges) >= max_edges:
            if self.worst_edge is None or self.edges.get(self.worst_edge, math.inf) <= dist:
                return 0
            self.rem_edge(self.worst_edge)
        self.add_edge(node, dist)
        return 1

    def is_neighbor(self, vertex):
        if False:
            print('Hello World!')
        return vertex in self.edges or vertex in self.r_edges

    def get_edge(self, vertex: Vertex):
        if False:
            i = 10
            return i + 15
        if vertex in self.edges:
            return (self, vertex, self.edges[vertex])
        return (vertex, self, self.r_edges[vertex])

    def has_neighbors(self) -> bool:
        if False:
            while True:
                i = 10
        return len(self.edges) > 0

    def has_rneighbors(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return len(self.r_edges) > 0

    @property
    def sample_flags(self):
        if False:
            print('Hello World!')
        return list(map(lambda n: n in self.flags, self.edges.keys()))

    @sample_flags.setter
    def sample_flags(self, sampled):
        if False:
            print('Hello World!')
        self.flags -= set(sampled)

    def neighbors(self) -> tuple[list[Vertex], list[float]]:
        if False:
            i = 10
            return i + 15
        res = tuple(map(list, zip(*((node, dist) for (node, dist) in self.edges.items()))))
        return res if len(res) > 0 else ([], [])

    def r_neighbors(self) -> tuple[list[Vertex], list[float]]:
        if False:
            i = 10
            return i + 15
        res = tuple(map(list, zip(*((vertex, dist) for (vertex, dist) in self.r_edges.items()))))
        return res if len(res) > 0 else ([], [])

    def all_neighbors(self):
        if False:
            while True:
                i = 10
        return set.union(set(self.edges.keys()), set(self.r_edges.keys()))

    def is_isolated(self):
        if False:
            while True:
                i = 10
        return len(self.edges) == 0 and len(self.r_edges) == 0

    def prune(self, prune_prob: float, prune_trigger: int, rng: random.Random):
        if False:
            while True:
                i = 10
        if prune_prob == 0:
            return
        total_degree = len(self.edges) + len(self.r_edges)
        if total_degree <= prune_trigger:
            return
        counter = itertools.count()
        edge_pool: list[tuple[float, int, Vertex, bool]] = []
        for (n, dist) in self.edges.items():
            heapq.heappush(edge_pool, (dist, next(counter), n, True))
        for (rn, dist) in self.r_edges.items():
            heapq.heappush(edge_pool, (dist, next(counter), rn, False))
        selected: list[Vertex] = [heapq.heappop(edge_pool)[2]]
        while len(edge_pool) > 0:
            (c_dist, _, c, c_isdir) = heapq.heappop(edge_pool)
            discarded = False
            for s in selected:
                if s.is_neighbor(c) and rng.random() < prune_prob:
                    (orig, dest, dist) = s.get_edge(c)
                    if dist < c_dist:
                        if c_isdir:
                            self.rem_edge(c)
                        else:
                            c.rem_edge(self)
                        discarded = True
                        break
                    else:
                        orig.rem_edge(dest)
            if not discarded:
                selected.append(c)