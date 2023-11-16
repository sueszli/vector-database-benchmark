from unittest.mock import Mock
from celery.utils.graph import DependencyGraph
from celery.utils.text import WhateverIO

class test_DependencyGraph:

    def graph1(self):
        if False:
            return 10
        res_a = self.app.AsyncResult('A')
        res_b = self.app.AsyncResult('B')
        res_c = self.app.GroupResult('C', [res_a])
        res_d = self.app.GroupResult('D', [res_c, res_b])
        node_a = (res_a, [])
        node_b = (res_b, [])
        node_c = (res_c, [res_a])
        node_d = (res_d, [res_c, res_b])
        return DependencyGraph([node_a, node_b, node_c, node_d])

    def test_repr(self):
        if False:
            while True:
                i = 10
        assert repr(self.graph1())

    def test_topsort(self):
        if False:
            i = 10
            return i + 15
        order = self.graph1().topsort()
        assert order.index('C') < order.index('D')
        assert order.index('B') < order.index('D')
        assert order.index('A') < order.index('C')

    def test_edges(self):
        if False:
            print('Hello World!')
        edges = self.graph1().edges()
        assert sorted(edges, key=str) == ['C', 'D']

    def test_connect(self):
        if False:
            i = 10
            return i + 15
        (x, y) = (self.graph1(), self.graph1())
        x.connect(y)

    def test_valency_of_when_missing(self):
        if False:
            for i in range(10):
                print('nop')
        x = self.graph1()
        assert x.valency_of('foobarbaz') == 0

    def test_format(self):
        if False:
            i = 10
            return i + 15
        x = self.graph1()
        x.formatter = Mock()
        obj = Mock()
        assert x.format(obj)
        x.formatter.assert_called_with(obj)
        x.formatter = None
        assert x.format(obj) is obj

    def test_items(self):
        if False:
            print('Hello World!')
        assert dict(self.graph1().items()) == {'A': [], 'B': [], 'C': ['A'], 'D': ['C', 'B']}

    def test_repr_node(self):
        if False:
            print('Hello World!')
        x = self.graph1()
        assert x.repr_node('fasdswewqewq')

    def test_to_dot(self):
        if False:
            return 10
        s = WhateverIO()
        self.graph1().to_dot(s)
        assert s.getvalue()