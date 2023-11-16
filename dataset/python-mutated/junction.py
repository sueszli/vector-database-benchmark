import os, sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.normpath(os.path.join(current_path, '../')))
from collections import OrderedDict

class Junction(object):

    def __init__(self, _id=None):
        if False:
            return 10
        self.idx = _id
        self.jc_nodes = list()
        self.connecting_road = dict()

    def add_jc_node(self, node):
        if False:
            while True:
                i = 10
        if node not in self.jc_nodes:
            self.jc_nodes.append(node)
        if self not in node.junctions:
            node.junctions.append(self)

    def remove_jc_node(self, node):
        if False:
            print('Hello World!')
        self.jc_nodes.remove(node)

    def get_jc_nodes(self):
        if False:
            while True:
                i = 10
        return self.jc_nodes

    def get_jc_node_points(self):
        if False:
            i = 10
            return i + 15
        pts = []
        for node in self.jc_nodes:
            pts.append(node.point)
        return pts

    def get_jc_node_indices(self):
        if False:
            for i in range(10):
                print('nop')
        indices = []
        for node in self.jc_nodes:
            indices.append(node.idx)
        return indices

    def item_prop(self):
        if False:
            for i in range(10):
                print('nop')
        prop_data = OrderedDict()
        prop_data['idx'] = {'type': 'string', 'value': self.idx}
        prop_data['jc nodes id'] = {'type': 'list<string>', 'value': self.get_jc_node_indices()}
        return prop_data