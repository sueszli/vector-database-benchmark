import os, sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.normpath(os.path.join(current_path, '../')))
from class_defs.key_maker import KeyMaker

class RoadPolygonSet:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.data = dict()
        self.key_maker = KeyMaker('RP')

    def append_data(self, RoadPoly, create_new_key=False):
        if False:
            for i in range(10):
                print('nop')
        if create_new_key:
            idx = self.key_maker.get_new()
            for idx in self.data.keys():
                idx = self.key_maker.get_new()
            RoadPoly.idx = idx
        self.data[RoadPoly.idx] = RoadPoly

    def remove_data(self, Poly):
        if False:
            i = 10
            return i + 15
        self.data.pop(Poly.idx)
    '\n    def draw_plot(self, axes):\n        for idx, Poly in self.data.items():\n            Poly.draw_plot(axes)\n\n    def erase_plot(self):\n        for idx, Poly in self.data.items():\n            Poly.erase_plot()\n    '