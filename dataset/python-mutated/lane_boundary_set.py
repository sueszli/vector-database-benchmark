import os, sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.normpath(os.path.join(current_path, '../')))
from class_defs.key_maker import KeyMaker

class LaneBoundarySet(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.lanes = dict()
        self.key_maker = KeyMaker(prefix='LM')

    def append_line(self, lane_obj, create_new_key=False):
        if False:
            while True:
                i = 10
        if create_new_key:
            idx = self.key_maker.get_new()
            while idx in self.lanes.keys():
                idx = self.key_maker.get_new()
            lane_obj.idx = idx
        self.lanes[lane_obj.idx] = lane_obj

    def draw_plot(self, axes):
        if False:
            i = 10
            return i + 15
        for (idx, lane) in self.lanes.items():
            lane.draw_plot(axes)

    def remove_line(self, line_obj):
        if False:
            return 10
        if line_obj.idx in self.lanes.keys():
            self.lanes.pop(line_obj.idx)