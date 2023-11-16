import os, sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.normpath(os.path.join(current_path, '../')))
from class_defs.key_maker import KeyMaker

class ParkingSpaceSet(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.data = dict()
        self.key_maker = KeyMaker('P')

    def append_data(self, parking_space, create_new_key=False):
        if False:
            while True:
                i = 10
        if create_new_key:
            idx = self.key_maker.get_new()
            while idx in self.data.keys():
                idx = self.key_maker.get_new()
            parking_space.idx = idx
        self.data[parking_space.idx] = parking_space

    def remove_data(self, parking_space):
        if False:
            return 10
        self.data.pop(parking_space.idx)

    def draw_plot(self, axes):
        if False:
            while True:
                i = 10
        for (idx, scw) in self.data.items():
            scw.draw_plot(axes)

    def erase_plot(self):
        if False:
            i = 10
            return i + 15
        for (idx, scw) in self.data.items():
            scw.erase_plot()