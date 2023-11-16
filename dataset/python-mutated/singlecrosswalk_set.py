import os, sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.normpath(os.path.join(current_path, '../')))
from class_defs.key_maker import KeyMaker

class SingleCrosswalkSet:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.data = dict()
        self.key_maker = KeyMaker('CW')
        self.ref_crosswalk_id = ''

    def append_data(self, scw, create_new_key=False):
        if False:
            print('Hello World!')
        if create_new_key:
            idx = self.key_maker.get_new()
            while idx in self.data.keys():
                idx = self.key_maker.get_new()
            scw.idx = idx
        self.data[scw.idx] = scw

    def remove_data(self, scw):
        if False:
            return 10
        self.data.pop(scw.idx)

    def draw_plot(self, axes):
        if False:
            return 10
        for (idx, scw) in self.data.items():
            scw.draw_plot(axes)

    def erase_plot(self):
        if False:
            return 10
        for (idx, scw) in self.data.items():
            scw.erase_plot()

    def get_singlecrosswalk_contain_crosswalkid(self, cw_id):
        if False:
            i = 10
            return i + 15
        scw_list = []
        for (idx, scw) in self.data.items():
            if scw.ref_crosswalk_id == cw_id:
                scw_list.append(idx)
        return scw_list