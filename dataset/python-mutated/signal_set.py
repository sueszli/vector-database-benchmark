import os, sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.normpath(os.path.join(current_path, '../')))
from class_defs.key_maker import KeyMaker

class SignalSet(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.signals = dict()
        self.key_maker = KeyMaker('SN')

    def append_signal(self, signal_obj, create_new_key=False):
        if False:
            for i in range(10):
                print('nop')
        if create_new_key:
            idx = self.key_maker.get_new()
            while idx in self.signals.keys():
                idx = self.key_maker.get_new()
            signal_obj.idx = idx
        self.signals[signal_obj.idx] = signal_obj

    def remove_signal(self, signal_obj):
        if False:
            for i in range(10):
                print('nop')
        self.signals.pop(signal_obj.idx)

    def draw_plot(self, axes):
        if False:
            i = 10
            return i + 15
        for (idx, signal) in self.signals.items():
            signal.draw_plot(axes)

    def erase_plot(self):
        if False:
            i = 10
            return i + 15
        for (idx, signal) in self.signals.items():
            signal.erase_plot()

    def to_list(self):
        if False:
            print('Hello World!')
        signal_list = []
        for (key, val) in self.signals.items():
            signal_list.append(val)
        return signal_list

    def merge_signal_set(self, a_signals):
        if False:
            i = 10
            return i + 15
        for signal in a_signals:
            if signal not in self.signals.keys():
                self.signals[signal] = a_signals[signal]
        return self.signals

    def get_signal_contain_crosswalkid(self, cw_id):
        if False:
            return 10
        tl_list = []
        for (idx, signal) in self.signals.items():
            if signal.ref_crosswalk_id == cw_id:
                tl_list.append(idx)
        return tl_list