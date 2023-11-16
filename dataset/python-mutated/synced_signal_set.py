import os, sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.normpath(os.path.join(current_path, '../')))
from class_defs.key_maker import KeyMaker

class SyncedSignalSet(object):

    def __init__(self):
        if False:
            return 10
        self.synced_signals = dict()
        self.key_maker = KeyMaker('SSN')

    def append_synced_signal(self, synced_signal_obj, create_new_key=False):
        if False:
            print('Hello World!')
        if create_new_key:
            idx = self.key_maker.get_new()
            while idx in self.synced_signals.keys():
                idx = self.key_maker.get_new()
            synced_signal_obj.idx = idx
        self.synced_signals[synced_signal_obj.idx] = synced_signal_obj
        self.synced_signals = dict(sorted(self.synced_signals.items()))

    def remove_synced_signal(self, synced_signal_obj):
        if False:
            i = 10
            return i + 15
        self.synced_signals.pop(synced_signal_obj.idx)

    def get_signal_list(self):
        if False:
            while True:
                i = 10
        signal_list = []
        for synced_signal in self.synced_signals.values():
            signal_list = signal_list + synced_signal.signal_set.to_list()
        return signal_list

    def remove_data(self, ss):
        if False:
            i = 10
            return i + 15
        self.synced_signals.pop(ss)