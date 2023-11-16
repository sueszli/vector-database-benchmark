import os, sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.normpath(os.path.join(current_path, '../')))

class ConnectingRoad(object):

    def __init__(self, _idx=None):
        if False:
            while True:
                i = 10
        self.idx = _idx
        self.connecting = None
        self.incoming = None
        self.from_lanes = list()
        self.to_lanes = list()

    def add_lanes(self, lane_id):
        if False:
            while True:
                i = 10
        self.from_lanes.append(lane_id)

    def get_lanes(self):
        if False:
            i = 10
            return i + 15
        return self.from_lanes