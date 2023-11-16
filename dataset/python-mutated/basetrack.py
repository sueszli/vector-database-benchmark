import numpy as np
from collections import OrderedDict

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class BaseTrack(object):
    _count = 0
    track_id = 0
    is_activated = False
    state = TrackState.New
    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        if False:
            print('Hello World!')
        return self.frame_id

    @staticmethod
    def next_id():
        if False:
            i = 10
            return i + 15
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def predict(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def update(self, *args, **kwargs):
        if False:
            return 10
        raise NotImplementedError

    def mark_lost(self):
        if False:
            return 10
        self.state = TrackState.Lost

    def mark_removed(self):
        if False:
            i = 10
            return i + 15
        self.state = TrackState.Removed