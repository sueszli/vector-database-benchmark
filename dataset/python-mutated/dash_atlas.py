from functools import lru_cache
import numpy as np

class DashAtlas(object):
    """  """

    def __init__(self, shape=(64, 1024, 4)):
        if False:
            for i in range(10):
                print('nop')
        self._data = np.zeros(shape, dtype=np.float32)
        self._index = 0
        self._atlas = {}
        self['solid'] = ((1e+20, 0), (1, 1))
        self['densely dotted'] = ((0, 1), (1, 1))
        self['dotted'] = ((0, 2), (1, 1))
        self['loosely dotted'] = ((0, 3), (1, 1))
        self['densely dashed'] = ((1, 1), (1, 1))
        self['dashed'] = ((1, 2), (1, 1))
        self['loosely dashed'] = ((1, 4), (1, 1))
        self['densely dashdotted'] = ((1, 1, 0, 1), (1, 1, 1, 1))
        self['dashdotted'] = ((1, 2, 0, 2), (1, 1, 1, 1))
        self['loosely dashdotted'] = ((1, 3, 0, 3), (1, 1, 1, 1))
        self['densely dashdotdotted'] = ((1, 1, 0, 1, 0, 1), (1, 1, 1, 1))
        self['dashdotdotted'] = ((1, 2, 0, 2, 0, 2), (1, 1, 1, 1, 1, 1))
        self['loosely dashdotdotted'] = ((1, 3, 0, 3, 0, 3), (1, 1, 1, 1))
        self._dirty = True

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        return self._atlas[key]

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        (data, period) = self.make_pattern(value[0], value[1])
        self._data[self._index] = data
        self._atlas[key] = [self._index / float(self._data.shape[0]), period]
        self._index += 1
        self._dirty = True

    def make_pattern(self, pattern, caps=(1, 1)):
        if False:
            print('Hello World!')
        length = self._data.shape[1]
        return _make_pattern(length, pattern, caps)

@lru_cache(maxsize=32)
def _make_pattern(length, pattern, caps):
    if False:
        print('Hello World!')
    'Make a concrete dash pattern of a given length.'
    if len(pattern) > 1 and len(pattern) % 2:
        pattern = [pattern[0] + pattern[-1]] + pattern[1:-1]
    P = np.array(pattern)
    period = np.cumsum(P)[-1]
    (C, c) = ([], 0)
    for i in range(0, len(P) + 2, 2):
        a = max(0.0001, P[i % len(P)])
        b = max(0.0001, P[(i + 1) % len(P)])
        C.extend([c, c + a])
        c += a + b
    C = np.array(C)
    Z = np.zeros((length, 4), dtype=np.float32)
    for i in np.arange(0, len(Z)):
        x = period * i / float(len(Z) - 1)
        index = np.argmin(abs(C - x))
        if index % 2 == 0:
            if x <= C[index]:
                dash_type = +1
            else:
                dash_type = 0
            (dash_start, dash_end) = (C[index], C[index + 1])
        else:
            if x > C[index]:
                dash_type = -1
            else:
                dash_type = 0
            (dash_start, dash_end) = (C[index - 1], C[index])
        Z[i] = (C[index], dash_type, dash_start, dash_end)
    return (Z, period)