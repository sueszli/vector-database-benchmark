import pandas as pd

class Finalize:
    param_names = ['series', 'frame']
    params = [pd.Series, pd.DataFrame]

    def setup(self, param):
        if False:
            print('Hello World!')
        N = 1000
        obj = param(dtype=float)
        for i in range(N):
            obj.attrs[i] = i
        self.obj = obj

    def time_finalize_micro(self, param):
        if False:
            while True:
                i = 10
        self.obj.__finalize__(self.obj, method='__finalize__')