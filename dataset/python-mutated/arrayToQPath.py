import numpy as np
import pyqtgraph as pg
rng = np.random.default_rng(12345)

class _TimeSuite:
    params = ([10000, 100000, 1000000], ['all', 'finite', 'pairs', 'array'])

    def setup(self, nelems, connect):
        if False:
            return 10
        self.xdata = np.arange(nelems, dtype=np.float64)
        self.ydata = rng.standard_normal(nelems, dtype=np.float64)
        if connect == 'array':
            self.connect_array = np.ones(nelems, dtype=bool)
        if self.have_nonfinite:
            self.ydata[::5000] = np.nan

    def time_test(self, nelems, connect):
        if False:
            for i in range(10):
                print('nop')
        if connect == 'array':
            connect = self.connect_array
        pg.arrayToQPath(self.xdata, self.ydata, connect=connect)

class TimeSuiteAllFinite(_TimeSuite):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.have_nonfinite = False

class TimeSuiteWithNonFinite(_TimeSuite):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.have_nonfinite = True