import numpy as np
import xarray as xr
from . import parameterized, randn, requires_dask
NDEGS = (2, 5, 20)
NX = (10 ** 2, 10 ** 6)

class Polyval:

    def setup(self, *args, **kwargs):
        if False:
            return 10
        self.xs = {nx: xr.DataArray(randn((nx,)), dims='x', name='x') for nx in NX}
        self.coeffs = {ndeg: xr.DataArray(randn((ndeg,)), dims='degree', coords={'degree': np.arange(ndeg)}) for ndeg in NDEGS}

    @parameterized(['nx', 'ndeg'], [NX, NDEGS])
    def time_polyval(self, nx, ndeg):
        if False:
            for i in range(10):
                print('nop')
        x = self.xs[nx]
        c = self.coeffs[ndeg]
        xr.polyval(x, c).compute()

    @parameterized(['nx', 'ndeg'], [NX, NDEGS])
    def peakmem_polyval(self, nx, ndeg):
        if False:
            while True:
                i = 10
        x = self.xs[nx]
        c = self.coeffs[ndeg]
        xr.polyval(x, c).compute()

class PolyvalDask(Polyval):

    def setup(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        requires_dask()
        super().setup(*args, **kwargs)
        self.xs = {k: v.chunk({'x': 10000}) for (k, v) in self.xs.items()}