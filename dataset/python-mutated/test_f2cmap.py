from . import util
import numpy as np

class TestF2Cmap(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'f2cmap', 'isoFortranEnvMap.f90'), util.getpath('tests', 'src', 'f2cmap', '.f2py_f2cmap')]

    def test_long_long_map(self):
        if False:
            while True:
                i = 10
        inp = np.ones(3)
        out = self.module.func1(inp)
        exp_out = 3
        assert out == exp_out