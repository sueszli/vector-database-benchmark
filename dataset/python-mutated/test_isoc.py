from . import util
import numpy as np

class TestISOC(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'isocintrin', 'isoCtests.f90')]

    def test_c_double(self):
        if False:
            return 10
        out = self.module.coddity.c_add(1, 2)
        exp_out = 3
        assert out == exp_out

    def test_bindc_function(self):
        if False:
            for i in range(10):
                print('nop')
        out = self.module.coddity.wat(1, 20)
        exp_out = 8
        assert out == exp_out