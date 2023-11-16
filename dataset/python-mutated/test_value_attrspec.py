import os
import pytest
from . import util

class TestValueAttr(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'value_attrspec', 'gh21665.f90')]

    def test_long_long_map(self):
        if False:
            while True:
                i = 10
        inp = 2
        out = self.module.fortfuncs.square(inp)
        exp_out = 4
        assert out == exp_out