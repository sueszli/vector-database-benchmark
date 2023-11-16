import sys
import os
import jittor as jt
import unittest
import time
import numpy as np
from .test_reorder_tuner import simple_parser
from .test_log import find_log_with_re

class TestBroadcastTuner(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if False:
            for i in range(10):
                print('nop')
        return

    def check(self, h, w, cs, rs, pa, rtp, dim):
        if False:
            while True:
                i = 10
        a = jt.random([h, w])
        a.data
        with jt.log_capture_scope(log_v=0, log_vprefix='tuner_manager=100', compile_options={'test_broadcast_tuner': 1}) as logs:
            amean = jt.mean(a, dims=[dim], keepdims=1)
            a2mean = jt.mean(a * a, dims=[dim], keepdims=1)
            norm_aa = (a - amean.broadcast_var(a)) / jt.sqrt(a2mean - amean * amean).broadcast_var(a)
            norm_aa.data
        logs = find_log_with_re(logs, 'Run tuner broadcast: confidence\\((20)\\) candidates\\((.*)\\)$')
        assert len(logs) == 1, logs
        assert logs[0][0] == '20', 'confidence of reorder should be 20'
        candidates = simple_parser(logs[0][1])
        assert candidates == {'order0': [0], 'order1': [1], 'order2': [0], 'split1': [2048], 'use_movnt': [1]}, candidates

    def test_broadcast_tuner(self):
        if False:
            i = 10
            return i + 15
        self.check(8192, 8192, 0, 0, 0, 5, 0)
if __name__ == '__main__':
    unittest.main()