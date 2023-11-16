import pytest
import pandas as pd
import numpy as np
from unittest import TestCase
from bigdl.chronos.data.utils.cycle_detection import cycle_length_est
from ... import op_torch, op_tf2, op_diff_set_all

class TestCycleDetectionTimeSeries(TestCase):

    def setup_method(self, method):
        if False:
            return 10
        pass

    def teardown_method(self, method):
        if False:
            print('Hello World!')
        pass

    @op_torch
    @op_tf2
    @op_diff_set_all
    def test_cycle_detection_timeseries_numpy(self):
        if False:
            for i in range(10):
                print('nop')
        data = np.random.randn(100)
        cycle_length = cycle_length_est(data)
        assert 1 <= cycle_length <= 100