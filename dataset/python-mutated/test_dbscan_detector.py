import pytest
import numpy as np
from unittest import TestCase
from bigdl.chronos.detector.anomaly.dbscan_detector import DBScanDetector
from ... import op_torch, op_tf2

@op_torch
@op_tf2
class TestDBScanDetector(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def create_data(self):
        if False:
            return 10
        cycles = 5
        time = np.arange(0, cycles * np.pi, 0.2)
        data = np.sin(time)
        data[3] += 10
        data[5] -= 2
        data[10] += 5
        data[17] -= 3
        return data

    def test_dbscan_fit_score(self):
        if False:
            print('Hello World!')
        y = self.create_data()
        ad = DBScanDetector(eps=0.1, min_samples=6)
        ad.fit(y)
        anomaly_scores = ad.score()
        assert len(anomaly_scores) == len(y)
        anomaly_indexes = ad.anomaly_indexes()
        assert len(anomaly_indexes) >= 4

    def test_corner_cases(self):
        if False:
            for i in range(10):
                print('nop')
        ad = DBScanDetector(eps=0.1, min_samples=6)
        with pytest.raises(RuntimeError):
            ad.score()
        with pytest.raises(RuntimeError):
            ad.anomaly_indexes()
        y = self.create_data()
        y = y[:-1].reshape(2, -1)
        with pytest.raises(RuntimeError):
            ad.fit(y)