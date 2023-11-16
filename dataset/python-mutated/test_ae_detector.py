import pytest
import numpy as np
from unittest import TestCase
from bigdl.chronos.detector.anomaly.ae_detector import AEDetector
from ... import op_torch, op_tf2

class TestAEDetector(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def tearDown(self):
        if False:
            print('Hello World!')
        pass

    def create_data(self):
        if False:
            print('Hello World!')
        cycles = 10
        time = np.arange(0, cycles * np.pi, 0.01)
        data = np.sin(time)
        data[600:800] = 10
        return data

    @op_tf2
    def test_ae_fit_score_rolled_keras(self):
        if False:
            return 10
        y = self.create_data()
        ad = AEDetector(roll_len=314)
        ad.fit(y)
        anomaly_scores = ad.score()
        assert len(anomaly_scores) == len(y)
        anomaly_indexes = ad.anomaly_indexes()
        assert len(anomaly_indexes) == int(ad.ratio * len(y))

    @op_torch
    def test_ae_fit_score_rolled_pytorch(self):
        if False:
            while True:
                i = 10
        y = self.create_data()
        ad = AEDetector(roll_len=314, backend='torch')
        ad.fit(y)
        anomaly_scores = ad.score()
        assert len(anomaly_scores) == len(y)
        anomaly_indexes = ad.anomaly_indexes()
        assert len(anomaly_indexes) == int(ad.ratio * len(y))

    @op_tf2
    def test_ae_fit_score_unrolled(self):
        if False:
            return 10
        y = self.create_data()
        ad = AEDetector(roll_len=0)
        ad.fit(y)
        anomaly_scores = ad.score()
        assert len(anomaly_scores) == len(y)
        anomaly_indexes = ad.anomaly_indexes()
        assert len(anomaly_indexes) == int(ad.ratio * len(y))

    @op_torch
    @op_tf2
    def test_corner_cases(self):
        if False:
            print('Hello World!')
        y = self.create_data()
        ad = AEDetector(roll_len=314, backend='dummy')
        with pytest.raises(RuntimeError):
            ad.fit(y)
        ad = AEDetector(roll_len=314)
        with pytest.raises(RuntimeError):
            ad.score()
        y = np.array([1])
        with pytest.raises(RuntimeError):
            ad.fit(y)
        y = self.create_data()
        y = y.reshape(2, -1)
        with pytest.raises(RuntimeError):
            ad.fit(y)