import pandas as pd
from h2o.estimators.xgboost import *
from tests import pyunit_utils
import unittest
import sys

class TestXGBoostUnicode(unittest.TestCase):

    @unittest.skipIf(sys.version_info[0] < 3, "not tested on 2.7: csv.writer fails on 'ascii' codec can't encode character")
    def test_xgboost_unicode(self):
        if False:
            for i in range(10):
                print('nop')
        assert H2OXGBoostEstimator.available()
        unic_df = pd.DataFrame({u'¥': [2, 3, 1], 'y': [0, 0, 1], 'x': [0.3, 0.1, 0.9]})
        h2o_unic = h2o.H2OFrame(unic_df, destination_frame='unic_df')
        xg1 = H2OXGBoostEstimator(model_id='xg1', ntrees=3)
        xg1.train(x=[u'¥', 'x'], y='y', training_frame=h2o_unic)
suite = unittest.TestLoader().loadTestsFromTestCase(TestXGBoostUnicode)
unittest.TextTestRunner().run(suite)