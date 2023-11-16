import pytest
from unittest import TestCase
from .. import op_diff_set_all
import numpy as np
import os
from numpy.testing import assert_array_almost_equal
import pandas as pd
from bigdl.chronos.utils import LazyImport
ARIMAModel = LazyImport('bigdl.chronos.model.arima.ARIMAModel')

@op_diff_set_all
class TestARIMAModel(TestCase):

    def setup_method(self, method):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(0)
        self.seq_len = 400
        self.config = {'p': np.random.randint(0, 4), 'q': np.random.randint(0, 4), 'seasonality_mode': np.random.choice([True, False]), 'P': 5, 'Q': 5, 'm': np.random.choice([4, 7]), 'metric': 'mse'}
        self.model = ARIMAModel()
        self.data = np.random.rand(self.seq_len)
        self.horizon = np.random.randint(2, 50)
        self.validation_data = np.random.rand(self.horizon)

    def teardown_method(self, method):
        if False:
            i = 10
            return i + 15
        del self.model
        del self.data
        del self.validation_data

    def test_arima(self):
        if False:
            print('Hello World!')
        evaluate_result = self.model.fit_eval(data=self.data, validation_data=self.validation_data, **self.config)
        result = self.model.predict(horizon=self.horizon)
        assert len(result) == self.horizon
        evaluate_result = self.model.evaluate(target=self.validation_data, metrics=['mae', 'smape'])
        assert len(evaluate_result) == 2
        rolling_result = self.model.predict(horizon=self.horizon, rolling=True)
        assert len(rolling_result) == self.horizon

    def test_error(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(RuntimeError, match='x should be None'):
            self.model.predict(x=1)
        with pytest.raises(RuntimeError, match="We don't support input x currently"):
            self.model.evaluate(target=self.validation_data, x=1)
        with pytest.raises(RuntimeError, match='Input invalid target of None'):
            self.model.evaluate(target=None)
        with pytest.raises(RuntimeError, match='Needs to call fit_eval or restore first before calling predict'):
            self.model.predict(horizon=self.horizon)
        with pytest.raises(RuntimeError, match="We don't support updating model without rolling prediction currently"):
            self.model.predict(horizon=self.horizon, update=True, rolling=False)
        with pytest.raises(RuntimeError, match='Needs to call fit_eval or restore first before calling evaluate'):
            self.model.evaluate(target=self.validation_data, x=None)
        with pytest.raises(RuntimeError, match='Needs to call fit_eval or restore first before calling save'):
            model_file = 'tmp.pkl'
            self.model.save(model_file)

    def test_save_restore(self):
        if False:
            i = 10
            return i + 15
        self.model.fit_eval(data=self.data, validation_data=self.validation_data, **self.config)
        result_save = self.model.predict(horizon=self.horizon, rolling=False)
        model_file = 'tmp.pkl'
        self.model.save(model_file)
        assert os.path.isfile(model_file)
        new_model = ARIMAModel()
        new_model.restore(model_file)
        assert new_model.model
        result_restore = new_model.predict(horizon=self.horizon, rolling=False)
        (assert_array_almost_equal(result_save, result_restore, decimal=2), 'Prediction values are not the same after restore: predict before is {}, and predict after is {}'.format(result_save, result_restore))
        os.remove(model_file)
if __name__ == '__main__':
    pytest.main([__file__])