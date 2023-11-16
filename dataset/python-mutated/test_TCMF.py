import pytest
from unittest import TestCase
from .. import op_torch, op_distributed
import numpy as np
import os
from numpy.testing import assert_array_almost_equal
import pandas as pd
from bigdl.chronos.utils import LazyImport
TCMF = LazyImport('bigdl.chronos.model.tcmf_model.TCMF')

@op_torch
@op_distributed
class TestTCMF(TestCase):

    def setup_method(self, method):
        if False:
            return 10
        self.seq_len = 480
        self.num_samples = 300
        self.config = {'y_iters': 1, 'init_FX_epoch': 1, 'max_FX_epoch': 1, 'max_TCN_epoch': 1, 'alt_iters': 2}
        self.model = TCMF()
        self.Ymat = np.random.rand(self.num_samples, self.seq_len)
        self.horizon = np.random.randint(2, 50)

    def teardown_method(self, method):
        if False:
            for i in range(10):
                print('nop')
        del self.model
        del self.Ymat

    def test_tcmf(self):
        if False:
            return 10
        self.model.fit_eval(data=(self.Ymat, None), **self.config)
        result = self.model.predict(horizon=self.horizon)
        assert result.shape[1] == self.horizon
        target = np.random.rand(self.num_samples, self.horizon)
        evaluate_result = self.model.evaluate(y=target, metrics=['mae', 'smape'])
        assert len(evaluate_result) == 2
        assert len(evaluate_result[0]) == self.horizon
        assert len(evaluate_result[1]) == self.horizon
        Ymat_before = self.model.model.Ymat
        self.model.fit_incremental(target)
        Ymat_after = self.model.model.Ymat
        assert Ymat_after.shape[1] - Ymat_before.shape[1] == self.horizon
        incr_result = self.model.predict(horizon=self.horizon)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, result, incr_result)

    def test_tcmf_covariates_dti(self):
        if False:
            return 10
        with pytest.raises(RuntimeError, match='Input covariates must be a ndarray. Got'):
            self.model.fit_eval(data=(self.Ymat, None), covariates='None', **self.config)
        with pytest.raises(RuntimeError, match='The second dimension shape of covariates should be'):
            self.model.fit_eval(data=(self.Ymat, None), covariates=np.random.randn(3, self.seq_len - 1), **self.config)
        with pytest.raises(RuntimeError, match='You should input a 2-D ndarray of covariates.'):
            self.model.fit_eval(data=(self.Ymat, None), covariates=np.random.randn(3, 4, 5), **self.config)
        with pytest.raises(RuntimeError, match='Input dti must be a pandas DatetimeIndex. Got'):
            self.model.fit_eval(data=(self.Ymat, None), covariates=np.random.randn(3, self.seq_len), dti='None', **self.config)
        with pytest.raises(RuntimeError, match='Input dti length should be equal to'):
            self.model.fit_eval(data=(self.Ymat, None), covariates=np.random.randn(3, self.seq_len), dti=pd.date_range('20130101', periods=self.seq_len - 1), **self.config)
        self.model.fit_eval(data=(self.Ymat, None), covariates=np.random.rand(3, self.seq_len), dti=pd.date_range('20130101', periods=self.seq_len), **self.config)
        with pytest.raises(RuntimeError, match='Find valid covariates in fit but invalid covariates in predict.'):
            self.model.predict(horizon=self.horizon)
        with pytest.raises(RuntimeError, match='be the same as the input covariates number in fit.'):
            self.model.predict(horizon=self.horizon, future_covariates=np.random.randn(2, self.horizon))
        with pytest.raises(RuntimeError, match='Find valid dti in fit but invalid dti in'):
            self.model.predict(horizon=self.horizon, future_covariates=np.random.randn(3, self.horizon))
        with pytest.raises(RuntimeError, match='Find valid covariates in fit but invalid covariates in fit_incremental.'):
            self.model.fit_incremental(x=np.random.rand(self.num_samples, self.horizon))
        with pytest.raises(RuntimeError, match='be the same as the input covariates number in fit.'):
            self.model.fit_incremental(x=np.random.rand(self.num_samples, self.horizon), covariates_new=np.random.randn(2, self.horizon))
        with pytest.raises(RuntimeError, match='Find valid dti in fit but invalid dti in'):
            self.model.fit_incremental(x=np.random.rand(self.num_samples, self.horizon), covariates_new=np.random.randn(3, self.horizon))
        self.model.predict(horizon=self.horizon, future_covariates=np.random.randn(3, self.horizon), future_dti=pd.date_range('20130101', periods=self.horizon))
        self.model.fit_incremental(x=np.random.rand(self.num_samples, self.horizon), covariates_new=np.random.randn(3, self.horizon), dti_new=pd.date_range('20130101', periods=self.horizon))

    def test_error(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(RuntimeError, match="We don't support input x directly"):
            self.model.predict(x=1)
        with pytest.raises(RuntimeError, match="We don't support input x directly"):
            self.model.evaluate(x=1, y=np.random.rand(self.num_samples, self.horizon))
        with pytest.raises(RuntimeError, match='Input invalid y of None'):
            self.model.evaluate(y=None)
        with pytest.raises(Exception, match='Needs to call fit_eval or restore first before calling predict'):
            self.model.predict(x=None)
        with pytest.raises(Exception, match='Needs to call fit_eval or restore first before calling predict'):
            self.model.evaluate(y=np.random.rand(self.num_samples, self.horizon))
        with pytest.raises(RuntimeError, match='Input invalid x of None'):
            self.model.fit_incremental(x=None)
        with pytest.raises(Exception, match='Needs to call fit_eval or restore first before calling fit_incremental'):
            self.model.fit_incremental(x=np.random.rand(self.num_samples, self.horizon))
        self.model.fit_eval(data=(self.Ymat, None), **self.config)
        with pytest.raises(Exception, match=f'Expected incremental input with {self.num_samples} time series, got {self.num_samples - 1} instead'):
            self.model.fit_incremental(x=np.random.rand(self.num_samples - 1, self.horizon))
        with pytest.raises(RuntimeError, match='but invalid covariates in fit. '):
            self.model.predict(horizon=self.horizon, future_covariates=np.random.randn(3, self.horizon))
        with pytest.raises(RuntimeError, match='but invalid dti in fit. '):
            self.model.predict(horizon=self.horizon, future_dti=pd.date_range('20130101', periods=self.horizon))
        with pytest.raises(RuntimeError, match='but invalid covariates in fit. '):
            self.model.fit_incremental(x=np.random.rand(self.num_samples, self.horizon), covariates_new=np.random.randn(3, self.horizon))
        with pytest.raises(RuntimeError, match='but invalid dti in fit. '):
            self.model.fit_incremental(x=np.random.rand(self.num_samples, self.horizon), dti_new=pd.date_range('20130101', periods=self.horizon))

    def test_save_restore(self):
        if False:
            return 10
        self.model.fit_eval(data=(self.Ymat, None), **self.config)
        result_save = self.model.predict(horizon=self.horizon)
        model_file = 'tmp.pkl'
        self.model.save(model_file)
        assert os.path.isfile(model_file)
        new_model = TCMF()
        new_model.restore(model_file)
        assert new_model.model
        result_restore = new_model.predict(horizon=self.horizon)
        (assert_array_almost_equal(result_save, result_restore, decimal=2), 'Prediction values are not the same after restore: predict before is {}, and predict after is {}'.format(result_save, result_restore))
        os.remove(model_file)
if __name__ == '__main__':
    pytest.main([__file__])