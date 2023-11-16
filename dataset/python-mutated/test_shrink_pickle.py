"""

Created on Fri Mar 09 16:00:27 2012

Author: Josef Perktold
"""
from statsmodels.compat.pandas import assert_series_equal
from io import BytesIO
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
log = np.log

def check_pickle(obj):
    if False:
        return 10
    fh = BytesIO()
    pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    plen = fh.tell()
    fh.seek(0, 0)
    res = pickle.load(fh)
    fh.close()
    return (res, plen)

class RemoveDataPickle:

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        nobs = 1000
        np.random.seed(987689)
        x = np.random.randn(nobs, 3)
        x = sm.add_constant(x)
        cls.exog = x
        cls.xf = 0.25 * np.ones((2, 4))
        cls.predict_kwds = {}
        cls.reduction_factor = 0.1

    def test_remove_data_pickle(self):
        if False:
            return 10
        results = self.results
        xf = self.xf
        pred_kwds = self.predict_kwds
        pred1 = results.predict(xf, **pred_kwds)
        results.summary()
        results.summary2()
        (res, orig_nbytes) = check_pickle(results._results)
        results.remove_data()
        pred2 = results.predict(xf, **pred_kwds)
        if isinstance(pred1, pd.Series) and isinstance(pred2, pd.Series):
            assert_series_equal(pred1, pred2)
        elif isinstance(pred1, pd.DataFrame) and isinstance(pred2, pd.DataFrame):
            assert pred1.equals(pred2)
        else:
            np.testing.assert_equal(pred2, pred1)
        (res, nbytes) = check_pickle(results._results)
        self.res = res
        msg = 'pickle length not %d < %d' % (nbytes, orig_nbytes)
        assert nbytes < orig_nbytes * self.reduction_factor, msg
        pred3 = results.predict(xf, **pred_kwds)
        if isinstance(pred1, pd.Series) and isinstance(pred3, pd.Series):
            assert_series_equal(pred1, pred3)
        elif isinstance(pred1, pd.DataFrame) and isinstance(pred3, pd.DataFrame):
            assert pred1.equals(pred3)
        else:
            np.testing.assert_equal(pred3, pred1)

    def test_remove_data_docstring(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.results.remove_data.__doc__ is not None

    def test_pickle_wrapper(self):
        if False:
            for i in range(10):
                print('nop')
        fh = BytesIO()
        self.results._results.save(fh)
        fh.seek(0, 0)
        res_unpickled = self.results._results.__class__.load(fh)
        assert type(res_unpickled) is type(self.results._results)
        fh.seek(0, 0)
        self.results.save(fh)
        fh.seek(0, 0)
        res_unpickled = self.results.__class__.load(fh)
        fh.close()
        assert type(res_unpickled) is type(self.results)
        before = sorted(self.results.__dict__.keys())
        after = sorted(res_unpickled.__dict__.keys())
        assert before == after, 'not equal %r and %r' % (before, after)
        before = sorted(self.results._results.__dict__.keys())
        after = sorted(res_unpickled._results.__dict__.keys())
        assert before == after, 'not equal %r and %r' % (before, after)
        before = sorted(self.results.model.__dict__.keys())
        after = sorted(res_unpickled.model.__dict__.keys())
        assert before == after, 'not equal %r and %r' % (before, after)
        before = sorted(self.results._cache.keys())
        after = sorted(res_unpickled._cache.keys())
        assert before == after, 'not equal %r and %r' % (before, after)

class TestRemoveDataPickleOLS(RemoveDataPickle):

    def setup_method(self):
        if False:
            return 10
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = sm.OLS(y, self.exog).fit()

class TestRemoveDataPickleWLS(RemoveDataPickle):

    def setup_method(self):
        if False:
            print('Hello World!')
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = sm.WLS(y, self.exog, weights=np.ones(len(y))).fit()

class TestRemoveDataPicklePoisson(RemoveDataPickle):

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        x = self.exog
        np.random.seed(987689)
        y_count = np.random.poisson(np.exp(x.sum(1) - x.mean()))
        model = sm.Poisson(y_count, x)
        start_params = np.array([0.75334818, 0.99425553, 1.00494724, 1.00247112])
        self.results = model.fit(start_params=start_params, method='bfgs', disp=0)
        self.predict_kwds = dict(exposure=1, offset=0)

class TestRemoveDataPickleNegativeBinomial(RemoveDataPickle):

    def setup_method(self):
        if False:
            return 10
        np.random.seed(987689)
        data = sm.datasets.randhie.load()
        mod = sm.NegativeBinomial(data.endog, data.exog)
        self.results = mod.fit(disp=0)

class TestRemoveDataPickleLogit(RemoveDataPickle):

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        x = self.exog
        nobs = x.shape[0]
        np.random.seed(987689)
        y_bin = (np.random.rand(nobs) < 1.0 / (1 + np.exp(x.sum(1) - x.mean()))).astype(int)
        model = sm.Logit(y_bin, x)
        start_params = np.array([-0.73403806, -1.00901514, -0.97754543, -0.95648212])
        self.results = model.fit(start_params=start_params, method='bfgs', disp=0)

class TestRemoveDataPickleRLM(RemoveDataPickle):

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = sm.RLM(y, self.exog).fit()

class TestRemoveDataPickleGLM(RemoveDataPickle):

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = sm.GLM(y, self.exog).fit()

    def test_cached_data_removed(self):
        if False:
            while True:
                i = 10
        res = self.results
        names = ['resid_response', 'resid_deviance', 'resid_pearson', 'resid_anscombe']
        for name in names:
            getattr(res, name)
        for name in names:
            assert name in res._cache
            assert res._cache[name] is not None
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            res.remove_data()
        for name in names:
            assert res._cache[name] is None

    def test_cached_values_evaluated(self):
        if False:
            return 10
        res = self.results
        assert res._cache == {}
        res.remove_data()
        assert 'aic' in res._cache

class TestRemoveDataPickleGLMConstrained(RemoveDataPickle):

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = sm.GLM(y, self.exog).fit_constrained('x1=x2')

class TestPickleFormula(RemoveDataPickle):

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        super(TestPickleFormula, cls).setup_class()
        nobs = 10000
        np.random.seed(987689)
        x = np.random.randn(nobs, 3)
        cls.exog = pd.DataFrame(x, columns=['A', 'B', 'C'])
        cls.xf = pd.DataFrame(0.25 * np.ones((2, 3)), columns=cls.exog.columns)
        cls.reduction_factor = 0.5

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        x = self.exog
        np.random.seed(123)
        y = x.sum(1) + np.random.randn(x.shape[0])
        y = pd.Series(y, name='Y')
        X = self.exog.copy()
        X['Y'] = y
        self.results = sm.OLS.from_formula('Y ~ A + B + C', data=X).fit()

class TestPickleFormula2(RemoveDataPickle):

    @classmethod
    def setup_class(cls):
        if False:
            i = 10
            return i + 15
        super(TestPickleFormula2, cls).setup_class()
        nobs = 500
        np.random.seed(987689)
        data = np.random.randn(nobs, 4)
        data[:, 0] = data[:, 1:].sum(1)
        cls.data = pd.DataFrame(data, columns=['Y', 'A', 'B', 'C'])
        cls.xf = pd.DataFrame(0.25 * np.ones((2, 3)), columns=cls.data.columns[1:])
        cls.reduction_factor = 0.5

    def setup_method(self):
        if False:
            return 10
        self.results = sm.OLS.from_formula('Y ~ A + B + C', data=self.data).fit()

class TestPickleFormula3(TestPickleFormula2):

    def setup_method(self):
        if False:
            print('Hello World!')
        self.results = sm.OLS.from_formula('Y ~ A + B * C', data=self.data).fit()

class TestPickleFormula4(TestPickleFormula2):

    def setup_method(self):
        if False:
            print('Hello World!')
        self.results = sm.OLS.from_formula('Y ~ np.log(abs(A) + 1) + B * C', data=self.data).fit()

class TestPickleFormula5(TestPickleFormula2):

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.results = sm.OLS.from_formula('Y ~ log(abs(A) + 1) + B * C', data=self.data).fit()

class TestRemoveDataPicklePoissonRegularized(RemoveDataPickle):

    def setup_method(self):
        if False:
            while True:
                i = 10
        x = self.exog
        np.random.seed(987689)
        y_count = np.random.poisson(np.exp(x.sum(1) - x.mean()))
        model = sm.Poisson(y_count, x)
        self.results = model.fit_regularized(method='l1', disp=0, alpha=10)