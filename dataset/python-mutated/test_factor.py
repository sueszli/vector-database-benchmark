"""
Tests for Factor terms.
"""
from functools import partial
from itertools import product
from nose_parameterized import parameterized
from unittest import TestCase, skipIf
from toolz import compose
import numpy as np
from numpy import apply_along_axis, arange, array, datetime64, empty, eye, inf, log1p, nan, ones, ones_like, rot90, where
from numpy.random import randn, seed
import pandas as pd
from scipy.stats.mstats import winsorize as scipy_winsorize
from zipline.errors import BadPercentileBounds, UnknownRankMethod
from zipline.lib.labelarray import LabelArray
from zipline.lib.rank import masked_rankdata_2d
from zipline.lib.normalize import naive_grouped_rowwise_apply as grouped_apply
from zipline.pipeline import Classifier, Factor, Filter, Pipeline
from zipline.pipeline.data import DataSet, Column, EquityPricing
from zipline.pipeline.factors import CustomFactor, DailyReturns, Returns, PercentChange
from zipline.pipeline.factors.factor import summary_funcs, winsorize as zp_winsorize
from zipline.testing import check_allclose, check_arrays, parameter_space, permute_rows
from zipline.testing.fixtures import WithUSEquityPricingPipelineEngine, ZiplineTestCase
from zipline.testing.predicates import assert_equal
from zipline.utils.numpy_utils import as_column, categorical_dtype, datetime64ns_dtype, float64_dtype, ignore_nanwarnings, int64_dtype, NaTns
from zipline.utils.math_utils import nanmean, nanstd
from zipline.utils.pandas_utils import new_pandas, skip_pipeline_new_pandas
from .base import BaseUSEquityPipelineTestCase

class F(Factor):
    dtype = float64_dtype
    inputs = ()
    window_length = 0

class OtherF(Factor):
    dtype = float64_dtype
    inputs = ()
    window_length = 0

class C(Classifier):
    dtype = int64_dtype
    missing_value = -1
    inputs = ()
    window_length = 0

class OtherC(Classifier):
    dtype = int64_dtype
    missing_value = -1
    inputs = ()
    window_length = 0

class Mask(Filter):
    inputs = ()
    window_length = 0
for_each_factor_dtype = parameterized.expand([('datetime64[ns]', datetime64ns_dtype), ('float', float64_dtype)])

def scipy_winsorize_with_nan_handling(array, limits):
    if False:
        while True:
            i = 10
    "\n    Wrapper around scipy.stats.mstats.winsorize that handles NaNs correctly.\n\n    scipy's winsorize sorts NaNs to the end of the array when calculating\n    percentiles.\n    "
    nancount = np.isnan(array).sum()
    if nancount == len(array):
        return array.copy()
    sorter = array.argsort()
    unsorter = sorter.argsort()
    if nancount:
        sorted_non_nans = array[sorter][:-nancount]
    else:
        sorted_non_nans = array[sorter]
    sorted_winsorized = np.hstack([scipy_winsorize(sorted_non_nans, limits).data, np.full(nancount, np.nan)])
    return sorted_winsorized[unsorter]

class FactorTestCase(BaseUSEquityPipelineTestCase):

    def init_instance_fixtures(self):
        if False:
            print('Hello World!')
        super(FactorTestCase, self).init_instance_fixtures()
        self.f = F()

    def test_bad_input(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(UnknownRankMethod):
            self.f.rank('not a real rank method')

    @parameter_space(method_name=['isnan', 'notnan', 'isfinite'])
    def test_float64_only_ops(self, method_name):
        if False:
            while True:
                i = 10

        class NotFloat(Factor):
            dtype = datetime64ns_dtype
            inputs = ()
            window_length = 0
        nf = NotFloat()
        meth = getattr(nf, method_name)
        with self.assertRaises(TypeError):
            meth()

    @parameter_space(custom_missing_value=[-1, 0])
    def test_isnull_int_dtype(self, custom_missing_value):
        if False:
            while True:
                i = 10

        class CustomMissingValue(Factor):
            dtype = int64_dtype
            window_length = 0
            missing_value = custom_missing_value
            inputs = ()
        factor = CustomMissingValue()
        data = arange(25).reshape(5, 5)
        data[eye(5, dtype=bool)] = custom_missing_value
        self.check_terms({'isnull': factor.isnull(), 'notnull': factor.notnull()}, {'isnull': eye(5, dtype=bool), 'notnull': ~eye(5, dtype=bool)}, initial_workspace={factor: data}, mask=self.build_mask(ones((5, 5))))

    def test_isnull_datetime_dtype(self):
        if False:
            while True:
                i = 10

        class DatetimeFactor(Factor):
            dtype = datetime64ns_dtype
            window_length = 0
            inputs = ()
        factor = DatetimeFactor()
        data = arange(25).reshape(5, 5).astype('datetime64[ns]')
        data[eye(5, dtype=bool)] = NaTns
        self.check_terms({'isnull': factor.isnull(), 'notnull': factor.notnull()}, {'isnull': eye(5, dtype=bool), 'notnull': ~eye(5, dtype=bool)}, initial_workspace={factor: data}, mask=self.build_mask(ones((5, 5))))

    @for_each_factor_dtype
    def test_rank_ascending(self, name, factor_dtype):
        if False:
            i = 10
            return i + 15
        f = F(dtype=factor_dtype)
        data = array([[0, 1, 2, 3, 0], [1, 2, 3, 0, 1], [2, 3, 0, 1, 2], [3, 0, 1, 2, 3], [0, 1, 2, 3, 0]], dtype=factor_dtype)
        expected_ranks = {'ordinal': array([[1.0, 3.0, 4.0, 5.0, 2.0], [2.0, 4.0, 5.0, 1.0, 3.0], [3.0, 5.0, 1.0, 2.0, 4.0], [4.0, 1.0, 2.0, 3.0, 5.0], [1.0, 3.0, 4.0, 5.0, 2.0]]), 'average': array([[1.5, 3.0, 4.0, 5.0, 1.5], [2.5, 4.0, 5.0, 1.0, 2.5], [3.5, 5.0, 1.0, 2.0, 3.5], [4.5, 1.0, 2.0, 3.0, 4.5], [1.5, 3.0, 4.0, 5.0, 1.5]]), 'min': array([[1.0, 3.0, 4.0, 5.0, 1.0], [2.0, 4.0, 5.0, 1.0, 2.0], [3.0, 5.0, 1.0, 2.0, 3.0], [4.0, 1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 4.0, 5.0, 1.0]]), 'max': array([[2.0, 3.0, 4.0, 5.0, 2.0], [3.0, 4.0, 5.0, 1.0, 3.0], [4.0, 5.0, 1.0, 2.0, 4.0], [5.0, 1.0, 2.0, 3.0, 5.0], [2.0, 3.0, 4.0, 5.0, 2.0]]), 'dense': array([[1.0, 2.0, 3.0, 4.0, 1.0], [2.0, 3.0, 4.0, 1.0, 2.0], [3.0, 4.0, 1.0, 2.0, 3.0], [4.0, 1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0, 1.0]])}

        def check(terms):
            if False:
                while True:
                    i = 10
            self.check_terms(terms, expected={name: expected_ranks[name] for name in terms}, initial_workspace={f: data}, mask=self.build_mask(ones((5, 5))))
        check({meth: f.rank(method=meth) for meth in expected_ranks})
        check({meth: f.rank(method=meth, ascending=True) for meth in expected_ranks})
        check({'ordinal': f.rank()})
        check({'ordinal': f.rank(ascending=True)})

    @for_each_factor_dtype
    def test_rank_descending(self, name, factor_dtype):
        if False:
            print('Hello World!')
        f = F(dtype=factor_dtype)
        data = array([[0, 1, 2, 3, 0], [1, 2, 3, 0, 1], [2, 3, 0, 1, 2], [3, 0, 1, 2, 3], [0, 1, 2, 3, 0]], dtype=factor_dtype)
        expected_ranks = {'ordinal': array([[4.0, 3.0, 2.0, 1.0, 5.0], [3.0, 2.0, 1.0, 5.0, 4.0], [2.0, 1.0, 5.0, 4.0, 3.0], [1.0, 5.0, 4.0, 3.0, 2.0], [4.0, 3.0, 2.0, 1.0, 5.0]]), 'average': array([[4.5, 3.0, 2.0, 1.0, 4.5], [3.5, 2.0, 1.0, 5.0, 3.5], [2.5, 1.0, 5.0, 4.0, 2.5], [1.5, 5.0, 4.0, 3.0, 1.5], [4.5, 3.0, 2.0, 1.0, 4.5]]), 'min': array([[4.0, 3.0, 2.0, 1.0, 4.0], [3.0, 2.0, 1.0, 5.0, 3.0], [2.0, 1.0, 5.0, 4.0, 2.0], [1.0, 5.0, 4.0, 3.0, 1.0], [4.0, 3.0, 2.0, 1.0, 4.0]]), 'max': array([[5.0, 3.0, 2.0, 1.0, 5.0], [4.0, 2.0, 1.0, 5.0, 4.0], [3.0, 1.0, 5.0, 4.0, 3.0], [2.0, 5.0, 4.0, 3.0, 2.0], [5.0, 3.0, 2.0, 1.0, 5.0]]), 'dense': array([[4.0, 3.0, 2.0, 1.0, 4.0], [3.0, 2.0, 1.0, 4.0, 3.0], [2.0, 1.0, 4.0, 3.0, 2.0], [1.0, 4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0, 4.0]])}

        def check(terms):
            if False:
                while True:
                    i = 10
            self.check_terms(terms, expected={name: expected_ranks[name] for name in terms}, initial_workspace={f: data}, mask=self.build_mask(ones((5, 5))))
        check({meth: f.rank(method=meth, ascending=False) for meth in expected_ranks})
        check({'ordinal': f.rank(ascending=False)})

    @for_each_factor_dtype
    def test_rank_after_mask(self, name, factor_dtype):
        if False:
            while True:
                i = 10
        f = F(dtype=factor_dtype)
        data = array([[0, 1, 2, 3, 0], [1, 2, 3, 0, 1], [2, 3, 0, 1, 2], [3, 0, 1, 2, 3], [0, 1, 2, 3, 0]], dtype=factor_dtype)
        mask_data = ~eye(5, dtype=bool)
        initial_workspace = {f: data, Mask(): mask_data}
        terms = {'ascending_nomask': f.rank(ascending=True), 'ascending_mask': f.rank(ascending=True, mask=Mask()), 'descending_nomask': f.rank(ascending=False), 'descending_mask': f.rank(ascending=False, mask=Mask())}
        expected = {'ascending_nomask': array([[1.0, 3.0, 4.0, 5.0, 2.0], [2.0, 4.0, 5.0, 1.0, 3.0], [3.0, 5.0, 1.0, 2.0, 4.0], [4.0, 1.0, 2.0, 3.0, 5.0], [1.0, 3.0, 4.0, 5.0, 2.0]]), 'descending_nomask': array([[4.0, 3.0, 2.0, 1.0, 5.0], [3.0, 2.0, 1.0, 5.0, 4.0], [2.0, 1.0, 5.0, 4.0, 3.0], [1.0, 5.0, 4.0, 3.0, 2.0], [4.0, 3.0, 2.0, 1.0, 5.0]]), 'ascending_mask': array([[nan, 2.0, 3.0, 4.0, 1.0], [2.0, nan, 4.0, 1.0, 3.0], [2.0, 4.0, nan, 1.0, 3.0], [3.0, 1.0, 2.0, nan, 4.0], [1.0, 2.0, 3.0, 4.0, nan]]), 'descending_mask': array([[nan, 3.0, 2.0, 1.0, 4.0], [2.0, nan, 1.0, 4.0, 3.0], [2.0, 1.0, nan, 4.0, 3.0], [1.0, 4.0, 3.0, nan, 2.0], [4.0, 3.0, 2.0, 1.0, nan]])}
        self.check_terms(terms, expected, initial_workspace, mask=self.build_mask(ones((5, 5))))

    @for_each_factor_dtype
    def test_grouped_rank_ascending(self, name, factor_dtype=float64_dtype):
        if False:
            for i in range(10):
                print('nop')
        f = F(dtype=factor_dtype)
        c = C()
        str_c = C(dtype=categorical_dtype, missing_value=None)
        data = array([[0, 1, 2, 3, 0], [1, 2, 3, 0, 1], [2, 3, 0, 1, 2], [3, 0, 1, 2, 3], [0, 1, 2, 3, 0]], dtype=factor_dtype)
        classifier_data = array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0]], dtype=int64_dtype)
        string_classifier_data = LabelArray(classifier_data.astype(str).astype(object), missing_value=None)
        expected_ranks = {'ordinal': array([[1.0, 1.0, 3.0, 2.0, 2.0], [1.0, 2.0, 3.0, 1.0, 2.0], [2.0, 2.0, 1.0, 1.0, 3.0], [2.0, 1.0, 1.0, 2.0, 3.0], [1.0, 1.0, 3.0, 2.0, 2.0]]), 'average': array([[1.5, 1.0, 3.0, 2.0, 1.5], [1.5, 2.0, 3.0, 1.0, 1.5], [2.5, 2.0, 1.0, 1.0, 2.5], [2.5, 1.0, 1.0, 2.0, 2.5], [1.5, 1.0, 3.0, 2.0, 1.5]]), 'min': array([[1.0, 1.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 1.0, 1.0], [2.0, 2.0, 1.0, 1.0, 2.0], [2.0, 1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 3.0, 2.0, 1.0]]), 'max': array([[2.0, 1.0, 3.0, 2.0, 2.0], [2.0, 2.0, 3.0, 1.0, 2.0], [3.0, 2.0, 1.0, 1.0, 3.0], [3.0, 1.0, 1.0, 2.0, 3.0], [2.0, 1.0, 3.0, 2.0, 2.0]]), 'dense': array([[1.0, 1.0, 2.0, 2.0, 1.0], [1.0, 2.0, 2.0, 1.0, 1.0], [2.0, 2.0, 1.0, 1.0, 2.0], [2.0, 1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0, 1.0]])}

        def check(terms):
            if False:
                i = 10
                return i + 15
            self.check_terms(terms, expected={name: expected_ranks[name] for name in terms}, initial_workspace={f: data, c: classifier_data, str_c: string_classifier_data}, mask=self.build_mask(ones((5, 5))))
        check({meth: f.rank(method=meth, groupby=c) for meth in expected_ranks})
        check({meth: f.rank(method=meth, groupby=str_c) for meth in expected_ranks})
        check({meth: f.rank(method=meth, groupby=c, ascending=True) for meth in expected_ranks})
        check({meth: f.rank(method=meth, groupby=str_c, ascending=True) for meth in expected_ranks})
        check({'ordinal': f.rank(groupby=c)})
        check({'ordinal': f.rank(groupby=str_c)})
        check({'ordinal': f.rank(groupby=c, ascending=True)})
        check({'ordinal': f.rank(groupby=str_c, ascending=True)})

    @for_each_factor_dtype
    def test_grouped_rank_descending(self, name, factor_dtype):
        if False:
            return 10
        f = F(dtype=factor_dtype)
        c = C()
        str_c = C(dtype=categorical_dtype, missing_value=None)
        data = array([[0, 1, 2, 3, 0], [1, 2, 3, 0, 1], [2, 3, 0, 1, 2], [3, 0, 1, 2, 3], [0, 1, 2, 3, 0]], dtype=factor_dtype)
        classifier_data = array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0]], dtype=int64_dtype)
        string_classifier_data = LabelArray(classifier_data.astype(str).astype(object), missing_value=None)
        expected_ranks = {'ordinal': array([[2.0, 2.0, 1.0, 1.0, 3.0], [2.0, 1.0, 1.0, 2.0, 3.0], [1.0, 1.0, 3.0, 2.0, 2.0], [1.0, 2.0, 3.0, 1.0, 2.0], [2.0, 2.0, 1.0, 1.0, 3.0]]), 'average': array([[2.5, 2.0, 1.0, 1.0, 2.5], [2.5, 1.0, 1.0, 2.0, 2.5], [1.5, 1.0, 3.0, 2.0, 1.5], [1.5, 2.0, 3.0, 1.0, 1.5], [2.5, 2.0, 1.0, 1.0, 2.5]]), 'min': array([[2.0, 2.0, 1.0, 1.0, 2.0], [2.0, 1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 1.0, 1.0], [2.0, 2.0, 1.0, 1.0, 2.0]]), 'max': array([[3.0, 2.0, 1.0, 1.0, 3.0], [3.0, 1.0, 1.0, 2.0, 3.0], [2.0, 1.0, 3.0, 2.0, 2.0], [2.0, 2.0, 3.0, 1.0, 2.0], [3.0, 2.0, 1.0, 1.0, 3.0]]), 'dense': array([[2.0, 2.0, 1.0, 1.0, 2.0], [2.0, 1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0, 1.0], [1.0, 2.0, 2.0, 1.0, 1.0], [2.0, 2.0, 1.0, 1.0, 2.0]])}

        def check(terms):
            if False:
                i = 10
                return i + 15
            self.check_terms(terms, expected={name: expected_ranks[name] for name in terms}, initial_workspace={f: data, c: classifier_data, str_c: string_classifier_data}, mask=self.build_mask(ones((5, 5))))
        check({meth: f.rank(method=meth, groupby=c, ascending=False) for meth in expected_ranks})
        check({meth: f.rank(method=meth, groupby=str_c, ascending=False) for meth in expected_ranks})
        check({'ordinal': f.rank(groupby=c, ascending=False)})
        check({'ordinal': f.rank(groupby=str_c, ascending=False)})

    @parameterized.expand([(100, 15), (101, 4), (102, 100)])
    def test_returns(self, seed_value, window_length):
        if False:
            return 10
        returns = Returns(window_length=window_length)
        today = datetime64(1, 'ns')
        assets = arange(3)
        seed(seed_value)
        test_data = abs(randn(window_length, 3))
        expected = (test_data[-1] - test_data[0]) / test_data[0]
        out = empty((3,), dtype=float)
        returns.compute(today, assets, out, test_data)
        check_allclose(expected, out)

    @parameterized.expand([(100, 15), (101, 4), (102, 100)])
    def test_percentchange(self, seed_value, window_length):
        if False:
            return 10
        pct_change = PercentChange(inputs=[EquityPricing.close], window_length=window_length)
        today = datetime64(1, 'ns')
        assets = arange(8)
        seed(seed_value)
        middle_rows = randn(window_length - 2, 8)
        first_row = array([1, 2, 2, 1, -1, -1, 0, nan])
        end_row = array([2, 1, 2, -2, 2, -2, 1, 1])
        test_data = np.vstack([first_row, middle_rows, end_row])
        expected = array([1, -0.5, 0, -3, 3, -1, inf, nan])
        out = empty((8,), dtype=float)
        pct_change.compute(today, assets, out, test_data)
        check_allclose(expected, out)
        with self.assertRaises(ValueError):
            PercentChange(inputs=(), window_length=2)
        with self.assertRaises(ValueError):
            PercentChange(inputs=[EquityPricing.close], window_length=1)

    def gen_ranking_cases():
        if False:
            while True:
                i = 10
        seeds = range(int(10000.0), int(100000.0), int(10000.0))
        methods = ('ordinal', 'average')
        use_mask_values = (True, False)
        set_missing_values = (True, False)
        ascending_values = (True, False)
        return product(seeds, methods, use_mask_values, set_missing_values, ascending_values)

    @parameterized.expand(gen_ranking_cases())
    def test_masked_rankdata_2d(self, seed_value, method, use_mask, set_missing, ascending):
        if False:
            i = 10
            return i + 15
        eyemask = ~eye(5, dtype=bool)
        nomask = ones((5, 5), dtype=bool)
        seed(seed_value)
        asfloat = randn(5, 5) * seed_value
        asdatetime = asfloat.copy().view('datetime64[ns]')
        mask = eyemask if use_mask else nomask
        if set_missing:
            asfloat[:, 2] = nan
            asdatetime[:, 2] = NaTns
        float_result = masked_rankdata_2d(data=asfloat, mask=mask, missing_value=nan, method=method, ascending=True)
        datetime_result = masked_rankdata_2d(data=asdatetime, mask=mask, missing_value=NaTns, method=method, ascending=True)
        check_arrays(float_result, datetime_result)

    def test_normalizations_hand_computed(self):
        if False:
            return 10
        '\n        Test the hand-computed example in factor.demean.\n        '
        f = self.f
        m = Mask()
        c = C()
        str_c = C(dtype=categorical_dtype, missing_value=None)
        factor_data = array([[1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 1.0], [2.0, 3.0, 4.0, 1.5], [2.5, 3.5, 1.0, 2.0]])
        filter_data = array([[False, True, True, True], [True, False, True, True], [True, True, False, True], [True, True, True, False]], dtype=bool)
        classifier_data = array([[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]], dtype=int64_dtype)
        string_classifier_data = LabelArray(classifier_data.astype(str).astype(object), missing_value=None)
        terms = {'vanilla': f.demean(), 'masked': f.demean(mask=m), 'grouped': f.demean(groupby=c), 'grouped_str': f.demean(groupby=str_c), 'grouped_masked': f.demean(mask=m, groupby=c), 'grouped_masked_str': f.demean(mask=m, groupby=str_c)}
        expected = {'vanilla': array([[-1.5, -0.5, 0.5, 1.5], [-0.625, 0.375, 1.375, -1.125], [-0.625, 0.375, 1.375, -1.125], [0.25, 1.25, -1.25, -0.25]]), 'masked': array([[nan, -1.0, 0.0, 1.0], [-0.5, nan, 1.5, -1.0], [-0.166, 0.833, nan, -0.666], [0.166, 1.166, -1.333, nan]]), 'grouped': array([[-0.5, 0.5, -0.5, 0.5], [-0.5, 0.5, 1.25, -1.25], [-0.5, 0.5, 1.25, -1.25], [-0.5, 0.5, -0.5, 0.5]]), 'grouped_masked': array([[nan, 0.0, -0.5, 0.5], [0.0, nan, 1.25, -1.25], [-0.5, 0.5, nan, 0.0], [-0.5, 0.5, 0.0, nan]])}
        expected['grouped_str'] = expected['grouped']
        expected['grouped_masked_str'] = expected['grouped_masked']
        self.check_terms(terms, expected, initial_workspace={f: factor_data, c: classifier_data, str_c: string_classifier_data, m: filter_data}, mask=self.build_mask(self.ones_mask(shape=factor_data.shape)), check=partial(check_allclose, atol=0.001))

    def test_winsorize_hand_computed(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the hand-computed example in factor.winsorize.\n        '
        f = self.f
        m = Mask()
        c = C()
        str_c = C(dtype=categorical_dtype, missing_value=None)
        factor_data = array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, nan, nan, nan], [1.0, 8.0, 27.0, 64.0, 125.0, 216.0, nan, nan, nan], [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, nan, nan, nan], [nan, nan, nan, nan, nan, nan, nan, nan, nan]])
        filter_data = array([[1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 1, 1, 1, 1, 1]], dtype=bool)
        classifier_data = array([[1, 1, 1, 2, 2, 2, 1, 1, 1], [1, 1, 1, 2, 2, 2, 1, 1, 1], [1, 1, 1, 2, 2, 2, 1, 1, 1], [1, 1, 1, 2, 2, 2, 1, 1, 1], [1, 1, 1, 2, 2, 2, 1, 1, 1]], dtype=int64_dtype)
        string_classifier_data = LabelArray(classifier_data.astype(str).astype(object), missing_value=None)
        terms = {'winsor_1': f.winsorize(min_percentile=0.33, max_percentile=0.67), 'winsor_2': f.winsorize(min_percentile=0.49, max_percentile=1), 'winsor_3': f.winsorize(min_percentile=0, max_percentile=0.67), 'masked': f.winsorize(min_percentile=0.33, max_percentile=0.67, mask=m), 'grouped': f.winsorize(min_percentile=0.34, max_percentile=0.66, groupby=c), 'grouped_str': f.winsorize(min_percentile=0.34, max_percentile=0.66, groupby=str_c), 'grouped_masked': f.winsorize(min_percentile=0.34, max_percentile=0.66, mask=m, groupby=c), 'grouped_masked_str': f.winsorize(min_percentile=0.34, max_percentile=0.66, mask=m, groupby=str_c)}
        expected = {'winsor_1': array([[3.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0], [2.0, 2.0, 3.0, 4.0, 5.0, 5.0, nan, nan, nan], [8.0, 8.0, 27.0, 64.0, 125.0, 125.0, nan, nan, nan], [5.0, 5.0, 4.0, 3.0, 2.0, 2.0, nan, nan, nan], [nan, nan, nan, nan, nan, nan, nan, nan, nan]]), 'winsor_2': array([[5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3.0, 3.0, 3.0, 4.0, 5.0, 6.0, nan, nan, nan], [27.0, 27.0, 27.0, 64.0, 125.0, 216.0, nan, nan, nan], [6.0, 5.0, 4.0, 3.0, 3.0, 3.0, nan, nan, nan], [nan, nan, nan, nan, nan, nan, nan, nan, nan]]), 'winsor_3': array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0], [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, nan, nan, nan], [1.0, 8.0, 27.0, 64.0, 125.0, 125.0, nan, nan, nan], [5.0, 5.0, 4.0, 3.0, 2.0, 1.0, nan, nan, nan], [nan, nan, nan, nan, nan, nan, nan, nan, nan]]), 'masked': array([[3.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0], [nan, 3.0, 3.0, 4.0, 5.0, 5.0, nan, nan, nan], [27.0, nan, 27.0, 64.0, 125.0, 125.0, nan, nan, nan], [5.0, 5.0, nan, 3.0, 2.0, 2.0, nan, nan, nan], [nan, nan, nan, nan, nan, nan, nan, nan, nan]]), 'grouped': array([[3.0, 3.0, 3.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0], [2.0, 2.0, 2.0, 5.0, 5.0, 5.0, nan, nan, nan], [8.0, 8.0, 8.0, 125.0, 125.0, 125.0, nan, nan, nan], [5.0, 5.0, 5.0, 2.0, 2.0, 2.0, nan, nan, nan], [nan, nan, nan, nan, nan, nan, nan, nan, nan]]), 'grouped_masked': array([[3.0, 3.0, 3.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0], [nan, 2.0, 3.0, 5.0, 5.0, 5.0, nan, nan, nan], [1.0, nan, 27.0, 125.0, 125.0, 125.0, nan, nan, nan], [6.0, 5.0, nan, 2.0, 2.0, 2.0, nan, nan, nan], [nan, nan, nan, nan, nan, nan, nan, nan, nan]])}
        expected['grouped_str'] = expected['grouped']
        expected['grouped_masked_str'] = expected['grouped_masked']
        self.check_terms(terms, expected, initial_workspace={f: factor_data, c: classifier_data, str_c: string_classifier_data, m: filter_data}, mask=self.build_mask(self.ones_mask(shape=factor_data.shape)), check=partial(check_allclose, atol=0.001))

    def test_winsorize_no_nans(self):
        if False:
            i = 10
            return i + 15
        data = array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        permutation = array([2, 1, 6, 8, 7, 5, 3, 9, 4, 0])
        for perm in (slice(None), permutation):
            result = zp_winsorize(data[perm], 0.1, 0.9)
            expected = array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0])[perm]
            assert_equal(result, expected)
            result = zp_winsorize(data[perm], 0.2, 0.8)
            expected = array([2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0])[perm]
            assert_equal(result, expected)
            result = zp_winsorize(data[perm], 0.0, 0.8)
            expected = array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0])[perm]
            assert_equal(result, expected)
            result = zp_winsorize(data[perm], 0.2, 1.0)
            expected = array([2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])[perm]
            assert_equal(result, expected)
            result = zp_winsorize(data[perm], 0.0, 1.0)
            expected = array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])[perm]
            assert_equal(result, expected)

    def test_winsorize_nans(self):
        if False:
            i = 10
            return i + 15
        data = array([4.0, 3.0, 0.0, 1.0, 2.0, nan, nan, nan, 9.0, 5.0, 6.0, 8.0, 7.0])
        result = zp_winsorize(data, 0.1, 0.9)
        expected = array([4.0, 3.0, 1.0, 1.0, 2.0, nan, nan, nan, 8.0, 5.0, 6.0, 8.0, 7.0])
        assert_equal(result, expected)
        result = zp_winsorize(data, 0.2, 0.8)
        expected = array([4.0, 3.0, 2.0, 2.0, 2.0, nan, nan, nan, 7.0, 5.0, 6.0, 7.0, 7.0])
        assert_equal(result, expected)
        result = zp_winsorize(data, 0, 0.8)
        expected = array([4.0, 3.0, 0.0, 1.0, 2.0, nan, nan, nan, 7.0, 5.0, 6.0, 7.0, 7.0])
        assert_equal(result, expected)
        result = zp_winsorize(data, 0.2, 1.0)
        expected = array([4.0, 3.0, 2.0, 2.0, 2.0, nan, nan, nan, 9.0, 5.0, 6.0, 8.0, 7.0])
        assert_equal(result, expected)

    def test_winsorize_bad_bounds(self):
        if False:
            print('Hello World!')
        '\n        Test out of bounds input for factor.winsorize.\n        '
        f = self.f
        bad_percentiles = [(-0.1, 1), (0, 95), (5, 95), (5, 5), (0.6, 0.4)]
        for (min_, max_) in bad_percentiles:
            with self.assertRaises(BadPercentileBounds):
                f.winsorize(min_percentile=min_, max_percentile=max_)

    @skipIf(new_pandas, skip_pipeline_new_pandas)
    @parameter_space(seed_value=[1, 2], normalizer_name_and_func=[('demean', {}, lambda row: row - nanmean(row)), ('zscore', {}, lambda row: (row - nanmean(row)) / nanstd(row)), ('winsorize', {'min_percentile': 0.25, 'max_percentile': 0.75}, lambda row: scipy_winsorize_with_nan_handling(row, limits=0.25))], add_nulls_to_factor=(False, True))
    def test_normalizations_randomized(self, seed_value, normalizer_name_and_func, add_nulls_to_factor):
        if False:
            for i in range(10):
                print('nop')
        (name, kwargs, func) = normalizer_name_and_func
        shape = (20, 20)
        nomask = self.ones_mask(shape=shape)
        eyemask = self.eye_mask(shape=shape)
        eyemask90 = rot90(eyemask)
        xmask = eyemask & eyemask90
        factor_data = self.randn_data(seed=seed_value, shape=shape)
        if add_nulls_to_factor:
            factor_data = where(eyemask, factor_data, nan)
        classifier_data = (self.arange_data(shape=shape, dtype=int64_dtype) + seed_value) % 3
        classifier_data_eyenulls = where(eyemask, classifier_data, -1)
        classifier_data_eyenulls90 = where(eyemask90, classifier_data, -1)
        classifier_data_xnulls = where(xmask, classifier_data, -1)
        f = self.f
        c = C()
        c_with_nulls = OtherC()
        m = Mask()
        method = partial(getattr(f, name), **kwargs)
        terms = {'vanilla': method(), 'masked': method(mask=m), 'grouped': method(groupby=c), 'grouped_with_nulls': method(groupby=c_with_nulls), 'both': method(mask=m, groupby=c), 'both_with_nulls': method(mask=m, groupby=c_with_nulls)}
        expected = {'vanilla': apply_along_axis(func, 1, factor_data), 'masked': where(eyemask, grouped_apply(factor_data, eyemask, func), nan), 'grouped': grouped_apply(factor_data, classifier_data, func), 'grouped_with_nulls': where(eyemask90, grouped_apply(factor_data, classifier_data_eyenulls90, func), nan), 'both': where(eyemask, grouped_apply(factor_data, classifier_data_eyenulls, func), nan), 'both_with_nulls': where(xmask, grouped_apply(factor_data, classifier_data_xnulls, func), nan)}
        self.check_terms(terms=terms, expected=expected, initial_workspace={f: factor_data, c: classifier_data, c_with_nulls: classifier_data_eyenulls90, Mask(): eyemask}, mask=self.build_mask(nomask))

    @parameter_space(method_name=['demean', 'zscore'])
    def test_cant_normalize_non_float(self, method_name):
        if False:
            print('Hello World!')

        class DateFactor(Factor):
            dtype = datetime64ns_dtype
            inputs = ()
            window_length = 0
        d = DateFactor()
        with self.assertRaises(TypeError) as e:
            getattr(d, method_name)()
        errmsg = str(e.exception)
        expected = '{normalizer}() is only defined on Factors of dtype float64, but it was called on a Factor of dtype datetime64[ns].'.format(normalizer=method_name)
        self.assertEqual(errmsg, expected)

    @parameter_space(seed=[1, 2, 3])
    def test_quantiles_unmasked(self, seed):
        if False:
            i = 10
            return i + 15
        permute = partial(permute_rows, seed)
        shape = (6, 6)
        factor_data = permute(log1p(arange(36, dtype=float).reshape(shape)))
        f = self.f
        permuted_array = compose(permute, partial(array, dtype=int64_dtype))
        self.check_terms(terms={'2': f.quantiles(bins=2), '3': f.quantiles(bins=3), '6': f.quantiles(bins=6)}, initial_workspace={f: factor_data}, expected={'2': permuted_array([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]), '3': permuted_array([[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]]), '6': permuted_array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])}, mask=self.build_mask(self.ones_mask(shape=shape)))

    @parameter_space(seed=[1, 2, 3])
    def test_quantiles_masked(self, seed):
        if False:
            return 10
        permute = partial(permute_rows, seed)
        shape = (7, 7)
        factor_data = permute(log1p(arange(49, dtype=float).reshape(shape)))
        factor_data_w_nans = where(permute(rot90(self.eye_mask(shape=shape))), factor_data, nan)
        mask_data = permute(self.eye_mask(shape=shape))
        f = F()
        f_nans = OtherF()
        m = Mask()
        permuted_array = compose(permute, partial(array, dtype=int64_dtype))
        self.check_terms(terms={'2_masked': f.quantiles(bins=2, mask=m), '3_masked': f.quantiles(bins=3, mask=m), '6_masked': f.quantiles(bins=6, mask=m), '2_nans': f_nans.quantiles(bins=2), '3_nans': f_nans.quantiles(bins=3), '6_nans': f_nans.quantiles(bins=6)}, initial_workspace={f: factor_data, f_nans: factor_data_w_nans, m: mask_data}, expected={'2_masked': permuted_array([[-1, 0, 0, 0, 1, 1, 1], [0, -1, 0, 0, 1, 1, 1], [0, 0, -1, 0, 1, 1, 1], [0, 0, 0, -1, 1, 1, 1], [0, 0, 0, 1, -1, 1, 1], [0, 0, 0, 1, 1, -1, 1], [0, 0, 0, 1, 1, 1, -1]]), '3_masked': permuted_array([[-1, 0, 0, 1, 1, 2, 2], [0, -1, 0, 1, 1, 2, 2], [0, 0, -1, 1, 1, 2, 2], [0, 0, 1, -1, 1, 2, 2], [0, 0, 1, 1, -1, 2, 2], [0, 0, 1, 1, 2, -1, 2], [0, 0, 1, 1, 2, 2, -1]]), '6_masked': permuted_array([[-1, 0, 1, 2, 3, 4, 5], [0, -1, 1, 2, 3, 4, 5], [0, 1, -1, 2, 3, 4, 5], [0, 1, 2, -1, 3, 4, 5], [0, 1, 2, 3, -1, 4, 5], [0, 1, 2, 3, 4, -1, 5], [0, 1, 2, 3, 4, 5, -1]]), '2_nans': permuted_array([[0, 0, 0, 1, 1, 1, -1], [0, 0, 0, 1, 1, -1, 1], [0, 0, 0, 1, -1, 1, 1], [0, 0, 0, -1, 1, 1, 1], [0, 0, -1, 0, 1, 1, 1], [0, -1, 0, 0, 1, 1, 1], [-1, 0, 0, 0, 1, 1, 1]]), '3_nans': permuted_array([[0, 0, 1, 1, 2, 2, -1], [0, 0, 1, 1, 2, -1, 2], [0, 0, 1, 1, -1, 2, 2], [0, 0, 1, -1, 1, 2, 2], [0, 0, -1, 1, 1, 2, 2], [0, -1, 0, 1, 1, 2, 2], [-1, 0, 0, 1, 1, 2, 2]]), '6_nans': permuted_array([[0, 1, 2, 3, 4, 5, -1], [0, 1, 2, 3, 4, -1, 5], [0, 1, 2, 3, -1, 4, 5], [0, 1, 2, -1, 3, 4, 5], [0, 1, -1, 2, 3, 4, 5], [0, -1, 1, 2, 3, 4, 5], [-1, 0, 1, 2, 3, 4, 5]])}, mask=self.build_mask(self.ones_mask(shape=shape)))

    def test_quantiles_uneven_buckets(self):
        if False:
            for i in range(10):
                print('nop')
        permute = partial(permute_rows, 5)
        shape = (5, 5)
        factor_data = permute(log1p(arange(25, dtype=float).reshape(shape)))
        mask_data = permute(self.eye_mask(shape=shape))
        f = F()
        m = Mask()
        permuted_array = compose(permute, partial(array, dtype=int64_dtype))
        self.check_terms(terms={'3_masked': f.quantiles(bins=3, mask=m), '7_masked': f.quantiles(bins=7, mask=m)}, initial_workspace={f: factor_data, m: mask_data}, expected={'3_masked': permuted_array([[-1, 0, 0, 1, 2], [0, -1, 0, 1, 2], [0, 0, -1, 1, 2], [0, 0, 1, -1, 2], [0, 0, 1, 2, -1]]), '7_masked': permuted_array([[-1, 0, 2, 4, 6], [0, -1, 2, 4, 6], [0, 2, -1, 4, 6], [0, 2, 4, -1, 6], [0, 2, 4, 6, -1]])}, mask=self.build_mask(self.ones_mask(shape=shape)))

    def test_quantile_helpers(self):
        if False:
            print('Hello World!')
        f = self.f
        m = Mask()
        self.assertIs(f.quartiles(), f.quantiles(bins=4))
        self.assertIs(f.quartiles(mask=m), f.quantiles(bins=4, mask=m))
        self.assertIsNot(f.quartiles(), f.quartiles(mask=m))
        self.assertIs(f.quintiles(), f.quantiles(bins=5))
        self.assertIs(f.quintiles(mask=m), f.quantiles(bins=5, mask=m))
        self.assertIsNot(f.quintiles(), f.quintiles(mask=m))
        self.assertIs(f.deciles(), f.quantiles(bins=10))
        self.assertIs(f.deciles(mask=m), f.quantiles(bins=10, mask=m))
        self.assertIsNot(f.deciles(), f.deciles(mask=m))

    @parameter_space(seed=[1, 2, 3])
    def test_clip(self, seed):
        if False:
            i = 10
            return i + 15
        rand = np.random.RandomState(seed)
        shape = (5, 5)
        original_min = -10
        original_max = +10
        input_array = rand.uniform(original_min, original_max, size=shape)
        (min_, max_) = np.percentile(input_array, [25, 75])
        self.assertGreater(min_, original_min)
        self.assertLess(max_, original_max)
        f = F()
        self.check_terms(terms={'clip': f.clip(min_, max_)}, initial_workspace={f: input_array}, expected={'clip': np.clip(input_array, min_, max_)}, mask=self.build_mask(self.ones_mask(shape=shape)))

class ReprTestCase(TestCase):
    """
    Tests for term reprs.
    """

    def test_demean(self):
        if False:
            while True:
                i = 10
        r = F().demean().graph_repr()
        self.assertEqual(r, "GroupedRowTransform('demean')")

    def test_zscore(self):
        if False:
            print('Hello World!')
        r = F().zscore().graph_repr()
        self.assertEqual(r, "GroupedRowTransform('zscore')")

    def test_winsorize(self):
        if False:
            return 10
        r = F().winsorize(min_percentile=0.05, max_percentile=0.95).graph_repr()
        self.assertEqual(r, "GroupedRowTransform('winsorize')")

    def test_recarray_field_repr(self):
        if False:
            return 10

        class MultipleOutputs(CustomFactor):
            outputs = ['a', 'b']
            inputs = ()
            window_length = 5

            def recursive_repr(self):
                if False:
                    i = 10
                    return i + 15
                return 'CustomRepr()'
        a = MultipleOutputs().a
        b = MultipleOutputs().b
        self.assertEqual(a.graph_repr(), 'CustomRepr().a')
        self.assertEqual(b.graph_repr(), 'CustomRepr().b')

    def test_latest_repr(self):
        if False:
            while True:
                i = 10

        class SomeDataSet(DataSet):
            a = Column(dtype=float64_dtype)
            b = Column(dtype=float64_dtype)
        self.assertEqual(SomeDataSet.a.latest.graph_repr(), 'Latest')
        self.assertEqual(SomeDataSet.b.latest.graph_repr(), 'Latest')

    def test_recursive_repr(self):
        if False:
            for i in range(10):
                print('nop')

        class DS(DataSet):
            a = Column(dtype=float64_dtype)
            b = Column(dtype=float64_dtype)

        class Input(CustomFactor):
            inputs = ()
            window_safe = True

        class HasInputs(CustomFactor):
            inputs = [Input(window_length=3), DS.a, DS.b]
            window_length = 3
        result = repr(HasInputs())
        expected = 'HasInputs([Input(...), DS.a, DS.b], 3)'
        self.assertEqual(result, expected)

    def test_rank_repr(self):
        if False:
            i = 10
            return i + 15
        rank = DailyReturns().rank()
        result = repr(rank)
        expected = "Rank(DailyReturns(...), method='ordinal')"
        self.assertEqual(result, expected)
        recursive_repr = rank.recursive_repr()
        self.assertEqual(recursive_repr, 'Rank(...)')

    def test_rank_repr_with_mask(self):
        if False:
            i = 10
            return i + 15
        rank = DailyReturns().rank(mask=Mask())
        result = repr(rank)
        expected = "Rank(DailyReturns(...), method='ordinal', mask=Mask(...))"
        self.assertEqual(result, expected)
        recursive_repr = rank.recursive_repr()
        self.assertEqual(recursive_repr, 'Rank(...)')

class TestWindowSafety(TestCase):

    def test_zscore_is_window_safe(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(F().zscore().window_safe)

    @parameter_space(__fail_fast=True, is_window_safe=[True, False])
    def test_window_safety_propagates_to_recarray_fields(self, is_window_safe):
        if False:
            while True:
                i = 10

        class MultipleOutputs(CustomFactor):
            outputs = ['a', 'b']
            inputs = ()
            window_length = 5
            window_safe = is_window_safe
        mo = MultipleOutputs()
        for attr in (mo.a, mo.b):
            self.assertEqual(attr.window_safe, mo.window_safe)

    def test_demean_is_window_safe_if_input_is_window_safe(self):
        if False:
            print('Hello World!')
        self.assertFalse(F().demean().window_safe)
        self.assertFalse(F(window_safe=False).demean().window_safe)
        self.assertTrue(F(window_safe=True).demean().window_safe)

    def test_winsorize_is_window_safe_if_input_is_window_safe(self):
        if False:
            return 10
        self.assertFalse(F().winsorize(min_percentile=0.05, max_percentile=0.95).window_safe)
        self.assertFalse(F(window_safe=False).winsorize(min_percentile=0.05, max_percentile=0.95).window_safe)
        self.assertTrue(F(window_safe=True).winsorize(min_percentile=0.05, max_percentile=0.95).window_safe)

class TestPostProcessAndToWorkSpaceValue(ZiplineTestCase):

    @parameter_space(dtype_=(float64_dtype, datetime64ns_dtype))
    def test_reversability(self, dtype_):
        if False:
            print('Hello World!')

        class F(Factor):
            inputs = ()
            dtype = dtype_
            window_length = 0
        f = F()
        column_data = array([[0, f.missing_value], [1, f.missing_value], [2, 3]], dtype=dtype_)
        assert_equal(f.postprocess(column_data.ravel()), column_data.ravel())
        pipeline_output = pd.Series(data=array([0, 1, 2, 3], dtype=dtype_), index=pd.MultiIndex.from_arrays([[pd.Timestamp('2014-01-01'), pd.Timestamp('2014-01-02'), pd.Timestamp('2014-01-03'), pd.Timestamp('2014-01-03')], [0, 0, 0, 1]]))
        assert_equal(f.to_workspace_value(pipeline_output, pd.Index([0, 1])), column_data)

class TestSpecialCases(WithUSEquityPricingPipelineEngine, ZiplineTestCase):
    ASSET_FINDER_COUNTRY_CODE = 'US'

    def check_equivalent_terms(self, terms):
        if False:
            while True:
                i = 10
        self.assertTrue(len(terms) > 1, 'Need at least two terms to compare')
        pipe = Pipeline(terms)
        (start, end) = self.trading_days[[-10, -1]]
        results = self.pipeline_engine.run_pipeline(pipe, start, end)
        first_column = results.iloc[:, 0]
        for name in terms:
            assert_equal(results.loc[:, name], first_column, check_names=False)

    def test_daily_returns_is_special_case_of_returns(self):
        if False:
            while True:
                i = 10
        self.check_equivalent_terms({'daily': DailyReturns(), 'manual_daily': Returns(window_length=2)})

class SummaryTestCase(BaseUSEquityPipelineTestCase, ZiplineTestCase):

    @parameter_space(seed=[1, 2, 3], mask=[np.zeros((10, 5), dtype=bool), ones((10, 5), dtype=bool), eye(10, 5, dtype=bool), ~eye(10, 5, dtype=bool)])
    def test_summary_methods(self, seed, mask):
        if False:
            print('Hello World!')
        'Test that summary funcs work the same as numpy NaN-aware funcs.\n        '
        rand = np.random.RandomState(seed)
        shape = (10, 5)
        data = rand.randn(*shape)
        data[~mask] = np.nan
        workspace = {F(): data}
        terms = {'mean': F().mean(), 'sum': F().sum(), 'median': F().median(), 'min': F().min(), 'max': F().max(), 'stddev': F().stddev(), 'notnull_count': F().notnull_count()}
        with ignore_nanwarnings():
            expected = {'mean': as_column(np.nanmean(data, axis=1)), 'sum': as_column(np.nansum(data, axis=1)), 'median': as_column(np.nanmedian(data, axis=1)), 'min': as_column(np.nanmin(data, axis=1)), 'max': as_column(np.nanmax(data, axis=1)), 'stddev': as_column(np.nanstd(data, axis=1)), 'notnull_count': as_column((~np.isnan(data)).sum(axis=1))}
        self.assertEqual(set(expected), summary_funcs.names)
        self.check_terms(terms=terms, expected=expected, initial_workspace=workspace, mask=self.build_mask(ones(shape)))

    @parameter_space(seed=[4, 5, 6], mask=[np.zeros((10, 5), dtype=bool), ones((10, 5), dtype=bool), eye(10, 5, dtype=bool), ~eye(10, 5, dtype=bool)])
    def test_built_in_vs_summary(self, seed, mask):
        if False:
            while True:
                i = 10
        'Test that summary funcs match normalization functions.\n        '
        rand = np.random.RandomState(seed)
        shape = (10, 5)
        data = rand.randn(*shape)
        data[~mask] = np.nan
        workspace = {F(): data}
        terms = {'demean': F().demean(), 'alt_demean': F() - F().mean(), 'zscore': F().zscore(), 'alt_zscore': (F() - F().mean()) / F().stddev(), 'mean': F().mean(), 'alt_mean': F().sum() / F().notnull_count()}
        result = self.run_terms(terms, initial_workspace=workspace, mask=self.build_mask(ones(shape)))
        assert_equal(result['demean'], result['alt_demean'])
        assert_equal(result['zscore'], result['alt_zscore'])

    @parameter_space(seed=[100, 200, 300], mask=[np.zeros((10, 5), dtype=bool), ones((10, 5), dtype=bool), eye(10, 5, dtype=bool), ~eye(10, 5, dtype=bool)])
    def test_complex_expression(self, seed, mask):
        if False:
            return 10
        rand = np.random.RandomState(seed)
        shape = (10, 5)
        data = rand.randn(*shape)
        data[~mask] = np.nan
        workspace = {F(): data}
        terms = {'rescaled': (F() - F().min()) / (F().max() - F().min())}
        with ignore_nanwarnings():
            mins = as_column(np.nanmin(data, axis=1))
            maxes = as_column(np.nanmax(data, axis=1))
        expected = {'rescaled': (data - mins) / (maxes - mins)}
        self.check_terms(terms, expected, initial_workspace=workspace, mask=self.build_mask(ones(shape)))

    @parameter_space(seed=[40, 41, 42], mask=[np.zeros((10, 5), dtype=bool), ones((10, 5), dtype=bool), eye(10, 5, dtype=bool), ~eye(10, 5, dtype=bool)], mask_mode=('none', 'param', 'root'))
    def test_summaries_after_fillna(self, seed, mask, mask_mode):
        if False:
            for i in range(10):
                print('nop')
        rand = np.random.RandomState(seed)
        shape = (10, 5)
        with_nans = np.where(mask, rand.randn(*shape), np.nan)
        with_minus_1s = np.where(mask, with_nans, -1)
        kwargs = {}
        workspace = {F(): with_nans}
        if mask_mode == 'param':
            kwargs['mask'] = Mask()
            workspace[Mask()] = mask
        terms = {'mean': F().fillna(-1).mean(**kwargs), 'sum': F().fillna(-1).sum(**kwargs), 'median': F().fillna(-1).median(**kwargs), 'min': F().fillna(-1).min(**kwargs), 'max': F().fillna(-1).max(**kwargs), 'stddev': F().fillna(-1).stddev(**kwargs), 'notnull_count': F().fillna(-1).notnull_count(**kwargs)}
        with ignore_nanwarnings():
            if mask_mode == 'none':
                expected_input = with_minus_1s
            else:
                expected_input = with_nans
            expected = {'mean': as_column(np.nanmean(expected_input, axis=1)), 'sum': as_column(np.nansum(expected_input, axis=1)), 'median': as_column(np.nanmedian(expected_input, axis=1)), 'min': as_column(np.nanmin(expected_input, axis=1)), 'max': as_column(np.nanmax(expected_input, axis=1)), 'stddev': as_column(np.nanstd(expected_input, axis=1)), 'notnull_count': as_column((~np.isnan(expected_input)).sum(axis=1))}
        self.assertEqual(set(expected), summary_funcs.names)
        if mask_mode == 'root':
            root_mask = self.build_mask(mask)
        else:
            root_mask = self.build_mask(ones_like(mask))
        self.check_terms(terms=terms, expected=expected, initial_workspace=workspace, mask=root_mask)

    def test_repr(self):
        if False:
            i = 10
            return i + 15

        class MyFactor(CustomFactor):
            window_length = 1
            inputs = ()

            def recursive_repr(self):
                if False:
                    print('Hello World!')
                return 'MyFactor()'
        f = MyFactor()
        for method in summary_funcs.names:
            summarized = getattr(f, method)()
            self.assertEqual(repr(summarized), 'MyFactor().{}()'.format(method))
            self.assertEqual(summarized.recursive_repr(), 'MyFactor().{}()'.format(method))