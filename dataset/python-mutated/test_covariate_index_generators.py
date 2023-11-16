import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries
from darts.dataprocessing.encoders.encoder_base import CovariatesIndexGenerator, FutureCovariatesIndexGenerator, PastCovariatesIndexGenerator
from darts.logging import get_logger
from darts.utils import timeseries_generation as tg
logger = get_logger(__name__)

class TestCovariatesIndexGenerator:
    n_target = 24
    target_time = tg.linear_timeseries(length=n_target, freq='MS')
    cov_time_train = tg.datetime_attribute_timeseries(target_time, attribute='month', cyclic=True)
    cov_time_train_short = cov_time_train[1:]
    target_int = tg.linear_timeseries(length=n_target, start=2)
    cov_int_train = target_int
    cov_int_train_short = cov_int_train[1:]
    input_chunk_length = 12
    output_chunk_length = 6
    n_short = 6
    n_long = 8
    cov_time_inf_short = TimeSeries.from_times_and_values(tg.generate_index(start=target_time.start_time(), length=n_target + n_short, freq=target_time.freq), np.arange(n_target + n_short))
    cov_time_inf_long = TimeSeries.from_times_and_values(tg.generate_index(start=target_time.start_time(), length=n_target + n_long, freq=target_time.freq), np.arange(n_target + n_long))
    cov_int_inf_short = TimeSeries.from_times_and_values(tg.generate_index(start=target_int.start_time(), length=n_target + n_short, freq=target_int.freq), np.arange(n_target + n_short))
    cov_int_inf_long = TimeSeries.from_times_and_values(tg.generate_index(start=target_int.start_time(), length=n_target + n_long, freq=target_int.freq), np.arange(n_target + n_long))

    def helper_test_index_types(self, ig: CovariatesIndexGenerator):
        if False:
            for i in range(10):
                print('nop')
        'test the index type of generated index'
        (idx, _) = ig.generate_train_idx(self.target_time, self.cov_time_train)
        assert isinstance(idx, pd.DatetimeIndex)
        (idx, _) = ig.generate_inference_idx(self.n_short, self.target_time, self.cov_time_inf_short)
        assert isinstance(idx, pd.DatetimeIndex)
        (idx, _) = ig.generate_train_inference_idx(self.n_short, self.target_time, self.cov_time_inf_short)
        assert isinstance(idx, pd.DatetimeIndex)
        (idx, _) = ig.generate_train_idx(self.target_time, None)
        assert isinstance(idx, pd.DatetimeIndex)
        (idx, _) = ig.generate_train_idx(self.target_int, self.cov_int_train)
        assert isinstance(idx, pd.RangeIndex)
        (idx, _) = ig.generate_inference_idx(self.n_short, self.target_int, self.cov_int_inf_short)
        assert isinstance(idx, pd.RangeIndex)
        (idx, _) = ig.generate_train_inference_idx(self.n_short, self.target_int, self.cov_int_inf_short)
        assert isinstance(idx, pd.RangeIndex)
        (idx, _) = ig.generate_train_idx(self.target_int, None)
        assert isinstance(idx, pd.RangeIndex)

    def helper_test_index_generator_train(self, ig: CovariatesIndexGenerator):
        if False:
            return 10
        "\n        If covariates are given, the index generators should return the covariate series' index.\n        If covariates are not given, the index generators should return the target series' index.\n        "
        (idx, _) = ig.generate_train_idx(self.target_time, self.cov_time_train)
        assert idx.equals(self.cov_time_train.time_index)
        (idx, _) = ig.generate_train_idx(self.target_time, self.cov_time_train_short)
        assert idx.equals(self.cov_time_train_short.time_index)
        (idx, _) = ig.generate_train_idx(self.target_time, None)
        assert idx[0] == self.target_time.start_time()
        if isinstance(ig, PastCovariatesIndexGenerator):
            assert idx[-1] == self.target_time.end_time() - self.output_chunk_length * self.target_time.freq
        else:
            assert idx[-1] == self.target_time.end_time()
        (idx, _) = ig.generate_train_idx(self.target_int, self.cov_int_train)
        assert idx.equals(self.cov_int_train.time_index)
        (idx, _) = ig.generate_train_idx(self.target_int, self.cov_int_train_short)
        assert idx.equals(self.cov_int_train_short.time_index)
        (idx, _) = ig.generate_train_idx(self.target_int, None)
        assert idx[0] == self.target_int.start_time()
        if isinstance(ig, PastCovariatesIndexGenerator):
            assert idx[-1] == self.target_int.end_time() - self.output_chunk_length * self.target_int.freq
        else:
            assert idx[-1] == self.target_int.end_time()

    def helper_test_index_generator_inference(self, ig, is_past=False):
        if False:
            i = 10
            return i + 15
        '\n        For prediction (`n` is given) with past covariates we have to distinguish between two cases:\n        1)  if past covariates are given, we can use them as reference\n        2)  if past covariates are missing, we need to generate a time index that starts `input_chunk_length`\n            before the end of `target` and ends `max(0, n - output_chunk_length)` after the end of `target`\n\n        For prediction (`n` is given) with future covariates we have to distinguish between two cases:\n        1)  if future covariates are given, we can use them as reference\n        2)  if future covariates are missing, we need to generate a time index that starts `input_chunk_length`\n            before the end of `target` and ends `max(n, output_chunk_length)` after the end of `target`\n        '
        (idx, _) = ig.generate_inference_idx(self.n_short, self.target_time, None)
        if is_past:
            n_out = self.input_chunk_length
            last_idx = self.target_time.end_time()
        else:
            n_out = self.input_chunk_length + self.output_chunk_length
            last_idx = self.cov_time_inf_short.end_time()
        assert len(idx) == n_out
        assert idx[-1] == last_idx
        (idx, _) = ig.generate_inference_idx(self.n_long, self.target_time, None)
        if is_past:
            n_out = self.input_chunk_length + self.n_long - self.output_chunk_length
            last_idx = self.target_time.end_time() + (self.n_long - self.output_chunk_length) * self.target_time.freq
        else:
            n_out = self.input_chunk_length + self.n_long
            last_idx = self.cov_time_inf_long.end_time()
        assert len(idx) == n_out
        assert idx[-1] == last_idx
        (idx, _) = ig.generate_inference_idx(self.n_short, self.target_time, self.cov_time_inf_short)
        assert idx.equals(self.cov_time_inf_short.time_index)
        (idx, _) = ig.generate_inference_idx(self.n_long, self.target_time, self.cov_time_inf_long)
        assert idx.equals(self.cov_time_inf_long.time_index)
        (idx, _) = ig.generate_inference_idx(self.n_short, self.target_int, self.cov_int_inf_short)
        assert idx.equals(self.cov_int_inf_short.time_index)
        (idx, _) = ig.generate_inference_idx(self.n_long, self.target_int, self.cov_int_inf_long)
        assert idx.equals(self.cov_int_inf_long.time_index)

    def helper_test_index_generator_creation(self, ig_cls, is_past=False):
        if False:
            print('Hello World!')
        with pytest.raises(ValueError):
            _ = ig_cls(output_chunk_length=3)
        with pytest.raises(ValueError):
            _ = ig_cls(output_chunk_length=3, lags_covariates=[-1])
        _ = ig_cls(input_chunk_length=3, output_chunk_length=3)
        _ = ig_cls(input_chunk_length=3, output_chunk_length=3, lags_covariates=[-1])
        if not is_past:
            _ = ig_cls()

    def test_past_index_generator_creation(self):
        if False:
            while True:
                i = 10
        self.helper_test_index_generator_creation(ig_cls=PastCovariatesIndexGenerator, is_past=True)
        with pytest.raises(ValueError):
            _ = PastCovariatesIndexGenerator(1, 1, lags_covariates=[-1, 1])
        with pytest.raises(ValueError):
            _ = PastCovariatesIndexGenerator(1, 1, lags_covariates=[0, -1])
        (min_lag, max_lag) = (-2, -1)
        ig = PastCovariatesIndexGenerator(1, 1, lags_covariates=[min_lag, max_lag])
        assert ig.shift_start == min_lag + 1
        assert ig.shift_end == max_lag + 1
        (min_lag, max_lag) = (-1, -1)
        ig = PastCovariatesIndexGenerator(1, 1, lags_covariates=[min_lag, max_lag])
        assert ig.shift_start == min_lag + 1
        assert ig.shift_end == max_lag + 1
        (min_lag, max_lag) = (-10, -3)
        ig = PastCovariatesIndexGenerator(1, 1, lags_covariates=[-5, min_lag, max_lag, -4])
        assert ig.shift_start == min_lag + 1
        assert ig.shift_end == max_lag + 1

    def test_future_index_generator_creation(self):
        if False:
            print('Hello World!')
        self.helper_test_index_generator_creation(ig_cls=FutureCovariatesIndexGenerator, is_past=False)
        (min_lag, max_lag) = (-2, -1)
        ig = FutureCovariatesIndexGenerator(1, 1, lags_covariates=[min_lag, max_lag])
        assert ig.shift_start == min_lag + 1
        assert ig.shift_end == max_lag + 1
        (min_lag, max_lag) = (-1, -1)
        ig = FutureCovariatesIndexGenerator(1, 1, lags_covariates=[min_lag, max_lag])
        assert ig.shift_start == min_lag + 1
        assert ig.shift_end == max_lag + 1
        (min_lag, max_lag) = (-2, 1)
        ig = FutureCovariatesIndexGenerator(1, 1, lags_covariates=[min_lag, max_lag])
        assert ig.shift_start == min_lag + 1
        assert ig.shift_end == max_lag + 1
        (min_lag, max_lag) = (-10, 5)
        ig = FutureCovariatesIndexGenerator(1, 1, lags_covariates=[-5, min_lag, max_lag, -1])
        assert ig.shift_start == min_lag + 1
        assert ig.shift_end == max_lag + 1

    def test_past_index_generator(self):
        if False:
            i = 10
            return i + 15
        ig = PastCovariatesIndexGenerator(self.input_chunk_length, self.output_chunk_length)
        self.helper_test_index_types(ig)
        self.helper_test_index_generator_train(ig)
        self.helper_test_index_generator_inference(ig, is_past=True)

    def test_past_index_generator_with_lags(self):
        if False:
            for i in range(10):
                print('nop')
        icl = self.input_chunk_length
        ocl = self.output_chunk_length
        target = self.target_time

        def test_routine_train(self, icl, ocl, min_lag, max_lag, start_expected, end_expected):
            if False:
                print('Hello World!')
            idxg = PastCovariatesIndexGenerator(icl, ocl, lags_covariates=[min_lag, max_lag])
            (idx, target_end) = idxg.generate_train_idx(target, None)
            assert idx[0] == pd.Timestamp(start_expected)
            assert idx[-1] == pd.Timestamp(end_expected)
            assert target_end == target.end_time()
            (idx, target_end) = idxg.generate_train_idx(target, self.cov_time_train)
            assert idx.equals(self.cov_time_train.time_index)
            assert target_end == target.end_time()
            return idxg

        def test_routine_inf(self, idxg, n, start_expected, end_expected):
            if False:
                print('Hello World!')
            (idx, target_end) = idxg.generate_inference_idx(n, target, None)
            assert idx[0] == pd.Timestamp(start_expected)
            assert idx[-1] == pd.Timestamp(end_expected)
            assert target_end == target.end_time()
            (idx, target_end) = idxg.generate_inference_idx(n, target, self.cov_time_inf_short)
            assert idx.equals(self.cov_time_inf_short.time_index)
            assert target_end == target.end_time()

        def test_routine_train_inf(self, idxg, n, start_expected, end_expected):
            if False:
                return 10
            (idx, target_end) = idxg.generate_train_inference_idx(n, target, None)
            assert idx[0] == pd.Timestamp(start_expected)
            assert idx[-1] == pd.Timestamp(end_expected)
            assert target_end == target.end_time()
            (idx, target_end) = idxg.generate_train_inference_idx(n, target, self.cov_time_inf_short)
            assert idx.equals(self.cov_time_inf_short.time_index)
            assert target_end == target.end_time()
        min_lag = -12
        max_lag = -1
        expected_start_train = '2000-01-01'
        expected_end_train = '2001-06-01'
        ig = test_routine_train(self, icl, ocl, min_lag, max_lag, expected_start_train, expected_end_train)
        self.helper_test_index_types(ig)
        self.helper_test_index_generator_train(ig)
        self.helper_test_index_generator_inference(ig, is_past=True)
        expected_start_inf = '2001-01-01'
        expected_end_inf = '2001-12-01'
        test_routine_inf(self, ig, 1, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, 1, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, ocl, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl + 1, expected_start_inf, '2002-01-01')
        test_routine_train_inf(self, ig, ocl + 1, expected_start_train, '2002-01-01')
        (min_lag, max_lag) = (-11, -1)
        expected_start_train = '2000-02-01'
        expected_end_train = '2001-06-01'
        ig = test_routine_train(self, icl, ocl, min_lag, max_lag, expected_start_train, expected_end_train)
        expected_start_inf = '2001-02-01'
        expected_end_inf = '2001-12-01'
        test_routine_inf(self, ig, 1, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, 1, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, ocl, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl + 1, expected_start_inf, '2002-01-01')
        test_routine_train_inf(self, ig, ocl + 1, expected_start_train, '2002-01-01')
        (min_lag, max_lag) = (-13, -1)
        expected_start_train = '1999-12-01'
        expected_end_train = '2001-06-01'
        ig = test_routine_train(self, icl, ocl, min_lag, max_lag, expected_start_train, expected_end_train)
        expected_start_inf = '2000-12-01'
        expected_end_inf = '2001-12-01'
        test_routine_inf(self, ig, 1, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, 1, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, ocl, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl + 1, expected_start_inf, '2002-01-01')
        test_routine_train_inf(self, ig, ocl + 1, expected_start_train, '2002-01-01')
        (min_lag, max_lag) = (-13, -2)
        expected_start_train = '1999-12-01'
        expected_end_train = '2001-05-01'
        ig = test_routine_train(self, icl, ocl, min_lag, max_lag, expected_start_train, expected_end_train)
        expected_start_inf = '2000-12-01'
        expected_end_inf = '2001-11-01'
        test_routine_inf(self, ig, 1, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, 1, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, ocl, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl + 1, expected_start_inf, '2001-12-01')
        test_routine_train_inf(self, ig, ocl + 1, expected_start_train, '2001-12-01')

    def test_future_index_generator(self):
        if False:
            for i in range(10):
                print('nop')
        ig = FutureCovariatesIndexGenerator(self.input_chunk_length, self.output_chunk_length)
        self.helper_test_index_types(ig)
        self.helper_test_index_generator_train(ig)
        self.helper_test_index_generator_inference(ig, is_past=False)

    def test_future_index_generator_with_lags(self):
        if False:
            print('Hello World!')
        icl = self.input_chunk_length
        ocl = self.output_chunk_length
        target = self.target_time

        def test_routine_train(self, icl, ocl, min_lag, max_lag, start_expected, end_expected):
            if False:
                for i in range(10):
                    print('nop')
            idxg = FutureCovariatesIndexGenerator(icl, ocl, lags_covariates=[min_lag, max_lag])
            (idx, target_end) = idxg.generate_train_idx(target, None)
            assert idx[0] == pd.Timestamp(start_expected)
            assert idx[-1] == pd.Timestamp(end_expected)
            assert target_end == target.end_time()
            (idx, target_end) = idxg.generate_train_idx(target, self.cov_time_train)
            assert idx.equals(self.cov_time_train.time_index)
            assert target_end == target.end_time()
            return idxg

        def test_routine_inf(self, idxg, n, start_expected, end_expected):
            if False:
                i = 10
                return i + 15
            (idx, target_end) = idxg.generate_inference_idx(n, target, None)
            assert idx[0] == pd.Timestamp(start_expected)
            assert idx[-1] == pd.Timestamp(end_expected)
            assert target_end == target.end_time()
            (idx, target_end) = idxg.generate_inference_idx(n, target, self.cov_time_inf_short)
            assert idx.equals(self.cov_time_inf_short.time_index)
            assert target_end == target.end_time()

        def test_routine_train_inf(self, idxg, n, start_expected, end_expected):
            if False:
                return 10
            (idx, target_end) = idxg.generate_train_inference_idx(n, target, None)
            assert idx[0] == pd.Timestamp(start_expected)
            assert idx[-1] == pd.Timestamp(end_expected)
            assert target_end == target.end_time()
            (idx, target_end) = idxg.generate_train_inference_idx(n, target, self.cov_time_inf_short)
            assert idx.equals(self.cov_time_inf_short.time_index)
            assert target_end == target.end_time()
        (min_lag, max_lag) = (-11, -1)
        expected_start_train = '2000-02-01'
        expected_end_train = '2001-06-01'
        ig = test_routine_train(self, icl, ocl, min_lag, max_lag, expected_start_train, expected_end_train)
        expected_start_inf = '2001-02-01'
        expected_end_inf = '2001-12-01'
        test_routine_inf(self, ig, 1, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, 1, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, ocl, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl + 1, expected_start_inf, '2002-01-01')
        test_routine_train_inf(self, ig, ocl + 1, expected_start_train, '2002-01-01')
        (min_lag, max_lag) = (-13, -1)
        expected_start_train = '1999-12-01'
        expected_end_train = '2001-06-01'
        ig = test_routine_train(self, icl, ocl, min_lag, max_lag, expected_start_train, expected_end_train)
        expected_start_inf = '2000-12-01'
        expected_end_inf = '2001-12-01'
        test_routine_inf(self, ig, 1, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, 1, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, ocl, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl + 1, expected_start_inf, '2002-01-01')
        test_routine_train_inf(self, ig, ocl + 1, expected_start_train, '2002-01-01')
        (min_lag, max_lag) = (-13, -2)
        expected_start_train = '1999-12-01'
        expected_end_train = '2001-05-01'
        ig = test_routine_train(self, icl, ocl, min_lag, max_lag, expected_start_train, expected_end_train)
        expected_start_inf = '2000-12-01'
        expected_end_inf = '2001-11-01'
        test_routine_inf(self, ig, 1, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, 1, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, ocl, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl + 1, expected_start_inf, '2001-12-01')
        test_routine_train_inf(self, ig, ocl + 1, expected_start_train, '2001-12-01')
        min_lag = -12
        max_lag = 5
        expected_start_train = '2000-01-01'
        expected_end_train = '2001-12-01'
        ig = test_routine_train(self, icl, ocl, min_lag, max_lag, expected_start_train, expected_end_train)
        self.helper_test_index_types(ig)
        self.helper_test_index_generator_train(ig)
        self.helper_test_index_generator_inference(ig, is_past=False)
        expected_start_inf = '2001-01-01'
        expected_end_inf = '2002-06-01'
        test_routine_inf(self, ig, 1, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, 1, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, ocl, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl + 1, expected_start_inf, '2002-07-01')
        test_routine_train_inf(self, ig, ocl + 1, expected_start_train, '2002-07-01')
        (min_lag, max_lag) = (-12, 0)
        expected_start_train = '2000-01-01'
        expected_end_train = '2001-07-01'
        ig = test_routine_train(self, icl, ocl, min_lag, max_lag, expected_start_train, expected_end_train)
        expected_start_inf = '2001-01-01'
        expected_end_inf = '2002-01-01'
        test_routine_inf(self, ig, 1, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, 1, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, ocl, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl + 1, expected_start_inf, '2002-02-01')
        test_routine_train_inf(self, ig, ocl + 1, expected_start_train, '2002-02-01')
        (min_lag, max_lag) = (-12, 17)
        expected_start_train = '2000-01-01'
        expected_end_train = '2002-12-01'
        ig = test_routine_train(self, icl, ocl, min_lag, max_lag, expected_start_train, expected_end_train)
        expected_start_inf = '2001-01-01'
        expected_end_inf = '2003-06-01'
        test_routine_inf(self, ig, 1, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, 1, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, ocl, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl + 1, expected_start_inf, '2003-07-01')
        test_routine_train_inf(self, ig, ocl + 1, expected_start_train, '2003-07-01')
        (min_lag, max_lag) = (0, 2)
        expected_start_train = '2001-01-01'
        expected_end_train = '2001-09-01'
        ig = test_routine_train(self, icl, ocl, min_lag, max_lag, expected_start_train, expected_end_train)
        expected_start_inf = '2002-01-01'
        expected_end_inf = '2002-03-01'
        test_routine_inf(self, ig, 1, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, 1, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, ocl, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl + 1, expected_start_inf, '2002-04-01')
        test_routine_train_inf(self, ig, ocl + 1, expected_start_train, '2002-04-01')
        (min_lag, max_lag) = (0, 17)
        expected_start_train = '2001-01-01'
        expected_end_train = '2002-12-01'
        ig = test_routine_train(self, icl, ocl, min_lag, max_lag, expected_start_train, expected_end_train)
        expected_start_inf = '2002-01-01'
        expected_end_inf = '2003-06-01'
        test_routine_inf(self, ig, 1, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, 1, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, ocl, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl + 1, expected_start_inf, '2003-07-01')
        test_routine_train_inf(self, ig, ocl + 1, expected_start_train, '2003-07-01')
        (min_lag, max_lag) = (-13, 17)
        expected_start_train = '1999-12-01'
        expected_end_train = '2002-12-01'
        ig = test_routine_train(self, icl, ocl, min_lag, max_lag, expected_start_train, expected_end_train)
        expected_start_inf = '2000-12-01'
        expected_end_inf = '2003-06-01'
        test_routine_inf(self, ig, 1, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, 1, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl, expected_start_inf, expected_end_inf)
        test_routine_train_inf(self, ig, ocl, expected_start_train, expected_end_inf)
        test_routine_inf(self, ig, ocl + 1, expected_start_inf, '2003-07-01')
        test_routine_train_inf(self, ig, ocl + 1, expected_start_train, '2003-07-01')

    def test_future_index_generator_local(self):
        if False:
            for i in range(10):
                print('nop')
        freq = self.target_time.freq
        target = self.target_time
        idxg = FutureCovariatesIndexGenerator()
        (idx, _) = idxg.generate_train_idx(target=target, covariates=None)
        assert idx.equals(target.time_index)
        (idx, _) = idxg.generate_train_idx(target=target, covariates=self.cov_time_train)
        assert idx.equals(self.cov_time_train.time_index)
        n = 10
        (idx, _) = idxg.generate_inference_idx(n=n, target=target, covariates=None)
        assert idx.freq == freq
        assert idx[0] == target.end_time() + 1 * freq
        assert idx[-1] == target.end_time() + n * freq
        (idx, _) = idxg.generate_inference_idx(n=n, target=target, covariates=self.cov_int_inf_short)
        assert idx.equals(self.cov_int_inf_short.time_index)