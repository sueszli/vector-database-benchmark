import warnings
from itertools import product
from typing import Sequence
import pandas as pd
import pytest
from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.utils.data.tabularization import _get_feature_times
from darts.utils.timeseries_generation import linear_timeseries

class TestGetFeatureTimes:
    """
    Tests `_get_feature_times` function defined in `darts.utils.data.tabularization`. There
    are broadly two 'groups' of tests defined in this module:
        1. 'Generated Test Cases': these test that `_get_feature_times` produces the same outputs
        as a simplified implementation of this same function. For these tests, the 'correct answer' is not
        directly specified; instead, it is generated from a set of input parameters using the set of simplified
        functions. The rationale behind this approach is that it allows for many different combinations of input
        values to be effortlessly tested. The drawback of this, however, is that the correctness of these tests
        assumes that the simplified functions have been implemented correctly - if this isn't the case, then these
        tests are not to be trusted. In saying this, these simplified functions are significantly easier to
        understand and debug than the `create_lagged_prediction_data` function they're helping to test.
        2. 'Specified Test Cases': these test that `_get_feature_times` returns an exactly specified output; these
        specified outputs are *not* 'generated' by another function. Although these 'specified' test cases tend to
        be simpler and less extensive than the 'generated' test cases, their correctness does not assume the correct
        implementation of any other function.
    """

    @staticmethod
    def get_feature_times_target_training(target_series: TimeSeries, lags: Sequence[int], output_chunk_length: int):
        if False:
            return 10
        "\n        Helper function that returns all the times within `target_series` that can be used to\n        create features and labels for training.\n\n        More specifically:\n            - The first `max_lag = -min(lags)` times are excluded, since these times\n            have fewer than `max_lag` values after them, which means that we can't create\n            features for these times.\n            - The last `output_chunk_length - 1` times are excluded, since these times don't\n            have `(output_chunk_length - 1)` values ahead of them and, therefore, we can't\n            create labels for these times.\n        "
        times = target_series.time_index
        max_lag = -min(lags)
        times = times[max_lag:]
        if output_chunk_length > 1:
            times = times[:-output_chunk_length + 1]
        return times

    @staticmethod
    def get_feature_times_past(past_covariates: TimeSeries, past_covariates_lags: Sequence[int]) -> pd.Index:
        if False:
            print('Hello World!')
        "\n        Helper function that returns all the times within `past_covariates` that can be used to\n        create features for training or prediction.\n\n        Unlike the `target_series` during training, features can be constructed for times that\n        occur after the end of `past_covariates`; this is because:\n            1. We don't need to have all the `past_covariates` values up to time `t` to construct\n            a feature for this time; instead, we only need to have the values from time `t - min_lag`\n            to `t - max_lag`, where `min_lag = -max(past_covariates_lags)` and\n            `max_lag = -min(past_covariates_lags)`. In other words, the latest feature we can create\n            for `past_covariates` occurs at `past_covariates.end_time() + min_lag * past_covariates.freq`.\n            2. We don't need to use the values of `past_covariates` to construct labels, so we're able\n            to create a feature for time `t` without having to worry about whether we can construct\n            a corresponding label for this time.\n        "
        times = past_covariates.time_index
        min_lag = -max(past_covariates_lags)
        times = times.union([times[-1] + i * past_covariates.freq for i in range(1, min_lag + 1)])
        max_lag = -min(past_covariates_lags)
        times = times[max_lag:]
        return times

    @staticmethod
    def get_feature_times_target_prediction(target_series: TimeSeries, lags: Sequence[int]):
        if False:
            for i in range(10):
                print('nop')
        "\n        Helper function that returns all the times within `target_series` that can be used to\n        create features for prediction.\n\n        Since we don't need to worry about creating labels for prediction data, the process\n        of constructing prediction features using the `target_series` is identical to\n        constructing features for the `past_covariates` series.\n        "
        return TestGetFeatureTimes.get_feature_times_past(target_series, lags)

    @staticmethod
    def get_feature_times_future(future_covariates: TimeSeries, future_covariates_lags: Sequence[int]) -> pd.Index:
        if False:
            i = 10
            return i + 15
        "\n        Helper function called by `_get_feature_times` that extracts all of the times within\n        `future_covariates` that can be used to create features for training or prediction.\n\n        Unlike the lag values for `target_series` and `past_covariates`, the values in\n        `future_covariates_lags` can be negative, zero, or positive. This means that\n        `min_lag = -max(future_covariates_lags)` and `max_lag = -min(future_covariates_lags)`\n        are *not* guaranteed to be positive here: they could be negative (corresponding to\n        a positive value in `future_covariates_lags`), zero, or positive (corresponding to\n        a negative value in `future_covariates_lags`). With that being said, the relationship\n        `min_lag <= max_lag` always holds.\n\n        Consequently, we need to consider three scenarios when finding feature times\n        for `future_covariates`:\n            1. Both `min_lag` and `max_lag` are positive, which indicates that all of\n            the lag values in `future_covariates_lags` are negative (i.e. only values before\n            time `t` are used to create a feature from time `t`). In this case, `min_lag`\n            and `max_lag` correspond to the smallest magnitude and largest magnitude *negative*\n            lags in `future_covariates_lags` respectively. This means we *can* create features for\n            times that extend beyond the end of `future_covariates`; additionally, we're unable\n            to create features for the first `min_lag` times (see docstring for `get_feature_times_past`).\n            2. Both `min_lag` and `max_lag` are non-positive. In this case, `abs(min_lag)` and `abs(max_lag)`\n            correspond to the largest and smallest magnitude lags in `future_covariates_lags` respectively;\n            note that, somewhat confusingly, `abs(max_lag) <= abs(min_lag)` here. This means that we *can* create f\n            features for times that occur before the start of `future_covariates`; the reasoning for this is\n            basically the inverse of Case 1 (i.e. we only need to know the values from times `t + abs(max_lag)`\n            to `t + abs(min_lag)` to create a feature for time `t`). Additionally, we're unable to create features\n            for the last `abs(min_lag)` times in the series, since these times do not have `abs(min_lag)` values\n            after them.\n            3. `min_lag` is non-positive (i.e. zero or negative), but `max_lag` is positive. In this case,\n            `abs(min_lag)` is the magnitude of the largest *non-negative* lag value in `future_covariates_lags`\n            and `max_lag` is the largest *negative* lag value in `future_covariates_lags`. This means that we\n            *cannot* create features for times that occur before the start of `future_covariates`, nor for\n            times that occur after the end of `future_covariates`; this is because we must have access to\n            both times before *and* after time `t` to create a feature for this time, which clearly can't\n            be acieved for times extending before the start or after the end of the series. Moreover,\n            we must exclude the first `max_lag` times and the last `abs(min_lag)` times, since these\n            times do not have enough values before or after them respectively.\n        "
        times = future_covariates.time_index
        min_lag = -max(future_covariates_lags)
        max_lag = -min(future_covariates_lags)
        if min_lag > 0 and max_lag > 0:
            times = times.union([times[-1] + i * future_covariates.freq for i in range(1, min_lag + 1)])
            times = times[max_lag:]
        elif min_lag <= 0 and max_lag <= 0:
            times = times.union([times[0] - i * future_covariates.freq for i in range(1, abs(max_lag) + 1)])
            times = times[:min_lag] if min_lag != 0 else times
        elif min_lag <= 0 and max_lag > 0:
            times = times[:min_lag] if min_lag != 0 else times
            times = times[max_lag:]
        else:
            error_msg = f'Unexpected `future_covariates_lags` case encountered: `min_lag` is positive, but `max_lag` is negative. Caused by `future_covariates_lags = {future_covariates_lags}`.'
            error = ValueError(error_msg)
            raise_log(error, get_logger(__name__))
        return times
    target_lag_combos = lags_past_combos = ([-1], [-2, -1], [-6, -4, -3], [-4, -6, -3])
    lags_future_combos = (*target_lag_combos, [0], [0, 1], [1, 3], [-2, 2])
    ocl_combos = (1, 2, 5, 10)

    def test_feature_times_training_range_idx(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests that `_get_feature_times` produces the same `times` output as\n        that generated by using the various `get_feature_times_*` helper\n        functions defined in this module when `is_training = True`. Consistency\n        is checked over all of the combinations of parameter values specified by\n        `self.target_lag_combos`, `self.lags_past_combos`, `self.lags_future_combos`\n        and `self.max_samples_per_ts_combos`. This particular test uses timeseries\n        with range time indices.\n        '
        target = linear_timeseries(start=1, length=20, freq=1)
        past = linear_timeseries(start=2, length=25, freq=2)
        future = linear_timeseries(start=3, length=30, freq=3)
        for (lags, lags_past, lags_future, ocl) in product(self.target_lag_combos, self.lags_past_combos, self.lags_future_combos, self.ocl_combos):
            feature_times = _get_feature_times(target_series=target, past_covariates=past, future_covariates=future, lags=lags, lags_past_covariates=lags_past, lags_future_covariates=lags_future, output_chunk_length=ocl, is_training=True)
            target_expected = self.get_feature_times_target_training(target, lags, ocl)
            past_expected = self.get_feature_times_past(past, lags_past)
            future_expected = self.get_feature_times_future(future, lags_future)
            assert target_expected.equals(feature_times[0])
            assert past_expected.equals(feature_times[1])
            assert future_expected.equals(feature_times[2])

    def test_feature_times_training_datetime_idx(self):
        if False:
            return 10
        '\n        Tests that `_get_feature_times` produces the same `times` output as\n        that generated by using the various `get_feature_times_*` helper\n        functions defined in this module when `is_training = True`. Consistency\n        is checked over all of the combinations of parameter values specified by\n        `self.target_lag_combos`, `self.lags_past_combos`, `self.lags_future_combos`\n        and `self.max_samples_per_ts_combos`. This particular test uses timeseries\n        with datetime time indices.\n        '
        target = linear_timeseries(start=pd.Timestamp('1/1/2000'), length=20, freq='1d')
        past = linear_timeseries(start=pd.Timestamp('1/2/2000'), length=25, freq='2d')
        future = linear_timeseries(start=pd.Timestamp('1/3/2000'), length=30, freq='3d')
        for (lags, lags_past, lags_future, ocl) in product(self.target_lag_combos, self.lags_past_combos, self.lags_future_combos, self.ocl_combos):
            feature_times = _get_feature_times(target_series=target, past_covariates=past, future_covariates=future, lags=lags, lags_past_covariates=lags_past, lags_future_covariates=lags_future, output_chunk_length=ocl, is_training=True)
            target_expected = self.get_feature_times_target_training(target, lags, ocl)
            past_expected = self.get_feature_times_past(past, lags_past)
            future_expected = self.get_feature_times_future(future, lags_future)
            assert target_expected.equals(feature_times[0])
            assert past_expected.equals(feature_times[1])
            assert future_expected.equals(feature_times[2])

    def test_feature_times_prediction_range_idx(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that `_get_feature_times` produces the same `times` output as\n        that generated by using the various `get_feature_times_*` helper\n        functions defined in this module when `is_training = False` (i.e. when creaiting\n        prediction data). Consistency is checked over all of the combinations of parameter\n        values specified by `self.target_lag_combos`, `self.lags_past_combos`,\n        `self.lags_future_combos` and `self.max_samples_per_ts_combos`. This particular test\n        uses timeseries with range time indices.\n        '
        target = linear_timeseries(start=1, length=20, freq=1)
        past = linear_timeseries(start=2, length=25, freq=2)
        future = linear_timeseries(start=3, length=30, freq=3)
        for (lags, lags_past, lags_future) in product(self.target_lag_combos, self.lags_past_combos, self.lags_future_combos):
            feature_times = _get_feature_times(target_series=target, past_covariates=past, future_covariates=future, lags=lags, lags_past_covariates=lags_past, lags_future_covariates=lags_future, is_training=False)
            target_expected = self.get_feature_times_target_prediction(target, lags)
            past_expected = self.get_feature_times_past(past, lags_past)
            future_expected = self.get_feature_times_future(future, lags_future)
            assert target_expected.equals(feature_times[0])
            assert past_expected.equals(feature_times[1])
            assert future_expected.equals(feature_times[2])

    def test_feature_times_prediction_datetime_idx(self):
        if False:
            return 10
        '\n        Tests that `_get_feature_times` produces the same `times` output as\n        that generated by using the various `get_feature_times_*` helper\n        functions defined in this module when `is_training = False` (i.e. when creaiting\n        prediction data). Consistency is checked over all of the combinations of parameter\n        values specified by `self.target_lag_combos`, `self.lags_past_combos`,\n        `self.lags_future_combos` and `self.max_samples_per_ts_combos`. This particular test\n        uses timeseries with datetime time indices.\n        '
        target = linear_timeseries(start=pd.Timestamp('1/1/2000'), length=20, freq='1d')
        past = linear_timeseries(start=pd.Timestamp('1/2/2000'), length=25, freq='2d')
        future = linear_timeseries(start=pd.Timestamp('1/3/2000'), length=30, freq='3d')
        for (lags, lags_past, lags_future) in product(self.target_lag_combos, self.lags_past_combos, self.lags_future_combos):
            feature_times = _get_feature_times(target_series=target, past_covariates=past, future_covariates=future, lags=lags, lags_past_covariates=lags_past, lags_future_covariates=lags_future, is_training=False)
            target_expected = self.get_feature_times_target_prediction(target, lags)
            past_expected = self.get_feature_times_past(past, lags_past)
            future_expected = self.get_feature_times_future(future, lags_future)
            assert target_expected.equals(feature_times[0])
            assert past_expected.equals(feature_times[1])
            assert future_expected.equals(feature_times[2])

    def test_feature_times_output_chunk_length_range_idx(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests that the last feature time for the `target_series`\n        returned by `_get_feature_times` corresponds to\n        `output_chunk_length - 1` timesteps *before* the end of\n        the target series; this is the last time point in\n        `target_series` which has enough values in front of it\n        to create a label. This particular test uses range time\n        index series to check this behaviour.\n        '
        target = linear_timeseries(start=0, length=20, freq=2)
        for ocl in (1, 2, 3, 4, 5):
            feature_times = _get_feature_times(target_series=target, lags=[-2, -3, -5], output_chunk_length=ocl, is_training=True)
            assert feature_times[0][-1] == target.end_time() - target.freq * (ocl - 1)

    def test_feature_times_output_chunk_length_datetime_idx(self):
        if False:
            print('Hello World!')
        '\n        Tests that the last feature time for the `target_series`\n        returned by `_get_feature_times` when `is_training = True`\n        corresponds to the time that is `(output_chunk_length - 1)`\n        timesteps *before* the end of the target series; this is the\n        last time point in `target_series` which has enough values\n        in front of it to create a label. This particular test uses\n        datetime time index series to check this behaviour.\n        '
        target = linear_timeseries(start=pd.Timestamp('1/1/2000'), length=20, freq='2d')
        for ocl in (1, 2, 3, 4, 5):
            feature_times = _get_feature_times(target_series=target, lags=[-2, -3, -5], output_chunk_length=ocl, is_training=True)
            assert feature_times[0][-1] == target.end_time() - target.freq * (ocl - 1)

    def test_feature_times_lags_range_idx(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that the first feature time for the `target_series`\n        returned by `_get_feature_times` corresponds to the time\n        that is `max_lags` timesteps *after* the start of\n        the target series; this is the first time point in\n        `target_series` which has enough values in preceeding it\n        to create a feature. This particular test uses range time\n        index series to check this behaviour.\n        '
        target = linear_timeseries(start=0, length=20, freq=2)
        for is_training in (False, True):
            for max_lags in (-1, -2, -3, -4, -5):
                feature_times = _get_feature_times(target_series=target, lags=[-1, max_lags], is_training=is_training)
                assert feature_times[0][0] == target.start_time() + target.freq * abs(max_lags)

    def test_feature_times_lags_datetime_idx(self):
        if False:
            while True:
                i = 10
        '\n        Tests that the first feature time for the `target_series`\n        returned by `_get_feature_times` corresponds to the time\n        that is `max_lags` timesteps *after* the start of\n        the target series; this is the first time point in\n        `target_series` which has enough values in preceeding it\n        to create a feature. This particular test uses datetime time\n        index series to check this behaviour.\n        '
        target = linear_timeseries(start=pd.Timestamp('1/1/2000'), length=20, freq='2d')
        for is_training in (False, True):
            for max_lags in (-1, -2, -3, -4, -5):
                feature_times = _get_feature_times(target_series=target, lags=[-1, max_lags], is_training=is_training)
                assert feature_times[0][0] == target.start_time() + target.freq * abs(max_lags)

    def test_feature_times_training_single_time_range_idx(self):
        if False:
            print('Hello World!')
        '\n        Tests that `_get_feature_times` correctly handles case where only\n        a single time can be used to create training features and labels.\n        This particular test uses range index timeseries.\n        '
        target = linear_timeseries(start=0, length=2, freq=1)
        lags = [-1]
        feature_times = _get_feature_times(target_series=target, output_chunk_length=1, lags=lags, is_training=True)
        assert len(feature_times[0]) == 1
        assert feature_times[0][0] == 1
        future = linear_timeseries(start=2, length=1, freq=2)
        future_lags = [-2]
        feature_times = _get_feature_times(target_series=target, future_covariates=future, output_chunk_length=1, lags=lags, lags_future_covariates=future_lags, is_training=True)
        assert len(feature_times[0]) == 1
        assert feature_times[0][0] == 1
        assert len(feature_times[2]) == 1
        assert feature_times[2][0] == 6

    def test_feature_times_training_single_time_datetime_idx(self):
        if False:
            while True:
                i = 10
        '\n        Tests that `_get_feature_times` correctly handles case where only\n        a single time can be used to create training features and labels.\n        This particular test uses datetime index timeseries.\n        '
        target = linear_timeseries(start=pd.Timestamp('1/1/2000'), length=2, freq='d')
        lags = [-1]
        feature_times = _get_feature_times(target_series=target, output_chunk_length=1, lags=lags, is_training=True)
        assert len(feature_times[0]) == 1
        assert feature_times[0][0] == pd.Timestamp('1/2/2000')
        future = linear_timeseries(start=pd.Timestamp('1/2/2000'), length=1, freq='2d')
        future_lags = [-2]
        feature_times = _get_feature_times(target_series=target, future_covariates=future, output_chunk_length=1, lags=lags, lags_future_covariates=future_lags, is_training=True)
        assert len(feature_times[0]) == 1
        assert feature_times[0][0] == pd.Timestamp('1/2/2000')
        assert len(feature_times[2]) == 1
        assert feature_times[2][0] == pd.Timestamp('1/6/2000')

    def test_feature_times_prediction_single_time_range_idx(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that `_get_feature_times` correctly handles case where only\n        a single time can be used to create prediction features.\n        This particular test uses range index timeseries.\n        '
        target = linear_timeseries(start=0, length=1, freq=1)
        lags = [-1]
        feature_times = _get_feature_times(target_series=target, lags=lags, is_training=False)
        assert len(feature_times[0]) == 1
        assert feature_times[0][0] == 1
        future = linear_timeseries(start=2, length=1, freq=2)
        lags_future = [-2]
        feature_times = _get_feature_times(target_series=target, future_covariates=future, lags=lags, lags_future_covariates=lags_future, is_training=False)
        assert len(feature_times[0]) == 1
        assert feature_times[0][0] == 1
        assert len(feature_times[2]) == 1
        assert feature_times[2][0] == 6

    def test_feature_times_prediction_single_time_datetime_idx(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests that `_get_feature_times` correctly handles case where only\n        a single time can be used to create prediction features.\n        This particular test uses datetime index timeseries.\n        '
        target = linear_timeseries(start=pd.Timestamp('1/1/2000'), length=1, freq='d')
        lags = [-1]
        feature_times = _get_feature_times(target_series=target, lags=lags, is_training=False)
        assert len(feature_times[0]) == 1
        assert feature_times[0][0] == pd.Timestamp('1/2/2000')
        future = linear_timeseries(start=pd.Timestamp('1/2/2000'), length=1, freq='2d')
        lags_future = [-2]
        feature_times = _get_feature_times(target_series=target, future_covariates=future, lags=lags, lags_future_covariates=lags_future, is_training=False)
        assert len(feature_times[0]) == 1
        assert feature_times[0][0] == pd.Timestamp('1/2/2000')
        assert len(feature_times[2]) == 1
        assert feature_times[2][0] == pd.Timestamp('1/6/2000')

    def test_feature_times_extend_time_index_range_idx(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that `_get_feature_times` is able to return feature\n        times that occur after the end of a series or occur before\n        the beginning of a series. This particular test uses range\n        index time series.\n        '
        target = linear_timeseries(start=10, length=1, freq=3)
        past = linear_timeseries(start=2, length=1, freq=2)
        future = linear_timeseries(start=3, length=1, freq=1)
        lags = lags_past = lags_future_1 = [-4]
        feature_times = _get_feature_times(target_series=target, past_covariates=past, future_covariates=future, lags=lags, lags_past_covariates=lags_past, lags_future_covariates=lags_future_1, is_training=False)
        assert len(feature_times[0]) == 1
        assert feature_times[0][0] == target.start_time() - lags[0] * target.freq
        assert len(feature_times[1]) == 1
        assert feature_times[1][0] == past.start_time() - lags_past[0] * past.freq
        assert len(feature_times[2]) == 1
        assert feature_times[2][0] == future.start_time() - lags_future_1[0] * future.freq
        lags_future_2 = [4]
        feature_times = _get_feature_times(future_covariates=future, lags_future_covariates=lags_future_2, is_training=False)
        assert len(feature_times[2]) == 1
        assert feature_times[2][0] == future.start_time() - lags_future_2[0] * future.freq

    def test_feature_times_extend_time_index_datetime_idx(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests that `_get_feature_times` is able to return feature\n        times that occur after the end of a series or occur before\n        the beginning of a series. This particular test uses datetime\n        index time series.\n        '
        target = linear_timeseries(start=pd.Timestamp('1/10/2000'), length=1, freq='3d')
        past = linear_timeseries(start=pd.Timestamp('1/2/2000'), length=1, freq='2d')
        future = linear_timeseries(start=pd.Timestamp('1/3/2000'), length=1, freq='1d')
        lags = lags_past = lags_future_1 = [-4]
        feature_times = _get_feature_times(target_series=target, past_covariates=past, future_covariates=future, lags=lags, lags_past_covariates=lags_past, lags_future_covariates=lags_future_1, is_training=False)
        assert len(feature_times[0]) == 1
        assert feature_times[0][0] == target.start_time() - lags[0] * target.freq
        assert len(feature_times[1]) == 1
        assert feature_times[1][0] == past.start_time() - lags_past[0] * past.freq
        assert len(feature_times[2]) == 1
        assert feature_times[2][0] == future.start_time() - lags_future_1[0] * future.freq
        lags_future_2 = [4]
        feature_times = _get_feature_times(future_covariates=future, lags_future_covariates=lags_future_2, is_training=False)
        assert len(feature_times[2]) == 1
        assert feature_times[2][0] == future.start_time() - lags_future_2[0] * future.freq

    def test_feature_times_future_lags_range_idx(self):
        if False:
            while True:
                i = 10
        '\n        Tests that `_get_feature_times` correctly handles the `lags_future_covariates`\n        argument for the following three cases:\n            1. `lags_future_covariates` contains only `0`\n            2. `lags_future_covariates` contains only a positive lag\n            3. `lags_future_covariates` contains a combination of positive,\n            zero, and negative lags\n        This particular test uses range index timeseries.\n        '
        future = linear_timeseries(start=0, length=10, freq=2)
        lags_future = [0]
        feature_times = _get_feature_times(future_covariates=future, lags_future_covariates=lags_future, is_training=False)
        assert len(feature_times[2]) == future.n_timesteps
        assert feature_times[2].equals(future.time_index)
        lags_future = [1]
        feature_times = _get_feature_times(future_covariates=future, lags_future_covariates=lags_future, is_training=False)
        extended_future = future.prepend_values([0])
        assert len(feature_times[2]) == extended_future.n_timesteps - 1
        assert feature_times[2].equals(extended_future.time_index[:-1])
        lags_future = [-1, 0, 1]
        feature_times = _get_feature_times(future_covariates=future, lags_future_covariates=lags_future, is_training=False)
        assert len(feature_times[2]) == future.n_timesteps - 2
        assert feature_times[2].equals(future.time_index[1:-1])

    def test_feature_times_future_lags_datetime_idx(self):
        if False:
            print('Hello World!')
        '\n        Tests that `_get_feature_times` correctly handles the `lags_future_covariates`\n        argument for the following three cases:\n            1. `lags_future_covariates` contains only `0`\n            2. `lags_future_covariates` contains only a positive lag\n            3. `lags_future_covariates` contains a combination of positive,\n            zero, and negative lags\n        This particular test uses datetime index timeseries.\n        '
        future = linear_timeseries(start=pd.Timestamp('1/1/2000'), length=10, freq='2d')
        lags_future = [0]
        feature_times = _get_feature_times(future_covariates=future, lags_future_covariates=lags_future, is_training=False)
        assert len(feature_times[2]) == future.n_timesteps
        assert feature_times[2].equals(future.time_index)
        lags_future = [1]
        feature_times = _get_feature_times(future_covariates=future, lags_future_covariates=lags_future, is_training=False)
        extended_future = future.prepend_values([0])
        assert len(feature_times[2]) == extended_future.n_timesteps - 1
        assert feature_times[2].equals(extended_future.time_index[:-1])
        lags_future = [-1, 0, 1]
        feature_times = _get_feature_times(future_covariates=future, lags_future_covariates=lags_future, is_training=False)
        assert len(feature_times[2]) == future.n_timesteps - 2
        assert feature_times[2].equals(future.time_index[1:-1])

    def test_feature_times_unspecified_series(self):
        if False:
            return 10
        '\n        Tests that `_get_feature_times` correctly returns\n        `None` in place of a sequence of times if a particular\n        series is not specified.\n        '
        target = linear_timeseries(start=1, length=20, freq=1)
        past = linear_timeseries(start=2, length=25, freq=2)
        future = linear_timeseries(start=3, length=30, freq=3)
        lags = [-1]
        lags_past = [-2]
        lags_future = [-3]
        expected_target = target.append_values([0]).time_index[1:]
        expected_past = past.append_values(2 * [0]).time_index[2:]
        expected_future = future.append_values(3 * [0]).time_index[3:]
        feature_times = _get_feature_times(target_series=target, lags=lags, is_training=False)
        assert expected_target.equals(feature_times[0])
        assert feature_times[1] is None
        assert feature_times[2] is None
        feature_times = _get_feature_times(past_covariates=past, lags_past_covariates=lags_past, is_training=False)
        assert feature_times[0] is None
        assert expected_past.equals(feature_times[1])
        assert feature_times[2] is None
        feature_times = _get_feature_times(future_covariates=future, lags_future_covariates=lags_future, is_training=False)
        assert feature_times[0] is None
        assert feature_times[1] is None
        assert expected_future.equals(feature_times[2])
        feature_times = _get_feature_times(target_series=target, past_covariates=past, lags=lags, lags_past_covariates=lags_past, is_training=False)
        assert expected_target.equals(feature_times[0])
        assert expected_past.equals(feature_times[1])
        assert feature_times[2] is None
        feature_times = _get_feature_times(target_series=target, future_covariates=future, lags=lags, lags_future_covariates=lags_future, is_training=False)
        assert expected_target.equals(feature_times[0])
        assert feature_times[1] is None
        assert expected_future.equals(feature_times[2])
        feature_times = _get_feature_times(past_covariates=past, future_covariates=future, lags_past_covariates=lags_past, lags_future_covariates=lags_future, is_training=False)
        assert feature_times[0] is None
        assert expected_past.equals(feature_times[1])
        assert expected_future.equals(feature_times[2])

    def test_feature_times_unspecified_lag_or_series_warning(self):
        if False:
            return 10
        '\n        Tests that `_get_feature_times` throws correct warning when\n        a series is specified by its corresponding lag is not, or\n        vice versa. The only circumstance under which a warning\n        should *not* be issued is when `target_series` is specified,\n        but `lags` is not when `is_training = True`; this is because\n        the user may not want to add auto-regressive features to `X`,\n        but they still need to specify `target_series` to create labels.\n        '
        target = linear_timeseries(start=1, length=20, freq=1)
        past = linear_timeseries(start=2, length=25, freq=2)
        future = linear_timeseries(start=3, length=30, freq=3)
        lags = [-1, -2]
        lags_past = [-2, -5]
        lags_future = [-3, -5]
        with warnings.catch_warnings(record=True) as w:
            _ = _get_feature_times(past_covariates=past, future_covariates=future, lags_past_covariates=lags_past, is_training=False)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert str(w[0].message) == '`future_covariates` was specified without accompanying `lags_future_covariates` and, thus, will be ignored.'
        with warnings.catch_warnings(record=True) as w:
            _ = _get_feature_times(past_covariates=past, lags_past_covariates=lags_past, lags_future_covariates=lags_future, is_training=False)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert str(w[0].message) == '`lags_future_covariates` was specified without accompanying `future_covariates` and, thus, will be ignored.'
        with warnings.catch_warnings(record=True) as w:
            _ = _get_feature_times(target_series=target, past_covariates=past, future_covariates=future, lags_past_covariates=lags_past, output_chunk_length=1, is_training=True)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert str(w[0].message) == '`future_covariates` was specified without accompanying `lags_future_covariates` and, thus, will be ignored.'
        with warnings.catch_warnings(record=True) as w:
            _ = _get_feature_times(target_series=target, past_covariates=past, lags_past_covariates=lags_past, lags_future_covariates=lags_future, output_chunk_length=1, is_training=True)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert str(w[0].message) == '`lags_future_covariates` was specified without accompanying `future_covariates` and, thus, will be ignored.'
        with warnings.catch_warnings(record=True) as w:
            _ = _get_feature_times(target_series=target, past_covariates=past, lags=lags, lags_future_covariates=lags_future, is_training=False)
            assert len(w) == 2
            assert issubclass(w[0].category, UserWarning)
            assert issubclass(w[1].category, UserWarning)
            assert str(w[0].message) == '`past_covariates` was specified without accompanying `lags_past_covariates` and, thus, will be ignored.'
            assert str(w[1].message) == '`lags_future_covariates` was specified without accompanying `future_covariates` and, thus, will be ignored.'
        with warnings.catch_warnings(record=True) as w:
            _ = _get_feature_times(target_series=target, past_covariates=past, lags_future_covariates=lags_future, output_chunk_length=1, is_training=True)
            assert len(w) == 2
            assert issubclass(w[0].category, UserWarning)
            assert issubclass(w[1].category, UserWarning)
            assert str(w[0].message) == '`past_covariates` was specified without accompanying `lags_past_covariates` and, thus, will be ignored.'
            assert str(w[1].message) == '`lags_future_covariates` was specified without accompanying `future_covariates` and, thus, will be ignored.'
        with warnings.catch_warnings(record=True) as w:
            _ = _get_feature_times(target_series=target, past_covariates=past, future_covariates=future, lags_past_covariates=lags_past, lags_future_covariates=lags_future, is_training=False)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
        with warnings.catch_warnings(record=True) as w:
            _ = _get_feature_times(target_series=target, past_covariates=past, future_covariates=future, lags_past_covariates=lags_past, lags_future_covariates=lags_future, is_training=True)
            assert len(w) == 0

    def test_feature_times_unspecified_training_inputs_error(self):
        if False:
            return 10
        "\n        Tests that `_get_feature_times` throws correct error when\n        `target_series` and/or `output_chunk_length` hasn't been\n        specified when `is_training = True`.\n        "
        output_chunk_length = 1
        with pytest.raises(ValueError) as err:
            _get_feature_times(output_chunk_length=output_chunk_length, is_training=True)
        assert 'Must specify `target_series` when `is_training = True`.' == str(err.value)
        with pytest.raises(ValueError) as err:
            _get_feature_times(is_training=True)
        assert 'Must specify `target_series` when `is_training = True`.' == str(err.value)

    def test_feature_times_no_lags_specified_error(self):
        if False:
            return 10
        '\n        Tests that `_get_feature_times` throws correct error\n        when no lags have been specified.\n        '
        target = linear_timeseries(start=1, length=20, freq=1)
        with pytest.raises(ValueError) as err:
            _get_feature_times(target_series=target, is_training=False)
        assert 'Must specify at least one of: `lags`, `lags_past_covariates`, `lags_future_covariates`.' == str(err.value)

    def test_feature_times_series_too_short_error(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests that `_get_feature_times` throws correct error\n        when provided series are too short for specified\n        lag and/or `output_chunk_length` values.\n        '
        series = linear_timeseries(start=1, length=2, freq=1)
        with pytest.raises(ValueError) as err:
            _get_feature_times(target_series=series, lags=[-20, -1], is_training=False)
        assert '`target_series` must have at least `-min(lags) + max(lags) + 1` = 20 timesteps; instead, it only has 2.' == str(err.value)
        with pytest.raises(ValueError) as err:
            _get_feature_times(target_series=series, lags=[-20], output_chunk_length=5, is_training=True)
        assert '`target_series` must have at least `-min(lags) + output_chunk_length` = 25 timesteps; instead, it only has 2.' == str(err.value)
        with pytest.raises(ValueError) as err:
            _get_feature_times(target_series=series, past_covariates=series, lags_past_covariates=[-20, -1], output_chunk_length=1, is_training=True)
        assert '`past_covariates` must have at least `-min(lags_past_covariates) + max(lags_past_covariates) + 1` = 20 timesteps; instead, it only has 2.' == str(err.value)

    def test_feature_times_invalid_lag_values_error(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that `_get_feature_times` throws correct error\n        when provided with invalid lag values (i.e. not less than\n        0 if `lags`, or not less than 1 if `lags_past_covariates` or\n        `lags_future_covariates`).\n        '
        series = linear_timeseries(start=1, length=3, freq=1)
        with pytest.raises(ValueError) as err:
            _get_feature_times(target_series=series, lags=[0], is_training=False)
        assert '`lags` must be a `Sequence` or `Dict` containing only `int` values less than 0.' == str(err.value)
        with pytest.raises(ValueError) as err:
            _get_feature_times(past_covariates=series, lags_past_covariates=[0], is_training=False)
        assert '`lags_past_covariates` must be a `Sequence` or `Dict` containing only `int` values less than 0.' == str(err.value)
        _get_feature_times(future_covariates=series, lags_future_covariates=[-1, 0, 1], is_training=False)