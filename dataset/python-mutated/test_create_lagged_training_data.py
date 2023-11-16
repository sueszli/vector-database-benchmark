import warnings
from itertools import product
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries
from darts import concatenate as darts_concatenate
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.utils.data.tabularization import create_lagged_component_names, create_lagged_training_data
from darts.utils.timeseries_generation import linear_timeseries

class TestCreateLaggedTrainingData:
    """
    Tests the `create_lagged_training_data` function defined in `darts.utils.data.tabularization`. There are broadly
    two 'groups' of tests defined in this module:
        1. 'Generated Test Cases': these test that `create_lagged_training_data` produces the same outputs
        as a simplified implementation of the 'time intersection' feature generation method (see
        `darts.utils.data.tabularization` for further details). For these tests, the 'correct answer' is not
        directly specified; instead, it is generated from a set of input parameters using the set of simplified
        functions. The rationale behind this approach is that it allows for many different combinations of input
        values to be effortlessly tested. The drawback of this, however, is that the correctness of these tests
        assumes that the simplified functions have been implemented correctly - if this isn't the case, then these
        tests are not to be trusted. In saying this, these simplified functions are significantly easier to
        understand and debug than the `create_lagged_training_data` function they're helping to test.
        2. 'Specified Test Cases': these test that `create_lagged_training_data` returns an exactly specified
        output; these specified outputs are *not* 'generated' by another function. Although these 'specified'
        test cases tend to be simpler and less extensive than the 'generated' test cases, their correctness does
        not assume the correct implementation of any other function.
    """

    @staticmethod
    def create_multivariate_linear_timeseries(n_components: int, components_names: Sequence[str]=None, **kwargs) -> TimeSeries:
        if False:
            print('Hello World!')
        '\n        Helper function that creates a `linear_timeseries` with a specified number of\n        components. To help distinguish each component from one another, `i` is added on\n        to each value of the `i`th component. Any additional keyword arguments are passed\n        to `linear_timeseries` (`start_value`, `end_value`, `start`, `end`, `length`, etc).\n        '
        timeseries = []
        if components_names is None or len(components_names) < n_components:
            components_names = [f'lin_ts_{i}' for i in range(n_components)]
        for i in range(n_components):
            timeseries_i = linear_timeseries(column_name=components_names[i], **kwargs) + i
            timeseries.append(timeseries_i)
        return darts_concatenate(timeseries, axis=1)

    @staticmethod
    def get_feature_times(target: TimeSeries, past: TimeSeries, future: TimeSeries, lags: Optional[Sequence[int]], lags_past: Optional[Sequence[int]], lags_future: Optional[Sequence[int]], output_chunk_length: Optional[int], max_samples_per_ts: Optional[int]):
        if False:
            print('Hello World!')
        "\n        Helper function that returns the times shared by all of the specified series that can be used\n        to create features and labels. This is performed by using the helper functions\n        `get_feature_times_target`, `get_feature_times_past`, and `get_feature_times_future` (all\n        defined below) to extract the feature times from the target series, past covariates, and future\n        covariates respectively, and then intersecting these features times with one another. A series is\n        considered to be 'specified' if its corresponding lag (e.g. `lags` for `target`, or `lags_future`\n        for `future`) is not `None`. If requested, the last `max_samples_per_ts` times are taken.\n\n        This function is basically a simplified implementation of `get_feature_times` in `tabularization.py`\n        that only works for `is_training = True`.\n        "
        times = TestCreateLaggedTrainingData.get_feature_times_target(target, lags, output_chunk_length)
        if lags_past is not None:
            past_times = TestCreateLaggedTrainingData.get_feature_times_past(past, lags_past)
            times = times.intersection(past_times)
        if lags_future is not None:
            future_times = TestCreateLaggedTrainingData.get_feature_times_future(future, lags_future)
            times = times.intersection(future_times)
        if max_samples_per_ts is not None and len(times) > max_samples_per_ts:
            times = times[-max_samples_per_ts:]
        return times

    @staticmethod
    def get_feature_times_target(target_series: TimeSeries, lags: Optional[Sequence[int]], output_chunk_length: int) -> pd.Index:
        if False:
            while True:
                i = 10
        '\n        Helper function called by `get_feature_times` that extracts all of the times within a\n        `target_series` that can be used to create a feature and label. More specifically,\n        we can create features and labels for times within `target_series` that have *both*:\n            1. At least `max_lag = -min(lags)` values preceeding them, since these preceeding\n            values are required to construct a feature vector for that time. Since the first `max_lag`\n            times do not fulfill this condition, they are exluded *if* values from `target_series` are\n            to be added to `X`.\n            2. At least `(output_chunk_length - 1)` values after them, because the all of the times from\n            time `t` to time `t + output_chunk_length - 1` will be used as labels. Since the last\n            `(output_chunk_length - 1)` times do not fulfil this condition, they are excluded.\n        '
        times = target_series.time_index
        if lags is not None:
            max_lag = -min(lags)
            times = times[max_lag:]
        if output_chunk_length > 1:
            times = times[:-output_chunk_length + 1]
        return times

    @staticmethod
    def get_feature_times_past(past_covariates: TimeSeries, past_covariates_lags: Sequence[int]) -> pd.Index:
        if False:
            while True:
                i = 10
        "\n        Helper function called by `get_feature_times` that extracts all of the times within\n        `past_covariates` that can be used to create features. More specifically, we can create\n        features for times within `past_covariates` that have at least `max_lag = -min(past_covariates_lags)`\n        values preceeding them, since these preceeding values are required to construct a feature vector for\n        that time. Since the first `max_lag` times do not fulfill this condition, they are exluded.\n\n        Unlike the `target_series`, features can be constructed for times that occur after the end of\n        `past_covariates`; this is because:\n            1. We don't need to have all the `past_covariates` values up to time `t` to construct\n            a feature for this time; instead, we only need to have the values from time `t - min_lag`\n            to `t - max_lag`, where `min_lag = -max(past_covariates_lags)` and\n            `max_lag = -min(past_covariates_lags)`. In other words, the latest feature we can create\n            for `past_covariates` occurs at `past_covariates.end_time() + min_lag * past_covariates.freq`.\n            2. We don't need to use the values of `past_covariates` to construct labels, so we're able\n            to create a feature for time `t` without having to worry about whether we can construct\n            a corresponding label for this time.\n        "
        times = past_covariates.time_index
        min_lag = -max(past_covariates_lags)
        times = times.union([times[-1] + i * past_covariates.freq for i in range(1, min_lag + 1)])
        max_lag = -min(past_covariates_lags)
        times = times[max_lag:]
        return times

    @staticmethod
    def get_feature_times_future(future_covariates: TimeSeries, future_covariates_lags: Sequence[int]) -> pd.Index:
        if False:
            i = 10
            return i + 15
        "\n        Helper function called by `get_feature_times` that extracts all of the times within\n        `future_covariates` that can be used to create features.\n\n        Unlike the lag values for `target_series` and `past_covariates`, the values in\n        `future_covariates_lags` can be negative, zero, or positive. This means that\n        `min_lag = -max(future_covariates_lags)` and `max_lag = -min(future_covariates_lags)`\n        are *not* guaranteed to be positive here: they could be negative (corresponding to\n        a positive value in `future_covariates_lags`), zero, or positive (corresponding to\n        a negative value in `future_covariates_lags`). With that being said, the relationship\n        `min_lag <= max_lag` always holds.\n\n        Consequently, we need to consider three scenarios when finding feature times\n        for `future_covariates`:\n            1. Both `min_lag` and `max_lag` are positive, which indicates that all of\n            the lag values in `future_covariates_lags` are negative (i.e. only values before\n            time `t` are used to create a feature from time `t`). In this case, `min_lag`\n            and `max_lag` correspond to the smallest magnitude and largest magnitude *negative*\n            lags in `future_covariates_lags` respectively. This means we *can* create features for\n            times that extend beyond the end of `future_covariates`; additionally, we're unable\n            to create features for the first `min_lag` times (see docstring for `get_feature_times_past`).\n            2. Both `min_lag` and `max_lag` are non-positive. In this case, `abs(min_lag)` and `abs(max_lag)`\n            correspond to the largest and smallest magnitude lags in `future_covariates_lags` respectively;\n            note that, somewhat confusingly, `abs(max_lag) <= abs(min_lag)` here. This means that we *can* create f\n            features for times that occur before the start of `future_covariates`; the reasoning for this is\n            basically the inverse of Case 1 (i.e. we only need to know the values from times `t + abs(max_lag)`\n            to `t + abs(min_lag)` to create a feature for time `t`). Additionally, we're unable to create features\n            for the last `abs(min_lag)` times in the series, since these times do not have `abs(min_lag)` values\n            after them.\n            3. `min_lag` is non-positive (i.e. zero or negative), but `max_lag` is positive. In this case,\n            `abs(min_lag)` is the magnitude of the largest *non-negative* lag value in `future_covariates_lags`\n            and `max_lag` is the largest *negative* lag value in `future_covariates_lags`. This means that we\n            *cannot* create features for times that occur before the start of `future_covariates`, nor for\n            times that occur after the end of `future_covariates`; this is because we must have access to\n            both times before *and* after time `t` to create a feature for this time, which clearly can't\n            be acieved for times extending before the start or after the end of the series. Moreover,\n            we must exclude the first `max_lag` times and the last `abs(min_lag)` times, since these\n            times do not have enough values before or after them respectively.\n        "
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

    @staticmethod
    def construct_X_block(series: TimeSeries, feature_times: pd.Index, lags: Optional[Sequence[int]]) -> np.array:
        if False:
            while True:
                i = 10
        "\n        Helper function that creates the lagged features 'block' of a specific\n        `series` (i.e. either `target_series`, `past_covariates`, or `future_covariates`);\n        the feature matrix `X` is formed by concatenating the blocks of all of the specified\n        series along the components axis. If `lags` is `None`, then `None` will be returned in\n        lieu of an array. Please refer to the `create_lagged_features` docstring for further\n        details about the structure of the `X` feature matrix.\n\n        The returned `X_block` is constructed by looping over each time in `feature_times`,\n        finding the index position of that time in the series, and then for each lag value in\n        `lags`, offset this index position by a particular lag value; this offset index is then\n        used to extract all of the components at a single lagged time.\n\n        Unlike the implementation found in `darts.utils.data.tabularization`, this function doesn't\n        use any 'vectorisation' tricks, which makes it slower to run, but more easily interpretable.\n\n        Some of the times in `feature_times` may occur before the start *or* after the end of `series`;\n        see the docstrings of `get_feature_times_past` and `get_feature_times_future` for why this is the\n        case. Because of this, we need to prepend or append these 'extended times' to `series.time_index`\n        before searching for the index of each time in the series. Even though the integer indices of the\n        'extended times' won't be contained within the original `series`, offsetting these found indices\n        by the requested lag value should 'bring us back' to a time within the original, unextended `series`.\n        However, if we've prepended times to `series.time_index`, we have to note that all of the indices will\n        be 'bumped up' by the number of values we've prepended, even after offsetting by a lag value. For example,\n        if we extended `series.time_index` by prepending two values to the start, the integer index of the first\n        actual value in `series` will occur at an index of `2` instead of `0`. To 'undo' this, we must subtract off\n        the number of prepended value from the lag-offseted indices before retrieving values from `series`.\n        "
        if lags is None:
            X_block = None
        else:
            series_times = series.time_index
            is_range_idx = isinstance(series_times[0], int)
            add_to_end = series_times[-1] < feature_times[-1]
            add_to_start = series_times[0] > feature_times[0]
            if add_to_end:
                num_prepended = 0
                if is_range_idx:
                    series_times = pd.RangeIndex(start=series_times[0], stop=feature_times[-1] + 1, step=series.freq)
                else:
                    series_times = pd.date_range(start=series_times[0], end=feature_times[-1], freq=series.freq)
            elif add_to_start:
                num_prepended = (series_times[0] - feature_times[0]) // series.freq
                if is_range_idx:
                    series_times = pd.RangeIndex(start=feature_times[0], stop=series_times[-1] + 1, step=series.freq)
                else:
                    series_times = pd.date_range(start=feature_times[0], end=series_times[-1], freq=series.freq)
            else:
                num_prepended = 0
            array_vals = series.all_values(copy=False)[:, :, 0]
            X_block = []
            for time in feature_times:
                time_idx = np.searchsorted(series_times, time)
                X_row = []
                for lag in lags:
                    idx_to_get = time_idx + lag
                    idx_to_get -= num_prepended
                    raise_if_not(idx_to_get >= 0, f'Unexpected case encountered: `time_idx + lag - num_prepended = {idx_to_get} < 0`.')
                    X_row.append(array_vals[idx_to_get, :].reshape(-1))
                X_row = np.concatenate(X_row, axis=0)
                X_block.append(X_row)
            X_block = np.stack(X_block, axis=0)
        return X_block

    @staticmethod
    def create_y(target: TimeSeries, feature_times: pd.Index, output_chunk_length: int, multi_models: bool) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        "\n        Helper function that constructs the labels array `y` from the target series.\n        This is done by by looping over each time in `feature_times`, finding the index\n        position of that time in the target series, and then for each timestep ahead of\n        this time we wish to predict, offset this time index by this timestep. This offset\n        index is then used to extract all components of the target series at this time which\n        is to be predicted. Please refer to the `create_lagged_features` docstring for further\n        details about the structure of the `y` labels matrix.\n\n        Unlike `construct_X_block`, we don't need to worry about times in `feature_times` lying\n        outside of `target.time_index` here: each label *must* be contained in the `target`\n        series already.\n        "
        array_vals = target.all_values(copy=False)
        y = []
        for time in feature_times:
            raise_if(time < target.start_time(), f'Unexpected label time at {time}, but `series` starts at {target.start_time()}.')
            raise_if(time > target.end_time(), f'Unexpected label time at {time}, but `series` ends at {target.end_time()}.')
            time_idx = np.searchsorted(target.time_index, time)
            timesteps_ahead = range(output_chunk_length) if multi_models else (output_chunk_length - 1,)
            y_row = []
            for i in timesteps_ahead:
                idx_to_get = time_idx + i
                y_row.append(array_vals[idx_to_get, :, 0].reshape(-1))
            y_row = np.concatenate(y_row, axis=0)
            y.append(y_row)
        y = np.stack(y, axis=0)
        return y
    output_chunk_length_combos = (1, 3)
    multi_models_combos = (False, True)
    max_samples_per_ts_combos = (1, 2, None)
    target_lag_combos = past_lag_combos = (None, [-1, -3], [-3, -1])
    future_lag_combos = (*target_lag_combos, [0], [2, 1], [-1, 1], [0, 2])

    def test_lagged_training_data_equal_freq_range_index(self):
        if False:
            print('Hello World!')
        "\n        Tests that `create_lagged_training_data` produces `X`, `y`, and `times`\n        outputs that are consistent with those generated by using the helper\n        functions `get_feature_times`, `construct_X_block`, and `construct_labels`.\n        Consistency is checked over all of the combinations of parameter values\n        specified by `self.target_lag_combos`, `self.covariates_lag_combos`,\n        `self.output_chunk_length_combos`, `self.multi_models_combos`, and\n        `self.max_samples_per_ts_combos`.\n\n        This particular test uses timeseries with range time indices of equal\n        frequencies. Since all of the timeseries are of the same frequency,\n        the implementation of the 'moving window' method is being tested here.\n        "
        target = self.create_multivariate_linear_timeseries(n_components=2, start_value=0, end_value=10, start=2, length=8, freq=2)
        past = self.create_multivariate_linear_timeseries(n_components=3, start_value=10, end_value=20, start=4, length=9, freq=2)
        future = self.create_multivariate_linear_timeseries(n_components=4, start_value=20, end_value=30, start=6, length=10, freq=2)
        for (lags, lags_past, lags_future, output_chunk_length, multi_models, max_samples_per_ts) in product(self.target_lag_combos, self.past_lag_combos, self.future_lag_combos, self.output_chunk_length_combos, self.multi_models_combos, self.max_samples_per_ts_combos):
            all_lags = (lags, lags_past, lags_future)
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
            (X, y, times, _) = create_lagged_training_data(target, output_chunk_length, past_covariates=past if lags_past else None, future_covariates=future if lags_future else None, lags=lags, lags_past_covariates=lags_past, lags_future_covariates=lags_future, uses_static_covariates=False, multi_models=multi_models, max_samples_per_ts=max_samples_per_ts, use_moving_windows=True)
            feats_times = self.get_feature_times(target, past, future, lags, lags_past, lags_future, output_chunk_length, max_samples_per_ts)
            X_target = self.construct_X_block(target, feats_times, lags)
            X_past = self.construct_X_block(past, feats_times, lags_past)
            X_future = self.construct_X_block(future, feats_times, lags_future)
            all_X = (X_target, X_past, X_future)
            to_concat = [X for X in all_X if X is not None]
            expected_X = np.concatenate(to_concat, axis=1)
            expected_y = self.create_y(target, feats_times, output_chunk_length, multi_models)
            assert X.shape[0] == len(feats_times)
            assert y.shape[0] == len(feats_times)
            assert X.shape[0] == len(times[0])
            assert y.shape[0] == len(times[0])
            assert np.allclose(expected_X, X[:, :, 0])
            assert np.allclose(expected_y, y[:, :, 0])
            assert feats_times.equals(times[0])

    def test_lagged_training_data_equal_freq_datetime_index(self):
        if False:
            return 10
        "\n        Tests that `create_lagged_training_data` produces `X`, `y`, and `times`\n        outputs that are consistent with those generated by using the helper\n        functions `get_feature_times`, `construct_X_block`, and `construct_labels`.\n        Consistency is checked over all of the combinations of parameter values\n        specified by `self.target_lag_combos`, `self.covariates_lag_combos`,\n        `self.output_chunk_length_combos`, `self.multi_models_combos`, and\n        `self.max_samples_per_ts_combos`.\n\n        This particular test uses timeseries with datetime time indices of equal\n        frequencies. Since all of the timeseries are of the same frequency,\n        the implementation of the 'moving window' method is being tested here.\n        "
        target = self.create_multivariate_linear_timeseries(n_components=2, start_value=0, end_value=10, start=pd.Timestamp('1/2/2000'), length=8, freq='2d')
        past = self.create_multivariate_linear_timeseries(n_components=3, start_value=10, end_value=20, start=pd.Timestamp('1/4/2000'), length=9, freq='2d')
        future = self.create_multivariate_linear_timeseries(n_components=4, start_value=20, end_value=30, start=pd.Timestamp('1/6/2000'), length=10, freq='2d')
        for (lags, lags_past, lags_future, output_chunk_length, multi_models, max_samples_per_ts) in product(self.target_lag_combos, self.past_lag_combos, self.future_lag_combos, self.output_chunk_length_combos, self.multi_models_combos, self.max_samples_per_ts_combos):
            all_lags = (lags, lags_past, lags_future)
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
            (X, y, times, _) = create_lagged_training_data(target, output_chunk_length, past_covariates=past if lags_past else None, future_covariates=future if lags_future else None, lags=lags, lags_past_covariates=lags_past, lags_future_covariates=lags_future, uses_static_covariates=False, multi_models=multi_models, max_samples_per_ts=max_samples_per_ts, use_moving_windows=True)
            feats_times = self.get_feature_times(target, past, future, lags, lags_past, lags_future, output_chunk_length, max_samples_per_ts)
            X_target = self.construct_X_block(target, feats_times, lags)
            X_past = self.construct_X_block(past, feats_times, lags_past)
            X_future = self.construct_X_block(future, feats_times, lags_future)
            all_X = (X_target, X_past, X_future)
            to_concat = [x for x in all_X if x is not None]
            expected_X = np.concatenate(to_concat, axis=1)
            expected_y = self.create_y(target, feats_times, output_chunk_length, multi_models)
            assert X.shape[0] == len(feats_times)
            assert y.shape[0] == len(feats_times)
            assert X.shape[0] == len(times[0])
            assert y.shape[0] == len(times[0])
            assert np.allclose(expected_X, X[:, :, 0])
            assert np.allclose(expected_y, y[:, :, 0])
            assert feats_times.equals(times[0])

    def test_lagged_training_data_unequal_freq_range_index(self):
        if False:
            return 10
        "\n        Tests that `create_lagged_training_data` produces `X`, `y`, and `times`\n        outputs that are consistent with those generated by using the helper\n        functions `get_feature_times`, `construct_X_block`, and `construct_labels`.\n        Consistency is checked over all of the combinations of parameter values\n        specified by `self.target_lag_combos`, `self.covariates_lag_combos`,\n        `self.output_chunk_length_combos`, `self.multi_models_combos`, and\n        `self.max_samples_per_ts_combos`.\n\n        This particular test uses timeseries with range time indices of unequal\n        frequencies. Since all of the timeseries are *not* of the same frequency,\n        the implementation of the 'time intersection' method is being tested here.\n        "
        target = self.create_multivariate_linear_timeseries(n_components=2, start_value=0, end_value=10, start=2, length=20, freq=1)
        past = self.create_multivariate_linear_timeseries(n_components=3, start_value=10, end_value=20, start=4, length=10, freq=2)
        future = self.create_multivariate_linear_timeseries(n_components=4, start_value=20, end_value=30, start=6, length=7, freq=3)
        for (lags, lags_past, lags_future, output_chunk_length, multi_models, max_samples_per_ts) in product(self.target_lag_combos, self.past_lag_combos, self.future_lag_combos, self.output_chunk_length_combos, self.multi_models_combos, self.max_samples_per_ts_combos):
            all_lags = (lags, lags_past, lags_future)
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
            (X, y, times, _) = create_lagged_training_data(target, output_chunk_length, past_covariates=past if lags_past else None, future_covariates=future if lags_future else None, lags=lags, lags_past_covariates=lags_past, lags_future_covariates=lags_future, uses_static_covariates=False, multi_models=multi_models, max_samples_per_ts=max_samples_per_ts, use_moving_windows=False)
            feats_times = self.get_feature_times(target, past, future, lags, lags_past, lags_future, output_chunk_length, max_samples_per_ts)
            X_target = self.construct_X_block(target, feats_times, lags)
            X_past = self.construct_X_block(past, feats_times, lags_past)
            X_future = self.construct_X_block(future, feats_times, lags_future)
            all_X = (X_target, X_past, X_future)
            to_concat = [x for x in all_X if x is not None]
            expected_X = np.concatenate(to_concat, axis=1)
            expected_y = self.create_y(target, feats_times, output_chunk_length, multi_models)
            assert X.shape[0] == len(feats_times)
            assert y.shape[0] == len(feats_times)
            assert X.shape[0] == len(times[0])
            assert y.shape[0] == len(times[0])
            assert np.allclose(expected_X, X[:, :, 0])
            assert np.allclose(expected_y, y[:, :, 0])
            assert feats_times.equals(times[0])

    def test_lagged_training_data_unequal_freq_datetime_index(self):
        if False:
            print('Hello World!')
        "\n        Tests that `create_lagged_training_data` produces `X`, `y`, and `times`\n        outputs that are consistent with those generated by using the helper\n        functions `get_feature_times`, `construct_X_block`, and `construct_labels`.\n        Consistency is checked over all of the combinations of parameter values\n        specified by `self.target_lag_combos`, `self.covariates_lag_combos`,\n        `self.output_chunk_length_combos`, `self.multi_models_combos`, and\n        `self.max_samples_per_ts_combos`.\n\n        This particular test uses timeseries with datetime time indices of unequal\n        frequencies. Since all of the timeseries are *not* of the same frequency,\n        the implementation of the 'time intersection' method is being tested here.\n        "
        target = self.create_multivariate_linear_timeseries(n_components=2, start_value=0, end_value=10, start=pd.Timestamp('1/1/2000'), length=20, freq='d')
        past = self.create_multivariate_linear_timeseries(n_components=3, start_value=10, end_value=20, start=pd.Timestamp('1/2/2000'), length=10, freq='2d')
        future = self.create_multivariate_linear_timeseries(n_components=4, start_value=20, end_value=30, start=pd.Timestamp('1/3/2000'), length=7, freq='3d')
        for (lags, lags_past, lags_future, output_chunk_length, multi_models, max_samples_per_ts) in product(self.target_lag_combos, self.past_lag_combos, self.future_lag_combos, self.output_chunk_length_combos, self.multi_models_combos, self.max_samples_per_ts_combos):
            all_lags = (lags, lags_past, lags_future)
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
            (X, y, times, _) = create_lagged_training_data(target, output_chunk_length, past_covariates=past if lags_past else None, future_covariates=future if lags_future else None, lags=lags, lags_past_covariates=lags_past, lags_future_covariates=lags_future, uses_static_covariates=False, multi_models=multi_models, max_samples_per_ts=max_samples_per_ts, use_moving_windows=False)
            feats_times = self.get_feature_times(target, past, future, lags, lags_past, lags_future, output_chunk_length, max_samples_per_ts)
            X_target = self.construct_X_block(target, feats_times, lags)
            X_past = self.construct_X_block(past, feats_times, lags_past)
            X_future = self.construct_X_block(future, feats_times, lags_future)
            all_X = (X_target, X_past, X_future)
            to_concat = [x for x in all_X if x is not None]
            expected_X = np.concatenate(to_concat, axis=1)
            expected_y = self.create_y(target, feats_times, output_chunk_length, multi_models)
            assert X.shape[0] == len(feats_times)
            assert y.shape[0] == len(feats_times)
            assert X.shape[0] == len(times[0])
            assert y.shape[0] == len(times[0])
            assert np.allclose(expected_X, X[:, :, 0])
            assert np.allclose(expected_y, y[:, :, 0])
            assert feats_times.equals(times[0])

    def test_lagged_training_data_method_consistency_range_index(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Tests that `create_lagged_training_data` produces the same result\n        when `use_moving_windows = False` and when `use_moving_windows = True`\n        for all of the parameter combinations used in the 'generated' test cases.\n\n        Obviously, if both the 'Moving Window Method' and the 'Time Intersection'\n        are both wrong in the same way, this test won't reveal any bugs. With this\n        being said, if this test fails, something is definitely wrong in either\n        one or both of the implemented methods.\n\n        This particular test uses range index timeseries.\n        "
        target = self.create_multivariate_linear_timeseries(n_components=2, start_value=0, end_value=10, start=2, length=20, freq=1)
        past = self.create_multivariate_linear_timeseries(n_components=3, start_value=10, end_value=20, start=4, length=10, freq=2)
        future = self.create_multivariate_linear_timeseries(n_components=4, start_value=20, end_value=30, start=6, length=7, freq=3)
        for (lags, lags_past, lags_future, output_chunk_length, multi_models, max_samples_per_ts) in product(self.target_lag_combos, self.past_lag_combos, self.future_lag_combos, self.output_chunk_length_combos, self.multi_models_combos, self.max_samples_per_ts_combos):
            all_lags = (lags, lags_past, lags_future)
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
            (X_mw, y_mw, times_mw, _) = create_lagged_training_data(target_series=target, output_chunk_length=output_chunk_length, past_covariates=past if lags_past else None, future_covariates=future if lags_future else None, lags=lags, lags_past_covariates=lags_past, lags_future_covariates=lags_future, uses_static_covariates=False, max_samples_per_ts=max_samples_per_ts, multi_models=multi_models, use_moving_windows=True)
            (X_ti, y_ti, times_ti, _) = create_lagged_training_data(target_series=target, output_chunk_length=output_chunk_length, past_covariates=past if lags_past else None, future_covariates=future if lags_future else None, lags=lags, lags_past_covariates=lags_past, lags_future_covariates=lags_future, uses_static_covariates=False, max_samples_per_ts=max_samples_per_ts, multi_models=multi_models, use_moving_windows=False)
            assert np.allclose(X_mw, X_ti)
            assert np.allclose(y_mw, y_ti)
            assert times_mw[0].equals(times_ti[0])

    def test_lagged_training_data_method_consistency_datetime_index(self):
        if False:
            i = 10
            return i + 15
        "\n        Tests that `create_lagged_training_data` produces the same result\n        when `use_moving_windows = False` and when `use_moving_windows = True`\n        for all of the parameter combinations used in the 'generated' test cases.\n\n        Obviously, if both the 'Moving Window Method' and the 'Time Intersection'\n        are both wrong in the same way, this test won't reveal any bugs. With this\n        being said, if this test fails, something is definitely wrong in either\n        one or both of the implemented methods.\n\n        This particular test uses datetime index timeseries.\n        "
        target = self.create_multivariate_linear_timeseries(n_components=2, start_value=0, end_value=10, start=pd.Timestamp('1/2/2000'), end=pd.Timestamp('1/16/2000'), freq='2d')
        past = self.create_multivariate_linear_timeseries(n_components=3, start_value=10, end_value=20, start=pd.Timestamp('1/4/2000'), end=pd.Timestamp('1/18/2000'), freq='2d')
        future = self.create_multivariate_linear_timeseries(n_components=4, start_value=20, end_value=30, start=pd.Timestamp('1/6/2000'), end=pd.Timestamp('1/20/2000'), freq='2d')
        for (lags, lags_past, lags_future, output_chunk_length, multi_models, max_samples_per_ts) in product(self.target_lag_combos, self.past_lag_combos, self.future_lag_combos, self.output_chunk_length_combos, self.multi_models_combos, self.max_samples_per_ts_combos):
            all_lags = (lags, lags_past, lags_future)
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
            (X_mw, y_mw, times_mw, _) = create_lagged_training_data(target_series=target, output_chunk_length=output_chunk_length, past_covariates=past if lags_past else None, future_covariates=future if lags_future else None, lags=lags, lags_past_covariates=lags_past, lags_future_covariates=lags_future, uses_static_covariates=False, max_samples_per_ts=max_samples_per_ts, multi_models=multi_models, use_moving_windows=True)
            (X_ti, y_ti, times_ti, _) = create_lagged_training_data(target_series=target, output_chunk_length=output_chunk_length, past_covariates=past if lags_past else None, future_covariates=future if lags_future else None, lags=lags, lags_past_covariates=lags_past, lags_future_covariates=lags_future, uses_static_covariates=False, max_samples_per_ts=max_samples_per_ts, multi_models=multi_models, use_moving_windows=False)
            assert np.allclose(X_mw, X_ti)
            assert np.allclose(y_mw, y_ti)
            assert times_mw[0].equals(times_ti[0])

    def test_lagged_training_data_single_lag_single_component_same_series_range_idx(self):
        if False:
            return 10
        '\n        Tests that `create_lagged_training_data` correctly produces `X`, `y` and `times`\n        when all the `series` inputs are identical, all the `lags` inputs consist\n        of a single value, and `output_chunk_length` is `1`. In this situation, the\n        expected `X` values can be found by concatenating three different slices of the\n        same time series, and the expected  `y` can be formed by taking a single slice\n        from the `target`. This particular test uses a time series with a range index.\n        '
        series = linear_timeseries(start=0, length=15)
        lags = [-1]
        output_chunk_length = 1
        past_lags = [-3]
        future_lags = [2]
        expected_times = series.time_index[3:-2]
        expected_y = series.all_values(copy=False)[3:-2, :, 0]
        expected_X_target = series.all_values(copy=False)[2:-3, :, 0]
        expected_X_past = series.all_values(copy=False)[:-5, :, 0]
        expected_X_future = series.all_values(copy=False)[5:, :, 0]
        expected_X = np.concatenate([expected_X_target, expected_X_past, expected_X_future], axis=1)
        for use_moving_windows in (False, True):
            (X, y, times, _) = create_lagged_training_data(target_series=series, output_chunk_length=output_chunk_length, past_covariates=series, future_covariates=series, lags=lags, lags_past_covariates=past_lags, lags_future_covariates=future_lags, uses_static_covariates=False, use_moving_windows=use_moving_windows)
            assert X.shape[0] == len(expected_times)
            assert X.shape[0] == len(times[0])
            assert y.shape[0] == len(expected_times)
            assert y.shape[0] == len(times[0])
            assert np.allclose(expected_X, X[:, :, 0])
            assert np.allclose(expected_y, y[:, :, 0])
            assert expected_times.equals(times[0])

    def test_lagged_training_data_single_lag_single_component_same_series_datetime_idx(self):
        if False:
            while True:
                i = 10
        '\n        Tests that `create_lagged_training_data` correctly produces `X`, `y` and `times`\n        when all the `series` inputs are identical, all the `lags` inputs consist\n        of a single value, and `output_chunk_length` is `1`. In this situation, the\n        expected `X` values can be found by concatenating three different slices of the\n        same time series, and the expected  `y` can be formed by taking a single slice\n        from the `target`. This particular test uses a time series with a datetime index.\n        '
        series = linear_timeseries(start=pd.Timestamp('1/1/2000'), length=15)
        lags = [-1]
        output_chunk_length = 1
        past_lags = [-3]
        future_lags = [2]
        expected_times = series.time_index[3:-2]
        expected_y = series.all_values(copy=False)[3:-2, :, 0]
        expected_X_target = series.all_values(copy=False)[2:-3, :, 0]
        expected_X_past = series.all_values(copy=False)[:-5, :, 0]
        expected_X_future = series.all_values(copy=False)[5:, :, 0]
        expected_X = np.concatenate([expected_X_target, expected_X_past, expected_X_future], axis=1)
        for use_moving_windows in (False, True):
            (X, y, times, _) = create_lagged_training_data(target_series=series, output_chunk_length=output_chunk_length, past_covariates=series, future_covariates=series, lags=lags, lags_past_covariates=past_lags, lags_future_covariates=future_lags, uses_static_covariates=False, use_moving_windows=use_moving_windows)
            assert X.shape[0] == len(expected_times)
            assert X.shape[0] == len(times[0])
            assert y.shape[0] == len(expected_times)
            assert y.shape[0] == len(times[0])
            assert np.allclose(expected_X, X[:, :, 0])
            assert np.allclose(expected_y, y[:, :, 0])
            assert expected_times.equals(times[0])

    def test_lagged_training_data_extend_past_and_future_covariates_range_idx(self):
        if False:
            return 10
        "\n        Tests that `create_lagged_training_data` correctly handles case where features\n        and labels can be created for a time that is *not* contained in `past_covariates`\n        and/or `future_covariates`. This particular test checks this behaviour by using\n        range index timeseries.\n\n        More specifically, we define the series and lags such that a training example can\n        be generated for time `target.end_time()`, even though this time isn't contained in\n        neither `past` nor `future`.\n        "
        target = linear_timeseries(start=0, end=10, start_value=1, end_value=2)
        lags = [-1]
        past = linear_timeseries(start=0, end=8, start_value=2, end_value=3)
        lags_past = [-2]
        future = linear_timeseries(start=0, end=6, start_value=3, end_value=4)
        lags_future = [-4]
        max_samples_per_ts = 1
        expected_X = np.concatenate([target.all_values(copy=False)[-2, :, 0], past.all_values(copy=False)[-1, :, 0], future.all_values(copy=False)[-1, :, 0]]).reshape(1, -1)
        expected_y = target.all_values(copy=False)[-1, :, 0]
        for use_moving_windows in (False, True):
            (X, y, times, _) = create_lagged_training_data(target, output_chunk_length=1, past_covariates=past, future_covariates=future, lags=lags, lags_past_covariates=lags_past, lags_future_covariates=lags_future, uses_static_covariates=False, max_samples_per_ts=max_samples_per_ts, use_moving_windows=use_moving_windows)
            assert times[0][0] == target.end_time()
            assert np.allclose(expected_X, X[:, :, 0])
            assert np.allclose(expected_y, y[:, :, 0])

    @pytest.mark.parametrize('freq', ['D', 'MS', 'Y'])
    def test_lagged_training_data_extend_past_and_future_covariates_datetime_idx(self, freq):
        if False:
            while True:
                i = 10
        "\n        Tests that `create_lagged_training_data` correctly handles case where features\n        and labels can be created for a time that is *not* contained in `past_covariates`\n        and/or `future_covariates`. This particular test checks this behaviour by using\n        datetime index timeseries and three different frequencies: daily, month start and\n        year end.\n\n        More specifically, we define the series and lags such that a training example can\n        be generated for time `target.end_time()`, even though this time isn't contained in\n        neither `past` nor `future`.\n        "
        target = linear_timeseries(start=pd.Timestamp('1/1/2000'), start_value=1, end_value=2, length=11, freq=freq)
        lags = [-1]
        past = linear_timeseries(start=pd.Timestamp('1/1/2000'), start_value=2, end_value=3, length=9, freq=freq)
        lags_past = [-2]
        future = linear_timeseries(start=pd.Timestamp('1/1/2000'), start_value=3, end_value=4, length=7, freq=freq)
        lags_future = [-4]
        max_samples_per_ts = 1
        expected_X = np.concatenate([target.all_values(copy=False)[-2, :, 0], past.all_values(copy=False)[-1, :, 0], future.all_values(copy=False)[-1, :, 0]]).reshape(1, -1)
        expected_y = target.all_values(copy=False)[-1, :, 0]
        for use_moving_windows in (False, True):
            (X, y, times, _) = create_lagged_training_data(target, output_chunk_length=1, past_covariates=past, future_covariates=future, lags=lags, lags_past_covariates=lags_past, lags_future_covariates=lags_future, uses_static_covariates=False, max_samples_per_ts=max_samples_per_ts, use_moving_windows=use_moving_windows)
            assert times[0][0] == target.end_time()
            assert np.allclose(expected_X, X[:, :, 0])
            assert np.allclose(expected_y, y[:, :, 0])

    def test_lagged_training_data_single_point_range_idx(self):
        if False:
            print('Hello World!')
        '\n        Tests that `create_lagged_training_data` correctly handles case\n        where only one possible training point can be generated.  This\n        particular test checks this behaviour by using range index timeseries.\n        '
        target = linear_timeseries(start=0, length=2, start_value=0, end_value=1)
        output_chunk_length = 1
        lags = [-1]
        expected_X = np.zeros((1, 1, 1))
        expected_y = np.ones((1, 1, 1))
        for (use_moving_windows, multi_models) in product([False, True], [False, True]):
            (X, y, times, _) = create_lagged_training_data(target, output_chunk_length, lags=lags, uses_static_covariates=False, multi_models=multi_models, use_moving_windows=use_moving_windows)
            assert np.allclose(expected_X, X)
            assert np.allclose(expected_y, y)
            assert len(times[0]) == 1
            assert times[0][0] == target.end_time()

    def test_lagged_training_data_single_point_datetime_idx(self):
        if False:
            return 10
        '\n        Tests that `create_lagged_training_data` correctly handles case\n        where only one possible training point can be generated. This\n        particular test checks this behaviour by using datetime index timeseries.\n        '
        target = linear_timeseries(start=pd.Timestamp('1/1/2000'), length=2, start_value=0, end_value=1)
        output_chunk_length = 1
        lags = [-1]
        expected_X = np.zeros((1, 1, 1))
        expected_y = np.ones((1, 1, 1))
        for (use_moving_windows, multi_models) in product([False, True], [False, True]):
            (X, y, times, _) = create_lagged_training_data(target, output_chunk_length, lags=lags, uses_static_covariates=False, multi_models=multi_models, use_moving_windows=use_moving_windows)
            assert np.allclose(expected_X, X)
            assert np.allclose(expected_y, y)
            assert len(times[0]) == 1
            assert times[0][0] == target.end_time()

    def test_lagged_training_data_zero_lags_range_idx(self):
        if False:
            print('Hello World!')
        "\n        Tests that `create_lagged_training_data` correctly handles case when\n        `0` is included in `lags_future_covariates` (i.e. when we're using the values\n        `future_covariates` at time `t` to predict the value of `target_series` at\n        that same time point). This particular test checks this behaviour by using\n        range index timeseries.\n        "
        target = linear_timeseries(start=0, length=2, start_value=0, end_value=1)
        future = linear_timeseries(start=target.end_time(), length=1, start_value=1, end_value=2)
        expected_X = np.array([0.0, 1.0]).reshape(1, 2, 1)
        expected_y = np.ones((1, 1, 1))
        for (use_moving_windows, multi_models) in product([False, True], [False, True]):
            (X, y, times, _) = create_lagged_training_data(target, output_chunk_length=1, future_covariates=future, lags=[-1], lags_future_covariates=[0], uses_static_covariates=False, multi_models=multi_models, use_moving_windows=use_moving_windows)
            assert np.allclose(expected_X, X)
            assert np.allclose(expected_y, y)
            assert len(times[0]) == 1
            assert times[0][0] == target.end_time()

    def test_lagged_training_data_zero_lags_datetime_idx(self):
        if False:
            print('Hello World!')
        "\n        Tests that `create_lagged_training_data` correctly handles case when\n        `0` is included in `lags_future_covariates` (i.e. when we're using the values\n        `future_covariates` at time `t` to predict the value of `target_series` at\n        that same time point). This particular test checks this behaviour by using\n        datetime index timeseries.\n        "
        target = linear_timeseries(start=pd.Timestamp('1/1/2000'), length=2, start_value=0, end_value=1)
        future = linear_timeseries(start=target.end_time(), length=1, start_value=1, end_value=2)
        expected_X = np.array([0.0, 1.0]).reshape(1, 2, 1)
        expected_y = np.ones((1, 1, 1))
        for (use_moving_windows, multi_models) in product([False, True], [False, True]):
            (X, y, times, _) = create_lagged_training_data(target, output_chunk_length=1, future_covariates=future, lags=[-1], lags_future_covariates=[0], uses_static_covariates=False, multi_models=multi_models, use_moving_windows=use_moving_windows)
            assert np.allclose(expected_X, X)
            assert np.allclose(expected_y, y)
            assert len(times[0]) == 1
            assert times[0][0] == target.end_time()

    def test_lagged_training_data_positive_lags_range_idx(self):
        if False:
            while True:
                i = 10
        "\n        Tests that `create_lagged_training_data` correctly handles case when\n        `0` is included in `lags_future_covariates` (i.e. when we're using the values\n        `future_covariates` at time `t` to predict the value of `target_series` at\n        that same time point). This particular test checks this behaviour by using\n        datetime index timeseries.\n        "
        target = linear_timeseries(start=0, length=2, start_value=0, end_value=1)
        future = linear_timeseries(start=target.end_time() + target.freq, length=1, start_value=1, end_value=2)
        expected_X = np.array([0.0, 1.0]).reshape(1, 2, 1)
        expected_y = np.ones((1, 1, 1))
        for (use_moving_windows, multi_models) in product([False, True], [False, True]):
            (X, y, times, _) = create_lagged_training_data(target, output_chunk_length=1, future_covariates=future, lags=[-1], lags_future_covariates=[1], uses_static_covariates=False, multi_models=multi_models, use_moving_windows=use_moving_windows)
            assert np.allclose(expected_X, X)
            assert np.allclose(expected_y, y)
            assert len(times[0]) == 1
            assert times[0][0] == target.end_time()

    def test_lagged_training_data_positive_lags_datetime_idx(self):
        if False:
            i = 10
            return i + 15
        "\n        Tests that `create_lagged_training_data` correctly handles case when\n        `0` is included in `lags_future_covariates` (i.e. when we're using the values\n        `future_covariates` at time `t` to predict the value of `target_series` at\n        that same time point). This particular test checks this behaviour by using\n        datetime index timeseries.\n        "
        target = linear_timeseries(start=pd.Timestamp('1/1/2000'), length=2, start_value=0, end_value=1)
        future = linear_timeseries(start=target.end_time() + target.freq, length=1, start_value=1, end_value=2)
        expected_X = np.array([0.0, 1.0]).reshape(1, 2, 1)
        expected_y = np.ones((1, 1, 1))
        for (use_moving_windows, multi_models) in product([False, True], [False, True]):
            (X, y, times, _) = create_lagged_training_data(target, output_chunk_length=1, future_covariates=future, lags=[-1], lags_future_covariates=[1], uses_static_covariates=False, multi_models=multi_models, use_moving_windows=use_moving_windows)
            assert np.allclose(expected_X, X)
            assert np.allclose(expected_y, y)
            assert len(times[0]) == 1
            assert times[0][0] == target.end_time()

    def test_lagged_training_data_sequence_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that `create_lagged_training_data` correctly handles being\n        passed a sequence of `TimeSeries` inputs, as opposed to individual\n        `TimeSeries`.\n        '
        target_1 = past_1 = future_1 = linear_timeseries(start=0, end=5)
        target_2 = past_2 = future_2 = linear_timeseries(start=6, end=11)
        lags = lags_past = lags_future = [-1]
        output_chunk_length = 1
        expected_X_1 = np.concatenate(3 * [target_1.all_values(copy=False)[:-1, :, :]], axis=1)
        expected_X_2 = np.concatenate(3 * [target_2.all_values(copy=False)[:-1, :, :]], axis=1)
        expected_X = np.concatenate([expected_X_1, expected_X_2], axis=0)
        expected_y_1 = target_1.all_values(copy=False)[1:, :, :]
        expected_y_2 = target_2.all_values(copy=False)[1:, :, :]
        expected_y = np.concatenate([expected_y_1, expected_y_2], axis=0)
        expected_times_1 = target_1.time_index[1:]
        expected_times_2 = target_2.time_index[1:]
        (X, y, times, _) = create_lagged_training_data((target_1, target_2), output_chunk_length=output_chunk_length, past_covariates=(past_1, past_2), future_covariates=(future_1, future_2), lags=lags, lags_past_covariates=lags_past, lags_future_covariates=lags_future, uses_static_covariates=False)
        assert np.allclose(X, expected_X)
        assert np.allclose(y, expected_y)
        assert len(times) == 2
        assert times[0].equals(expected_times_1)
        assert times[1].equals(expected_times_2)
        (X, y, times, _) = create_lagged_training_data((target_1, target_2), output_chunk_length=output_chunk_length, past_covariates=(past_1, past_2), future_covariates=(future_1, future_2), lags=lags, lags_past_covariates=lags_past, lags_future_covariates=lags_future, uses_static_covariates=False, concatenate=False)
        assert len(X) == 2
        assert len(y) == 2
        assert np.allclose(X[0], expected_X_1)
        assert np.allclose(X[1], expected_X_2)
        assert np.allclose(y[0], expected_y_1)
        assert np.allclose(y[1], expected_y_2)
        assert len(times) == 2
        assert times[0].equals(expected_times_1)
        assert times[1].equals(expected_times_2)

    def test_lagged_training_data_stochastic_series(self):
        if False:
            return 10
        '\n        Tests that `create_lagged_training_data` is correctly vectorised\n        over the sample axes of the input `TimeSeries`.\n        '
        target_1 = past_1 = future_1 = linear_timeseries(start=0, end=5)
        target_2 = past_2 = future_2 = 2 * target_1
        target = target_1.concatenate(target_2, axis=2)
        past = past_1.concatenate(past_2, axis=2)
        future = future_1.concatenate(future_2, axis=2)
        lags = lags_past = lags_future = [-1]
        output_chunk_length = 1
        expected_X = np.concatenate(3 * [target.all_values(copy=False)[:-1, :, :]], axis=1)
        expected_y = target.all_values(copy=False)[1:, :, :]
        expected_times = target.time_index[1:]
        (X, y, times, _) = create_lagged_training_data(target, output_chunk_length=output_chunk_length, past_covariates=past, future_covariates=future, lags=lags, lags_past_covariates=lags_past, lags_future_covariates=lags_future, uses_static_covariates=False)
        assert np.allclose(X, expected_X)
        assert np.allclose(y, expected_y)
        assert times[0].equals(expected_times)

    def test_lagged_training_data_no_shared_times_error(self):
        if False:
            return 10
        '\n        Tests that `create_lagged_training_data` throws correct error\n        when the specified series do not share any times in common\n        for creating features and labels.\n        '
        series_1 = linear_timeseries(start=0, length=4, freq=1)
        series_2 = linear_timeseries(start=series_1.end_time() + 1, length=4, freq=1)
        lags = [-1]
        for use_moving_windows in (False, True):
            with pytest.raises(ValueError) as err:
                create_lagged_training_data(target_series=series_1, output_chunk_length=1, lags=lags, past_covariates=series_2, lags_past_covariates=lags, uses_static_covariates=False, use_moving_windows=use_moving_windows)
            assert 'Specified series do not share any common times for which features can be created.' == str(err.value)

    def test_lagged_training_data_no_specified_series_lags_pairs_error(self):
        if False:
            while True:
                i = 10
        '\n        Tests that `create_lagged_training_data` throws correct error\n        when no lags-series pairs are specified.\n        '
        series_1 = linear_timeseries(start=1, length=10, freq=1)
        series_2 = linear_timeseries(start=1, length=10, freq=2)
        lags = [-1]
        for use_moving_windows in (False, True):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with pytest.raises(ValueError) as err:
                    create_lagged_training_data(target_series=series_1, output_chunk_length=1, lags_past_covariates=lags, uses_static_covariates=False, use_moving_windows=use_moving_windows)
            assert 'Must specify at least one series-lags pair.' == str(err.value)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with pytest.raises(ValueError) as err:
                    create_lagged_training_data(target_series=series_1, output_chunk_length=1, lags_future_covariates=lags, past_covariates=series_2, uses_static_covariates=False, use_moving_windows=use_moving_windows)
            assert 'Must specify at least one series-lags pair.' == str(err.value)

    def test_lagged_training_data_invalid_output_chunk_length_error(self):
        if False:
            while True:
                i = 10
        '\n        Tests that `create_lagged_training_data` throws correct error\n        when `output_chunk_length` is set to a non-`int` value (e.g. a\n        `float`) or a non-positive value (e.g. `0`).\n        '
        target = linear_timeseries(start=1, length=20, freq=1)
        lags = [-1]
        for use_moving_windows in (False, True):
            with pytest.raises(ValueError) as err:
                create_lagged_training_data(target_series=target, output_chunk_length=0, lags=lags, uses_static_covariates=False, use_moving_windows=use_moving_windows)
            assert '`output_chunk_length` must be a positive `int`.' == str(err.value)
            with pytest.raises(ValueError) as err:
                create_lagged_training_data(target_series=target, output_chunk_length=1.1, lags=lags, uses_static_covariates=False, use_moving_windows=use_moving_windows)
            assert '`output_chunk_length` must be a positive `int`.' == str(err.value)

    def test_lagged_training_data_no_lags_specified_error(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests that `create_lagged_training_data` throws correct error\n        when no lags are specified.\n        '
        target = linear_timeseries(start=1, length=20, freq=1)
        for use_moving_windows in (False, True):
            with pytest.raises(ValueError) as err:
                create_lagged_training_data(target_series=target, output_chunk_length=1, uses_static_covariates=False, use_moving_windows=use_moving_windows)
            assert 'Must specify at least one of: `lags`, `lags_past_covariates`, `lags_future_covariates`.' == str(err.value)

    def test_lagged_training_data_series_too_short_error(self):
        if False:
            while True:
                i = 10
        '\n        Tests that `create_lagged_training_data` throws correct error\n        when supplied `target_series` is too short to generate any\n        features/labels from using the specified `lags` and\n        `output_chunk_length` values, and when supplied\n        `past_covariates`/`future_covariates` is too short to generate\n        any features from using the specified\n        `lags_past_covariates`/`lags_future_covariates`.\n        '
        series = linear_timeseries(start=1, length=2, freq=1)
        for use_moving_windows in (False, True):
            with pytest.raises(ValueError) as err:
                create_lagged_training_data(target_series=series, output_chunk_length=5, lags=[-20, -10], uses_static_covariates=False, use_moving_windows=use_moving_windows)
            assert '`target_series` must have at least `-min(lags) + output_chunk_length` = 25 timesteps; instead, it only has 2.' == str(err.value)
            with pytest.raises(ValueError) as err:
                create_lagged_training_data(target_series=series, output_chunk_length=1, past_covariates=series, lags_past_covariates=[-5, -3], uses_static_covariates=False, use_moving_windows=use_moving_windows)
            assert '`past_covariates` must have at least `-min(lags_past_covariates) + max(lags_past_covariates) + 1` = 3 timesteps; instead, it only has 2.' == str(err.value)

    def test_lagged_training_data_invalid_lag_values_error(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Tests that `create_lagged_training_data` throws correct\n        error when invalid lag values are specified. More specifically:\n            1. If `lags` contains any value greater than `-1`, an error\n            should be thrown (since times > `-1` are used for labels).\n            2. If `lags_past_covariates` contains any value greater than\n            `-1` (since, by definition, past covariates are only 'know' )\n            3. `lags_future_covariates` should be able to contain positive\n            values, negative values, and/or zero without throwing any errors.\n        "
        series = linear_timeseries(start=1, length=3, freq=1)
        for use_moving_windows in (False, True):
            with pytest.raises(ValueError) as err:
                create_lagged_training_data(target_series=series, output_chunk_length=1, lags=[0], uses_static_covariates=False, use_moving_windows=use_moving_windows)
            assert '`lags` must be a `Sequence` or `Dict` containing only `int` values less than 0.' == str(err.value)
            with pytest.raises(ValueError) as err:
                create_lagged_training_data(target_series=series, output_chunk_length=1, past_covariates=series, lags_past_covariates=[0], uses_static_covariates=False, use_moving_windows=use_moving_windows)
            assert '`lags_past_covariates` must be a `Sequence` or `Dict` containing only `int` values less than 0.' == str(err.value)
            create_lagged_training_data(target_series=series, output_chunk_length=1, future_covariates=series, lags_future_covariates=[-1, 0, 1], uses_static_covariates=False, use_moving_windows=use_moving_windows)

    def test_lagged_training_data_unspecified_lag_or_series_warning(self):
        if False:
            while True:
                i = 10
        "\n        Tests that `create_lagged_training_data` throws correct\n        user warnings when a series is specified without any\n        corresponding lags, or vice versa. The only exception\n        to this is that a warning shouldn't be thrown if\n        `target_series` is specified without any `lags`, since\n        the `target_series` is still used to construct labels,\n        even if its not used to create features (i.e. `target_series`\n        is not ignored if `lags` is not specified).\n        "
        series = linear_timeseries(start=1, length=20, freq=1)
        lags = [-1]
        for use_moving_windows in (False, True):
            with warnings.catch_warnings(record=True) as w:
                _ = create_lagged_training_data(target_series=series, output_chunk_length=1, lags=lags, future_covariates=series, uses_static_covariates=False, use_moving_windows=use_moving_windows)
                assert len(w) == 1
                assert issubclass(w[0].category, UserWarning)
                assert str(w[0].message) == '`future_covariates` was specified without accompanying `lags_future_covariates` and, thus, will be ignored.'
            with warnings.catch_warnings(record=True) as w:
                _ = create_lagged_training_data(target_series=series, output_chunk_length=1, lags=lags, lags_future_covariates=lags, uses_static_covariates=False, use_moving_windows=use_moving_windows)
                assert len(w) == 1
                assert issubclass(w[0].category, UserWarning)
                assert str(w[0].message) == '`lags_future_covariates` was specified without accompanying `future_covariates` and, thus, will be ignored.'
            with warnings.catch_warnings(record=True) as w:
                _ = create_lagged_training_data(target_series=series, lags=lags, output_chunk_length=1, past_covariates=series, lags_future_covariates=lags, uses_static_covariates=False, use_moving_windows=use_moving_windows)
                assert len(w) == 2
                assert issubclass(w[0].category, UserWarning)
                assert issubclass(w[1].category, UserWarning)
                assert str(w[0].message) == '`past_covariates` was specified without accompanying `lags_past_covariates` and, thus, will be ignored.'
                assert str(w[1].message) == '`lags_future_covariates` was specified without accompanying `future_covariates` and, thus, will be ignored.'
            with warnings.catch_warnings(record=True) as w:
                _ = create_lagged_training_data(target_series=series, output_chunk_length=1, past_covariates=series, lags_past_covariates=lags, uses_static_covariates=False, use_moving_windows=use_moving_windows)
                assert len(w) == 0

    def test_create_lagged_component_names(self):
        if False:
            return 10
        '\n        Tests that `create_lagged_component_names` produces the expected features name depending\n        on the lags, output_chunk_length and covariates.\n        '
        target_with_no_cov = self.create_multivariate_linear_timeseries(n_components=1, components_names=['no_static'], start_value=0, end_value=10, start=2, length=10, freq=2)
        n_comp = 2
        target_with_static_cov = self.create_multivariate_linear_timeseries(n_components=n_comp, components_names=['static_0', 'static_1'], start_value=0, end_value=10, start=2, length=10, freq=2)
        target_with_static_cov = target_with_static_cov.with_static_covariates(pd.DataFrame({'dummy': [1]}))
        target_with_static_cov2 = target_with_static_cov.with_static_covariates(pd.DataFrame({'dummy': [i for i in range(n_comp)]}))
        target_with_static_cov3 = target_with_static_cov.with_static_covariates(pd.DataFrame({'dummy': [i for i in range(n_comp)], 'dummy1': [i for i in range(n_comp)]}))
        past = self.create_multivariate_linear_timeseries(n_components=3, components_names=['past_0', 'past_1', 'past_2'], start_value=10, end_value=20, start=2, length=10, freq=2)
        future = self.create_multivariate_linear_timeseries(n_components=4, components_names=['future_0', 'future_1', 'future_2', 'future_3'], start_value=20, end_value=30, start=2, length=10, freq=2)
        expected_lagged_features = ['no_static_target_lag-2', 'no_static_target_lag-1']
        (created_lagged_features, _) = create_lagged_component_names(target_series=target_with_no_cov, past_covariates=None, future_covariates=None, lags=[-2, -1], lags_past_covariates=None, lags_future_covariates=None, concatenate=False, use_static_covariates=False)
        assert expected_lagged_features == created_lagged_features
        expected_lagged_features = ['static_0_target_lag-4', 'static_1_target_lag-4', 'static_0_target_lag-1', 'static_1_target_lag-1']
        (created_lagged_features, _) = create_lagged_component_names(target_series=target_with_static_cov, past_covariates=None, future_covariates=None, lags=[-4, -1], lags_past_covariates=None, lags_future_covariates=None, concatenate=False, use_static_covariates=False)
        assert expected_lagged_features == created_lagged_features
        expected_lagged_features = ['static_0_target_lag-4', 'static_1_target_lag-4', 'static_0_target_lag-1', 'static_1_target_lag-1', 'dummy_statcov_target_global_components']
        (created_lagged_features, _) = create_lagged_component_names(target_series=target_with_static_cov, past_covariates=None, future_covariates=None, lags=[-4, -1], lags_past_covariates=None, lags_future_covariates=None, concatenate=False, use_static_covariates=True)
        assert expected_lagged_features == created_lagged_features
        expected_lagged_features = ['static_0_target_lag-4', 'static_1_target_lag-4', 'static_0_target_lag-1', 'static_1_target_lag-1', 'dummy_statcov_target_static_0', 'dummy_statcov_target_static_1']
        (created_lagged_features, _) = create_lagged_component_names(target_series=target_with_static_cov2, past_covariates=None, future_covariates=None, lags=[-4, -1], lags_past_covariates=None, lags_future_covariates=None, concatenate=False, use_static_covariates=True)
        assert expected_lagged_features == created_lagged_features
        expected_lagged_features = ['static_0_target_lag-4', 'static_1_target_lag-4', 'static_0_target_lag-1', 'static_1_target_lag-1', 'dummy_statcov_target_static_0', 'dummy_statcov_target_static_1', 'dummy1_statcov_target_static_0', 'dummy1_statcov_target_static_1']
        (created_lagged_features, _) = create_lagged_component_names(target_series=target_with_static_cov3, past_covariates=None, future_covariates=None, lags=[-4, -1], lags_past_covariates=None, lags_future_covariates=None, concatenate=False, use_static_covariates=True)
        assert expected_lagged_features == created_lagged_features
        expected_lagged_features = ['no_static_target_lag-4', 'no_static_target_lag-3', 'past_0_pastcov_lag-1', 'past_1_pastcov_lag-1', 'past_2_pastcov_lag-1']
        (created_lagged_features, _) = create_lagged_component_names(target_series=target_with_no_cov, past_covariates=past, future_covariates=None, lags=[-4, -3], lags_past_covariates=[-1], lags_future_covariates=None, concatenate=False)
        assert expected_lagged_features == created_lagged_features
        expected_lagged_features = ['no_static_target_lag-2', 'no_static_target_lag-1', 'future_0_futcov_lag3', 'future_1_futcov_lag3', 'future_2_futcov_lag3', 'future_3_futcov_lag3']
        (created_lagged_features, _) = create_lagged_component_names(target_series=target_with_no_cov, past_covariates=None, future_covariates=future, lags=[-2, -1], lags_past_covariates=None, lags_future_covariates=[3], concatenate=False)
        assert expected_lagged_features == created_lagged_features
        expected_lagged_features = ['past_0_pastcov_lag-1', 'past_1_pastcov_lag-1', 'past_2_pastcov_lag-1', 'future_0_futcov_lag2', 'future_1_futcov_lag2', 'future_2_futcov_lag2', 'future_3_futcov_lag2']
        (created_lagged_features, _) = create_lagged_component_names(target_series=target_with_no_cov, past_covariates=past, future_covariates=future, lags=None, lags_past_covariates=[-1], lags_future_covariates=[2], concatenate=False)
        assert expected_lagged_features == created_lagged_features
        expected_lagged_features = ['static_0_target_lag-2', 'static_1_target_lag-2', 'static_0_target_lag-1', 'static_1_target_lag-1', 'past_0_pastcov_lag-1', 'past_1_pastcov_lag-1', 'past_2_pastcov_lag-1', 'future_0_futcov_lag2', 'future_1_futcov_lag2', 'future_2_futcov_lag2', 'future_3_futcov_lag2']
        (created_lagged_features, _) = create_lagged_component_names(target_series=target_with_static_cov, past_covariates=past, future_covariates=future, lags=[-2, -1], lags_past_covariates=[-1], lags_future_covariates=[2], concatenate=False)
        assert expected_lagged_features == created_lagged_features
        expected_lagged_features = ['static_0_target_lag-3', 'static_1_target_lag-3', 'past_0_pastcov_lag-1', 'past_1_pastcov_lag-1', 'past_2_pastcov_lag-1', 'future_0_futcov_lag2', 'future_1_futcov_lag2', 'future_2_futcov_lag2', 'future_3_futcov_lag2']
        (created_lagged_features, _) = create_lagged_component_names(target_series=[target_with_static_cov, target_with_static_cov], past_covariates=[past, past], future_covariates=[future, future], lags=[-3], lags_past_covariates=[-1], lags_future_covariates=[2], concatenate=False)
        assert expected_lagged_features == created_lagged_features
        expected_lagged_features = ['static_0_target_lag-2', 'static_1_target_lag-2', 'static_0_target_lag-1', 'static_1_target_lag-1', 'past_0_pastcov_lag-1', 'past_1_pastcov_lag-1', 'past_2_pastcov_lag-1', 'future_0_futcov_lag2', 'future_1_futcov_lag2', 'future_2_futcov_lag2', 'future_3_futcov_lag2']
        (created_lagged_features, _) = create_lagged_component_names(target_series=[target_with_static_cov, target_with_no_cov.stack(target_with_no_cov)], past_covariates=[past, past], future_covariates=[future, past.stack(target_with_no_cov)], lags=[-2, -1], lags_past_covariates=[-1], lags_future_covariates=[2], concatenate=False)
        assert expected_lagged_features == created_lagged_features