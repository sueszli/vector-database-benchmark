import copy
import numpy as np
import pandas as pd
import pytest
from darts.utils.data.tabularization import add_static_covariates_to_lagged_data
from darts.utils.timeseries_generation import linear_timeseries

class TestAddStaticToLaggedData:
    series = linear_timeseries(length=6)
    series = series.stack(series)
    series_stcov_single = series.with_static_covariates(pd.DataFrame({'a': [0.0]}))
    series_stcov_multi = series.with_static_covariates(pd.DataFrame({'a': [0.0], 'b': [1.0]}))
    series_stcov_multivar = series.with_static_covariates(pd.DataFrame({'a': [0.0, 1.0], 'b': [10.0, 20.0]}))
    features = np.empty(shape=(len(series), 2))

    def test_add_static_covs_train(self):
        if False:
            print('Hello World!')
        (features, last_shape) = add_static_covariates_to_lagged_data(copy.deepcopy(self.features), self.series, uses_static_covariates=False, last_shape=None)
        assert features.shape == self.features.shape
        assert last_shape is None
        with pytest.raises(ValueError):
            add_static_covariates_to_lagged_data(copy.deepcopy(self.features), self.series, uses_static_covariates=True, last_shape=None)
        with pytest.raises(ValueError):
            add_static_covariates_to_lagged_data([copy.deepcopy(self.features), copy.deepcopy(self.features)], [self.series, self.series_stcov_single], uses_static_covariates=True, last_shape=None)
        with pytest.raises(ValueError):
            add_static_covariates_to_lagged_data([copy.deepcopy(self.features), copy.deepcopy(self.features)], [self.series_stcov_single, self.series_stcov_multi], uses_static_covariates=True, last_shape=None)
        expected_shape = (self.features.shape[0], self.features.shape[1] + 1)
        (features, last_shape) = add_static_covariates_to_lagged_data(copy.deepcopy(self.features), self.series_stcov_single, uses_static_covariates=True, last_shape=None)
        assert features.shape == expected_shape
        assert last_shape == self.series_stcov_single.static_covariates.shape
        expected_shape = (self.features.shape[0], self.features.shape[1] + 2)
        (features, last_shape) = add_static_covariates_to_lagged_data(copy.deepcopy(self.features), self.series_stcov_multi, uses_static_covariates=True, last_shape=None)
        assert features.shape == expected_shape
        assert last_shape == self.series_stcov_multi.static_covariates.shape
        expected_shape = (self.features.shape[0], self.features.shape[1] + 4)
        (features, last_shape) = add_static_covariates_to_lagged_data(copy.deepcopy(self.features), self.series_stcov_multivar, uses_static_covariates=True, last_shape=None)
        assert features.shape == expected_shape
        assert last_shape == self.series_stcov_multivar.static_covariates.shape
        expected_shape = (self.features.shape[0], self.features.shape[1] + 4)
        (features, last_shape) = add_static_covariates_to_lagged_data([copy.deepcopy(self.features), copy.deepcopy(self.features)], [self.series_stcov_multivar, self.series_stcov_multivar], uses_static_covariates=True, last_shape=None)
        assert [features_.shape == expected_shape for features_ in features]
        assert last_shape == self.series_stcov_multivar.static_covariates.shape
        assert np.all(features[0][:, -sum(last_shape):] == np.array([0.0, 1.0, 10.0, 20.0]))

    def test_add_static_covs_predict(self):
        if False:
            return 10
        (features, last_shape) = add_static_covariates_to_lagged_data(copy.deepcopy(self.features), self.series, uses_static_covariates=False, last_shape=(10, 10))
        assert features.shape == self.features.shape
        assert last_shape == (10, 10)
        with pytest.raises(ValueError):
            add_static_covariates_to_lagged_data(copy.deepcopy(self.features), self.series, uses_static_covariates=True, last_shape=(10, 10))
        with pytest.raises(ValueError):
            add_static_covariates_to_lagged_data(copy.deepcopy(self.features), self.series_stcov_single, uses_static_covariates=True, last_shape=(10, 10))
        expected_shape = (self.features.shape[0], self.features.shape[1] + 1)
        (features, last_shape) = add_static_covariates_to_lagged_data(copy.deepcopy(self.features), self.series_stcov_single, uses_static_covariates=True, last_shape=self.series_stcov_single.static_covariates.shape)
        assert features.shape == expected_shape
        assert last_shape == self.series_stcov_single.static_covariates.shape
        expected_shape = (self.features.shape[0], self.features.shape[1] + 2)
        (features, last_shape) = add_static_covariates_to_lagged_data(copy.deepcopy(self.features), self.series_stcov_multi, uses_static_covariates=True, last_shape=self.series_stcov_multi.static_covariates.shape)
        assert features.shape == expected_shape
        assert last_shape == self.series_stcov_multi.static_covariates.shape
        expected_shape = (self.features.shape[0], self.features.shape[1] + 4)
        (features, last_shape) = add_static_covariates_to_lagged_data(copy.deepcopy(self.features), self.series_stcov_multivar, uses_static_covariates=True, last_shape=self.series_stcov_multivar.static_covariates.shape)
        assert features.shape == expected_shape
        assert last_shape == self.series_stcov_multivar.static_covariates.shape
        expected_shape = (self.features.shape[0], self.features.shape[1] + 4)
        (features, last_shape) = add_static_covariates_to_lagged_data([copy.deepcopy(self.features), copy.deepcopy(self.features)], [self.series_stcov_multivar, self.series_stcov_multivar], uses_static_covariates=True, last_shape=self.series_stcov_multivar.static_covariates.shape)
        assert [features_.shape == expected_shape for features_ in features]
        assert last_shape == self.series_stcov_multivar.static_covariates.shape
        assert np.all(features[0][:, -sum(last_shape):] == np.array([0.0, 1.0, 10.0, 20.0]))