"""
Gaussian Processes
------------------
"""
from typing import Optional
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from darts.models.filtering.filtering_model import FilteringModel
from darts.timeseries import TimeSeries

class GaussianProcessFilter(FilteringModel):

    def __init__(self, kernel: Optional[Kernel]=None, **kwargs):
        if False:
            return 10
        "\n        This model uses the ``GaussianProcessRegressor`` of scikit-learn to fit a Gaussian Process to the\n        supplied TimeSeries. This can then be used to obtain samples from the\n        Gaussian Process at the times of the TimeSeries.\n\n        It can for instance be used to fill in missing (NaN) values from a TimeSeries.\n\n        Parameters\n        ----------\n        kernel : sklearn.gaussian_process.kernels.Kernel, default: None\n            The kernel specifying the covariance function of the Gaussian Process. If None is passed,\n            the default in scikit-learn is used. Note that the kernel hyperparameters are optimized\n            during fitting unless the bounds are marked as 'fixed'.\n        **kwargs\n            Additional keyword arguments passed to ``sklearn.gaussian_process.GaussianProcessRegressor``.\n        "
        super().__init__()
        self.model = GaussianProcessRegressor(kernel=kernel, **kwargs)

    def filter(self, series: TimeSeries, num_samples: int=1) -> TimeSeries:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fits the Gaussian Process on the observations and returns samples from the Gaussian Process,\n        or its mean values if `num_samples` is set to 1.\n\n        Parameters\n        ----------\n        series\n            The series of observations used to infer the values according to the specified Gaussian Process.\n            This must be a deterministic series (containing one sample).\n        num_samples: int, default: 1\n            Number of times a prediction is sampled from the Gaussian Process. If set to 1,\n            the mean values will be returned instead.\n\n        Returns\n        -------\n        TimeSeries\n            A stochastic ``TimeSeries`` sampled from the Gaussian Process, or its mean\n            if `num_samples` is set to 1.\n        '
        super().filter(series)
        values = series.values(copy=False)
        if series.has_datetime_index:
            times = np.arange(series.n_timesteps).reshape(-1, 1)
        else:
            times = series.time_index.values.reshape(-1, 1)
        not_nan_mask = np.all(~np.isnan(values), axis=1)
        self.model.fit(times[not_nan_mask, :], values[not_nan_mask, :])
        if num_samples == 1:
            filtered_values = self.model.predict(times)
        else:
            filtered_values = self.model.sample_y(times, n_samples=num_samples)
        filtered_values = filtered_values.reshape(len(times), -1, num_samples)
        return series.with_values(filtered_values)