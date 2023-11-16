import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.datasets import AirPassengersDataset
from darts.metrics import mae
from darts.models import LinearRegressionModel, StatsForecastAutoETS

class TestStatsForecastAutoETS:
    ts_passengers = AirPassengersDataset().load()
    (ts_pass_train, ts_pass_val) = ts_passengers.split_after(pd.Timestamp('19570101'))
    trend_values = np.arange(start=1, stop=len(ts_passengers) + 1)
    trend_times = ts_passengers.time_index
    ts_trend = TimeSeries.from_times_and_values(times=trend_times, values=trend_values, columns=['trend'])
    (ts_trend_train, ts_trend_val) = ts_trend.split_after(pd.Timestamp('19570101'))

    def test_fit_on_residuals(self):
        if False:
            while True:
                i = 10
        model = StatsForecastAutoETS(season_length=12, model='ZZZ')
        model.fit(series=self.ts_pass_train, future_covariates=self.ts_trend_train)
        fitted_values_linreg = model._linreg.model.predict(X=self.ts_trend_train.values(copy=False))
        fitted_values_linreg_ts = TimeSeries.from_times_and_values(times=self.ts_pass_train.time_index, values=fitted_values_linreg)
        resids = self.ts_pass_train - fitted_values_linreg_ts
        in_sample_preds = model.model.predict_in_sample()['fitted']
        ts_in_sample_preds = TimeSeries.from_times_and_values(times=self.ts_pass_train.time_index, values=in_sample_preds)
        current_mae = mae(resids, ts_in_sample_preds)
        assert current_mae < 9

    def test_fit_a_linreg(self):
        if False:
            i = 10
            return i + 15
        model = StatsForecastAutoETS(season_length=12, model='ZZZ')
        model.fit(series=self.ts_pass_train, future_covariates=self.ts_trend_train)
        assert model._linreg is not None
        assert model._linreg._fit_called
        linreg = LinearRegressionModel(lags_future_covariates=[0])
        linreg.fit(series=self.ts_pass_train, future_covariates=self.ts_trend_train)
        assert model._linreg.model.coef_ == linreg.model.coef_