import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
'\nUnivariate Local Linear Trend Model\n'

class LocalLinearTrend(sm.tsa.statespace.MLEModel):

    def __init__(self, endog):
        if False:
            while True:
                i = 10
        k_states = k_posdef = 2
        super(LocalLinearTrend, self).__init__(endog, k_states=k_states, k_posdef=k_posdef, initialization='approximate_diffuse', loglikelihood_burn=k_states)
        self.ssm['design'] = np.array([1, 0])
        self.ssm['transition'] = np.array([[1, 1], [0, 1]])
        self.ssm['selection'] = np.eye(k_states)
        self._state_cov_idx = ('state_cov',) + np.diag_indices(k_posdef)

    @property
    def param_names(self):
        if False:
            print('Hello World!')
        return ['sigma2.measurement', 'sigma2.level', 'sigma2.trend']

    @property
    def start_params(self):
        if False:
            for i in range(10):
                print('nop')
        return [np.std(self.endog)] * 3

    def transform_params(self, unconstrained):
        if False:
            while True:
                i = 10
        return unconstrained ** 2

    def untransform_params(self, constrained):
        if False:
            i = 10
            return i + 15
        return constrained ** 0.5

    def update(self, params, *args, **kwargs):
        if False:
            return 10
        params = super(LocalLinearTrend, self).update(params, *args, **kwargs)
        self.ssm['obs_cov', 0, 0] = params[0]
        self.ssm[self._state_cov_idx] = params[1:]
import requests
from io import BytesIO
from zipfile import ZipFile
ck = requests.get('http://staff.feweb.vu.nl/koopman/projects/ckbook/OxCodeAll.zip').content
zipped = ZipFile(BytesIO(ck))
df = pd.read_table(BytesIO(zipped.read('OxCodeIntroStateSpaceBook/Chapter_2/NorwayFinland.txt')), skiprows=1, header=None, sep='\\s+', engine='python', names=['date', 'nf', 'ff'])
df.index = pd.date_range(start='%d-01-01' % df.date[0], end='%d-01-01' % df.iloc[-1, 0], freq='AS')
df['lff'] = np.log(df['ff'])
mod = LocalLinearTrend(df['lff'])
res = mod.fit(disp=False)
print(res.summary())
predict = res.get_prediction()
forecast = res.get_forecast('2014')
(fig, ax) = plt.subplots(figsize=(10, 4))
df['lff'].plot(ax=ax, style='k.', label='Observations')
predict.predicted_mean.plot(ax=ax, label='One-step-ahead Prediction')
predict_ci = predict.conf_int(alpha=0.05)
predict_index = np.arange(len(predict_ci))
ax.fill_between(predict_index[2:], predict_ci.iloc[2:, 0], predict_ci.iloc[2:, 1], alpha=0.1)
forecast.predicted_mean.plot(ax=ax, style='r', label='Forecast')
forecast_ci = forecast.conf_int()
forecast_index = np.arange(len(predict_ci), len(predict_ci) + len(forecast_ci))
ax.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], alpha=0.1)
ax.set_ylim((4, 8))
legend = ax.legend(loc='lower left')