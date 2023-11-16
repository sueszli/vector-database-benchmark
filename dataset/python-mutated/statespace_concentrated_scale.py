import numpy as np
import pandas as pd
import statsmodels.api as sm
dta = sm.datasets.macrodata.load_pandas().data
dta.index = pd.date_range(start='1959Q1', end='2009Q4', freq='Q')

class LocalLevel(sm.tsa.statespace.MLEModel):
    _start_params = [1.0, 1.0]
    _param_names = ['var.level', 'var.irregular']

    def __init__(self, endog):
        if False:
            return 10
        super(LocalLevel, self).__init__(endog, k_states=1, initialization='diffuse')
        self['design', 0, 0] = 1
        self['transition', 0, 0] = 1
        self['selection', 0, 0] = 1

    def transform_params(self, unconstrained):
        if False:
            for i in range(10):
                print('nop')
        return unconstrained ** 2

    def untransform_params(self, unconstrained):
        if False:
            i = 10
            return i + 15
        return unconstrained ** 0.5

    def update(self, params, **kwargs):
        if False:
            return 10
        params = super(LocalLevel, self).update(params, **kwargs)
        self['state_cov', 0, 0] = params[0]
        self['obs_cov', 0, 0] = params[1]
mod = LocalLevel(dta.infl)
res = mod.fit(disp=False)
print(res.summary())
print(res.mle_retvals)

class LocalLevelConcentrated(sm.tsa.statespace.MLEModel):
    _start_params = [1.0]
    _param_names = ['ratio.irregular']

    def __init__(self, endog):
        if False:
            print('Hello World!')
        super(LocalLevelConcentrated, self).__init__(endog, k_states=1, initialization='diffuse')
        self['design', 0, 0] = 1
        self['transition', 0, 0] = 1
        self['selection', 0, 0] = 1
        self['state_cov', 0, 0] = 1
        self.ssm.filter_concentrated = True

    def transform_params(self, unconstrained):
        if False:
            i = 10
            return i + 15
        return unconstrained ** 2

    def untransform_params(self, unconstrained):
        if False:
            while True:
                i = 10
        return unconstrained ** 0.5

    def update(self, params, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        params = super(LocalLevelConcentrated, self).update(params, **kwargs)
        self['obs_cov', 0, 0] = params[0]
mod_conc = LocalLevelConcentrated(dta.infl)
res_conc = mod_conc.fit(disp=False)
print(res_conc.summary())
print(res_conc.mle_retvals)
print('Original model')
print('var.level     = %.5f' % res.params[0])
print('var.irregular = %.5f' % res.params[1])
print('\nConcentrated model')
print('scale         = %.5f' % res_conc.scale)
print('h * scale     = %.5f' % (res_conc.params[0] * res_conc.scale))
mod_ar = sm.tsa.SARIMAX(dta.cpi, order=(1, 0, 0), trend='ct')
res_ar = mod_ar.fit(disp=False)
mod_ar_conc = sm.tsa.SARIMAX(dta.cpi, order=(1, 0, 0), trend='ct', concentrate_scale=True)
res_ar_conc = mod_ar_conc.fit(disp=False)
print('Loglikelihood')
print('- Original model:     %.4f' % res_ar.llf)
print('- Concentrated model: %.4f' % res_ar_conc.llf)
print('\nParameters')
print('- Original model:     %.4f, %.4f, %.4f, %.4f' % tuple(res_ar.params))
print('- Concentrated model: %.4f, %.4f, %.4f, %.4f' % (tuple(res_ar_conc.params) + (res_ar_conc.scale,)))
print('Optimizer iterations')
print('- Original model:     %d' % res_ar.mle_retvals['iterations'])
print('- Concentrated model: %d' % res_ar_conc.mle_retvals['iterations'])