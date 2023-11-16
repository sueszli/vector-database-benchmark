from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
plt.rc('figure', figsize=(16, 8))
plt.rc('font', size=15)

def gen_data_for_model1():
    if False:
        while True:
            i = 10
    nobs = 1000
    rs = np.random.RandomState(seed=93572)
    d = 5
    var_y = 5
    var_coeff_x = 0.01
    var_coeff_w = 0.5
    x_t = rs.uniform(size=nobs)
    w_t = rs.uniform(size=nobs)
    eps = rs.normal(scale=var_y ** 0.5, size=nobs)
    beta_x = np.cumsum(rs.normal(size=nobs, scale=var_coeff_x ** 0.5))
    beta_w = np.cumsum(rs.normal(size=nobs, scale=var_coeff_w ** 0.5))
    y_t = d + beta_x * x_t + beta_w * w_t + eps
    return (y_t, x_t, w_t, beta_x, beta_w)
(y_t, x_t, w_t, beta_x, beta_w) = gen_data_for_model1()
_ = plt.plot(y_t)

class TVRegression(sm.tsa.statespace.MLEModel):

    def __init__(self, y_t, x_t, w_t):
        if False:
            print('Hello World!')
        exog = np.c_[x_t, w_t]
        super(TVRegression, self).__init__(endog=y_t, exog=exog, k_states=2, initialization='diffuse')
        self.ssm['design'] = exog.T[np.newaxis, :, :]
        self.ssm['selection'] = np.eye(self.k_states)
        self.ssm['transition'] = np.eye(self.k_states)
        self.positive_parameters = slice(1, 4)

    @property
    def param_names(self):
        if False:
            i = 10
            return i + 15
        return ['intercept', 'var.e', 'var.x.coeff', 'var.w.coeff']

    @property
    def start_params(self):
        if False:
            return 10
        '\n        Defines the starting values for the parameters\n        The linear regression gives us reasonable starting values for the constant\n        d and the variance of the epsilon error\n        '
        exog = sm.add_constant(self.exog)
        res = sm.OLS(self.endog, exog).fit()
        params = np.r_[res.params[0], res.scale, 0.001, 0.001]
        return params

    def transform_params(self, unconstrained):
        if False:
            print('Hello World!')
        "\n        We constraint the last three parameters\n        ('var.e', 'var.x.coeff', 'var.w.coeff') to be positive,\n        because they are variances\n        "
        constrained = unconstrained.copy()
        constrained[self.positive_parameters] = constrained[self.positive_parameters] ** 2
        return constrained

    def untransform_params(self, constrained):
        if False:
            i = 10
            return i + 15
        '\n        Need to unstransform all the parameters you transformed\n        in the `transform_params` function\n        '
        unconstrained = constrained.copy()
        unconstrained[self.positive_parameters] = unconstrained[self.positive_parameters] ** 0.5
        return unconstrained

    def update(self, params, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        params = super(TVRegression, self).update(params, **kwargs)
        self['obs_intercept', 0, 0] = params[0]
        self['obs_cov', 0, 0] = params[1]
        self['state_cov'] = np.diag(params[2:4])
mod = TVRegression(y_t, x_t, w_t)
res = mod.fit()
print(res.summary())
(fig, axes) = plt.subplots(2, figsize=(16, 8))
ss = pd.DataFrame(res.smoothed_state.T, columns=['x', 'w'])
axes[0].plot(beta_x, label='True')
axes[0].plot(ss['x'], label='Smoothed estimate')
axes[0].set(title='Time-varying coefficient on x_t')
axes[0].legend()
axes[1].plot(beta_w, label='True')
axes[1].plot(ss['w'], label='Smoothed estimate')
axes[1].set(title='Time-varying coefficient on w_t')
axes[1].legend()
fig.tight_layout()

class TVRegressionExtended(sm.tsa.statespace.MLEModel):

    def __init__(self, y_t, x_t, w_t):
        if False:
            for i in range(10):
                print('nop')
        exog = np.c_[x_t, w_t]
        super(TVRegressionExtended, self).__init__(endog=y_t, exog=exog, k_states=2, initialization='diffuse')
        self.ssm['design'] = exog.T[np.newaxis, :, :]
        self.ssm['selection'] = np.eye(self.k_states)
        self.ssm['transition'] = np.eye(self.k_states)
        self.positive_parameters = slice(1, 4)

    @property
    def param_names(self):
        if False:
            for i in range(10):
                print('nop')
        return ['intercept', 'var.e', 'var.x.coeff', 'var.w.coeff', 'rho1', 'rho2']

    @property
    def start_params(self):
        if False:
            print('Hello World!')
        '\n        Defines the starting values for the parameters\n        The linear regression gives us reasonable starting values for the constant\n        d and the variance of the epsilon error\n        '
        exog = sm.add_constant(self.exog)
        res = sm.OLS(self.endog, exog).fit()
        params = np.r_[res.params[0], res.scale, 0.001, 0.001, 0.7, 0.8]
        return params

    def transform_params(self, unconstrained):
        if False:
            while True:
                i = 10
        "\n        We constraint the last three parameters\n        ('var.e', 'var.x.coeff', 'var.w.coeff') to be positive,\n        because they are variances\n        "
        constrained = unconstrained.copy()
        constrained[self.positive_parameters] = constrained[self.positive_parameters] ** 2
        return constrained

    def untransform_params(self, constrained):
        if False:
            return 10
        '\n        Need to unstransform all the parameters you transformed\n        in the `transform_params` function\n        '
        unconstrained = constrained.copy()
        unconstrained[self.positive_parameters] = unconstrained[self.positive_parameters] ** 0.5
        return unconstrained

    def update(self, params, **kwargs):
        if False:
            print('Hello World!')
        params = super(TVRegressionExtended, self).update(params, **kwargs)
        self['obs_intercept', 0, 0] = params[0]
        self['obs_cov', 0, 0] = params[1]
        self['state_cov'] = np.diag(params[2:4])
        self['transition', 0, 0] = params[4]
        self['transition', 1, 1] = params[5]
mod = TVRegressionExtended(y_t, x_t, w_t)
res = mod.fit(maxiter=2000)
print(res.summary())
true_values = {'var_e1': 0.01, 'var_e2': 0.01, 'var_w1': 0.01, 'var_w2': 0.01, 'delta1': 0.8, 'delta2': 0.5, 'delta3': 0.7}

def gen_data_for_model3():
    if False:
        print('Hello World!')
    alpha1_0 = 2.1
    alpha2_0 = 1.1
    t_max = 500

    def gen_i(alpha1, s):
        if False:
            for i in range(10):
                print('nop')
        return alpha1 * s + np.sqrt(true_values['var_e1']) * np.random.randn()

    def gen_m_hat(alpha2):
        if False:
            i = 10
            return i + 15
        return 1 * alpha2 + np.sqrt(true_values['var_e2']) * np.random.randn()

    def gen_alpha1(alpha1, alpha2):
        if False:
            while True:
                i = 10
        w1 = np.sqrt(true_values['var_w1']) * np.random.randn()
        return true_values['delta1'] * alpha1 + true_values['delta2'] * alpha2 + w1

    def gen_alpha2(alpha2):
        if False:
            return 10
        w2 = np.sqrt(true_values['var_w2']) * np.random.randn()
        return true_values['delta3'] * alpha2 + w2
    s_t = 0.3 + np.sqrt(1.4) * np.random.randn(t_max)
    i_hat = np.empty(t_max)
    m_hat = np.empty(t_max)
    current_alpha1 = alpha1_0
    current_alpha2 = alpha2_0
    for t in range(t_max):
        i_hat[t] = gen_i(current_alpha1, s_t[t])
        m_hat[t] = gen_m_hat(current_alpha2)
        new_alpha1 = gen_alpha1(current_alpha1, current_alpha2)
        new_alpha2 = gen_alpha2(current_alpha2)
        current_alpha1 = new_alpha1
        current_alpha2 = new_alpha2
    return (i_hat, m_hat, s_t)
(i_hat, m_hat, s_t) = gen_data_for_model3()
starting_values = {'var_e1': 0.2, 'var_e2': 0.1, 'var_w1': 0.15, 'var_w2': 0.18, 'delta1': 0.7, 'delta2': 0.1, 'delta3': 0.85}

class MultipleYsModel(sm.tsa.statespace.MLEModel):

    def __init__(self, i_t: np.array, s_t: np.array, m_t: np.array):
        if False:
            return 10
        exog = np.c_[s_t, np.repeat(1, len(s_t))]
        super(MultipleYsModel, self).__init__(endog=np.c_[i_t, m_t], exog=exog, k_states=2, initialization='diffuse')
        self.ssm['design'] = np.zeros((self.k_endog, self.k_states, self.nobs))
        self.ssm['design', 0, 0, :] = s_t
        self.ssm['design', 1, 1, :] = 1
        self.ssm['selection'] = np.eye(self.k_states)
        self.ssm['transition'] = np.eye(self.k_states)
        self.position_dict = OrderedDict(var_e1=1, var_e2=2, var_w1=3, var_w2=4, delta1=5, delta2=6, delta3=7)
        self.initial_values = starting_values
        self.positive_parameters = slice(0, 4)

    @property
    def param_names(self):
        if False:
            i = 10
            return i + 15
        return list(self.position_dict.keys())

    @property
    def start_params(self):
        if False:
            print('Hello World!')
        '\n        Initial values\n        '
        params = np.r_[self.initial_values['var_e1'], self.initial_values['var_e2'], self.initial_values['var_w1'], self.initial_values['var_w2'], self.initial_values['delta1'], self.initial_values['delta2'], self.initial_values['delta3']]
        return params

    def transform_params(self, unconstrained):
        if False:
            for i in range(10):
                print('nop')
        '\n        If you need to restrict parameters\n        For example, variances should be > 0\n        Parameters maybe have to be within -1 and 1\n        '
        constrained = unconstrained.copy()
        constrained[self.positive_parameters] = constrained[self.positive_parameters] ** 2
        return constrained

    def untransform_params(self, constrained):
        if False:
            return 10
        '\n        Need to reverse what you did in transform_params()\n        '
        unconstrained = constrained.copy()
        unconstrained[self.positive_parameters] = unconstrained[self.positive_parameters] ** 0.5
        return unconstrained

    def update(self, params, **kwargs):
        if False:
            return 10
        params = super(MultipleYsModel, self).update(params, **kwargs)
        self['obs_intercept'] = np.repeat([np.array([0, 0])], self.nobs, axis=0).T
        self['obs_cov', 0, 0] = params[0]
        self['obs_cov', 1, 1] = params[1]
        self['state_cov'] = np.diag(params[2:4])
        self['transition', 0, 0] = params[4]
        self['transition', 0, 1] = params[5]
        self['transition', 1, 1] = params[6]
mod = MultipleYsModel(i_hat, s_t, m_hat)
res = mod.fit()
print(res.summary())
import pymc3 as pm
import theano
import theano.tensor as tt

class Loglike(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, model):
        if False:
            print('Hello World!')
        self.model = model
        self.score = Score(self.model)

    def perform(self, node, inputs, outputs):
        if False:
            while True:
                i = 10
        (theta,) = inputs
        llf = self.model.loglike(theta)
        outputs[0][0] = np.array(llf)

    def grad(self, inputs, g):
        if False:
            for i in range(10):
                print('nop')
        (theta,) = inputs
        out = [g[0] * self.score(theta)]
        return out

class Score(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, model):
        if False:
            print('Hello World!')
        self.model = model

    def perform(self, node, inputs, outputs):
        if False:
            i = 10
            return i + 15
        (theta,) = inputs
        outputs[0][0] = self.model.score(theta)
(y_t, x_t, w_t, beta_x, beta_w) = gen_data_for_model1()
plt.plot(y_t)
mod = TVRegression(y_t, x_t, w_t)
res_mle = mod.fit(disp=False)
print(res_mle.summary())
ndraws = 3000
nburn = 600
loglike = Loglike(mod)
with pm.Model():
    intercept = pm.Uniform('intercept', 1, 10)
    var_e = pm.InverseGamma('var.e', 2.3, 0.5)
    var_x_coeff = pm.InverseGamma('var.x.coeff', 2.3, 0.1)
    var_w_coeff = pm.InverseGamma('var.w.coeff', 2.3, 0.1)
    theta = tt.as_tensor_variable([intercept, var_e, var_x_coeff, var_w_coeff])
    pm.DensityDist('likelihood', loglike, observed=theta)
    trace = pm.sample(ndraws, tune=nburn, return_inferencedata=True, cores=1, compute_convergence_checks=False)
results_dict = {'intercept': res_mle.params[0], 'var.e': res_mle.params[1], 'var.x.coeff': res_mle.params[2], 'var.w.coeff': res_mle.params[3]}
plt.tight_layout()
_ = pm.plot_trace(trace, lines=[(k, {}, [v]) for (k, v) in dict(results_dict).items()], combined=True, figsize=(12, 12))