from importlib import reload
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import invwishart, invgamma
dta = sm.datasets.macrodata.load_pandas().data
dta.index = pd.date_range('1959Q1', '2009Q3', freq='QS')
mod = sm.tsa.UnobservedComponents(dta.infl, 'llevel')
res = mod.fit()
print(res.params)
sim_kfs = mod.simulation_smoother()
sim_cfa = mod.simulation_smoother(method='cfa')
nsimulations = 20
simulated_state_kfs = pd.DataFrame(np.zeros((mod.nobs, nsimulations)), index=dta.index)
simulated_state_cfa = pd.DataFrame(np.zeros((mod.nobs, nsimulations)), index=dta.index)
for i in range(nsimulations):
    sim_kfs.simulate()
    simulated_state_kfs.iloc[:, i] = sim_kfs.simulated_state[0]
    sim_cfa.simulate()
    simulated_state_cfa.iloc[:, i] = sim_cfa.simulated_state[0]
(fig, axes) = plt.subplots(2, figsize=(15, 6))
dta.infl.plot(ax=axes[0], color='k')
axes[0].set_title('Simulations based on KFS approach, MLE parameters')
simulated_state_kfs.plot(ax=axes[0], color='C0', alpha=0.25, legend=False)
dta.infl.plot(ax=axes[1], color='k')
axes[1].set_title('Simulations based on CFA approach, MLE parameters')
simulated_state_cfa.plot(ax=axes[1], color='C0', alpha=0.25, legend=False)
(handles, labels) = axes[0].get_legend_handles_labels()
axes[0].legend(handles[:2], ['Data', 'Simulated state'])
fig.tight_layout()
(fig, ax) = plt.subplots(figsize=(15, 3))
mod.update([4, 0.05])
for i in range(nsimulations):
    sim_kfs.simulate()
    ax.plot(dta.index, sim_kfs.simulated_state[0], color='C0', alpha=0.25, label='Simulated state')
dta.infl.plot(ax=ax, color='k', label='Data', zorder=-1)
ax.set_title('Simulations with alternative parameterization yielding a smoother trend')
(handles, labels) = ax.get_legend_handles_labels()
ax.legend(handles[-2:], labels[-2:])
fig.tight_layout()
y = dta[['realgdp', 'cpi', 'unemp', 'tbilrate']].copy()
y.columns = ['gdp', 'inf', 'unemp', 'int']
y[['gdp', 'inf']] = np.log(y[['gdp', 'inf']]).diff() * 100
y = y.iloc[1:]
(fig, ax) = plt.subplots(figsize=(15, 5))
y.plot(ax=ax)
ax.set_title('Evolution of macroeconomic variables included in TVP-VAR exercise')

class TVPVAR(sm.tsa.statespace.MLEModel):

    def __init__(self, y):
        if False:
            while True:
                i = 10
        augmented = sm.tsa.lagmat(y, 1, trim='both', original='in', use_pandas=True)
        p = y.shape[1]
        y_t = augmented.iloc[:, :p]
        z_t = sm.add_constant(augmented.iloc[:, p:])
        k_states = p * (p + 1)
        super().__init__(y_t, exog=z_t, k_states=k_states)
        self['design'] = np.zeros((self.k_endog, self.k_states, self.nobs))
        for i in range(self.k_endog):
            start = i * (self.k_endog + 1)
            end = start + self.k_endog + 1
            self['design', i, start:end, :] = z_t.T
        self['transition'] = np.eye(k_states)
        self['selection'] = np.eye(k_states)
        self.ssm.initialize('known', stationary_cov=5 * np.eye(self.k_states))

    def update_variances(self, obs_cov, state_cov_diag):
        if False:
            while True:
                i = 10
        self['obs_cov'] = obs_cov
        self['state_cov'] = np.diag(state_cov_diag)

    @property
    def state_names(self):
        if False:
            for i in range(10):
                print('nop')
        state_names = np.empty((self.k_endog, self.k_endog + 1), dtype=object)
        for i in range(self.k_endog):
            endog_name = self.endog_names[i]
            state_names[i] = ['intercept.%s' % endog_name] + ['L1.%s->%s' % (other_name, endog_name) for other_name in self.endog_names]
        return state_names.ravel().tolist()
mod = TVPVAR(y)
initial_obs_cov = np.cov(y.T)
initial_state_cov_diag = [0.01] * mod.k_states
mod.update_variances(initial_obs_cov, initial_state_cov_diag)
initial_res = mod.smooth([])

def plot_coefficients_by_equation(states):
    if False:
        i = 10
        return i + 15
    (fig, axes) = plt.subplots(2, 2, figsize=(15, 8))
    ax = axes[0, 0]
    states.iloc[:, :5].plot(ax=ax)
    ax.set_title('GDP growth')
    ax.legend()
    ax = axes[0, 1]
    states.iloc[:, 5:10].plot(ax=ax)
    ax.set_title('Inflation rate')
    ax.legend()
    ax = axes[1, 0]
    states.iloc[:, 10:15].plot(ax=ax)
    ax.set_title('Unemployment equation')
    ax.legend()
    ax = axes[1, 1]
    states.iloc[:, 15:20].plot(ax=ax)
    ax.set_title('Interest rate equation')
    ax.legend()
    return ax
plot_coefficients_by_equation(initial_res.states.smoothed)
v10 = mod.k_endog + 3
S10 = np.eye(mod.k_endog)
vi20 = 6
Si20 = 0.01
niter = 11000
nburn = 1000
store_states = np.zeros((niter + 1, mod.nobs, mod.k_states))
store_obs_cov = np.zeros((niter + 1, mod.k_endog, mod.k_endog))
store_state_cov = np.zeros((niter + 1, mod.k_states))
store_obs_cov[0] = initial_obs_cov
store_state_cov[0] = initial_state_cov_diag
mod.update_variances(store_obs_cov[0], store_state_cov[0])
sim = mod.simulation_smoother(method='cfa')
for i in range(niter):
    mod.update_variances(store_obs_cov[i], store_state_cov[i])
    sim.simulate()
    store_states[i + 1] = sim.simulated_state.T
    fitted = np.matmul(mod['design'].transpose(2, 0, 1), store_states[i + 1][..., None])[..., 0]
    resid = mod.endog - fitted
    store_obs_cov[i + 1] = invwishart.rvs(v10 + mod.nobs, S10 + resid.T @ resid)
    resid = store_states[i + 1, 1:] - store_states[i + 1, :-1]
    sse = np.sum(resid ** 2, axis=0)
    for j in range(mod.k_states):
        rv = invgamma.rvs((vi20 + mod.nobs - 1) / 2, scale=(Si20 + sse[j]) / 2)
        store_state_cov[i + 1, j] = rv
states_posterior_mean = pd.DataFrame(np.mean(store_states[nburn + 1:], axis=0), index=mod._index, columns=mod.state_names)
plot_coefficients_by_equation(states_posterior_mean)
import arviz as az
az_obs_cov = az.convert_to_inference_data({'Var[%s]' % mod.endog_names[i] if i == j else 'Cov[%s, %s]' % (mod.endog_names[i], mod.endog_names[j]): store_obs_cov[nburn + 1:, i, j] for i in range(mod.k_endog) for j in range(i, mod.k_endog)})
az.plot_forest(az_obs_cov, figsize=(8, 7))
az_state_cov = az.convert_to_inference_data({'$\\sigma^2$[%s]' % mod.state_names[i]: store_state_cov[nburn + 1:, i] for i in range(mod.k_states)})
az.plot_forest(az_state_cov, figsize=(8, 7))
from statsmodels.tsa.statespace.simulation_smoother import SIMULATION_STATE
sim_cfa = mod.simulation_smoother(method='cfa')
sim_kfs = mod.simulation_smoother(simulation_output=SIMULATION_STATE)