import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
print(sm.datasets.fair.NOTE)
data = sm.datasets.fair.load_pandas().data
data.describe()
data[:3]
data['affairs'] = np.ceil(data['affairs'])
data[:3]
(data['affairs'] == 0).mean()
np.bincount(data['affairs'].astype(int))
data2 = data.copy()
data2['const'] = 1
dc = data2['affairs rate_marriage age yrs_married const'.split()].groupby('affairs rate_marriage age yrs_married'.split()).count()
dc.reset_index(inplace=True)
dc.rename(columns={'const': 'freq'}, inplace=True)
print(dc.shape)
dc.head()
gr = data['affairs rate_marriage age yrs_married'.split()].groupby('rate_marriage age yrs_married'.split())
df_a = gr.agg(['mean', 'sum', 'count'])

def merge_tuple(tpl):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(tpl, tuple) and len(tpl) > 1:
        return '_'.join(map(str, tpl))
    else:
        return tpl
df_a.columns = df_a.columns.map(merge_tuple)
df_a.reset_index(inplace=True)
print(df_a.shape)
df_a.head()
print('number of rows: \noriginal, with unique observations, with unique exog')
(data.shape[0], dc.shape[0], df_a.shape[0])
glm = smf.glm('affairs ~ rate_marriage + age + yrs_married', data=data, family=sm.families.Poisson())
res_o = glm.fit()
print(res_o.summary())
res_o.pearson_chi2 / res_o.df_resid
glm = smf.glm('affairs ~ rate_marriage + age + yrs_married', data=dc, family=sm.families.Poisson(), freq_weights=np.asarray(dc['freq']))
res_f = glm.fit()
print(res_f.summary())
res_f.pearson_chi2 / res_f.df_resid
glm = smf.glm('affairs ~ rate_marriage + age + yrs_married', data=dc, family=sm.families.Poisson(), var_weights=np.asarray(dc['freq']))
res_fv = glm.fit()
print(res_fv.summary())
(res_fv.pearson_chi2 / res_fv.df_resid, res_f.pearson_chi2 / res_f.df_resid)
glm = smf.glm('affairs_sum ~ rate_marriage + age + yrs_married', data=df_a, family=sm.families.Poisson(), exposure=np.asarray(df_a['affairs_count']))
res_e = glm.fit()
print(res_e.summary())
res_e.pearson_chi2 / res_e.df_resid
glm = smf.glm('affairs_mean ~ rate_marriage + age + yrs_married', data=df_a, family=sm.families.Poisson(), var_weights=np.asarray(df_a['affairs_count']))
res_a = glm.fit()
print(res_a.summary())
results_all = [res_o, res_f, res_e, res_a]
names = 'res_o res_f res_e res_a'.split()
pd.concat([r.params for r in results_all], axis=1, keys=names)
pd.concat([r.bse for r in results_all], axis=1, keys=names)
pd.concat([r.pvalues for r in results_all], axis=1, keys=names)
pd.DataFrame(np.column_stack([[r.llf, r.deviance, r.pearson_chi2] for r in results_all]), columns=names, index=['llf', 'deviance', 'pearson chi2'])
glm = smf.glm('affairs ~ rate_marriage + yrs_married', data=data, family=sm.families.Poisson())
res_o2 = glm.fit()
(res_o2.pearson_chi2 - res_o.pearson_chi2, res_o2.deviance - res_o.deviance, res_o2.llf - res_o.llf)
glm = smf.glm('affairs ~ rate_marriage + yrs_married', data=dc, family=sm.families.Poisson(), freq_weights=np.asarray(dc['freq']))
res_f2 = glm.fit()
(res_f2.pearson_chi2 - res_f.pearson_chi2, res_f2.deviance - res_f.deviance, res_f2.llf - res_f.llf)
glm = smf.glm('affairs_sum ~ rate_marriage + yrs_married', data=df_a, family=sm.families.Poisson(), exposure=np.asarray(df_a['affairs_count']))
res_e2 = glm.fit()
(res_e2.pearson_chi2 - res_e.pearson_chi2, res_e2.deviance - res_e.deviance, res_e2.llf - res_e.llf)
glm = smf.glm('affairs_mean ~ rate_marriage + yrs_married', data=df_a, family=sm.families.Poisson(), var_weights=np.asarray(df_a['affairs_count']))
res_a2 = glm.fit()
(res_a2.pearson_chi2 - res_a.pearson_chi2, res_a2.deviance - res_a.deviance, res_a2.llf - res_a.llf)
(res_e2.pearson_chi2, res_e.pearson_chi2, (res_e2.resid_pearson ** 2).sum(), (res_e.resid_pearson ** 2).sum())
(res_e._results.resid_response.mean(), res_e.model.family.variance(res_e.mu)[:5], res_e.mu[:5])
(res_e._results.resid_response ** 2 / res_e.model.family.variance(res_e.mu)).sum()
(res_e2._results.resid_response.mean(), res_e2.model.family.variance(res_e2.mu)[:5], res_e2.mu[:5])
(res_e2._results.resid_response ** 2 / res_e2.model.family.variance(res_e2.mu)).sum()
((res_e2._results.resid_response ** 2).sum(), (res_e._results.resid_response ** 2).sum())
((res_e2._results.resid_response ** 2 - res_e._results.resid_response ** 2) / res_e2.model.family.variance(res_e2.mu)).sum()
((res_a2._results.resid_response ** 2 - res_a._results.resid_response ** 2) / res_a2.model.family.variance(res_a2.mu) * res_a2.model.var_weights).sum()
((res_f2._results.resid_response ** 2 - res_f._results.resid_response ** 2) / res_f2.model.family.variance(res_f2.mu) * res_f2.model.freq_weights).sum()
((res_o2._results.resid_response ** 2 - res_o._results.resid_response ** 2) / res_o2.model.family.variance(res_o2.mu)).sum()
(np.exp(res_e2.model.exposure)[:5], np.asarray(df_a['affairs_count'])[:5])
res_e2.resid_pearson.sum() - res_e.resid_pearson.sum()
res_e2.mu[:5]
(res_a2.pearson_chi2, res_a.pearson_chi2, res_a2.resid_pearson.sum(), res_a.resid_pearson.sum())
(res_a2._results.resid_response ** 2 / res_a2.model.family.variance(res_a2.mu) * res_a2.model.var_weights).sum()
(res_a._results.resid_response ** 2 / res_a.model.family.variance(res_a.mu) * res_a.model.var_weights).sum()
(res_a._results.resid_response ** 2 / res_a.model.family.variance(res_a2.mu) * res_a.model.var_weights).sum()
(res_e.model.endog[:5], res_e2.model.endog[:5])
(res_a.model.endog[:5], res_a2.model.endog[:5])
res_a2.model.endog[:5] * np.exp(res_e2.model.exposure)[:5]
res_a2.model.endog[:5] * res_a2.model.var_weights[:5]
from scipy import stats
(stats.chi2.sf(27.19530754604785, 1), stats.chi2.sf(29.083798806764687, 1))
res_o.pvalues
print(res_e2.summary())
print(res_e.summary())
print(res_f2.summary())
print(res_f.summary())