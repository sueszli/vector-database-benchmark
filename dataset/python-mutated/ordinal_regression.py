import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
url = 'https://stats.idre.ucla.edu/stat/data/ologit.dta'
data_student = pd.read_stata(url)
data_student.head(5)
data_student.dtypes
data_student['apply'].dtype
mod_prob = OrderedModel(data_student['apply'], data_student[['pared', 'public', 'gpa']], distr='probit')
res_prob = mod_prob.fit(method='bfgs')
res_prob.summary()
num_of_thresholds = 2
mod_prob.transform_threshold_params(res_prob.params[-num_of_thresholds:])
mod_log = OrderedModel(data_student['apply'], data_student[['pared', 'public', 'gpa']], distr='logit')
res_log = mod_log.fit(method='bfgs', disp=False)
res_log.summary()
predicted = res_log.model.predict(res_log.params, exog=data_student[['pared', 'public', 'gpa']])
predicted
pred_choice = predicted.argmax(1)
print('Fraction of correct choice predictions')
print((np.asarray(data_student['apply'].values.codes) == pred_choice).mean())
res_exp = OrderedModel(data_student['apply'], data_student[['pared', 'public', 'gpa']], distr=stats.expon).fit(method='bfgs', disp=False)
res_exp.summary()

class CLogLog(stats.rv_continuous):

    def _ppf(self, q):
        if False:
            i = 10
            return i + 15
        return np.log(-np.log(1 - q))

    def _cdf(self, x):
        if False:
            return 10
        return 1 - np.exp(-np.exp(x))
cloglog = CLogLog()
res_cloglog = OrderedModel(data_student['apply'], data_student[['pared', 'public', 'gpa']], distr=cloglog).fit(method='bfgs', disp=False)
res_cloglog.summary()
modf_logit = OrderedModel.from_formula('apply ~ 0 + pared + public + gpa', data_student, distr='logit')
resf_logit = modf_logit.fit(method='bfgs')
resf_logit.summary()
data_student['apply_codes'] = data_student['apply'].cat.codes * 2 + 5
data_student['apply_codes'].head()
OrderedModel.from_formula('apply_codes ~ 0 + pared + public + gpa', data_student, distr='logit').fit().summary()
resf_logit.predict(data_student.iloc[:5])
data_student['apply_str'] = np.asarray(data_student['apply'])
data_student['apply_str'].head()
data_student.apply_str = pd.Categorical(data_student.apply_str, ordered=True)
data_student.public = data_student.public.astype(float)
data_student.pared = data_student.pared.astype(float)
OrderedModel.from_formula('apply_str ~ 0 + pared + public + gpa', data_student, distr='logit')
nobs = len(data_student)
data_student['dummy'] = (np.arange(nobs) < nobs / 2).astype(float)
modfd_logit = OrderedModel.from_formula('apply ~ 1 + pared + public + gpa + C(dummy)', data_student, distr='logit')
resfd_logit = modfd_logit.fit(method='bfgs')
print(resfd_logit.summary())
modfd_logit.k_vars
modfd_logit.k_constant
modfd2_logit = OrderedModel.from_formula('apply ~ 0 + pared + public + gpa + C(dummy)', data_student, distr='logit', hasconst=False)
resfd2_logit = modfd2_logit.fit(method='bfgs')
print(resfd2_logit.summary())
resfd2_logit.predict(data_student.iloc[:5])
resf_logit.predict()
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
mask_drop = data_student['apply'] == 'somewhat likely'
data2 = data_student.loc[~mask_drop, :]
data2['apply'].cat.remove_categories('somewhat likely', inplace=True)
data2.head()
mod_log = OrderedModel(data2['apply'], data2[['pared', 'public', 'gpa']], distr='logit')
res_log = mod_log.fit(method='bfgs', disp=False)
res_log.summary()
ex = add_constant(data2[['pared', 'public', 'gpa']], prepend=False)
mod_logit = Logit(data2['apply'].cat.codes, ex)
res_logit = mod_logit.fit(method='bfgs', disp=False)
res_logit.summary()
res_logit_hac = mod_logit.fit(method='bfgs', disp=False, cov_type='hac', cov_kwds={'maxlags': 2})
res_log_hac = mod_log.fit(method='bfgs', disp=False, cov_type='hac', cov_kwds={'maxlags': 2})
res_logit_hac.bse.values - res_log_hac.bse