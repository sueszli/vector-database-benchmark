import numpy as np
import pandas as pd
import statsmodels.api as sm
data = '\n  x   y y_err\n201 592    61\n244 401    25\n 47 583    38\n287 402    15\n203 495    21\n 58 173    15\n210 479    27\n202 504    14\n198 510    30\n158 416    16\n165 393    14\n201 442    25\n157 317    52\n131 311    16\n166 400    34\n160 337    31\n186 423    42\n125 334    26\n218 533    16\n146 344    22\n'
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
data = pd.read_csv(StringIO(data), delim_whitespace=True).astype(float)
data.head()
exog = sm.add_constant(data['x'])
endog = data['y']
weights = 1.0 / data['y_err'] ** 2
wls = sm.WLS(endog, exog, weights)
results = wls.fit(cov_type='fixed scale')
print(results.summary())
from scipy.optimize import curve_fit

def f(x, a, b):
    if False:
        i = 10
        return i + 15
    return a * x + b
xdata = data['x']
ydata = data['y']
p0 = [0, 0]
sigma = data['y_err']
(popt, pcov) = curve_fit(f, xdata, ydata, p0, sigma, absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))
print('a = {0:10.3f} +- {1:10.3f}'.format(popt[0], perr[0]))
print('b = {0:10.3f} +- {1:10.3f}'.format(popt[1], perr[1]))
from scipy.optimize import minimize

def chi2(pars):
    if False:
        return 10
    'Cost function.'
    y_model = pars[0] * data['x'] + pars[1]
    chi = (data['y'] - y_model) / data['y_err']
    return np.sum(chi ** 2)
result = minimize(fun=chi2, x0=[0, 0])
popt = result.x
print('a = {0:10.3f}'.format(popt[0]))
print('b = {0:10.3f}'.format(popt[1]))