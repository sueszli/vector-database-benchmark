from functools import partial
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.tools.numdiff import approx_fprime, approx_hess
data = sm.datasets.spector.load()
data.exog = sm.add_constant(data.exog, prepend=False)
probit_mod = sm.Probit(data.endog, data.exog)
probit_res = probit_mod.fit()
loglike = probit_mod.loglike
score = probit_mod.score
mod = GenericLikelihoodModel(data.endog, data.exog * 2, loglike, score)
res = mod.fit(method='nm', maxiter=500)

def probitloglike(params, endog, exog):
    if False:
        print('Hello World!')
    '\n    Log likelihood for the probit\n    '
    q = 2 * endog - 1
    X = exog
    return np.add.reduce(stats.norm.logcdf(q * np.dot(X, params)))
model_loglike = partial(probitloglike, endog=data.endog, exog=data.exog)
mod = GenericLikelihoodModel(data.endog, data.exog, loglike=model_loglike)
res = mod.fit(method='nm', maxiter=500)
print(res)
np.allclose(res.params, probit_res.params, rtol=0.0001)
print(res.params, probit_res.params)
datal = sm.datasets.ccard.load()
datal.exog = sm.add_constant(datal.exog, prepend=False)
nobs = 5000
rvs = np.random.randn(nobs, 6)
datal.exog = rvs[:, :-1]
datal.exog = sm.add_constant(datal.exog, prepend=False)
datal.endog = 1 + rvs.sum(1)
show_error = False
show_error2 = 1
if show_error:

    def loglike_norm_xb(self, params):
        if False:
            for i in range(10):
                print('nop')
        beta = params[:-1]
        sigma = params[-1]
        xb = np.dot(self.exog, beta)
        return stats.norm.logpdf(self.endog, loc=xb, scale=sigma)
    mod_norm = GenericLikelihoodModel(datal.endog, datal.exog, loglike_norm_xb)
    res_norm = mod_norm.fit(method='nm', maxiter=500)
    print(res_norm.params)
if show_error2:

    def loglike_norm_xb(params, endog, exog):
        if False:
            return 10
        beta = params[:-1]
        sigma = params[-1]
        xb = np.dot(exog, beta)
        return stats.norm.logpdf(endog, loc=xb, scale=sigma).sum()
    model_loglike3 = partial(loglike_norm_xb, endog=datal.endog, exog=datal.exog)
    mod_norm = GenericLikelihoodModel(datal.endog, datal.exog, model_loglike3)
    res_norm = mod_norm.fit(start_params=np.ones(datal.exog.shape[1] + 1), method='nm', maxiter=5000)
    print(res_norm.params)

class MygMLE(GenericLikelihoodModel):

    def loglike(self, params):
        if False:
            return 10
        beta = params[:-1]
        sigma = params[-1]
        xb = np.dot(self.exog, beta)
        return stats.norm.logpdf(self.endog, loc=xb, scale=sigma).sum()

    def loglikeobs(self, params):
        if False:
            return 10
        beta = params[:-1]
        sigma = params[-1]
        xb = np.dot(self.exog, beta)
        return stats.norm.logpdf(self.endog, loc=xb, scale=sigma)
mod_norm2 = MygMLE(datal.endog, datal.exog)
res_norm2 = mod_norm2.fit(start_params=[1.0] * datal.exog.shape[1] + [1], method='nm', maxiter=500)
np.allclose(res_norm.params, res_norm2.params)
print(res_norm2.params)
res2 = sm.OLS(datal.endog, datal.exog).fit()
start_params = np.hstack((res2.params, np.sqrt(res2.mse_resid)))
res_norm3 = mod_norm2.fit(start_params=start_params, method='nm', maxiter=500, retall=0)
print(start_params)
print(res_norm3.params)
print(res2.bse)
print(res_norm3.bse)
print('llf', res2.llf, res_norm3.llf)
bse = np.sqrt(np.diag(np.linalg.inv(res_norm3.model.hessian(res_norm3.params))))
res_norm3.model.score(res_norm3.params)
res_bfgs = mod_norm2.fit(start_params=start_params, method='bfgs', fprime=None, maxiter=500, retall=0)
hb = -approx_hess(res_norm3.params, mod_norm2.loglike, epsilon=-0.0001)
hf = -approx_hess(res_norm3.params, mod_norm2.loglike, epsilon=0.0001)
hh = (hf + hb) / 2.0
print(np.linalg.eigh(hh))
grad = -approx_fprime(res_norm3.params, mod_norm2.loglike, epsilon=-0.0001)
print(grad)
gradb = -approx_fprime(res_norm3.params, mod_norm2.loglike, epsilon=-0.0001)
gradf = -approx_fprime(res_norm3.params, mod_norm2.loglike, epsilon=0.0001)
print((gradb + gradf) / 2.0)
print(res_norm3.model.score(res_norm3.params))
print(res_norm3.model.score(start_params))
mod_norm2.loglike(start_params / 2.0)
print(np.linalg.inv(-1 * mod_norm2.hessian(res_norm3.params)))
print(np.sqrt(np.diag(res_bfgs.cov_params())))
print(res_norm3.bse)
print('MLE - OLS parameter estimates')
print(res_norm3.params[:-1] - res2.params)
print('bse diff in percent')
print(res_norm3.bse[:-1] / res2.bse * 100.0 - 100)
'\nOptimization terminated successfully.\n         Current function value: 12.818804\n         Iterations 6\nOptimization terminated successfully.\n         Current function value: 12.818804\n         Iterations: 439\n         Function evaluations: 735\nOptimization terminated successfully.\n         Current function value: 12.818804\n         Iterations: 439\n         Function evaluations: 735\n<statsmodels.model.LikelihoodModelResults object at 0x02131290>\n[ 1.6258006   0.05172931  1.42632252 -7.45229732] [ 1.62581004  0.05172895  1.42633234 -7.45231965]\nWarning: Maximum number of function evaluations has been exceeded.\n[  -1.18109149  246.94438535  -16.21235536   24.05282629 -324.80867176\n  274.07378453]\nWarning: Maximum number of iterations has been exceeded\n[  17.57107    -149.87528787   19.89079376  -72.49810777  -50.06067953\n  306.14170418]\nOptimization terminated successfully.\n         Current function value: 506.488765\n         Iterations: 339\n         Function evaluations: 550\n[  -3.08181404  234.34702702  -14.99684418   27.94090839 -237.1465136\n  284.75079529]\n[  -3.08181304  234.34701361  -14.99684381   27.94088692 -237.14649571\n  274.6857294 ]\n[   5.51471653   80.36595035    7.46933695   82.92232357  199.35166485]\nllf -506.488764864 -506.488764864\nOptimization terminated successfully.\n         Current function value: 506.488765\n         Iterations: 9\n         Function evaluations: 13\n         Gradient evaluations: 13\n(array([  2.41772580e-05,   1.62492628e-04,   2.79438138e-04,\n         1.90996240e-03,   2.07117946e-01,   1.28747174e+00]), array([[  1.52225754e-02,   2.01838216e-02,   6.90127235e-02,\n         -2.57002471e-04,  -5.25941060e-01,  -8.47339404e-01],\n       [  2.39797491e-01,  -2.32325602e-01,  -9.36235262e-01,\n          3.02434938e-03,   3.95614029e-02,  -1.02035585e-01],\n       [ -2.11381471e-02,   3.01074776e-02,   7.97208277e-02,\n         -2.94955832e-04,   8.49402362e-01,  -5.20391053e-01],\n       [ -1.55821981e-01,  -9.66926643e-01,   2.01517298e-01,\n          1.52397702e-03,   4.13805882e-03,  -1.19878714e-02],\n       [ -9.57881586e-01,   9.87911166e-02,  -2.67819451e-01,\n          1.55192932e-03,  -1.78717579e-02,  -2.55757014e-02],\n       [ -9.96486655e-04,  -2.03697290e-03,  -2.98130314e-03,\n         -9.99992985e-01,  -1.71500426e-05,   4.70854949e-06]]))\n[[ -4.91007768e-05  -7.28732630e-07  -2.51941401e-05  -2.50111043e-08\n   -4.77484718e-08  -9.72022463e-08]]\n[[ -1.64845915e-08  -2.87059265e-08  -2.88764568e-07  -6.82121026e-09\n    2.84217094e-10  -1.70530257e-09]]\n[ -4.90678076e-05  -6.71320777e-07  -2.46166110e-05  -1.13686838e-08\n  -4.83169060e-08  -9.37916411e-08]\n[ -4.56753924e-05  -6.50857146e-07  -2.31756303e-05  -1.70530257e-08\n  -4.43378667e-08  -1.75592936e-02]\n[[  2.99386348e+01  -1.24442928e+02   9.67254672e+00  -1.58968536e+02\n   -5.91960010e+02  -2.48738183e+00]\n [ -1.24442928e+02   5.62972166e+03  -5.00079203e+02  -7.13057475e+02\n   -7.82440674e+03  -1.05126925e+01]\n [  9.67254672e+00  -5.00079203e+02   4.87472259e+01   3.37373299e+00\n    6.96960872e+02   7.69866589e-01]\n [ -1.58968536e+02  -7.13057475e+02   3.37373299e+00   6.82417837e+03\n    4.84485862e+03   3.21440021e+01]\n [ -5.91960010e+02  -7.82440674e+03   6.96960872e+02   4.84485862e+03\n    3.43753691e+04   9.37524459e+01]\n [ -2.48738183e+00  -1.05126925e+01   7.69866589e-01   3.21440021e+01\n    9.37524459e+01   5.23915258e+02]]\n>>> res_norm3.bse\narray([   5.47162086,   75.03147114,    6.98192136,   82.60858536,\n        185.40595756,   22.88919522])\n>>> print res_norm3.model.score(res_norm3.params)\n[ -4.90678076e-05  -6.71320777e-07  -2.46166110e-05  -1.13686838e-08\n  -4.83169060e-08  -9.37916411e-08]\n>>> print res_norm3.model.score(start_params)\n[ -4.56753924e-05  -6.50857146e-07  -2.31756303e-05  -1.70530257e-08\n  -4.43378667e-08  -1.75592936e-02]\n>>> mod_norm2.loglike(start_params/2.)\n-598.56178102781314\n>>> print np.linalg.inv(-1*mod_norm2.hessian(res_norm3.params))\n[[  2.99386348e+01  -1.24442928e+02   9.67254672e+00  -1.58968536e+02\n   -5.91960010e+02  -2.48738183e+00]\n [ -1.24442928e+02   5.62972166e+03  -5.00079203e+02  -7.13057475e+02\n   -7.82440674e+03  -1.05126925e+01]\n [  9.67254672e+00  -5.00079203e+02   4.87472259e+01   3.37373299e+00\n    6.96960872e+02   7.69866589e-01]\n [ -1.58968536e+02  -7.13057475e+02   3.37373299e+00   6.82417837e+03\n    4.84485862e+03   3.21440021e+01]\n [ -5.91960010e+02  -7.82440674e+03   6.96960872e+02   4.84485862e+03\n    3.43753691e+04   9.37524459e+01]\n [ -2.48738183e+00  -1.05126925e+01   7.69866589e-01   3.21440021e+01\n    9.37524459e+01   5.23915258e+02]]\n>>> print np.sqrt(np.diag(res_bfgs.cov_params()))\n[   5.10032831   74.34988912    6.96522122   76.7091604   169.8117832\n   22.91695494]\n>>> print res_norm3.bse\n[   5.47162086   75.03147114    6.98192136   82.60858536  185.40595756\n   22.88919522]\n>>> res_norm3.conf_int\n<bound method LikelihoodModelResults.conf_int of <statsmodels.model.LikelihoodModelResults object at 0x021317F0>>\n>>> res_norm3.conf_int()\narray([[0.96421437, 1.01999835],\n       [0.99251725, 1.04863332],\n       [0.95721328, 1.01246222],\n       [0.97134549, 1.02695393],\n       [0.97050081, 1.02660988],\n       [0.97773434, 1.03290028],\n       [0.97529207, 1.01428874]])\n\n>>> res_norm3.params\narray([  -3.08181304,  234.34701361,  -14.99684381,   27.94088692,\n       -237.14649571,  274.6857294 ])\n>>> res2.params\narray([  -3.08181404,  234.34702702,  -14.99684418,   27.94090839,\n       -237.1465136 ])\n>>>\n>>> res_norm3.params - res2.params\nTraceback (most recent call last):\n  File "<stdin>", line 1, in <module>\nValueError: shape mismatch: objects cannot be broadcast to a single shape\n\n>>> res_norm3.params[:-1] - res2.params\narray([  9.96859735e-07,  -1.34122981e-05,   3.72278400e-07,\n        -2.14645839e-05,   1.78919019e-05])\n>>>\n>>> res_norm3.bse[:-1] - res2.bse\narray([ -0.04309567,  -5.33447922,  -0.48741559,  -0.31373822, -13.94570729])\n>>> (res_norm3.bse[:-1] / res2.bse) - 1\narray([-0.00781467, -0.06637735, -0.06525554, -0.00378352, -0.06995531])\n>>> (res_norm3.bse[:-1] / res2.bse)*100. - 100\narray([-0.7814667 , -6.6377355 , -6.52555369, -0.37835193, -6.99553089])\n>>> np.sqrt(np.diag(np.linalg.inv(res_norm3.model.hessian(res_bfgs.params))))\narray([ NaN,  NaN,  NaN,  NaN,  NaN,  NaN])\n>>> np.sqrt(np.diag(np.linalg.inv(-res_norm3.model.hessian(res_bfgs.params))))\narray([   5.10032831,   74.34988912,    6.96522122,   76.7091604 ,\n        169.8117832 ,   22.91695494])\n>>> res_norm3.bse\narray([   5.47162086,   75.03147114,    6.98192136,   82.60858536,\n        185.40595756,   22.88919522])\n>>> res2.bse\narray([   5.51471653,   80.36595035,    7.46933695,   82.92232357,\n        199.35166485])\n>>>\n>>> bse_bfgs = np.sqrt(np.diag(np.linalg.inv(-res_norm3.model.hessian(res_bfgs.params))))\n>>> (bse_bfgs[:-1] / res2.bse)*100. - 100\narray([ -7.51422527,  -7.4858335 ,  -6.74913633,  -7.49275094, -14.8179759 ])\n>>> hb=-approx_hess(res_bfgs.params, mod_norm2.loglike, epsilon=-1e-4)\n>>> hf=-approx_hess(res_bfgs.params, mod_norm2.loglike, epsilon=1e-4)\n>>> hh = (hf+hb)/2.\n>>> bse_bfgs = np.sqrt(np.diag(np.linalg.inv(-hh)))\n>>> bse_bfgs\narray([ NaN,  NaN,  NaN,  NaN,  NaN,  NaN])\n>>> bse_bfgs = np.sqrt(np.diag(np.linalg.inv(hh)))\n>>> np.diag(hh)\narray([  9.81680159e-01,   1.39920076e-02,   4.98101826e-01,\n         3.60955710e-04,   9.57811608e-04,   1.90709670e-03])\n>>> np.diag(np.inv(hh))\nTraceback (most recent call last):\n  File "<stdin>", line 1, in <module>\nAttributeError: \'module\' object has no attribute \'inv\'\n\n>>> np.diag(np.linalg.inv(hh))\narray([  2.64875153e+01,   5.91578496e+03,   5.13279911e+01,\n         6.11533345e+03,   3.33775960e+04,   5.24357391e+02])\n>>> res2.bse**2\narray([  3.04120984e+01,   6.45868598e+03,   5.57909945e+01,\n         6.87611175e+03,   3.97410863e+04])\n>>> bse_bfgs\narray([   5.14660231,   76.91414015,    7.1643556 ,   78.20059751,\n        182.69536402,   22.89885131])\n>>> bse_bfgs - res_norm3.bse\narray([-0.32501855,  1.88266901,  0.18243424, -4.40798785, -2.71059354,\n        0.00965609])\n>>> (bse_bfgs[:-1] / res2.bse)*100. - 100\narray([-6.67512508, -4.29511526, -4.0831115 , -5.69415552, -8.35523538])\n>>> (res_norm3.bse[:-1] / res2.bse)*100. - 100\narray([-0.7814667 , -6.6377355 , -6.52555369, -0.37835193, -6.99553089])\n>>> (bse_bfgs / res_norm3.bse)*100. - 100\narray([-5.94007812,  2.50917247,  2.61295176, -5.33599242, -1.46197759,\n        0.04218624])\n>>> bse_bfgs\narray([   5.14660231,   76.91414015,    7.1643556 ,   78.20059751,\n        182.69536402,   22.89885131])\n>>> res_norm3.bse\narray([   5.47162086,   75.03147114,    6.98192136,   82.60858536,\n        185.40595756,   22.88919522])\n>>> res2.bse\narray([   5.51471653,   80.36595035,    7.46933695,   82.92232357,\n        199.35166485])\n>>> dir(res_bfgs)\n[\'__class__\', \'__delattr__\', \'__dict__\', \'__doc__\', \'__getattribute__\', \'__hash__\', \'__init__\', \'__module__\', \'__new__\', \'__reduce__\', \'__reduce_ex__\', \'__repr__\', \'__setattr__\', \'__str__\', \'__weakref__\', \'bse\', \'conf_int\', \'cov_params\', \'f_test\', \'initialize\', \'llf\', \'mle_retvals\', \'mle_settings\', \'model\', \'normalized_cov_params\', \'params\', \'scale\', \'t\', \'t_test\']\n>>> res_bfgs.scale\n1.0\n>>> res2.scale\n81083.015420213851\n>>> res2.mse_resid\n81083.015420213851\n>>> print np.sqrt(np.diag(np.linalg.inv(-1*mod_norm2.hessian(res_bfgs.params))))\n[   5.10032831   74.34988912    6.96522122   76.7091604   169.8117832\n   22.91695494]\n>>> print np.sqrt(np.diag(np.linalg.inv(-1*res_bfgs.model.hessian(res_bfgs.params))))\n[   5.10032831   74.34988912    6.96522122   76.7091604   169.8117832\n   22.91695494]\n\nIs scale a misnomer, actually scale squared, i.e. variance of error term ?\n'
print(res_norm3.model.score_obs(res_norm3.params).shape)
jac = res_norm3.model.score_obs(res_norm3.params)
print(np.sqrt(np.diag(np.dot(jac.T, jac))) / start_params)
jac2 = res_norm3.model.score_obs(res_norm3.params, centered=True)
print(np.sqrt(np.diag(np.linalg.inv(np.dot(jac.T, jac)))))
print(res_norm3.bse)
print(res2.bse)