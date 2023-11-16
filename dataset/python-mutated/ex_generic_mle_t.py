"""
Created on Wed Jul 28 08:28:04 2010

Author: josef-pktd
"""
import numpy as np
from scipy import special
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.tools.numdiff import approx_hess
np_log = np.log
np_pi = np.pi
sps_gamln = special.gammaln

def maxabs(arr1, arr2):
    if False:
        return 10
    return np.max(np.abs(arr1 - arr2))

def maxabsrel(arr1, arr2):
    if False:
        while True:
            i = 10
    return np.max(np.abs(arr2 / arr1 - 1))

class MyT(GenericLikelihoodModel):
    """Maximum Likelihood Estimation of Poisson Model

    This is an example for generic MLE which has the same
    statistical model as discretemod.Poisson.

    Except for defining the negative log-likelihood method, all
    methods and results are generic. Gradients and Hessian
    and all resulting statistics are based on numerical
    differentiation.

    """

    def loglike(self, params):
        if False:
            return 10
        return -self.nloglikeobs(params).sum(0)

    def nloglikeobs(self, params):
        if False:
            i = 10
            return i + 15
        '\n        Loglikelihood of Poisson model\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model.\n\n        Returns\n        -------\n        The log likelihood of the model evaluated at `params`\n\n        Notes\n        -----\n        .. math:: \\ln L=\\sum_{i=1}^{n}\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]\n        '
        beta = params[:-2]
        df = params[-2]
        scale = params[-1]
        loc = np.dot(self.exog, beta)
        endog = self.endog
        x = (endog - loc) / scale
        lPx = sps_gamln((df + 1) / 2) - sps_gamln(df / 2.0)
        lPx -= 0.5 * np_log(df * np_pi) + (df + 1) / 2.0 * np_log(1 + x ** 2 / df)
        lPx -= np_log(scale)
        return -lPx
np.random.seed(98765678)
nobs = 1000
rvs = np.random.randn(nobs, 5)
data_exog = sm.add_constant(rvs, prepend=False)
xbeta = 0.9 + 0.1 * rvs.sum(1)
data_endog = xbeta + 0.1 * np.random.standard_t(5, size=nobs)
modp = MyT(data_endog, data_exog)
modp.start_value = np.ones(data_exog.shape[1] + 2)
modp.start_value[-2] = 10
modp.start_params = modp.start_value
resp = modp.fit(start_params=modp.start_value)
print(resp.params)
print(resp.bse)
hb = -approx_hess(modp.start_value, modp.loglike, epsilon=-0.0001)
tmp = modp.loglike(modp.start_value)
print(tmp.shape)
'\n>>> tmp = modp.loglike(modp.start_value)\n8\n>>> tmp.shape\n(100,)\n>>> tmp.sum(0)\n-24220.877108016182\n>>> tmp = modp.nloglikeobs(modp.start_value)\n8\n>>> tmp.shape\n(100, 100)\n\n>>> params = modp.start_value\n>>> beta = params[:-2]\n>>> beta.shape\n(6,)\n>>> np.dot(modp.exog, beta).shape\n(100,)\n>>> modp.endog.shape\n(100, 100)\n>>> xbeta.shape\n(100,)\n>>>\n'
'\nrepr(start_params) array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])\nOptimization terminated successfully.\n         Current function value: 91.897859\n         Iterations: 108\n         Function evaluations: 173\n         Gradient evaluations: 173\n[  1.58253308e-01   1.73188603e-01   1.77357447e-01   2.06707494e-02\n  -1.31174789e-01   8.79915580e-01   6.47663840e+03   6.73457641e+02]\n[         NaN          NaN          NaN          NaN          NaN\n  28.26906182          NaN          NaN]\n()\n>>> resp.params\narray([  1.58253308e-01,   1.73188603e-01,   1.77357447e-01,\n         2.06707494e-02,  -1.31174789e-01,   8.79915580e-01,\n         6.47663840e+03,   6.73457641e+02])\n>>> resp.bse\narray([         NaN,          NaN,          NaN,          NaN,\n                NaN,  28.26906182,          NaN,          NaN])\n>>> resp.jac\nTraceback (most recent call last):\n  File "<stdin>", line 1, in <module>\nAttributeError: \'GenericLikelihoodModelResults\' object has no attribute \'jac\'\n\n>>> resp.bsejac\narray([    45243.35919908,     51997.80776897,     41418.33021984,\n           42763.46575168,     50101.91631612,     42804.92083525,\n         3005625.35649203,  13826948.68708931])\n>>> resp.bsejhj\narray([ 1.51643931,  0.80229636,  0.27720185,  0.4711138 ,  0.9028682 ,\n        0.31673747,  0.00524426,  0.69729368])\n>>> resp.covjac\narray([[  2.04696155e+09,   1.46643494e+08,   7.59932781e+06,\n         -2.39993397e+08,   5.62644255e+08,   2.34300598e+08,\n         -3.07824799e+09,  -1.93425470e+10],\n       [  1.46643494e+08,   2.70377201e+09,   1.06005712e+08,\n          3.76824011e+08,  -1.21778986e+08,   5.38612723e+08,\n         -2.12575784e+10,  -1.69503271e+11],\n       [  7.59932781e+06,   1.06005712e+08,   1.71547808e+09,\n         -5.94451158e+07,  -1.44586401e+08,  -5.41830441e+06,\n          1.25899515e+10,   1.06372065e+11],\n       [ -2.39993397e+08,   3.76824011e+08,  -5.94451158e+07,\n          1.82871400e+09,  -5.66930891e+08,   3.75061111e+08,\n         -6.84681772e+09,  -7.29993789e+10],\n       [  5.62644255e+08,  -1.21778986e+08,  -1.44586401e+08,\n         -5.66930891e+08,   2.51020202e+09,  -4.67886982e+08,\n          1.78890380e+10,   1.75428694e+11],\n       [  2.34300598e+08,   5.38612723e+08,  -5.41830441e+06,\n          3.75061111e+08,  -4.67886982e+08,   1.83226125e+09,\n         -1.27484996e+10,  -1.12550321e+11],\n       [ -3.07824799e+09,  -2.12575784e+10,   1.25899515e+10,\n         -6.84681772e+09,   1.78890380e+10,  -1.27484996e+10,\n          9.03378378e+12,   2.15188047e+13],\n       [ -1.93425470e+10,  -1.69503271e+11,   1.06372065e+11,\n         -7.29993789e+10,   1.75428694e+11,  -1.12550321e+11,\n          2.15188047e+13,   1.91184510e+14]])\n>>> hb\narray([[  33.68732564,   -2.33209221,  -13.51255321,   -1.60840159,\n         -13.03920385,   -9.3506543 ,    4.86239173,   -9.30409101],\n       [  -2.33209221,    3.12512611,   -6.08530968,   -6.79232244,\n           3.66804898,    1.26497071,    5.10113409,   -2.53482995],\n       [ -13.51255321,   -6.08530968,   31.14883498,   -5.01514705,\n         -10.48819911,   -2.62533035,    3.82241581,  -12.51046342],\n       [  -1.60840159,   -6.79232244,   -5.01514705,   28.40141917,\n          -8.72489636,   -8.82449456,    5.47584023,  -18.20500017],\n       [ -13.03920385,    3.66804898,  -10.48819911,   -8.72489636,\n           9.03650914,    3.65206176,    6.55926726,   -1.8233635 ],\n       [  -9.3506543 ,    1.26497071,   -2.62533035,   -8.82449456,\n           3.65206176,   21.41825348,   -1.28610793,    4.28101146],\n       [   4.86239173,    5.10113409,    3.82241581,    5.47584023,\n           6.55926726,   -1.28610793,   46.52354448,  -32.23861427],\n       [  -9.30409101,   -2.53482995,  -12.51046342,  -18.20500017,\n          -1.8233635 ,    4.28101146,  -32.23861427,  178.61978279]])\n>>> np.linalg.eigh(hb)\n(array([ -10.50373649,    0.7460258 ,   14.73131793,   29.72453087,\n         36.24103832,   41.98042979,   48.99815223,  190.04303734]), array([[-0.40303259,  0.10181305,  0.18164206,  0.48201456,  0.03916688,\n         0.00903695,  0.74620692,  0.05853619],\n       [-0.3201713 , -0.88444855, -0.19867642,  0.02828812,  0.16733946,\n        -0.21440765, -0.02927317,  0.01176904],\n       [-0.41847094,  0.00170161,  0.04973298,  0.43276118, -0.55894304,\n         0.26454728, -0.49745582,  0.07251685],\n       [-0.3508729 , -0.08302723,  0.25004884, -0.73495077, -0.38936448,\n         0.20677082,  0.24464779,  0.11448238],\n       [-0.62065653,  0.44662675, -0.37388565, -0.19453047,  0.29084735,\n        -0.34151809, -0.19088978,  0.00342713],\n       [-0.15119802, -0.01099165,  0.84377273,  0.00554863,  0.37332324,\n        -0.17917015, -0.30371283, -0.03635211],\n       [ 0.15813581,  0.0293601 ,  0.09882271,  0.03515962, -0.48768565,\n        -0.81960996,  0.05248464,  0.22533642],\n       [-0.06118044, -0.00549223,  0.03205047, -0.01782649, -0.21128588,\n        -0.14391393,  0.05973658, -0.96226835]]))\n>>> np.linalg.eigh(np.linalg.inv(hb))\n(array([-0.09520422,  0.00526197,  0.02040893,  0.02382062,  0.02759303,\n        0.03364225,  0.06788259,  1.34043621]), array([[-0.40303259,  0.05853619,  0.74620692, -0.00903695, -0.03916688,\n         0.48201456,  0.18164206,  0.10181305],\n       [-0.3201713 ,  0.01176904, -0.02927317,  0.21440765, -0.16733946,\n         0.02828812, -0.19867642, -0.88444855],\n       [-0.41847094,  0.07251685, -0.49745582, -0.26454728,  0.55894304,\n         0.43276118,  0.04973298,  0.00170161],\n       [-0.3508729 ,  0.11448238,  0.24464779, -0.20677082,  0.38936448,\n        -0.73495077,  0.25004884, -0.08302723],\n       [-0.62065653,  0.00342713, -0.19088978,  0.34151809, -0.29084735,\n        -0.19453047, -0.37388565,  0.44662675],\n       [-0.15119802, -0.03635211, -0.30371283,  0.17917015, -0.37332324,\n         0.00554863,  0.84377273, -0.01099165],\n       [ 0.15813581,  0.22533642,  0.05248464,  0.81960996,  0.48768565,\n         0.03515962,  0.09882271,  0.0293601 ],\n       [-0.06118044, -0.96226835,  0.05973658,  0.14391393,  0.21128588,\n        -0.01782649,  0.03205047, -0.00549223]]))\n>>> np.diag(np.linalg.inv(hb))\narray([ 0.01991288,  1.0433882 ,  0.00516616,  0.02642799,  0.24732871,\n        0.05281555,  0.02236704,  0.00643486])\n>>> np.sqrt(np.diag(np.linalg.inv(hb)))\narray([ 0.14111302,  1.02146375,  0.07187597,  0.16256686,  0.49732154,\n        0.22981633,  0.14955616,  0.08021756])\n>>> hess = modp.hessian(resp.params)\n>>> np.sqrt(np.diag(np.linalg.inv(hess)))\narray([ 231.3823423 ,  117.79508218,   31.46595143,   53.44753106,\n        132.4855704 ,           NaN,    5.47881705,   90.75332693])\n>>> hb=-approx_hess(resp.params, modp.loglike, epsilon=-1e-4)\n>>> np.sqrt(np.diag(np.linalg.inv(hb)))\narray([ 31.93524822,  22.0333515 ,          NaN,  29.90198792,\n        38.82615785,          NaN,          NaN,          NaN])\n>>> hb=-approx_hess(resp.params, modp.loglike, epsilon=-1e-8)\n>>> np.sqrt(np.diag(np.linalg.inv(hb)))\nTraceback (most recent call last):\n  [...]\n    raise LinAlgError, \'Singular matrix\'\nnumpy.linalg.linalg.LinAlgError: Singular matrix\n>>> resp.params\narray([  1.58253308e-01,   1.73188603e-01,   1.77357447e-01,\n         2.06707494e-02,  -1.31174789e-01,   8.79915580e-01,\n         6.47663840e+03,   6.73457641e+02])\n>>>\n'