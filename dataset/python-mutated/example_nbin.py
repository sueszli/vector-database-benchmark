"""
Author: Vincent Arel-Bundock <varel@umich.edu>
Date: 2012-08-25

This example file implements 5 variations of the negative binomial regression
model for count data: NB-P, NB-1, NB-2, geometric and left-truncated.

The NBin class inherits from the GenericMaximumLikelihood statsmodels class
which provides automatic numerical differentiation for the score and hessian.

NB-1, NB-2 and geometric are implemented as special cases of the NB-P model
described in Greene (2008) Functional forms for the negative binomial model for
count data. Economics Letters, v99n3.

Tests are included to check how NB-1, NB-2 and geometric coefficient estimates
compare to equivalent models in R. Results usually agree up to the 4th digit.

The NB-P and left-truncated model results have not been compared to other
implementations. Note that NB-P appears to only have been implemented in the
LIMDEP software.
"""
from urllib.request import urlopen
import numpy as np
from numpy.testing import assert_almost_equal
from scipy.special import digamma
from scipy.stats import nbinom
import pandas
import patsy
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.base.model import GenericLikelihoodModelResults

def _ll_nbp(y, X, beta, alph, Q):
    if False:
        for i in range(10):
            print('nop')
    '\n    Negative Binomial Log-likelihood -- type P\n\n    References:\n\n    Greene, W. 2008. "Functional forms for the negative binomial model\n        for count data". Economics Letters. Volume 99, Number 3, pp.585-590.\n    Hilbe, J.M. 2011. "Negative binomial regression". Cambridge University Press.\n\n    Following notation in Greene (2008), with negative binomial heterogeneity\n    parameter :math:`\\alpha`:\n\n    .. math::\n\n        \\lambda_i = exp(X\\beta)\\\\\n        \\theta = 1 / \\alpha \\\\\n        g_i = \\theta \\lambda_i^Q \\\\\n        w_i = g_i/(g_i + \\lambda_i) \\\\\n        r_i = \\theta / (\\theta+\\lambda_i) \\\\\n        ln \\mathcal{L}_i = ln \\Gamma(y_i+g_i) - ln \\Gamma(1+y_i) + g_iln (r_i) + y_i ln(1-r_i)\n    '
    mu = np.exp(np.dot(X, beta))
    size = 1 / alph * mu ** Q
    prob = size / (size + mu)
    ll = nbinom.logpmf(y, size, prob)
    return ll

def _ll_nb1(y, X, beta, alph):
    if False:
        return 10
    'Negative Binomial regression (type 1 likelihood)'
    ll = _ll_nbp(y, X, beta, alph, Q=1)
    return ll

def _ll_nb2(y, X, beta, alph):
    if False:
        return 10
    'Negative Binomial regression (type 2 likelihood)'
    ll = _ll_nbp(y, X, beta, alph, Q=0)
    return ll

def _ll_geom(y, X, beta):
    if False:
        while True:
            i = 10
    'Geometric regression'
    ll = _ll_nbp(y, X, beta, alph=1, Q=0)
    return ll

def _ll_nbt(y, X, beta, alph, C=0):
    if False:
        print('Hello World!')
    '\n    Negative Binomial (truncated)\n\n    Truncated densities for count models (Cameron & Trivedi, 2005, 680):\n\n    .. math::\n\n        f(y|\\beta, y \\geq C+1) = \\frac{f(y|\\beta)}{1-F(C|\\beta)}\n    '
    Q = 0
    mu = np.exp(np.dot(X, beta))
    size = 1 / alph * mu ** Q
    prob = size / (size + mu)
    ll = nbinom.logpmf(y, size, prob) - np.log(1 - nbinom.cdf(C, size, prob))
    return ll

class NBin(GenericLikelihoodModel):
    """
    Negative Binomial regression

    Parameters
    ----------
    endog : array_like
        1-d array of the response variable.
    exog : array_like
        `exog` is an n x p array where n is the number of observations and p
        is the number of regressors including the intercept if one is
        included in the data.
    ll_type: str
        log-likelihood type
        `nb2`: Negative Binomial type-2 (most common)
        `nb1`: Negative Binomial type-1
        `nbp`: Negative Binomial type-P (Greene, 2008)
        `nbt`: Left-truncated Negative Binomial (type-2)
        `geom`: Geometric regression model
    C: int
        Cut-point for `nbt` model
    """

    def __init__(self, endog, exog, ll_type='nb2', C=0, **kwds):
        if False:
            i = 10
            return i + 15
        self.exog = np.array(exog)
        self.endog = np.array(endog)
        self.C = C
        super(NBin, self).__init__(endog, exog, **kwds)
        if ll_type not in ['nb2', 'nb1', 'nbp', 'nbt', 'geom']:
            raise NameError('Valid ll_type are: nb2, nb1, nbp,  nbt, geom')
        self.ll_type = ll_type
        if ll_type == 'geom':
            self.start_params_default = np.zeros(self.exog.shape[1])
        elif ll_type == 'nbp':
            start_mod = NBin(endog, exog, 'nb2')
            start_res = start_mod.fit(disp=False)
            self.start_params_default = np.append(start_res.params, 0)
        else:
            self.start_params_default = np.append(np.zeros(self.exog.shape[1]), 0.5)
        self.start_params_default[0] = np.log(self.endog.mean())
        if ll_type == 'nb1':
            self.ll_func = _ll_nb1
        elif ll_type == 'nb2':
            self.ll_func = _ll_nb2
        elif ll_type == 'geom':
            self.ll_func = _ll_geom
        elif ll_type == 'nbp':
            self.ll_func = _ll_nbp
        elif ll_type == 'nbt':
            self.ll_func = _ll_nbt

    def nloglikeobs(self, params):
        if False:
            while True:
                i = 10
        alph = params[-1]
        beta = params[:self.exog.shape[1]]
        if self.ll_type == 'geom':
            return -self.ll_func(self.endog, self.exog, beta)
        elif self.ll_type == 'nbt':
            return -self.ll_func(self.endog, self.exog, beta, alph, self.C)
        elif self.ll_type == 'nbp':
            Q = params[-2]
            return -self.ll_func(self.endog, self.exog, beta, alph, Q)
        else:
            return -self.ll_func(self.endog, self.exog, beta, alph)

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if False:
            while True:
                i = 10
        if start_params is None:
            countfit = super(NBin, self).fit(start_params=self.start_params_default, maxiter=maxiter, maxfun=maxfun, **kwds)
        else:
            countfit = super(NBin, self).fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds)
        countfit = CountResults(self, countfit)
        return countfit

class CountResults(GenericLikelihoodModelResults):

    def __init__(self, model, mlefit):
        if False:
            for i in range(10):
                print('nop')
        self.model = model
        self.__dict__.update(mlefit.__dict__)

    def summary(self, yname=None, xname=None, title=None, alpha=0.05, yname_list=None):
        if False:
            while True:
                i = 10
        top_left = [('Dep. Variable:', None), ('Model:', [self.model.__class__.__name__]), ('Method:', ['MLE']), ('Date:', None), ('Time:', None), ('Converged:', ['%s' % self.mle_retvals['converged']])]
        top_right = [('No. Observations:', None), ('Log-Likelihood:', None)]
        if title is None:
            title = self.model.__class__.__name__ + ' ' + 'Regression Results'
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right, yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname_list, xname=xname, alpha=alpha, use_t=True)
        return smry

def _score_nbp(y, X, beta, thet, Q):
    if False:
        for i in range(10):
            print('nop')
    '\n    Negative Binomial Score -- type P likelihood from Greene (2007)\n    .. math::\n\n        \\lambda_i = exp(X\\beta)\\\\\n        g_i = \\theta \\lambda_i^Q \\\\\n        w_i = g_i/(g_i + \\lambda_i) \\\\\n        r_i = \\theta / (\\theta+\\lambda_i) \\\\\n        A_i = \\left [ \\Psi(y_i+g_i) - \\Psi(g_i) + ln w_i \\right ] \\\\\n        B_i = \\left [ g_i (1-w_i) - y_iw_i \\right ] \\\\\n        \\partial ln \\mathcal{L}_i / \\partial\n            \\begin{pmatrix} \\lambda_i \\\\ \\theta \\\\ Q \\end{pmatrix}=\n            [A_i+B_i]\n            \\begin{pmatrix} Q/\\lambda_i \\\\ 1/\\theta \\\\ ln(\\lambda_i) \\end{pmatrix}\n            -B_i\n            \\begin{pmatrix} 1/\\lambda_i\\\\ 0 \\\\ 0 \\end{pmatrix} \\\\\n        \\frac{\\partial \\lambda}{\\partial \\beta} = \\lambda_i \\mathbf{x}_i \\\\\n        \\frac{\\partial \\mathcal{L}_i}{\\partial \\beta} =\n            \\left (\\frac{\\partial\\mathcal{L}_i}{\\partial \\lambda_i} \\right )\n            \\frac{\\partial \\lambda_i}{\\partial \\beta}\n    '
    lamb = np.exp(np.dot(X, beta))
    g = thet * lamb ** Q
    w = g / (g + lamb)
    r = thet / (thet + lamb)
    A = digamma(y + g) - digamma(g) + np.log(w)
    B = g * (1 - w) - y * w
    dl = (A + B) * Q / lamb - B * 1 / lamb
    dt = (A + B) * 1 / thet
    dq = (A + B) * np.log(lamb)
    db = X * (dl * lamb)[:, np.newaxis]
    sc = np.array([dt.sum(), dq.sum()])
    sc = np.concatenate([db.sum(axis=0), sc])
    return sc
medpar = pandas.read_csv(urlopen('https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/csv/COUNT/medpar.csv'))
mdvis = pandas.read_csv(urlopen('https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/csv/COUNT/mdvis.csv'))
'\n# R v2.15.1\nlibrary(MASS)\nlibrary(COUNT)\ndata(medpar)\nf <- los~factor(type)+hmo+white\nmod <- glm.nb(f, medpar)\nsummary(mod)\nCall:\nglm.nb(formula = f, data = medpar, init.theta = 2.243376203,\n    link = log)\n\nDeviance Residuals:\n    Min       1Q   Median       3Q      Max\n-2.4671  -0.9090  -0.2693   0.4320   3.8668\n\nCoefficients:\n              Estimate Std. Error z value Pr(>|z|)\n(Intercept)    2.31028    0.06745  34.253  < 2e-16 ***\nfactor(type)2  0.22125    0.05046   4.385 1.16e-05 ***\nfactor(type)3  0.70616    0.07600   9.292  < 2e-16 ***\nhmo           -0.06796    0.05321  -1.277    0.202\nwhite         -0.12907    0.06836  -1.888    0.059 .\n---\nSignif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1\n\n(Dispersion parameter for Negative Binomial(2.2434) family taken to be 1)\n\n    Null deviance: 1691.1  on 1494  degrees of freedom\nResidual deviance: 1568.1  on 1490  degrees of freedom\nAIC: 9607\n\nNumber of Fisher Scoring iterations: 1\n\n\n              Theta:  2.2434\n          Std. Err.:  0.0997\n\n 2 x log-likelihood:  -9594.9530\n'

def test_nb2():
    if False:
        return 10
    (y, X) = patsy.dmatrices('los ~ C(type) + hmo + white', medpar)
    y = np.array(y)[:, 0]
    nb2 = NBin(y, X, 'nb2').fit(maxiter=10000, maxfun=5000)
    assert_almost_equal(nb2.params, [2.31027893349935, 0.221248978197356, 0.706158824346228, -0.067955221930748, -0.129065442248951, 0.4457567], decimal=2)
'\n# R v2.15.1\n# COUNT v1.2.3\nlibrary(COUNT)\ndata(medpar)\nf <- los~factor(type)+hmo+white\nml.nb1(f, medpar)\n\n                 Estimate         SE          Z         LCL         UCL\n(Intercept)    2.34918407 0.06023641 38.9994023  2.23112070  2.46724744\nfactor(type)2  0.16175471 0.04585569  3.5274735  0.07187757  0.25163186\nfactor(type)3  0.41879257 0.06553258  6.3906006  0.29034871  0.54723643\nhmo           -0.04533566 0.05004714 -0.9058592 -0.14342805  0.05275673\nwhite         -0.12951295 0.06071130 -2.1332593 -0.24850710 -0.01051880\nalpha          4.57898241 0.22015968 20.7984603  4.14746943  5.01049539\n'
'\nMASS v7.3-20\nR v2.15.1\nlibrary(MASS)\ndata(medpar)\nf <- los~factor(type)+hmo+white\nmod <- glm(f, family=negative.binomial(1), data=medpar)\nsummary(mod)\nCall:\nglm(formula = f, family = negative.binomial(1), data = medpar)\n\nDeviance Residuals:\n    Min       1Q   Median       3Q      Max\n-1.7942  -0.6545  -0.1896   0.3044   2.6844\n\nCoefficients:\n              Estimate Std. Error t value Pr(>|t|)\n(Intercept)    2.30849    0.07071  32.649  < 2e-16 ***\nfactor(type)2  0.22121    0.05283   4.187 2.99e-05 ***\nfactor(type)3  0.70599    0.08092   8.724  < 2e-16 ***\nhmo           -0.06779    0.05521  -1.228   0.2197\nwhite         -0.12709    0.07169  -1.773   0.0765 .\n---\nSignif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1\n\n(Dispersion parameter for Negative Binomial(1) family taken to be 0.5409721)\n\n    Null deviance: 872.29  on 1494  degrees of freedom\nResidual deviance: 811.95  on 1490  degrees of freedom\nAIC: 9927.3\n\nNumber of Fisher Scoring iterations: 5\n'
test_nb2()