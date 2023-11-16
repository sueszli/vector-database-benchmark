"""conditional logit and nested conditional logit

nested conditional logit is supposed to be the random utility version
(RU2 and maybe RU1)

References:
-----------
currently based on:
Greene, Econometric Analysis, 5th edition and draft (?)
Hess, Florian, 2002, Structural Choice analysis with nested logit models,
    The Stats Journal 2(3) pp 227-252

not yet used:
Silberhorn Nadja, Yasemin Boztug, Lutz Hildebrandt, 2008, Estimation with the
    nested logit model: specifications and software particularities,
    OR Spectrum
Koppelman, Frank S., and Chandra Bhat with technical support from Vaneet Sethi,
    Sriram Subramanian, Vincent Bernardin and Jian Zhang, 2006,
    A Self Instructing Course in Mode Choice Modeling: Multinomial and
    Nested Logit Models

Author: josef-pktd
License: BSD (simplified)
"""
import numpy as np
import numpy.lib.recfunctions as recf
from scipy import optimize

class TryCLogit:
    """
    Conditional Logit, data handling test

    Parameters
    ----------

    endog : array (nobs,nchoices)
        dummy encoding of realized choices
    exog_bychoices : list of arrays
        explanatory variables, one array of exog for each choice. Variables
        with common coefficients have to be first in each array
    ncommon : int
        number of explanatory variables with common coefficients

    Notes
    -----

    Utility for choice j is given by

        $V_j = X_j * beta + Z * gamma_j$

    where X_j contains generic variables (terminology Hess) that have the same
    coefficient across choices, and Z are variables, like individual-specific
    variables that have different coefficients across variables.

    If there are choice specific constants, then they should be contained in Z.
    For identification, the constant of one choice should be dropped.


    """

    def __init__(self, endog, exog_bychoices, ncommon):
        if False:
            return 10
        self.endog = endog
        self.exog_bychoices = exog_bychoices
        self.ncommon = ncommon
        (self.nobs, self.nchoices) = endog.shape
        self.nchoices = len(exog_bychoices)
        betaind = [exog_bychoices[ii].shape[1] - ncommon for ii in range(4)]
        zi = np.r_[[ncommon], ncommon + np.array(betaind).cumsum()]
        beta_indices = [np.r_[np.array([0, 1]), z[zi[ii]:zi[ii + 1]]] for ii in range(len(zi) - 1)]
        self.beta_indices = beta_indices
        beta = np.arange(7)
        betaidx_bychoices = [beta[idx] for idx in beta_indices]

    def xbetas(self, params):
        if False:
            print('Hello World!')
        'these are the V_i\n        '
        res = np.empty((self.nobs, self.nchoices))
        for choiceind in range(self.nchoices):
            res[:, choiceind] = np.dot(self.exog_bychoices[choiceind], params[self.beta_indices[choiceind]])
        return res

    def loglike(self, params):
        if False:
            i = 10
            return i + 15
        xb = self.xbetas(params)
        expxb = np.exp(xb)
        sumexpxb = expxb.sum(1)
        probs = expxb / expxb.sum(1)[:, None]
        loglike = (self.endog * np.log(probs)).sum(1)
        return -loglike.sum()

    def fit(self, start_params=None):
        if False:
            print('Hello World!')
        if start_params is None:
            start_params = np.zeros(6)
        return optimize.fmin(self.loglike, start_params, maxfun=10000)

class TryNCLogit:
    """
    Nested Conditional Logit (RUNMNL), data handling test

    unfinished, does not do anything yet

    """

    def __init__(self, endog, exog_bychoices, ncommon):
        if False:
            i = 10
            return i + 15
        self.endog = endog
        self.exog_bychoices = exog_bychoices
        self.ncommon = ncommon
        (self.nobs, self.nchoices) = endog.shape
        self.nchoices = len(exog_bychoices)
        betaind = [exog_bychoices[ii].shape[1] - ncommon for ii in range(4)]
        zi = np.r_[[ncommon], ncommon + np.array(betaind).cumsum()]
        beta_indices = [np.r_[np.array([0, 1]), z[zi[ii]:zi[ii + 1]]] for ii in range(len(zi) - 1)]
        self.beta_indices = beta_indices
        beta = np.arange(7)
        betaidx_bychoices = [beta[idx] for idx in beta_indices]

    def xbetas(self, params):
        if False:
            i = 10
            return i + 15
        'these are the V_i\n        '
        res = np.empty((self.nobs, self.nchoices))
        for choiceind in range(self.nchoices):
            res[:, choiceind] = np.dot(self.exog_bychoices[choiceind], params[self.beta_indices[choiceind]])
        return res

    def loglike_leafbranch(self, params, tau):
        if False:
            i = 10
            return i + 15
        xb = self.xbetas(params)
        expxb = np.exp(xb / tau)
        sumexpxb = expxb.sum(1)
        logsumexpxb = np.log(sumexpxb)
        probs = expxb / sumexpxb[:, None]
        return (probs, logsumexpxp)

    def loglike_branch(self, params, tau):
        if False:
            return 10
        ivs = []
        for b in branches:
            (probs, iv) = self.loglike_leafbranch(params, tau)
            ivs.append(iv)
        ivs = np.column_stack(ivs)
        exptiv = np.exp(tau * ivs)
        sumexptiv = exptiv.sum(1)
        logsumexpxb = np.log(sumexpxb)
        probs = exptiv / sumexptiv[:, None]
testxb = 0

class RU2NMNL:
    """Nested Multinomial Logit with Random Utility 2 parameterization

    """

    def __init__(self, endog, exog, tree, paramsind):
        if False:
            return 10
        self.endog = endog
        self.datadict = exog
        self.tree = tree
        self.paramsind = paramsind
        self.branchsum = ''
        self.probs = {}

    def calc_prob(self, tree, keys=None):
        if False:
            print('Hello World!')
        'walking a tree bottom-up based on dictionary\n        '
        endog = self.endog
        datadict = self.datadict
        paramsind = self.paramsind
        branchsum = self.branchsum
        if isinstance(tree, tuple):
            (name, subtree) = tree
            print(name, datadict[name])
            print('subtree', subtree)
            keys = []
            if testxb:
                branchsum = datadict[name]
            else:
                branchsum = name
            for b in subtree:
                print(b)
                branchsum = branchsum + self.calc_prob(b, keys)
            print('branchsum', branchsum, keys)
            for k in keys:
                self.probs[k] = self.probs[k] + ['*' + name + '-prob']
        else:
            keys.append(tree)
            self.probs[tree] = [tree + '-prob' + '(%s)' % ', '.join(self.paramsind[tree])]
            if testxb:
                leavessum = sum((datadict[bi] for bi in tree))
                print('final branch with', tree, ''.join(tree), leavessum)
                return leavessum
            else:
                return ''.join(tree)
        print('working on branch', tree, branchsum)
        return branchsum
dta = np.genfromtxt('TableF23-2.txt', skip_header=1, names='Mode   Ttme   Invc    Invt      GC     Hinc    PSize'.split())
endog = dta['Mode'].reshape(-1, 4).copy()
(nobs, nchoices) = endog.shape
datafloat = dta.view(float).reshape(-1, 7)
exog = datafloat[:, 1:].reshape(-1, 6 * nchoices).copy()
print(endog.sum(0))
varnames = dta.dtype.names
print(varnames[1:])
modes = ['Air', 'Train', 'Bus', 'Car']
print(exog.mean(0).reshape(nchoices, -1))
exog_choice_names = ['GC', 'Ttme']
exog_choice = np.column_stack([dta[name] for name in exog_choice_names])
exog_choice = exog_choice.reshape(-1, len(exog_choice_names) * nchoices)
exog_choice = np.c_[endog, exog_choice]
exog_individual = dta['Hinc'][:, None]
choice_index = np.arange(dta.shape[0]) % nchoices
hinca = dta['Hinc'] * (choice_index == 0)
dta2 = recf.append_fields(dta, ['Hinca'], [hinca], usemask=False)
xi = []
for ii in range(4):
    xi.append(datafloat[choice_index == ii])
dta1 = recf.append_fields(dta, ['Const'], [np.ones(dta.shape[0])], usemask=False)
xivar = [['GC', 'Ttme', 'Const', 'Hinc'], ['GC', 'Ttme', 'Const'], ['GC', 'Ttme', 'Const'], ['GC', 'Ttme']]
xi = []
for ii in range(4):
    xi.append(dta1[xivar[ii]][choice_index == ii])
ncommon = 2
betaind = [len(xi[ii].dtype.names) - ncommon for ii in range(4)]
zi = np.r_[[ncommon], ncommon + np.array(betaind).cumsum()]
z = np.arange(7)
betaindices = [np.r_[np.array([0, 1]), z[zi[ii]:zi[ii + 1]]] for ii in range(len(zi) - 1)]
beta = np.arange(7)
betai = [beta[idx] for idx in betaindices]
xifloat = [xx.view(float).reshape(nobs, -1) for xx in xi]
clogit = TryCLogit(endog, xifloat, 2)
debug = 0
if debug:
    res = optimize.fmin(clogit.loglike, np.ones(6))
tab2324 = [-0.15501, -0.09612, 0.01329, 5.2074, 3.869, 3.1632]
if debug:
    res2 = optimize.fmin(clogit.loglike, tab2324)
res3 = optimize.fmin(clogit.loglike, np.zeros(6), maxfun=10000)
'\nOptimization terminated successfully.\n         Current function value: 199.128369\n         Iterations: 957\n         Function evaluations: 1456\narray([-0.0961246 , -0.0155019 ,  0.01328757,  5.20741244,  3.86905293,\n        3.16319074])\n'
res3corr = res3[[1, 0, 2, 3, 4, 5]]
res3corr[0] *= 10
print(res3corr - tab2324)
print(clogit.fit())
tree0 = ('top', [('Fly', ['Air']), ('Ground', ['Train', 'Car', 'Bus'])])
datadict = dict(zip(['Air', 'Train', 'Bus', 'Car'], [xifloat[i] for i in range(4)]))
datadict = dict(zip(['Air', 'Train', 'Bus', 'Car'], ['Airdata', 'Traindata', 'Busdata', 'Cardata']))
datadict.update({'top': [], 'Fly': [], 'Ground': []})
paramsind = {'top': [], 'Fly': [], 'Ground': [], 'Air': ['GC', 'Ttme', 'ConstA', 'Hinc'], 'Train': ['GC', 'Ttme', 'ConstT'], 'Bus': ['GC', 'Ttme', 'ConstB'], 'Car': ['GC', 'Ttme']}
modru = RU2NMNL(endog, datadict, tree0, paramsind)
print(modru.calc_prob(modru.tree))
print('\nmodru.probs')
print(modru.probs)