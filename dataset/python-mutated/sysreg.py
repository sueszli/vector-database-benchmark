from statsmodels.regression.linear_model import GLS
import numpy as np
from statsmodels.base.model import LikelihoodModelResults
from scipy import sparse
__all__ = ['SUR', 'Sem2SLS']

class SUR:
    """
    Seemingly Unrelated Regression

    Parameters
    ----------
    sys : list
        [endog1, exog1, endog2, exog2,...] It will be of length 2 x M,
        where M is the number of equations endog = exog.
    sigma : array_like
        M x M array where sigma[i,j] is the covariance between equation i and j
    dfk : None, 'dfk1', or 'dfk2'
        Default is None.  Correction for the degrees of freedom
        should be specified for small samples.  See the notes for more
        information.

    Attributes
    ----------
    cholsigmainv : ndarray
        The transpose of the Cholesky decomposition of `pinv_wexog`
    df_model : ndarray
        Model degrees of freedom of each equation. p_{m} - 1 where p is
        the number of regressors for each equation m and one is subtracted
        for the constant.
    df_resid : ndarray
        Residual degrees of freedom of each equation. Number of observations
        less the number of parameters.
    endog : ndarray
        The LHS variables for each equation in the system.
        It is a M x nobs array where M is the number of equations.
    exog : ndarray
        The RHS variable for each equation in the system.
        It is a nobs x sum(p_{m}) array.  Which is just each
        RHS array stacked next to each other in columns.
    history : dict
        Contains the history of fitting the model. Probably not of interest
        if the model is fit with `igls` = False.
    iterations : int
        The number of iterations until convergence if the model is fit
        iteratively.
    nobs : float
        The number of observations of the equations.
    normalized_cov_params : ndarray
        sum(p_{m}) x sum(p_{m}) array
        :math:`\\left[X^{T}\\left(\\Sigma^{-1}\\otimes\\boldsymbol{I}\\right)X\\right]^{-1}`
    pinv_wexog : ndarray
        The pseudo-inverse of the `wexog`
    sigma : ndarray
        M x M covariance matrix of the cross-equation disturbances. See notes.
    sp_exog : CSR sparse matrix
        Contains a block diagonal sparse matrix of the design so that
        exog1 ... exogM are on the diagonal.
    wendog : ndarray
        M * nobs x 1 array of the endogenous variables whitened by
        `cholsigmainv` and stacked into a single column.
    wexog : ndarray
        M*nobs x sum(p_{m}) array of the whitened exogenous variables.

    Notes
    -----
    All individual equations are assumed to be well-behaved, homoskedastic
    iid errors.  This is basically an extension of GLS, using sparse matrices.

    .. math:: \\Sigma=\\left[\\begin{array}{cccc}
              \\sigma_{11} & \\sigma_{12} & \\cdots & \\sigma_{1M}\\\\
              \\sigma_{21} & \\sigma_{22} & \\cdots & \\sigma_{2M}\\\\
              \\vdots & \\vdots & \\ddots & \\vdots\\\\
              \\sigma_{M1} & \\sigma_{M2} & \\cdots & \\sigma_{MM}\\end{array}\\right]

    References
    ----------
    Zellner (1962), Greene (2003)
    """

    def __init__(self, sys, sigma=None, dfk=None):
        if False:
            print('Hello World!')
        if len(sys) % 2 != 0:
            raise ValueError('sys must be a list of pairs of endogenous and exogenous variables.  Got length %s' % len(sys))
        if dfk:
            if not dfk.lower() in ['dfk1', 'dfk2']:
                raise ValueError('dfk option %s not understood' % dfk)
        self._dfk = dfk
        M = len(sys[1::2])
        self._M = M
        exog = np.column_stack((np.asarray(sys[1::2][i]) for i in range(M)))
        self.exog = exog
        endog = np.asarray(sys[::2])
        self.endog = endog
        self.nobs = float(self.endog[0].shape[0])
        df_resid = []
        df_model = []
        [df_resid.append(self.nobs - np.linalg.matrix_rank(_)) for _ in sys[1::2]]
        [df_model.append(np.linalg.matrix_rank(_) - 1) for _ in sys[1::2]]
        self.df_resid = np.asarray(df_resid)
        self.df_model = np.asarray(df_model)
        sp_exog = sparse.lil_matrix((int(self.nobs * M), int(np.sum(self.df_model + 1))))
        self._cols = np.cumsum(np.hstack((0, self.df_model + 1)))
        for i in range(M):
            sp_exog[i * self.nobs:(i + 1) * self.nobs, self._cols[i]:self._cols[i + 1]] = sys[1::2][i]
        self.sp_exog = sp_exog.tocsr()
        if np.any(sigma):
            sigma = np.asarray(sigma)
        elif sigma is None:
            resids = []
            for i in range(M):
                resids.append(GLS(endog[i], exog[:, self._cols[i]:self._cols[i + 1]]).fit().resid)
            resids = np.asarray(resids).reshape(M, -1)
            sigma = self._compute_sigma(resids)
        self.sigma = sigma
        self.cholsigmainv = np.linalg.cholesky(np.linalg.pinv(self.sigma)).T
        self.initialize()

    def initialize(self):
        if False:
            i = 10
            return i + 15
        self.wendog = self.whiten(self.endog)
        self.wexog = self.whiten(self.sp_exog)
        self.pinv_wexog = np.linalg.pinv(self.wexog)
        self.normalized_cov_params = np.dot(self.pinv_wexog, np.transpose(self.pinv_wexog))
        self.history = {'params': [np.inf]}
        self.iterations = 0

    def _update_history(self, params):
        if False:
            return 10
        self.history['params'].append(params)

    def _compute_sigma(self, resids):
        if False:
            return 10
        '\n        Computes the sigma matrix and update the cholesky decomposition.\n        '
        M = self._M
        nobs = self.nobs
        sig = np.dot(resids, resids.T)
        if not self._dfk:
            div = nobs
        elif self._dfk.lower() == 'dfk1':
            div = np.zeros(M ** 2)
            for i in range(M):
                for j in range(M):
                    div[i + j] = ((self.df_model[i] + 1) * (self.df_model[j] + 1)) ** (1 / 2)
            div.reshape(M, M)
        else:
            div = np.zeros(M ** 2)
            for i in range(M):
                for j in range(M):
                    div[i + j] = nobs - np.max(self.df_model[i] + 1, self.df_model[j] + 1)
            div.reshape(M, M)
        self.cholsigmainv = np.linalg.cholesky(np.linalg.pinv(sig / div)).T
        return sig / div

    def whiten(self, X):
        if False:
            i = 10
            return i + 15
        '\n        SUR whiten method.\n\n        Parameters\n        ----------\n        X : list of arrays\n            Data to be whitened.\n\n        Returns\n        -------\n        If X is the exogenous RHS of the system.\n        ``np.dot(np.kron(cholsigmainv,np.eye(M)),np.diag(X))``\n\n        If X is the endogenous LHS of the system.\n        '
        nobs = self.nobs
        if X is self.endog:
            return np.dot(np.kron(self.cholsigmainv, np.eye(nobs)), X.reshape(-1, 1))
        elif X is self.sp_exog:
            return (sparse.kron(self.cholsigmainv, sparse.eye(nobs, nobs)) * X).toarray()

    def fit(self, igls=False, tol=1e-05, maxiter=100):
        if False:
            i = 10
            return i + 15
        '\n        igls : bool\n            Iterate until estimates converge if sigma is None instead of\n            two-step GLS, which is the default is sigma is None.\n\n        tol : float\n\n        maxiter : int\n\n        Notes\n        -----\n        This ia naive implementation that does not exploit the block\n        diagonal structure. It should work for ill-conditioned `sigma`\n        but this is untested.\n        '
        if not np.any(self.sigma):
            self.sigma = self._compute_sigma(self.endog, self.exog)
        M = self._M
        beta = np.dot(self.pinv_wexog, self.wendog)
        self._update_history(beta)
        self.iterations += 1
        if not igls:
            sur_fit = SysResults(self, beta, self.normalized_cov_params)
            return sur_fit
        conv = self.history['params']
        while igls and np.any(np.abs(conv[-2] - conv[-1]) > tol) and (self.iterations < maxiter):
            fittedvalues = (self.sp_exog * beta).reshape(M, -1)
            resids = self.endog - fittedvalues
            self.sigma = self._compute_sigma(resids)
            self.wendog = self.whiten(self.endog)
            self.wexog = self.whiten(self.sp_exog)
            self.pinv_wexog = np.linalg.pinv(self.wexog)
            self.normalized_cov_params = np.dot(self.pinv_wexog, np.transpose(self.pinv_wexog))
            beta = np.dot(self.pinv_wexog, self.wendog)
            self._update_history(beta)
            self.iterations += 1
        sur_fit = SysResults(self, beta, self.normalized_cov_params)
        return sur_fit

    def predict(self, design):
        if False:
            while True:
                i = 10
        pass

class Sem2SLS:
    """
    Two-Stage Least Squares for Simultaneous equations

    Parameters
    ----------
    sys : list
        [endog1, exog1, endog2, exog2,...] It will be of length 2 x M,
        where M is the number of equations endog = exog.
    indep_endog : dict
        A dictionary mapping the equation to the column numbers of the
        the independent endogenous regressors in each equation.
        It is assumed that the system is entered as broken up into
        LHS and RHS. For now, the values of the dict have to be sequences.
        Note that the keys for the equations should be zero-indexed.
    instruments : ndarray
        Array of the exogenous independent variables.

    Notes
    -----
    This is unfinished, and the design should be refactored.
    Estimation is done by brute force and there is no exploitation of
    the structure of the system.
    """

    def __init__(self, sys, indep_endog=None, instruments=None):
        if False:
            return 10
        if len(sys) % 2 != 0:
            raise ValueError('sys must be a list of pairs of endogenous and exogenous variables.  Got length %s' % len(sys))
        M = len(sys[1::2])
        self._M = M
        self.endog = sys[::2]
        self.exog = sys[1::2]
        self._K = [np.linalg.matrix_rank(_) for _ in sys[1::2]]
        self.instruments = instruments
        instr_endog = {}
        [instr_endog.setdefault(_, []) for _ in indep_endog.keys()]
        for eq_key in indep_endog:
            for varcol in indep_endog[eq_key]:
                instr_endog[eq_key].append(self.exog[eq_key][:, varcol])
        self._indep_endog = indep_endog
        _col_map = np.cumsum(np.hstack((0, self._K)))
        for eq_key in indep_endog:
            try:
                iter(indep_endog[eq_key])
            except:
                raise TypeError('The values of the indep_exog dict must be iterable. Got type %s for converter %s' % (type(indep_endog[eq_key]), eq_key))
        self.wexog = self.whiten(instr_endog)

    def whiten(self, Y):
        if False:
            print('Hello World!')
        '\n        Runs the first stage of the 2SLS.\n\n        Returns the RHS variables that include the instruments.\n        '
        wexog = []
        indep_endog = self._indep_endog
        instruments = self.instruments
        for eq in range(self._M):
            instr_eq = Y.get(eq, None)
            newRHS = self.exog[eq].copy()
            if instr_eq:
                for (i, LHS) in enumerate(instr_eq):
                    yhat = GLS(LHS, self.instruments).fit().fittedvalues
                    newRHS[:, indep_endog[eq][i]] = yhat
            wexog.append(newRHS)
        return wexog

    def fit(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        '
        delta = []
        wexog = self.wexog
        endog = self.endog
        for j in range(self._M):
            delta.append(GLS(endog[j], wexog[j]).fit().params)
        return delta

class SysResults(LikelihoodModelResults):
    """
    Not implemented yet.
    """

    def __init__(self, model, params, normalized_cov_params=None, scale=1.0):
        if False:
            return 10
        super(SysResults, self).__init__(model, params, normalized_cov_params, scale)
        self._get_results()

    def _get_results(self):
        if False:
            return 10
        pass