"""General linear model

author: Yichuan Liu
"""
import numpy as np
from numpy.linalg import eigvals, inv, solve, matrix_rank, pinv, svd
from scipy import stats
import pandas as pd
from patsy import DesignInfo
from statsmodels.compat.pandas import Substitution
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
__docformat__ = 'restructuredtext en'
_hypotheses_doc = 'hypotheses : list[tuple]\n    Hypothesis `L*B*M = C` to be tested where B is the parameters in\n    regression Y = X*B. Each element is a tuple of length 2, 3, or 4:\n\n      * (name, contrast_L)\n      * (name, contrast_L, transform_M)\n      * (name, contrast_L, transform_M, constant_C)\n\n    containing a string `name`, the contrast matrix L, the transform\n    matrix M (for transforming dependent variables), and right-hand side\n    constant matrix constant_C, respectively.\n\n    contrast_L : 2D array or an array of strings\n        Left-hand side contrast matrix for hypotheses testing.\n        If 2D array, each row is an hypotheses and each column is an\n        independent variable. At least 1 row\n        (1 by k_exog, the number of independent variables) is required.\n        If an array of strings, it will be passed to\n        patsy.DesignInfo().linear_constraint.\n\n    transform_M : 2D array or an array of strings or None, optional\n        Left hand side transform matrix.\n        If `None` or left out, it is set to a k_endog by k_endog\n        identity matrix (i.e. do not transform y matrix).\n        If an array of strings, it will be passed to\n        patsy.DesignInfo().linear_constraint.\n\n    constant_C : 2D array or None, optional\n        Right-hand side constant matrix.\n        if `None` or left out it is set to a matrix of zeros\n        Must has the same number of rows as contrast_L and the same\n        number of columns as transform_M\n\n    If `hypotheses` is None: 1) the effect of each independent variable\n    on the dependent variables will be tested. Or 2) if model is created\n    using a formula,  `hypotheses` will be created according to\n    `design_info`. 1) and 2) is equivalent if no additional variables\n    are created by the formula (e.g. dummy variables for categorical\n    variables and interaction terms)\n'

def _multivariate_ols_fit(endog, exog, method='svd', tolerance=1e-08):
    if False:
        return 10
    "\n    Solve multivariate linear model y = x * params\n    where y is dependent variables, x is independent variables\n\n    Parameters\n    ----------\n    endog : array_like\n        each column is a dependent variable\n    exog : array_like\n        each column is a independent variable\n    method : str\n        'svd' - Singular value decomposition\n        'pinv' - Moore-Penrose pseudoinverse\n    tolerance : float, a small positive number\n        Tolerance for eigenvalue. Values smaller than tolerance is considered\n        zero.\n    Returns\n    -------\n    a tuple of matrices or values necessary for hypotheses testing\n\n    .. [*] https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introreg_sect012.htm\n    Notes\n    -----\n    Status: experimental and incomplete\n    "
    y = endog
    x = exog
    (nobs, k_endog) = y.shape
    (nobs1, k_exog) = x.shape
    if nobs != nobs1:
        raise ValueError('x(n=%d) and y(n=%d) should have the same number of rows!' % (nobs1, nobs))
    df_resid = nobs - k_exog
    if method == 'pinv':
        pinv_x = pinv(x)
        params = pinv_x.dot(y)
        inv_cov = pinv_x.dot(pinv_x.T)
        if matrix_rank(inv_cov, tol=tolerance) < k_exog:
            raise ValueError('Covariance of x singular!')
        t = x.dot(params)
        sscpr = np.subtract(y.T.dot(y), t.T.dot(t))
        return (params, df_resid, inv_cov, sscpr)
    elif method == 'svd':
        (u, s, v) = svd(x, 0)
        if (s > tolerance).sum() < len(s):
            raise ValueError('Covariance of x singular!')
        invs = 1.0 / s
        params = v.T.dot(np.diag(invs)).dot(u.T).dot(y)
        inv_cov = v.T.dot(np.diag(np.power(invs, 2))).dot(v)
        t = np.diag(s).dot(v).dot(params)
        sscpr = np.subtract(y.T.dot(y), t.T.dot(t))
        return (params, df_resid, inv_cov, sscpr)
    else:
        raise ValueError('%s is not a supported method!' % method)

def multivariate_stats(eigenvals, r_err_sscp, r_contrast, df_resid, tolerance=1e-08):
    if False:
        i = 10
        return i + 15
    "\n    For multivariate linear model Y = X * B\n    Testing hypotheses\n        L*B*M = 0\n    where L is contrast matrix, B is the parameters of the\n    multivariate linear model and M is dependent variable transform matrix.\n        T = L*inv(X'X)*L'\n        H = M'B'L'*inv(T)*LBM\n        E =  M'(Y'Y - B'X'XB)M\n\n    Parameters\n    ----------\n    eigenvals : ndarray\n        The eigenvalues of inv(E + H)*H\n    r_err_sscp : int\n        Rank of E + H\n    r_contrast : int\n        Rank of T matrix\n    df_resid : int\n        Residual degree of freedom (n_samples minus n_variables of X)\n    tolerance : float\n        smaller than which eigenvalue is considered 0\n\n    Returns\n    -------\n    A DataFrame\n\n    References\n    ----------\n    .. [*] https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introreg_sect012.htm\n    "
    v = df_resid
    p = r_err_sscp
    q = r_contrast
    s = np.min([p, q])
    ind = eigenvals > tolerance
    n_e = ind.sum()
    eigv2 = eigenvals[ind]
    eigv1 = np.array([i / (1 - i) for i in eigv2])
    m = (np.abs(p - q) - 1) / 2
    n = (v - p - 1) / 2
    cols = ['Value', 'Num DF', 'Den DF', 'F Value', 'Pr > F']
    index = ["Wilks' lambda", "Pillai's trace", 'Hotelling-Lawley trace', "Roy's greatest root"]
    results = pd.DataFrame(columns=cols, index=index)

    def fn(x):
        if False:
            i = 10
            return i + 15
        return np.real([x])[0]
    results.loc["Wilks' lambda", 'Value'] = fn(np.prod(1 - eigv2))
    results.loc["Pillai's trace", 'Value'] = fn(eigv2.sum())
    results.loc['Hotelling-Lawley trace', 'Value'] = fn(eigv1.sum())
    results.loc["Roy's greatest root", 'Value'] = fn(eigv1.max())
    r = v - (p - q + 1) / 2
    u = (p * q - 2) / 4
    df1 = p * q
    if p * p + q * q - 5 > 0:
        t = np.sqrt((p * p * q * q - 4) / (p * p + q * q - 5))
    else:
        t = 1
    df2 = r * t - 2 * u
    lmd = results.loc["Wilks' lambda", 'Value']
    lmd = np.power(lmd, 1 / t)
    F = (1 - lmd) / lmd * df2 / df1
    results.loc["Wilks' lambda", 'Num DF'] = df1
    results.loc["Wilks' lambda", 'Den DF'] = df2
    results.loc["Wilks' lambda", 'F Value'] = F
    pval = stats.f.sf(F, df1, df2)
    results.loc["Wilks' lambda", 'Pr > F'] = pval
    V = results.loc["Pillai's trace", 'Value']
    df1 = s * (2 * m + s + 1)
    df2 = s * (2 * n + s + 1)
    F = df2 / df1 * V / (s - V)
    results.loc["Pillai's trace", 'Num DF'] = df1
    results.loc["Pillai's trace", 'Den DF'] = df2
    results.loc["Pillai's trace", 'F Value'] = F
    pval = stats.f.sf(F, df1, df2)
    results.loc["Pillai's trace", 'Pr > F'] = pval
    U = results.loc['Hotelling-Lawley trace', 'Value']
    if n > 0:
        b = (p + 2 * n) * (q + 2 * n) / 2 / (2 * n + 1) / (n - 1)
        df1 = p * q
        df2 = 4 + (p * q + 2) / (b - 1)
        c = (df2 - 2) / 2 / n
        F = df2 / df1 * U / c
    else:
        df1 = s * (2 * m + s + 1)
        df2 = s * (s * n + 1)
        F = df2 / df1 / s * U
    results.loc['Hotelling-Lawley trace', 'Num DF'] = df1
    results.loc['Hotelling-Lawley trace', 'Den DF'] = df2
    results.loc['Hotelling-Lawley trace', 'F Value'] = F
    pval = stats.f.sf(F, df1, df2)
    results.loc['Hotelling-Lawley trace', 'Pr > F'] = pval
    sigma = results.loc["Roy's greatest root", 'Value']
    r = np.max([p, q])
    df1 = r
    df2 = v - r + q
    F = df2 / df1 * sigma
    results.loc["Roy's greatest root", 'Num DF'] = df1
    results.loc["Roy's greatest root", 'Den DF'] = df2
    results.loc["Roy's greatest root", 'F Value'] = F
    pval = stats.f.sf(F, df1, df2)
    results.loc["Roy's greatest root", 'Pr > F'] = pval
    return results

def _multivariate_ols_test(hypotheses, fit_results, exog_names, endog_names):
    if False:
        for i in range(10):
            print('nop')

    def fn(L, M, C):
        if False:
            i = 10
            return i + 15
        (params, df_resid, inv_cov, sscpr) = fit_results
        t1 = L.dot(params).dot(M) - C
        t2 = L.dot(inv_cov).dot(L.T)
        q = matrix_rank(t2)
        H = t1.T.dot(inv(t2)).dot(t1)
        E = M.T.dot(sscpr).dot(M)
        return (E, H, q, df_resid)
    return _multivariate_test(hypotheses, exog_names, endog_names, fn)

@Substitution(hypotheses_doc=_hypotheses_doc)
def _multivariate_test(hypotheses, exog_names, endog_names, fn):
    if False:
        i = 10
        return i + 15
    "\n    Multivariate linear model hypotheses testing\n\n    For y = x * params, where y are the dependent variables and x are the\n    independent variables, testing L * params * M = 0 where L is the contrast\n    matrix for hypotheses testing and M is the transformation matrix for\n    transforming the dependent variables in y.\n\n    Algorithm:\n        T = L*inv(X'X)*L'\n        H = M'B'L'*inv(T)*LBM\n        E =  M'(Y'Y - B'X'XB)M\n    where H and E correspond to the numerator and denominator of a univariate\n    F-test. Then find the eigenvalues of inv(H + E)*H from which the\n    multivariate test statistics are calculated.\n\n    .. [*] https://support.sas.com/documentation/cdl/en/statug/63033/HTML\n           /default/viewer.htm#statug_introreg_sect012.htm\n\n    Parameters\n    ----------\n    %(hypotheses_doc)s\n    k_xvar : int\n        The number of independent variables\n    k_yvar : int\n        The number of dependent variables\n    fn : function\n        a function fn(contrast_L, transform_M) that returns E, H, q, df_resid\n        where q is the rank of T matrix\n\n    Returns\n    -------\n    results : MANOVAResults\n    "
    k_xvar = len(exog_names)
    k_yvar = len(endog_names)
    results = {}
    for hypo in hypotheses:
        if len(hypo) == 2:
            (name, L) = hypo
            M = None
            C = None
        elif len(hypo) == 3:
            (name, L, M) = hypo
            C = None
        elif len(hypo) == 4:
            (name, L, M, C) = hypo
        else:
            raise ValueError('hypotheses must be a tuple of length 2, 3 or 4. len(hypotheses)=%d' % len(hypo))
        if any((isinstance(j, str) for j in L)):
            L = DesignInfo(exog_names).linear_constraint(L).coefs
        else:
            if not isinstance(L, np.ndarray) or len(L.shape) != 2:
                raise ValueError('Contrast matrix L must be a 2-d array!')
            if L.shape[1] != k_xvar:
                raise ValueError('Contrast matrix L should have the same number of columns as exog! %d != %d' % (L.shape[1], k_xvar))
        if M is None:
            M = np.eye(k_yvar)
        elif any((isinstance(j, str) for j in M)):
            M = DesignInfo(endog_names).linear_constraint(M).coefs.T
        elif M is not None:
            if not isinstance(M, np.ndarray) or len(M.shape) != 2:
                raise ValueError('Transform matrix M must be a 2-d array!')
            if M.shape[0] != k_yvar:
                raise ValueError('Transform matrix M should have the same number of rows as the number of columns of endog! %d != %d' % (M.shape[0], k_yvar))
        if C is None:
            C = np.zeros([L.shape[0], M.shape[1]])
        elif not isinstance(C, np.ndarray):
            raise ValueError('Constant matrix C must be a 2-d array!')
        if C.shape[0] != L.shape[0]:
            raise ValueError('contrast L and constant C must have the same number of rows! %d!=%d' % (L.shape[0], C.shape[0]))
        if C.shape[1] != M.shape[1]:
            raise ValueError('transform M and constant C must have the same number of columns! %d!=%d' % (M.shape[1], C.shape[1]))
        (E, H, q, df_resid) = fn(L, M, C)
        EH = np.add(E, H)
        p = matrix_rank(EH)
        eigv2 = np.sort(eigvals(solve(EH, H)))
        stat_table = multivariate_stats(eigv2, p, q, df_resid)
        results[name] = {'stat': stat_table, 'contrast_L': L, 'transform_M': M, 'constant_C': C, 'E': E, 'H': H}
    return results

class _MultivariateOLS(Model):
    """
    Multivariate linear model via least squares


    Parameters
    ----------
    endog : array_like
        Dependent variables. A nobs x k_endog array where nobs is
        the number of observations and k_endog is the number of dependent
        variables
    exog : array_like
        Independent variables. A nobs x k_exog array where nobs is the
        number of observations and k_exog is the number of independent
        variables. An intercept is not included by default and should be added
        by the user (models specified using a formula include an intercept by
        default)

    Attributes
    ----------
    endog : ndarray
        See Parameters.
    exog : ndarray
        See Parameters.
    """
    _formula_max_endog = None

    def __init__(self, endog, exog, missing='none', hasconst=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if len(endog.shape) == 1 or endog.shape[1] == 1:
            raise ValueError('There must be more than one dependent variable to fit multivariate OLS!')
        super(_MultivariateOLS, self).__init__(endog, exog, missing=missing, hasconst=hasconst, **kwargs)

    def fit(self, method='svd'):
        if False:
            for i in range(10):
                print('nop')
        self._fittedmod = _multivariate_ols_fit(self.endog, self.exog, method=method)
        return _MultivariateOLSResults(self)

class _MultivariateOLSResults:
    """
    _MultivariateOLS results class
    """

    def __init__(self, fitted_mv_ols):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(fitted_mv_ols, 'data') and hasattr(fitted_mv_ols.data, 'design_info'):
            self.design_info = fitted_mv_ols.data.design_info
        else:
            self.design_info = None
        self.exog_names = fitted_mv_ols.exog_names
        self.endog_names = fitted_mv_ols.endog_names
        self._fittedmod = fitted_mv_ols._fittedmod

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.summary().__str__()

    @Substitution(hypotheses_doc=_hypotheses_doc)
    def mv_test(self, hypotheses=None, skip_intercept_test=False):
        if False:
            i = 10
            return i + 15
        '\n        Linear hypotheses testing\n\n        Parameters\n        ----------\n        %(hypotheses_doc)s\n        skip_intercept_test : bool\n            If true, then testing the intercept is skipped, the model is not\n            changed.\n            Note: If a term has a numerically insignificant effect, then\n            an exception because of emtpy arrays may be raised. This can\n            happen for the intercept if the data has been demeaned.\n\n        Returns\n        -------\n        results: _MultivariateOLSResults\n\n        Notes\n        -----\n        Tests hypotheses of the form\n\n            L * params * M = C\n\n        where `params` is the regression coefficient matrix for the\n        linear model y = x * params, `L` is the contrast matrix, `M` is the\n        dependent variable transform matrix and C is the constant matrix.\n        '
        k_xvar = len(self.exog_names)
        if hypotheses is None:
            if self.design_info is not None:
                terms = self.design_info.term_name_slices
                hypotheses = []
                for key in terms:
                    if skip_intercept_test and key == 'Intercept':
                        continue
                    L_contrast = np.eye(k_xvar)[terms[key], :]
                    hypotheses.append([key, L_contrast, None])
            else:
                hypotheses = []
                for i in range(k_xvar):
                    name = 'x%d' % i
                    L = np.zeros([1, k_xvar])
                    L[i] = 1
                    hypotheses.append([name, L, None])
        results = _multivariate_ols_test(hypotheses, self._fittedmod, self.exog_names, self.endog_names)
        return MultivariateTestResults(results, self.endog_names, self.exog_names)

    def summary(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

class MultivariateTestResults:
    """
    Multivariate test results class

    Returned by `mv_test` method of `_MultivariateOLSResults` class

    Parameters
    ----------
    results : dict[str, dict]
        Dictionary containing test results. See the description
        below for the expected format.
    endog_names : sequence[str]
        A list or other sequence of endogenous variables names
    exog_names : sequence[str]
        A list of other sequence of exogenous variables names

    Attributes
    ----------
    results : dict
        Each hypothesis is contained in a single`key`. Each test must
        have the following keys:

        * 'stat' - contains the multivariate test results
        * 'contrast_L' - contains the contrast_L matrix
        * 'transform_M' - contains the transform_M matrix
        * 'constant_C' - contains the constant_C matrix
        * 'H' - contains an intermediate Hypothesis matrix,
          or the between groups sums of squares and cross-products matrix,
          corresponding to the numerator of the univariate F test.
        * 'E' - contains an intermediate Error matrix,
          corresponding to the denominator of the univariate F test.
          The Hypotheses and Error matrices can be used to calculate
          the same test statistics in 'stat', as well as to calculate
          the discriminant function (canonical correlates) from the
          eigenvectors of inv(E)H.

    endog_names : list[str]
        The endogenous names
    exog_names : list[str]
        The exogenous names
    summary_frame : DataFrame
        Returns results as a MultiIndex DataFrame
    """

    def __init__(self, results, endog_names, exog_names):
        if False:
            i = 10
            return i + 15
        self.results = results
        self.endog_names = list(endog_names)
        self.exog_names = list(exog_names)

    def __str__(self):
        if False:
            print('Hello World!')
        return self.summary().__str__()

    def __getitem__(self, item):
        if False:
            while True:
                i = 10
        return self.results[item]

    @property
    def summary_frame(self):
        if False:
            while True:
                i = 10
        '\n        Return results as a multiindex dataframe\n        '
        df = []
        for key in self.results:
            tmp = self.results[key]['stat'].copy()
            tmp.loc[:, 'Effect'] = key
            df.append(tmp.reset_index())
        df = pd.concat(df, axis=0)
        df = df.set_index(['Effect', 'index'])
        df.index.set_names(['Effect', 'Statistic'], inplace=True)
        return df

    def summary(self, show_contrast_L=False, show_transform_M=False, show_constant_C=False):
        if False:
            print('Hello World!')
        '\n        Summary of test results\n\n        Parameters\n        ----------\n        show_contrast_L : bool\n            Whether to show contrast_L matrix\n        show_transform_M : bool\n            Whether to show transform_M matrix\n        show_constant_C : bool\n            Whether to show the constant_C\n        '
        summ = summary2.Summary()
        summ.add_title('Multivariate linear model')
        for key in self.results:
            summ.add_dict({'': ''})
            df = self.results[key]['stat'].copy()
            df = df.reset_index()
            c = list(df.columns)
            c[0] = key
            df.columns = c
            df.index = ['', '', '', '']
            summ.add_df(df)
            if show_contrast_L:
                summ.add_dict({key: ' contrast L='})
                df = pd.DataFrame(self.results[key]['contrast_L'], columns=self.exog_names)
                summ.add_df(df)
            if show_transform_M:
                summ.add_dict({key: ' transform M='})
                df = pd.DataFrame(self.results[key]['transform_M'], index=self.endog_names)
                summ.add_df(df)
            if show_constant_C:
                summ.add_dict({key: ' constant C='})
                df = pd.DataFrame(self.results[key]['constant_C'])
                summ.add_df(df)
        return summ