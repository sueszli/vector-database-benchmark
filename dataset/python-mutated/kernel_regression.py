"""
Multivariate Conditional and Unconditional Kernel Density Estimation
with Mixed Data Types

References
----------
[1] Racine, J., Li, Q. Nonparametric econometrics: theory and practice.
    Princeton University Press. (2007)
[2] Racine, Jeff. "Nonparametric Econometrics: A Primer," Foundation
    and Trends in Econometrics: Vol 3: No 1, pp1-88. (2008)
    http://dx.doi.org/10.1561/0800000009
[3] Racine, J., Li, Q. "Nonparametric Estimation of Distributions
    with Categorical and Continuous Data." Working Paper. (2000)
[4] Racine, J. Li, Q. "Kernel Estimation of Multivariate Conditional
    Distributions Annals of Economics and Finance 5, 211-235 (2004)
[5] Liu, R., Yang, L. "Kernel estimation of multivariate
    cumulative distribution function."
    Journal of Nonparametric Statistics (2008)
[6] Li, R., Ju, G. "Nonparametric Estimation of Multivariate CDF
    with Categorical and Continuous Data." Working Paper
[7] Li, Q., Racine, J. "Cross-validated local linear nonparametric
    regression" Statistica Sinica 14(2004), pp. 485-512
[8] Racine, J.: "Consistent Significance Testing for Nonparametric
        Regression" Journal of Business & Economics Statistics
[9] Racine, J., Hart, J., Li, Q., "Testing the Significance of
        Categorical Predictor Variables in Nonparametric Regression
        Models", 2006, Econometric Reviews 25, 523-544

"""
import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from ._kernel_base import GenericKDE, EstimatorSettings, gpke, LeaveOneOut, _get_type_pos, _adjust_shape, _compute_min_std_IQR, kernel_func
__all__ = ['KernelReg', 'KernelCensoredReg']

class KernelReg(GenericKDE):
    """
    Nonparametric kernel regression class.

    Calculates the conditional mean ``E[y|X]`` where ``y = g(X) + e``.
    Note that the "local constant" type of regression provided here is also
    known as Nadaraya-Watson kernel regression; "local linear" is an extension
    of that which suffers less from bias issues at the edge of the support. Note
    that specifying a custom kernel works only with "local linear" kernel
    regression. For example, a custom ``tricube`` kernel yields LOESS regression.

    Parameters
    ----------
    endog : array_like
        This is the dependent variable.
    exog : array_like
        The training data for the independent variable(s)
        Each element in the list is a separate variable
    var_type : str
        The type of the variables, one character per variable:

            - c: continuous
            - u: unordered (discrete)
            - o: ordered (discrete)

    reg_type : {'lc', 'll'}, optional
        Type of regression estimator. 'lc' means local constant and
        'll' local Linear estimator.  Default is 'll'
    bw : str or array_like, optional
        Either a user-specified bandwidth or the method for bandwidth
        selection. If a string, valid values are 'cv_ls' (least-squares
        cross-validation) and 'aic' (AIC Hurvich bandwidth estimation).
        Default is 'cv_ls'. User specified bandwidth must have as many
        entries as the number of variables.
    ckertype : str, optional
        The kernel used for the continuous variables.
    okertype : str, optional
        The kernel used for the ordered discrete variables.
    ukertype : str, optional
        The kernel used for the unordered discrete variables.
    defaults : EstimatorSettings instance, optional
        The default values for the efficient bandwidth estimation.

    Attributes
    ----------
    bw : array_like
        The bandwidth parameters.
    """

    def __init__(self, endog, exog, var_type, reg_type='ll', bw='cv_ls', ckertype='gaussian', okertype='wangryzin', ukertype='aitchisonaitken', defaults=None):
        if False:
            while True:
                i = 10
        self.var_type = var_type
        self.data_type = var_type
        self.reg_type = reg_type
        self.ckertype = ckertype
        self.okertype = okertype
        self.ukertype = ukertype
        if not (self.ckertype in kernel_func and self.ukertype in kernel_func and (self.okertype in kernel_func)):
            raise ValueError('user specified kernel must be a supported kernel from statsmodels.nonparametric.kernels.')
        self.k_vars = len(self.var_type)
        self.endog = _adjust_shape(endog, 1)
        self.exog = _adjust_shape(exog, self.k_vars)
        self.data = np.column_stack((self.endog, self.exog))
        self.nobs = np.shape(self.exog)[0]
        self.est = dict(lc=self._est_loc_constant, ll=self._est_loc_linear)
        defaults = EstimatorSettings() if defaults is None else defaults
        self._set_defaults(defaults)
        if not isinstance(bw, str):
            bw = np.asarray(bw)
            if len(bw) != self.k_vars:
                raise ValueError('bw must have the same dimension as the number of variables.')
        if not self.efficient:
            self.bw = self._compute_reg_bw(bw)
        else:
            self.bw = self._compute_efficient(bw)

    def _compute_reg_bw(self, bw):
        if False:
            i = 10
            return i + 15
        if not isinstance(bw, str):
            self._bw_method = 'user-specified'
            return np.asarray(bw)
        else:
            self._bw_method = bw
            if bw == 'cv_ls':
                res = self.cv_loo
            else:
                res = self.aic_hurvich
            X = np.std(self.exog, axis=0)
            h0 = 1.06 * X * self.nobs ** (-1.0 / (4 + np.size(self.exog, axis=1)))
            func = self.est[self.reg_type]
            bw_estimated = optimize.fmin(res, x0=h0, args=(func,), maxiter=1000.0, maxfun=1000.0, disp=0)
            return bw_estimated

    def _est_loc_linear(self, bw, endog, exog, data_predict):
        if False:
            for i in range(10):
                print('nop')
        '\n        Local linear estimator of g(x) in the regression ``y = g(x) + e``.\n\n        Parameters\n        ----------\n        bw : array_like\n            Vector of bandwidth value(s).\n        endog : 1D array_like\n            The dependent variable.\n        exog : 1D or 2D array_like\n            The independent variable(s).\n        data_predict : 1D array_like of length K, where K is the number of variables.\n            The point at which the density is estimated.\n\n        Returns\n        -------\n        D_x : array_like\n            The value of the conditional mean at `data_predict`.\n\n        Notes\n        -----\n        See p. 81 in [1] and p.38 in [2] for the formulas.\n        Unlike other methods, this one requires that `data_predict` be 1D.\n        '
        (nobs, k_vars) = exog.shape
        ker = gpke(bw, data=exog, data_predict=data_predict, var_type=self.var_type, ckertype=self.ckertype, ukertype=self.ukertype, okertype=self.okertype, tosum=False) / float(nobs)
        ker = ker[:, np.newaxis]
        M12 = exog - data_predict
        M22 = np.dot(M12.T, M12 * ker)
        M12 = (M12 * ker).sum(axis=0)
        M = np.empty((k_vars + 1, k_vars + 1))
        M[0, 0] = ker.sum()
        M[0, 1:] = M12
        M[1:, 0] = M12
        M[1:, 1:] = M22
        ker_endog = ker * endog
        V = np.empty((k_vars + 1, 1))
        V[0, 0] = ker_endog.sum()
        V[1:, 0] = ((exog - data_predict) * ker_endog).sum(axis=0)
        mean_mfx = np.dot(np.linalg.pinv(M), V)
        mean = mean_mfx[0]
        mfx = mean_mfx[1:, :]
        return (mean, mfx)

    def _est_loc_constant(self, bw, endog, exog, data_predict):
        if False:
            while True:
                i = 10
        '\n        Local constant estimator of g(x) in the regression\n        y = g(x) + e\n\n        Parameters\n        ----------\n        bw : array_like\n            Array of bandwidth value(s).\n        endog : 1D array_like\n            The dependent variable.\n        exog : 1D or 2D array_like\n            The independent variable(s).\n        data_predict : 1D or 2D array_like\n            The point(s) at which the density is estimated.\n\n        Returns\n        -------\n        G : ndarray\n            The value of the conditional mean at `data_predict`.\n        B_x : ndarray\n            The marginal effects.\n        '
        ker_x = gpke(bw, data=exog, data_predict=data_predict, var_type=self.var_type, ckertype=self.ckertype, ukertype=self.ukertype, okertype=self.okertype, tosum=False)
        ker_x = np.reshape(ker_x, np.shape(endog))
        G_numer = (ker_x * endog).sum(axis=0)
        G_denom = ker_x.sum(axis=0)
        G = G_numer / G_denom
        nobs = exog.shape[0]
        f_x = G_denom / float(nobs)
        ker_xc = gpke(bw, data=exog, data_predict=data_predict, var_type=self.var_type, ckertype='d_gaussian', tosum=False)
        ker_xc = ker_xc[:, np.newaxis]
        d_mx = -(endog * ker_xc).sum(axis=0) / float(nobs)
        d_fx = -ker_xc.sum(axis=0) / float(nobs)
        B_x = d_mx / f_x - G * d_fx / f_x
        B_x = (G_numer * d_fx - G_denom * d_mx) / G_denom ** 2
        return (G, B_x)

    def aic_hurvich(self, bw, func=None):
        if False:
            i = 10
            return i + 15
        "\n        Computes the AIC Hurvich criteria for the estimation of the bandwidth.\n\n        Parameters\n        ----------\n        bw : str or array_like\n            See the ``bw`` parameter of `KernelReg` for details.\n\n        Returns\n        -------\n        aic : ndarray\n            The AIC Hurvich criteria, one element for each variable.\n        func : None\n            Unused here, needed in signature because it's used in `cv_loo`.\n\n        References\n        ----------\n        See ch.2 in [1] and p.35 in [2].\n        "
        H = np.empty((self.nobs, self.nobs))
        for j in range(self.nobs):
            H[:, j] = gpke(bw, data=self.exog, data_predict=self.exog[j, :], ckertype=self.ckertype, ukertype=self.ukertype, okertype=self.okertype, var_type=self.var_type, tosum=False)
        denom = H.sum(axis=1)
        H = H / denom
        gx = KernelReg(endog=self.endog, exog=self.exog, var_type=self.var_type, reg_type=self.reg_type, bw=bw, defaults=EstimatorSettings(efficient=False)).fit()[0]
        gx = np.reshape(gx, (self.nobs, 1))
        sigma = ((self.endog - gx) ** 2).sum(axis=0) / float(self.nobs)
        frac = (1 + np.trace(H) / float(self.nobs)) / (1 - (np.trace(H) + 2) / float(self.nobs))
        aic = np.log(sigma) + frac
        return aic

    def cv_loo(self, bw, func):
        if False:
            for i in range(10):
                print('nop')
        '\n        The cross-validation function with leave-one-out estimator.\n\n        Parameters\n        ----------\n        bw : array_like\n            Vector of bandwidth values.\n        func : callable function\n            Returns the estimator of g(x).  Can be either ``_est_loc_constant``\n            (local constant) or ``_est_loc_linear`` (local_linear).\n\n        Returns\n        -------\n        L : float\n            The value of the CV function.\n\n        Notes\n        -----\n        Calculates the cross-validation least-squares function. This function\n        is minimized by compute_bw to calculate the optimal value of `bw`.\n\n        For details see p.35 in [2]\n\n        .. math:: CV(h)=n^{-1}\\sum_{i=1}^{n}(Y_{i}-g_{-i}(X_{i}))^{2}\n\n        where :math:`g_{-i}(X_{i})` is the leave-one-out estimator of g(X)\n        and :math:`h` is the vector of bandwidths\n        '
        LOO_X = LeaveOneOut(self.exog)
        LOO_Y = LeaveOneOut(self.endog).__iter__()
        L = 0
        for (ii, X_not_i) in enumerate(LOO_X):
            Y = next(LOO_Y)
            G = func(bw, endog=Y, exog=-X_not_i, data_predict=-self.exog[ii, :])[0]
            L += (self.endog[ii] - G) ** 2
        return L / self.nobs

    def r_squared(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the R-Squared for the nonparametric regression.\n\n        Notes\n        -----\n        For more details see p.45 in [2]\n        The R-Squared is calculated by:\n\n        .. math:: R^{2}=\\frac{\\left[\\sum_{i=1}^{n}\n            (Y_{i}-\\bar{y})(\\hat{Y_{i}}-\\bar{y}\\right]^{2}}{\\sum_{i=1}^{n}\n            (Y_{i}-\\bar{y})^{2}\\sum_{i=1}^{n}(\\hat{Y_{i}}-\\bar{y})^{2}},\n\n        where :math:`\\hat{Y_{i}}` is the mean calculated in `fit` at the exog\n        points.\n        '
        Y = np.squeeze(self.endog)
        Yhat = self.fit()[0]
        Y_bar = np.mean(Yhat)
        R2_numer = ((Y - Y_bar) * (Yhat - Y_bar)).sum() ** 2
        R2_denom = ((Y - Y_bar) ** 2).sum(axis=0) * ((Yhat - Y_bar) ** 2).sum(axis=0)
        return R2_numer / R2_denom

    def fit(self, data_predict=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the mean and marginal effects at the `data_predict` points.\n\n        Parameters\n        ----------\n        data_predict : array_like, optional\n            Points at which to return the mean and marginal effects.  If not\n            given, ``data_predict == exog``.\n\n        Returns\n        -------\n        mean : ndarray\n            The regression result for the mean (i.e. the actual curve).\n        mfx : ndarray\n            The marginal effects, i.e. the partial derivatives of the mean.\n        '
        func = self.est[self.reg_type]
        if data_predict is None:
            data_predict = self.exog
        else:
            data_predict = _adjust_shape(data_predict, self.k_vars)
        N_data_predict = np.shape(data_predict)[0]
        mean = np.empty((N_data_predict,))
        mfx = np.empty((N_data_predict, self.k_vars))
        for i in range(N_data_predict):
            mean_mfx = func(self.bw, self.endog, self.exog, data_predict=data_predict[i, :])
            mean[i] = np.squeeze(mean_mfx[0])
            mfx_c = np.squeeze(mean_mfx[1])
            mfx[i, :] = mfx_c
        return (mean, mfx)

    def sig_test(self, var_pos, nboot=50, nested_res=25, pivot=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Significance test for the variables in the regression.\n\n        Parameters\n        ----------\n        var_pos : sequence\n            The position of the variable in exog to be tested.\n\n        Returns\n        -------\n        sig : str\n            The level of significance:\n\n                - `*` : at 90% confidence level\n                - `**` : at 95% confidence level\n                - `***` : at 99* confidence level\n                - "Not Significant" : if not significant\n        '
        var_pos = np.asarray(var_pos)
        (ix_cont, ix_ord, ix_unord) = _get_type_pos(self.var_type)
        if np.any(ix_cont[var_pos]):
            if np.any(ix_ord[var_pos]) or np.any(ix_unord[var_pos]):
                raise ValueError('Discrete variable in hypothesis. Must be continuous')
            Sig = TestRegCoefC(self, var_pos, nboot, nested_res, pivot)
        else:
            Sig = TestRegCoefD(self, var_pos, nboot)
        return Sig.sig

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        'Provide something sane to print.'
        rpr = 'KernelReg instance\n'
        rpr += 'Number of variables: k_vars = ' + str(self.k_vars) + '\n'
        rpr += 'Number of samples:   N = ' + str(self.nobs) + '\n'
        rpr += 'Variable types:      ' + self.var_type + '\n'
        rpr += 'BW selection method: ' + self._bw_method + '\n'
        rpr += 'Estimator type: ' + self.reg_type + '\n'
        return rpr

    def _get_class_vars_type(self):
        if False:
            print('Hello World!')
        'Helper method to be able to pass needed vars to _compute_subset.'
        class_type = 'KernelReg'
        class_vars = (self.var_type, self.k_vars, self.reg_type)
        return (class_type, class_vars)

    def _compute_dispersion(self, data):
        if False:
            while True:
                i = 10
        '\n        Computes the measure of dispersion.\n\n        The minimum of the standard deviation and interquartile range / 1.349\n\n        References\n        ----------\n        See the user guide for the np package in R.\n        In the notes on bwscaling option in npreg, npudens, npcdens there is\n        a discussion on the measure of dispersion\n        '
        data = data[:, 1:]
        return _compute_min_std_IQR(data)

class KernelCensoredReg(KernelReg):
    """
    Nonparametric censored regression.

    Calculates the conditional mean ``E[y|X]`` where ``y = g(X) + e``,
    where y is left-censored.  Left censored variable Y is defined as
    ``Y = min {Y', L}`` where ``L`` is the value at which ``Y`` is censored
    and ``Y'`` is the true value of the variable.

    Parameters
    ----------
    endog : list with one element which is array_like
        This is the dependent variable.
    exog : list
        The training data for the independent variable(s)
        Each element in the list is a separate variable
    dep_type : str
        The type of the dependent variable(s)
        c: Continuous
        u: Unordered (Discrete)
        o: Ordered (Discrete)
    reg_type : str
        Type of regression estimator
        lc: Local Constant Estimator
        ll: Local Linear Estimator
    bw : array_like
        Either a user-specified bandwidth or
        the method for bandwidth selection.
        cv_ls: cross-validation least squares
        aic: AIC Hurvich Estimator
    ckertype : str, optional
        The kernel used for the continuous variables.
    okertype : str, optional
        The kernel used for the ordered discrete variables.
    ukertype : str, optional
        The kernel used for the unordered discrete variables.
    censor_val : float
        Value at which the dependent variable is censored
    defaults : EstimatorSettings instance, optional
        The default values for the efficient bandwidth estimation

    Attributes
    ----------
    bw : array_like
        The bandwidth parameters
    """

    def __init__(self, endog, exog, var_type, reg_type, bw='cv_ls', ckertype='gaussian', ukertype='aitchison_aitken_reg', okertype='wangryzin_reg', censor_val=0, defaults=None):
        if False:
            for i in range(10):
                print('nop')
        self.var_type = var_type
        self.data_type = var_type
        self.reg_type = reg_type
        self.ckertype = ckertype
        self.okertype = okertype
        self.ukertype = ukertype
        if not (self.ckertype in kernel_func and self.ukertype in kernel_func and (self.okertype in kernel_func)):
            raise ValueError('user specified kernel must be a supported kernel from statsmodels.nonparametric.kernels.')
        self.k_vars = len(self.var_type)
        self.endog = _adjust_shape(endog, 1)
        self.exog = _adjust_shape(exog, self.k_vars)
        self.data = np.column_stack((self.endog, self.exog))
        self.nobs = np.shape(self.exog)[0]
        self.est = dict(lc=self._est_loc_constant, ll=self._est_loc_linear)
        defaults = EstimatorSettings() if defaults is None else defaults
        self._set_defaults(defaults)
        self.censor_val = censor_val
        if self.censor_val is not None:
            self.censored(censor_val)
        else:
            self.W_in = np.ones((self.nobs, 1))
        if not self.efficient:
            self.bw = self._compute_reg_bw(bw)
        else:
            self.bw = self._compute_efficient(bw)

    def censored(self, censor_val):
        if False:
            print('Hello World!')
        self.d = (self.endog != censor_val) * 1.0
        ix = np.argsort(np.squeeze(self.endog))
        self.sortix = ix
        self.sortix_rev = np.zeros(ix.shape, int)
        self.sortix_rev[ix] = np.arange(len(ix))
        self.endog = np.squeeze(self.endog[ix])
        self.endog = _adjust_shape(self.endog, 1)
        self.exog = np.squeeze(self.exog[ix])
        self.d = np.squeeze(self.d[ix])
        self.W_in = np.empty((self.nobs, 1))
        for i in range(1, self.nobs + 1):
            P = 1
            for j in range(1, i):
                P *= ((self.nobs - j) / (float(self.nobs) - j + 1)) ** self.d[j - 1]
            self.W_in[i - 1, 0] = P * self.d[i - 1] / (float(self.nobs) - i + 1)

    def __repr__(self):
        if False:
            print('Hello World!')
        'Provide something sane to print.'
        rpr = 'KernelCensoredReg instance\n'
        rpr += 'Number of variables: k_vars = ' + str(self.k_vars) + '\n'
        rpr += 'Number of samples:   nobs = ' + str(self.nobs) + '\n'
        rpr += 'Variable types:      ' + self.var_type + '\n'
        rpr += 'BW selection method: ' + self._bw_method + '\n'
        rpr += 'Estimator type: ' + self.reg_type + '\n'
        return rpr

    def _est_loc_linear(self, bw, endog, exog, data_predict, W):
        if False:
            i = 10
            return i + 15
        '\n        Local linear estimator of g(x) in the regression ``y = g(x) + e``.\n\n        Parameters\n        ----------\n        bw : array_like\n            Vector of bandwidth value(s)\n        endog : 1D array_like\n            The dependent variable\n        exog : 1D or 2D array_like\n            The independent variable(s)\n        data_predict : 1D array_like of length K, where K is\n            the number of variables. The point at which\n            the density is estimated\n\n        Returns\n        -------\n        D_x : array_like\n            The value of the conditional mean at data_predict\n\n        Notes\n        -----\n        See p. 81 in [1] and p.38 in [2] for the formulas\n        Unlike other methods, this one requires that data_predict be 1D\n        '
        (nobs, k_vars) = exog.shape
        ker = gpke(bw, data=exog, data_predict=data_predict, var_type=self.var_type, ckertype=self.ckertype, ukertype=self.ukertype, okertype=self.okertype, tosum=False)
        ker = W * ker[:, np.newaxis]
        M12 = exog - data_predict
        M22 = np.dot(M12.T, M12 * ker)
        M12 = (M12 * ker).sum(axis=0)
        M = np.empty((k_vars + 1, k_vars + 1))
        M[0, 0] = ker.sum()
        M[0, 1:] = M12
        M[1:, 0] = M12
        M[1:, 1:] = M22
        ker_endog = ker * endog
        V = np.empty((k_vars + 1, 1))
        V[0, 0] = ker_endog.sum()
        V[1:, 0] = ((exog - data_predict) * ker_endog).sum(axis=0)
        mean_mfx = np.dot(np.linalg.pinv(M), V)
        mean = mean_mfx[0]
        mfx = mean_mfx[1:, :]
        return (mean, mfx)

    def cv_loo(self, bw, func):
        if False:
            for i in range(10):
                print('nop')
        '\n        The cross-validation function with leave-one-out\n        estimator\n\n        Parameters\n        ----------\n        bw : array_like\n            Vector of bandwidth values\n        func : callable function\n            Returns the estimator of g(x).\n            Can be either ``_est_loc_constant`` (local constant) or\n            ``_est_loc_linear`` (local_linear).\n\n        Returns\n        -------\n        L : float\n            The value of the CV function\n\n        Notes\n        -----\n        Calculates the cross-validation least-squares\n        function. This function is minimized by compute_bw\n        to calculate the optimal value of bw\n\n        For details see p.35 in [2]\n\n        .. math:: CV(h)=n^{-1}\\sum_{i=1}^{n}(Y_{i}-g_{-i}(X_{i}))^{2}\n\n        where :math:`g_{-i}(X_{i})` is the leave-one-out estimator of g(X)\n        and :math:`h` is the vector of bandwidths\n        '
        LOO_X = LeaveOneOut(self.exog)
        LOO_Y = LeaveOneOut(self.endog).__iter__()
        LOO_W = LeaveOneOut(self.W_in).__iter__()
        L = 0
        for (ii, X_not_i) in enumerate(LOO_X):
            Y = next(LOO_Y)
            w = next(LOO_W)
            G = func(bw, endog=Y, exog=-X_not_i, data_predict=-self.exog[ii, :], W=w)[0]
            L += (self.endog[ii] - G) ** 2
        return L / self.nobs

    def fit(self, data_predict=None):
        if False:
            while True:
                i = 10
        '\n        Returns the marginal effects at the data_predict points.\n        '
        func = self.est[self.reg_type]
        if data_predict is None:
            data_predict = self.exog
        else:
            data_predict = _adjust_shape(data_predict, self.k_vars)
        N_data_predict = np.shape(data_predict)[0]
        mean = np.empty((N_data_predict,))
        mfx = np.empty((N_data_predict, self.k_vars))
        for i in range(N_data_predict):
            mean_mfx = func(self.bw, self.endog, self.exog, data_predict=data_predict[i, :], W=self.W_in)
            mean[i] = np.squeeze(mean_mfx[0])
            mfx_c = np.squeeze(mean_mfx[1])
            mfx[i, :] = mfx_c
        return (mean, mfx)

class TestRegCoefC:
    """
    Significance test for continuous variables in a nonparametric regression.

    The null hypothesis is ``dE(Y|X)/dX_not_i = 0``, the alternative hypothesis
    is ``dE(Y|X)/dX_not_i != 0``.

    Parameters
    ----------
    model : KernelReg instance
        This is the nonparametric regression model whose elements
        are tested for significance.
    test_vars : tuple, list of integers, array_like
        index of position of the continuous variables to be tested
        for significance. E.g. (1,3,5) jointly tests variables at
        position 1,3 and 5 for significance.
    nboot : int
        Number of bootstrap samples used to determine the distribution
        of the test statistic in a finite sample. Default is 400
    nested_res : int
        Number of nested resamples used to calculate lambda.
        Must enable the pivot option
    pivot : bool
        Pivot the test statistic by dividing by its standard error
        Significantly increases computational time. But pivot statistics
        have more desirable properties
        (See references)

    Attributes
    ----------
    sig : str
        The significance level of the variable(s) tested
        "Not Significant": Not significant at the 90% confidence level
                            Fails to reject the null
        "*": Significant at the 90% confidence level
        "**": Significant at the 95% confidence level
        "***": Significant at the 99% confidence level

    Notes
    -----
    This class allows testing of joint hypothesis as long as all variables
    are continuous.

    References
    ----------
    Racine, J.: "Consistent Significance Testing for Nonparametric Regression"
    Journal of Business & Economics Statistics.

    Chapter 12 in [1].
    """

    def __init__(self, model, test_vars, nboot=400, nested_res=400, pivot=False):
        if False:
            return 10
        self.nboot = nboot
        self.nres = nested_res
        self.test_vars = test_vars
        self.model = model
        self.bw = model.bw
        self.var_type = model.var_type
        self.k_vars = len(self.var_type)
        self.endog = model.endog
        self.exog = model.exog
        self.gx = model.est[model.reg_type]
        self.test_vars = test_vars
        self.pivot = pivot
        self.run()

    def run(self):
        if False:
            i = 10
            return i + 15
        self.test_stat = self._compute_test_stat(self.endog, self.exog)
        self.sig = self._compute_sig()

    def _compute_test_stat(self, Y, X):
        if False:
            while True:
                i = 10
        '\n        Computes the test statistic.  See p.371 in [8].\n        '
        lam = self._compute_lambda(Y, X)
        t = lam
        if self.pivot:
            se_lam = self._compute_se_lambda(Y, X)
            t = lam / float(se_lam)
        return t

    def _compute_lambda(self, Y, X):
        if False:
            print('Hello World!')
        'Computes only lambda -- the main part of the test statistic'
        n = np.shape(X)[0]
        Y = _adjust_shape(Y, 1)
        X = _adjust_shape(X, self.k_vars)
        b = KernelReg(Y, X, self.var_type, self.model.reg_type, self.bw, defaults=EstimatorSettings(efficient=False)).fit()[1]
        b = b[:, self.test_vars]
        b = np.reshape(b, (n, len(self.test_vars)))
        fct = 1.0
        lam = ((b / fct) ** 2).sum() / float(n)
        return lam

    def _compute_se_lambda(self, Y, X):
        if False:
            print('Hello World!')
        '\n        Calculates the SE of lambda by nested resampling\n        Used to pivot the statistic.\n        Bootstrapping works better with estimating pivotal statistics\n        but slows down computation significantly.\n        '
        n = np.shape(Y)[0]
        lam = np.empty(shape=(self.nres,))
        for i in range(self.nres):
            ind = np.random.randint(0, n, size=(n, 1))
            Y1 = Y[ind, 0]
            X1 = X[ind, :]
            lam[i] = self._compute_lambda(Y1, X1)
        se_lambda = np.std(lam)
        return se_lambda

    def _compute_sig(self):
        if False:
            return 10
        '\n        Computes the significance value for the variable(s) tested.\n\n        The empirical distribution of the test statistic is obtained through\n        bootstrapping the sample.  The null hypothesis is rejected if the test\n        statistic is larger than the 90, 95, 99 percentiles.\n        '
        t_dist = np.empty(shape=(self.nboot,))
        Y = self.endog
        X = copy.deepcopy(self.exog)
        n = np.shape(Y)[0]
        X[:, self.test_vars] = np.mean(X[:, self.test_vars], axis=0)
        M = KernelReg(Y, X, self.var_type, self.model.reg_type, self.bw, defaults=EstimatorSettings(efficient=False)).fit()[0]
        M = np.reshape(M, (n, 1))
        e = Y - M
        e = e - np.mean(e)
        for i in range(self.nboot):
            ind = np.random.randint(0, n, size=(n, 1))
            e_boot = e[ind, 0]
            Y_boot = M + e_boot
            t_dist[i] = self._compute_test_stat(Y_boot, self.exog)
        self.t_dist = t_dist
        sig = 'Not Significant'
        if self.test_stat > mquantiles(t_dist, 0.9):
            sig = '*'
        if self.test_stat > mquantiles(t_dist, 0.95):
            sig = '**'
        if self.test_stat > mquantiles(t_dist, 0.99):
            sig = '***'
        return sig

class TestRegCoefD(TestRegCoefC):
    """
    Significance test for the categorical variables in a nonparametric
    regression.

    Parameters
    ----------
    model : Instance of KernelReg class
        This is the nonparametric regression model whose elements
        are tested for significance.
    test_vars : tuple, list of one element
        index of position of the discrete variable to be tested
        for significance. E.g. (3) tests variable at
        position 3 for significance.
    nboot : int
        Number of bootstrap samples used to determine the distribution
        of the test statistic in a finite sample. Default is 400

    Attributes
    ----------
    sig : str
        The significance level of the variable(s) tested
        "Not Significant": Not significant at the 90% confidence level
                            Fails to reject the null
        "*": Significant at the 90% confidence level
        "**": Significant at the 95% confidence level
        "***": Significant at the 99% confidence level

    Notes
    -----
    This class currently does not allow joint hypothesis.
    Only one variable can be tested at a time

    References
    ----------
    See [9] and chapter 12 in [1].
    """

    def _compute_test_stat(self, Y, X):
        if False:
            print('Hello World!')
        'Computes the test statistic'
        dom_x = np.sort(np.unique(self.exog[:, self.test_vars]))
        n = np.shape(X)[0]
        model = KernelReg(Y, X, self.var_type, self.model.reg_type, self.bw, defaults=EstimatorSettings(efficient=False))
        X1 = copy.deepcopy(X)
        X1[:, self.test_vars] = 0
        m0 = model.fit(data_predict=X1)[0]
        m0 = np.reshape(m0, (n, 1))
        zvec = np.zeros((n, 1))
        for i in dom_x[1:]:
            X1[:, self.test_vars] = i
            m1 = model.fit(data_predict=X1)[0]
            m1 = np.reshape(m1, (n, 1))
            zvec += (m1 - m0) ** 2
        avg = zvec.sum(axis=0) / float(n)
        return avg

    def _compute_sig(self):
        if False:
            while True:
                i = 10
        'Calculates the significance level of the variable tested'
        m = self._est_cond_mean()
        Y = self.endog
        X = self.exog
        n = np.shape(X)[0]
        u = Y - m
        u = u - np.mean(u)
        fct1 = (1 - 5 ** 0.5) / 2.0
        fct2 = (1 + 5 ** 0.5) / 2.0
        u1 = fct1 * u
        u2 = fct2 * u
        r = fct2 / 5 ** 0.5
        I_dist = np.empty((self.nboot, 1))
        for j in range(self.nboot):
            u_boot = copy.deepcopy(u2)
            prob = np.random.uniform(0, 1, size=(n, 1))
            ind = prob < r
            u_boot[ind] = u1[ind]
            Y_boot = m + u_boot
            I_dist[j] = self._compute_test_stat(Y_boot, X)
        sig = 'Not Significant'
        if self.test_stat > mquantiles(I_dist, 0.9):
            sig = '*'
        if self.test_stat > mquantiles(I_dist, 0.95):
            sig = '**'
        if self.test_stat > mquantiles(I_dist, 0.99):
            sig = '***'
        return sig

    def _est_cond_mean(self):
        if False:
            print('Hello World!')
        '\n        Calculates the expected conditional mean\n        m(X, Z=l) for all possible l\n        '
        self.dom_x = np.sort(np.unique(self.exog[:, self.test_vars]))
        X = copy.deepcopy(self.exog)
        m = 0
        for i in self.dom_x:
            X[:, self.test_vars] = i
            m += self.model.fit(data_predict=X)[0]
        m = m / float(len(self.dom_x))
        m = np.reshape(m, (np.shape(self.exog)[0], 1))
        return m