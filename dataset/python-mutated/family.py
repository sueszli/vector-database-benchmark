"""
The one parameter exponential family distributions used by GLM.
"""
import inspect
import warnings
import numpy as np
from scipy import special, stats
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.tools.sm_exceptions import ValueWarning
from . import links as L, varfuncs as V
FLOAT_EPS = np.finfo(float).eps

class Family:
    """
    The parent class for one-parameter exponential families.

    Parameters
    ----------
    link : a link function instance
        Link is the linear transformation function.
        See the individual families for available links.
    variance : a variance function
        Measures the variance as a function of the mean probabilities.
        See the individual families for the default variance function.
    check_link : bool
        If True (default), then and exception is raised if the link is invalid
        for the family.
        If False, then the link is not checked.

    See Also
    --------
    :ref:`links` : Further details on links.
    """
    valid = [-np.inf, np.inf]
    links = []

    def _setlink(self, link):
        if False:
            print('Hello World!')
        '\n        Helper method to set the link for a family.\n\n        Raises a ``ValueError`` exception if the link is not available. Note\n        that  the error message might not be that informative because it tells\n        you that the link should be in the base class for the link function.\n\n        See statsmodels.genmod.generalized_linear_model.GLM for a list of\n        appropriate links for each family but note that not all of these are\n        currently available.\n        '
        self._link = link
        if self._check_link:
            if not isinstance(link, L.Link):
                raise TypeError('The input should be a valid Link object.')
            if hasattr(self, 'links'):
                validlink = max([isinstance(link, _) for _ in self.links])
                if not validlink:
                    msg = 'Invalid link for family, should be in %s. (got %s)'
                    raise ValueError(msg % (repr(self.links), link))

    def _getlink(self):
        if False:
            i = 10
            return i + 15
        '\n        Helper method to get the link for a family.\n        '
        return self._link
    link = property(_getlink, _setlink, doc='Link function for family')

    def __init__(self, link, variance, check_link=True):
        if False:
            while True:
                i = 10
        self._check_link = check_link
        if inspect.isclass(link):
            warnmssg = 'Calling Family(..) with a link class is not allowed. Use an instance of a link class instead.'
            raise TypeError(warnmssg)
        self.link = link
        self.variance = variance

    def starting_mu(self, y):
        if False:
            for i in range(10):
                print('nop')
        '\n        Starting value for mu in the IRLS algorithm.\n\n        Parameters\n        ----------\n        y : ndarray\n            The untransformed response variable.\n\n        Returns\n        -------\n        mu_0 : ndarray\n            The first guess on the transformed response variable.\n\n        Notes\n        -----\n        .. math::\n\n           \\mu_0 = (Y + \\overline{Y})/2\n\n        Only the Binomial family takes a different initial value.\n        '
        return (y + y.mean()) / 2.0

    def weights(self, mu):
        if False:
            while True:
                i = 10
        "\n        Weights for IRLS steps\n\n        Parameters\n        ----------\n        mu : array_like\n            The transformed mean response variable in the exponential family\n\n        Returns\n        -------\n        w : ndarray\n            The weights for the IRLS steps\n\n        Notes\n        -----\n        .. math::\n\n           w = 1 / (g'(\\mu)^2  * Var(\\mu))\n        "
        return 1.0 / (self.link.deriv(mu) ** 2 * self.variance(mu))

    def deviance(self, endog, mu, var_weights=1.0, freq_weights=1.0, scale=1.0):
        if False:
            i = 10
            return i + 15
        '\n        The deviance function evaluated at (endog, mu, var_weights,\n        freq_weights, scale) for the distribution.\n\n        Deviance is usually defined as twice the loglikelihood ratio.\n\n        Parameters\n        ----------\n        endog : array_like\n            The endogenous response variable\n        mu : array_like\n            The inverse of the link function at the linear predicted values.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n        freq_weights : array_like\n            1d array of frequency weights. The default is 1.\n        scale : float, optional\n            An optional scale argument. The default is 1.\n\n        Returns\n        -------\n        Deviance : ndarray\n            The value of deviance function defined below.\n\n        Notes\n        -----\n        Deviance is defined\n\n        .. math::\n\n           D = 2\\sum_i (freq\\_weights_i * var\\_weights *\n           (llf(endog_i, endog_i) - llf(endog_i, \\mu_i)))\n\n        where y is the endogenous variable. The deviance functions are\n        analytically defined for each family.\n\n        Internally, we calculate deviance as:\n\n        .. math::\n           D = \\sum_i freq\\_weights_i * var\\_weights * resid\\_dev_i  / scale\n        '
        resid_dev = self._resid_dev(endog, mu)
        return np.sum(resid_dev * freq_weights * var_weights / scale)

    def resid_dev(self, endog, mu, var_weights=1.0, scale=1.0):
        if False:
            while True:
                i = 10
        '\n        The deviance residuals\n\n        Parameters\n        ----------\n        endog : array_like\n            The endogenous response variable\n        mu : array_like\n            The inverse of the link function at the linear predicted values.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n        scale : float, optional\n            An optional scale argument. The default is 1.\n\n        Returns\n        -------\n        resid_dev : float\n            Deviance residuals as defined below.\n\n        Notes\n        -----\n        The deviance residuals are defined by the contribution D_i of\n        observation i to the deviance as\n\n        .. math::\n           resid\\_dev_i = sign(y_i-\\mu_i) \\sqrt{D_i}\n\n        D_i is calculated from the _resid_dev method in each family.\n        Distribution-specific documentation of the calculation is available\n        there.\n        '
        resid_dev = self._resid_dev(endog, mu)
        resid_dev *= var_weights / scale
        return np.sign(endog - mu) * np.sqrt(np.clip(resid_dev, 0.0, np.inf))

    def fitted(self, lin_pred):
        if False:
            while True:
                i = 10
        '\n        Fitted values based on linear predictors lin_pred.\n\n        Parameters\n        ----------\n        lin_pred : ndarray\n            Values of the linear predictor of the model.\n            :math:`X \\cdot \\beta` in a classical linear model.\n\n        Returns\n        -------\n        mu : ndarray\n            The mean response variables given by the inverse of the link\n            function.\n        '
        fits = self.link.inverse(lin_pred)
        return fits

    def predict(self, mu):
        if False:
            for i in range(10):
                print('nop')
        '\n        Linear predictors based on given mu values.\n\n        Parameters\n        ----------\n        mu : ndarray\n            The mean response variables\n\n        Returns\n        -------\n        lin_pred : ndarray\n            Linear predictors based on the mean response variables.  The value\n            of the link function at the given mu.\n        '
        return self.link(mu)

    def loglike_obs(self, endog, mu, var_weights=1.0, scale=1.0):
        if False:
            for i in range(10):
                print('nop')
        '\n        The log-likelihood function for each observation in terms of the fitted\n        mean response for the distribution.\n\n        Parameters\n        ----------\n        endog : ndarray\n            Usually the endogenous response variable.\n        mu : ndarray\n            Usually but not always the fitted mean response variable.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n        scale : float\n            The scale parameter. The default is 1.\n\n        Returns\n        -------\n        ll_i : float\n            The value of the loglikelihood evaluated at\n            (endog, mu, var_weights, scale) as defined below.\n\n        Notes\n        -----\n        This is defined for each family. endog and mu are not restricted to\n        ``endog`` and ``mu`` respectively.  For instance, you could call\n        both ``loglike(endog, endog)`` and ``loglike(endog, mu)`` to get the\n        log-likelihood ratio.\n        '
        raise NotImplementedError

    def loglike(self, endog, mu, var_weights=1.0, freq_weights=1.0, scale=1.0):
        if False:
            while True:
                i = 10
        '\n        The log-likelihood function in terms of the fitted mean response.\n\n        Parameters\n        ----------\n        endog : ndarray\n            Usually the endogenous response variable.\n        mu : ndarray\n            Usually but not always the fitted mean response variable.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n        freq_weights : array_like\n            1d array of frequency weights. The default is 1.\n        scale : float\n            The scale parameter. The default is 1.\n\n        Returns\n        -------\n        ll : float\n            The value of the loglikelihood evaluated at\n            (endog, mu, var_weights, freq_weights, scale) as defined below.\n\n        Notes\n        -----\n        Where :math:`ll_i` is the by-observation log-likelihood:\n\n        .. math::\n           ll = \\sum(ll_i * freq\\_weights_i)\n\n        ``ll_i`` is defined for each family. endog and mu are not restricted\n        to ``endog`` and ``mu`` respectively.  For instance, you could call\n        both ``loglike(endog, endog)`` and ``loglike(endog, mu)`` to get the\n        log-likelihood ratio.\n        '
        ll_obs = self.loglike_obs(endog, mu, var_weights, scale)
        return np.sum(ll_obs * freq_weights)

    def resid_anscombe(self, endog, mu, var_weights=1.0, scale=1.0):
        if False:
            for i in range(10):
                print('nop')
        "\n        The Anscombe residuals\n\n        Parameters\n        ----------\n        endog : ndarray\n            The endogenous response variable\n        mu : ndarray\n            The inverse of the link function at the linear predicted values.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n        scale : float, optional\n            An optional argument to divide the residuals by sqrt(scale).\n            The default is 1.\n\n        See Also\n        --------\n        statsmodels.genmod.families.family.Family : `resid_anscombe` for the\n          individual families for more information\n\n        Notes\n        -----\n        Anscombe residuals are defined by\n\n        .. math::\n           resid\\_anscombe_i = \\frac{A(y)-A(\\mu)}{A'(\\mu)\\sqrt{Var[\\mu]}} *\n           \\sqrt(var\\_weights)\n\n        where :math:`A'(y)=v(y)^{-\\frac{1}{3}}` and :math:`v(\\mu)` is the\n        variance function :math:`Var[y]=\\frac{\\phi}{w}v(mu)`.\n        The transformation :math:`A(y)` makes the residuals more normal\n        distributed.\n        "
        raise NotImplementedError

    def _clean(self, x):
        if False:
            print('Hello World!')
        '\n        Helper function to trim the data so that it is in (0,inf)\n\n        Notes\n        -----\n        The need for this function was discovered through usage and its\n        possible that other families might need a check for validity of the\n        domain.\n        '
        return np.clip(x, FLOAT_EPS, np.inf)

class Poisson(Family):
    """
    Poisson exponential family.

    Parameters
    ----------
    link : a link instance, optional
        The default link for the Poisson family is the log link. Available
        links are log, identity, and sqrt. See statsmodels.families.links for
        more information.
    check_link : bool
        If True (default), then and exception is raised if the link is invalid
        for the family.
        If False, then the link is not checked.

    Attributes
    ----------
    Poisson.link : a link instance
        The link function of the Poisson instance.
    Poisson.variance : varfuncs instance
        ``variance`` is an instance of
        statsmodels.genmod.families.varfuncs.mu

    See Also
    --------
    statsmodels.genmod.families.family.Family : Parent class for all links.
    :ref:`links` : Further details on links.
    """
    links = [L.Log, L.Identity, L.Sqrt]
    variance = V.mu
    valid = [0, np.inf]
    safe_links = [L.Log]

    def __init__(self, link=None, check_link=True):
        if False:
            return 10
        if link is None:
            link = L.Log()
        super(Poisson, self).__init__(link=link, variance=Poisson.variance, check_link=check_link)

    def _resid_dev(self, endog, mu):
        if False:
            while True:
                i = 10
        '\n        Poisson deviance residuals\n\n        Parameters\n        ----------\n        endog : ndarray\n            The endogenous response variable.\n        mu : ndarray\n            The inverse of the link function at the linear predicted values.\n\n        Returns\n        -------\n        resid_dev : float\n            Deviance residuals as defined below.\n\n        Notes\n        -----\n        .. math::\n\n           resid\\_dev_i = 2 * (endog_i * \\ln(endog_i / \\mu_i) -\n           (endog_i - \\mu_i))\n        '
        endog_mu = self._clean(endog / mu)
        resid_dev = endog * np.log(endog_mu) - (endog - mu)
        return 2 * resid_dev

    def loglike_obs(self, endog, mu, var_weights=1.0, scale=1.0):
        if False:
            while True:
                i = 10
        '\n        The log-likelihood function for each observation in terms of the fitted\n        mean response for the Poisson distribution.\n\n        Parameters\n        ----------\n        endog : ndarray\n            Usually the endogenous response variable.\n        mu : ndarray\n            Usually but not always the fitted mean response variable.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n        scale : float\n            The scale parameter. The default is 1.\n\n        Returns\n        -------\n        ll_i : float\n            The value of the loglikelihood evaluated at\n            (endog, mu, var_weights, scale) as defined below.\n\n        Notes\n        -----\n        .. math::\n            ll_i = var\\_weights_i / scale * (endog_i * \\ln(\\mu_i) - \\mu_i -\n            \\ln \\Gamma(endog_i + 1))\n        '
        return var_weights / scale * (endog * np.log(mu) - mu - special.gammaln(endog + 1))

    def resid_anscombe(self, endog, mu, var_weights=1.0, scale=1.0):
        if False:
            i = 10
            return i + 15
        '\n        The Anscombe residuals\n\n        Parameters\n        ----------\n        endog : ndarray\n            The endogenous response variable\n        mu : ndarray\n            The inverse of the link function at the linear predicted values.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n        scale : float, optional\n            An optional argument to divide the residuals by sqrt(scale).\n            The default is 1.\n\n        Returns\n        -------\n        resid_anscombe : ndarray\n            The Anscombe residuals for the Poisson family defined below\n\n        Notes\n        -----\n        .. math::\n\n           resid\\_anscombe_i = (3/2) * (endog_i^{2/3} - \\mu_i^{2/3}) /\n           \\mu_i^{1/6} * \\sqrt(var\\_weights)\n        '
        resid = 3 / 2.0 * (endog ** (2 / 3.0) - mu ** (2 / 3.0)) / (mu ** (1 / 6.0) * scale ** 0.5)
        resid *= np.sqrt(var_weights)
        return resid

    def get_distribution(self, mu, scale=1.0, var_weights=1.0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Frozen Poisson distribution instance for given parameters\n\n        Parameters\n        ----------\n        mu : ndarray\n            Usually but not always the fitted mean response variable.\n        scale : float\n            The scale parameter is ignored.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n            var_weights are ignored for Poisson.\n\n        Returns\n        -------\n        distribution instance\n\n        '
        return stats.poisson(mu)

class Gaussian(Family):
    """
    Gaussian exponential family distribution.

    Parameters
    ----------
    link : a link instance, optional
        The default link for the Gaussian family is the identity link.
        Available links are log, identity, and inverse.
        See statsmodels.genmod.families.links for more information.
    check_link : bool
        If True (default), then and exception is raised if the link is invalid
        for the family.
        If False, then the link is not checked.

    Attributes
    ----------
    Gaussian.link : a link instance
        The link function of the Gaussian instance
    Gaussian.variance : varfunc instance
        ``variance`` is an instance of
        statsmodels.genmod.families.varfuncs.constant

    See Also
    --------
    statsmodels.genmod.families.family.Family : Parent class for all links.
    :ref:`links` : Further details on links.
    """
    links = [L.Log, L.Identity, L.InversePower]
    variance = V.constant
    safe_links = links

    def __init__(self, link=None, check_link=True):
        if False:
            for i in range(10):
                print('nop')
        if link is None:
            link = L.Identity()
        super(Gaussian, self).__init__(link=link, variance=Gaussian.variance, check_link=check_link)

    def _resid_dev(self, endog, mu):
        if False:
            return 10
        '\n        Gaussian deviance residuals\n\n        Parameters\n        ----------\n        endog : ndarray\n            The endogenous response variable.\n        mu : ndarray\n            The inverse of the link function at the linear predicted values.\n\n        Returns\n        -------\n        resid_dev : float\n            Deviance residuals as defined below.\n\n        Notes\n        -----\n        .. math::\n\n           resid\\_dev_i = (endog_i - \\mu_i) ** 2\n        '
        return (endog - mu) ** 2

    def loglike_obs(self, endog, mu, var_weights=1.0, scale=1.0):
        if False:
            i = 10
            return i + 15
        '\n        The log-likelihood function for each observation in terms of the fitted\n        mean response for the Gaussian distribution.\n\n        Parameters\n        ----------\n        endog : ndarray\n            Usually the endogenous response variable.\n        mu : ndarray\n            Usually but not always the fitted mean response variable.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n        scale : float\n            The scale parameter. The default is 1.\n\n        Returns\n        -------\n        ll_i : float\n            The value of the loglikelihood evaluated at\n            (endog, mu, var_weights, scale) as defined below.\n\n        Notes\n        -----\n        If the link is the identity link function then the\n        loglikelihood function is the same as the classical OLS model.\n\n        .. math::\n\n           llf = -nobs / 2 * (\\log(SSR) + (1 + \\log(2 \\pi / nobs)))\n\n        where\n\n        .. math::\n\n           SSR = \\sum_i (Y_i - g^{-1}(\\mu_i))^2\n\n        If the links is not the identity link then the loglikelihood\n        function is defined as\n\n        .. math::\n\n           ll_i = -1 / 2 \\sum_i  * var\\_weights * ((Y_i - mu_i)^2 / scale +\n                                                \\log(2 * \\pi * scale))\n        '
        ll_obs = -var_weights * (endog - mu) ** 2 / scale
        ll_obs += -np.log(scale / var_weights) - np.log(2 * np.pi)
        ll_obs /= 2
        return ll_obs

    def resid_anscombe(self, endog, mu, var_weights=1.0, scale=1.0):
        if False:
            for i in range(10):
                print('nop')
        '\n        The Anscombe residuals\n\n        Parameters\n        ----------\n        endog : ndarray\n            The endogenous response variable\n        mu : ndarray\n            The inverse of the link function at the linear predicted values.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n        scale : float, optional\n            An optional argument to divide the residuals by sqrt(scale).\n            The default is 1.\n\n        Returns\n        -------\n        resid_anscombe : ndarray\n            The Anscombe residuals for the Gaussian family defined below\n\n        Notes\n        -----\n        For the Gaussian distribution, Anscombe residuals are the same as\n        deviance residuals.\n\n        .. math::\n\n           resid\\_anscombe_i = (Y_i - \\mu_i) / \\sqrt{scale} *\n           \\sqrt(var\\_weights)\n        '
        resid = (endog - mu) / scale ** 0.5
        resid *= np.sqrt(var_weights)
        return resid

    def get_distribution(self, mu, scale, var_weights=1.0):
        if False:
            print('Hello World!')
        '\n        Frozen Gaussian distribution instance for given parameters\n\n        Parameters\n        ----------\n        mu : ndarray\n            Usually but not always the fitted mean response variable.\n        scale : float\n            The scale parameter is required argument for get_distribution.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n\n        Returns\n        -------\n        distribution instance\n\n        '
        scale_n = scale / var_weights
        return stats.norm(loc=mu, scale=np.sqrt(scale_n))

class Gamma(Family):
    """
    Gamma exponential family distribution.

    Parameters
    ----------
    link : a link instance, optional
        The default link for the Gamma family is the inverse link.
        Available links are log, identity, and inverse.
        See statsmodels.genmod.families.links for more information.
    check_link : bool
        If True (default), then and exception is raised if the link is invalid
        for the family.
        If False, then the link is not checked.

    Attributes
    ----------
    Gamma.link : a link instance
        The link function of the Gamma instance
    Gamma.variance : varfunc instance
        ``variance`` is an instance of
        statsmodels.genmod.family.varfuncs.mu_squared

    See Also
    --------
    statsmodels.genmod.families.family.Family : Parent class for all links.
    :ref:`links` : Further details on links.
    """
    links = [L.Log, L.Identity, L.InversePower]
    variance = V.mu_squared
    safe_links = [L.Log]

    def __init__(self, link=None, check_link=True):
        if False:
            for i in range(10):
                print('nop')
        if link is None:
            link = L.InversePower()
        super(Gamma, self).__init__(link=link, variance=Gamma.variance, check_link=check_link)

    def _resid_dev(self, endog, mu):
        if False:
            return 10
        '\n        Gamma deviance residuals\n\n        Parameters\n        ----------\n        endog : ndarray\n            The endogenous response variable.\n        mu : ndarray\n            The inverse of the link function at the linear predicted values.\n\n        Returns\n        -------\n        resid_dev : float\n            Deviance residuals as defined below.\n\n        Notes\n        -----\n        .. math::\n\n           resid\\_dev_i = 2 * ((endog_i - \\mu_i) / \\mu_i -\n           \\log(endog_i / \\mu_i))\n        '
        endog_mu = self._clean(endog / mu)
        resid_dev = -np.log(endog_mu) + (endog - mu) / mu
        return 2 * resid_dev

    def loglike_obs(self, endog, mu, var_weights=1.0, scale=1.0):
        if False:
            while True:
                i = 10
        '\n        The log-likelihood function for each observation in terms of the fitted\n        mean response for the Gamma distribution.\n\n        Parameters\n        ----------\n        endog : ndarray\n            Usually the endogenous response variable.\n        mu : ndarray\n            Usually but not always the fitted mean response variable.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n        scale : float\n            The scale parameter. The default is 1.\n\n        Returns\n        -------\n        ll_i : float\n            The value of the loglikelihood evaluated at\n            (endog, mu, var_weights, scale) as defined below.\n\n        Notes\n        -----\n        .. math::\n\n           ll_i = var\\_weights_i / scale * (\\ln(var\\_weights_i * endog_i /\n           (scale * \\mu_i)) - (var\\_weights_i * endog_i) /\n           (scale * \\mu_i)) - \\ln \\Gamma(var\\_weights_i / scale) - \\ln(\\mu_i)\n        '
        endog_mu = self._clean(endog / mu)
        weight_scale = var_weights / scale
        ll_obs = weight_scale * np.log(weight_scale * endog_mu)
        ll_obs -= weight_scale * endog_mu
        ll_obs -= special.gammaln(weight_scale) + np.log(endog)
        return ll_obs

    def resid_anscombe(self, endog, mu, var_weights=1.0, scale=1.0):
        if False:
            return 10
        '\n        The Anscombe residuals\n\n        Parameters\n        ----------\n        endog : ndarray\n            The endogenous response variable\n        mu : ndarray\n            The inverse of the link function at the linear predicted values.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n        scale : float, optional\n            An optional argument to divide the residuals by sqrt(scale).\n            The default is 1.\n\n        Returns\n        -------\n        resid_anscombe : ndarray\n            The Anscombe residuals for the Gamma family defined below\n\n        Notes\n        -----\n        .. math::\n\n           resid\\_anscombe_i = 3 * (endog_i^{1/3} - \\mu_i^{1/3}) / \\mu_i^{1/3}\n           / \\sqrt{scale} * \\sqrt(var\\_weights)\n        '
        resid = 3 * (endog ** (1 / 3.0) - mu ** (1 / 3.0)) / mu ** (1 / 3.0) / scale ** 0.5
        resid *= np.sqrt(var_weights)
        return resid

    def get_distribution(self, mu, scale, var_weights=1.0):
        if False:
            return 10
        '\n        Frozen Gamma distribution instance for given parameters\n\n        Parameters\n        ----------\n        mu : ndarray\n            Usually but not always the fitted mean response variable.\n        scale : float\n            The scale parameter is required argument for get_distribution.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n\n        Returns\n        -------\n        distribution instance\n\n        '
        scale_ = scale / var_weights
        shape = 1 / scale_
        scale_g = mu * scale_
        return stats.gamma(shape, scale=scale_g)

class Binomial(Family):
    """
    Binomial exponential family distribution.

    Parameters
    ----------
    link : a link instance, optional
        The default link for the Binomial family is the logit link.
        Available links are logit, probit, cauchy, log, loglog, and cloglog.
        See statsmodels.genmod.families.links for more information.
    check_link : bool
        If True (default), then and exception is raised if the link is invalid
        for the family.
        If False, then the link is not checked.

    Attributes
    ----------
    Binomial.link : a link instance
        The link function of the Binomial instance
    Binomial.variance : varfunc instance
        ``variance`` is an instance of
        statsmodels.genmod.families.varfuncs.binary

    See Also
    --------
    statsmodels.genmod.families.family.Family : Parent class for all links.
    :ref:`links` : Further details on links.

    Notes
    -----
    endog for Binomial can be specified in one of three ways:
    A 1d array of 0 or 1 values, indicating failure or success
    respectively.
    A 2d array, with two columns. The first column represents the
    success count and the second column represents the failure
    count.
    A 1d array of proportions, indicating the proportion of
    successes, with parameter `var_weights` containing the
    number of trials for each row.
    """
    links = [L.Logit, L.Probit, L.Cauchy, L.Log, L.LogC, L.CLogLog, L.LogLog, L.Identity]
    variance = V.binary
    safe_links = [L.Logit, L.CDFLink]

    def __init__(self, link=None, check_link=True):
        if False:
            while True:
                i = 10
        if link is None:
            link = L.Logit()
        self.n = 1
        super(Binomial, self).__init__(link=link, variance=V.Binomial(n=self.n), check_link=check_link)

    def starting_mu(self, y):
        if False:
            print('Hello World!')
        '\n        The starting values for the IRLS algorithm for the Binomial family.\n        A good choice for the binomial family is :math:`\\mu_0 = (Y_i + 0.5)/2`\n        '
        return (y + 0.5) / 2

    def initialize(self, endog, freq_weights):
        if False:
            i = 10
            return i + 15
        '\n        Initialize the response variable.\n\n        Parameters\n        ----------\n        endog : ndarray\n            Endogenous response variable\n        freq_weights : ndarray\n            1d array of frequency weights\n\n        Returns\n        -------\n        If `endog` is binary, returns `endog`\n\n        If `endog` is a 2d array, then the input is assumed to be in the format\n        (successes, failures) and\n        successes/(success + failures) is returned.  And n is set to\n        successes + failures.\n        '
        if endog.ndim > 1 and endog.shape[1] > 2:
            raise ValueError('endog has more than 2 columns. The Binomial link supports either a single response variable or a paired response variable.')
        elif endog.ndim > 1 and endog.shape[1] > 1:
            y = endog[:, 0]
            self.n = endog.sum(1)
            return (y * 1.0 / self.n, self.n)
        else:
            return (endog, np.ones(endog.shape[0]))

    def _resid_dev(self, endog, mu):
        if False:
            while True:
                i = 10
        '\n        Binomial deviance residuals\n\n        Parameters\n        ----------\n        endog : ndarray\n            The endogenous response variable.\n        mu : ndarray\n            The inverse of the link function at the linear predicted values.\n\n        Returns\n        -------\n        resid_dev : float\n            Deviance residuals as defined below.\n\n        Notes\n        -----\n        .. math::\n\n           resid\\_dev_i = 2 * n * (endog_i * \\ln(endog_i /\\mu_i) +\n           (1 - endog_i) * \\ln((1 - endog_i) / (1 - \\mu_i)))\n        '
        endog_mu = self._clean(endog / (mu + 1e-20))
        n_endog_mu = self._clean((1.0 - endog) / (1.0 - mu + 1e-20))
        resid_dev = endog * np.log(endog_mu) + (1 - endog) * np.log(n_endog_mu)
        return 2 * self.n * resid_dev

    def loglike_obs(self, endog, mu, var_weights=1.0, scale=1.0):
        if False:
            return 10
        '\n        The log-likelihood function for each observation in terms of the fitted\n        mean response for the Binomial distribution.\n\n        Parameters\n        ----------\n        endog : ndarray\n            Usually the endogenous response variable.\n        mu : ndarray\n            Usually but not always the fitted mean response variable.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n        scale : float\n            The scale parameter. The default is 1.\n\n        Returns\n        -------\n        ll_i : float\n            The value of the loglikelihood evaluated at\n            (endog, mu, var_weights, scale) as defined below.\n\n        Notes\n        -----\n        If the endogenous variable is binary:\n\n        .. math::\n\n         ll_i = \\sum_i (y_i * \\log(\\mu_i/(1-\\mu_i)) + \\log(1-\\mu_i)) *\n               var\\_weights_i\n\n        If the endogenous variable is binomial:\n\n        .. math::\n\n           ll_i = \\sum_i var\\_weights_i * (\\ln \\Gamma(n+1) -\n                  \\ln \\Gamma(y_i + 1) - \\ln \\Gamma(n_i - y_i +1) + y_i *\n                  \\log(\\mu_i / (n_i - \\mu_i)) + n * \\log(1 - \\mu_i/n_i))\n\n        where :math:`y_i = Y_i * n_i` with :math:`Y_i` and :math:`n_i` as\n        defined in Binomial initialize.  This simply makes :math:`y_i` the\n        original number of successes.\n        '
        n = self.n
        y = endog * n
        return (special.gammaln(n + 1) - special.gammaln(y + 1) - special.gammaln(n - y + 1) + y * np.log(mu / (1 - mu + 1e-20)) + n * np.log(1 - mu + 1e-20)) * var_weights

    def resid_anscombe(self, endog, mu, var_weights=1.0, scale=1.0):
        if False:
            return 10
        '\n        The Anscombe residuals\n\n        Parameters\n        ----------\n        endog : ndarray\n            The endogenous response variable\n        mu : ndarray\n            The inverse of the link function at the linear predicted values.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n        scale : float, optional\n            An optional argument to divide the residuals by sqrt(scale).\n            The default is 1.\n\n        Returns\n        -------\n        resid_anscombe : ndarray\n            The Anscombe residuals as defined below.\n\n        Notes\n        -----\n        .. math::\n\n            n^{2/3}*(cox\\_snell(endog)-cox\\_snell(mu)) /\n            (mu*(1-mu/n)*scale^3)^{1/6} * \\sqrt(var\\_weights)\n\n        where cox_snell is defined as\n        cox_snell(x) = betainc(2/3., 2/3., x)*betainc(2/3.,2/3.)\n        where betainc is the incomplete beta function as defined in scipy,\n        which uses a regularized version (with the unregularized version, one\n        would just have :math:`cox_snell(x) = Betainc(2/3., 2/3., x)`).\n\n        The name \'cox_snell\' is idiosyncratic and is simply used for\n        convenience following the approach suggested in Cox and Snell (1968).\n        Further note that\n        :math:`cox\\_snell(x) = \\frac{3}{2}*x^{2/3} *\n        hyp2f1(2/3.,1/3.,5/3.,x)`\n        where hyp2f1 is the hypergeometric 2f1 function.  The Anscombe\n        residuals are sometimes defined in the literature using the\n        hyp2f1 formulation.  Both betainc and hyp2f1 can be found in scipy.\n\n        References\n        ----------\n        Anscombe, FJ. (1953) "Contribution to the discussion of H. Hotelling\'s\n            paper." Journal of the Royal Statistical Society B. 15, 229-30.\n\n        Cox, DR and Snell, EJ. (1968) "A General Definition of Residuals."\n            Journal of the Royal Statistical Society B. 30, 248-75.\n        '
        endog = endog * self.n
        mu = mu * self.n

        def cox_snell(x):
            if False:
                print('Hello World!')
            return special.betainc(2 / 3.0, 2 / 3.0, x) * special.beta(2 / 3.0, 2 / 3.0)
        resid = self.n ** (2 / 3.0) * (cox_snell(endog * 1.0 / self.n) - cox_snell(mu * 1.0 / self.n)) / (mu * (1 - mu * 1.0 / self.n) * scale ** 3) ** (1 / 6.0)
        resid *= np.sqrt(var_weights)
        return resid

    def get_distribution(self, mu, scale=1.0, var_weights=1.0, n_trials=1):
        if False:
            return 10
        '\n        Frozen Binomial distribution instance for given parameters\n\n        Parameters\n        ----------\n        mu : ndarray\n            Usually but not always the fitted mean response variable.\n        scale : float\n            The scale parameter is ignored.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n            var_weights are ignored for Poisson.\n        n_trials : int\n            Number of trials for the binomial distribution. The default is 1\n            which corresponds to a Bernoulli random variable.\n\n        Returns\n        -------\n        distribution instance\n\n        '
        return stats.binom(n=n_trials, p=mu)

class InverseGaussian(Family):
    """
    InverseGaussian exponential family.

    Parameters
    ----------
    link : a link instance, optional
        The default link for the inverse Gaussian family is the
        inverse squared link.
        Available links are InverseSquared, Inverse, Log, and Identity.
        See statsmodels.genmod.families.links for more information.
    check_link : bool
        If True (default), then and exception is raised if the link is invalid
        for the family.
        If False, then the link is not checked.

    Attributes
    ----------
    InverseGaussian.link : a link instance
        The link function of the inverse Gaussian instance
    InverseGaussian.variance : varfunc instance
        ``variance`` is an instance of
        statsmodels.genmod.families.varfuncs.mu_cubed

    See Also
    --------
    statsmodels.genmod.families.family.Family : Parent class for all links.
    :ref:`links` : Further details on links.

    Notes
    -----
    The inverse Gaussian distribution is sometimes referred to in the
    literature as the Wald distribution.
    """
    links = [L.InverseSquared, L.InversePower, L.Identity, L.Log]
    variance = V.mu_cubed
    safe_links = [L.InverseSquared, L.Log]

    def __init__(self, link=None, check_link=True):
        if False:
            while True:
                i = 10
        if link is None:
            link = L.InverseSquared()
        super(InverseGaussian, self).__init__(link=link, variance=InverseGaussian.variance, check_link=check_link)

    def _resid_dev(self, endog, mu):
        if False:
            for i in range(10):
                print('nop')
        '\n        Inverse Gaussian deviance residuals\n\n        Parameters\n        ----------\n        endog : ndarray\n            The endogenous response variable.\n        mu : ndarray\n            The inverse of the link function at the linear predicted values.\n\n        Returns\n        -------\n        resid_dev : float\n            Deviance residuals as defined below.\n\n        Notes\n        -----\n        .. math::\n\n           resid\\_dev_i = 1 / (endog_i * \\mu_i^2) * (endog_i - \\mu_i)^2\n        '
        return 1.0 / (endog * mu ** 2) * (endog - mu) ** 2

    def loglike_obs(self, endog, mu, var_weights=1.0, scale=1.0):
        if False:
            return 10
        '\n        The log-likelihood function for each observation in terms of the fitted\n        mean response for the Inverse Gaussian distribution.\n\n        Parameters\n        ----------\n        endog : ndarray\n            Usually the endogenous response variable.\n        mu : ndarray\n            Usually but not always the fitted mean response variable.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n        scale : float\n            The scale parameter. The default is 1.\n\n        Returns\n        -------\n        ll_i : float\n            The value of the loglikelihood evaluated at\n            (endog, mu, var_weights, scale) as defined below.\n\n        Notes\n        -----\n        .. math::\n\n           ll_i = -1/2 * (var\\_weights_i * (endog_i - \\mu_i)^2 /\n           (scale * endog_i * \\mu_i^2) + \\ln(scale * \\endog_i^3 /\n           var\\_weights_i) - \\ln(2 * \\pi))\n        '
        ll_obs = -var_weights * (endog - mu) ** 2 / (scale * endog * mu ** 2)
        ll_obs += -np.log(scale * endog ** 3 / var_weights) - np.log(2 * np.pi)
        ll_obs /= 2
        return ll_obs

    def resid_anscombe(self, endog, mu, var_weights=1.0, scale=1.0):
        if False:
            i = 10
            return i + 15
        '\n        The Anscombe residuals\n\n        Parameters\n        ----------\n        endog : ndarray\n            The endogenous response variable\n        mu : ndarray\n            The inverse of the link function at the linear predicted values.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n        scale : float, optional\n            An optional argument to divide the residuals by sqrt(scale).\n            The default is 1.\n\n        Returns\n        -------\n        resid_anscombe : ndarray\n            The Anscombe residuals for the inverse Gaussian distribution  as\n            defined below\n\n        Notes\n        -----\n        .. math::\n\n           resid\\_anscombe_i = \\log(Y_i / \\mu_i) / \\sqrt{\\mu_i * scale} *\n           \\sqrt(var\\_weights)\n        '
        resid = np.log(endog / mu) / np.sqrt(mu * scale)
        resid *= np.sqrt(var_weights)
        return resid

    def get_distribution(self, mu, scale, var_weights=1.0):
        if False:
            i = 10
            return i + 15
        '\n        Frozen Inverse Gaussian distribution instance for given parameters\n\n        Parameters\n        ----------\n        mu : ndarray\n            Usually but not always the fitted mean response variable.\n        scale : float\n            The scale parameter is required argument for get_distribution.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n\n        Returns\n        -------\n        distribution instance\n\n        '
        scale_ = scale / var_weights
        mu_ig = mu * scale_
        return stats.invgauss(mu_ig, scale=1 / scale_)

class NegativeBinomial(Family):
    """
    Negative Binomial exponential family (corresponds to NB2).

    Parameters
    ----------
    link : a link instance, optional
        The default link for the negative binomial family is the log link.
        Available links are log, cloglog, identity, nbinom and power.
        See statsmodels.genmod.families.links for more information.
    alpha : float, optional
        The ancillary parameter for the negative binomial distribution.
        For now ``alpha`` is assumed to be nonstochastic.  The default value
        is 1.  Permissible values are usually assumed to be between .01 and 2.
    check_link : bool
        If True (default), then and exception is raised if the link is invalid
        for the family.
        If False, then the link is not checked.

    Attributes
    ----------
    NegativeBinomial.link : a link instance
        The link function of the negative binomial instance
    NegativeBinomial.variance : varfunc instance
        ``variance`` is an instance of
        statsmodels.genmod.families.varfuncs.nbinom

    See Also
    --------
    statsmodels.genmod.families.family.Family : Parent class for all links.
    :ref:`links` : Further details on links.

    Notes
    -----
    Power link functions are not yet supported.

    Parameterization for :math:`y=0, 1, 2, \\ldots` is

    .. math::

       f(y) = \\frac{\\Gamma(y+\\frac{1}{\\alpha})}{y!\\Gamma(\\frac{1}{\\alpha})}
              \\left(\\frac{1}{1+\\alpha\\mu}\\right)^{\\frac{1}{\\alpha}}
              \\left(\\frac{\\alpha\\mu}{1+\\alpha\\mu}\\right)^y

    with :math:`E[Y]=\\mu\\,` and :math:`Var[Y]=\\mu+\\alpha\\mu^2`.
    """
    links = [L.Log, L.CLogLog, L.Identity, L.NegativeBinomial, L.Power]
    variance = V.nbinom
    safe_links = [L.Log]

    def __init__(self, link=None, alpha=1.0, check_link=True):
        if False:
            while True:
                i = 10
        self.alpha = 1.0 * alpha
        if alpha is self.__init__.__defaults__[1]:
            warnings.warn(f'Negative binomial dispersion parameter alpha not set. Using default value alpha={alpha}.', ValueWarning)
        if link is None:
            link = L.Log()
        super(NegativeBinomial, self).__init__(link=link, variance=V.NegativeBinomial(alpha=self.alpha), check_link=check_link)

    def _resid_dev(self, endog, mu):
        if False:
            return 10
        '\n        Negative Binomial deviance residuals\n\n        Parameters\n        ----------\n        endog : ndarray\n            The endogenous response variable.\n        mu : ndarray\n            The inverse of the link function at the linear predicted values.\n\n        Returns\n        -------\n        resid_dev : float\n            Deviance residuals as defined below.\n\n        Notes\n        -----\n        .. math::\n\n            resid_dev_i = 2 * (endog_i * \\ln(endog_i /\n            \\mu_i) - (endog_i + 1 / \\alpha) * \\ln((endog_i + 1 / \\alpha) /\n            (\\mu_i + 1 / \\alpha)))\n        '
        endog_mu = self._clean(endog / mu)
        endog_alpha = endog + 1 / self.alpha
        mu_alpha = mu + 1 / self.alpha
        resid_dev = endog * np.log(endog_mu)
        resid_dev -= endog_alpha * np.log(endog_alpha / mu_alpha)
        return 2 * resid_dev

    def loglike_obs(self, endog, mu, var_weights=1.0, scale=1.0):
        if False:
            while True:
                i = 10
        '\n        The log-likelihood function for each observation in terms of the fitted\n        mean response for the Negative Binomial distribution.\n\n        Parameters\n        ----------\n        endog : ndarray\n            Usually the endogenous response variable.\n        mu : ndarray\n            Usually but not always the fitted mean response variable.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n        scale : float\n            The scale parameter. The default is 1.\n\n        Returns\n        -------\n        ll_i : float\n            The value of the loglikelihood evaluated at\n            (endog, mu, var_weights, scale) as defined below.\n\n        Notes\n        -----\n        Defined as:\n\n        .. math::\n\n           llf = \\sum_i var\\_weights_i / scale * (Y_i * \\log{(\\alpha * \\mu_i /\n                 (1 + \\alpha * \\mu_i))} - \\log{(1 + \\alpha * \\mu_i)}/\n                 \\alpha + Constant)\n\n        where :math:`Constant` is defined as:\n\n        .. math::\n\n           Constant = \\ln \\Gamma{(Y_i + 1/ \\alpha )} - \\ln \\Gamma(Y_i + 1) -\n                      \\ln \\Gamma{(1/ \\alpha )}\n\n        constant = (special.gammaln(endog + 1 / self.alpha) -\n                    special.gammaln(endog+1)-special.gammaln(1/self.alpha))\n        return (endog * np.log(self.alpha * mu / (1 + self.alpha * mu)) -\n                np.log(1 + self.alpha * mu) / self.alpha +\n                constant) * var_weights / scale\n        '
        ll_obs = endog * np.log(self.alpha * mu)
        ll_obs -= (endog + 1 / self.alpha) * np.log(1 + self.alpha * mu)
        ll_obs += special.gammaln(endog + 1 / self.alpha)
        ll_obs -= special.gammaln(1 / self.alpha)
        ll_obs -= special.gammaln(endog + 1)
        return var_weights / scale * ll_obs

    def resid_anscombe(self, endog, mu, var_weights=1.0, scale=1.0):
        if False:
            while True:
                i = 10
        '\n        The Anscombe residuals\n\n        Parameters\n        ----------\n        endog : ndarray\n            The endogenous response variable\n        mu : ndarray\n            The inverse of the link function at the linear predicted values.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n        scale : float, optional\n            An optional argument to divide the residuals by sqrt(scale).\n            The default is 1.\n\n        Returns\n        -------\n        resid_anscombe : ndarray\n            The Anscombe residuals as defined below.\n\n        Notes\n        -----\n        Anscombe residuals for Negative Binomial are the same as for Binomial\n        upon setting :math:`n=-\\frac{1}{\\alpha}`. Due to the negative value of\n        :math:`-\\alpha*Y` the representation with the hypergeometric function\n        :math:`H2F1(x) =  hyp2f1(2/3.,1/3.,5/3.,x)` is advantageous\n\n        .. math::\n\n            resid\\_anscombe_i = \\frac{3}{2} *\n            (Y_i^(2/3)*H2F1(-\\alpha*Y_i) - \\mu_i^(2/3)*H2F1(-\\alpha*\\mu_i))\n            / (\\mu_i * (1+\\alpha*\\mu_i) * scale^3)^(1/6) * \\sqrt(var\\_weights)\n\n        Note that for the (unregularized) Beta function, one has\n        :math:`Beta(z,a,b) = z^a/a * H2F1(a,1-b,a+1,z)`\n        '

        def hyp2f1(x):
            if False:
                print('Hello World!')
            return special.hyp2f1(2 / 3.0, 1 / 3.0, 5 / 3.0, x)
        resid = 3 / 2.0 * (endog ** (2 / 3.0) * hyp2f1(-self.alpha * endog) - mu ** (2 / 3.0) * hyp2f1(-self.alpha * mu)) / (mu * (1 + self.alpha * mu) * scale ** 3) ** (1 / 6.0)
        resid *= np.sqrt(var_weights)
        return resid

    def get_distribution(self, mu, scale=1.0, var_weights=1.0):
        if False:
            print('Hello World!')
        '\n        Frozen NegativeBinomial distribution instance for given parameters\n\n        Parameters\n        ----------\n        mu : ndarray\n            Usually but not always the fitted mean response variable.\n        scale : float\n            The scale parameter is ignored.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n            var_weights are ignored for NegativeBinomial.\n\n        Returns\n        -------\n        distribution instance\n\n        '
        size = 1.0 / self.alpha
        prob = size / (size + mu)
        return stats.nbinom(size, prob)

class Tweedie(Family):
    """
    Tweedie family.

    Parameters
    ----------
    link : a link instance, optional
        The default link for the Tweedie family is the log link.
        Available links are log, Power and any aliases of power.
        See statsmodels.genmod.families.links for more information.
    var_power : float, optional
        The variance power. The default is 1.
    eql : bool
        If True, the Extended Quasi-Likelihood is used, else the
        likelihood is used.
        In both cases, for likelihood computations the var_power
        must be between 1 and 2.
    check_link : bool
        If True (default), then and exception is raised if the link is invalid
        for the family.
        If False, then the link is not checked.

    Attributes
    ----------
    Tweedie.link : a link instance
        The link function of the Tweedie instance
    Tweedie.variance : varfunc instance
        ``variance`` is an instance of
        statsmodels.genmod.families.varfuncs.Power
    Tweedie.var_power : float
        The power parameter of the variance function.

    See Also
    --------
    statsmodels.genmod.families.family.Family : Parent class for all links.
    :ref:`links` : Further details on links.

    Notes
    -----
    Loglikelihood function not implemented because of the complexity of
    calculating an infinite series of summations. The variance power can be
    estimated using the ``estimate_tweedie_power`` function that is part of the
    statsmodels.genmod.generalized_linear_model.GLM class.
    """
    links = [L.Log, L.Power]
    variance = V.Power(power=1.5)
    safe_links = [L.Log, L.Power]

    def __init__(self, link=None, var_power=1.0, eql=False, check_link=True):
        if False:
            print('Hello World!')
        self.var_power = var_power
        self.eql = eql
        if eql and (var_power < 1 or var_power > 2):
            raise ValueError('Tweedie: if EQL=True then var_power must fall between 1 and 2')
        if link is None:
            link = L.Log()
        super(Tweedie, self).__init__(link=link, variance=V.Power(power=var_power * 1.0), check_link=check_link)

    def _resid_dev(self, endog, mu):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tweedie deviance residuals\n\n        Parameters\n        ----------\n        endog : ndarray\n            The endogenous response variable.\n        mu : ndarray\n            The inverse of the link function at the linear predicted values.\n\n        Returns\n        -------\n        resid_dev : float\n            Deviance residuals as defined below.\n\n        Notes\n        -----\n        When :math:`p = 1`,\n\n        .. math::\n\n            dev_i = \\mu_i\n\n        when :math:`endog_i = 0` and\n\n        .. math::\n\n            dev_i = endog_i * \\log(endog_i / \\mu_i) + (\\mu_i - endog_i)\n\n        otherwise.\n\n        When :math:`p = 2`,\n\n        .. math::\n\n            dev_i =  (endog_i - \\mu_i) / \\mu_i - \\log(endog_i / \\mu_i)\n\n        For all other p,\n\n        .. math::\n\n            dev_i = endog_i^{2 - p} / ((1 - p) * (2 - p)) -\n                    endog_i * \\mu_i^{1 - p} / (1 - p) + \\mu_i^{2 - p} /\n                    (2 - p)\n\n        The deviance residual is then\n\n        .. math::\n\n            resid\\_dev_i = 2 * dev_i\n        '
        p = self.var_power
        if p == 1:
            dev = np.where(endog == 0, mu, endog * np.log(endog / mu) + (mu - endog))
        elif p == 2:
            endog1 = self._clean(endog)
            dev = (endog - mu) / mu - np.log(endog1 / mu)
        else:
            dev = endog ** (2 - p) / ((1 - p) * (2 - p)) - endog * mu ** (1 - p) / (1 - p) + mu ** (2 - p) / (2 - p)
        return 2 * dev

    def loglike_obs(self, endog, mu, var_weights=1.0, scale=1.0):
        if False:
            return 10
        "\n        The log-likelihood function for each observation in terms of the fitted\n        mean response for the Tweedie distribution.\n\n        Parameters\n        ----------\n        endog : ndarray\n            Usually the endogenous response variable.\n        mu : ndarray\n            Usually but not always the fitted mean response variable.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n        scale : float\n            The scale parameter. The default is 1.\n\n        Returns\n        -------\n        ll_i : float\n            The value of the loglikelihood evaluated at\n            (endog, mu, var_weights, scale) as defined below.\n\n        Notes\n        -----\n        If eql is True, the Extended Quasi-Likelihood is used.  At present,\n        this method returns NaN if eql is False.  When the actual likelihood\n        is implemented, it will be accessible by setting eql to False.\n\n        References\n        ----------\n        R Kaas (2005).  Compound Poisson Distributions and GLM's -- Tweedie's\n        Distribution.\n        https://core.ac.uk/download/pdf/6347266.pdf#page=11\n\n        JA Nelder, D Pregibon (1987).  An extended quasi-likelihood function.\n        Biometrika 74:2, pp 221-232.  https://www.jstor.org/stable/2336136\n        "
        p = self.var_power
        endog = np.atleast_1d(endog)
        if p == 1:
            return Poisson().loglike_obs(endog=endog, mu=mu, var_weights=var_weights, scale=scale)
        elif p == 2:
            return Gamma().loglike_obs(endog=endog, mu=mu, var_weights=var_weights, scale=scale)
        if not self.eql:
            if p < 1 or p > 2:
                return np.nan
            if SP_LT_17:
                return np.nan
            scale = scale / var_weights
            theta = mu ** (1 - p) / (1 - p)
            kappa = mu ** (2 - p) / (2 - p)
            alpha = (2 - p) / (1 - p)
            ll_obs = (endog * theta - kappa) / scale
            idx = endog > 0
            if np.any(idx):
                if not np.isscalar(endog):
                    endog = endog[idx]
                if not np.isscalar(scale):
                    scale = scale[idx]
                x = ((p - 1) * scale / endog) ** alpha
                x /= (2 - p) * scale
                wb = special.wright_bessel(-alpha, 0, x)
                ll_obs[idx] += np.log(1 / endog * wb)
            return ll_obs
        else:
            llf = np.log(2 * np.pi * scale) + p * np.log(endog)
            llf -= np.log(var_weights)
            llf /= -2
            u = endog ** (2 - p) - (2 - p) * endog * mu ** (1 - p) + (1 - p) * mu ** (2 - p)
            u *= var_weights / (scale * (1 - p) * (2 - p))
        return llf - u

    def resid_anscombe(self, endog, mu, var_weights=1.0, scale=1.0):
        if False:
            while True:
                i = 10
        '\n        The Anscombe residuals\n\n        Parameters\n        ----------\n        endog : ndarray\n            The endogenous response variable\n        mu : ndarray\n            The inverse of the link function at the linear predicted values.\n        var_weights : array_like\n            1d array of variance (analytic) weights. The default is 1.\n        scale : float, optional\n            An optional argument to divide the residuals by sqrt(scale).\n            The default is 1.\n\n        Returns\n        -------\n        resid_anscombe : ndarray\n            The Anscombe residuals as defined below.\n\n        Notes\n        -----\n        When :math:`p = 3`, then\n\n        .. math::\n\n            resid\\_anscombe_i = \\log(endog_i / \\mu_i) / \\sqrt{\\mu_i * scale} *\n            \\sqrt(var\\_weights)\n\n        Otherwise,\n\n        .. math::\n\n            c = (3 - p) / 3\n\n        .. math::\n\n            resid\\_anscombe_i = (1 / c) * (endog_i^c - \\mu_i^c) / \\mu_i^{p / 6}\n            / \\sqrt{scale} * \\sqrt(var\\_weights)\n        '
        if self.var_power == 3:
            resid = np.log(endog / mu) / np.sqrt(mu * scale)
        else:
            c = (3.0 - self.var_power) / 3.0
            resid = 1.0 / c * (endog ** c - mu ** c) / mu ** (self.var_power / 6.0) / scale ** 0.5
        resid *= np.sqrt(var_weights)
        return resid