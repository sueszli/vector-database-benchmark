"""
Impulse reponse-related code
"""
import numpy as np
import numpy.linalg as la
import scipy.linalg as L
from statsmodels.tools.decorators import cache_readonly
import statsmodels.tsa.tsatools as tsa
import statsmodels.tsa.vector_ar.plotting as plotting
import statsmodels.tsa.vector_ar.util as util
mat = np.array

class BaseIRAnalysis:
    """
    Base class for plotting and computing IRF-related statistics, want to be
    able to handle known and estimated processes
    """

    def __init__(self, model, P=None, periods=10, order=None, svar=False, vecm=False):
        if False:
            for i in range(10):
                print('nop')
        self.model = model
        self.periods = periods
        (self.neqs, self.lags, self.T) = (model.neqs, model.k_ar, model.nobs)
        self.order = order
        if P is None:
            sigma = model.sigma_u
            P = la.cholesky(sigma)
        self.P = P
        self.svar = svar
        self.irfs = model.ma_rep(periods)
        if svar:
            self.svar_irfs = model.svar_ma_rep(periods, P=P)
        else:
            self.orth_irfs = model.orth_ma_rep(periods, P=P)
        self.cum_effects = self.irfs.cumsum(axis=0)
        if svar:
            self.svar_cum_effects = self.svar_irfs.cumsum(axis=0)
        else:
            self.orth_cum_effects = self.orth_irfs.cumsum(axis=0)
        if not vecm:
            self.lr_effects = model.long_run_effects()
            if svar:
                self.svar_lr_effects = np.dot(model.long_run_effects(), P)
            else:
                self.orth_lr_effects = np.dot(model.long_run_effects(), P)
        if vecm:
            self._A = util.comp_matrix(model.var_rep)
        else:
            self._A = util.comp_matrix(model.coefs)

    def _choose_irfs(self, orth=False, svar=False):
        if False:
            for i in range(10):
                print('nop')
        if orth:
            return self.orth_irfs
        elif svar:
            return self.svar_irfs
        else:
            return self.irfs

    def cov(self, *args, **kwargs):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def cum_effect_cov(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def plot(self, orth=False, *, impulse=None, response=None, signif=0.05, plot_params=None, figsize=(10, 10), subplot_params=None, plot_stderr=True, stderr_type='asym', repl=1000, seed=None, component=None):
        if False:
            print('Hello World!')
        "\n        Plot impulse responses\n\n        Parameters\n        ----------\n        orth : bool, default False\n            Compute orthogonalized impulse responses\n        impulse : {str, int}\n            variable providing the impulse\n        response : {str, int}\n            variable affected by the impulse\n        signif : float (0 < signif < 1)\n            Significance level for error bars, defaults to 95% CI\n        subplot_params : dict\n            To pass to subplot plotting funcions. Example: if fonts are too big,\n            pass {'fontsize' : 8} or some number to your taste.\n        plot_params : dict\n\n        figsize : (float, float), default (10, 10)\n            Figure size (width, height in inches)\n        plot_stderr : bool, default True\n            Plot standard impulse response error bands\n        stderr_type : str\n            'asym': default, computes asymptotic standard errors\n            'mc': monte carlo standard errors (use rpl)\n        repl : int, default 1000\n            Number of replications for Monte Carlo and Sims-Zha standard errors\n        seed : int\n            np.random.seed for Monte Carlo replications\n        component: array or vector of principal component indices\n        "
        periods = self.periods
        model = self.model
        svar = self.svar
        if orth and svar:
            raise ValueError('For SVAR system, set orth=False')
        irfs = self._choose_irfs(orth, svar)
        if orth:
            title = 'Impulse responses (orthogonalized)'
        elif svar:
            title = 'Impulse responses (structural)'
        else:
            title = 'Impulse responses'
        if plot_stderr is False:
            stderr = None
        elif stderr_type not in ['asym', 'mc', 'sz1', 'sz2', 'sz3']:
            raise ValueError("Error type must be either 'asym', 'mc','sz1','sz2', or 'sz3'")
        else:
            if stderr_type == 'asym':
                stderr = self.cov(orth=orth)
            if stderr_type == 'mc':
                stderr = self.errband_mc(orth=orth, svar=svar, repl=repl, signif=signif, seed=seed)
            if stderr_type == 'sz1':
                stderr = self.err_band_sz1(orth=orth, svar=svar, repl=repl, signif=signif, seed=seed, component=component)
            if stderr_type == 'sz2':
                stderr = self.err_band_sz2(orth=orth, svar=svar, repl=repl, signif=signif, seed=seed, component=component)
            if stderr_type == 'sz3':
                stderr = self.err_band_sz3(orth=orth, svar=svar, repl=repl, signif=signif, seed=seed, component=component)
        fig = plotting.irf_grid_plot(irfs, stderr, impulse, response, self.model.names, title, signif=signif, subplot_params=subplot_params, plot_params=plot_params, figsize=figsize, stderr_type=stderr_type)
        return fig

    def plot_cum_effects(self, orth=False, *, impulse=None, response=None, signif=0.05, plot_params=None, figsize=(10, 10), subplot_params=None, plot_stderr=True, stderr_type='asym', repl=1000, seed=None):
        if False:
            while True:
                i = 10
        "\n        Plot cumulative impulse response functions\n\n        Parameters\n        ----------\n        orth : bool, default False\n            Compute orthogonalized impulse responses\n        impulse : {str, int}\n            variable providing the impulse\n        response : {str, int}\n            variable affected by the impulse\n        signif : float (0 < signif < 1)\n            Significance level for error bars, defaults to 95% CI\n        subplot_params : dict\n            To pass to subplot plotting funcions. Example: if fonts are too big,\n            pass {'fontsize' : 8} or some number to your taste.\n        plot_params : dict\n\n        figsize: (float, float), default (10, 10)\n            Figure size (width, height in inches)\n        plot_stderr : bool, default True\n            Plot standard impulse response error bands\n        stderr_type : str\n            'asym': default, computes asymptotic standard errors\n            'mc': monte carlo standard errors (use rpl)\n        repl : int, default 1000\n            Number of replications for monte carlo standard errors\n        seed : int\n            np.random.seed for Monte Carlo replications\n        "
        if orth:
            title = 'Cumulative responses responses (orthogonalized)'
            cum_effects = self.orth_cum_effects
            lr_effects = self.orth_lr_effects
        else:
            title = 'Cumulative responses'
            cum_effects = self.cum_effects
            lr_effects = self.lr_effects
        if stderr_type not in ['asym', 'mc']:
            raise ValueError("`stderr_type` must be one of 'asym', 'mc'")
        else:
            if stderr_type == 'asym':
                stderr = self.cum_effect_cov(orth=orth)
            if stderr_type == 'mc':
                stderr = self.cum_errband_mc(orth=orth, repl=repl, signif=signif, seed=seed)
        if not plot_stderr:
            stderr = None
        fig = plotting.irf_grid_plot(cum_effects, stderr, impulse, response, self.model.names, title, signif=signif, hlines=lr_effects, subplot_params=subplot_params, plot_params=plot_params, figsize=figsize, stderr_type=stderr_type)
        return fig

class IRAnalysis(BaseIRAnalysis):
    """
    Impulse response analysis class. Computes impulse responses, asymptotic
    standard errors, and produces relevant plots

    Parameters
    ----------
    model : VAR instance

    Notes
    -----
    Using Lütkepohl (2005) notation
    """

    def __init__(self, model, P=None, periods=10, order=None, svar=False, vecm=False):
        if False:
            print('Hello World!')
        BaseIRAnalysis.__init__(self, model, P=P, periods=periods, order=order, svar=svar, vecm=vecm)
        if vecm:
            self.cov_a = model.cov_var_repr
        else:
            self.cov_a = model._cov_alpha
        self.cov_sig = model._cov_sigma
        self._g_memo = {}

    def cov(self, orth=False):
        if False:
            i = 10
            return i + 15
        '\n        Compute asymptotic standard errors for impulse response coefficients\n\n        Notes\n        -----\n        Lütkepohl eq 3.7.5\n\n        Returns\n        -------\n        '
        if orth:
            return self._orth_cov()
        covs = self._empty_covm(self.periods + 1)
        covs[0] = np.zeros((self.neqs ** 2, self.neqs ** 2))
        for i in range(1, self.periods + 1):
            Gi = self.G[i - 1]
            covs[i] = Gi @ self.cov_a @ Gi.T
        return covs

    def errband_mc(self, orth=False, svar=False, repl=1000, signif=0.05, seed=None, burn=100):
        if False:
            for i in range(10):
                print('nop')
        '\n        IRF Monte Carlo integrated error bands\n        '
        model = self.model
        periods = self.periods
        if svar:
            return model.sirf_errband_mc(orth=orth, repl=repl, steps=periods, signif=signif, seed=seed, burn=burn, cum=False)
        else:
            return model.irf_errband_mc(orth=orth, repl=repl, steps=periods, signif=signif, seed=seed, burn=burn, cum=False)

    def err_band_sz1(self, orth=False, svar=False, repl=1000, signif=0.05, seed=None, burn=100, component=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        IRF Sims-Zha error band method 1. Assumes symmetric error bands around\n        mean.\n\n        Parameters\n        ----------\n        orth : bool, default False\n            Compute orthogonalized impulse responses\n        repl : int, default 1000\n            Number of MC replications\n        signif : float (0 < signif < 1)\n            Significance level for error bars, defaults to 95% CI\n        seed : int, default None\n            np.random seed\n        burn : int, default 100\n            Number of initial simulated obs to discard\n        component : neqs x neqs array, default to largest for each\n            Index of column of eigenvector/value to use for each error band\n            Note: period of impulse (t=0) is not included when computing\n            principle component\n\n        References\n        ----------\n        Sims, Christopher A., and Tao Zha. 1999. "Error Bands for Impulse\n        Response". Econometrica 67: 1113-1155.\n        '
        model = self.model
        periods = self.periods
        irfs = self._choose_irfs(orth, svar)
        neqs = self.neqs
        irf_resim = model.irf_resim(orth=orth, repl=repl, steps=periods, seed=seed, burn=burn)
        q = util.norm_signif_level(signif)
        (W, eigva, k) = self._eigval_decomp_SZ(irf_resim)
        if component is not None:
            if np.shape(component) != (neqs, neqs):
                raise ValueError('Component array must be ' + str(neqs) + ' x ' + str(neqs))
            if np.argmax(component) >= neqs * periods:
                raise ValueError('Atleast one of the components does not exist')
            else:
                k = component
        lower = np.copy(irfs)
        upper = np.copy(irfs)
        for i in range(neqs):
            for j in range(neqs):
                lower[1:, i, j] = irfs[1:, i, j] + W[i, j, :, k[i, j]] * q * np.sqrt(eigva[i, j, k[i, j]])
                upper[1:, i, j] = irfs[1:, i, j] - W[i, j, :, k[i, j]] * q * np.sqrt(eigva[i, j, k[i, j]])
        return (lower, upper)

    def err_band_sz2(self, orth=False, svar=False, repl=1000, signif=0.05, seed=None, burn=100, component=None):
        if False:
            i = 10
            return i + 15
        '\n        IRF Sims-Zha error band method 2.\n\n        This method Does not assume symmetric error bands around mean.\n\n        Parameters\n        ----------\n        orth : bool, default False\n            Compute orthogonalized impulse responses\n        repl : int, default 1000\n            Number of MC replications\n        signif : float (0 < signif < 1)\n            Significance level for error bars, defaults to 95% CI\n        seed : int, default None\n            np.random seed\n        burn : int, default 100\n            Number of initial simulated obs to discard\n        component : neqs x neqs array, default to largest for each\n            Index of column of eigenvector/value to use for each error band\n            Note: period of impulse (t=0) is not included when computing\n            principle component\n\n        References\n        ----------\n        Sims, Christopher A., and Tao Zha. 1999. "Error Bands for Impulse\n        Response". Econometrica 67: 1113-1155.\n        '
        model = self.model
        periods = self.periods
        irfs = self._choose_irfs(orth, svar)
        neqs = self.neqs
        irf_resim = model.irf_resim(orth=orth, repl=repl, steps=periods, seed=seed, burn=100)
        (W, eigva, k) = self._eigval_decomp_SZ(irf_resim)
        if component is not None:
            if np.shape(component) != (neqs, neqs):
                raise ValueError('Component array must be ' + str(neqs) + ' x ' + str(neqs))
            if np.argmax(component) >= neqs * periods:
                raise ValueError('Atleast one of the components does not exist')
            else:
                k = component
        gamma = np.zeros((repl, periods + 1, neqs, neqs))
        for p in range(repl):
            for i in range(neqs):
                for j in range(neqs):
                    gamma[p, 1:, i, j] = W[i, j, k[i, j], :] * irf_resim[p, 1:, i, j]
        gamma_sort = np.sort(gamma, axis=0)
        indx = (round(signif / 2 * repl) - 1, round((1 - signif / 2) * repl) - 1)
        lower = np.copy(irfs)
        upper = np.copy(irfs)
        for i in range(neqs):
            for j in range(neqs):
                lower[:, i, j] = irfs[:, i, j] + gamma_sort[indx[0], :, i, j]
                upper[:, i, j] = irfs[:, i, j] + gamma_sort[indx[1], :, i, j]
        return (lower, upper)

    def err_band_sz3(self, orth=False, svar=False, repl=1000, signif=0.05, seed=None, burn=100, component=None):
        if False:
            i = 10
            return i + 15
        '\n        IRF Sims-Zha error band method 3. Does not assume symmetric error bands around mean.\n\n        Parameters\n        ----------\n        orth : bool, default False\n            Compute orthogonalized impulse responses\n        repl : int, default 1000\n            Number of MC replications\n        signif : float (0 < signif < 1)\n            Significance level for error bars, defaults to 95% CI\n        seed : int, default None\n            np.random seed\n        burn : int, default 100\n            Number of initial simulated obs to discard\n        component : vector length neqs, default to largest for each\n            Index of column of eigenvector/value to use for each error band\n            Note: period of impulse (t=0) is not included when computing\n            principle component\n\n        References\n        ----------\n        Sims, Christopher A., and Tao Zha. 1999. "Error Bands for Impulse\n        Response". Econometrica 67: 1113-1155.\n        '
        model = self.model
        periods = self.periods
        irfs = self._choose_irfs(orth, svar)
        neqs = self.neqs
        irf_resim = model.irf_resim(orth=orth, repl=repl, steps=periods, seed=seed, burn=100)
        stack = np.zeros((neqs, repl, periods * neqs))
        for p in range(repl):
            for i in range(neqs):
                stack[i, p, :] = np.ravel(irf_resim[p, 1:, :, i].T)
        stack_cov = np.zeros((neqs, periods * neqs, periods * neqs))
        W = np.zeros((neqs, periods * neqs, periods * neqs))
        eigva = np.zeros((neqs, periods * neqs))
        k = np.zeros(neqs, dtype=int)
        if component is not None:
            if np.size(component) != neqs:
                raise ValueError('Component array must be of length ' + str(neqs))
            if np.argmax(component) >= neqs * periods:
                raise ValueError('Atleast one of the components does not exist')
            else:
                k = component
        for i in range(neqs):
            stack_cov[i] = np.cov(stack[i], rowvar=0)
            (W[i], eigva[i], k[i]) = util.eigval_decomp(stack_cov[i])
        gamma = np.zeros((repl, periods + 1, neqs, neqs))
        for p in range(repl):
            c = 0
            for j in range(neqs):
                for i in range(neqs):
                    gamma[p, 1:, i, j] = W[j, k[j], i * periods:(i + 1) * periods] * irf_resim[p, 1:, i, j]
                    if i == neqs - 1:
                        gamma[p, 1:, i, j] = W[j, k[j], i * periods:] * irf_resim[p, 1:, i, j]
        gamma_sort = np.sort(gamma, axis=0)
        indx = (round(signif / 2 * repl) - 1, round((1 - signif / 2) * repl) - 1)
        lower = np.copy(irfs)
        upper = np.copy(irfs)
        for i in range(neqs):
            for j in range(neqs):
                lower[:, i, j] = irfs[:, i, j] + gamma_sort[indx[0], :, i, j]
                upper[:, i, j] = irfs[:, i, j] + gamma_sort[indx[1], :, i, j]
        return (lower, upper)

    def _eigval_decomp_SZ(self, irf_resim):
        if False:
            print('Hello World!')
        '\n        Returns\n        -------\n        W: array of eigenvectors\n        eigva: list of eigenvalues\n        k: matrix indicating column # of largest eigenvalue for each c_i,j\n        '
        neqs = self.neqs
        periods = self.periods
        cov_hold = np.zeros((neqs, neqs, periods, periods))
        for i in range(neqs):
            for j in range(neqs):
                cov_hold[i, j, :, :] = np.cov(irf_resim[:, 1:, i, j], rowvar=0)
        W = np.zeros((neqs, neqs, periods, periods))
        eigva = np.zeros((neqs, neqs, periods, 1))
        k = np.zeros((neqs, neqs), dtype=int)
        for i in range(neqs):
            for j in range(neqs):
                (W[i, j, :, :], eigva[i, j, :, 0], k[i, j]) = util.eigval_decomp(cov_hold[i, j, :, :])
        return (W, eigva, k)

    @cache_readonly
    def G(self):
        if False:
            while True:
                i = 10
        K = self.neqs

        def _make_g(i):
            if False:
                i = 10
                return i + 15
            G = 0.0
            for m in range(i):
                idx = i - 1 - m
                if idx in self._g_memo:
                    apow = self._g_memo[idx]
                else:
                    apow = la.matrix_power(self._A.T, idx)
                    apow = apow[:K]
                    self._g_memo[idx] = apow
                piece = np.kron(apow, self.irfs[m])
                G = G + piece
            return G
        return [_make_g(i) for i in range(1, self.periods + 1)]

    def _orth_cov(self):
        if False:
            print('Hello World!')
        Ik = np.eye(self.neqs)
        PIk = np.kron(self.P.T, Ik)
        H = self.H
        covs = self._empty_covm(self.periods + 1)
        for i in range(self.periods + 1):
            if i == 0:
                apiece = 0
            else:
                Ci = np.dot(PIk, self.G[i - 1])
                apiece = Ci @ self.cov_a @ Ci.T
            Cibar = np.dot(np.kron(Ik, self.irfs[i]), H)
            bpiece = Cibar @ self.cov_sig @ Cibar.T / self.T
            covs[i] = apiece + bpiece
        return covs

    def cum_effect_cov(self, orth=False):
        if False:
            return 10
        '\n        Compute asymptotic standard errors for cumulative impulse response\n        coefficients\n\n        Parameters\n        ----------\n        orth : bool\n\n        Notes\n        -----\n        eq. 3.7.7 (non-orth), 3.7.10 (orth)\n\n        Returns\n        -------\n        '
        Ik = np.eye(self.neqs)
        PIk = np.kron(self.P.T, Ik)
        F = 0.0
        covs = self._empty_covm(self.periods + 1)
        for i in range(self.periods + 1):
            if i > 0:
                F = F + self.G[i - 1]
            if orth:
                if i == 0:
                    apiece = 0
                else:
                    Bn = np.dot(PIk, F)
                    apiece = Bn @ self.cov_a @ Bn.T
                Bnbar = np.dot(np.kron(Ik, self.cum_effects[i]), self.H)
                bpiece = Bnbar @ self.cov_sig @ Bnbar.T / self.T
                covs[i] = apiece + bpiece
            else:
                if i == 0:
                    covs[i] = np.zeros((self.neqs ** 2, self.neqs ** 2))
                    continue
                covs[i] = F @ self.cov_a @ F.T
        return covs

    def cum_errband_mc(self, orth=False, repl=1000, signif=0.05, seed=None, burn=100):
        if False:
            while True:
                i = 10
        '\n        IRF Monte Carlo integrated error bands of cumulative effect\n        '
        model = self.model
        periods = self.periods
        return model.irf_errband_mc(orth=orth, repl=repl, steps=periods, signif=signif, seed=seed, burn=burn, cum=True)

    def lr_effect_cov(self, orth=False):
        if False:
            while True:
                i = 10
        '\n        Returns\n        -------\n        '
        lre = self.lr_effects
        Finfty = np.kron(np.tile(lre.T, self.lags), lre)
        Ik = np.eye(self.neqs)
        if orth:
            Binf = np.dot(np.kron(self.P.T, np.eye(self.neqs)), Finfty)
            Binfbar = np.dot(np.kron(Ik, lre), self.H)
            return Binf @ self.cov_a @ Binf.T + Binfbar @ self.cov_sig @ Binfbar.T
        else:
            return Finfty @ self.cov_a @ Finfty.T

    def stderr(self, orth=False):
        if False:
            i = 10
            return i + 15
        return np.array([tsa.unvec(np.sqrt(np.diag(c))) for c in self.cov(orth=orth)])

    def cum_effect_stderr(self, orth=False):
        if False:
            while True:
                i = 10
        return np.array([tsa.unvec(np.sqrt(np.diag(c))) for c in self.cum_effect_cov(orth=orth)])

    def lr_effect_stderr(self, orth=False):
        if False:
            while True:
                i = 10
        cov = self.lr_effect_cov(orth=orth)
        return tsa.unvec(np.sqrt(np.diag(cov)))

    def _empty_covm(self, periods):
        if False:
            while True:
                i = 10
        return np.zeros((periods, self.neqs ** 2, self.neqs ** 2), dtype=float)

    @cache_readonly
    def H(self):
        if False:
            for i in range(10):
                print('nop')
        k = self.neqs
        Lk = tsa.elimination_matrix(k)
        Kkk = tsa.commutation_matrix(k, k)
        Ik = np.eye(k)
        B = Lk @ (np.kron(Ik, self.P) @ Kkk + np.kron(self.P, Ik)) @ Lk.T
        return np.dot(Lk.T, L.inv(B))

    def fevd_table(self):
        if False:
            return 10
        raise NotImplementedError