"""
Least Angle Regression algorithm. See the documentation on the
Generalized Linear Model for a complete discussion.
"""
import sys
import warnings
from math import log
from numbers import Integral, Real
import numpy as np
from scipy import interpolate, linalg
from scipy.linalg.lapack import get_lapack_funcs
from ..base import MultiOutputMixin, RegressorMixin, _fit_context
from ..exceptions import ConvergenceWarning
from ..model_selection import check_cv
from ..utils import Bunch, arrayfuncs, as_float_array, check_random_state
from ..utils._metadata_requests import MetadataRouter, MethodMapping, _raise_for_params, _routing_enabled, process_routing
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.parallel import Parallel, delayed
from ._base import LinearModel, LinearRegression, _deprecate_normalize, _preprocess_data
SOLVE_TRIANGULAR_ARGS = {'check_finite': False}

@validate_params({'X': [np.ndarray, None], 'y': [np.ndarray, None], 'Xy': [np.ndarray, None], 'Gram': [StrOptions({'auto'}), 'boolean', np.ndarray, None], 'max_iter': [Interval(Integral, 0, None, closed='left')], 'alpha_min': [Interval(Real, 0, None, closed='left')], 'method': [StrOptions({'lar', 'lasso'})], 'copy_X': ['boolean'], 'eps': [Interval(Real, 0, None, closed='neither'), None], 'copy_Gram': ['boolean'], 'verbose': ['verbose'], 'return_path': ['boolean'], 'return_n_iter': ['boolean'], 'positive': ['boolean']}, prefer_skip_nested_validation=True)
def lars_path(X, y, Xy=None, *, Gram=None, max_iter=500, alpha_min=0, method='lar', copy_X=True, eps=np.finfo(float).eps, copy_Gram=True, verbose=0, return_path=True, return_n_iter=False, positive=False):
    if False:
        return 10
    'Compute Least Angle Regression or Lasso path using the LARS algorithm [1].\n\n    The optimization objective for the case method=\'lasso\' is::\n\n    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1\n\n    in the case of method=\'lar\', the objective function is only known in\n    the form of an implicit equation (see discussion in [1]).\n\n    Read more in the :ref:`User Guide <least_angle_regression>`.\n\n    Parameters\n    ----------\n    X : None or ndarray of shape (n_samples, n_features)\n        Input data. Note that if X is `None` then the Gram matrix must be\n        specified, i.e., cannot be `None` or `False`.\n\n    y : None or ndarray of shape (n_samples,)\n        Input targets.\n\n    Xy : array-like of shape (n_features,) or (n_features, n_targets),             default=None\n        `Xy = X.T @ y` that can be precomputed. It is useful\n        only when the Gram matrix is precomputed.\n\n    Gram : None, \'auto\', bool, ndarray of shape (n_features, n_features),             default=None\n        Precomputed Gram matrix `X.T @ X`, if `\'auto\'`, the Gram\n        matrix is precomputed from the given X, if there are more samples\n        than features.\n\n    max_iter : int, default=500\n        Maximum number of iterations to perform, set to infinity for no limit.\n\n    alpha_min : float, default=0\n        Minimum correlation along the path. It corresponds to the\n        regularization parameter `alpha` in the Lasso.\n\n    method : {\'lar\', \'lasso\'}, default=\'lar\'\n        Specifies the returned model. Select `\'lar\'` for Least Angle\n        Regression, `\'lasso\'` for the Lasso.\n\n    copy_X : bool, default=True\n        If `False`, `X` is overwritten.\n\n    eps : float, default=np.finfo(float).eps\n        The machine-precision regularization in the computation of the\n        Cholesky diagonal factors. Increase this for very ill-conditioned\n        systems. Unlike the `tol` parameter in some iterative\n        optimization-based algorithms, this parameter does not control\n        the tolerance of the optimization.\n\n    copy_Gram : bool, default=True\n        If `False`, `Gram` is overwritten.\n\n    verbose : int, default=0\n        Controls output verbosity.\n\n    return_path : bool, default=True\n        If `True`, returns the entire path, else returns only the\n        last point of the path.\n\n    return_n_iter : bool, default=False\n        Whether to return the number of iterations.\n\n    positive : bool, default=False\n        Restrict coefficients to be >= 0.\n        This option is only allowed with method \'lasso\'. Note that the model\n        coefficients will not converge to the ordinary-least-squares solution\n        for small values of alpha. Only coefficients up to the smallest alpha\n        value (`alphas_[alphas_ > 0.].min()` when fit_path=True) reached by\n        the stepwise Lars-Lasso algorithm are typically in congruence with the\n        solution of the coordinate descent `lasso_path` function.\n\n    Returns\n    -------\n    alphas : ndarray of shape (n_alphas + 1,)\n        Maximum of covariances (in absolute value) at each iteration.\n        `n_alphas` is either `max_iter`, `n_features`, or the\n        number of nodes in the path with `alpha >= alpha_min`, whichever\n        is smaller.\n\n    active : ndarray of shape (n_alphas,)\n        Indices of active variables at the end of the path.\n\n    coefs : ndarray of shape (n_features, n_alphas + 1)\n        Coefficients along the path.\n\n    n_iter : int\n        Number of iterations run. Returned only if `return_n_iter` is set\n        to True.\n\n    See Also\n    --------\n    lars_path_gram : Compute LARS path in the sufficient stats mode.\n    lasso_path : Compute Lasso path with coordinate descent.\n    LassoLars : Lasso model fit with Least Angle Regression a.k.a. Lars.\n    Lars : Least Angle Regression model a.k.a. LAR.\n    LassoLarsCV : Cross-validated Lasso, using the LARS algorithm.\n    LarsCV : Cross-validated Least Angle Regression model.\n    sklearn.decomposition.sparse_encode : Sparse coding.\n\n    References\n    ----------\n    .. [1] "Least Angle Regression", Efron et al.\n           http://statweb.stanford.edu/~tibs/ftp/lars.pdf\n\n    .. [2] `Wikipedia entry on the Least-angle regression\n           <https://en.wikipedia.org/wiki/Least-angle_regression>`_\n\n    .. [3] `Wikipedia entry on the Lasso\n           <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_\n    '
    if X is None and Gram is not None:
        raise ValueError('X cannot be None if Gram is not NoneUse lars_path_gram to avoid passing X and y.')
    return _lars_path_solver(X=X, y=y, Xy=Xy, Gram=Gram, n_samples=None, max_iter=max_iter, alpha_min=alpha_min, method=method, copy_X=copy_X, eps=eps, copy_Gram=copy_Gram, verbose=verbose, return_path=return_path, return_n_iter=return_n_iter, positive=positive)

@validate_params({'Xy': [np.ndarray], 'Gram': [np.ndarray], 'n_samples': [Interval(Integral, 0, None, closed='left')], 'max_iter': [Interval(Integral, 0, None, closed='left')], 'alpha_min': [Interval(Real, 0, None, closed='left')], 'method': [StrOptions({'lar', 'lasso'})], 'copy_X': ['boolean'], 'eps': [Interval(Real, 0, None, closed='neither'), None], 'copy_Gram': ['boolean'], 'verbose': ['verbose'], 'return_path': ['boolean'], 'return_n_iter': ['boolean'], 'positive': ['boolean']}, prefer_skip_nested_validation=True)
def lars_path_gram(Xy, Gram, *, n_samples, max_iter=500, alpha_min=0, method='lar', copy_X=True, eps=np.finfo(float).eps, copy_Gram=True, verbose=0, return_path=True, return_n_iter=False, positive=False):
    if False:
        i = 10
        return i + 15
    'The lars_path in the sufficient stats mode [1].\n\n    The optimization objective for the case method=\'lasso\' is::\n\n    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1\n\n    in the case of method=\'lars\', the objective function is only known in\n    the form of an implicit equation (see discussion in [1])\n\n    Read more in the :ref:`User Guide <least_angle_regression>`.\n\n    Parameters\n    ----------\n    Xy : ndarray of shape (n_features,) or (n_features, n_targets)\n        `Xy = X.T @ y`.\n\n    Gram : ndarray of shape (n_features, n_features)\n        `Gram = X.T @ X`.\n\n    n_samples : int\n        Equivalent size of sample.\n\n    max_iter : int, default=500\n        Maximum number of iterations to perform, set to infinity for no limit.\n\n    alpha_min : float, default=0\n        Minimum correlation along the path. It corresponds to the\n        regularization parameter alpha parameter in the Lasso.\n\n    method : {\'lar\', \'lasso\'}, default=\'lar\'\n        Specifies the returned model. Select `\'lar\'` for Least Angle\n        Regression, ``\'lasso\'`` for the Lasso.\n\n    copy_X : bool, default=True\n        If `False`, `X` is overwritten.\n\n    eps : float, default=np.finfo(float).eps\n        The machine-precision regularization in the computation of the\n        Cholesky diagonal factors. Increase this for very ill-conditioned\n        systems. Unlike the `tol` parameter in some iterative\n        optimization-based algorithms, this parameter does not control\n        the tolerance of the optimization.\n\n    copy_Gram : bool, default=True\n        If `False`, `Gram` is overwritten.\n\n    verbose : int, default=0\n        Controls output verbosity.\n\n    return_path : bool, default=True\n        If `return_path==True` returns the entire path, else returns only the\n        last point of the path.\n\n    return_n_iter : bool, default=False\n        Whether to return the number of iterations.\n\n    positive : bool, default=False\n        Restrict coefficients to be >= 0.\n        This option is only allowed with method \'lasso\'. Note that the model\n        coefficients will not converge to the ordinary-least-squares solution\n        for small values of alpha. Only coefficients up to the smallest alpha\n        value (`alphas_[alphas_ > 0.].min()` when `fit_path=True`) reached by\n        the stepwise Lars-Lasso algorithm are typically in congruence with the\n        solution of the coordinate descent lasso_path function.\n\n    Returns\n    -------\n    alphas : ndarray of shape (n_alphas + 1,)\n        Maximum of covariances (in absolute value) at each iteration.\n        `n_alphas` is either `max_iter`, `n_features` or the\n        number of nodes in the path with `alpha >= alpha_min`, whichever\n        is smaller.\n\n    active : ndarray of shape (n_alphas,)\n        Indices of active variables at the end of the path.\n\n    coefs : ndarray of shape (n_features, n_alphas + 1)\n        Coefficients along the path.\n\n    n_iter : int\n        Number of iterations run. Returned only if `return_n_iter` is set\n        to True.\n\n    See Also\n    --------\n    lars_path_gram : Compute LARS path.\n    lasso_path : Compute Lasso path with coordinate descent.\n    LassoLars : Lasso model fit with Least Angle Regression a.k.a. Lars.\n    Lars : Least Angle Regression model a.k.a. LAR.\n    LassoLarsCV : Cross-validated Lasso, using the LARS algorithm.\n    LarsCV : Cross-validated Least Angle Regression model.\n    sklearn.decomposition.sparse_encode : Sparse coding.\n\n    References\n    ----------\n    .. [1] "Least Angle Regression", Efron et al.\n           http://statweb.stanford.edu/~tibs/ftp/lars.pdf\n\n    .. [2] `Wikipedia entry on the Least-angle regression\n           <https://en.wikipedia.org/wiki/Least-angle_regression>`_\n\n    .. [3] `Wikipedia entry on the Lasso\n           <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_\n    '
    return _lars_path_solver(X=None, y=None, Xy=Xy, Gram=Gram, n_samples=n_samples, max_iter=max_iter, alpha_min=alpha_min, method=method, copy_X=copy_X, eps=eps, copy_Gram=copy_Gram, verbose=verbose, return_path=return_path, return_n_iter=return_n_iter, positive=positive)

def _lars_path_solver(X, y, Xy=None, Gram=None, n_samples=None, max_iter=500, alpha_min=0, method='lar', copy_X=True, eps=np.finfo(float).eps, copy_Gram=True, verbose=0, return_path=True, return_n_iter=False, positive=False):
    if False:
        print('Hello World!')
    'Compute Least Angle Regression or Lasso path using LARS algorithm [1]\n\n    The optimization objective for the case method=\'lasso\' is::\n\n    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1\n\n    in the case of method=\'lars\', the objective function is only known in\n    the form of an implicit equation (see discussion in [1])\n\n    Read more in the :ref:`User Guide <least_angle_regression>`.\n\n    Parameters\n    ----------\n    X : None or ndarray of shape (n_samples, n_features)\n        Input data. Note that if X is None then Gram must be specified,\n        i.e., cannot be None or False.\n\n    y : None or ndarray of shape (n_samples,)\n        Input targets.\n\n    Xy : array-like of shape (n_features,) or (n_features, n_targets),             default=None\n        `Xy = np.dot(X.T, y)` that can be precomputed. It is useful\n        only when the Gram matrix is precomputed.\n\n    Gram : None, \'auto\' or array-like of shape (n_features, n_features),             default=None\n        Precomputed Gram matrix `(X\' * X)`, if ``\'auto\'``, the Gram\n        matrix is precomputed from the given X, if there are more samples\n        than features.\n\n    n_samples : int or float, default=None\n        Equivalent size of sample. If `None`, it will be `n_samples`.\n\n    max_iter : int, default=500\n        Maximum number of iterations to perform, set to infinity for no limit.\n\n    alpha_min : float, default=0\n        Minimum correlation along the path. It corresponds to the\n        regularization parameter alpha parameter in the Lasso.\n\n    method : {\'lar\', \'lasso\'}, default=\'lar\'\n        Specifies the returned model. Select ``\'lar\'`` for Least Angle\n        Regression, ``\'lasso\'`` for the Lasso.\n\n    copy_X : bool, default=True\n        If ``False``, ``X`` is overwritten.\n\n    eps : float, default=np.finfo(float).eps\n        The machine-precision regularization in the computation of the\n        Cholesky diagonal factors. Increase this for very ill-conditioned\n        systems. Unlike the ``tol`` parameter in some iterative\n        optimization-based algorithms, this parameter does not control\n        the tolerance of the optimization.\n\n    copy_Gram : bool, default=True\n        If ``False``, ``Gram`` is overwritten.\n\n    verbose : int, default=0\n        Controls output verbosity.\n\n    return_path : bool, default=True\n        If ``return_path==True`` returns the entire path, else returns only the\n        last point of the path.\n\n    return_n_iter : bool, default=False\n        Whether to return the number of iterations.\n\n    positive : bool, default=False\n        Restrict coefficients to be >= 0.\n        This option is only allowed with method \'lasso\'. Note that the model\n        coefficients will not converge to the ordinary-least-squares solution\n        for small values of alpha. Only coefficients up to the smallest alpha\n        value (``alphas_[alphas_ > 0.].min()`` when fit_path=True) reached by\n        the stepwise Lars-Lasso algorithm are typically in congruence with the\n        solution of the coordinate descent lasso_path function.\n\n    Returns\n    -------\n    alphas : array-like of shape (n_alphas + 1,)\n        Maximum of covariances (in absolute value) at each iteration.\n        ``n_alphas`` is either ``max_iter``, ``n_features`` or the\n        number of nodes in the path with ``alpha >= alpha_min``, whichever\n        is smaller.\n\n    active : array-like of shape (n_alphas,)\n        Indices of active variables at the end of the path.\n\n    coefs : array-like of shape (n_features, n_alphas + 1)\n        Coefficients along the path\n\n    n_iter : int\n        Number of iterations run. Returned only if return_n_iter is set\n        to True.\n\n    See Also\n    --------\n    lasso_path\n    LassoLars\n    Lars\n    LassoLarsCV\n    LarsCV\n    sklearn.decomposition.sparse_encode\n\n    References\n    ----------\n    .. [1] "Least Angle Regression", Efron et al.\n           http://statweb.stanford.edu/~tibs/ftp/lars.pdf\n\n    .. [2] `Wikipedia entry on the Least-angle regression\n           <https://en.wikipedia.org/wiki/Least-angle_regression>`_\n\n    .. [3] `Wikipedia entry on the Lasso\n           <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_\n\n    '
    if method == 'lar' and positive:
        raise ValueError("Positive constraint not supported for 'lar' coding method.")
    n_samples = n_samples if n_samples is not None else y.size
    if Xy is None:
        Cov = np.dot(X.T, y)
    else:
        Cov = Xy.copy()
    if Gram is None or Gram is False:
        Gram = None
        if X is None:
            raise ValueError('X and Gram cannot both be unspecified.')
    elif isinstance(Gram, str) and Gram == 'auto' or Gram is True:
        if Gram is True or X.shape[0] > X.shape[1]:
            Gram = np.dot(X.T, X)
        else:
            Gram = None
    elif copy_Gram:
        Gram = Gram.copy()
    if Gram is None:
        n_features = X.shape[1]
    else:
        n_features = Cov.shape[0]
        if Gram.shape != (n_features, n_features):
            raise ValueError('The shapes of the inputs Gram and Xy do not match.')
    if copy_X and X is not None and (Gram is None):
        X = X.copy('F')
    max_features = min(max_iter, n_features)
    dtypes = set((a.dtype for a in (X, y, Xy, Gram) if a is not None))
    if len(dtypes) == 1:
        return_dtype = next(iter(dtypes))
    else:
        return_dtype = np.float64
    if return_path:
        coefs = np.zeros((max_features + 1, n_features), dtype=return_dtype)
        alphas = np.zeros(max_features + 1, dtype=return_dtype)
    else:
        (coef, prev_coef) = (np.zeros(n_features, dtype=return_dtype), np.zeros(n_features, dtype=return_dtype))
        (alpha, prev_alpha) = (np.array([0.0], dtype=return_dtype), np.array([0.0], dtype=return_dtype))
    (n_iter, n_active) = (0, 0)
    (active, indices) = (list(), np.arange(n_features))
    sign_active = np.empty(max_features, dtype=np.int8)
    drop = False
    if Gram is None:
        L = np.empty((max_features, max_features), dtype=X.dtype)
        (swap, nrm2) = linalg.get_blas_funcs(('swap', 'nrm2'), (X,))
    else:
        L = np.empty((max_features, max_features), dtype=Gram.dtype)
        (swap, nrm2) = linalg.get_blas_funcs(('swap', 'nrm2'), (Cov,))
    (solve_cholesky,) = get_lapack_funcs(('potrs',), (L,))
    if verbose:
        if verbose > 1:
            print('Step\t\tAdded\t\tDropped\t\tActive set size\t\tC')
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
    tiny32 = np.finfo(np.float32).tiny
    cov_precision = np.finfo(Cov.dtype).precision
    equality_tolerance = np.finfo(np.float32).eps
    if Gram is not None:
        Gram_copy = Gram.copy()
        Cov_copy = Cov.copy()
    while True:
        if Cov.size:
            if positive:
                C_idx = np.argmax(Cov)
            else:
                C_idx = np.argmax(np.abs(Cov))
            C_ = Cov[C_idx]
            if positive:
                C = C_
            else:
                C = np.fabs(C_)
        else:
            C = 0.0
        if return_path:
            alpha = alphas[n_iter, np.newaxis]
            coef = coefs[n_iter]
            prev_alpha = alphas[n_iter - 1, np.newaxis]
            prev_coef = coefs[n_iter - 1]
        alpha[0] = C / n_samples
        if alpha[0] <= alpha_min + equality_tolerance:
            if abs(alpha[0] - alpha_min) > equality_tolerance:
                if n_iter > 0:
                    ss = (prev_alpha[0] - alpha_min) / (prev_alpha[0] - alpha[0])
                    coef[:] = prev_coef + ss * (coef - prev_coef)
                alpha[0] = alpha_min
            if return_path:
                coefs[n_iter] = coef
            break
        if n_iter >= max_iter or n_active >= n_features:
            break
        if not drop:
            if positive:
                sign_active[n_active] = np.ones_like(C_)
            else:
                sign_active[n_active] = np.sign(C_)
            (m, n) = (n_active, C_idx + n_active)
            (Cov[C_idx], Cov[0]) = swap(Cov[C_idx], Cov[0])
            (indices[n], indices[m]) = (indices[m], indices[n])
            Cov_not_shortened = Cov
            Cov = Cov[1:]
            if Gram is None:
                (X.T[n], X.T[m]) = swap(X.T[n], X.T[m])
                c = nrm2(X.T[n_active]) ** 2
                L[n_active, :n_active] = np.dot(X.T[n_active], X.T[:n_active].T)
            else:
                (Gram[m], Gram[n]) = swap(Gram[m], Gram[n])
                (Gram[:, m], Gram[:, n]) = swap(Gram[:, m], Gram[:, n])
                c = Gram[n_active, n_active]
                L[n_active, :n_active] = Gram[n_active, :n_active]
            if n_active:
                linalg.solve_triangular(L[:n_active, :n_active], L[n_active, :n_active], trans=0, lower=1, overwrite_b=True, **SOLVE_TRIANGULAR_ARGS)
            v = np.dot(L[n_active, :n_active], L[n_active, :n_active])
            diag = max(np.sqrt(np.abs(c - v)), eps)
            L[n_active, n_active] = diag
            if diag < 1e-07:
                warnings.warn('Regressors in active set degenerate. Dropping a regressor, after %i iterations, i.e. alpha=%.3e, with an active set of %i regressors, and the smallest cholesky pivot element being %.3e. Reduce max_iter or increase eps parameters.' % (n_iter, alpha.item(), n_active, diag), ConvergenceWarning)
                Cov = Cov_not_shortened
                Cov[0] = 0
                (Cov[C_idx], Cov[0]) = swap(Cov[C_idx], Cov[0])
                continue
            active.append(indices[n_active])
            n_active += 1
            if verbose > 1:
                print('%s\t\t%s\t\t%s\t\t%s\t\t%s' % (n_iter, active[-1], '', n_active, C))
        if method == 'lasso' and n_iter > 0 and (prev_alpha[0] < alpha[0]):
            warnings.warn('Early stopping the lars path, as the residues are small and the current value of alpha is no longer well controlled. %i iterations, alpha=%.3e, previous alpha=%.3e, with an active set of %i regressors.' % (n_iter, alpha.item(), prev_alpha.item(), n_active), ConvergenceWarning)
            break
        (least_squares, _) = solve_cholesky(L[:n_active, :n_active], sign_active[:n_active], lower=True)
        if least_squares.size == 1 and least_squares == 0:
            least_squares[...] = 1
            AA = 1.0
        else:
            AA = 1.0 / np.sqrt(np.sum(least_squares * sign_active[:n_active]))
            if not np.isfinite(AA):
                i = 0
                L_ = L[:n_active, :n_active].copy()
                while not np.isfinite(AA):
                    L_.flat[::n_active + 1] += 2 ** i * eps
                    (least_squares, _) = solve_cholesky(L_, sign_active[:n_active], lower=True)
                    tmp = max(np.sum(least_squares * sign_active[:n_active]), eps)
                    AA = 1.0 / np.sqrt(tmp)
                    i += 1
            least_squares *= AA
        if Gram is None:
            eq_dir = np.dot(X.T[:n_active].T, least_squares)
            corr_eq_dir = np.dot(X.T[n_active:], eq_dir)
        else:
            corr_eq_dir = np.dot(Gram[:n_active, n_active:].T, least_squares)
        np.around(corr_eq_dir, decimals=cov_precision, out=corr_eq_dir)
        g1 = arrayfuncs.min_pos((C - Cov) / (AA - corr_eq_dir + tiny32))
        if positive:
            gamma_ = min(g1, C / AA)
        else:
            g2 = arrayfuncs.min_pos((C + Cov) / (AA + corr_eq_dir + tiny32))
            gamma_ = min(g1, g2, C / AA)
        drop = False
        z = -coef[active] / (least_squares + tiny32)
        z_pos = arrayfuncs.min_pos(z)
        if z_pos < gamma_:
            idx = np.where(z == z_pos)[0][::-1]
            sign_active[idx] = -sign_active[idx]
            if method == 'lasso':
                gamma_ = z_pos
            drop = True
        n_iter += 1
        if return_path:
            if n_iter >= coefs.shape[0]:
                del coef, alpha, prev_alpha, prev_coef
                add_features = 2 * max(1, max_features - n_active)
                coefs = np.resize(coefs, (n_iter + add_features, n_features))
                coefs[-add_features:] = 0
                alphas = np.resize(alphas, n_iter + add_features)
                alphas[-add_features:] = 0
            coef = coefs[n_iter]
            prev_coef = coefs[n_iter - 1]
        else:
            prev_coef = coef
            prev_alpha[0] = alpha[0]
            coef = np.zeros_like(coef)
        coef[active] = prev_coef[active] + gamma_ * least_squares
        Cov -= gamma_ * corr_eq_dir
        if drop and method == 'lasso':
            for ii in idx:
                arrayfuncs.cholesky_delete(L[:n_active, :n_active], ii)
            n_active -= 1
            drop_idx = [active.pop(ii) for ii in idx]
            if Gram is None:
                for ii in idx:
                    for i in range(ii, n_active):
                        (X.T[i], X.T[i + 1]) = swap(X.T[i], X.T[i + 1])
                        (indices[i], indices[i + 1]) = (indices[i + 1], indices[i])
                residual = y - np.dot(X[:, :n_active], coef[active])
                temp = np.dot(X.T[n_active], residual)
                Cov = np.r_[temp, Cov]
            else:
                for ii in idx:
                    for i in range(ii, n_active):
                        (indices[i], indices[i + 1]) = (indices[i + 1], indices[i])
                        (Gram[i], Gram[i + 1]) = swap(Gram[i], Gram[i + 1])
                        (Gram[:, i], Gram[:, i + 1]) = swap(Gram[:, i], Gram[:, i + 1])
                temp = Cov_copy[drop_idx] - np.dot(Gram_copy[drop_idx], coef)
                Cov = np.r_[temp, Cov]
            sign_active = np.delete(sign_active, idx)
            sign_active = np.append(sign_active, 0.0)
            if verbose > 1:
                print('%s\t\t%s\t\t%s\t\t%s\t\t%s' % (n_iter, '', drop_idx, n_active, abs(temp)))
    if return_path:
        alphas = alphas[:n_iter + 1]
        coefs = coefs[:n_iter + 1]
        if return_n_iter:
            return (alphas, active, coefs.T, n_iter)
        else:
            return (alphas, active, coefs.T)
    elif return_n_iter:
        return (alpha, active, coef, n_iter)
    else:
        return (alpha, active, coef)

class Lars(MultiOutputMixin, RegressorMixin, LinearModel):
    """Least Angle Regression model a.k.a. LAR.

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    verbose : bool or int, default=False
        Sets the verbosity amount.

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

        .. versionchanged:: 1.2
           default changed from True to False in 1.2.

        .. deprecated:: 1.2
            ``normalize`` was deprecated in version 1.2 and will be removed in 1.4.

    precompute : bool, 'auto' or array-like , default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    n_nonzero_coefs : int, default=500
        Target number of non-zero coefficients. Use ``np.inf`` for no limit.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    fit_path : bool, default=True
        If True the full path is stored in the ``coef_path_`` attribute.
        If you compute the solution for a large problem or many targets,
        setting ``fit_path`` to ``False`` will lead to a speedup, especially
        with a small alpha.

    jitter : float, default=None
        Upper bound on a uniform noise parameter to be added to the
        `y` values, to satisfy the model's assumption of
        one-at-a-time computations. Might help with stability.

        .. versionadded:: 0.23

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for jittering. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`. Ignored if `jitter` is None.

        .. versionadded:: 0.23

    Attributes
    ----------
    alphas_ : array-like of shape (n_alphas + 1,) or list of such arrays
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter``, ``n_features`` or the
        number of nodes in the path with ``alpha >= alpha_min``, whichever
        is smaller. If this is a list of array-like, the length of the outer
        list is `n_targets`.

    active_ : list of shape (n_alphas,) or list of such lists
        Indices of active variables at the end of the path.
        If this is a list of list, the length of the outer list is `n_targets`.

    coef_path_ : array-like of shape (n_features, n_alphas + 1) or list             of such arrays
        The varying values of the coefficients along the path. It is not
        present if the ``fit_path`` parameter is ``False``. If this is a list
        of array-like, the length of the outer list is `n_targets`.

    coef_ : array-like of shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the formulation formula).

    intercept_ : float or array-like of shape (n_targets,)
        Independent term in decision function.

    n_iter_ : array-like or int
        The number of iterations taken by lars_path to find the
        grid of alphas for each target.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    lars_path: Compute Least Angle Regression or Lasso
        path using LARS algorithm.
    LarsCV : Cross-validated Least Angle Regression model.
    sklearn.decomposition.sparse_encode : Sparse coding.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> reg = linear_model.Lars(n_nonzero_coefs=1)
    >>> reg.fit([[-1, 1], [0, 0], [1, 1]], [-1.1111, 0, -1.1111])
    Lars(n_nonzero_coefs=1)
    >>> print(reg.coef_)
    [ 0. -1.11...]
    """
    _parameter_constraints: dict = {'fit_intercept': ['boolean'], 'verbose': ['verbose'], 'normalize': ['boolean', Hidden(StrOptions({'deprecated'}))], 'precompute': ['boolean', StrOptions({'auto'}), np.ndarray, Hidden(None)], 'n_nonzero_coefs': [Interval(Integral, 1, None, closed='left')], 'eps': [Interval(Real, 0, None, closed='left')], 'copy_X': ['boolean'], 'fit_path': ['boolean'], 'jitter': [Interval(Real, 0, None, closed='left'), None], 'random_state': ['random_state']}
    method = 'lar'
    positive = False

    def __init__(self, *, fit_intercept=True, verbose=False, normalize='deprecated', precompute='auto', n_nonzero_coefs=500, eps=np.finfo(float).eps, copy_X=True, fit_path=True, jitter=None, random_state=None):
        if False:
            i = 10
            return i + 15
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.normalize = normalize
        self.precompute = precompute
        self.n_nonzero_coefs = n_nonzero_coefs
        self.eps = eps
        self.copy_X = copy_X
        self.fit_path = fit_path
        self.jitter = jitter
        self.random_state = random_state

    @staticmethod
    def _get_gram(precompute, X, y):
        if False:
            i = 10
            return i + 15
        if not hasattr(precompute, '__array__') and (precompute is True or (precompute == 'auto' and X.shape[0] > X.shape[1]) or (precompute == 'auto' and y.shape[1] > 1)):
            precompute = np.dot(X.T, X)
        return precompute

    def _fit(self, X, y, max_iter, alpha, fit_path, normalize, Xy=None):
        if False:
            for i in range(10):
                print('nop')
        'Auxiliary method to fit the model using X, y as training data'
        n_features = X.shape[1]
        (X, y, X_offset, y_offset, X_scale) = _preprocess_data(X, y, self.fit_intercept, normalize, self.copy_X)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        n_targets = y.shape[1]
        Gram = self._get_gram(self.precompute, X, y)
        self.alphas_ = []
        self.n_iter_ = []
        self.coef_ = np.empty((n_targets, n_features), dtype=X.dtype)
        if fit_path:
            self.active_ = []
            self.coef_path_ = []
            for k in range(n_targets):
                this_Xy = None if Xy is None else Xy[:, k]
                (alphas, active, coef_path, n_iter_) = lars_path(X, y[:, k], Gram=Gram, Xy=this_Xy, copy_X=self.copy_X, copy_Gram=True, alpha_min=alpha, method=self.method, verbose=max(0, self.verbose - 1), max_iter=max_iter, eps=self.eps, return_path=True, return_n_iter=True, positive=self.positive)
                self.alphas_.append(alphas)
                self.active_.append(active)
                self.n_iter_.append(n_iter_)
                self.coef_path_.append(coef_path)
                self.coef_[k] = coef_path[:, -1]
            if n_targets == 1:
                (self.alphas_, self.active_, self.coef_path_, self.coef_) = [a[0] for a in (self.alphas_, self.active_, self.coef_path_, self.coef_)]
                self.n_iter_ = self.n_iter_[0]
        else:
            for k in range(n_targets):
                this_Xy = None if Xy is None else Xy[:, k]
                (alphas, _, self.coef_[k], n_iter_) = lars_path(X, y[:, k], Gram=Gram, Xy=this_Xy, copy_X=self.copy_X, copy_Gram=True, alpha_min=alpha, method=self.method, verbose=max(0, self.verbose - 1), max_iter=max_iter, eps=self.eps, return_path=False, return_n_iter=True, positive=self.positive)
                self.alphas_.append(alphas)
                self.n_iter_.append(n_iter_)
            if n_targets == 1:
                self.alphas_ = self.alphas_[0]
                self.n_iter_ = self.n_iter_[0]
        self._set_intercept(X_offset, y_offset, X_scale)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, Xy=None):
        if False:
            i = 10
            return i + 15
        'Fit the model using X, y as training data.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Training data.\n\n        y : array-like of shape (n_samples,) or (n_samples, n_targets)\n            Target values.\n\n        Xy : array-like of shape (n_features,) or (n_features, n_targets),                 default=None\n            Xy = np.dot(X.T, y) that can be precomputed. It is useful\n            only when the Gram matrix is precomputed.\n\n        Returns\n        -------\n        self : object\n            Returns an instance of self.\n        '
        (X, y) = self._validate_data(X, y, y_numeric=True, multi_output=True)
        _normalize = _deprecate_normalize(self.normalize, estimator_name=self.__class__.__name__)
        alpha = getattr(self, 'alpha', 0.0)
        if hasattr(self, 'n_nonzero_coefs'):
            alpha = 0.0
            max_iter = self.n_nonzero_coefs
        else:
            max_iter = self.max_iter
        if self.jitter is not None:
            rng = check_random_state(self.random_state)
            noise = rng.uniform(high=self.jitter, size=len(y))
            y = y + noise
        self._fit(X, y, max_iter=max_iter, alpha=alpha, fit_path=self.fit_path, normalize=_normalize, Xy=Xy)
        return self

class LassoLars(Lars):
    """Lasso model fit with Least Angle Regression a.k.a. Lars.

    It is a Linear Model trained with an L1 prior as regularizer.

    The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the penalty term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square, solved
        by :class:`LinearRegression`. For numerical reasons, using
        ``alpha = 0`` with the LassoLars object is not advised and you
        should prefer the LinearRegression object.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    verbose : bool or int, default=False
        Sets the verbosity amount.

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

        .. versionchanged:: 1.2
           default changed from True to False in 1.2.

        .. deprecated:: 1.2
            ``normalize`` was deprecated in version 1.2 and will be removed in 1.4.

    precompute : bool, 'auto' or array-like, default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    max_iter : int, default=500
        Maximum number of iterations to perform.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    fit_path : bool, default=True
        If ``True`` the full path is stored in the ``coef_path_`` attribute.
        If you compute the solution for a large problem or many targets,
        setting ``fit_path`` to ``False`` will lead to a speedup, especially
        with a small alpha.

    positive : bool, default=False
        Restrict coefficients to be >= 0. Be aware that you might want to
        remove fit_intercept which is set True by default.
        Under the positive restriction the model coefficients will not converge
        to the ordinary-least-squares solution for small values of alpha.
        Only coefficients up to the smallest alpha value (``alphas_[alphas_ >
        0.].min()`` when fit_path=True) reached by the stepwise Lars-Lasso
        algorithm are typically in congruence with the solution of the
        coordinate descent Lasso estimator.

    jitter : float, default=None
        Upper bound on a uniform noise parameter to be added to the
        `y` values, to satisfy the model's assumption of
        one-at-a-time computations. Might help with stability.

        .. versionadded:: 0.23

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for jittering. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`. Ignored if `jitter` is None.

        .. versionadded:: 0.23

    Attributes
    ----------
    alphas_ : array-like of shape (n_alphas + 1,) or list of such arrays
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter``, ``n_features`` or the
        number of nodes in the path with ``alpha >= alpha_min``, whichever
        is smaller. If this is a list of array-like, the length of the outer
        list is `n_targets`.

    active_ : list of length n_alphas or list of such lists
        Indices of active variables at the end of the path.
        If this is a list of list, the length of the outer list is `n_targets`.

    coef_path_ : array-like of shape (n_features, n_alphas + 1) or list             of such arrays
        If a list is passed it's expected to be one of n_targets such arrays.
        The varying values of the coefficients along the path. It is not
        present if the ``fit_path`` parameter is ``False``. If this is a list
        of array-like, the length of the outer list is `n_targets`.

    coef_ : array-like of shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the formulation formula).

    intercept_ : float or array-like of shape (n_targets,)
        Independent term in decision function.

    n_iter_ : array-like or int
        The number of iterations taken by lars_path to find the
        grid of alphas for each target.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    lars_path : Compute Least Angle Regression or Lasso
        path using LARS algorithm.
    lasso_path : Compute Lasso path with coordinate descent.
    Lasso : Linear Model trained with L1 prior as
        regularizer (aka the Lasso).
    LassoCV : Lasso linear model with iterative fitting
        along a regularization path.
    LassoLarsCV: Cross-validated Lasso, using the LARS algorithm.
    LassoLarsIC : Lasso model fit with Lars using BIC
        or AIC for model selection.
    sklearn.decomposition.sparse_encode : Sparse coding.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> reg = linear_model.LassoLars(alpha=0.01)
    >>> reg.fit([[-1, 1], [0, 0], [1, 1]], [-1, 0, -1])
    LassoLars(alpha=0.01)
    >>> print(reg.coef_)
    [ 0.         -0.955...]
    """
    _parameter_constraints: dict = {**Lars._parameter_constraints, 'alpha': [Interval(Real, 0, None, closed='left')], 'max_iter': [Interval(Integral, 0, None, closed='left')], 'positive': ['boolean']}
    _parameter_constraints.pop('n_nonzero_coefs')
    method = 'lasso'

    def __init__(self, alpha=1.0, *, fit_intercept=True, verbose=False, normalize='deprecated', precompute='auto', max_iter=500, eps=np.finfo(float).eps, copy_X=True, fit_path=True, positive=False, jitter=None, random_state=None):
        if False:
            i = 10
            return i + 15
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.verbose = verbose
        self.normalize = normalize
        self.positive = positive
        self.precompute = precompute
        self.copy_X = copy_X
        self.eps = eps
        self.fit_path = fit_path
        self.jitter = jitter
        self.random_state = random_state

def _check_copy_and_writeable(array, copy=False):
    if False:
        return 10
    if copy or not array.flags.writeable:
        return array.copy()
    return array

def _lars_path_residues(X_train, y_train, X_test, y_test, Gram=None, copy=True, method='lar', verbose=False, fit_intercept=True, normalize=False, max_iter=500, eps=np.finfo(float).eps, positive=False):
    if False:
        return 10
    "Compute the residues on left-out data for a full LARS path\n\n    Parameters\n    -----------\n    X_train : array-like of shape (n_samples, n_features)\n        The data to fit the LARS on\n\n    y_train : array-like of shape (n_samples,)\n        The target variable to fit LARS on\n\n    X_test : array-like of shape (n_samples, n_features)\n        The data to compute the residues on\n\n    y_test : array-like of shape (n_samples,)\n        The target variable to compute the residues on\n\n    Gram : None, 'auto' or array-like of shape (n_features, n_features),             default=None\n        Precomputed Gram matrix (X' * X), if ``'auto'``, the Gram\n        matrix is precomputed from the given X, if there are more samples\n        than features\n\n    copy : bool, default=True\n        Whether X_train, X_test, y_train and y_test should be copied;\n        if False, they may be overwritten.\n\n    method : {'lar' , 'lasso'}, default='lar'\n        Specifies the returned model. Select ``'lar'`` for Least Angle\n        Regression, ``'lasso'`` for the Lasso.\n\n    verbose : bool or int, default=False\n        Sets the amount of verbosity\n\n    fit_intercept : bool, default=True\n        whether to calculate the intercept for this model. If set\n        to false, no intercept will be used in calculations\n        (i.e. data is expected to be centered).\n\n    positive : bool, default=False\n        Restrict coefficients to be >= 0. Be aware that you might want to\n        remove fit_intercept which is set True by default.\n        See reservations for using this option in combination with method\n        'lasso' for expected small values of alpha in the doc of LassoLarsCV\n        and LassoLarsIC.\n\n    normalize : bool, default=False\n        This parameter is ignored when ``fit_intercept`` is set to False.\n        If True, the regressors X will be normalized before regression by\n        subtracting the mean and dividing by the l2-norm.\n        If you wish to standardize, please use\n        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``\n        on an estimator with ``normalize=False``.\n\n        .. versionchanged:: 1.2\n           default changed from True to False in 1.2.\n\n        .. deprecated:: 1.2\n            ``normalize`` was deprecated in version 1.2 and will be removed in 1.4.\n\n    max_iter : int, default=500\n        Maximum number of iterations to perform.\n\n    eps : float, default=np.finfo(float).eps\n        The machine-precision regularization in the computation of the\n        Cholesky diagonal factors. Increase this for very ill-conditioned\n        systems. Unlike the ``tol`` parameter in some iterative\n        optimization-based algorithms, this parameter does not control\n        the tolerance of the optimization.\n\n    Returns\n    --------\n    alphas : array-like of shape (n_alphas,)\n        Maximum of covariances (in absolute value) at each iteration.\n        ``n_alphas`` is either ``max_iter`` or ``n_features``, whichever\n        is smaller.\n\n    active : list\n        Indices of active variables at the end of the path.\n\n    coefs : array-like of shape (n_features, n_alphas)\n        Coefficients along the path\n\n    residues : array-like of shape (n_alphas, n_samples)\n        Residues of the prediction on the test data\n    "
    X_train = _check_copy_and_writeable(X_train, copy)
    y_train = _check_copy_and_writeable(y_train, copy)
    X_test = _check_copy_and_writeable(X_test, copy)
    y_test = _check_copy_and_writeable(y_test, copy)
    if fit_intercept:
        X_mean = X_train.mean(axis=0)
        X_train -= X_mean
        X_test -= X_mean
        y_mean = y_train.mean(axis=0)
        y_train = as_float_array(y_train, copy=False)
        y_train -= y_mean
        y_test = as_float_array(y_test, copy=False)
        y_test -= y_mean
    if normalize:
        norms = np.sqrt(np.sum(X_train ** 2, axis=0))
        nonzeros = np.flatnonzero(norms)
        X_train[:, nonzeros] /= norms[nonzeros]
    (alphas, active, coefs) = lars_path(X_train, y_train, Gram=Gram, copy_X=False, copy_Gram=False, method=method, verbose=max(0, verbose - 1), max_iter=max_iter, eps=eps, positive=positive)
    if normalize:
        coefs[nonzeros] /= norms[nonzeros][:, np.newaxis]
    residues = np.dot(X_test, coefs) - y_test[:, np.newaxis]
    return (alphas, active, coefs, residues.T)

class LarsCV(Lars):
    """Cross-validated Least Angle Regression model.

    See glossary entry for :term:`cross-validation estimator`.

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    verbose : bool or int, default=False
        Sets the verbosity amount.

    max_iter : int, default=500
        Maximum number of iterations to perform.

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

        .. versionchanged:: 1.2
           default changed from True to False in 1.2.

        .. deprecated:: 1.2
            ``normalize`` was deprecated in version 1.2 and will be removed in 1.4.

    precompute : bool, 'auto' or array-like , default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram matrix
        cannot be passed as argument since we will use only subsets of X.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    max_n_alphas : int, default=1000
        The maximum number of points on the path used to compute the
        residuals in the cross-validation.

    n_jobs : int or None, default=None
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    Attributes
    ----------
    active_ : list of length n_alphas or list of such lists
        Indices of active variables at the end of the path.
        If this is a list of lists, the outer list length is `n_targets`.

    coef_ : array-like of shape (n_features,)
        parameter vector (w in the formulation formula)

    intercept_ : float
        independent term in decision function

    coef_path_ : array-like of shape (n_features, n_alphas)
        the varying values of the coefficients along the path

    alpha_ : float
        the estimated regularization parameter alpha

    alphas_ : array-like of shape (n_alphas,)
        the different values of alpha along the path

    cv_alphas_ : array-like of shape (n_cv_alphas,)
        all the values of alpha along the path for the different folds

    mse_path_ : array-like of shape (n_folds, n_cv_alphas)
        the mean square error on left-out for each fold along the path
        (alpha values given by ``cv_alphas``)

    n_iter_ : array-like or int
        the number of iterations run by Lars with the optimal alpha.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    lars_path : Compute Least Angle Regression or Lasso
        path using LARS algorithm.
    lasso_path : Compute Lasso path with coordinate descent.
    Lasso : Linear Model trained with L1 prior as
        regularizer (aka the Lasso).
    LassoCV : Lasso linear model with iterative fitting
        along a regularization path.
    LassoLars : Lasso model fit with Least Angle Regression a.k.a. Lars.
    LassoLarsIC : Lasso model fit with Lars using BIC
        or AIC for model selection.
    sklearn.decomposition.sparse_encode : Sparse coding.

    Notes
    -----
    In `fit`, once the best parameter `alpha` is found through
    cross-validation, the model is fit again using the entire training set.

    Examples
    --------
    >>> from sklearn.linear_model import LarsCV
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=200, noise=4.0, random_state=0)
    >>> reg = LarsCV(cv=5).fit(X, y)
    >>> reg.score(X, y)
    0.9996...
    >>> reg.alpha_
    0.2961...
    >>> reg.predict(X[:1,])
    array([154.3996...])
    """
    _parameter_constraints: dict = {**Lars._parameter_constraints, 'max_iter': [Interval(Integral, 0, None, closed='left')], 'cv': ['cv_object'], 'max_n_alphas': [Interval(Integral, 1, None, closed='left')], 'n_jobs': [Integral, None]}
    for parameter in ['n_nonzero_coefs', 'jitter', 'fit_path', 'random_state']:
        _parameter_constraints.pop(parameter)
    method = 'lar'

    def __init__(self, *, fit_intercept=True, verbose=False, max_iter=500, normalize='deprecated', precompute='auto', cv=None, max_n_alphas=1000, n_jobs=None, eps=np.finfo(float).eps, copy_X=True):
        if False:
            print('Hello World!')
        self.max_iter = max_iter
        self.cv = cv
        self.max_n_alphas = max_n_alphas
        self.n_jobs = n_jobs
        super().__init__(fit_intercept=fit_intercept, verbose=verbose, normalize=normalize, precompute=precompute, n_nonzero_coefs=500, eps=eps, copy_X=copy_X, fit_path=True)

    def _more_tags(self):
        if False:
            return 10
        return {'multioutput': False}

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, **params):
        if False:
            return 10
        'Fit the model using X, y as training data.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Training data.\n\n        y : array-like of shape (n_samples,)\n            Target values.\n\n        **params : dict, default=None\n            Parameters to be passed to the CV splitter.\n\n            .. versionadded:: 1.4\n                Only available if `enable_metadata_routing=True`,\n                which can be set by using\n                ``sklearn.set_config(enable_metadata_routing=True)``.\n                See :ref:`Metadata Routing User Guide <metadata_routing>` for\n                more details.\n\n        Returns\n        -------\n        self : object\n            Returns an instance of self.\n        '
        _raise_for_params(params, self, 'fit')
        _normalize = _deprecate_normalize(self.normalize, estimator_name=self.__class__.__name__)
        (X, y) = self._validate_data(X, y, y_numeric=True)
        X = as_float_array(X, copy=self.copy_X)
        y = as_float_array(y, copy=self.copy_X)
        cv = check_cv(self.cv, classifier=False)
        if _routing_enabled():
            routed_params = process_routing(self, 'fit', **params)
        else:
            routed_params = Bunch(splitter=Bunch(split={}))
        Gram = self.precompute
        if hasattr(Gram, '__array__'):
            warnings.warn('Parameter "precompute" cannot be an array in %s. Automatically switch to "auto" instead.' % self.__class__.__name__)
            Gram = 'auto'
        cv_paths = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)((delayed(_lars_path_residues)(X[train], y[train], X[test], y[test], Gram=Gram, copy=False, method=self.method, verbose=max(0, self.verbose - 1), normalize=_normalize, fit_intercept=self.fit_intercept, max_iter=self.max_iter, eps=self.eps, positive=self.positive) for (train, test) in cv.split(X, y, **routed_params.splitter.split)))
        all_alphas = np.concatenate(list(zip(*cv_paths))[0])
        all_alphas = np.unique(all_alphas)
        stride = int(max(1, int(len(all_alphas) / float(self.max_n_alphas))))
        all_alphas = all_alphas[::stride]
        mse_path = np.empty((len(all_alphas), len(cv_paths)))
        for (index, (alphas, _, _, residues)) in enumerate(cv_paths):
            alphas = alphas[::-1]
            residues = residues[::-1]
            if alphas[0] != 0:
                alphas = np.r_[0, alphas]
                residues = np.r_[residues[0, np.newaxis], residues]
            if alphas[-1] != all_alphas[-1]:
                alphas = np.r_[alphas, all_alphas[-1]]
                residues = np.r_[residues, residues[-1, np.newaxis]]
            this_residues = interpolate.interp1d(alphas, residues, axis=0)(all_alphas)
            this_residues **= 2
            mse_path[:, index] = np.mean(this_residues, axis=-1)
        mask = np.all(np.isfinite(mse_path), axis=-1)
        all_alphas = all_alphas[mask]
        mse_path = mse_path[mask]
        i_best_alpha = np.argmin(mse_path.mean(axis=-1))
        best_alpha = all_alphas[i_best_alpha]
        self.alpha_ = best_alpha
        self.cv_alphas_ = all_alphas
        self.mse_path_ = mse_path
        self._fit(X, y, max_iter=self.max_iter, alpha=best_alpha, Xy=None, fit_path=True, normalize=_normalize)
        return self

    def get_metadata_routing(self):
        if False:
            while True:
                i = 10
        'Get metadata routing of this object.\n\n        Please check :ref:`User Guide <metadata_routing>` on how the routing\n        mechanism works.\n\n        .. versionadded:: 1.4\n\n        Returns\n        -------\n        routing : MetadataRouter\n            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating\n            routing information.\n        '
        router = MetadataRouter(owner=self.__class__.__name__).add(splitter=check_cv(self.cv), method_mapping=MethodMapping().add(callee='split', caller='fit'))
        return router

class LassoLarsCV(LarsCV):
    """Cross-validated Lasso, using the LARS algorithm.

    See glossary entry for :term:`cross-validation estimator`.

    The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    verbose : bool or int, default=False
        Sets the verbosity amount.

    max_iter : int, default=500
        Maximum number of iterations to perform.

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

        .. versionchanged:: 1.2
           default changed from True to False in 1.2.

        .. deprecated:: 1.2
            ``normalize`` was deprecated in version 1.2 and will be removed in 1.4.

    precompute : bool or 'auto' , default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram matrix
        cannot be passed as argument since we will use only subsets of X.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    max_n_alphas : int, default=1000
        The maximum number of points on the path used to compute the
        residuals in the cross-validation.

    n_jobs : int or None, default=None
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    positive : bool, default=False
        Restrict coefficients to be >= 0. Be aware that you might want to
        remove fit_intercept which is set True by default.
        Under the positive restriction the model coefficients do not converge
        to the ordinary-least-squares solution for small values of alpha.
        Only coefficients up to the smallest alpha value (``alphas_[alphas_ >
        0.].min()`` when fit_path=True) reached by the stepwise Lars-Lasso
        algorithm are typically in congruence with the solution of the
        coordinate descent Lasso estimator.
        As a consequence using LassoLarsCV only makes sense for problems where
        a sparse solution is expected and/or reached.

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        parameter vector (w in the formulation formula)

    intercept_ : float
        independent term in decision function.

    coef_path_ : array-like of shape (n_features, n_alphas)
        the varying values of the coefficients along the path

    alpha_ : float
        the estimated regularization parameter alpha

    alphas_ : array-like of shape (n_alphas,)
        the different values of alpha along the path

    cv_alphas_ : array-like of shape (n_cv_alphas,)
        all the values of alpha along the path for the different folds

    mse_path_ : array-like of shape (n_folds, n_cv_alphas)
        the mean square error on left-out for each fold along the path
        (alpha values given by ``cv_alphas``)

    n_iter_ : array-like or int
        the number of iterations run by Lars with the optimal alpha.

    active_ : list of int
        Indices of active variables at the end of the path.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    lars_path : Compute Least Angle Regression or Lasso
        path using LARS algorithm.
    lasso_path : Compute Lasso path with coordinate descent.
    Lasso : Linear Model trained with L1 prior as
        regularizer (aka the Lasso).
    LassoCV : Lasso linear model with iterative fitting
        along a regularization path.
    LassoLars : Lasso model fit with Least Angle Regression a.k.a. Lars.
    LassoLarsIC : Lasso model fit with Lars using BIC
        or AIC for model selection.
    sklearn.decomposition.sparse_encode : Sparse coding.

    Notes
    -----
    The object solves the same problem as the
    :class:`~sklearn.linear_model.LassoCV` object. However, unlike the
    :class:`~sklearn.linear_model.LassoCV`, it find the relevant alphas values
    by itself. In general, because of this property, it will be more stable.
    However, it is more fragile to heavily multicollinear datasets.

    It is more efficient than the :class:`~sklearn.linear_model.LassoCV` if
    only a small number of features are selected compared to the total number,
    for instance if there are very few samples compared to the number of
    features.

    In `fit`, once the best parameter `alpha` is found through
    cross-validation, the model is fit again using the entire training set.

    Examples
    --------
    >>> from sklearn.linear_model import LassoLarsCV
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(noise=4.0, random_state=0)
    >>> reg = LassoLarsCV(cv=5).fit(X, y)
    >>> reg.score(X, y)
    0.9993...
    >>> reg.alpha_
    0.3972...
    >>> reg.predict(X[:1,])
    array([-78.4831...])
    """
    _parameter_constraints = {**LarsCV._parameter_constraints, 'positive': ['boolean']}
    method = 'lasso'

    def __init__(self, *, fit_intercept=True, verbose=False, max_iter=500, normalize='deprecated', precompute='auto', cv=None, max_n_alphas=1000, n_jobs=None, eps=np.finfo(float).eps, copy_X=True, positive=False):
        if False:
            while True:
                i = 10
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.max_iter = max_iter
        self.normalize = normalize
        self.precompute = precompute
        self.cv = cv
        self.max_n_alphas = max_n_alphas
        self.n_jobs = n_jobs
        self.eps = eps
        self.copy_X = copy_X
        self.positive = positive

class LassoLarsIC(LassoLars):
    """Lasso model fit with Lars using BIC or AIC for model selection.

    The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    AIC is the Akaike information criterion [2]_ and BIC is the Bayes
    Information criterion [3]_. Such criteria are useful to select the value
    of the regularization parameter by making a trade-off between the
    goodness of fit and the complexity of the model. A good model should
    explain well the data while being simple.

    Read more in the :ref:`User Guide <lasso_lars_ic>`.

    Parameters
    ----------
    criterion : {'aic', 'bic'}, default='aic'
        The type of criterion to use.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    verbose : bool or int, default=False
        Sets the verbosity amount.

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

        .. versionchanged:: 1.2
           default changed from True to False in 1.2.

        .. deprecated:: 1.2
            ``normalize`` was deprecated in version 1.2 and will be removed in 1.4.

    precompute : bool, 'auto' or array-like, default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    max_iter : int, default=500
        Maximum number of iterations to perform. Can be used for
        early stopping.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    positive : bool, default=False
        Restrict coefficients to be >= 0. Be aware that you might want to
        remove fit_intercept which is set True by default.
        Under the positive restriction the model coefficients do not converge
        to the ordinary-least-squares solution for small values of alpha.
        Only coefficients up to the smallest alpha value (``alphas_[alphas_ >
        0.].min()`` when fit_path=True) reached by the stepwise Lars-Lasso
        algorithm are typically in congruence with the solution of the
        coordinate descent Lasso estimator.
        As a consequence using LassoLarsIC only makes sense for problems where
        a sparse solution is expected and/or reached.

    noise_variance : float, default=None
        The estimated noise variance of the data. If `None`, an unbiased
        estimate is computed by an OLS model. However, it is only possible
        in the case where `n_samples > n_features + fit_intercept`.

        .. versionadded:: 1.1

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        parameter vector (w in the formulation formula)

    intercept_ : float
        independent term in decision function.

    alpha_ : float
        the alpha parameter chosen by the information criterion

    alphas_ : array-like of shape (n_alphas + 1,) or list of such arrays
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter``, ``n_features`` or the
        number of nodes in the path with ``alpha >= alpha_min``, whichever
        is smaller. If a list, it will be of length `n_targets`.

    n_iter_ : int
        number of iterations run by lars_path to find the grid of
        alphas.

    criterion_ : array-like of shape (n_alphas,)
        The value of the information criteria ('aic', 'bic') across all
        alphas. The alpha which has the smallest information criterion is
        chosen, as specified in [1]_.

    noise_variance_ : float
        The estimated noise variance from the data used to compute the
        criterion.

        .. versionadded:: 1.1

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    lars_path : Compute Least Angle Regression or Lasso
        path using LARS algorithm.
    lasso_path : Compute Lasso path with coordinate descent.
    Lasso : Linear Model trained with L1 prior as
        regularizer (aka the Lasso).
    LassoCV : Lasso linear model with iterative fitting
        along a regularization path.
    LassoLars : Lasso model fit with Least Angle Regression a.k.a. Lars.
    LassoLarsCV: Cross-validated Lasso, using the LARS algorithm.
    sklearn.decomposition.sparse_encode : Sparse coding.

    Notes
    -----
    The number of degrees of freedom is computed as in [1]_.

    To have more details regarding the mathematical formulation of the
    AIC and BIC criteria, please refer to :ref:`User Guide <lasso_lars_ic>`.

    References
    ----------
    .. [1] :arxiv:`Zou, Hui, Trevor Hastie, and Robert Tibshirani.
            "On the degrees of freedom of the lasso."
            The Annals of Statistics 35.5 (2007): 2173-2192.
            <0712.0881>`

    .. [2] `Wikipedia entry on the Akaike information criterion
            <https://en.wikipedia.org/wiki/Akaike_information_criterion>`_

    .. [3] `Wikipedia entry on the Bayesian information criterion
            <https://en.wikipedia.org/wiki/Bayesian_information_criterion>`_

    Examples
    --------
    >>> from sklearn import linear_model
    >>> reg = linear_model.LassoLarsIC(criterion='bic')
    >>> X = [[-2, 2], [-1, 1], [0, 0], [1, 1], [2, 2]]
    >>> y = [-2.2222, -1.1111, 0, -1.1111, -2.2222]
    >>> reg.fit(X, y)
    LassoLarsIC(criterion='bic')
    >>> print(reg.coef_)
    [ 0.  -1.11...]
    """
    _parameter_constraints: dict = {**LassoLars._parameter_constraints, 'criterion': [StrOptions({'aic', 'bic'})], 'noise_variance': [Interval(Real, 0, None, closed='left'), None]}
    for parameter in ['jitter', 'fit_path', 'alpha', 'random_state']:
        _parameter_constraints.pop(parameter)

    def __init__(self, criterion='aic', *, fit_intercept=True, verbose=False, normalize='deprecated', precompute='auto', max_iter=500, eps=np.finfo(float).eps, copy_X=True, positive=False, noise_variance=None):
        if False:
            while True:
                i = 10
        self.criterion = criterion
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.max_iter = max_iter
        self.verbose = verbose
        self.normalize = normalize
        self.copy_X = copy_X
        self.precompute = precompute
        self.eps = eps
        self.fit_path = True
        self.noise_variance = noise_variance

    def _more_tags(self):
        if False:
            print('Hello World!')
        return {'multioutput': False}

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, copy_X=None):
        if False:
            i = 10
            return i + 15
        "Fit the model using X, y as training data.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Training data.\n\n        y : array-like of shape (n_samples,)\n            Target values. Will be cast to X's dtype if necessary.\n\n        copy_X : bool, default=None\n            If provided, this parameter will override the choice\n            of copy_X made at instance creation.\n            If ``True``, X will be copied; else, it may be overwritten.\n\n        Returns\n        -------\n        self : object\n            Returns an instance of self.\n        "
        _normalize = _deprecate_normalize(self.normalize, estimator_name=self.__class__.__name__)
        if copy_X is None:
            copy_X = self.copy_X
        (X, y) = self._validate_data(X, y, y_numeric=True)
        (X, y, Xmean, ymean, Xstd) = _preprocess_data(X, y, self.fit_intercept, _normalize, copy_X)
        Gram = self.precompute
        (alphas_, _, coef_path_, self.n_iter_) = lars_path(X, y, Gram=Gram, copy_X=copy_X, copy_Gram=True, alpha_min=0.0, method='lasso', verbose=self.verbose, max_iter=self.max_iter, eps=self.eps, return_n_iter=True, positive=self.positive)
        n_samples = X.shape[0]
        if self.criterion == 'aic':
            criterion_factor = 2
        elif self.criterion == 'bic':
            criterion_factor = log(n_samples)
        else:
            raise ValueError(f'criterion should be either bic or aic, got {self.criterion!r}')
        residuals = y[:, np.newaxis] - np.dot(X, coef_path_)
        residuals_sum_squares = np.sum(residuals ** 2, axis=0)
        degrees_of_freedom = np.zeros(coef_path_.shape[1], dtype=int)
        for (k, coef) in enumerate(coef_path_.T):
            mask = np.abs(coef) > np.finfo(coef.dtype).eps
            if not np.any(mask):
                continue
            degrees_of_freedom[k] = np.sum(mask)
        self.alphas_ = alphas_
        if self.noise_variance is None:
            self.noise_variance_ = self._estimate_noise_variance(X, y, positive=self.positive)
        else:
            self.noise_variance_ = self.noise_variance
        self.criterion_ = n_samples * np.log(2 * np.pi * self.noise_variance_) + residuals_sum_squares / self.noise_variance_ + criterion_factor * degrees_of_freedom
        n_best = np.argmin(self.criterion_)
        self.alpha_ = alphas_[n_best]
        self.coef_ = coef_path_[:, n_best]
        self._set_intercept(Xmean, ymean, Xstd)
        return self

    def _estimate_noise_variance(self, X, y, positive):
        if False:
            i = 10
            return i + 15
        'Compute an estimate of the variance with an OLS model.\n\n        Parameters\n        ----------\n        X : ndarray of shape (n_samples, n_features)\n            Data to be fitted by the OLS model. We expect the data to be\n            centered.\n\n        y : ndarray of shape (n_samples,)\n            Associated target.\n\n        positive : bool, default=False\n            Restrict coefficients to be >= 0. This should be inline with\n            the `positive` parameter from `LassoLarsIC`.\n\n        Returns\n        -------\n        noise_variance : float\n            An estimator of the noise variance of an OLS model.\n        '
        if X.shape[0] <= X.shape[1] + self.fit_intercept:
            raise ValueError(f'You are using {self.__class__.__name__} in the case where the number of samples is smaller than the number of features. In this setting, getting a good estimate for the variance of the noise is not possible. Provide an estimate of the noise variance in the constructor.')
        ols_model = LinearRegression(positive=positive, fit_intercept=False)
        y_pred = ols_model.fit(X, y).predict(X)
        return np.sum((y - y_pred) ** 2) / (X.shape[0] - X.shape[1] - self.fit_intercept)