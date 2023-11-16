"""
Loss functions for linear models with raw_prediction = X @ coef
"""
import numpy as np
from scipy import sparse
from ..utils.extmath import squared_norm

class LinearModelLoss:
    """General class for loss functions with raw_prediction = X @ coef + intercept.

    Note that raw_prediction is also known as linear predictor.

    The loss is the average of per sample losses and includes a term for L2
    regularization::

        loss = 1 / s_sum * sum_i s_i loss(y_i, X_i @ coef + intercept)
               + 1/2 * l2_reg_strength * ||coef||_2^2

    with sample weights s_i=1 if sample_weight=None and s_sum=sum_i s_i.

    Gradient and hessian, for simplicity without intercept, are::

        gradient = 1 / s_sum * X.T @ loss.gradient + l2_reg_strength * coef
        hessian = 1 / s_sum * X.T @ diag(loss.hessian) @ X
                  + l2_reg_strength * identity

    Conventions:
        if fit_intercept:
            n_dof =  n_features + 1
        else:
            n_dof = n_features

        if base_loss.is_multiclass:
            coef.shape = (n_classes, n_dof) or ravelled (n_classes * n_dof,)
        else:
            coef.shape = (n_dof,)

        The intercept term is at the end of the coef array:
        if base_loss.is_multiclass:
            if coef.shape (n_classes, n_dof):
                intercept = coef[:, -1]
            if coef.shape (n_classes * n_dof,)
                intercept = coef[n_features::n_dof] = coef[(n_dof-1)::n_dof]
            intercept.shape = (n_classes,)
        else:
            intercept = coef[-1]

    Note: If coef has shape (n_classes * n_dof,), the 2d-array can be reconstructed as

        coef.reshape((n_classes, -1), order="F")

    The option order="F" makes coef[:, i] contiguous. This, in turn, makes the
    coefficients without intercept, coef[:, :-1], contiguous and speeds up
    matrix-vector computations.

    Note: If the average loss per sample is wanted instead of the sum of the loss per
    sample, one can simply use a rescaled sample_weight such that
    sum(sample_weight) = 1.

    Parameters
    ----------
    base_loss : instance of class BaseLoss from sklearn._loss.
    fit_intercept : bool
    """

    def __init__(self, base_loss, fit_intercept):
        if False:
            print('Hello World!')
        self.base_loss = base_loss
        self.fit_intercept = fit_intercept

    def init_zero_coef(self, X, dtype=None):
        if False:
            i = 10
            return i + 15
        'Allocate coef of correct shape with zeros.\n\n        Parameters:\n        -----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Training data.\n        dtype : data-type, default=None\n            Overrides the data type of coef. With dtype=None, coef will have the same\n            dtype as X.\n\n        Returns\n        -------\n        coef : ndarray of shape (n_dof,) or (n_classes, n_dof)\n            Coefficients of a linear model.\n        '
        n_features = X.shape[1]
        n_classes = self.base_loss.n_classes
        if self.fit_intercept:
            n_dof = n_features + 1
        else:
            n_dof = n_features
        if self.base_loss.is_multiclass:
            coef = np.zeros_like(X, shape=(n_classes, n_dof), dtype=dtype, order='F')
        else:
            coef = np.zeros_like(X, shape=n_dof, dtype=dtype)
        return coef

    def weight_intercept(self, coef):
        if False:
            print('Hello World!')
        'Helper function to get coefficients and intercept.\n\n        Parameters\n        ----------\n        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)\n            Coefficients of a linear model.\n            If shape (n_classes * n_dof,), the classes of one feature are contiguous,\n            i.e. one reconstructs the 2d-array via\n            coef.reshape((n_classes, -1), order="F").\n\n        Returns\n        -------\n        weights : ndarray of shape (n_features,) or (n_classes, n_features)\n            Coefficients without intercept term.\n        intercept : float or ndarray of shape (n_classes,)\n            Intercept terms.\n        '
        if not self.base_loss.is_multiclass:
            if self.fit_intercept:
                intercept = coef[-1]
                weights = coef[:-1]
            else:
                intercept = 0.0
                weights = coef
        else:
            if coef.ndim == 1:
                weights = coef.reshape((self.base_loss.n_classes, -1), order='F')
            else:
                weights = coef
            if self.fit_intercept:
                intercept = weights[:, -1]
                weights = weights[:, :-1]
            else:
                intercept = 0.0
        return (weights, intercept)

    def weight_intercept_raw(self, coef, X):
        if False:
            return 10
        'Helper function to get coefficients, intercept and raw_prediction.\n\n        Parameters\n        ----------\n        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)\n            Coefficients of a linear model.\n            If shape (n_classes * n_dof,), the classes of one feature are contiguous,\n            i.e. one reconstructs the 2d-array via\n            coef.reshape((n_classes, -1), order="F").\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Training data.\n\n        Returns\n        -------\n        weights : ndarray of shape (n_features,) or (n_classes, n_features)\n            Coefficients without intercept term.\n        intercept : float or ndarray of shape (n_classes,)\n            Intercept terms.\n        raw_prediction : ndarray of shape (n_samples,) or             (n_samples, n_classes)\n        '
        (weights, intercept) = self.weight_intercept(coef)
        if not self.base_loss.is_multiclass:
            raw_prediction = X @ weights + intercept
        else:
            raw_prediction = X @ weights.T + intercept
        return (weights, intercept, raw_prediction)

    def l2_penalty(self, weights, l2_reg_strength):
        if False:
            while True:
                i = 10
        'Compute L2 penalty term l2_reg_strength/2 *||w||_2^2.'
        norm2_w = weights @ weights if weights.ndim == 1 else squared_norm(weights)
        return 0.5 * l2_reg_strength * norm2_w

    def loss(self, coef, X, y, sample_weight=None, l2_reg_strength=0.0, n_threads=1, raw_prediction=None):
        if False:
            i = 10
            return i + 15
        'Compute the loss as weighted average over point-wise losses.\n\n        Parameters\n        ----------\n        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)\n            Coefficients of a linear model.\n            If shape (n_classes * n_dof,), the classes of one feature are contiguous,\n            i.e. one reconstructs the 2d-array via\n            coef.reshape((n_classes, -1), order="F").\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Training data.\n        y : contiguous array of shape (n_samples,)\n            Observed, true target values.\n        sample_weight : None or contiguous array of shape (n_samples,), default=None\n            Sample weights.\n        l2_reg_strength : float, default=0.0\n            L2 regularization strength\n        n_threads : int, default=1\n            Number of OpenMP threads to use.\n        raw_prediction : C-contiguous array of shape (n_samples,) or array of             shape (n_samples, n_classes)\n            Raw prediction values (in link space). If provided, these are used. If\n            None, then raw_prediction = X @ coef + intercept is calculated.\n\n        Returns\n        -------\n        loss : float\n            Weighted average of losses per sample, plus penalty.\n        '
        if raw_prediction is None:
            (weights, intercept, raw_prediction) = self.weight_intercept_raw(coef, X)
        else:
            (weights, intercept) = self.weight_intercept(coef)
        loss = self.base_loss.loss(y_true=y, raw_prediction=raw_prediction, sample_weight=None, n_threads=n_threads)
        loss = np.average(loss, weights=sample_weight)
        return loss + self.l2_penalty(weights, l2_reg_strength)

    def loss_gradient(self, coef, X, y, sample_weight=None, l2_reg_strength=0.0, n_threads=1, raw_prediction=None):
        if False:
            i = 10
            return i + 15
        'Computes the sum of loss and gradient w.r.t. coef.\n\n        Parameters\n        ----------\n        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)\n            Coefficients of a linear model.\n            If shape (n_classes * n_dof,), the classes of one feature are contiguous,\n            i.e. one reconstructs the 2d-array via\n            coef.reshape((n_classes, -1), order="F").\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Training data.\n        y : contiguous array of shape (n_samples,)\n            Observed, true target values.\n        sample_weight : None or contiguous array of shape (n_samples,), default=None\n            Sample weights.\n        l2_reg_strength : float, default=0.0\n            L2 regularization strength\n        n_threads : int, default=1\n            Number of OpenMP threads to use.\n        raw_prediction : C-contiguous array of shape (n_samples,) or array of             shape (n_samples, n_classes)\n            Raw prediction values (in link space). If provided, these are used. If\n            None, then raw_prediction = X @ coef + intercept is calculated.\n\n        Returns\n        -------\n        loss : float\n            Weighted average of losses per sample, plus penalty.\n\n        gradient : ndarray of shape coef.shape\n             The gradient of the loss.\n        '
        ((n_samples, n_features), n_classes) = (X.shape, self.base_loss.n_classes)
        n_dof = n_features + int(self.fit_intercept)
        if raw_prediction is None:
            (weights, intercept, raw_prediction) = self.weight_intercept_raw(coef, X)
        else:
            (weights, intercept) = self.weight_intercept(coef)
        (loss, grad_pointwise) = self.base_loss.loss_gradient(y_true=y, raw_prediction=raw_prediction, sample_weight=sample_weight, n_threads=n_threads)
        sw_sum = n_samples if sample_weight is None else np.sum(sample_weight)
        loss = loss.sum() / sw_sum
        loss += self.l2_penalty(weights, l2_reg_strength)
        grad_pointwise /= sw_sum
        if not self.base_loss.is_multiclass:
            grad = np.empty_like(coef, dtype=weights.dtype)
            grad[:n_features] = X.T @ grad_pointwise + l2_reg_strength * weights
            if self.fit_intercept:
                grad[-1] = grad_pointwise.sum()
        else:
            grad = np.empty((n_classes, n_dof), dtype=weights.dtype, order='F')
            grad[:, :n_features] = grad_pointwise.T @ X + l2_reg_strength * weights
            if self.fit_intercept:
                grad[:, -1] = grad_pointwise.sum(axis=0)
            if coef.ndim == 1:
                grad = grad.ravel(order='F')
        return (loss, grad)

    def gradient(self, coef, X, y, sample_weight=None, l2_reg_strength=0.0, n_threads=1, raw_prediction=None):
        if False:
            i = 10
            return i + 15
        'Computes the gradient w.r.t. coef.\n\n        Parameters\n        ----------\n        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)\n            Coefficients of a linear model.\n            If shape (n_classes * n_dof,), the classes of one feature are contiguous,\n            i.e. one reconstructs the 2d-array via\n            coef.reshape((n_classes, -1), order="F").\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Training data.\n        y : contiguous array of shape (n_samples,)\n            Observed, true target values.\n        sample_weight : None or contiguous array of shape (n_samples,), default=None\n            Sample weights.\n        l2_reg_strength : float, default=0.0\n            L2 regularization strength\n        n_threads : int, default=1\n            Number of OpenMP threads to use.\n        raw_prediction : C-contiguous array of shape (n_samples,) or array of             shape (n_samples, n_classes)\n            Raw prediction values (in link space). If provided, these are used. If\n            None, then raw_prediction = X @ coef + intercept is calculated.\n\n        Returns\n        -------\n        gradient : ndarray of shape coef.shape\n             The gradient of the loss.\n        '
        ((n_samples, n_features), n_classes) = (X.shape, self.base_loss.n_classes)
        n_dof = n_features + int(self.fit_intercept)
        if raw_prediction is None:
            (weights, intercept, raw_prediction) = self.weight_intercept_raw(coef, X)
        else:
            (weights, intercept) = self.weight_intercept(coef)
        grad_pointwise = self.base_loss.gradient(y_true=y, raw_prediction=raw_prediction, sample_weight=sample_weight, n_threads=n_threads)
        sw_sum = n_samples if sample_weight is None else np.sum(sample_weight)
        grad_pointwise /= sw_sum
        if not self.base_loss.is_multiclass:
            grad = np.empty_like(coef, dtype=weights.dtype)
            grad[:n_features] = X.T @ grad_pointwise + l2_reg_strength * weights
            if self.fit_intercept:
                grad[-1] = grad_pointwise.sum()
            return grad
        else:
            grad = np.empty((n_classes, n_dof), dtype=weights.dtype, order='F')
            grad[:, :n_features] = grad_pointwise.T @ X + l2_reg_strength * weights
            if self.fit_intercept:
                grad[:, -1] = grad_pointwise.sum(axis=0)
            if coef.ndim == 1:
                return grad.ravel(order='F')
            else:
                return grad

    def gradient_hessian(self, coef, X, y, sample_weight=None, l2_reg_strength=0.0, n_threads=1, gradient_out=None, hessian_out=None, raw_prediction=None):
        if False:
            return 10
        'Computes gradient and hessian w.r.t. coef.\n\n        Parameters\n        ----------\n        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)\n            Coefficients of a linear model.\n            If shape (n_classes * n_dof,), the classes of one feature are contiguous,\n            i.e. one reconstructs the 2d-array via\n            coef.reshape((n_classes, -1), order="F").\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Training data.\n        y : contiguous array of shape (n_samples,)\n            Observed, true target values.\n        sample_weight : None or contiguous array of shape (n_samples,), default=None\n            Sample weights.\n        l2_reg_strength : float, default=0.0\n            L2 regularization strength\n        n_threads : int, default=1\n            Number of OpenMP threads to use.\n        gradient_out : None or ndarray of shape coef.shape\n            A location into which the gradient is stored. If None, a new array\n            might be created.\n        hessian_out : None or ndarray\n            A location into which the hessian is stored. If None, a new array\n            might be created.\n        raw_prediction : C-contiguous array of shape (n_samples,) or array of             shape (n_samples, n_classes)\n            Raw prediction values (in link space). If provided, these are used. If\n            None, then raw_prediction = X @ coef + intercept is calculated.\n\n        Returns\n        -------\n        gradient : ndarray of shape coef.shape\n             The gradient of the loss.\n\n        hessian : ndarray\n            Hessian matrix.\n\n        hessian_warning : bool\n            True if pointwise hessian has more than half of its elements non-positive.\n        '
        (n_samples, n_features) = X.shape
        n_dof = n_features + int(self.fit_intercept)
        if raw_prediction is None:
            (weights, intercept, raw_prediction) = self.weight_intercept_raw(coef, X)
        else:
            (weights, intercept) = self.weight_intercept(coef)
        (grad_pointwise, hess_pointwise) = self.base_loss.gradient_hessian(y_true=y, raw_prediction=raw_prediction, sample_weight=sample_weight, n_threads=n_threads)
        sw_sum = n_samples if sample_weight is None else np.sum(sample_weight)
        grad_pointwise /= sw_sum
        hess_pointwise /= sw_sum
        hessian_warning = np.mean(hess_pointwise <= 0) > 0.25
        hess_pointwise = np.abs(hess_pointwise)
        if not self.base_loss.is_multiclass:
            if gradient_out is None:
                grad = np.empty_like(coef, dtype=weights.dtype)
            else:
                grad = gradient_out
            grad[:n_features] = X.T @ grad_pointwise + l2_reg_strength * weights
            if self.fit_intercept:
                grad[-1] = grad_pointwise.sum()
            if hessian_out is None:
                hess = np.empty(shape=(n_dof, n_dof), dtype=weights.dtype)
            else:
                hess = hessian_out
            if hessian_warning:
                return (grad, hess, hessian_warning)
            if sparse.issparse(X):
                hess[:n_features, :n_features] = (X.T @ sparse.dia_matrix((hess_pointwise, 0), shape=(n_samples, n_samples)) @ X).toarray()
            else:
                WX = hess_pointwise[:, None] * X
                hess[:n_features, :n_features] = np.dot(X.T, WX)
            if l2_reg_strength > 0:
                hess.reshape(-1)[:n_features * n_dof:n_dof + 1] += l2_reg_strength
            if self.fit_intercept:
                Xh = X.T @ hess_pointwise
                hess[:-1, -1] = Xh
                hess[-1, :-1] = Xh
                hess[-1, -1] = hess_pointwise.sum()
        else:
            raise NotImplementedError
        return (grad, hess, hessian_warning)

    def gradient_hessian_product(self, coef, X, y, sample_weight=None, l2_reg_strength=0.0, n_threads=1):
        if False:
            i = 10
            return i + 15
        'Computes gradient and hessp (hessian product function) w.r.t. coef.\n\n        Parameters\n        ----------\n        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)\n            Coefficients of a linear model.\n            If shape (n_classes * n_dof,), the classes of one feature are contiguous,\n            i.e. one reconstructs the 2d-array via\n            coef.reshape((n_classes, -1), order="F").\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Training data.\n        y : contiguous array of shape (n_samples,)\n            Observed, true target values.\n        sample_weight : None or contiguous array of shape (n_samples,), default=None\n            Sample weights.\n        l2_reg_strength : float, default=0.0\n            L2 regularization strength\n        n_threads : int, default=1\n            Number of OpenMP threads to use.\n\n        Returns\n        -------\n        gradient : ndarray of shape coef.shape\n             The gradient of the loss.\n\n        hessp : callable\n            Function that takes in a vector input of shape of gradient and\n            and returns matrix-vector product with hessian.\n        '
        ((n_samples, n_features), n_classes) = (X.shape, self.base_loss.n_classes)
        n_dof = n_features + int(self.fit_intercept)
        (weights, intercept, raw_prediction) = self.weight_intercept_raw(coef, X)
        sw_sum = n_samples if sample_weight is None else np.sum(sample_weight)
        if not self.base_loss.is_multiclass:
            (grad_pointwise, hess_pointwise) = self.base_loss.gradient_hessian(y_true=y, raw_prediction=raw_prediction, sample_weight=sample_weight, n_threads=n_threads)
            grad_pointwise /= sw_sum
            hess_pointwise /= sw_sum
            grad = np.empty_like(coef, dtype=weights.dtype)
            grad[:n_features] = X.T @ grad_pointwise + l2_reg_strength * weights
            if self.fit_intercept:
                grad[-1] = grad_pointwise.sum()
            hessian_sum = hess_pointwise.sum()
            if sparse.issparse(X):
                hX = sparse.dia_matrix((hess_pointwise, 0), shape=(n_samples, n_samples)) @ X
            else:
                hX = hess_pointwise[:, np.newaxis] * X
            if self.fit_intercept:
                hX_sum = np.squeeze(np.asarray(hX.sum(axis=0)))
                hX_sum = np.atleast_1d(hX_sum)

            def hessp(s):
                if False:
                    while True:
                        i = 10
                ret = np.empty_like(s)
                if sparse.issparse(X):
                    ret[:n_features] = X.T @ (hX @ s[:n_features])
                else:
                    ret[:n_features] = np.linalg.multi_dot([X.T, hX, s[:n_features]])
                ret[:n_features] += l2_reg_strength * s[:n_features]
                if self.fit_intercept:
                    ret[:n_features] += s[-1] * hX_sum
                    ret[-1] = hX_sum @ s[:n_features] + hessian_sum * s[-1]
                return ret
        else:
            (grad_pointwise, proba) = self.base_loss.gradient_proba(y_true=y, raw_prediction=raw_prediction, sample_weight=sample_weight, n_threads=n_threads)
            grad_pointwise /= sw_sum
            grad = np.empty((n_classes, n_dof), dtype=weights.dtype, order='F')
            grad[:, :n_features] = grad_pointwise.T @ X + l2_reg_strength * weights
            if self.fit_intercept:
                grad[:, -1] = grad_pointwise.sum(axis=0)

            def hessp(s):
                if False:
                    i = 10
                    return i + 15
                s = s.reshape((n_classes, -1), order='F')
                if self.fit_intercept:
                    s_intercept = s[:, -1]
                    s = s[:, :-1]
                else:
                    s_intercept = 0
                tmp = X @ s.T + s_intercept
                tmp += (-proba * tmp).sum(axis=1)[:, np.newaxis]
                tmp *= proba
                if sample_weight is not None:
                    tmp *= sample_weight[:, np.newaxis]
                hess_prod = np.empty((n_classes, n_dof), dtype=weights.dtype, order='F')
                hess_prod[:, :n_features] = tmp.T @ X / sw_sum + l2_reg_strength * s
                if self.fit_intercept:
                    hess_prod[:, -1] = tmp.sum(axis=0) / sw_sum
                if coef.ndim == 1:
                    return hess_prod.ravel(order='F')
                else:
                    return hess_prod
            if coef.ndim == 1:
                return (grad.ravel(order='F'), hessp)
        return (grad, hessp)