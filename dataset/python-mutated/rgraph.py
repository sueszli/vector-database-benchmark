""" R-graph

"""
from __future__ import division
from __future__ import print_function
import warnings
import numpy as np
from scipy import sparse
from sklearn.decomposition import sparse_encode
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.utils import check_array
from .base import BaseDetector

class RGraph(BaseDetector):
    """ Outlier Detection via R-graph.
    Paper: https://openaccess.thecvf.com/content_cvpr_2017/papers/You_Provable_Self-Representation_Based_CVPR_2017_paper.pdf
    See :cite:`you2017provable` for details.

    Parameters
    ----------
    transition_steps : int, optional (default=20)
        Number of transition steps that are taken in the graph, after which 
        the outlier scores are determined.

    gamma : float

    gamma_nz : boolean, default True
        gamma and gamma_nz together determines the parameter alpha.
        When ``gamma_nz = False``, alpha = gamma.
        When ``gamma_nz = True``, then alpha = gamma * alpha0, where alpha0 is
        the largest number such that the solution to the optimization problem
        with alpha = alpha0 is the zero vector (see Proposition 1 in [1]).
        Therefore, when ``gamma_nz = True``, gamma should be a value greater
        than 1.0. A good choice is typically in the range [5, 500].

    tau : float, default 1.0
        Parameter for elastic net penalty term. 
        When tau = 1.0, the method reduces to sparse subspace clustering with
        basis pursuit (SSC-BP) [2].
        When tau = 0.0, the method reduces to least squares regression (LSR).

    algorithm : string, default ``lasso_lars``
        Algorithm for computing the representation. Either lasso_lars or
        lasso_cd.
        Note: ``lasso_lars`` and ``lasso_cd`` only support tau = 1.
        For cases tau << 1 linear regression is used.


    fit_intercept_LR: bool, optional (default=False)
        For  ``gamma`` > 10000 linear regression is used instead of
        ``lasso_lars`` or ``lasso_cd``. This parameter determines whether the
        intercept for the model is calculated.

    maxiter_lasso : int, default 1000
        The maximum number of iterations for ``lasso_lars`` and ``lasso_cd``.

    n_nonzero : int, default 50
        This is an upper bound on the number of nonzero entries of each
        representation vector.
        If there are more than n_nonzero nonzero entries,
        only the top n_nonzero number of
        entries with largest absolute value are kept.

    active_support: boolean, default True
        Set to True to use the active support algorithm in [1] for solving the
        optimization problem. This should significantly reduce the running time
        when n_samples is large.

    active_support_params: dictionary of string to any, optional
        Parameters (keyword arguments) and values for the active support
        algorithm. It may be used to set the parameters ``support_init``,
        ``support_size`` and ``maxiter``, see
        ``active_support_elastic_net`` for details. 
        Example: active_support_params={'support_size':50, 'maxiter':100}
        Ignored when ``active_support=False``

    preprocessing : bool, optional (default=True)
        If True, apply standardization on the data.


    verbose : int, optional (default=1)
        Verbosity mode.

        - 0 = silent
        - 1 = progress bar
        - 2 = one line per epoch.

        For verbose >= 1, model summary may be printed.

    random_state : random_state: int, RandomState instance or None, optional
        (default=None)
        If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. When fitting this is used
        to define the threshold on the decision function.

    blocksize_test_data: int, optional (default=10)
        Test set is splitted into blocks of the size ``blocksize_test_data``
        to at least partially separate test - and train set

    Attributes
    ----------
    
    transition_matrix_ : numpy array of shape (n_samples,)
        Transition matrix from the last fitted data, this might include 
        training + test data


    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, transition_steps=10, n_nonzero=10, gamma=50.0, gamma_nz=True, algorithm='lasso_lars', tau=1.0, maxiter_lasso=1000, preprocessing=True, contamination=0.1, blocksize_test_data=10, support_init='L2', maxiter=40, support_size=100, active_support=True, fit_intercept_LR=False, verbose=True):
        if False:
            print('Hello World!')
        super(RGraph, self).__init__(contamination=contamination)
        self.transition_steps = transition_steps
        self.n_nonzero = n_nonzero
        self.gamma = gamma
        self.gamma_nz = gamma_nz
        self.algorithm = algorithm
        self.tau = tau
        self.preprocessing = preprocessing
        self.contamination = contamination
        self.maxiter_lasso = maxiter_lasso
        self.support_init = support_init
        self.maxiter = maxiter
        self.support_size = support_size
        self.active_support = active_support
        self.verbose = verbose
        self.blocksize_test_data = blocksize_test_data
        self.fit_intercept_LR = fit_intercept_LR

    def active_support_elastic_net(self, X, y, alpha, tau=1.0, algorithm='lasso_lars', support_init='L2', support_size=100, maxiter=40, maxiter_lasso=1000):
        if False:
            print('Hello World!')
        '\n        Source: https://github.com/ChongYou/subspace-clustering/blob/master/cluster/selfrepresentation.py\n            An active support based algorithm for solving the elastic net optimization problem\n            min_{c} tau ||c||_1 + (1-tau)/2 ||c||_2^2 + alpha / 2 ||y - c X ||_2^2.\n        \n        Parameters\n        -----------\n        X : array-like, shape (n_samples, n_features)\n\n        y : array-like, shape (1, n_features)\n\n        alpha : float\n\n        tau : float, default 1.0\n\n        algorithm : string, default ``spams``\n            Algorithm for computing solving the subproblems. Either lasso_lars\n            or lasso_cd or spams\n            (installation of spams package is required).\n            Note: ``lasso_lars`` and ``lasso_cd`` only support tau = 1.\n\n        support_init: string, default ``knn``\n            This determines how the active support is initialized.\n            It can be either ``knn`` or ``L2``.\n\n        support_size: int, default 100\n            This determines the size of the working set.\n            A small support_size decreases the runtime per iteration while\n            increase the number of iterations.\n\n        maxiter: int default 40\n            Termination condition for active support update.\n        \n        Returns\n        -------\n        c : shape n_samples\n            The optimal solution to the optimization problem.\n        '
        n_samples = X.shape[0]
        if n_samples <= support_size:
            supp = np.arange(n_samples, dtype=int)
        elif support_init == 'L2':
            L2sol = np.linalg.solve(np.identity(y.shape[1]) * alpha + np.dot(X.T, X), y.T)
            c0 = np.dot(X, L2sol)[:, 0]
            supp = np.argpartition(-np.abs(c0), support_size)[0:support_size]
        elif support_init == 'knn':
            supp = np.argpartition(-np.abs(np.dot(y, X.T)[0]), support_size)[0:support_size]
        curr_obj = float('inf')
        for _ in range(maxiter):
            Xs = X[supp, :]
            cs = sparse_encode(y, Xs, algorithm=algorithm, alpha=alpha, max_iter=maxiter_lasso)
            delta = (y - np.dot(cs, Xs)) / alpha
            obj = tau * np.sum(np.abs(cs[0])) + (1.0 - tau) / 2.0 * np.sum(np.power(cs[0], 2.0)) + alpha / 2.0 * np.sum(np.power(delta, 2.0))
            if curr_obj - obj < 1e-10 * curr_obj:
                break
            curr_obj = obj
            coherence = np.abs(np.dot(delta, X.T))[0]
            coherence[supp] = 0
            addedsupp = np.nonzero(coherence > tau + 1e-10)[0]
            if addedsupp.size == 0:
                break
            activesupp = supp[np.abs(cs[0]) > 1e-10]
            if activesupp.size > 0.8 * support_size:
                support_size = min([round(max([activesupp.size, support_size]) * 1.1), n_samples])
            if addedsupp.size + activesupp.size > support_size:
                ord = np.argpartition(-coherence[addedsupp], support_size - activesupp.size)[0:support_size - activesupp.size]
                addedsupp = addedsupp[ord]
            supp = np.concatenate([activesupp, addedsupp])
        c = np.zeros(n_samples)
        c[supp] = cs
        return c

    def elastic_net_subspace_clustering(self, X, gamma=50.0, gamma_nz=True, tau=1.0, algorithm='lasso_lars', fit_intercept_LR=False, active_support=True, active_support_params=None, n_nonzero=50, maxiter_lasso=1000):
        if False:
            while True:
                i = 10
        "\n        Source: https://github.com/ChongYou/subspace-clustering/blob/master/cluster/selfrepresentation.py\n        \n        Elastic net subspace clustering (EnSC) [1]. \n        Compute self-representation matrix C from solving the following optimization problem\n        min_{c_j} tau ||c_j||_1 + (1-tau)/2 ||c_j||_2^2 + alpha / 2 ||x_j - c_j X ||_2^2 s.t. c_jj = 0,\n        where c_j and x_j are the j-th rows of C and X, respectively.\n        \n        Parameter ``algorithm`` specifies the algorithm for solving the optimization problem.\n        ``lasso_lars`` and ``lasso_cd`` are algorithms implemented in sklearn, \n        ``spams`` refers to the same algorithm as ``lasso_lars`` but is implemented in \n        spams package available at http://spams-devel.gforge.inria.fr/ (installation required)\n        In principle, all three algorithms give the same result.    \n        For large scale data (e.g. with > 5000 data points), use any of these algorithms in\n        conjunction with ``active_support=True``. It adopts an efficient active support \n        strategy that solves the optimization problem by breaking it into a sequence of \n        small scale optimization problems as described in [1].\n        If tau = 1.0, the method reduces to sparse subspace clustering with basis pursuit (SSC-BP) [2].\n        If tau = 0.0, the method reduces to least squares regression (LSR) [3].\n        Note: ``lasso_lars`` and ``lasso_cd`` only support tau = 1.\n        Parameters\n        -----------\n        X : array-like, shape (n_samples, n_features)\n            Input data to be clustered\n        gamma : float\n        gamma_nz : boolean, default True\n            gamma and gamma_nz together determines the parameter alpha. When ``gamma_nz = False``, \n            alpha = gamma. When ``gamma_nz = True``, then alpha = gamma * alpha0, where alpha0 is \n            the largest number such that the solution to the optimization problem with alpha = alpha0\n            is the zero vector (see Proposition 1 in [1]). Therefore, when ``gamma_nz = True``, gamma\n            should be a value greater than 1.0. A good choice is typically in the range [5, 500].   \n        tau : float, default 1.0\n            Parameter for elastic net penalty term. \n            When tau = 1.0, the method reduces to sparse subspace clustering with basis pursuit (SSC-BP) [2].\n            When tau = 0.0, the method reduces to least squares regression (LSR) [3].\n        algorithm : string, default ``lasso_lars``\n            Algorithm for computing the representation. Either lasso_lars or lasso_cd or spams \n            (installation of spams package is required).\n            Note: ``lasso_lars`` and ``lasso_cd`` only support tau = 1.\n        n_nonzero : int, default 50\n            This is an upper bound on the number of nonzero entries of each representation vector. \n            If there are more than n_nonzero nonzero entries,  only the top n_nonzero number of\n            entries with largest absolute value are kept.\n        active_support: boolean, default True\n            Set to True to use the active support algorithm in [1] for solving the optimization problem.\n            This should significantly reduce the running time when n_samples is large.\n        active_support_params: dictionary of string to any, optional\n            Parameters (keyword arguments) and values for the active support algorithm. It may be\n            used to set the parameters ``support_init``, ``support_size`` and ``maxiter``, see\n            ``active_support_elastic_net`` for details. \n            Example: active_support_params={'support_size':50, 'maxiter':100}\n            Ignored when ``active_support=False``\n        \n        Returns\n        -------\n        representation_matrix_ : csr matrix, shape: n_samples by n_samples\n            The self-representation matrix.\n        \n        References\n        ----------- \n        [1] C. You, C.-G. Li, D. Robinson, R. Vidal, Oracle Based Active Set Algorithm for Scalable Elastic Net Subspace Clustering, CVPR 2016\n        [2] E. Elhaifar, R. Vidal, Sparse Subspace Clustering: Algorithm, Theory, and Applications, TPAMI 2013\n        [3] C. Lu, et al. Robust and efficient subspace segmentation via least squares regression, ECCV 2012\n        "
        if algorithm in ('lasso_lars', 'lasso_cd') and tau < 1.0 - 1e-10:
            warnings.warn('algorithm {} cannot handle tau smaller than 1. Using tau = 1'.format(algorithm))
            tau = 1.0
        if active_support == True and active_support_params == None:
            active_support_params = {}
        n_samples = X.shape[0]
        rows = np.zeros(n_samples * n_nonzero)
        cols = np.zeros(n_samples * n_nonzero)
        vals = np.zeros(n_samples * n_nonzero)
        curr_pos = 0
        gamma_is_zero_notification = False
        for i in range(n_samples):
            if i % 25 == 0 and self.verbose == 1:
                print('{}/{}'.format(i, n_samples))
            y = X[i, :].copy().reshape(1, -1)
            X[i, :] = 0
            if algorithm in ('lasso_lars', 'lasso_cd'):
                if gamma_nz == True:
                    coh = np.delete(np.absolute(np.dot(X, y.T)), i)
                    alpha0 = np.amax(coh) / tau
                    alpha = alpha0 / gamma
                else:
                    alpha = 1.0 / gamma
                if gamma >= 10 ** 4:
                    if gamma_is_zero_notification == False:
                        warnings.warn('Set alpha = 0 i.e. LinearRegression() is used')
                        gamma_is_zero_notification = True
                    alpha = 0
                if alpha == 0:
                    lr = LinearRegression(fit_intercept=fit_intercept_LR)
                    lr.fit(X.T, y[0])
                    c = lr.coef_
                elif active_support == True:
                    c = self.active_support_elastic_net(X, y, alpha, tau, algorithm, **active_support_params)
                else:
                    c = sparse_encode(y, X, algorithm=algorithm, alpha=alpha, max_iter=maxiter_lasso)[0]
            else:
                warnings.warn('algorithm {} not found'.format(algorithm))
            index = np.flatnonzero(c)
            if index.size > n_nonzero:
                index = index[np.argsort(-np.absolute(c[index]))[0:n_nonzero]]
            rows[curr_pos:curr_pos + len(index)] = i
            cols[curr_pos:curr_pos + len(index)] = index
            vals[curr_pos:curr_pos + len(index)] = c[index]
            curr_pos += len(index)
            X[i, :] = y
        return sparse.csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))

    def fit(self, X, y=None):
        if False:
            i = 10
            return i + 15
        'Fit detector. y is ignored in unsupervised methods.\n        Parameters\n        ----------\n        X : numpy array of shape (n_samples, n_features)\n            The input samples.\n        y : Ignored\n            Not used, present for API consistency by convention.\n        Returns\n        -------\n        self : object\n            Fitted estimator.\n        '
        if hasattr(self, 'X_train'):
            del self.X_train
        X = check_array(X)
        if self.preprocessing:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X)
        self._set_n_classes(y)
        self.decision_scores_ = self.decision_function(X)
        self.X_train = X
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        if False:
            i = 10
            return i + 15
        'Predict raw anomaly score of X using the fitted detector.\n\n        The anomaly score of an input sample is computed based on different\n        detector algorithms. For consistency, outliers are assigned with\n        larger anomaly scores.\n\n        Parameters\n        ----------\n        X : numpy array of shape (n_samples, n_features)\n            The training input samples. Sparse matrices are accepted only\n            if they are supported by the base estimator.\n\n        Returns\n        -------\n        anomaly_scores : numpy array of shape (n_samples,)\n            The anomaly score of the input samples.\n        '
        X = check_array(X)
        if hasattr(self, 'X_train'):
            N = int(X.shape[0] / self.blocksize_test_data) + 1
            scores = []
            for i in range(N):
                if self.verbose == 1:
                    print('Test block {}/{}'.format(i, N))
                X_block_i = np.copy(X[i * self.blocksize_test_data:(i + 1) * self.blocksize_test_data])
                if X_block_i.shape[0] >= 1:
                    original_size_i = X_block_i.shape[0]
                    X_i = np.concatenate((self.X_train, X_block_i), axis=0)
                    if self.preprocessing:
                        X_i_norm = self.scaler_.transform(X_i)
                    else:
                        X_i_norm = np.copy(X_i)
                    scores_i = self._decision_function(X_i_norm)
                    scores_i = scores_i[-original_size_i:]
                    scores.extend(list(scores_i))
            scores = np.array(scores)
            return scores
        else:
            if self.preprocessing:
                X_norm = self.scaler_.transform(X)
            else:
                X_norm = np.copy(X)
            scores = self._decision_function(X_norm)
            return scores

    def _decision_function(self, X_norm):
        if False:
            return 10
        A = self.elastic_net_subspace_clustering(X_norm, gamma=self.gamma, gamma_nz=self.gamma_nz, tau=self.tau, algorithm=self.algorithm, fit_intercept_LR=self.fit_intercept_LR, active_support=self.active_support, n_nonzero=self.n_nonzero, maxiter_lasso=self.maxiter_lasso, active_support_params={'support_init': self.support_init, 'support_size': self.support_size, 'maxiter': self.maxiter})
        self.transition_matrix_ = normalize(np.abs(A.toarray()), norm='l1')
        pi = np.ones((1, len(self.transition_matrix_)), dtype='float64') / len(self.transition_matrix_)
        pi_bar = np.zeros((1, len(self.transition_matrix_)), dtype='float64')
        for _ in range(self.transition_steps):
            pi = pi @ self.transition_matrix_
            pi_bar += pi
        pi_bar /= self.transition_steps
        scores = pi_bar[0]
        scores = -1 * scores
        return scores