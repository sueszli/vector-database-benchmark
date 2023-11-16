"""
Benchmarks of Non-Negative Matrix Factorization
"""
import numbers
import sys
import warnings
from time import time
import matplotlib.pyplot as plt
import numpy as np
import pandas
from joblib import Memory
from sklearn.decomposition import NMF
from sklearn.decomposition._nmf import _beta_divergence, _check_init, _initialize_nmf
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import check_array
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.extmath import safe_sparse_dot, squared_norm
from sklearn.utils.validation import check_is_fitted, check_non_negative
mem = Memory(cachedir='.', verbose=0)

def _norm(x):
    if False:
        print('Hello World!')
    'Dot product-based Euclidean norm implementation\n    See: https://fa.bianp.net/blog/2011/computing-the-vector-norm/\n    '
    return np.sqrt(squared_norm(x))

def _nls_subproblem(X, W, H, tol, max_iter, alpha=0.0, l1_ratio=0.0, sigma=0.01, beta=0.1):
    if False:
        for i in range(10):
            print('nop')
    'Non-negative least square solver\n    Solves a non-negative least squares subproblem using the projected\n    gradient descent algorithm.\n    Parameters\n    ----------\n    X : array-like, shape (n_samples, n_features)\n        Constant matrix.\n    W : array-like, shape (n_samples, n_components)\n        Constant matrix.\n    H : array-like, shape (n_components, n_features)\n        Initial guess for the solution.\n    tol : float\n        Tolerance of the stopping condition.\n    max_iter : int\n        Maximum number of iterations before timing out.\n    alpha : double, default: 0.\n        Constant that multiplies the regularization terms. Set it to zero to\n        have no regularization.\n    l1_ratio : double, default: 0.\n        The regularization mixing parameter, with 0 <= l1_ratio <= 1.\n        For l1_ratio = 0 the penalty is an L2 penalty.\n        For l1_ratio = 1 it is an L1 penalty.\n        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.\n    sigma : float\n        Constant used in the sufficient decrease condition checked by the line\n        search.  Smaller values lead to a looser sufficient decrease condition,\n        thus reducing the time taken by the line search, but potentially\n        increasing the number of iterations of the projected gradient\n        procedure. 0.01 is a commonly used value in the optimization\n        literature.\n    beta : float\n        Factor by which the step size is decreased (resp. increased) until\n        (resp. as long as) the sufficient decrease condition is satisfied.\n        Larger values allow to find a better step size but lead to longer line\n        search. 0.1 is a commonly used value in the optimization literature.\n    Returns\n    -------\n    H : array-like, shape (n_components, n_features)\n        Solution to the non-negative least squares problem.\n    grad : array-like, shape (n_components, n_features)\n        The gradient.\n    n_iter : int\n        The number of iterations done by the algorithm.\n    References\n    ----------\n    C.-J. Lin. Projected gradient methods for non-negative matrix\n    factorization. Neural Computation, 19(2007), 2756-2779.\n    https://www.csie.ntu.edu.tw/~cjlin/nmf/\n    '
    WtX = safe_sparse_dot(W.T, X)
    WtW = np.dot(W.T, W)
    gamma = 1
    for n_iter in range(1, max_iter + 1):
        grad = np.dot(WtW, H) - WtX
        if alpha > 0 and l1_ratio == 1.0:
            grad += alpha
        elif alpha > 0:
            grad += alpha * (l1_ratio + (1 - l1_ratio) * H)
        if _norm(grad * np.logical_or(grad < 0, H > 0)) < tol:
            break
        Hp = H
        for inner_iter in range(20):
            Hn = H - gamma * grad
            Hn *= Hn > 0
            d = Hn - H
            gradd = np.dot(grad.ravel(), d.ravel())
            dQd = np.dot(np.dot(WtW, d).ravel(), d.ravel())
            suff_decr = (1 - sigma) * gradd + 0.5 * dQd < 0
            if inner_iter == 0:
                decr_gamma = not suff_decr
            if decr_gamma:
                if suff_decr:
                    H = Hn
                    break
                else:
                    gamma *= beta
            elif not suff_decr or (Hp == Hn).all():
                H = Hp
                break
            else:
                gamma /= beta
                Hp = Hn
    if n_iter == max_iter:
        warnings.warn('Iteration limit reached in nls subproblem.', ConvergenceWarning)
    return (H, grad, n_iter)

def _fit_projected_gradient(X, W, H, tol, max_iter, nls_max_iter, alpha, l1_ratio):
    if False:
        while True:
            i = 10
    gradW = np.dot(W, np.dot(H, H.T)) - safe_sparse_dot(X, H.T, dense_output=True)
    gradH = np.dot(np.dot(W.T, W), H) - safe_sparse_dot(W.T, X, dense_output=True)
    init_grad = squared_norm(gradW) + squared_norm(gradH.T)
    tolW = max(0.001, tol) * np.sqrt(init_grad)
    tolH = tolW
    for n_iter in range(1, max_iter + 1):
        proj_grad_W = squared_norm(gradW * np.logical_or(gradW < 0, W > 0))
        proj_grad_H = squared_norm(gradH * np.logical_or(gradH < 0, H > 0))
        if (proj_grad_W + proj_grad_H) / init_grad < tol ** 2:
            break
        (Wt, gradWt, iterW) = _nls_subproblem(X.T, H.T, W.T, tolW, nls_max_iter, alpha=alpha, l1_ratio=l1_ratio)
        (W, gradW) = (Wt.T, gradWt.T)
        if iterW == 1:
            tolW = 0.1 * tolW
        (H, gradH, iterH) = _nls_subproblem(X, W, H, tolH, nls_max_iter, alpha=alpha, l1_ratio=l1_ratio)
        if iterH == 1:
            tolH = 0.1 * tolH
    H[H == 0] = 0
    if n_iter == max_iter:
        (Wt, _, _) = _nls_subproblem(X.T, H.T, W.T, tolW, nls_max_iter, alpha=alpha, l1_ratio=l1_ratio)
        W = Wt.T
    return (W, H, n_iter)

class _PGNMF(NMF):
    """Non-Negative Matrix Factorization (NMF) with projected gradient solver.

    This class is private and for comparison purpose only.
    It may change or disappear without notice.

    """

    def __init__(self, n_components=None, solver='pg', init=None, tol=0.0001, max_iter=200, random_state=None, alpha=0.0, l1_ratio=0.0, nls_max_iter=10):
        if False:
            print('Hello World!')
        super().__init__(n_components=n_components, init=init, solver=solver, tol=tol, max_iter=max_iter, random_state=random_state, alpha_W=alpha, alpha_H=alpha, l1_ratio=l1_ratio)
        self.nls_max_iter = nls_max_iter

    def fit(self, X, y=None, **params):
        if False:
            for i in range(10):
                print('nop')
        self.fit_transform(X, **params)
        return self

    def transform(self, X):
        if False:
            for i in range(10):
                print('nop')
        check_is_fitted(self)
        H = self.components_
        (W, _, self.n_iter_) = self._fit_transform(X, H=H, update_H=False)
        return W

    def inverse_transform(self, W):
        if False:
            i = 10
            return i + 15
        check_is_fitted(self)
        return np.dot(W, self.components_)

    def fit_transform(self, X, y=None, W=None, H=None):
        if False:
            print('Hello World!')
        (W, H, self.n_iter) = self._fit_transform(X, W=W, H=H, update_H=True)
        self.components_ = H
        return W

    def _fit_transform(self, X, y=None, W=None, H=None, update_H=True):
        if False:
            print('Hello World!')
        X = check_array(X, accept_sparse=('csr', 'csc'))
        check_non_negative(X, 'NMF (input X)')
        (n_samples, n_features) = X.shape
        n_components = self.n_components
        if n_components is None:
            n_components = n_features
        if not isinstance(n_components, numbers.Integral) or n_components <= 0:
            raise ValueError('Number of components must be a positive integer; got (n_components=%r)' % n_components)
        if not isinstance(self.max_iter, numbers.Integral) or self.max_iter < 0:
            raise ValueError('Maximum number of iterations must be a positive integer; got (max_iter=%r)' % self.max_iter)
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError('Tolerance for stopping criteria must be positive; got (tol=%r)' % self.tol)
        if self.init == 'custom' and update_H:
            _check_init(H, (n_components, n_features), 'NMF (input H)')
            _check_init(W, (n_samples, n_components), 'NMF (input W)')
        elif not update_H:
            _check_init(H, (n_components, n_features), 'NMF (input H)')
            W = np.zeros((n_samples, n_components))
        else:
            (W, H) = _initialize_nmf(X, n_components, init=self.init, random_state=self.random_state)
        if update_H:
            (W, H, n_iter) = _fit_projected_gradient(X, W, H, self.tol, self.max_iter, self.nls_max_iter, self.alpha, self.l1_ratio)
        else:
            (Wt, _, n_iter) = _nls_subproblem(X.T, H.T, W.T, self.tol, self.nls_max_iter, alpha=self.alpha, l1_ratio=self.l1_ratio)
            W = Wt.T
        if n_iter == self.max_iter and self.tol > 0:
            warnings.warn('Maximum number of iteration %d reached. Increase it to improve convergence.' % self.max_iter, ConvergenceWarning)
        return (W, H, n_iter)

def plot_results(results_df, plot_name):
    if False:
        return 10
    if results_df is None:
        return None
    plt.figure(figsize=(16, 6))
    colors = 'bgr'
    markers = 'ovs'
    ax = plt.subplot(1, 3, 1)
    for (i, init) in enumerate(np.unique(results_df['init'])):
        plt.subplot(1, 3, i + 1, sharex=ax, sharey=ax)
        for (j, method) in enumerate(np.unique(results_df['method'])):
            mask = np.logical_and(results_df['init'] == init, results_df['method'] == method)
            selected_items = results_df[mask]
            plt.plot(selected_items['time'], selected_items['loss'], color=colors[j % len(colors)], ls='-', marker=markers[j % len(markers)], label=method)
        plt.legend(loc=0, fontsize='x-small')
        plt.xlabel('Time (s)')
        plt.ylabel('loss')
        plt.title('%s' % init)
    plt.suptitle(plot_name, fontsize=16)

@ignore_warnings(category=ConvergenceWarning)
@mem.cache(ignore=['X', 'W0', 'H0'])
def bench_one(name, X, W0, H0, X_shape, clf_type, clf_params, init, n_components, random_state):
    if False:
        for i in range(10):
            print('nop')
    W = W0.copy()
    H = H0.copy()
    clf = clf_type(**clf_params)
    st = time()
    W = clf.fit_transform(X, W=W, H=H)
    end = time()
    H = clf.components_
    this_loss = _beta_divergence(X, W, H, 2.0, True)
    duration = end - st
    return (this_loss, duration)

def run_bench(X, clfs, plot_name, n_components, tol, alpha, l1_ratio):
    if False:
        i = 10
        return i + 15
    start = time()
    results = []
    for (name, clf_type, iter_range, clf_params) in clfs:
        print('Training %s:' % name)
        for (rs, init) in enumerate(('nndsvd', 'nndsvdar', 'random')):
            print('    %s %s: ' % (init, ' ' * (8 - len(init))), end='')
            (W, H) = _initialize_nmf(X, n_components, init, 1e-06, rs)
            for max_iter in iter_range:
                clf_params['alpha'] = alpha
                clf_params['l1_ratio'] = l1_ratio
                clf_params['max_iter'] = max_iter
                clf_params['tol'] = tol
                clf_params['random_state'] = rs
                clf_params['init'] = 'custom'
                clf_params['n_components'] = n_components
                (this_loss, duration) = bench_one(name, X, W, H, X.shape, clf_type, clf_params, init, n_components, rs)
                init_name = "init='%s'" % init
                results.append((name, this_loss, duration, init_name))
                print('.', end='')
                sys.stdout.flush()
            print(' ')
    results_df = pandas.DataFrame(results, columns='method loss time init'.split())
    print('Total time = %0.3f sec\n' % (time() - start))
    plot_results(results_df, plot_name)
    return results_df

def load_20news():
    if False:
        print('Hello World!')
    print('Loading 20 newsgroups dataset')
    print('-----------------------------')
    from sklearn.datasets import fetch_20newsgroups
    dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = vectorizer.fit_transform(dataset.data)
    return tfidf

def load_faces():
    if False:
        for i in range(10):
            print('nop')
    print('Loading Olivetti face dataset')
    print('-----------------------------')
    from sklearn.datasets import fetch_olivetti_faces
    faces = fetch_olivetti_faces(shuffle=True)
    return faces.data

def build_clfs(cd_iters, pg_iters, mu_iters):
    if False:
        print('Hello World!')
    clfs = [('Coordinate Descent', NMF, cd_iters, {'solver': 'cd'}), ('Projected Gradient', _PGNMF, pg_iters, {'solver': 'pg'}), ('Multiplicative Update', NMF, mu_iters, {'solver': 'mu'})]
    return clfs
if __name__ == '__main__':
    alpha = 0.0
    l1_ratio = 0.5
    n_components = 10
    tol = 1e-15
    plot_name = '20 Newsgroups sparse dataset'
    cd_iters = np.arange(1, 30)
    pg_iters = np.arange(1, 6)
    mu_iters = np.arange(1, 30)
    clfs = build_clfs(cd_iters, pg_iters, mu_iters)
    X_20news = load_20news()
    run_bench(X_20news, clfs, plot_name, n_components, tol, alpha, l1_ratio)
    plot_name = 'Olivetti Faces dense dataset'
    cd_iters = np.arange(1, 30)
    pg_iters = np.arange(1, 12)
    mu_iters = np.arange(1, 30)
    clfs = build_clfs(cd_iters, pg_iters, mu_iters)
    X_faces = load_faces()
    run_bench(X_faces, clfs, plot_name, n_components, tol, alpha, l1_ratio)
    plt.show()