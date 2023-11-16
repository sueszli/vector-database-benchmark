"""
Linear mixed effects models are regression models for dependent data.
They can be used to estimate regression relationships involving both
means and variances.

These models are also known as multilevel linear models, and
hierarchical linear models.

The MixedLM class fits linear mixed effects models to data, and
provides support for some common post-estimation tasks.  This is a
group-based implementation that is most efficient for models in which
the data can be partitioned into independent groups.  Some models with
crossed effects can be handled by specifying a model with a single
group.

The data are partitioned into disjoint groups.  The probability model
for group i is:

Y = X*beta + Z*gamma + epsilon

where

* n_i is the number of observations in group i

* Y is a n_i dimensional response vector (called endog in MixedLM)

* X is a n_i x k_fe dimensional design matrix for the fixed effects
  (called exog in MixedLM)

* beta is a k_fe-dimensional vector of fixed effects parameters
  (called fe_params in MixedLM)

* Z is a design matrix for the random effects with n_i rows (called
  exog_re in MixedLM).  The number of columns in Z can vary by group
  as discussed below.

* gamma is a random vector with mean 0.  The covariance matrix for the
  first `k_re` elements of `gamma` (called cov_re in MixedLM) is
  common to all groups.  The remaining elements of `gamma` are
  variance components as discussed in more detail below. Each group
  receives its own independent realization of gamma.

* epsilon is a n_i dimensional vector of iid normal
  errors with mean 0 and variance sigma^2; the epsilon
  values are independent both within and between groups

Y, X and Z must be entirely observed.  beta, Psi, and sigma^2 are
estimated using ML or REML estimation, and gamma and epsilon are
random so define the probability model.

The marginal mean structure is E[Y | X, Z] = X*beta.  If only the mean
structure is of interest, GEE is an alternative to using linear mixed
models.

Two types of random effects are supported.  Standard random effects
are correlated with each other in arbitrary ways.  Every group has the
same number (`k_re`) of standard random effects, with the same joint
distribution (but with independent realizations across the groups).

Variance components are uncorrelated with each other, and with the
standard random effects.  Each variance component has mean zero, and
all realizations of a given variance component have the same variance
parameter.  The number of realized variance components per variance
parameter can differ across the groups.

The primary reference for the implementation details is:

MJ Lindstrom, DM Bates (1988).  "Newton Raphson and EM algorithms for
linear mixed effects models for repeated measures data".  Journal of
the American Statistical Association. Volume 83, Issue 404, pages
1014-1022.

See also this more recent document:

http://econ.ucsb.edu/~doug/245a/Papers/Mixed%20Effects%20Implement.pdf

All the likelihood, gradient, and Hessian calculations closely follow
Lindstrom and Bates 1988, adapted to support variance components.

The following two documents are written more from the perspective of
users:

http://lme4.r-forge.r-project.org/lMMwR/lrgprt.pdf

http://lme4.r-forge.r-project.org/slides/2009-07-07-Rennes/3Longitudinal-4.pdf

Notation:

* `cov_re` is the random effects covariance matrix (referred to above
  as Psi) and `scale` is the (scalar) error variance.  For a single
  group, the marginal covariance matrix of endog given exog is scale*I
  + Z * cov_re * Z', where Z is the design matrix for the random
  effects in one group.

* `vcomp` is a vector of variance parameters.  The length of `vcomp`
  is determined by the number of keys in either the `exog_vc` argument
  to ``MixedLM``, or the `vc_formula` argument when using formulas to
  fit a model.

Notes:

1. Three different parameterizations are used in different places.
The regression slopes (usually called `fe_params`) are identical in
all three parameterizations, but the variance parameters differ.  The
parameterizations are:

* The "user parameterization" in which cov(endog) = scale*I + Z *
  cov_re * Z', as described above.  This is the main parameterization
  visible to the user.

* The "profile parameterization" in which cov(endog) = I +
  Z * cov_re1 * Z'.  This is the parameterization of the profile
  likelihood that is maximized to produce parameter estimates.
  (see Lindstrom and Bates for details).  The "user" cov_re is
  equal to the "profile" cov_re1 times the scale.

* The "square root parameterization" in which we work with the Cholesky
  factor of cov_re1 instead of cov_re directly.  This is hidden from the
  user.

All three parameterizations can be packed into a vector by
(optionally) concatenating `fe_params` together with the lower
triangle or Cholesky square root of the dependence structure, followed
by the variance parameters for the variance components.  The are
stored as square roots if (and only if) the random effects covariance
matrix is stored as its Cholesky factor.  Note that when unpacking, it
is important to either square or reflect the dependence structure
depending on which parameterization is being used.

Two score methods are implemented.  One takes the score with respect
to the elements of the random effects covariance matrix (used for
inference once the MLE is reached), and the other takes the score with
respect to the parameters of the Cholesky square root of the random
effects covariance matrix (used for optimization).

The numerical optimization uses GLS to avoid explicitly optimizing
over the fixed effects parameters.  The likelihood that is optimized
is profiled over both the scale parameter (a scalar) and the fixed
effects parameters (if any).  As a result of this profiling, it is
difficult and unnecessary to calculate the Hessian of the profiled log
likelihood function, so that calculation is not implemented here.
Therefore, optimization methods requiring the Hessian matrix such as
the Newton-Raphson algorithm cannot be used for model fitting.
"""
import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.stats.distributions import norm
from statsmodels.base._penalties import Penalty
import statsmodels.base.model as base
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
_warn_cov_sing = 'The random effects covariance matrix is singular.'

def _dot(x, y):
    if False:
        return 10
    '\n    Returns the dot product of the arrays, works for sparse and dense.\n    '
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return np.dot(x, y)
    elif sparse.issparse(x):
        return x.dot(y)
    elif sparse.issparse(y):
        return y.T.dot(x.T).T

def _multi_dot_three(A, B, C):
    if False:
        for i in range(10):
            print('nop')
    '\n    Find best ordering for three arrays and do the multiplication.\n\n    Doing in manually instead of using dynamic programing is\n    approximately 15 times faster.\n    '
    cost1 = A.shape[0] * A.shape[1] * B.shape[1] + A.shape[0] * B.shape[1] * C.shape[1]
    cost2 = B.shape[0] * B.shape[1] * C.shape[1] + A.shape[0] * A.shape[1] * C.shape[1]
    if cost1 < cost2:
        return _dot(_dot(A, B), C)
    else:
        return _dot(A, _dot(B, C))

def _dotsum(x, y):
    if False:
        while True:
            i = 10
    "\n    Returns sum(x * y), where '*' is the pointwise product, computed\n    efficiently for dense and sparse matrices.\n    "
    if sparse.issparse(x):
        return x.multiply(y).sum()
    else:
        return np.dot(x.ravel(), y.ravel())

class VCSpec:
    """
    Define the variance component structure of a multilevel model.

    An instance of the class contains three attributes:

    - names : names[k] is the name of variance component k.

    - mats : mats[k][i] is the design matrix for group index
      i in variance component k.

    - colnames : colnames[k][i] is the list of column names for
      mats[k][i].

    The groups in colnames and mats must be in sorted order.
    """

    def __init__(self, names, colnames, mats):
        if False:
            i = 10
            return i + 15
        self.names = names
        self.colnames = colnames
        self.mats = mats

def _get_exog_re_names(self, exog_re):
    if False:
        print('Hello World!')
    '\n    Passes through if given a list of names. Otherwise, gets pandas names\n    or creates some generic variable names as needed.\n    '
    if self.k_re == 0:
        return []
    if isinstance(exog_re, pd.DataFrame):
        return exog_re.columns.tolist()
    elif isinstance(exog_re, pd.Series) and exog_re.name is not None:
        return [exog_re.name]
    elif isinstance(exog_re, list):
        return exog_re
    defnames = ['x_re{0:1d}'.format(k + 1) for k in range(exog_re.shape[1])]
    return defnames

class MixedLMParams:
    """
    This class represents a parameter state for a mixed linear model.

    Parameters
    ----------
    k_fe : int
        The number of covariates with fixed effects.
    k_re : int
        The number of covariates with random coefficients (excluding
        variance components).
    k_vc : int
        The number of variance components parameters.

    Notes
    -----
    This object represents the parameter state for the model in which
    the scale parameter has been profiled out.
    """

    def __init__(self, k_fe, k_re, k_vc):
        if False:
            return 10
        self.k_fe = k_fe
        self.k_re = k_re
        self.k_re2 = k_re * (k_re + 1) // 2
        self.k_vc = k_vc
        self.k_tot = self.k_fe + self.k_re2 + self.k_vc
        self._ix = np.tril_indices(self.k_re)

    def from_packed(params, k_fe, k_re, use_sqrt, has_fe):
        if False:
            i = 10
            return i + 15
        '\n        Create a MixedLMParams object from packed parameter vector.\n\n        Parameters\n        ----------\n        params : array_like\n            The mode parameters packed into a single vector.\n        k_fe : int\n            The number of covariates with fixed effects\n        k_re : int\n            The number of covariates with random effects (excluding\n            variance components).\n        use_sqrt : bool\n            If True, the random effects covariance matrix is provided\n            as its Cholesky factor, otherwise the lower triangle of\n            the covariance matrix is stored.\n        has_fe : bool\n            If True, `params` contains fixed effects parameters.\n            Otherwise, the fixed effects parameters are set to zero.\n\n        Returns\n        -------\n        A MixedLMParams object.\n        '
        k_re2 = int(k_re * (k_re + 1) / 2)
        if has_fe:
            k_vc = len(params) - k_fe - k_re2
        else:
            k_vc = len(params) - k_re2
        pa = MixedLMParams(k_fe, k_re, k_vc)
        cov_re = np.zeros((k_re, k_re))
        ix = pa._ix
        if has_fe:
            pa.fe_params = params[0:k_fe]
            cov_re[ix] = params[k_fe:k_fe + k_re2]
        else:
            pa.fe_params = np.zeros(k_fe)
            cov_re[ix] = params[0:k_re2]
        if use_sqrt:
            cov_re = np.dot(cov_re, cov_re.T)
        else:
            cov_re = cov_re + cov_re.T - np.diag(np.diag(cov_re))
        pa.cov_re = cov_re
        if k_vc > 0:
            if use_sqrt:
                pa.vcomp = params[-k_vc:] ** 2
            else:
                pa.vcomp = params[-k_vc:]
        else:
            pa.vcomp = np.array([])
        return pa
    from_packed = staticmethod(from_packed)

    def from_components(fe_params=None, cov_re=None, cov_re_sqrt=None, vcomp=None):
        if False:
            print('Hello World!')
        '\n        Create a MixedLMParams object from each parameter component.\n\n        Parameters\n        ----------\n        fe_params : array_like\n            The fixed effects parameter (a 1-dimensional array).  If\n            None, there are no fixed effects.\n        cov_re : array_like\n            The random effects covariance matrix (a square, symmetric\n            2-dimensional array).\n        cov_re_sqrt : array_like\n            The Cholesky (lower triangular) square root of the random\n            effects covariance matrix.\n        vcomp : array_like\n            The variance component parameters.  If None, there are no\n            variance components.\n\n        Returns\n        -------\n        A MixedLMParams object.\n        '
        if vcomp is None:
            vcomp = np.empty(0)
        if fe_params is None:
            fe_params = np.empty(0)
        if cov_re is None and cov_re_sqrt is None:
            cov_re = np.empty((0, 0))
        k_fe = len(fe_params)
        k_vc = len(vcomp)
        k_re = cov_re.shape[0] if cov_re is not None else cov_re_sqrt.shape[0]
        pa = MixedLMParams(k_fe, k_re, k_vc)
        pa.fe_params = fe_params
        if cov_re_sqrt is not None:
            pa.cov_re = np.dot(cov_re_sqrt, cov_re_sqrt.T)
        elif cov_re is not None:
            pa.cov_re = cov_re
        pa.vcomp = vcomp
        return pa
    from_components = staticmethod(from_components)

    def copy(self):
        if False:
            return 10
        '\n        Returns a copy of the object.\n        '
        obj = MixedLMParams(self.k_fe, self.k_re, self.k_vc)
        obj.fe_params = self.fe_params.copy()
        obj.cov_re = self.cov_re.copy()
        obj.vcomp = self.vcomp.copy()
        return obj

    def get_packed(self, use_sqrt, has_fe=False):
        if False:
            while True:
                i = 10
        '\n        Return the model parameters packed into a single vector.\n\n        Parameters\n        ----------\n        use_sqrt : bool\n            If True, the Cholesky square root of `cov_re` is\n            included in the packed result.  Otherwise the\n            lower triangle of `cov_re` is included.\n        has_fe : bool\n            If True, the fixed effects parameters are included\n            in the packed result, otherwise they are omitted.\n        '
        if self.k_re > 0:
            if use_sqrt:
                try:
                    L = np.linalg.cholesky(self.cov_re)
                except np.linalg.LinAlgError:
                    L = np.diag(np.sqrt(np.diag(self.cov_re)))
                cpa = L[self._ix]
            else:
                cpa = self.cov_re[self._ix]
        else:
            cpa = np.zeros(0)
        if use_sqrt:
            vcomp = np.sqrt(self.vcomp)
        else:
            vcomp = self.vcomp
        if has_fe:
            pa = np.concatenate((self.fe_params, cpa, vcomp))
        else:
            pa = np.concatenate((cpa, vcomp))
        return pa

def _smw_solver(s, A, AtA, Qi, di):
    if False:
        while True:
            i = 10
    '\n    Returns a solver for the linear system:\n\n    .. math::\n\n        (sI + ABA^\\prime) y = x\n\n    The returned function f satisfies f(x) = y as defined above.\n\n    B and its inverse matrix are block diagonal.  The upper left block\n    of :math:`B^{-1}` is Qi and its lower right block is diag(di).\n\n    Parameters\n    ----------\n    s : scalar\n        See above for usage\n    A : ndarray\n        p x q matrix, in general q << p, may be sparse.\n    AtA : square ndarray\n        :math:`A^\\prime  A`, a q x q matrix.\n    Qi : square symmetric ndarray\n        The matrix `B` is q x q, where q = r + d.  `B` consists of a r\n        x r diagonal block whose inverse is `Qi`, and a d x d diagonal\n        block, whose inverse is diag(di).\n    di : 1d array_like\n        See documentation for Qi.\n\n    Returns\n    -------\n    A function for solving a linear system, as documented above.\n\n    Notes\n    -----\n    Uses Sherman-Morrison-Woodbury identity:\n        https://en.wikipedia.org/wiki/Woodbury_matrix_identity\n    '
    qmat = AtA / s
    m = Qi.shape[0]
    qmat[0:m, 0:m] += Qi
    if sparse.issparse(A):
        qmat[m:, m:] += sparse.diags(di)

        def solver(rhs):
            if False:
                for i in range(10):
                    print('nop')
            ql = A.T.dot(rhs)
            ql = sparse.linalg.spsolve(qmat, ql)
            if ql.ndim < rhs.ndim:
                ql = ql[:, None]
            ql = A.dot(ql)
            return rhs / s - ql / s ** 2
    else:
        d = qmat.shape[0]
        qmat.flat[m * (d + 1)::d + 1] += di
        qmati = np.linalg.solve(qmat, A.T)

        def solver(rhs):
            if False:
                for i in range(10):
                    print('nop')
            ql = np.dot(qmati, rhs)
            ql = np.dot(A, ql)
            return rhs / s - ql / s ** 2
    return solver

def _smw_logdet(s, A, AtA, Qi, di, B_logdet):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns the log determinant of\n\n    .. math::\n\n        sI + ABA^\\prime\n\n    Uses the matrix determinant lemma to accelerate the calculation.\n    B is assumed to be positive definite, and s > 0, therefore the\n    determinant is positive.\n\n    Parameters\n    ----------\n    s : positive scalar\n        See above for usage\n    A : ndarray\n        p x q matrix, in general q << p.\n    AtA : square ndarray\n        :math:`A^\\prime  A`, a q x q matrix.\n    Qi : square symmetric ndarray\n        The matrix `B` is q x q, where q = r + d.  `B` consists of a r\n        x r diagonal block whose inverse is `Qi`, and a d x d diagonal\n        block, whose inverse is diag(di).\n    di : 1d array_like\n        See documentation for Qi.\n    B_logdet : real\n        The log determinant of B\n\n    Returns\n    -------\n    The log determinant of s*I + A*B*A'.\n\n    Notes\n    -----\n    Uses the matrix determinant lemma:\n        https://en.wikipedia.org/wiki/Matrix_determinant_lemma\n    "
    p = A.shape[0]
    ld = p * np.log(s)
    qmat = AtA / s
    m = Qi.shape[0]
    qmat[0:m, 0:m] += Qi
    if sparse.issparse(qmat):
        qmat[m:, m:] += sparse.diags(di)
        lu = sparse.linalg.splu(qmat)
        dl = lu.L.diagonal().astype(np.complex128)
        du = lu.U.diagonal().astype(np.complex128)
        ld1 = np.log(dl).sum() + np.log(du).sum()
        ld1 = ld1.real
    else:
        d = qmat.shape[0]
        qmat.flat[m * (d + 1)::d + 1] += di
        (_, ld1) = np.linalg.slogdet(qmat)
    return B_logdet + ld + ld1

def _convert_vc(exog_vc):
    if False:
        i = 10
        return i + 15
    vc_names = []
    vc_colnames = []
    vc_mats = []
    groups = set()
    for (k, v) in exog_vc.items():
        groups |= set(v.keys())
    groups = list(groups)
    groups.sort()
    for (k, v) in exog_vc.items():
        vc_names.append(k)
        (colnames, mats) = ([], [])
        for g in groups:
            try:
                colnames.append(v[g].columns)
            except AttributeError:
                colnames.append([str(j) for j in range(v[g].shape[1])])
            mats.append(v[g])
        vc_colnames.append(colnames)
        vc_mats.append(mats)
    ii = np.argsort(vc_names)
    vc_names = [vc_names[i] for i in ii]
    vc_colnames = [vc_colnames[i] for i in ii]
    vc_mats = [vc_mats[i] for i in ii]
    return VCSpec(vc_names, vc_colnames, vc_mats)

class MixedLM(base.LikelihoodModel):
    """
    Linear Mixed Effects Model

    Parameters
    ----------
    endog : 1d array_like
        The dependent variable
    exog : 2d array_like
        A matrix of covariates used to determine the
        mean structure (the "fixed effects" covariates).
    groups : 1d array_like
        A vector of labels determining the groups -- data from
        different groups are independent
    exog_re : 2d array_like
        A matrix of covariates used to determine the variance and
        covariance structure (the "random effects" covariates).  If
        None, defaults to a random intercept for each group.
    exog_vc : VCSpec instance or dict-like (deprecated)
        A VCSPec instance defines the structure of the variance
        components in the model.  Alternatively, see notes below
        for a dictionary-based format.  The dictionary format is
        deprecated and may be removed at some point in the future.
    use_sqrt : bool
        If True, optimization is carried out using the lower
        triangle of the square root of the random effects
        covariance matrix, otherwise it is carried out using the
        lower triangle of the random effects covariance matrix.
    missing : str
        The approach to missing data handling

    Notes
    -----
    If `exog_vc` is not a `VCSpec` instance, then it must be a
    dictionary of dictionaries.  Specifically, `exog_vc[a][g]` is a
    matrix whose columns are linearly combined using independent
    random coefficients.  This random term then contributes to the
    variance structure of the data for group `g`.  The random
    coefficients all have mean zero, and have the same variance.  The
    matrix must be `m x k`, where `m` is the number of observations in
    group `g`.  The number of columns may differ among the top-level
    groups.

    The covariates in `exog`, `exog_re` and `exog_vc` may (but need
    not) partially or wholly overlap.

    `use_sqrt` should almost always be set to True.  The main use case
    for use_sqrt=False is when complicated patterns of fixed values in
    the covariance structure are set (using the `free` argument to
    `fit`) that cannot be expressed in terms of the Cholesky factor L.

    Examples
    --------
    A basic mixed model with fixed effects for the columns of
    ``exog`` and a random intercept for each distinct value of
    ``group``:

    >>> model = sm.MixedLM(endog, exog, groups)
    >>> result = model.fit()

    A mixed model with fixed effects for the columns of ``exog`` and
    correlated random coefficients for the columns of ``exog_re``:

    >>> model = sm.MixedLM(endog, exog, groups, exog_re=exog_re)
    >>> result = model.fit()

    A mixed model with fixed effects for the columns of ``exog`` and
    independent random coefficients for the columns of ``exog_re``:

    >>> free = MixedLMParams.from_components(
                     fe_params=np.ones(exog.shape[1]),
                     cov_re=np.eye(exog_re.shape[1]))
    >>> model = sm.MixedLM(endog, exog, groups, exog_re=exog_re)
    >>> result = model.fit(free=free)

    A different way to specify independent random coefficients for the
    columns of ``exog_re``.  In this example ``groups`` must be a
    Pandas Series with compatible indexing with ``exog_re``, and
    ``exog_re`` has two columns.

    >>> g = pd.groupby(groups, by=groups).groups
    >>> vc = {}
    >>> vc['1'] = {k : exog_re.loc[g[k], 0] for k in g}
    >>> vc['2'] = {k : exog_re.loc[g[k], 1] for k in g}
    >>> model = sm.MixedLM(endog, exog, groups, vcomp=vc)
    >>> result = model.fit()
    """

    def __init__(self, endog, exog, groups, exog_re=None, exog_vc=None, use_sqrt=True, missing='none', **kwargs):
        if False:
            return 10
        _allowed_kwargs = ['missing_idx', 'design_info', 'formula']
        for x in kwargs.keys():
            if x not in _allowed_kwargs:
                raise ValueError('argument %s not permitted for MixedLM initialization' % x)
        self.use_sqrt = use_sqrt
        self.reml = True
        self.fe_pen = None
        self.re_pen = None
        if isinstance(exog_vc, dict):
            warnings.warn('Using deprecated variance components format')
            exog_vc = _convert_vc(exog_vc)
        if exog_vc is not None:
            self.k_vc = len(exog_vc.names)
            self.exog_vc = exog_vc
        else:
            self.k_vc = 0
            self.exog_vc = VCSpec([], [], [])
        if exog is not None and data_tools._is_using_ndarray_type(exog, None) and (exog.ndim == 1):
            exog = exog[:, None]
        if exog_re is not None and data_tools._is_using_ndarray_type(exog_re, None) and (exog_re.ndim == 1):
            exog_re = exog_re[:, None]
        super(MixedLM, self).__init__(endog, exog, groups=groups, exog_re=exog_re, missing=missing, **kwargs)
        self._init_keys.extend(['use_sqrt', 'exog_vc'])
        self.k_fe = exog.shape[1]
        if exog_re is None and len(self.exog_vc.names) == 0:
            self.k_re = 1
            self.k_re2 = 1
            self.exog_re = np.ones((len(endog), 1), dtype=np.float64)
            self.data.exog_re = self.exog_re
            names = ['Group Var']
            self.data.param_names = self.exog_names + names
            self.data.exog_re_names = names
            self.data.exog_re_names_full = names
        elif exog_re is not None:
            self.data.exog_re = exog_re
            self.exog_re = np.asarray(exog_re)
            if self.exog_re.ndim == 1:
                self.exog_re = self.exog_re[:, None]
            self.k_re = self.exog_re.shape[1]
            self.k_re2 = self.k_re * (self.k_re + 1) // 2
        else:
            self.k_re = 0
            self.k_re2 = 0
        if not self.data._param_names:
            (param_names, exog_re_names, exog_re_names_full) = self._make_param_names(exog_re)
            self.data.param_names = param_names
            self.data.exog_re_names = exog_re_names
            self.data.exog_re_names_full = exog_re_names_full
        self.k_params = self.k_fe + self.k_re2
        group_labels = list(set(groups))
        group_labels.sort()
        row_indices = dict(((s, []) for s in group_labels))
        for (i, g) in enumerate(groups):
            row_indices[g].append(i)
        self.row_indices = row_indices
        self.group_labels = group_labels
        self.n_groups = len(self.group_labels)
        self.endog_li = self.group_list(self.endog)
        self.exog_li = self.group_list(self.exog)
        self.exog_re_li = self.group_list(self.exog_re)
        if self.exog_re is None:
            self.exog_re2_li = None
        else:
            self.exog_re2_li = [np.dot(x.T, x) for x in self.exog_re_li]
        self.nobs = len(self.endog)
        self.n_totobs = self.nobs
        if self.exog_names is None:
            self.exog_names = ['FE%d' % (k + 1) for k in range(self.exog.shape[1])]
        self._aex_r = []
        self._aex_r2 = []
        for i in range(self.n_groups):
            a = self._augment_exog(i)
            self._aex_r.append(a)
            ma = _dot(a.T, a)
            self._aex_r2.append(ma)
        (self._lin, self._quad) = self._reparam()

    def _make_param_names(self, exog_re):
        if False:
            while True:
                i = 10
        '\n        Returns the full parameter names list, just the exogenous random\n        effects variables, and the exogenous random effects variables with\n        the interaction terms.\n        '
        exog_names = list(self.exog_names)
        exog_re_names = _get_exog_re_names(self, exog_re)
        param_names = []
        jj = self.k_fe
        for i in range(len(exog_re_names)):
            for j in range(i + 1):
                if i == j:
                    param_names.append(exog_re_names[i] + ' Var')
                else:
                    param_names.append(exog_re_names[j] + ' x ' + exog_re_names[i] + ' Cov')
                jj += 1
        vc_names = [x + ' Var' for x in self.exog_vc.names]
        return (exog_names + param_names + vc_names, exog_re_names, param_names)

    @classmethod
    def from_formula(cls, formula, data, re_formula=None, vc_formula=None, subset=None, use_sparse=False, missing='none', *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a Model from a formula and dataframe.\n\n        Parameters\n        ----------\n        formula : str or generic Formula object\n            The formula specifying the model\n        data : array_like\n            The data for the model. See Notes.\n        re_formula : str\n            A one-sided formula defining the variance structure of the\n            model.  The default gives a random intercept for each\n            group.\n        vc_formula : dict-like\n            Formulas describing variance components.  `vc_formula[vc]` is\n            the formula for the component with variance parameter named\n            `vc`.  The formula is processed into a matrix, and the columns\n            of this matrix are linearly combined with independent random\n            coefficients having mean zero and a common variance.\n        subset : array_like\n            An array-like object of booleans, integers, or index\n            values that indicate the subset of df to use in the\n            model. Assumes df is a `pandas.DataFrame`\n        missing : str\n            Either \'none\' or \'drop\'\n        args : extra arguments\n            These are passed to the model\n        kwargs : extra keyword arguments\n            These are passed to the model with one exception. The\n            ``eval_env`` keyword is passed to patsy. It can be either a\n            :class:`patsy:patsy.EvalEnvironment` object or an integer\n            indicating the depth of the namespace to use. For example, the\n            default ``eval_env=0`` uses the calling namespace. If you wish\n            to use a "clean" environment set ``eval_env=-1``.\n\n        Returns\n        -------\n        model : Model instance\n\n        Notes\n        -----\n        `data` must define __getitem__ with the keys in the formula\n        terms args and kwargs are passed on to the model\n        instantiation. E.g., a numpy structured or rec array, a\n        dictionary, or a pandas DataFrame.\n\n        If the variance component is intended to produce random\n        intercepts for disjoint subsets of a group, specified by\n        string labels or a categorical data value, always use \'0 +\' in\n        the formula so that no overall intercept is included.\n\n        If the variance components specify random slopes and you do\n        not also want a random group-level intercept in the model,\n        then use \'0 +\' in the formula to exclude the intercept.\n\n        The variance components formulas are processed separately for\n        each group.  If a variable is categorical the results will not\n        be affected by whether the group labels are distinct or\n        re-used over the top-level groups.\n\n        Examples\n        --------\n        Suppose we have data from an educational study with students\n        nested in classrooms nested in schools.  The students take a\n        test, and we want to relate the test scores to the students\'\n        ages, while accounting for the effects of classrooms and\n        schools.  The school will be the top-level group, and the\n        classroom is a nested group that is specified as a variance\n        component.  Note that the schools may have different number of\n        classrooms, and the classroom labels may (but need not be)\n        different across the schools.\n\n        >>> vc = {\'classroom\': \'0 + C(classroom)\'}\n        >>> MixedLM.from_formula(\'test_score ~ age\', vc_formula=vc,                                   re_formula=\'1\', groups=\'school\', data=data)\n\n        Now suppose we also have a previous test score called\n        \'pretest\'.  If we want the relationship between pretest\n        scores and the current test to vary by classroom, we can\n        specify a random slope for the pretest score\n\n        >>> vc = {\'classroom\': \'0 + C(classroom)\', \'pretest\': \'0 + pretest\'}\n        >>> MixedLM.from_formula(\'test_score ~ age + pretest\', vc_formula=vc,                                   re_formula=\'1\', groups=\'school\', data=data)\n\n        The following model is almost equivalent to the previous one,\n        but here the classroom random intercept and pretest slope may\n        be correlated.\n\n        >>> vc = {\'classroom\': \'0 + C(classroom)\'}\n        >>> MixedLM.from_formula(\'test_score ~ age + pretest\', vc_formula=vc,                                   re_formula=\'1 + pretest\', groups=\'school\',                                   data=data)\n        '
        if 'groups' not in kwargs.keys():
            raise AttributeError("'groups' is a required keyword argument " + 'in MixedLM.from_formula')
        groups = kwargs['groups']
        group_name = 'Group'
        if isinstance(groups, str):
            group_name = groups
            groups = np.asarray(data[groups])
        else:
            groups = np.asarray(groups)
        del kwargs['groups']
        if missing == 'drop':
            (data, groups) = _handle_missing(data, groups, formula, re_formula, vc_formula)
            missing = 'none'
        if re_formula is not None:
            if re_formula.strip() == '1':
                exog_re = np.ones((data.shape[0], 1))
                exog_re_names = [group_name]
            else:
                eval_env = kwargs.get('eval_env', None)
                if eval_env is None:
                    eval_env = 1
                elif eval_env == -1:
                    from patsy import EvalEnvironment
                    eval_env = EvalEnvironment({})
                exog_re = patsy.dmatrix(re_formula, data, eval_env=eval_env)
                exog_re_names = exog_re.design_info.column_names
                exog_re_names = [x.replace('Intercept', group_name) for x in exog_re_names]
                exog_re = np.asarray(exog_re)
            if exog_re.ndim == 1:
                exog_re = exog_re[:, None]
        else:
            exog_re = None
            if vc_formula is None:
                exog_re_names = [group_name]
            else:
                exog_re_names = []
        if vc_formula is not None:
            eval_env = kwargs.get('eval_env', None)
            if eval_env is None:
                eval_env = 1
            elif eval_env == -1:
                from patsy import EvalEnvironment
                eval_env = EvalEnvironment({})
            vc_mats = []
            vc_colnames = []
            vc_names = []
            gb = data.groupby(groups)
            kylist = sorted(gb.groups.keys())
            vcf = sorted(vc_formula.keys())
            for vc_name in vcf:
                md = patsy.ModelDesc.from_formula(vc_formula[vc_name])
                vc_names.append(vc_name)
                (evc_mats, evc_colnames) = ([], [])
                for (group_ix, group) in enumerate(kylist):
                    ii = gb.groups[group]
                    mat = patsy.dmatrix(md, data.loc[ii, :], eval_env=eval_env, return_type='dataframe')
                    evc_colnames.append(mat.columns.tolist())
                    if use_sparse:
                        evc_mats.append(sparse.csr_matrix(mat))
                    else:
                        evc_mats.append(np.asarray(mat))
                vc_mats.append(evc_mats)
                vc_colnames.append(evc_colnames)
            exog_vc = VCSpec(vc_names, vc_colnames, vc_mats)
        else:
            exog_vc = VCSpec([], [], [])
        kwargs['subset'] = None
        kwargs['exog_re'] = exog_re
        kwargs['exog_vc'] = exog_vc
        kwargs['groups'] = groups
        mod = super(MixedLM, cls).from_formula(formula, data, *args, **kwargs)
        (param_names, exog_re_names, exog_re_names_full) = mod._make_param_names(exog_re_names)
        mod.data.param_names = param_names
        mod.data.exog_re_names = exog_re_names
        mod.data.exog_re_names_full = exog_re_names_full
        if vc_formula is not None:
            mod.data.vcomp_names = mod.exog_vc.names
        return mod

    def predict(self, params, exog=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return predicted values from a design matrix.\n\n        Parameters\n        ----------\n        params : array_like\n            Parameters of a mixed linear model.  Can be either a\n            MixedLMParams instance, or a vector containing the packed\n            model parameters in which the fixed effects parameters are\n            at the beginning of the vector, or a vector containing\n            only the fixed effects parameters.\n        exog : array_like, optional\n            Design / exogenous data for the fixed effects. Model exog\n            is used if None.\n\n        Returns\n        -------\n        An array of fitted values.  Note that these predicted values\n        only reflect the fixed effects mean structure of the model.\n        '
        if exog is None:
            exog = self.exog
        if isinstance(params, MixedLMParams):
            params = params.fe_params
        else:
            params = params[0:self.k_fe]
        return np.dot(exog, params)

    def group_list(self, array):
        if False:
            return 10
        '\n        Returns `array` split into subarrays corresponding to the\n        grouping structure.\n        '
        if array is None:
            return None
        if array.ndim == 1:
            return [np.array(array[self.row_indices[k]]) for k in self.group_labels]
        else:
            return [np.array(array[self.row_indices[k], :]) for k in self.group_labels]

    def fit_regularized(self, start_params=None, method='l1', alpha=0, ceps=0.0001, ptol=1e-06, maxit=200, **fit_kwargs):
        if False:
            return 10
        '\n        Fit a model in which the fixed effects parameters are\n        penalized.  The dependence parameters are held fixed at their\n        estimated values in the unpenalized model.\n\n        Parameters\n        ----------\n        method : str of Penalty object\n            Method for regularization.  If a string, must be \'l1\'.\n        alpha : array_like\n            Scalar or vector of penalty weights.  If a scalar, the\n            same weight is applied to all coefficients; if a vector,\n            it contains a weight for each coefficient.  If method is a\n            Penalty object, the weights are scaled by alpha.  For L1\n            regularization, the weights are used directly.\n        ceps : positive real scalar\n            Fixed effects parameters smaller than this value\n            in magnitude are treated as being zero.\n        ptol : positive real scalar\n            Convergence occurs when the sup norm difference\n            between successive values of `fe_params` is less than\n            `ptol`.\n        maxit : int\n            The maximum number of iterations.\n        **fit_kwargs\n            Additional keyword arguments passed to fit.\n\n        Returns\n        -------\n        A MixedLMResults instance containing the results.\n\n        Notes\n        -----\n        The covariance structure is not updated as the fixed effects\n        parameters are varied.\n\n        The algorithm used here for L1 regularization is a"shooting"\n        or cyclic coordinate descent algorithm.\n\n        If method is \'l1\', then `fe_pen` and `cov_pen` are used to\n        obtain the covariance structure, but are ignored during the\n        L1-penalized fitting.\n\n        References\n        ----------\n        Friedman, J. H., Hastie, T. and Tibshirani, R. Regularized\n        Paths for Generalized Linear Models via Coordinate\n        Descent. Journal of Statistical Software, 33(1) (2008)\n        http://www.jstatsoft.org/v33/i01/paper\n\n        http://statweb.stanford.edu/~tibs/stat315a/Supplements/fuse.pdf\n        '
        if isinstance(method, str) and method.lower() != 'l1':
            raise ValueError('Invalid regularization method')
        if isinstance(method, Penalty):
            method.alpha = alpha
            fit_kwargs.update({'fe_pen': method})
            return self.fit(**fit_kwargs)
        if np.isscalar(alpha):
            alpha = alpha * np.ones(self.k_fe, dtype=np.float64)
        mdf = self.fit(**fit_kwargs)
        fe_params = mdf.fe_params
        cov_re = mdf.cov_re
        vcomp = mdf.vcomp
        scale = mdf.scale
        try:
            cov_re_inv = np.linalg.inv(cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = None
        for itr in range(maxit):
            fe_params_s = fe_params.copy()
            for j in range(self.k_fe):
                if abs(fe_params[j]) < ceps:
                    continue
                fe_params[j] = 0.0
                expval = np.dot(self.exog, fe_params)
                resid_all = self.endog - expval
                (a, b) = (0.0, 0.0)
                for (group_ix, group) in enumerate(self.group_labels):
                    vc_var = self._expand_vcomp(vcomp, group_ix)
                    exog = self.exog_li[group_ix]
                    (ex_r, ex2_r) = (self._aex_r[group_ix], self._aex_r2[group_ix])
                    resid = resid_all[self.row_indices[group]]
                    solver = _smw_solver(scale, ex_r, ex2_r, cov_re_inv, 1 / vc_var)
                    x = exog[:, j]
                    u = solver(x)
                    a += np.dot(u, x)
                    b -= 2 * np.dot(u, resid)
                pwt1 = alpha[j]
                if b > pwt1:
                    fe_params[j] = -(b - pwt1) / (2 * a)
                elif b < -pwt1:
                    fe_params[j] = -(b + pwt1) / (2 * a)
            if np.abs(fe_params_s - fe_params).max() < ptol:
                break
        params_prof = mdf.params.copy()
        params_prof[0:self.k_fe] = fe_params
        scale = self.get_scale(fe_params, mdf.cov_re_unscaled, mdf.vcomp)
        (hess, sing) = self.hessian(params_prof)
        if sing:
            warnings.warn(_warn_cov_sing)
        pcov = np.nan * np.ones_like(hess)
        ii = np.abs(params_prof) > ceps
        ii[self.k_fe:] = True
        ii = np.flatnonzero(ii)
        hess1 = hess[ii, :][:, ii]
        pcov[np.ix_(ii, ii)] = np.linalg.inv(-hess1)
        params_object = MixedLMParams.from_components(fe_params, cov_re=cov_re)
        results = MixedLMResults(self, params_prof, pcov / scale)
        results.params_object = params_object
        results.fe_params = fe_params
        results.cov_re = cov_re
        results.vcomp = vcomp
        results.scale = scale
        results.cov_re_unscaled = mdf.cov_re_unscaled
        results.method = mdf.method
        results.converged = True
        results.cov_pen = self.cov_pen
        results.k_fe = self.k_fe
        results.k_re = self.k_re
        results.k_re2 = self.k_re2
        results.k_vc = self.k_vc
        return MixedLMResultsWrapper(results)

    def get_fe_params(self, cov_re, vcomp, tol=1e-10):
        if False:
            print('Hello World!')
        '\n        Use GLS to update the fixed effects parameter estimates.\n\n        Parameters\n        ----------\n        cov_re : array_like (2d)\n            The covariance matrix of the random effects.\n        vcomp : array_like (1d)\n            The variance components.\n        tol : float\n            A tolerance parameter to determine when covariances\n            are singular.\n\n        Returns\n        -------\n        params : ndarray\n            The GLS estimates of the fixed effects parameters.\n        singular : bool\n            True if the covariance is singular\n        '
        if self.k_fe == 0:
            return (np.array([]), False)
        sing = False
        if self.k_re == 0:
            cov_re_inv = np.empty((0, 0))
        else:
            (w, v) = np.linalg.eigh(cov_re)
            if w.min() < tol:
                sing = True
                ii = np.flatnonzero(w >= tol)
                if len(ii) == 0:
                    cov_re_inv = np.zeros_like(cov_re)
                else:
                    vi = v[:, ii]
                    wi = w[ii]
                    cov_re_inv = np.dot(vi / wi, vi.T)
            else:
                cov_re_inv = np.linalg.inv(cov_re)
        if not hasattr(self, '_endex_li'):
            self._endex_li = []
            for (group_ix, _) in enumerate(self.group_labels):
                mat = np.concatenate((self.exog_li[group_ix], self.endog_li[group_ix][:, None]), axis=1)
                self._endex_li.append(mat)
        xtxy = 0.0
        for (group_ix, group) in enumerate(self.group_labels):
            vc_var = self._expand_vcomp(vcomp, group_ix)
            if vc_var.size > 0:
                if vc_var.min() < tol:
                    sing = True
                    ii = np.flatnonzero(vc_var >= tol)
                    vc_vari = np.zeros_like(vc_var)
                    vc_vari[ii] = 1 / vc_var[ii]
                else:
                    vc_vari = 1 / vc_var
            else:
                vc_vari = np.empty(0)
            exog = self.exog_li[group_ix]
            (ex_r, ex2_r) = (self._aex_r[group_ix], self._aex_r2[group_ix])
            solver = _smw_solver(1.0, ex_r, ex2_r, cov_re_inv, vc_vari)
            u = solver(self._endex_li[group_ix])
            xtxy += np.dot(exog.T, u)
        if sing:
            fe_params = np.dot(np.linalg.pinv(xtxy[:, 0:-1]), xtxy[:, -1])
        else:
            fe_params = np.linalg.solve(xtxy[:, 0:-1], xtxy[:, -1])
        return (fe_params, sing)

    def _reparam(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns parameters of the map converting parameters from the\n        form used in optimization to the form returned to the user.\n\n        Returns\n        -------\n        lin : list-like\n            Linear terms of the map\n        quad : list-like\n            Quadratic terms of the map\n\n        Notes\n        -----\n        If P are the standard form parameters and R are the\n        transformed parameters (i.e. with the Cholesky square root\n        covariance and square root transformed variance components),\n        then P[i] = lin[i] * R + R' * quad[i] * R\n        "
        (k_fe, k_re, k_re2, k_vc) = (self.k_fe, self.k_re, self.k_re2, self.k_vc)
        k_tot = k_fe + k_re2 + k_vc
        ix = np.tril_indices(self.k_re)
        lin = []
        for k in range(k_fe):
            e = np.zeros(k_tot)
            e[k] = 1
            lin.append(e)
        for k in range(k_re2):
            lin.append(np.zeros(k_tot))
        for k in range(k_vc):
            lin.append(np.zeros(k_tot))
        quad = []
        for k in range(k_tot):
            quad.append(np.zeros((k_tot, k_tot)))
        ii = np.tril_indices(k_re)
        ix = [(a, b) for (a, b) in zip(ii[0], ii[1])]
        for i1 in range(k_re2):
            for i2 in range(k_re2):
                ix1 = ix[i1]
                ix2 = ix[i2]
                if ix1[1] == ix2[1] and ix1[0] <= ix2[0]:
                    ii = (ix2[0], ix1[0])
                    k = ix.index(ii)
                    quad[k_fe + k][k_fe + i2, k_fe + i1] += 1
        for k in range(k_tot):
            quad[k] = 0.5 * (quad[k] + quad[k].T)
        km = k_fe + k_re2
        for k in range(km, km + k_vc):
            quad[k][k, k] = 1
        return (lin, quad)

    def _expand_vcomp(self, vcomp, group_ix):
        if False:
            for i in range(10):
                print('nop')
        "\n        Replicate variance parameters to match a group's design.\n\n        Parameters\n        ----------\n        vcomp : array_like\n            The variance parameters for the variance components.\n        group_ix : int\n            The group index\n\n        Returns an expanded version of vcomp, in which each variance\n        parameter is copied as many times as there are independent\n        realizations of the variance component in the given group.\n        "
        if len(vcomp) == 0:
            return np.empty(0)
        vc_var = []
        for j in range(len(self.exog_vc.names)):
            d = self.exog_vc.mats[j][group_ix].shape[1]
            vc_var.append(vcomp[j] * np.ones(d))
        if len(vc_var) > 0:
            return np.concatenate(vc_var)
        else:
            return np.empty(0)

    def _augment_exog(self, group_ix):
        if False:
            while True:
                i = 10
        '\n        Concatenate the columns for variance components to the columns\n        for other random effects to obtain a single random effects\n        exog matrix for a given group.\n        '
        ex_r = self.exog_re_li[group_ix] if self.k_re > 0 else None
        if self.k_vc == 0:
            return ex_r
        ex = [ex_r] if self.k_re > 0 else []
        any_sparse = False
        for (j, _) in enumerate(self.exog_vc.names):
            ex.append(self.exog_vc.mats[j][group_ix])
            any_sparse |= sparse.issparse(ex[-1])
        if any_sparse:
            for (j, x) in enumerate(ex):
                if not sparse.issparse(x):
                    ex[j] = sparse.csr_matrix(x)
            ex = sparse.hstack(ex)
            ex = sparse.csr_matrix(ex)
        else:
            ex = np.concatenate(ex, axis=1)
        return ex

    def loglike(self, params, profile_fe=True):
        if False:
            return 10
        '\n        Evaluate the (profile) log-likelihood of the linear mixed\n        effects model.\n\n        Parameters\n        ----------\n        params : MixedLMParams, or array_like.\n            The parameter value.  If array-like, must be a packed\n            parameter vector containing only the covariance\n            parameters.\n        profile_fe : bool\n            If True, replace the provided value of `fe_params` with\n            the GLS estimates.\n\n        Returns\n        -------\n        The log-likelihood value at `params`.\n\n        Notes\n        -----\n        The scale parameter `scale` is always profiled out of the\n        log-likelihood.  In addition, if `profile_fe` is true the\n        fixed effects parameters are also profiled out.\n        '
        if type(params) is not MixedLMParams:
            params = MixedLMParams.from_packed(params, self.k_fe, self.k_re, self.use_sqrt, has_fe=False)
        cov_re = params.cov_re
        vcomp = params.vcomp
        if profile_fe:
            (fe_params, sing) = self.get_fe_params(cov_re, vcomp)
            if sing:
                self._cov_sing += 1
        else:
            fe_params = params.fe_params
        if self.k_re > 0:
            try:
                cov_re_inv = np.linalg.inv(cov_re)
            except np.linalg.LinAlgError:
                cov_re_inv = np.linalg.pinv(cov_re)
                self._cov_sing += 1
            (_, cov_re_logdet) = np.linalg.slogdet(cov_re)
        else:
            cov_re_inv = np.zeros((0, 0))
            cov_re_logdet = 0
        expval = np.dot(self.exog, fe_params)
        resid_all = self.endog - expval
        likeval = 0.0
        if self.cov_pen is not None and self.k_re > 0:
            likeval -= self.cov_pen.func(cov_re, cov_re_inv)
        if self.fe_pen is not None:
            likeval -= self.fe_pen.func(fe_params)
        (xvx, qf) = (0.0, 0.0)
        for (group_ix, group) in enumerate(self.group_labels):
            vc_var = self._expand_vcomp(vcomp, group_ix)
            cov_aug_logdet = cov_re_logdet + np.sum(np.log(vc_var))
            exog = self.exog_li[group_ix]
            (ex_r, ex2_r) = (self._aex_r[group_ix], self._aex_r2[group_ix])
            solver = _smw_solver(1.0, ex_r, ex2_r, cov_re_inv, 1 / vc_var)
            resid = resid_all[self.row_indices[group]]
            ld = _smw_logdet(1.0, ex_r, ex2_r, cov_re_inv, 1 / vc_var, cov_aug_logdet)
            likeval -= ld / 2.0
            u = solver(resid)
            qf += np.dot(resid, u)
            if self.reml:
                mat = solver(exog)
                xvx += np.dot(exog.T, mat)
        if self.reml:
            likeval -= (self.n_totobs - self.k_fe) * np.log(qf) / 2.0
            (_, ld) = np.linalg.slogdet(xvx)
            likeval -= ld / 2.0
            likeval -= (self.n_totobs - self.k_fe) * np.log(2 * np.pi) / 2.0
            likeval += (self.n_totobs - self.k_fe) * np.log(self.n_totobs - self.k_fe) / 2.0
            likeval -= (self.n_totobs - self.k_fe) / 2.0
        else:
            likeval -= self.n_totobs * np.log(qf) / 2.0
            likeval -= self.n_totobs * np.log(2 * np.pi) / 2.0
            likeval += self.n_totobs * np.log(self.n_totobs) / 2.0
            likeval -= self.n_totobs / 2.0
        return likeval

    def _gen_dV_dPar(self, ex_r, solver, group_ix, max_ix=None):
        if False:
            print('Hello World!')
        "\n        A generator that yields the element-wise derivative of the\n        marginal covariance matrix with respect to the random effects\n        variance and covariance parameters.\n\n        ex_r : array_like\n            The random effects design matrix\n        solver : function\n            A function that given x returns V^{-1}x, where V\n            is the group's marginal covariance matrix.\n        group_ix : int\n            The group index\n        max_ix : {int, None}\n            If not None, the generator ends when this index\n            is reached.\n        "
        axr = solver(ex_r)
        jj = 0
        for j1 in range(self.k_re):
            for j2 in range(j1 + 1):
                if max_ix is not None and jj > max_ix:
                    return
                (mat_l, mat_r) = (ex_r[:, j1:j1 + 1], ex_r[:, j2:j2 + 1])
                (vsl, vsr) = (axr[:, j1:j1 + 1], axr[:, j2:j2 + 1])
                yield (jj, mat_l, mat_r, vsl, vsr, j1 == j2)
                jj += 1
        for (j, _) in enumerate(self.exog_vc.names):
            if max_ix is not None and jj > max_ix:
                return
            mat = self.exog_vc.mats[j][group_ix]
            axmat = solver(mat)
            yield (jj, mat, mat, axmat, axmat, True)
            jj += 1

    def score(self, params, profile_fe=True):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns the score vector of the profile log-likelihood.\n\n        Notes\n        -----\n        The score vector that is returned is computed with respect to\n        the parameterization defined by this model instance's\n        `use_sqrt` attribute.\n        "
        if type(params) is not MixedLMParams:
            params = MixedLMParams.from_packed(params, self.k_fe, self.k_re, self.use_sqrt, has_fe=False)
        if profile_fe:
            (params.fe_params, sing) = self.get_fe_params(params.cov_re, params.vcomp)
            if sing:
                msg = 'Random effects covariance is singular'
                warnings.warn(msg)
        if self.use_sqrt:
            (score_fe, score_re, score_vc) = self.score_sqrt(params, calc_fe=not profile_fe)
        else:
            (score_fe, score_re, score_vc) = self.score_full(params, calc_fe=not profile_fe)
        if self._freepat is not None:
            score_fe *= self._freepat.fe_params
            score_re *= self._freepat.cov_re[self._freepat._ix]
            score_vc *= self._freepat.vcomp
        if profile_fe:
            return np.concatenate((score_re, score_vc))
        else:
            return np.concatenate((score_fe, score_re, score_vc))

    def score_full(self, params, calc_fe):
        if False:
            while True:
                i = 10
        '\n        Returns the score with respect to untransformed parameters.\n\n        Calculates the score vector for the profiled log-likelihood of\n        the mixed effects model with respect to the parameterization\n        in which the random effects covariance matrix is represented\n        in its full form (not using the Cholesky factor).\n\n        Parameters\n        ----------\n        params : MixedLMParams or array_like\n            The parameter at which the score function is evaluated.\n            If array-like, must contain the packed random effects\n            parameters (cov_re and vcomp) without fe_params.\n        calc_fe : bool\n            If True, calculate the score vector for the fixed effects\n            parameters.  If False, this vector is not calculated, and\n            a vector of zeros is returned in its place.\n\n        Returns\n        -------\n        score_fe : array_like\n            The score vector with respect to the fixed effects\n            parameters.\n        score_re : array_like\n            The score vector with respect to the random effects\n            parameters (excluding variance components parameters).\n        score_vc : array_like\n            The score vector with respect to variance components\n            parameters.\n\n        Notes\n        -----\n        `score_re` is taken with respect to the parameterization in\n        which `cov_re` is represented through its lower triangle\n        (without taking the Cholesky square root).\n        '
        fe_params = params.fe_params
        cov_re = params.cov_re
        vcomp = params.vcomp
        try:
            cov_re_inv = np.linalg.inv(cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = np.linalg.pinv(cov_re)
            self._cov_sing += 1
        score_fe = np.zeros(self.k_fe)
        score_re = np.zeros(self.k_re2)
        score_vc = np.zeros(self.k_vc)
        if self.cov_pen is not None:
            score_re -= self.cov_pen.deriv(cov_re, cov_re_inv)
        if calc_fe and self.fe_pen is not None:
            score_fe -= self.fe_pen.deriv(fe_params)
        rvir = 0.0
        xtvir = 0.0
        xtvix = 0.0
        xtax = [0.0] * (self.k_re2 + self.k_vc)
        dlv = np.zeros(self.k_re2 + self.k_vc)
        rvavr = np.zeros(self.k_re2 + self.k_vc)
        for (group_ix, group) in enumerate(self.group_labels):
            vc_var = self._expand_vcomp(vcomp, group_ix)
            exog = self.exog_li[group_ix]
            (ex_r, ex2_r) = (self._aex_r[group_ix], self._aex_r2[group_ix])
            solver = _smw_solver(1.0, ex_r, ex2_r, cov_re_inv, 1 / vc_var)
            resid = self.endog_li[group_ix]
            if self.k_fe > 0:
                expval = np.dot(exog, fe_params)
                resid = resid - expval
            if self.reml:
                viexog = solver(exog)
                xtvix += np.dot(exog.T, viexog)
            vir = solver(resid)
            for (jj, matl, matr, vsl, vsr, sym) in self._gen_dV_dPar(ex_r, solver, group_ix):
                dlv[jj] = _dotsum(matr, vsl)
                if not sym:
                    dlv[jj] += _dotsum(matl, vsr)
                ul = _dot(vir, matl)
                ur = ul.T if sym else _dot(matr.T, vir)
                ulr = np.dot(ul, ur)
                rvavr[jj] += ulr
                if not sym:
                    rvavr[jj] += ulr.T
                if self.reml:
                    ul = _dot(viexog.T, matl)
                    ur = ul.T if sym else _dot(matr.T, viexog)
                    ulr = np.dot(ul, ur)
                    xtax[jj] += ulr
                    if not sym:
                        xtax[jj] += ulr.T
            if self.k_re > 0:
                score_re -= 0.5 * dlv[0:self.k_re2]
            if self.k_vc > 0:
                score_vc -= 0.5 * dlv[self.k_re2:]
            rvir += np.dot(resid, vir)
            if calc_fe:
                xtvir += np.dot(exog.T, vir)
        fac = self.n_totobs
        if self.reml:
            fac -= self.k_fe
        if calc_fe and self.k_fe > 0:
            score_fe += fac * xtvir / rvir
        if self.k_re > 0:
            score_re += 0.5 * fac * rvavr[0:self.k_re2] / rvir
        if self.k_vc > 0:
            score_vc += 0.5 * fac * rvavr[self.k_re2:] / rvir
        if self.reml:
            xtvixi = np.linalg.inv(xtvix)
            for j in range(self.k_re2):
                score_re[j] += 0.5 * _dotsum(xtvixi.T, xtax[j])
            for j in range(self.k_vc):
                score_vc[j] += 0.5 * _dotsum(xtvixi.T, xtax[self.k_re2 + j])
        return (score_fe, score_re, score_vc)

    def score_sqrt(self, params, calc_fe=True):
        if False:
            i = 10
            return i + 15
        '\n        Returns the score with respect to transformed parameters.\n\n        Calculates the score vector with respect to the\n        parameterization in which the random effects covariance matrix\n        is represented through its Cholesky square root.\n\n        Parameters\n        ----------\n        params : MixedLMParams or array_like\n            The model parameters.  If array-like must contain packed\n            parameters that are compatible with this model instance.\n        calc_fe : bool\n            If True, calculate the score vector for the fixed effects\n            parameters.  If False, this vector is not calculated, and\n            a vector of zeros is returned in its place.\n\n        Returns\n        -------\n        score_fe : array_like\n            The score vector with respect to the fixed effects\n            parameters.\n        score_re : array_like\n            The score vector with respect to the random effects\n            parameters (excluding variance components parameters).\n        score_vc : array_like\n            The score vector with respect to variance components\n            parameters.\n        '
        (score_fe, score_re, score_vc) = self.score_full(params, calc_fe=calc_fe)
        params_vec = params.get_packed(use_sqrt=True, has_fe=True)
        score_full = np.concatenate((score_fe, score_re, score_vc))
        scr = 0.0
        for i in range(len(params_vec)):
            v = self._lin[i] + 2 * np.dot(self._quad[i], params_vec)
            scr += score_full[i] * v
        score_fe = scr[0:self.k_fe]
        score_re = scr[self.k_fe:self.k_fe + self.k_re2]
        score_vc = scr[self.k_fe + self.k_re2:]
        return (score_fe, score_re, score_vc)

    def hessian(self, params):
        if False:
            return 10
        "\n        Returns the model's Hessian matrix.\n\n        Calculates the Hessian matrix for the linear mixed effects\n        model with respect to the parameterization in which the\n        covariance matrix is represented directly (without square-root\n        transformation).\n\n        Parameters\n        ----------\n        params : MixedLMParams or array_like\n            The model parameters at which the Hessian is calculated.\n            If array-like, must contain the packed parameters in a\n            form that is compatible with this model instance.\n\n        Returns\n        -------\n        hess : 2d ndarray\n            The Hessian matrix, evaluated at `params`.\n        sing : boolean\n            If True, the covariance matrix is singular and a\n            pseudo-inverse is returned.\n        "
        if type(params) is not MixedLMParams:
            params = MixedLMParams.from_packed(params, self.k_fe, self.k_re, use_sqrt=self.use_sqrt, has_fe=True)
        fe_params = params.fe_params
        vcomp = params.vcomp
        cov_re = params.cov_re
        sing = False
        if self.k_re > 0:
            try:
                cov_re_inv = np.linalg.inv(cov_re)
            except np.linalg.LinAlgError:
                cov_re_inv = np.linalg.pinv(cov_re)
                sing = True
        else:
            cov_re_inv = np.empty((0, 0))
        hess_fe = 0.0
        hess_re = np.zeros((self.k_re2 + self.k_vc, self.k_re2 + self.k_vc))
        hess_fere = np.zeros((self.k_re2 + self.k_vc, self.k_fe))
        fac = self.n_totobs
        if self.reml:
            fac -= self.exog.shape[1]
        rvir = 0.0
        xtvix = 0.0
        xtax = [0.0] * (self.k_re2 + self.k_vc)
        m = self.k_re2 + self.k_vc
        B = np.zeros(m)
        D = np.zeros((m, m))
        F = [[0.0] * m for k in range(m)]
        for (group_ix, group) in enumerate(self.group_labels):
            vc_var = self._expand_vcomp(vcomp, group_ix)
            vc_vari = np.zeros_like(vc_var)
            ii = np.flatnonzero(vc_var >= 1e-10)
            if len(ii) > 0:
                vc_vari[ii] = 1 / vc_var[ii]
            if len(ii) < len(vc_var):
                sing = True
            exog = self.exog_li[group_ix]
            (ex_r, ex2_r) = (self._aex_r[group_ix], self._aex_r2[group_ix])
            solver = _smw_solver(1.0, ex_r, ex2_r, cov_re_inv, vc_vari)
            resid = self.endog_li[group_ix]
            if self.k_fe > 0:
                expval = np.dot(exog, fe_params)
                resid = resid - expval
            viexog = solver(exog)
            xtvix += np.dot(exog.T, viexog)
            vir = solver(resid)
            rvir += np.dot(resid, vir)
            for (jj1, matl1, matr1, vsl1, vsr1, sym1) in self._gen_dV_dPar(ex_r, solver, group_ix):
                ul = _dot(viexog.T, matl1)
                ur = _dot(matr1.T, vir)
                hess_fere[jj1, :] += np.dot(ul, ur)
                if not sym1:
                    ul = _dot(viexog.T, matr1)
                    ur = _dot(matl1.T, vir)
                    hess_fere[jj1, :] += np.dot(ul, ur)
                if self.reml:
                    ul = _dot(viexog.T, matl1)
                    ur = ul if sym1 else np.dot(viexog.T, matr1)
                    ulr = _dot(ul, ur.T)
                    xtax[jj1] += ulr
                    if not sym1:
                        xtax[jj1] += ulr.T
                ul = _dot(vir, matl1)
                ur = ul if sym1 else _dot(vir, matr1)
                B[jj1] += np.dot(ul, ur) * (1 if sym1 else 2)
                E = [(vsl1, matr1)]
                if not sym1:
                    E.append((vsr1, matl1))
                for (jj2, matl2, matr2, vsl2, vsr2, sym2) in self._gen_dV_dPar(ex_r, solver, group_ix, jj1):
                    re = sum([_multi_dot_three(matr2.T, x[0], x[1].T) for x in E])
                    vt = 2 * _dot(_multi_dot_three(vir[None, :], matl2, re), vir[:, None])
                    if not sym2:
                        le = sum([_multi_dot_three(matl2.T, x[0], x[1].T) for x in E])
                        vt += 2 * _dot(_multi_dot_three(vir[None, :], matr2, le), vir[:, None])
                    D[jj1, jj2] += np.squeeze(vt)
                    if jj1 != jj2:
                        D[jj2, jj1] += np.squeeze(vt)
                    rt = _dotsum(vsl2, re.T) / 2
                    if not sym2:
                        rt += _dotsum(vsr2, le.T) / 2
                    hess_re[jj1, jj2] += rt
                    if jj1 != jj2:
                        hess_re[jj2, jj1] += rt
                    if self.reml:
                        ev = sum([_dot(x[0], _dot(x[1].T, viexog)) for x in E])
                        u1 = _dot(viexog.T, matl2)
                        u2 = _dot(matr2.T, ev)
                        um = np.dot(u1, u2)
                        F[jj1][jj2] += um + um.T
                        if not sym2:
                            u1 = np.dot(viexog.T, matr2)
                            u2 = np.dot(matl2.T, ev)
                            um = np.dot(u1, u2)
                            F[jj1][jj2] += um + um.T
        hess_fe -= fac * xtvix / rvir
        hess_re = hess_re - 0.5 * fac * (D / rvir - np.outer(B, B) / rvir ** 2)
        hess_fere = -fac * hess_fere / rvir
        if self.reml:
            QL = [np.linalg.solve(xtvix, x) for x in xtax]
            for j1 in range(self.k_re2 + self.k_vc):
                for j2 in range(j1 + 1):
                    a = _dotsum(QL[j1].T, QL[j2])
                    a -= np.trace(np.linalg.solve(xtvix, F[j1][j2]))
                    a *= 0.5
                    hess_re[j1, j2] += a
                    if j1 > j2:
                        hess_re[j2, j1] += a
        m = self.k_fe + self.k_re2 + self.k_vc
        hess = np.zeros((m, m))
        hess[0:self.k_fe, 0:self.k_fe] = hess_fe
        hess[0:self.k_fe, self.k_fe:] = hess_fere.T
        hess[self.k_fe:, 0:self.k_fe] = hess_fere
        hess[self.k_fe:, self.k_fe:] = hess_re
        return (hess, sing)

    def get_scale(self, fe_params, cov_re, vcomp):
        if False:
            print('Hello World!')
        '\n        Returns the estimated error variance based on given estimates\n        of the slopes and random effects covariance matrix.\n\n        Parameters\n        ----------\n        fe_params : array_like\n            The regression slope estimates\n        cov_re : 2d array_like\n            Estimate of the random effects covariance matrix\n        vcomp : array_like\n            Estimate of the variance components\n\n        Returns\n        -------\n        scale : float\n            The estimated error variance.\n        '
        try:
            cov_re_inv = np.linalg.inv(cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = np.linalg.pinv(cov_re)
            warnings.warn(_warn_cov_sing)
        qf = 0.0
        for (group_ix, group) in enumerate(self.group_labels):
            vc_var = self._expand_vcomp(vcomp, group_ix)
            exog = self.exog_li[group_ix]
            (ex_r, ex2_r) = (self._aex_r[group_ix], self._aex_r2[group_ix])
            solver = _smw_solver(1.0, ex_r, ex2_r, cov_re_inv, 1 / vc_var)
            resid = self.endog_li[group_ix]
            if self.k_fe > 0:
                expval = np.dot(exog, fe_params)
                resid = resid - expval
            mat = solver(resid)
            qf += np.dot(resid, mat)
        if self.reml:
            qf /= self.n_totobs - self.k_fe
        else:
            qf /= self.n_totobs
        return qf

    def fit(self, start_params=None, reml=True, niter_sa=0, do_cg=True, fe_pen=None, cov_pen=None, free=None, full_output=False, method=None, **fit_kwargs):
        if False:
            print('Hello World!')
        '\n        Fit a linear mixed model to the data.\n\n        Parameters\n        ----------\n        start_params : array_like or MixedLMParams\n            Starting values for the profile log-likelihood.  If not a\n            `MixedLMParams` instance, this should be an array\n            containing the packed parameters for the profile\n            log-likelihood, including the fixed effects\n            parameters.\n        reml : bool\n            If true, fit according to the REML likelihood, else\n            fit the standard likelihood using ML.\n        niter_sa : int\n            Currently this argument is ignored and has no effect\n            on the results.\n        cov_pen : CovariancePenalty object\n            A penalty for the random effects covariance matrix\n        do_cg : bool, defaults to True\n            If False, the optimization is skipped and a results\n            object at the given (or default) starting values is\n            returned.\n        fe_pen : Penalty object\n            A penalty on the fixed effects\n        free : MixedLMParams object\n            If not `None`, this is a mask that allows parameters to be\n            held fixed at specified values.  A 1 indicates that the\n            corresponding parameter is estimated, a 0 indicates that\n            it is fixed at its starting value.  Setting the `cov_re`\n            component to the identity matrix fits a model with\n            independent random effects.  Note that some optimization\n            methods do not respect this constraint (bfgs and lbfgs both\n            work).\n        full_output : bool\n            If true, attach iteration history to results\n        method : str\n            Optimization method.  Can be a scipy.optimize method name,\n            or a list of such names to be tried in sequence.\n        **fit_kwargs\n            Additional keyword arguments passed to fit.\n\n        Returns\n        -------\n        A MixedLMResults instance.\n        '
        _allowed_kwargs = ['gtol', 'maxiter', 'eps', 'maxcor', 'ftol', 'tol', 'disp', 'maxls']
        for x in fit_kwargs.keys():
            if x not in _allowed_kwargs:
                warnings.warn('Argument %s not used by MixedLM.fit' % x)
        if method is None:
            method = ['bfgs', 'lbfgs', 'cg']
        elif isinstance(method, str):
            method = [method]
        for meth in method:
            if meth.lower() in ['newton', 'ncg']:
                raise ValueError('method %s not available for MixedLM' % meth)
        self.reml = reml
        self.cov_pen = cov_pen
        self.fe_pen = fe_pen
        self._cov_sing = 0
        self._freepat = free
        if full_output:
            hist = []
        else:
            hist = None
        if start_params is None:
            params = MixedLMParams(self.k_fe, self.k_re, self.k_vc)
            params.fe_params = np.zeros(self.k_fe)
            params.cov_re = np.eye(self.k_re)
            params.vcomp = np.ones(self.k_vc)
        elif isinstance(start_params, MixedLMParams):
            params = start_params
        elif len(start_params) == self.k_fe + self.k_re2 + self.k_vc:
            params = MixedLMParams.from_packed(start_params, self.k_fe, self.k_re, self.use_sqrt, has_fe=True)
        elif len(start_params) == self.k_re2 + self.k_vc:
            params = MixedLMParams.from_packed(start_params, self.k_fe, self.k_re, self.use_sqrt, has_fe=False)
        else:
            raise ValueError('invalid start_params')
        if do_cg:
            fit_kwargs['retall'] = hist is not None
            if 'disp' not in fit_kwargs:
                fit_kwargs['disp'] = False
            packed = params.get_packed(use_sqrt=self.use_sqrt, has_fe=False)
            if niter_sa > 0:
                warnings.warn('niter_sa is currently ignored')
            for j in range(len(method)):
                rslt = super(MixedLM, self).fit(start_params=packed, skip_hessian=True, method=method[j], **fit_kwargs)
                if rslt.mle_retvals['converged']:
                    break
                packed = rslt.params
                if j + 1 < len(method):
                    next_method = method[j + 1]
                    warnings.warn('Retrying MixedLM optimization with %s' % next_method, ConvergenceWarning)
                else:
                    msg = 'MixedLM optimization failed, ' + 'trying a different optimizer may help.'
                    warnings.warn(msg, ConvergenceWarning)
            params = np.atleast_1d(rslt.params)
            if hist is not None:
                hist.append(rslt.mle_retvals)
        converged = rslt.mle_retvals['converged']
        if not converged:
            gn = self.score(rslt.params)
            gn = np.sqrt(np.sum(gn ** 2))
            msg = 'Gradient optimization failed, |grad| = %f' % gn
            warnings.warn(msg, ConvergenceWarning)
        params = MixedLMParams.from_packed(params, self.k_fe, self.k_re, use_sqrt=self.use_sqrt, has_fe=False)
        cov_re_unscaled = params.cov_re
        vcomp_unscaled = params.vcomp
        (fe_params, sing) = self.get_fe_params(cov_re_unscaled, vcomp_unscaled)
        params.fe_params = fe_params
        scale = self.get_scale(fe_params, cov_re_unscaled, vcomp_unscaled)
        cov_re = scale * cov_re_unscaled
        vcomp = scale * vcomp_unscaled
        f1 = self.k_re > 0 and np.min(np.abs(np.diag(cov_re))) < 0.01
        f2 = self.k_vc > 0 and np.min(np.abs(vcomp)) < 0.01
        if f1 or f2:
            msg = 'The MLE may be on the boundary of the parameter space.'
            warnings.warn(msg, ConvergenceWarning)
        (hess, sing) = self.hessian(params)
        if sing:
            warnings.warn(_warn_cov_sing)
        hess_diag = np.diag(hess)
        if free is not None:
            pcov = np.zeros_like(hess)
            pat = self._freepat.get_packed(use_sqrt=False, has_fe=True)
            ii = np.flatnonzero(pat)
            hess_diag = hess_diag[ii]
            if len(ii) > 0:
                hess1 = hess[np.ix_(ii, ii)]
                pcov[np.ix_(ii, ii)] = np.linalg.inv(-hess1)
        else:
            pcov = np.linalg.inv(-hess)
        if np.any(hess_diag >= 0):
            msg = 'The Hessian matrix at the estimated parameter values ' + 'is not positive definite.'
            warnings.warn(msg, ConvergenceWarning)
        params_packed = params.get_packed(use_sqrt=False, has_fe=True)
        results = MixedLMResults(self, params_packed, pcov / scale)
        results.params_object = params
        results.fe_params = fe_params
        results.cov_re = cov_re
        results.vcomp = vcomp
        results.scale = scale
        results.cov_re_unscaled = cov_re_unscaled
        results.method = 'REML' if self.reml else 'ML'
        results.converged = converged
        results.hist = hist
        results.reml = self.reml
        results.cov_pen = self.cov_pen
        results.k_fe = self.k_fe
        results.k_re = self.k_re
        results.k_re2 = self.k_re2
        results.k_vc = self.k_vc
        results.use_sqrt = self.use_sqrt
        results.freepat = self._freepat
        return MixedLMResultsWrapper(results)

    def get_distribution(self, params, scale, exog):
        if False:
            for i in range(10):
                print('nop')
        return _mixedlm_distribution(self, params, scale, exog)

class _mixedlm_distribution:
    """
    A private class for simulating data from a given mixed linear model.

    Parameters
    ----------
    model : MixedLM instance
        A mixed linear model
    params : array_like
        A parameter vector defining a mixed linear model.  See
        notes for more information.
    scale : scalar
        The unexplained variance
    exog : array_like
        An array of fixed effect covariates.  If None, model.exog
        is used.

    Notes
    -----
    The params array is a vector containing fixed effects parameters,
    random effects parameters, and variance component parameters, in
    that order.  The lower triangle of the random effects covariance
    matrix is stored.  The random effects and variance components
    parameters are divided by the scale parameter.

    This class is used in Mediation, and possibly elsewhere.
    """

    def __init__(self, model, params, scale, exog):
        if False:
            for i in range(10):
                print('nop')
        self.model = model
        self.exog = exog if exog is not None else model.exog
        po = MixedLMParams.from_packed(params, model.k_fe, model.k_re, False, True)
        self.fe_params = po.fe_params
        self.cov_re = scale * po.cov_re
        self.vcomp = scale * po.vcomp
        self.scale = scale
        group_idx = np.zeros(model.nobs, dtype=int)
        for (k, g) in enumerate(model.group_labels):
            group_idx[model.row_indices[g]] = k
        self.group_idx = group_idx

    def rvs(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a vector of simulated values from a mixed linear\n        model.\n\n        The parameter n is ignored, but required by the interface\n        '
        model = self.model
        y = np.dot(self.exog, self.fe_params)
        u = np.random.normal(size=(model.n_groups, model.k_re))
        u = np.dot(u, np.linalg.cholesky(self.cov_re).T)
        y += (u[self.group_idx, :] * model.exog_re).sum(1)
        for (j, _) in enumerate(model.exog_vc.names):
            ex = model.exog_vc.mats[j]
            v = self.vcomp[j]
            for (i, g) in enumerate(model.group_labels):
                exg = ex[i]
                ii = model.row_indices[g]
                u = np.random.normal(size=exg.shape[1])
                y[ii] += np.sqrt(v) * np.dot(exg, u)
        y += np.sqrt(self.scale) * np.random.normal(size=len(y))
        return y

class MixedLMResults(base.LikelihoodModelResults, base.ResultMixin):
    """
    Class to contain results of fitting a linear mixed effects model.

    MixedLMResults inherits from statsmodels.LikelihoodModelResults

    Parameters
    ----------
    See statsmodels.LikelihoodModelResults

    Attributes
    ----------
    model : class instance
        Pointer to MixedLM model instance that called fit.
    normalized_cov_params : ndarray
        The sampling covariance matrix of the estimates
    params : ndarray
        A packed parameter vector for the profile parameterization.
        The first `k_fe` elements are the estimated fixed effects
        coefficients.  The remaining elements are the estimated
        variance parameters.  The variance parameters are all divided
        by `scale` and are not the variance parameters shown
        in the summary.
    fe_params : ndarray
        The fitted fixed-effects coefficients
    cov_re : ndarray
        The fitted random-effects covariance matrix
    bse_fe : ndarray
        The standard errors of the fitted fixed effects coefficients
    bse_re : ndarray
        The standard errors of the fitted random effects covariance
        matrix and variance components.  The first `k_re * (k_re + 1)`
        parameters are the standard errors for the lower triangle of
        `cov_re`, the remaining elements are the standard errors for
        the variance components.

    See Also
    --------
    statsmodels.LikelihoodModelResults
    """

    def __init__(self, model, params, cov_params):
        if False:
            print('Hello World!')
        super(MixedLMResults, self).__init__(model, params, normalized_cov_params=cov_params)
        self.nobs = self.model.nobs
        self.df_resid = self.nobs - np.linalg.matrix_rank(self.model.exog)

    @cache_readonly
    def fittedvalues(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the fitted values for the model.\n\n        The fitted values reflect the mean structure specified by the\n        fixed effects and the predicted random effects.\n        '
        fit = np.dot(self.model.exog, self.fe_params)
        re = self.random_effects
        for (group_ix, group) in enumerate(self.model.group_labels):
            ix = self.model.row_indices[group]
            mat = []
            if self.model.exog_re_li is not None:
                mat.append(self.model.exog_re_li[group_ix])
            for j in range(self.k_vc):
                mat.append(self.model.exog_vc.mats[j][group_ix])
            mat = np.concatenate(mat, axis=1)
            fit[ix] += np.dot(mat, re[group])
        return fit

    @cache_readonly
    def resid(self):
        if False:
            while True:
                i = 10
        '\n        Returns the residuals for the model.\n\n        The residuals reflect the mean structure specified by the\n        fixed effects and the predicted random effects.\n        '
        return self.model.endog - self.fittedvalues

    @cache_readonly
    def bse_fe(self):
        if False:
            return 10
        '\n        Returns the standard errors of the fixed effect regression\n        coefficients.\n        '
        p = self.model.exog.shape[1]
        return np.sqrt(np.diag(self.cov_params())[0:p])

    @cache_readonly
    def bse_re(self):
        if False:
            return 10
        '\n        Returns the standard errors of the variance parameters.\n\n        The first `k_re x (k_re + 1)` elements of the returned array\n        are the standard errors of the lower triangle of `cov_re`.\n        The remaining elements are the standard errors of the variance\n        components.\n\n        Note that the sampling distribution of variance parameters is\n        strongly skewed unless the sample size is large, so these\n        standard errors may not give meaningful confidence intervals\n        or p-values if used in the usual way.\n        '
        p = self.model.exog.shape[1]
        return np.sqrt(self.scale * np.diag(self.cov_params())[p:])

    def _expand_re_names(self, group_ix):
        if False:
            return 10
        names = list(self.model.data.exog_re_names)
        for (j, v) in enumerate(self.model.exog_vc.names):
            vg = self.model.exog_vc.colnames[j][group_ix]
            na = ['%s[%s]' % (v, s) for s in vg]
            names.extend(na)
        return names

    @cache_readonly
    def random_effects(self):
        if False:
            i = 10
            return i + 15
        '\n        The conditional means of random effects given the data.\n\n        Returns\n        -------\n        random_effects : dict\n            A dictionary mapping the distinct `group` values to the\n            conditional means of the random effects for the group\n            given the data.\n        '
        try:
            cov_re_inv = np.linalg.inv(self.cov_re)
        except np.linalg.LinAlgError:
            raise ValueError('Cannot predict random effects from ' + 'singular covariance structure.')
        vcomp = self.vcomp
        k_re = self.k_re
        ranef_dict = {}
        for (group_ix, group) in enumerate(self.model.group_labels):
            endog = self.model.endog_li[group_ix]
            exog = self.model.exog_li[group_ix]
            ex_r = self.model._aex_r[group_ix]
            ex2_r = self.model._aex_r2[group_ix]
            vc_var = self.model._expand_vcomp(vcomp, group_ix)
            resid = endog
            if self.k_fe > 0:
                expval = np.dot(exog, self.fe_params)
                resid = resid - expval
            solver = _smw_solver(self.scale, ex_r, ex2_r, cov_re_inv, 1 / vc_var)
            vir = solver(resid)
            xtvir = _dot(ex_r.T, vir)
            xtvir[0:k_re] = np.dot(self.cov_re, xtvir[0:k_re])
            xtvir[k_re:] *= vc_var
            ranef_dict[group] = pd.Series(xtvir, index=self._expand_re_names(group_ix))
        return ranef_dict

    @cache_readonly
    def random_effects_cov(self):
        if False:
            while True:
                i = 10
        '\n        Returns the conditional covariance matrix of the random\n        effects for each group given the data.\n\n        Returns\n        -------\n        random_effects_cov : dict\n            A dictionary mapping the distinct values of the `group`\n            variable to the conditional covariance matrix of the\n            random effects given the data.\n        '
        try:
            cov_re_inv = np.linalg.inv(self.cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = None
        vcomp = self.vcomp
        ranef_dict = {}
        for group_ix in range(self.model.n_groups):
            ex_r = self.model._aex_r[group_ix]
            ex2_r = self.model._aex_r2[group_ix]
            label = self.model.group_labels[group_ix]
            vc_var = self.model._expand_vcomp(vcomp, group_ix)
            solver = _smw_solver(self.scale, ex_r, ex2_r, cov_re_inv, 1 / vc_var)
            n = ex_r.shape[0]
            m = self.cov_re.shape[0]
            mat1 = np.empty((n, m + len(vc_var)))
            mat1[:, 0:m] = np.dot(ex_r[:, 0:m], self.cov_re)
            mat1[:, m:] = np.dot(ex_r[:, m:], np.diag(vc_var))
            mat2 = solver(mat1)
            mat2 = np.dot(mat1.T, mat2)
            v = -mat2
            v[0:m, 0:m] += self.cov_re
            ix = np.arange(m, v.shape[0])
            v[ix, ix] += vc_var
            na = self._expand_re_names(group_ix)
            v = pd.DataFrame(v, index=na, columns=na)
            ranef_dict[label] = v
        return ranef_dict

    def t_test(self, r_matrix, use_t=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute a t-test for a each linear hypothesis of the form Rb = q\n\n        Parameters\n        ----------\n        r_matrix : array_like\n            If an array is given, a p x k 2d array or length k 1d\n            array specifying the linear restrictions. It is assumed\n            that the linear combination is equal to zero.\n        scale : float, optional\n            An optional `scale` to use.  Default is the scale specified\n            by the model fit.\n        use_t : bool, optional\n            If use_t is None, then the default of the model is used.\n            If use_t is True, then the p-values are based on the t\n            distribution.\n            If use_t is False, then the p-values are based on the normal\n            distribution.\n\n        Returns\n        -------\n        res : ContrastResults instance\n            The results for the test are attributes of this results instance.\n            The available results have the same elements as the parameter table\n            in `summary()`.\n        '
        if r_matrix.shape[1] != self.k_fe:
            raise ValueError('r_matrix for t-test should have %d columns' % self.k_fe)
        d = self.k_re2 + self.k_vc
        z0 = np.zeros((r_matrix.shape[0], d))
        r_matrix = np.concatenate((r_matrix, z0), axis=1)
        tst_rslt = super(MixedLMResults, self).t_test(r_matrix, use_t=use_t)
        return tst_rslt

    def summary(self, yname=None, xname_fe=None, xname_re=None, title=None, alpha=0.05):
        if False:
            for i in range(10):
                print('nop')
        '\n        Summarize the mixed model regression results.\n\n        Parameters\n        ----------\n        yname : str, optional\n            Default is `y`\n        xname_fe : list[str], optional\n            Fixed effects covariate names\n        xname_re : list[str], optional\n            Random effects covariate names\n        title : str, optional\n            Title for the top table. If not None, then this replaces\n            the default title\n        alpha : float\n            significance level for the confidence intervals\n\n        Returns\n        -------\n        smry : Summary instance\n            this holds the summary tables and text, which can be\n            printed or converted to various output formats.\n\n        See Also\n        --------\n        statsmodels.iolib.summary2.Summary : class to hold summary results\n        '
        from statsmodels.iolib import summary2
        smry = summary2.Summary()
        info = {}
        info['Model:'] = 'MixedLM'
        if yname is None:
            yname = self.model.endog_names
        param_names = self.model.data.param_names[:]
        k_fe_params = len(self.fe_params)
        k_re_params = len(param_names) - len(self.fe_params)
        if xname_fe is not None:
            if len(xname_fe) != k_fe_params:
                msg = 'xname_fe should be a list of length %d' % k_fe_params
                raise ValueError(msg)
            param_names[:k_fe_params] = xname_fe
        if xname_re is not None:
            if len(xname_re) != k_re_params:
                msg = 'xname_re should be a list of length %d' % k_re_params
                raise ValueError(msg)
            param_names[k_fe_params:] = xname_re
        info['No. Observations:'] = str(self.model.n_totobs)
        info['No. Groups:'] = str(self.model.n_groups)
        gs = np.array([len(x) for x in self.model.endog_li])
        info['Min. group size:'] = '%.0f' % min(gs)
        info['Max. group size:'] = '%.0f' % max(gs)
        info['Mean group size:'] = '%.1f' % np.mean(gs)
        info['Dependent Variable:'] = yname
        info['Method:'] = self.method
        info['Scale:'] = self.scale
        info['Log-Likelihood:'] = self.llf
        info['Converged:'] = 'Yes' if self.converged else 'No'
        smry.add_dict(info)
        smry.add_title('Mixed Linear Model Regression Results')
        float_fmt = '%.3f'
        sdf = np.nan * np.ones((self.k_fe + self.k_re2 + self.k_vc, 6))
        sdf[0:self.k_fe, 0] = self.fe_params
        sdf[0:self.k_fe, 1] = np.sqrt(np.diag(self.cov_params()[0:self.k_fe]))
        sdf[0:self.k_fe, 2] = sdf[0:self.k_fe, 0] / sdf[0:self.k_fe, 1]
        sdf[0:self.k_fe, 3] = 2 * norm.cdf(-np.abs(sdf[0:self.k_fe, 2]))
        qm = -norm.ppf(alpha / 2)
        sdf[0:self.k_fe, 4] = sdf[0:self.k_fe, 0] - qm * sdf[0:self.k_fe, 1]
        sdf[0:self.k_fe, 5] = sdf[0:self.k_fe, 0] + qm * sdf[0:self.k_fe, 1]
        jj = self.k_fe
        for i in range(self.k_re):
            for j in range(i + 1):
                sdf[jj, 0] = self.cov_re[i, j]
                sdf[jj, 1] = np.sqrt(self.scale) * self.bse[jj]
                jj += 1
        for i in range(self.k_vc):
            sdf[jj, 0] = self.vcomp[i]
            sdf[jj, 1] = np.sqrt(self.scale) * self.bse[jj]
            jj += 1
        sdf = pd.DataFrame(index=param_names, data=sdf)
        sdf.columns = ['Coef.', 'Std.Err.', 'z', 'P>|z|', '[' + str(alpha / 2), str(1 - alpha / 2) + ']']
        for col in sdf.columns:
            sdf[col] = [float_fmt % x if np.isfinite(x) else '' for x in sdf[col]]
        smry.add_df(sdf, align='r')
        return smry

    @cache_readonly
    def llf(self):
        if False:
            for i in range(10):
                print('nop')
        return self.model.loglike(self.params_object, profile_fe=False)

    @cache_readonly
    def aic(self):
        if False:
            for i in range(10):
                print('nop')
        'Akaike information criterion'
        if self.reml:
            return np.nan
        if self.freepat is not None:
            df = self.freepat.get_packed(use_sqrt=False, has_fe=True).sum() + 1
        else:
            df = self.params.size + 1
        return -2 * (self.llf - df)

    @cache_readonly
    def bic(self):
        if False:
            return 10
        'Bayesian information criterion'
        if self.reml:
            return np.nan
        if self.freepat is not None:
            df = self.freepat.get_packed(use_sqrt=False, has_fe=True).sum() + 1
        else:
            df = self.params.size + 1
        return -2 * self.llf + np.log(self.nobs) * df

    def profile_re(self, re_ix, vtype, num_low=5, dist_low=1.0, num_high=5, dist_high=1.0, **fit_kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Profile-likelihood inference for variance parameters.\n\n        Parameters\n        ----------\n        re_ix : int\n            If vtype is `re`, this value is the index of the variance\n            parameter for which to construct a profile likelihood.  If\n            `vtype` is 'vc' then `re_ix` is the name of the variance\n            parameter to be profiled.\n        vtype : str\n            Either 're' or 'vc', depending on whether the profile\n            analysis is for a random effect or a variance component.\n        num_low : int\n            The number of points at which to calculate the likelihood\n            below the MLE of the parameter of interest.\n        dist_low : float\n            The distance below the MLE of the parameter of interest to\n            begin calculating points on the profile likelihood.\n        num_high : int\n            The number of points at which to calculate the likelihood\n            above the MLE of the parameter of interest.\n        dist_high : float\n            The distance above the MLE of the parameter of interest to\n            begin calculating points on the profile likelihood.\n        **fit_kwargs\n            Additional keyword arguments passed to fit.\n\n        Returns\n        -------\n        An array with two columns.  The first column contains the\n        values to which the parameter of interest is constrained.  The\n        second column contains the corresponding likelihood values.\n\n        Notes\n        -----\n        Only variance parameters can be profiled.\n        "
        pmodel = self.model
        k_fe = pmodel.k_fe
        k_re = pmodel.k_re
        k_vc = pmodel.k_vc
        (endog, exog) = (pmodel.endog, pmodel.exog)
        if vtype == 're':
            ix = np.arange(k_re)
            ix[0] = re_ix
            ix[re_ix] = 0
            exog_re = pmodel.exog_re.copy()[:, ix]
            params = self.params_object.copy()
            cov_re_unscaled = params.cov_re
            cov_re_unscaled = cov_re_unscaled[np.ix_(ix, ix)]
            params.cov_re = cov_re_unscaled
            ru0 = cov_re_unscaled[0, 0]
            cov_re = self.scale * cov_re_unscaled
            low = (cov_re[0, 0] - dist_low) / self.scale
            high = (cov_re[0, 0] + dist_high) / self.scale
        elif vtype == 'vc':
            re_ix = self.model.exog_vc.names.index(re_ix)
            params = self.params_object.copy()
            vcomp = self.vcomp
            low = (vcomp[re_ix] - dist_low) / self.scale
            high = (vcomp[re_ix] + dist_high) / self.scale
            ru0 = vcomp[re_ix] / self.scale
        if low <= 0:
            raise ValueError('dist_low is too large and would result in a negative variance. Try a smaller value.')
        left = np.linspace(low, ru0, num_low + 1)
        right = np.linspace(ru0, high, num_high + 1)[1:]
        rvalues = np.concatenate((left, right))
        free = MixedLMParams(k_fe, k_re, k_vc)
        if self.freepat is None:
            free.fe_params = np.ones(k_fe)
            vcomp = np.ones(k_vc)
            mat = np.ones((k_re, k_re))
        else:
            free.fe_params = self.freepat.fe_params
            vcomp = self.freepat.vcomp
            mat = self.freepat.cov_re
            if vtype == 're':
                mat = mat[np.ix_(ix, ix)]
        if vtype == 're':
            mat[0, 0] = 0
        else:
            vcomp[re_ix] = 0
        free.cov_re = mat
        free.vcomp = vcomp
        klass = self.model.__class__
        init_kwargs = pmodel._get_init_kwds()
        if vtype == 're':
            init_kwargs['exog_re'] = exog_re
        likev = []
        for x in rvalues:
            model = klass(endog, exog, **init_kwargs)
            if vtype == 're':
                cov_re = params.cov_re.copy()
                cov_re[0, 0] = x
                params.cov_re = cov_re
            else:
                params.vcomp[re_ix] = x
            rslt = model.fit(start_params=params, free=free, reml=self.reml, cov_pen=self.cov_pen, **fit_kwargs)._results
            likev.append([x * rslt.scale, rslt.llf])
        likev = np.asarray(likev)
        return likev

class MixedLMResultsWrapper(base.LikelihoodResultsWrapper):
    _attrs = {'bse_re': ('generic_columns', 'exog_re_names_full'), 'fe_params': ('generic_columns', 'xnames'), 'bse_fe': ('generic_columns', 'xnames'), 'cov_re': ('generic_columns_2d', 'exog_re_names'), 'cov_re_unscaled': ('generic_columns_2d', 'exog_re_names')}
    _upstream_attrs = base.LikelihoodResultsWrapper._wrap_attrs
    _wrap_attrs = base.wrap.union_dicts(_attrs, _upstream_attrs)
    _methods = {}
    _upstream_methods = base.LikelihoodResultsWrapper._wrap_methods
    _wrap_methods = base.wrap.union_dicts(_methods, _upstream_methods)

def _handle_missing(data, groups, formula, re_formula, vc_formula):
    if False:
        i = 10
        return i + 15
    tokens = set()
    forms = [formula]
    if re_formula is not None:
        forms.append(re_formula)
    if vc_formula is not None:
        forms.extend(vc_formula.values())
    from statsmodels.compat.python import asunicode
    from io import StringIO
    import tokenize
    skiptoks = {'(', ')', '*', ':', '+', '-', '**', '/'}
    for fml in forms:
        rl = StringIO(fml)

        def rlu():
            if False:
                for i in range(10):
                    print('nop')
            line = rl.readline()
            return asunicode(line, 'ascii')
        g = tokenize.generate_tokens(rlu)
        for tok in g:
            if tok not in skiptoks:
                tokens.add(tok.string)
    tokens = sorted(tokens & set(data.columns))
    data = data[tokens]
    ii = pd.notnull(data).all(1)
    if type(groups) is not str:
        ii &= pd.notnull(groups)
    return (data.loc[ii, :], groups[np.asarray(ii)])