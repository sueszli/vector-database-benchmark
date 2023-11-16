"""
"Cholesky Factor Algorithm" (CFA) simulation smoothing for state space models

Author: Chad Fulton
License: BSD-3
"""
import numpy as np
from . import tools

class CFASimulationSmoother:
    """
    "Cholesky Factor Algorithm" (CFA) simulation smoother

    Parameters
    ----------
    model : Representation
        The state space model.

    Notes
    -----
    This class allows simulation smoothing by the "Cholesky Factor Algorithm"
    (CFA) described in [1]_ and [2]_, which essentially takes advantage of the
    existence of an efficient sparse Cholesky factor algorithm for banded
    matrices that are held in a sparse matrix format.

    In particular, this simulation smoother computes the joint posterior mean
    and covariance matrix for the unobserved state vector all at once, rather
    than using the recursive computations of the Kalman filter and smoother. It
    then uses these posterior moments to sample directly from this joint
    posterior. For some models, it can be more computationally efficient than
    the simulation smoother based on the Kalman filter and smoother.

    **Important caveat**:

    However, this simulation smoother cannot be used with all state space
    models, including several of the most popular. In particular, the CFA
    algorithm cannot support degenerate distributions (i.e. positive
    semi-definite covariance matrices) for the initial state (which is the
    prior for the first state) or the observation or state innovations.

    One practical problem with this algorithm is that an autoregressive term
    with order higher than one is typically put into state space form by
    augmenting the states using identities. As identities, these augmenting
    terms will not be subject to random innovations, and so the state
    innovation will be degenerate. It is possible to take these higher order
    terms directly into account when constructing the posterior covariance
    matrix, but this has not yet been implemented.

    Similarly, some state space forms of SARIMA and VARMA models make
    the observation equation an identity, which is not compatible with the CFA
    simulation smoothing approach.

    This simulation smoother has so-far found most of its use with dynamic
    factor and stochastic volatility models, which satisfy the restrictions
    described above.

    **Not-yet-implemented**:

    There are several features that are not yet available with this simulation
    smoother:

    - It does not yet allow diffuse initialization of the state vector.
    - It produces simulated states only for exactly the observations in the
      model (i.e. it cannot produce simulations for a subset of the model
      observations or for observations outside the model).


    References
    ----------
    .. [1] McCausland, William J., Shirley Miller, and Denis Pelletier.
           "Simulation smoothing for state-space models: A computational
           efficiency analysis."
           Computational Statistics & Data Analysis 55, no. 1 (2011): 199-212.
    .. [2] Chan, Joshua CC, and Ivan Jeliazkov.
           "Efficient simulation and integrated likelihood estimation in
           state space models."
           International Journal of Mathematical Modelling and Numerical
           Optimisation 1, no. 1-2 (2009): 101-120.
    """

    def __init__(self, model, cfa_simulation_smoother_classes=None):
        if False:
            for i in range(10):
                print('nop')
        self.model = model
        self.prefix_simulation_smoother_map = cfa_simulation_smoother_classes if cfa_simulation_smoother_classes is not None else tools.prefix_cfa_simulation_smoother_map.copy()
        self._simulation_smoothers = {}
        self._posterior_mean = None
        self._posterior_cov_inv_chol = None
        self._posterior_cov = None
        self._simulated_state = None

    @property
    def _simulation_smoother(self):
        if False:
            print('Hello World!')
        prefix = self.model.prefix
        if prefix in self._simulation_smoothers:
            return self._simulation_smoothers[prefix]
        return None

    @property
    def posterior_mean(self):
        if False:
            return 10
        '\n        Posterior mean of the states conditional on the data\n\n        Notes\n        -----\n\n        .. math::\n\n            \\hat \\alpha_t = E[\\alpha_t \\mid Y^n ]\n\n        This posterior mean is identical to the `smoothed_state` computed by\n        the Kalman smoother.\n        '
        if self._posterior_mean is None:
            self._posterior_mean = np.array(self._simulation_smoother.posterior_mean, copy=True)
        return self._posterior_mean

    @property
    def posterior_cov_inv_chol_sparse(self):
        if False:
            i = 10
            return i + 15
        "\n        Sparse Cholesky factor of inverse posterior covariance matrix\n\n        Notes\n        -----\n        This attribute holds in sparse diagonal banded storage the Cholesky\n        factor of the inverse of the posterior covariance matrix. If we denote\n        :math:`P = Var[\\alpha \\mid Y^n ]`, then the this attribute holds the\n        lower Cholesky factor :math:`L`, defined from :math:`L L' = P^{-1}`.\n        This attribute uses the sparse diagonal banded storage described in the\n        documentation of, for example, the SciPy function\n        `scipy.linalg.solveh_banded`.\n        "
        if self._posterior_cov_inv_chol is None:
            self._posterior_cov_inv_chol = np.array(self._simulation_smoother.posterior_cov_inv_chol, copy=True)
        return self._posterior_cov_inv_chol

    @property
    def posterior_cov(self):
        if False:
            print('Hello World!')
        '\n        Posterior covariance of the states conditional on the data\n\n        Notes\n        -----\n        **Warning**: the matrix computed when accessing this property can be\n        extremely large: it is shaped `(nobs * k_states, nobs * k_states)`. In\n        most cases, it is better to use the `posterior_cov_inv_chol_sparse`\n        property if possible, which holds in sparse diagonal banded storage\n        the Cholesky factor of the inverse of the posterior covariance matrix.\n\n        .. math::\n\n            Var[\\alpha \\mid Y^n ]\n\n        This posterior covariance matrix is *not* identical to the\n        `smoothed_state_cov` attribute produced by the Kalman smoother, because\n        it additionally contains all cross-covariance terms. Instead,\n        `smoothed_state_cov` contains the `(k_states, k_states)` block\n        diagonal entries of this posterior covariance matrix.\n        '
        if self._posterior_cov is None:
            from scipy.linalg import cho_solve_banded
            inv_chol = self.posterior_cov_inv_chol_sparse
            self._posterior_cov = cho_solve_banded((inv_chol, True), np.eye(inv_chol.shape[1]))
        return self._posterior_cov

    def simulate(self, variates=None, update_posterior=True):
        if False:
            print('Hello World!')
        "\n        Perform simulation smoothing (via Cholesky factor algorithm)\n\n        Does not return anything, but populates the object's `simulated_state`\n        attribute, and also makes available the attributes `posterior_mean`,\n        `posterior_cov`, and `posterior_cov_inv_chol_sparse`.\n\n        Parameters\n        ----------\n        variates : array_like, optional\n            Random variates, distributed standard Normal. Usually only\n            specified if results are to be replicated (e.g. to enforce a seed)\n            or for testing. If not specified, random variates are drawn. Must\n            be shaped (nobs, k_states).\n\n        Notes\n        -----\n        The first step in simulating from the joint posterior of the state\n        vector conditional on the data is to compute the two relevant moments\n        of the joint posterior distribution:\n\n        .. math::\n\n            \\alpha \\mid Y_n \\sim N(\\hat \\alpha, Var(\\alpha \\mid Y_n))\n\n        Let :math:`L L' = Var(\\alpha \\mid Y_n)^{-1}`. Then simulation proceeds\n        according to the following steps:\n\n        1. Draw :math:`u \\sim N(0, I)`\n        2. Compute :math:`x = \\hat \\alpha + (L')^{-1} u`\n\n        And then :math:`x` is a draw from the joint posterior of the states.\n        The output of the function is as follows:\n\n        - The simulated draw :math:`x` is held in the `simulated_state`\n          attribute.\n        - The posterior mean :math:`\\hat \\alpha` is held in the\n          `posterior_mean` attribute.\n        - The (lower triangular) Cholesky factor of the inverse posterior\n          covariance matrix, :math:`L`, is held in sparse diagonal banded\n          storage in the `posterior_cov_inv_chol` attribute.\n        - The posterior covariance matrix :math:`Var(\\alpha \\mid Y_n)` can be\n          computed on demand by accessing the `posterior_cov` property. Note\n          that this matrix can be extremely large, so care must be taken when\n          accessing this property. In most cases, it will be preferred to make\n          use of the `posterior_cov_inv_chol` attribute rather than the\n          `posterior_cov` attribute.\n\n        "
        (prefix, dtype, create) = self.model._initialize_representation()
        if variates is not None:
            tools.validate_matrix_shape('variates', variates.shape, self.model.k_states, self.model.nobs, 1)
            variates = np.ravel(variates, order='F').astype(dtype)
        self.model._initialize_state(prefix=prefix)
        if create or prefix not in self._simulation_smoothers:
            cls = self.prefix_simulation_smoother_map[prefix]
            self._simulation_smoothers[prefix] = cls(self.model._statespaces[prefix])
        sim = self._simulation_smoothers[prefix]
        if update_posterior:
            sim.update_sparse_posterior_moments()
            self._posterior_mean = None
            self._posterior_cov_inv_chol = None
            self._posterior_cov = None
        self.simulated_state = sim.simulate(variates=variates)