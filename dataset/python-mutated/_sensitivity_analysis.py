from __future__ import annotations
import inspect
from dataclasses import dataclass
from typing import Callable, Literal, Protocol, TYPE_CHECKING
import numpy as np
from scipy.stats._common import ConfidenceInterval
from scipy.stats._qmc import check_random_state
from scipy.stats._resampling import BootstrapResult
from scipy.stats import qmc, bootstrap
if TYPE_CHECKING:
    import numpy.typing as npt
    from scipy._lib._util import DecimalNumber, IntNumber, SeedType
__all__ = ['sobol_indices']

def f_ishigami(x: npt.ArrayLike) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    'Ishigami function.\n\n    .. math::\n\n        Y(\\mathbf{x}) = \\sin x_1 + 7 \\sin^2 x_2 + 0.1 x_3^4 \\sin x_1\n\n    with :math:`\\mathbf{x} \\in [-\\pi, \\pi]^3`.\n\n    Parameters\n    ----------\n    x : array_like ([x1, x2, x3], n)\n\n    Returns\n    -------\n    f : array_like (n,)\n        Function evaluation.\n\n    References\n    ----------\n    .. [1] Ishigami, T. and T. Homma. "An importance quantification technique\n       in uncertainty analysis for computer models." IEEE,\n       :doi:`10.1109/ISUMA.1990.151285`, 1990.\n    '
    x = np.atleast_2d(x)
    f_eval = np.sin(x[0]) + 7 * np.sin(x[1]) ** 2 + 0.1 * x[2] ** 4 * np.sin(x[0])
    return f_eval

def sample_A_B(n: IntNumber, dists: list[PPFDist], random_state: SeedType=None) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    'Sample two matrices A and B.\n\n    Uses a Sobol\' sequence with 2`d` columns to have 2 uncorrelated matrices.\n    This is more efficient than using 2 random draw of Sobol\'.\n    See sec. 5 from [1]_.\n\n    Output shape is (d, n).\n\n    References\n    ----------\n    .. [1] Saltelli, A., P. Annoni, I. Azzini, F. Campolongo, M. Ratto, and\n       S. Tarantola. "Variance based sensitivity analysis of model\n       output. Design and estimator for the total sensitivity index."\n       Computer Physics Communications, 181(2):259-270,\n       :doi:`10.1016/j.cpc.2009.09.018`, 2010.\n    '
    d = len(dists)
    A_B = qmc.Sobol(d=2 * d, seed=random_state, bits=64).random(n).T
    A_B = A_B.reshape(2, d, -1)
    try:
        for (d_, dist) in enumerate(dists):
            A_B[:, d_] = dist.ppf(A_B[:, d_])
    except AttributeError as exc:
        message = 'Each distribution in `dists` must have method `ppf`.'
        raise ValueError(message) from exc
    return A_B

def sample_AB(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if False:
        print('Hello World!')
    'AB matrix.\n\n    AB: rows of B into A. Shape (d, d, n).\n    - Copy A into d "pages"\n    - In the first page, replace 1st rows of A with 1st row of B.\n    ...\n    - In the dth page, replace dth row of A with dth row of B.\n    - return the stack of pages\n    '
    (d, n) = A.shape
    AB = np.tile(A, (d, 1, 1))
    i = np.arange(d)
    AB[i, i] = B[i]
    return AB

def saltelli_2010(f_A: np.ndarray, f_B: np.ndarray, f_AB: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if False:
        i = 10
        return i + 15
    'Saltelli2010 formulation.\n\n    .. math::\n\n        S_i = \\frac{1}{N} \\sum_{j=1}^N\n        f(\\mathbf{B})_j (f(\\mathbf{AB}^{(i)})_j - f(\\mathbf{A})_j)\n\n    .. math::\n\n        S_{T_i} = \\frac{1}{N} \\sum_{j=1}^N\n        (f(\\mathbf{A})_j - f(\\mathbf{AB}^{(i)})_j)^2\n\n    Parameters\n    ----------\n    f_A, f_B : array_like (s, n)\n        Function values at A and B, respectively\n    f_AB : array_like (d, s, n)\n        Function values at each of the AB pages\n\n    Returns\n    -------\n    s, st : array_like (s, d)\n        First order and total order Sobol\' indices.\n\n    References\n    ----------\n    .. [1] Saltelli, A., P. Annoni, I. Azzini, F. Campolongo, M. Ratto, and\n       S. Tarantola. "Variance based sensitivity analysis of model\n       output. Design and estimator for the total sensitivity index."\n       Computer Physics Communications, 181(2):259-270,\n       :doi:`10.1016/j.cpc.2009.09.018`, 2010.\n    '
    var = np.var([f_A, f_B], axis=(0, -1))
    s = np.mean(f_B * (f_AB - f_A), axis=-1) / var
    st = 0.5 * np.mean((f_A - f_AB) ** 2, axis=-1) / var
    return (s.T, st.T)

@dataclass
class BootstrapSobolResult:
    first_order: BootstrapResult
    total_order: BootstrapResult

@dataclass
class SobolResult:
    first_order: np.ndarray
    total_order: np.ndarray
    _indices_method: Callable
    _f_A: np.ndarray
    _f_B: np.ndarray
    _f_AB: np.ndarray
    _A: np.ndarray | None = None
    _B: np.ndarray | None = None
    _AB: np.ndarray | None = None
    _bootstrap_result: BootstrapResult | None = None

    def bootstrap(self, confidence_level: DecimalNumber=0.95, n_resamples: IntNumber=999) -> BootstrapSobolResult:
        if False:
            for i in range(10):
                print('nop')
        "Bootstrap Sobol' indices to provide confidence intervals.\n\n        Parameters\n        ----------\n        confidence_level : float, default: ``0.95``\n            The confidence level of the confidence intervals.\n        n_resamples : int, default: ``999``\n            The number of resamples performed to form the bootstrap\n            distribution of the indices.\n\n        Returns\n        -------\n        res : BootstrapSobolResult\n            Bootstrap result containing the confidence intervals and the\n            bootstrap distribution of the indices.\n\n            An object with attributes:\n\n            first_order : BootstrapResult\n                Bootstrap result of the first order indices.\n            total_order : BootstrapResult\n                Bootstrap result of the total order indices.\n            See `BootstrapResult` for more details.\n\n        "

        def statistic(idx):
            if False:
                print('Hello World!')
            f_A_ = self._f_A[:, idx]
            f_B_ = self._f_B[:, idx]
            f_AB_ = self._f_AB[..., idx]
            return self._indices_method(f_A_, f_B_, f_AB_)
        n = self._f_A.shape[1]
        res = bootstrap([np.arange(n)], statistic=statistic, method='BCa', n_resamples=n_resamples, confidence_level=confidence_level, bootstrap_result=self._bootstrap_result)
        self._bootstrap_result = res
        first_order = BootstrapResult(confidence_interval=ConfidenceInterval(res.confidence_interval.low[0], res.confidence_interval.high[0]), bootstrap_distribution=res.bootstrap_distribution[0], standard_error=res.standard_error[0])
        total_order = BootstrapResult(confidence_interval=ConfidenceInterval(res.confidence_interval.low[1], res.confidence_interval.high[1]), bootstrap_distribution=res.bootstrap_distribution[1], standard_error=res.standard_error[1])
        return BootstrapSobolResult(first_order=first_order, total_order=total_order)

class PPFDist(Protocol):

    @property
    def ppf(self) -> Callable[..., float]:
        if False:
            print('Hello World!')
        ...

def sobol_indices(*, func: Callable[[np.ndarray], npt.ArrayLike] | dict[Literal['f_A', 'f_B', 'f_AB'], np.ndarray], n: IntNumber, dists: list[PPFDist] | None=None, method: Callable | Literal['saltelli_2010']='saltelli_2010', random_state: SeedType=None) -> SobolResult:
    if False:
        for i in range(10):
            print('nop')
    'Global sensitivity indices of Sobol\'.\n\n    Parameters\n    ----------\n    func : callable or dict(str, array_like)\n        If `func` is a callable, function to compute the Sobol\' indices from.\n        Its signature must be::\n\n            func(x: ArrayLike) -> ArrayLike\n\n        with ``x`` of shape ``(d, n)`` and output of shape ``(s, n)`` where:\n\n        - ``d`` is the input dimensionality of `func`\n          (number of input variables),\n        - ``s`` is the output dimensionality of `func`\n          (number of output variables), and\n        - ``n`` is the number of samples (see `n` below).\n\n        Function evaluation values must be finite.\n\n        If `func` is a dictionary, contains the function evaluations from three\n        different arrays. Keys must be: ``f_A``, ``f_B`` and ``f_AB``.\n        ``f_A`` and ``f_B`` should have a shape ``(s, n)`` and ``f_AB``\n        should have a shape ``(d, s, n)``.\n        This is an advanced feature and misuse can lead to wrong analysis.\n    n : int\n        Number of samples used to generate the matrices ``A`` and ``B``.\n        Must be a power of 2. The total number of points at which `func` is\n        evaluated will be ``n*(d+2)``.\n    dists : list(distributions), optional\n        List of each parameter\'s distribution. The distribution of parameters\n        depends on the application and should be carefully chosen.\n        Parameters are assumed to be independently distributed, meaning there\n        is no constraint nor relationship between their values.\n\n        Distributions must be an instance of a class with a ``ppf``\n        method.\n\n        Must be specified if `func` is a callable, and ignored otherwise.\n    method : Callable or str, default: \'saltelli_2010\'\n        Method used to compute the first and total Sobol\' indices.\n\n        If a callable, its signature must be::\n\n            func(f_A: np.ndarray, f_B: np.ndarray, f_AB: np.ndarray)\n            -> Tuple[np.ndarray, np.ndarray]\n\n        with ``f_A, f_B`` of shape ``(s, n)`` and ``f_AB`` of shape\n        ``(d, s, n)``.\n        These arrays contain the function evaluations from three different sets\n        of samples.\n        The output is a tuple of the first and total indices with\n        shape ``(s, d)``.\n        This is an advanced feature and misuse can lead to wrong analysis.\n    random_state : {None, int, `numpy.random.Generator`}, optional\n        If `random_state` is an int or None, a new `numpy.random.Generator` is\n        created using ``np.random.default_rng(random_state)``.\n        If `random_state` is already a ``Generator`` instance, then the\n        provided instance is used.\n\n    Returns\n    -------\n    res : SobolResult\n        An object with attributes:\n\n        first_order : ndarray of shape (s, d)\n            First order Sobol\' indices.\n        total_order : ndarray of shape (s, d)\n            Total order Sobol\' indices.\n\n        And method:\n\n        bootstrap(confidence_level: float, n_resamples: int)\n        -> BootstrapSobolResult\n\n            A method providing confidence intervals on the indices.\n            See `scipy.stats.bootstrap` for more details.\n\n            The bootstrapping is done on both first and total order indices,\n            and they are available in `BootstrapSobolResult` as attributes\n            ``first_order`` and ``total_order``.\n\n    Notes\n    -----\n    The Sobol\' method [1]_, [2]_ is a variance-based Sensitivity Analysis which\n    obtains the contribution of each parameter to the variance of the\n    quantities of interest (QoIs; i.e., the outputs of `func`).\n    Respective contributions can be used to rank the parameters and\n    also gauge the complexity of the model by computing the\n    model\'s effective (or mean) dimension.\n\n    .. note::\n\n        Parameters are assumed to be independently distributed. Each\n        parameter can still follow any distribution. In fact, the distribution\n        is very important and should match the real distribution of the\n        parameters.\n\n    It uses a functional decomposition of the variance of the function to\n    explore\n\n    .. math::\n\n        \\mathbb{V}(Y) = \\sum_{i}^{d} \\mathbb{V}_i (Y) + \\sum_{i<j}^{d}\n        \\mathbb{V}_{ij}(Y) + ... + \\mathbb{V}_{1,2,...,d}(Y),\n\n    introducing conditional variances:\n\n    .. math::\n\n        \\mathbb{V}_i(Y) = \\mathbb{\\mathbb{V}}[\\mathbb{E}(Y|x_i)]\n        \\qquad\n        \\mathbb{V}_{ij}(Y) = \\mathbb{\\mathbb{V}}[\\mathbb{E}(Y|x_i x_j)]\n        - \\mathbb{V}_i(Y) - \\mathbb{V}_j(Y),\n\n    Sobol\' indices are expressed as\n\n    .. math::\n\n        S_i = \\frac{\\mathbb{V}_i(Y)}{\\mathbb{V}[Y]}\n        \\qquad\n        S_{ij} =\\frac{\\mathbb{V}_{ij}(Y)}{\\mathbb{V}[Y]}.\n\n    :math:`S_{i}` corresponds to the first-order term which apprises the\n    contribution of the i-th parameter, while :math:`S_{ij}` corresponds to the\n    second-order term which informs about the contribution of interactions\n    between the i-th and the j-th parameters. These equations can be\n    generalized to compute higher order terms; however, they are expensive to\n    compute and their interpretation is complex.\n    This is why only first order indices are provided.\n\n    Total order indices represent the global contribution of the parameters\n    to the variance of the QoI and are defined as:\n\n    .. math::\n\n        S_{T_i} = S_i + \\sum_j S_{ij} + \\sum_{j,k} S_{ijk} + ...\n        = 1 - \\frac{\\mathbb{V}[\\mathbb{E}(Y|x_{\\sim i})]}{\\mathbb{V}[Y]}.\n\n    First order indices sum to at most 1, while total order indices sum to at\n    least 1. If there are no interactions, then first and total order indices\n    are equal, and both first and total order indices sum to 1.\n\n    .. warning::\n\n        Negative Sobol\' values are due to numerical errors. Increasing the\n        number of points `n` should help.\n\n        The number of sample required to have a good analysis increases with\n        the dimensionality of the problem. e.g. for a 3 dimension problem,\n        consider at minima ``n >= 2**12``. The more complex the model is,\n        the more samples will be needed.\n\n        Even for a purely addiditive model, the indices may not sum to 1 due\n        to numerical noise.\n\n    References\n    ----------\n    .. [1] Sobol, I. M.. "Sensitivity analysis for nonlinear mathematical\n       models." Mathematical Modeling and Computational Experiment, 1:407-414,\n       1993.\n    .. [2] Sobol, I. M. (2001). "Global sensitivity indices for nonlinear\n       mathematical models and their Monte Carlo estimates." Mathematics\n       and Computers in Simulation, 55(1-3):271-280,\n       :doi:`10.1016/S0378-4754(00)00270-6`, 2001.\n    .. [3] Saltelli, A. "Making best use of model evaluations to\n       compute sensitivity indices."  Computer Physics Communications,\n       145(2):280-297, :doi:`10.1016/S0010-4655(02)00280-1`, 2002.\n    .. [4] Saltelli, A., M. Ratto, T. Andres, F. Campolongo, J. Cariboni,\n       D. Gatelli, M. Saisana, and S. Tarantola. "Global Sensitivity Analysis.\n       The Primer." 2007.\n    .. [5] Saltelli, A., P. Annoni, I. Azzini, F. Campolongo, M. Ratto, and\n       S. Tarantola. "Variance based sensitivity analysis of model\n       output. Design and estimator for the total sensitivity index."\n       Computer Physics Communications, 181(2):259-270,\n       :doi:`10.1016/j.cpc.2009.09.018`, 2010.\n    .. [6] Ishigami, T. and T. Homma. "An importance quantification technique\n       in uncertainty analysis for computer models." IEEE,\n       :doi:`10.1109/ISUMA.1990.151285`, 1990.\n\n    Examples\n    --------\n    The following is an example with the Ishigami function [6]_\n\n    .. math::\n\n        Y(\\mathbf{x}) = \\sin x_1 + 7 \\sin^2 x_2 + 0.1 x_3^4 \\sin x_1,\n\n    with :math:`\\mathbf{x} \\in [-\\pi, \\pi]^3`. This function exhibits strong\n    non-linearity and non-monotonicity.\n\n    Remember, Sobol\' indices assumes that samples are independently\n    distributed. In this case we use a uniform distribution on each marginals.\n\n    >>> import numpy as np\n    >>> from scipy.stats import sobol_indices, uniform\n    >>> rng = np.random.default_rng()\n    >>> def f_ishigami(x):\n    ...     f_eval = (\n    ...         np.sin(x[0])\n    ...         + 7 * np.sin(x[1])**2\n    ...         + 0.1 * (x[2]**4) * np.sin(x[0])\n    ...     )\n    ...     return f_eval\n    >>> indices = sobol_indices(\n    ...     func=f_ishigami, n=1024,\n    ...     dists=[\n    ...         uniform(loc=-np.pi, scale=2*np.pi),\n    ...         uniform(loc=-np.pi, scale=2*np.pi),\n    ...         uniform(loc=-np.pi, scale=2*np.pi)\n    ...     ],\n    ...     random_state=rng\n    ... )\n    >>> indices.first_order\n    array([0.31637954, 0.43781162, 0.00318825])\n    >>> indices.total_order\n    array([0.56122127, 0.44287857, 0.24229595])\n\n    Confidence interval can be obtained using bootstrapping.\n\n    >>> boot = indices.bootstrap()\n\n    Then, this information can be easily visualized.\n\n    >>> import matplotlib.pyplot as plt\n    >>> fig, axs = plt.subplots(1, 2, figsize=(9, 4))\n    >>> _ = axs[0].errorbar(\n    ...     [1, 2, 3], indices.first_order, fmt=\'o\',\n    ...     yerr=[\n    ...         indices.first_order - boot.first_order.confidence_interval.low,\n    ...         boot.first_order.confidence_interval.high - indices.first_order\n    ...     ],\n    ... )\n    >>> axs[0].set_ylabel("First order Sobol\' indices")\n    >>> axs[0].set_xlabel(\'Input parameters\')\n    >>> axs[0].set_xticks([1, 2, 3])\n    >>> _ = axs[1].errorbar(\n    ...     [1, 2, 3], indices.total_order, fmt=\'o\',\n    ...     yerr=[\n    ...         indices.total_order - boot.total_order.confidence_interval.low,\n    ...         boot.total_order.confidence_interval.high - indices.total_order\n    ...     ],\n    ... )\n    >>> axs[1].set_ylabel("Total order Sobol\' indices")\n    >>> axs[1].set_xlabel(\'Input parameters\')\n    >>> axs[1].set_xticks([1, 2, 3])\n    >>> plt.tight_layout()\n    >>> plt.show()\n\n    .. note::\n\n        By default, `scipy.stats.uniform` has support ``[0, 1]``.\n        Using the parameters ``loc`` and ``scale``, one obtains the uniform\n        distribution on ``[loc, loc + scale]``.\n\n    This result is particularly interesting because the first order index\n    :math:`S_{x_3} = 0` whereas its total order is :math:`S_{T_{x_3}} = 0.244`.\n    This means that higher order interactions with :math:`x_3` are responsible\n    for the difference. Almost 25% of the observed variance\n    on the QoI is due to the correlations between :math:`x_3` and :math:`x_1`,\n    although :math:`x_3` by itself has no impact on the QoI.\n\n    The following gives a visual explanation of Sobol\' indices on this\n    function. Let\'s generate 1024 samples in :math:`[-\\pi, \\pi]^3` and\n    calculate the value of the output.\n\n    >>> from scipy.stats import qmc\n    >>> n_dim = 3\n    >>> p_labels = [\'$x_1$\', \'$x_2$\', \'$x_3$\']\n    >>> sample = qmc.Sobol(d=n_dim, seed=rng).random(1024)\n    >>> sample = qmc.scale(\n    ...     sample=sample,\n    ...     l_bounds=[-np.pi, -np.pi, -np.pi],\n    ...     u_bounds=[np.pi, np.pi, np.pi]\n    ... )\n    >>> output = f_ishigami(sample.T)\n\n    Now we can do scatter plots of the output with respect to each parameter.\n    This gives a visual way to understand how each parameter impacts the\n    output of the function.\n\n    >>> fig, ax = plt.subplots(1, n_dim, figsize=(12, 4))\n    >>> for i in range(n_dim):\n    ...     xi = sample[:, i]\n    ...     ax[i].scatter(xi, output, marker=\'+\')\n    ...     ax[i].set_xlabel(p_labels[i])\n    >>> ax[0].set_ylabel(\'Y\')\n    >>> plt.tight_layout()\n    >>> plt.show()\n\n    Now Sobol\' goes a step further:\n    by conditioning the output value by given values of the parameter\n    (black lines), the conditional output mean is computed. It corresponds to\n    the term :math:`\\mathbb{E}(Y|x_i)`. Taking the variance of this term gives\n    the numerator of the Sobol\' indices.\n\n    >>> mini = np.min(output)\n    >>> maxi = np.max(output)\n    >>> n_bins = 10\n    >>> bins = np.linspace(-np.pi, np.pi, num=n_bins, endpoint=False)\n    >>> dx = bins[1] - bins[0]\n    >>> fig, ax = plt.subplots(1, n_dim, figsize=(12, 4))\n    >>> for i in range(n_dim):\n    ...     xi = sample[:, i]\n    ...     ax[i].scatter(xi, output, marker=\'+\')\n    ...     ax[i].set_xlabel(p_labels[i])\n    ...     for bin_ in bins:\n    ...         idx = np.where((bin_ <= xi) & (xi <= bin_ + dx))\n    ...         xi_ = xi[idx]\n    ...         y_ = output[idx]\n    ...         ave_y_ = np.mean(y_)\n    ...         ax[i].plot([bin_ + dx/2] * 2, [mini, maxi], c=\'k\')\n    ...         ax[i].scatter(bin_ + dx/2, ave_y_, c=\'r\')\n    >>> ax[0].set_ylabel(\'Y\')\n    >>> plt.tight_layout()\n    >>> plt.show()\n\n    Looking at :math:`x_3`, the variance\n    of the mean is zero leading to :math:`S_{x_3} = 0`. But we can further\n    observe that the variance of the output is not constant along the parameter\n    values of :math:`x_3`. This heteroscedasticity is explained by higher order\n    interactions. Moreover, an heteroscedasticity is also noticeable on\n    :math:`x_1` leading to an interaction between :math:`x_3` and :math:`x_1`.\n    On :math:`x_2`, the variance seems to be constant and thus null interaction\n    with this parameter can be supposed.\n\n    This case is fairly simple to analyse visually---although it is only a\n    qualitative analysis. Nevertheless, when the number of input parameters\n    increases such analysis becomes unrealistic as it would be difficult to\n    conclude on high-order terms. Hence the benefit of using Sobol\' indices.\n\n    '
    random_state = check_random_state(random_state)
    n_ = int(n)
    if not n_ & n_ - 1 == 0 or n != n_:
        raise ValueError("The balance properties of Sobol' points require 'n' to be a power of 2.")
    n = n_
    if not callable(method):
        indices_methods: dict[str, Callable] = {'saltelli_2010': saltelli_2010}
        try:
            method = method.lower()
            indices_method_ = indices_methods[method]
        except KeyError as exc:
            message = f"{method!r} is not a valid 'method'. It must be one of {set(indices_methods)!r} or a callable."
            raise ValueError(message) from exc
    else:
        indices_method_ = method
        sig = inspect.signature(indices_method_)
        if set(sig.parameters) != {'f_A', 'f_B', 'f_AB'}:
            message = f"If 'method' is a callable, it must have the following signature: {inspect.signature(saltelli_2010)}"
            raise ValueError(message)

    def indices_method(f_A, f_B, f_AB):
        if False:
            i = 10
            return i + 15
        'Wrap indices method to ensure proper output dimension.\n\n        1D when single output, 2D otherwise.\n        '
        return np.squeeze(indices_method_(f_A=f_A, f_B=f_B, f_AB=f_AB))
    if callable(func):
        if dists is None:
            raise ValueError("'dists' must be defined when 'func' is a callable.")

        def wrapped_func(x):
            if False:
                while True:
                    i = 10
            return np.atleast_2d(func(x))
        (A, B) = sample_A_B(n=n, dists=dists, random_state=random_state)
        AB = sample_AB(A=A, B=B)
        f_A = wrapped_func(A)
        if f_A.shape[1] != n:
            raise ValueError("'func' output should have a shape ``(s, -1)`` with ``s`` the number of output.")

        def funcAB(AB):
            if False:
                i = 10
                return i + 15
            (d, d, n) = AB.shape
            AB = np.moveaxis(AB, 0, -1).reshape(d, n * d)
            f_AB = wrapped_func(AB)
            return np.moveaxis(f_AB.reshape((-1, n, d)), -1, 0)
        f_B = wrapped_func(B)
        f_AB = funcAB(AB)
    else:
        message = "When 'func' is a dictionary, it must contain the following keys: 'f_A', 'f_B' and 'f_AB'.'f_A' and 'f_B' should have a shape ``(s, n)`` and 'f_AB' should have a shape ``(d, s, n)``."
        try:
            (f_A, f_B, f_AB) = np.atleast_2d(func['f_A'], func['f_B'], func['f_AB'])
        except KeyError as exc:
            raise ValueError(message) from exc
        if f_A.shape[1] != n or f_A.shape != f_B.shape or f_AB.shape == f_A.shape or (f_AB.shape[-1] % n != 0):
            raise ValueError(message)
    mean = np.mean([f_A, f_B], axis=(0, -1)).reshape(-1, 1)
    f_A -= mean
    f_B -= mean
    f_AB -= mean
    with np.errstate(divide='ignore', invalid='ignore'):
        (first_order, total_order) = indices_method(f_A=f_A, f_B=f_B, f_AB=f_AB)
    first_order[~np.isfinite(first_order)] = 0
    total_order[~np.isfinite(total_order)] = 0
    res = dict(first_order=first_order, total_order=total_order, _indices_method=indices_method, _f_A=f_A, _f_B=f_B, _f_AB=f_AB)
    if callable(func):
        res.update(dict(_A=A, _B=B, _AB=AB))
    return SobolResult(**res)