from ._analytic_rotation import target_rotation
from ._gpa_rotation import oblimin_objective, orthomax_objective, CF_objective
from ._gpa_rotation import ff_partial_target, ff_target
from ._gpa_rotation import vgQ_partial_target, vgQ_target
from ._gpa_rotation import rotateA, GPA
__all__ = []

def rotate_factors(A, method, *method_args, **algorithm_kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Subroutine for orthogonal and oblique rotation of the matrix :math:`A`.\n    For orthogonal rotations :math:`A` is rotated to :math:`L` according to\n\n    .. math::\n\n        L =  AT,\n\n    where :math:`T` is an orthogonal matrix. And, for oblique rotations\n    :math:`A` is rotated to :math:`L` according to\n\n    .. math::\n\n        L =  A(T^*)^{-1},\n\n    where :math:`T` is a normal matrix.\n\n    Parameters\n    ----------\n    A : numpy matrix (default None)\n        non rotated factors\n    method : str\n        should be one of the methods listed below\n    method_args : list\n        additional arguments that should be provided with each method\n    algorithm_kwargs : dictionary\n        algorithm : str (default gpa)\n            should be one of:\n\n            * 'gpa': a numerical method\n            * 'gpa_der_free': a derivative free numerical method\n            * 'analytic' : an analytic method\n\n        Depending on the algorithm, there are algorithm specific keyword\n        arguments. For the gpa and gpa_der_free, the following\n        keyword arguments are available:\n\n        max_tries : int (default 501)\n            maximum number of iterations\n\n        tol : float\n            stop criterion, algorithm stops if Frobenius norm of gradient is\n            smaller then tol\n\n        For analytic, the supported arguments depend on the method, see above.\n\n        See the lower level functions for more details.\n\n    Returns\n    -------\n    The tuple :math:`(L,T)`\n\n    Notes\n    -----\n    What follows is a list of available methods. Depending on the method\n    additional argument are required and different algorithms\n    are available. The algorithm_kwargs are additional keyword arguments\n    passed to the selected algorithm (see the parameters section).\n    Unless stated otherwise, only the gpa and\n    gpa_der_free algorithm are available.\n\n    Below,\n\n        * :math:`L` is a :math:`p\\times k` matrix;\n        * :math:`N` is :math:`k\\times k` matrix with zeros on the diagonal and ones\n          elsewhere;\n        * :math:`M` is :math:`p\\times p` matrix with zeros on the diagonal and ones\n          elsewhere;\n        * :math:`C` is a :math:`p\\times p` matrix with elements equal to\n          :math:`1/p`;\n        * :math:`(X,Y)=\\operatorname{Tr}(X^*Y)` is the Frobenius norm;\n        * :math:`\\circ` is the element-wise product or Hadamard product.\n\n    oblimin : orthogonal or oblique rotation that minimizes\n        .. math::\n            \\phi(L) = \\frac{1}{4}(L\\circ L,(I-\\gamma C)(L\\circ L)N).\n\n        For orthogonal rotations:\n\n        * :math:`\\gamma=0` corresponds to quartimax,\n        * :math:`\\gamma=\\frac{1}{2}` corresponds to biquartimax,\n        * :math:`\\gamma=1` corresponds to varimax,\n        * :math:`\\gamma=\\frac{1}{p}` corresponds to equamax.\n\n        For oblique rotations rotations:\n\n        * :math:`\\gamma=0` corresponds to quartimin,\n        * :math:`\\gamma=\\frac{1}{2}` corresponds to biquartimin.\n\n        method_args:\n\n        gamma : float\n            oblimin family parameter\n        rotation_method : str\n            should be one of {orthogonal, oblique}\n\n    orthomax : orthogonal rotation that minimizes\n\n        .. math::\n            \\phi(L) = -\\frac{1}{4}(L\\circ L,(I-\\gamma C)(L\\circ L)),\n\n        where :math:`0\\leq\\gamma\\leq1`. The orthomax family is equivalent to\n        the oblimin family (when restricted to orthogonal rotations).\n        Furthermore,\n\n        * :math:`\\gamma=0` corresponds to quartimax,\n        * :math:`\\gamma=\\frac{1}{2}` corresponds to biquartimax,\n        * :math:`\\gamma=1` corresponds to varimax,\n        * :math:`\\gamma=\\frac{1}{p}` corresponds to equamax.\n\n        method_args:\n\n        gamma : float (between 0 and 1)\n            orthomax family parameter\n\n    CF : Crawford-Ferguson family for orthogonal and oblique rotation which\n    minimizes:\n\n        .. math::\n\n            \\phi(L) =\\frac{1-\\kappa}{4} (L\\circ L,(L\\circ L)N)\n                     -\\frac{1}{4}(L\\circ L,M(L\\circ L)),\n\n        where :math:`0\\leq\\kappa\\leq1`. For orthogonal rotations the oblimin\n        (and orthomax) family of rotations is equivalent to the\n        Crawford-Ferguson family.\n        To be more precise:\n\n        * :math:`\\kappa=0` corresponds to quartimax,\n        * :math:`\\kappa=\\frac{1}{p}` corresponds to varimax,\n        * :math:`\\kappa=\\frac{k-1}{p+k-2}` corresponds to parsimax,\n        * :math:`\\kappa=1` corresponds to factor parsimony.\n\n        method_args:\n\n        kappa : float (between 0 and 1)\n            Crawford-Ferguson family parameter\n        rotation_method : str\n            should be one of {orthogonal, oblique}\n\n    quartimax : orthogonal rotation method\n        minimizes the orthomax objective with :math:`\\gamma=0`\n\n    biquartimax : orthogonal rotation method\n        minimizes the orthomax objective with :math:`\\gamma=\\frac{1}{2}`\n\n    varimax : orthogonal rotation method\n        minimizes the orthomax objective with :math:`\\gamma=1`\n\n    equamax : orthogonal rotation method\n        minimizes the orthomax objective with :math:`\\gamma=\\frac{1}{p}`\n\n    parsimax : orthogonal rotation method\n        minimizes the Crawford-Ferguson family objective with\n        :math:`\\kappa=\\frac{k-1}{p+k-2}`\n\n    parsimony : orthogonal rotation method\n        minimizes the Crawford-Ferguson family objective with :math:`\\kappa=1`\n\n    quartimin : oblique rotation method that minimizes\n        minimizes the oblimin objective with :math:`\\gamma=0`\n\n    quartimin : oblique rotation method that minimizes\n        minimizes the oblimin objective with :math:`\\gamma=\\frac{1}{2}`\n\n    target : orthogonal or oblique rotation that rotates towards a target\n\n    matrix : math:`H` by minimizing the objective\n\n        .. math::\n\n            \\phi(L) =\\frac{1}{2}\\|L-H\\|^2.\n\n        method_args:\n\n        H : numpy matrix\n            target matrix\n        rotation_method : str\n            should be one of {orthogonal, oblique}\n\n        For orthogonal rotations the algorithm can be set to analytic in which\n        case the following keyword arguments are available:\n\n        full_rank : bool (default False)\n            if set to true full rank is assumed\n\n    partial_target : orthogonal (default) or oblique rotation that partially\n    rotates towards a target matrix :math:`H` by minimizing the objective:\n\n        .. math::\n\n            \\phi(L) =\\frac{1}{2}\\|W\\circ(L-H)\\|^2.\n\n        method_args:\n\n        H : numpy matrix\n            target matrix\n        W : numpy matrix (default matrix with equal weight one for all entries)\n            matrix with weights, entries can either be one or zero\n\n    Examples\n    --------\n    >>> A = np.random.randn(8,2)\n    >>> L, T = rotate_factors(A,'varimax')\n    >>> np.allclose(L,A.dot(T))\n    >>> L, T = rotate_factors(A,'orthomax',0.5)\n    >>> np.allclose(L,A.dot(T))\n    >>> L, T = rotate_factors(A,'quartimin',0.5)\n    >>> np.allclose(L,A.dot(np.linalg.inv(T.T)))\n    "
    if 'algorithm' in algorithm_kwargs:
        algorithm = algorithm_kwargs['algorithm']
        algorithm_kwargs.pop('algorithm')
    else:
        algorithm = 'gpa'
    assert not 'rotation_method' in algorithm_kwargs, 'rotation_method cannot be provided as keyword argument'
    L = None
    T = None
    ff = None
    vgQ = None
    (p, k) = A.shape
    if method == 'orthomax':
        assert len(method_args) == 1, 'Only %s family parameter should be provided' % method
        rotation_method = 'orthogonal'
        gamma = method_args[0]
        if algorithm == 'gpa':
            vgQ = lambda L=None, A=None, T=None: orthomax_objective(L=L, A=A, T=T, gamma=gamma, return_gradient=True)
        elif algorithm == 'gpa_der_free':
            ff = lambda L=None, A=None, T=None: orthomax_objective(L=L, A=A, T=T, gamma=gamma, return_gradient=False)
        else:
            raise ValueError('Algorithm %s is not possible for %s rotation' % (algorithm, method))
    elif method == 'oblimin':
        assert len(method_args) == 2, 'Both %s family parameter and rotation_method should be provided' % method
        rotation_method = method_args[1]
        assert rotation_method in ['orthogonal', 'oblique'], 'rotation_method should be one of {orthogonal, oblique}'
        gamma = method_args[0]
        if algorithm == 'gpa':
            vgQ = lambda L=None, A=None, T=None: oblimin_objective(L=L, A=A, T=T, gamma=gamma, return_gradient=True)
        elif algorithm == 'gpa_der_free':
            ff = lambda L=None, A=None, T=None: oblimin_objective(L=L, A=A, T=T, gamma=gamma, rotation_method=rotation_method, return_gradient=False)
        else:
            raise ValueError('Algorithm %s is not possible for %s rotation' % (algorithm, method))
    elif method == 'CF':
        assert len(method_args) == 2, 'Both %s family parameter and rotation_method should be provided' % method
        rotation_method = method_args[1]
        assert rotation_method in ['orthogonal', 'oblique'], 'rotation_method should be one of {orthogonal, oblique}'
        kappa = method_args[0]
        if algorithm == 'gpa':
            vgQ = lambda L=None, A=None, T=None: CF_objective(L=L, A=A, T=T, kappa=kappa, rotation_method=rotation_method, return_gradient=True)
        elif algorithm == 'gpa_der_free':
            ff = lambda L=None, A=None, T=None: CF_objective(L=L, A=A, T=T, kappa=kappa, rotation_method=rotation_method, return_gradient=False)
        else:
            raise ValueError('Algorithm %s is not possible for %s rotation' % (algorithm, method))
    elif method == 'quartimax':
        return rotate_factors(A, 'orthomax', 0, **algorithm_kwargs)
    elif method == 'biquartimax':
        return rotate_factors(A, 'orthomax', 0.5, **algorithm_kwargs)
    elif method == 'varimax':
        return rotate_factors(A, 'orthomax', 1, **algorithm_kwargs)
    elif method == 'equamax':
        return rotate_factors(A, 'orthomax', 1 / p, **algorithm_kwargs)
    elif method == 'parsimax':
        return rotate_factors(A, 'CF', (k - 1) / (p + k - 2), 'orthogonal', **algorithm_kwargs)
    elif method == 'parsimony':
        return rotate_factors(A, 'CF', 1, 'orthogonal', **algorithm_kwargs)
    elif method == 'quartimin':
        return rotate_factors(A, 'oblimin', 0, 'oblique', **algorithm_kwargs)
    elif method == 'biquartimin':
        return rotate_factors(A, 'oblimin', 0.5, 'oblique', **algorithm_kwargs)
    elif method == 'target':
        assert len(method_args) == 2, 'only the rotation target and orthogonal/oblique should be provide for %s rotation' % method
        H = method_args[0]
        rotation_method = method_args[1]
        assert rotation_method in ['orthogonal', 'oblique'], 'rotation_method should be one of {orthogonal, oblique}'
        if algorithm == 'gpa':
            vgQ = lambda L=None, A=None, T=None: vgQ_target(H, L=L, A=A, T=T, rotation_method=rotation_method)
        elif algorithm == 'gpa_der_free':
            ff = lambda L=None, A=None, T=None: ff_target(H, L=L, A=A, T=T, rotation_method=rotation_method)
        elif algorithm == 'analytic':
            assert rotation_method == 'orthogonal', 'For analytic %s rotation only orthogonal rotation is supported'
            T = target_rotation(A, H, **algorithm_kwargs)
        else:
            raise ValueError('Algorithm %s is not possible for %s rotation' % (algorithm, method))
    elif method == 'partial_target':
        assert len(method_args) == 2, '2 additional arguments are expected for %s rotation' % method
        H = method_args[0]
        W = method_args[1]
        rotation_method = 'orthogonal'
        if algorithm == 'gpa':
            vgQ = lambda L=None, A=None, T=None: vgQ_partial_target(H, W=W, L=L, A=A, T=T)
        elif algorithm == 'gpa_der_free':
            ff = lambda L=None, A=None, T=None: ff_partial_target(H, W=W, L=L, A=A, T=T)
        else:
            raise ValueError('Algorithm %s is not possible for %s rotation' % (algorithm, method))
    else:
        raise ValueError('Invalid method')
    if T is None:
        (L, phi, T, table) = GPA(A, vgQ=vgQ, ff=ff, rotation_method=rotation_method, **algorithm_kwargs)
    if L is None:
        assert T is not None, 'Cannot compute L without T'
        L = rotateA(A, T, rotation_method=rotation_method)
    return (L, T)