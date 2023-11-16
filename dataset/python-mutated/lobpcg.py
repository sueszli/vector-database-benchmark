"""
Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG).

References
----------
.. [1] A. V. Knyazev (2001),
       Toward the Optimal Preconditioned Eigensolver: Locally Optimal
       Block Preconditioned Conjugate Gradient Method.
       SIAM Journal on Scientific Computing 23, no. 2,
       pp. 517-541. :doi:`10.1137/S1064827500366124`

.. [2] A. V. Knyazev, I. Lashuk, M. E. Argentati, and E. Ovchinnikov (2007),
       Block Locally Optimal Preconditioned Eigenvalue Xolvers (BLOPEX)
       in hypre and PETSc.  :arxiv:`0705.2626`

.. [3] A. V. Knyazev's C and MATLAB implementations:
       https://github.com/lobpcg/blopex
"""
import warnings
import numpy as np
from scipy.linalg import inv, eigh, cho_factor, cho_solve, cholesky, LinAlgError
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import issparse
__all__ = ['lobpcg']

def _report_nonhermitian(M, name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Report if `M` is not a Hermitian matrix given its type.\n    '
    from scipy.linalg import norm
    md = M - M.T.conj()
    nmd = norm(md, 1)
    tol = 10 * np.finfo(M.dtype).eps
    tol = max(tol, tol * norm(M, 1))
    if nmd > tol:
        warnings.warn(f'Matrix {name} of the type {M.dtype} is not Hermitian: condition: {nmd} < {tol} fails.', UserWarning, stacklevel=4)

def _as2d(ar):
    if False:
        i = 10
        return i + 15
    '\n    If the input array is 2D return it, if it is 1D, append a dimension,\n    making it a column vector.\n    '
    if ar.ndim == 2:
        return ar
    else:
        aux = np.array(ar, copy=False)
        aux.shape = (ar.shape[0], 1)
        return aux

def _makeMatMat(m):
    if False:
        while True:
            i = 10
    if m is None:
        return None
    elif callable(m):
        return lambda v: m(v)
    else:
        return lambda v: m @ v

def _matmul_inplace(x, y, verbosityLevel=0):
    if False:
        for i in range(10):
            print('nop')
    "Perform 'np.matmul' in-place if possible.\n\n    If some sufficient conditions for inplace matmul are met, do so.\n    Otherwise try inplace update and fall back to overwrite if that fails.\n    "
    if x.flags['CARRAY'] and x.shape[1] == y.shape[1] and (x.dtype == y.dtype):
        np.matmul(x, y, out=x)
    else:
        try:
            np.matmul(x, y, out=x)
        except Exception:
            if verbosityLevel:
                warnings.warn('Inplace update of x = x @ y failed, x needs to be overwritten.', UserWarning, stacklevel=3)
            x = x @ y
    return x

def _applyConstraints(blockVectorV, factYBY, blockVectorBY, blockVectorY):
    if False:
        for i in range(10):
            print('nop')
    'Changes blockVectorV in-place.'
    YBV = blockVectorBY.T.conj() @ blockVectorV
    tmp = cho_solve(factYBY, YBV)
    blockVectorV -= blockVectorY @ tmp

def _b_orthonormalize(B, blockVectorV, blockVectorBV=None, verbosityLevel=0):
    if False:
        print('Hello World!')
    'in-place B-orthonormalize the given block vector using Cholesky.'
    if blockVectorBV is None:
        if B is None:
            blockVectorBV = blockVectorV
        else:
            try:
                blockVectorBV = B(blockVectorV)
            except Exception as e:
                if verbosityLevel:
                    warnings.warn(f'Secondary MatMul call failed with error\n{e}\n', UserWarning, stacklevel=3)
                    return (None, None, None)
            if blockVectorBV.shape != blockVectorV.shape:
                raise ValueError(f'The shape {blockVectorV.shape} of the orthogonalized matrix not preserved\nand changed to {blockVectorBV.shape} after multiplying by the secondary matrix.\n')
    VBV = blockVectorV.T.conj() @ blockVectorBV
    try:
        VBV = cholesky(VBV, overwrite_a=True)
        VBV = inv(VBV, overwrite_a=True)
        blockVectorV = _matmul_inplace(blockVectorV, VBV, verbosityLevel=verbosityLevel)
        if B is not None:
            blockVectorBV = _matmul_inplace(blockVectorBV, VBV, verbosityLevel=verbosityLevel)
        return (blockVectorV, blockVectorBV, VBV)
    except LinAlgError:
        if verbosityLevel:
            warnings.warn('Cholesky has failed.', UserWarning, stacklevel=3)
        return (None, None, None)

def _get_indx(_lambda, num, largest):
    if False:
        for i in range(10):
            print('nop')
    'Get `num` indices into `_lambda` depending on `largest` option.'
    ii = np.argsort(_lambda)
    if largest:
        ii = ii[:-num - 1:-1]
    else:
        ii = ii[:num]
    return ii

def _handle_gramA_gramB_verbosity(gramA, gramB, verbosityLevel):
    if False:
        return 10
    if verbosityLevel:
        _report_nonhermitian(gramA, 'gramA')
        _report_nonhermitian(gramB, 'gramB')

def lobpcg(A, X, B=None, M=None, Y=None, tol=None, maxiter=None, largest=True, verbosityLevel=0, retLambdaHistory=False, retResidualNormsHistory=False, restartControl=20):
    if False:
        while True:
            i = 10
    'Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG).\n\n    LOBPCG is a preconditioned eigensolver for large real symmetric and complex\n    Hermitian definite generalized eigenproblems.\n\n    Parameters\n    ----------\n    A : {sparse matrix, ndarray, LinearOperator, callable object}\n        The Hermitian linear operator of the problem, usually given by a\n        sparse matrix.  Often called the "stiffness matrix".\n    X : ndarray, float32 or float64\n        Initial approximation to the ``k`` eigenvectors (non-sparse).\n        If `A` has ``shape=(n,n)`` then `X` must have ``shape=(n,k)``.\n    B : {sparse matrix, ndarray, LinearOperator, callable object}\n        Optional. By default ``B = None``, which is equivalent to identity.\n        The right hand side operator in a generalized eigenproblem if present.\n        Often called the "mass matrix". Must be Hermitian positive definite.\n    M : {sparse matrix, ndarray, LinearOperator, callable object}\n        Optional. By default ``M = None``, which is equivalent to identity.\n        Preconditioner aiming to accelerate convergence.\n    Y : ndarray, float32 or float64, default: None\n        An ``n-by-sizeY`` ndarray of constraints with ``sizeY < n``.\n        The iterations will be performed in the ``B``-orthogonal complement\n        of the column-space of `Y`. `Y` must be full rank if present.\n    tol : scalar, optional\n        The default is ``tol=n*sqrt(eps)``.\n        Solver tolerance for the stopping criterion.\n    maxiter : int, default: 20\n        Maximum number of iterations.\n    largest : bool, default: True\n        When True, solve for the largest eigenvalues, otherwise the smallest.\n    verbosityLevel : int, optional\n        By default ``verbosityLevel=0`` no output.\n        Controls the solver standard/screen output.\n    retLambdaHistory : bool, default: False\n        Whether to return iterative eigenvalue history.\n    retResidualNormsHistory : bool, default: False\n        Whether to return iterative history of residual norms.\n    restartControl : int, optional.\n        Iterations restart if the residuals jump ``2**restartControl`` times\n        compared to the smallest recorded in ``retResidualNormsHistory``.\n        The default is ``restartControl=20``, making the restarts rare for\n        backward compatibility.\n\n    Returns\n    -------\n    lambda : ndarray of the shape ``(k, )``.\n        Array of ``k`` approximate eigenvalues.\n    v : ndarray of the same shape as ``X.shape``.\n        An array of ``k`` approximate eigenvectors.\n    lambdaHistory : ndarray, optional.\n        The eigenvalue history, if `retLambdaHistory` is ``True``.\n    ResidualNormsHistory : ndarray, optional.\n        The history of residual norms, if `retResidualNormsHistory`\n        is ``True``.\n\n    Notes\n    -----\n    The iterative loop runs ``maxit=maxiter`` (20 if ``maxit=None``)\n    iterations at most and finishes earler if the tolerance is met.\n    Breaking backward compatibility with the previous version, LOBPCG\n    now returns the block of iterative vectors with the best accuracy rather\n    than the last one iterated, as a cure for possible divergence.\n\n    If ``X.dtype == np.float32`` and user-provided operations/multiplications\n    by `A`, `B`, and `M` all preserve the ``np.float32`` data type,\n    all the calculations and the output are in ``np.float32``.\n\n    The size of the iteration history output equals to the number of the best\n    (limited by `maxit`) iterations plus 3: initial, final, and postprocessing.\n\n    If both `retLambdaHistory` and `retResidualNormsHistory` are ``True``,\n    the return tuple has the following format\n    ``(lambda, V, lambda history, residual norms history)``.\n\n    In the following ``n`` denotes the matrix size and ``k`` the number\n    of required eigenvalues (smallest or largest).\n\n    The LOBPCG code internally solves eigenproblems of the size ``3k`` on every\n    iteration by calling the dense eigensolver `eigh`, so if ``k`` is not\n    small enough compared to ``n``, it makes no sense to call the LOBPCG code.\n    Moreover, if one calls the LOBPCG algorithm for ``5k > n``, it would likely\n    break internally, so the code calls the standard function `eigh` instead.\n    It is not that ``n`` should be large for the LOBPCG to work, but rather the\n    ratio ``n / k`` should be large. It you call LOBPCG with ``k=1``\n    and ``n=10``, it works though ``n`` is small. The method is intended\n    for extremely large ``n / k``.\n\n    The convergence speed depends basically on three factors:\n\n    1. Quality of the initial approximations `X` to the seeking eigenvectors.\n       Randomly distributed around the origin vectors work well if no better\n       choice is known.\n\n    2. Relative separation of the desired eigenvalues from the rest\n       of the eigenvalues. One can vary ``k`` to improve the separation.\n\n    3. Proper preconditioning to shrink the spectral spread.\n       For example, a rod vibration test problem (under tests\n       directory) is ill-conditioned for large ``n``, so convergence will be\n       slow, unless efficient preconditioning is used. For this specific\n       problem, a good simple preconditioner function would be a linear solve\n       for `A`, which is easy to code since `A` is tridiagonal.\n\n    References\n    ----------\n    .. [1] A. V. Knyazev (2001),\n           Toward the Optimal Preconditioned Eigensolver: Locally Optimal\n           Block Preconditioned Conjugate Gradient Method.\n           SIAM Journal on Scientific Computing 23, no. 2,\n           pp. 517-541. :doi:`10.1137/S1064827500366124`\n\n    .. [2] A. V. Knyazev, I. Lashuk, M. E. Argentati, and E. Ovchinnikov\n           (2007), Block Locally Optimal Preconditioned Eigenvalue Xolvers\n           (BLOPEX) in hypre and PETSc. :arxiv:`0705.2626`\n\n    .. [3] A. V. Knyazev\'s C and MATLAB implementations:\n           https://github.com/lobpcg/blopex\n\n    Examples\n    --------\n    Our first example is minimalistic - find the largest eigenvalue of\n    a diagonal matrix by solving the non-generalized eigenvalue problem\n    ``A x = lambda x`` without constraints or preconditioning.\n\n    >>> import numpy as np\n    >>> from scipy.sparse import spdiags\n    >>> from scipy.sparse.linalg import LinearOperator, aslinearoperator\n    >>> from scipy.sparse.linalg import lobpcg\n\n    The square matrix size is\n\n    >>> n = 100\n\n    and its diagonal entries are 1, ..., 100 defined by\n\n    >>> vals = np.arange(1, n + 1).astype(np.int16)\n\n    The first mandatory input parameter in this test is\n    the sparse diagonal matrix `A`\n    of the eigenvalue problem ``A x = lambda x`` to solve.\n\n    >>> A = spdiags(vals, 0, n, n)\n    >>> A = A.astype(np.int16)\n    >>> A.toarray()\n    array([[  1,   0,   0, ...,   0,   0,   0],\n           [  0,   2,   0, ...,   0,   0,   0],\n           [  0,   0,   3, ...,   0,   0,   0],\n           ...,\n           [  0,   0,   0, ...,  98,   0,   0],\n           [  0,   0,   0, ...,   0,  99,   0],\n           [  0,   0,   0, ...,   0,   0, 100]], dtype=int16)\n\n    The second mandatory input parameter `X` is a 2D array with the\n    row dimension determining the number of requested eigenvalues.\n    `X` is an initial guess for targeted eigenvectors.\n    `X` must have linearly independent columns.\n    If no initial approximations available, randomly oriented vectors\n    commonly work best, e.g., with components normally distributed\n    around zero or uniformly distributed on the interval [-1 1].\n    Setting the initial approximations to dtype ``np.float32``\n    forces all iterative values to dtype ``np.float32`` speeding up\n    the run while still allowing accurate eigenvalue computations.\n\n    >>> k = 1\n    >>> rng = np.random.default_rng()\n    >>> X = rng.normal(size=(n, k))\n    >>> X = X.astype(np.float32)\n\n    >>> eigenvalues, _ = lobpcg(A, X, maxiter=60)\n    >>> eigenvalues\n    array([100.])\n    >>> eigenvalues.dtype\n    dtype(\'float32\')\n\n    `lobpcg` needs only access the matrix product with `A` rather\n    then the matrix itself. Since the matrix `A` is diagonal in\n    this example, one can write a function of the matrix product\n    ``A @ X`` using the diagonal values ``vals`` only, e.g., by\n    element-wise multiplication with broadcasting in the lambda-function\n\n    >>> A_lambda = lambda X: vals[:, np.newaxis] * X\n\n    or the regular function\n\n    >>> def A_matmat(X):\n    ...     return vals[:, np.newaxis] * X\n\n    and use the handle to one of these callables as an input\n\n    >>> eigenvalues, _ = lobpcg(A_lambda, X, maxiter=60)\n    >>> eigenvalues\n    array([100.])\n    >>> eigenvalues, _ = lobpcg(A_matmat, X, maxiter=60)\n    >>> eigenvalues\n    array([100.])\n\n    The traditional callable `LinearOperator` is no longer\n    necessary but still supported as the input to `lobpcg`.\n    Specifying ``matmat=A_matmat`` explicitely improves performance. \n\n    >>> A_lo = LinearOperator((n, n), matvec=A_matmat, matmat=A_matmat, dtype=np.int16)\n    >>> eigenvalues, _ = lobpcg(A_lo, X, maxiter=80)\n    >>> eigenvalues\n    array([100.])\n\n    The least efficient callable option is `aslinearoperator`:\n\n    >>> eigenvalues, _ = lobpcg(aslinearoperator(A), X, maxiter=80)\n    >>> eigenvalues\n    array([100.])\n\n    We now switch to computing the three smallest eigenvalues specifying\n\n    >>> k = 3\n    >>> X = np.random.default_rng().normal(size=(n, k))\n\n    and ``largest=False`` parameter\n\n    >>> eigenvalues, _ = lobpcg(A, X, largest=False, maxiter=80)\n    >>> print(eigenvalues)  \n    [1. 2. 3.]\n\n    The next example illustrates computing 3 smallest eigenvalues of\n    the same matrix `A` given by the function handle ``A_matmat`` but\n    with constraints and preconditioning.\n\n    Constraints - an optional input parameter is a 2D array comprising\n    of column vectors that the eigenvectors must be orthogonal to\n\n    >>> Y = np.eye(n, 3)\n\n    The preconditioner acts as the inverse of `A` in this example, but\n    in the reduced precision ``np.float32`` even though the initial `X`\n    and thus all iterates and the output are in full ``np.float64``.\n\n    >>> inv_vals = 1./vals\n    >>> inv_vals = inv_vals.astype(np.float32)\n    >>> M = lambda X: inv_vals[:, np.newaxis] * X\n\n    Let us now solve the eigenvalue problem for the matrix `A` first\n    without preconditioning requesting 80 iterations\n\n    >>> eigenvalues, _ = lobpcg(A_matmat, X, Y=Y, largest=False, maxiter=80)\n    >>> eigenvalues\n    array([4., 5., 6.])\n    >>> eigenvalues.dtype\n    dtype(\'float64\')\n\n    With preconditioning we need only 20 iterations from the same `X`\n\n    >>> eigenvalues, _ = lobpcg(A_matmat, X, Y=Y, M=M, largest=False, maxiter=20)\n    >>> eigenvalues\n    array([4., 5., 6.])\n\n    Note that the vectors passed in `Y` are the eigenvectors of the 3\n    smallest eigenvalues. The results returned above are orthogonal to those.\n\n    The primary matrix `A` may be indefinite, e.g., after shifting\n    ``vals`` by 50 from 1, ..., 100 to -49, ..., 50, we still can compute\n    the 3 smallest or largest eigenvalues.\n\n    >>> vals = vals - 50\n    >>> X = rng.normal(size=(n, k))\n    >>> eigenvalues, _ = lobpcg(A_matmat, X, largest=False, maxiter=99)\n    >>> eigenvalues\n    array([-49., -48., -47.])\n    >>> eigenvalues, _ = lobpcg(A_matmat, X, largest=True, maxiter=99)\n    >>> eigenvalues\n    array([50., 49., 48.])\n\n    '
    blockVectorX = X
    bestblockVectorX = blockVectorX
    blockVectorY = Y
    residualTolerance = tol
    if maxiter is None:
        maxiter = 20
    bestIterationNumber = maxiter
    sizeY = 0
    if blockVectorY is not None:
        if len(blockVectorY.shape) != 2:
            warnings.warn(f'Expected rank-2 array for argument Y, instead got {len(blockVectorY.shape)}, so ignore it and use no constraints.', UserWarning, stacklevel=2)
            blockVectorY = None
        else:
            sizeY = blockVectorY.shape[1]
    if blockVectorX is None:
        raise ValueError('The mandatory initial matrix X cannot be None')
    if len(blockVectorX.shape) != 2:
        raise ValueError('expected rank-2 array for argument X')
    (n, sizeX) = blockVectorX.shape
    if not np.issubdtype(blockVectorX.dtype, np.inexact):
        warnings.warn(f'Data type for argument X is {blockVectorX.dtype}, which is not inexact, so casted to np.float32.', UserWarning, stacklevel=2)
        blockVectorX = np.asarray(blockVectorX, dtype=np.float32)
    if retLambdaHistory:
        lambdaHistory = np.zeros((maxiter + 3, sizeX), dtype=blockVectorX.dtype)
    if retResidualNormsHistory:
        residualNormsHistory = np.zeros((maxiter + 3, sizeX), dtype=blockVectorX.dtype)
    if verbosityLevel:
        aux = 'Solving '
        if B is None:
            aux += 'standard'
        else:
            aux += 'generalized'
        aux += ' eigenvalue problem with'
        if M is None:
            aux += 'out'
        aux += ' preconditioning\n\n'
        aux += 'matrix size %d\n' % n
        aux += 'block size %d\n\n' % sizeX
        if blockVectorY is None:
            aux += 'No constraints\n\n'
        elif sizeY > 1:
            aux += '%d constraints\n\n' % sizeY
        else:
            aux += '%d constraint\n\n' % sizeY
        print(aux)
    if n - sizeY < 5 * sizeX:
        warnings.warn(f'The problem size {n} minus the constraints size {sizeY} is too small relative to the block size {sizeX}. Using a dense eigensolver instead of LOBPCG iterations.No output of the history of the iterations.', UserWarning, stacklevel=2)
        sizeX = min(sizeX, n)
        if blockVectorY is not None:
            raise NotImplementedError('The dense eigensolver does not support constraints.')
        if largest:
            eigvals = (n - sizeX, n - 1)
        else:
            eigvals = (0, sizeX - 1)
        try:
            if isinstance(A, LinearOperator):
                A = A(np.eye(n, dtype=int))
            elif callable(A):
                A = A(np.eye(n, dtype=int))
                if A.shape != (n, n):
                    raise ValueError(f'The shape {A.shape} of the primary matrix\ndefined by a callable object is wrong.\n')
            elif issparse(A):
                A = A.toarray()
            else:
                A = np.asarray(A)
        except Exception as e:
            raise Exception(f'Primary MatMul call failed with error\n{e}\n')
        if B is not None:
            try:
                if isinstance(B, LinearOperator):
                    B = B(np.eye(n, dtype=int))
                elif callable(B):
                    B = B(np.eye(n, dtype=int))
                    if B.shape != (n, n):
                        raise ValueError(f'The shape {B.shape} of the secondary matrix\ndefined by a callable object is wrong.\n')
                elif issparse(B):
                    B = B.toarray()
                else:
                    B = np.asarray(B)
            except Exception as e:
                raise Exception(f'Secondary MatMul call failed with error\n{e}\n')
        try:
            (vals, vecs) = eigh(A, B, subset_by_index=eigvals, check_finite=False)
            if largest:
                vals = vals[::-1]
                vecs = vecs[:, ::-1]
            return (vals, vecs)
        except Exception as e:
            raise Exception(f'Dense eigensolver failed with error\n{e}\n')
    if residualTolerance is None or residualTolerance <= 0.0:
        residualTolerance = np.sqrt(np.finfo(blockVectorX.dtype).eps) * n
    A = _makeMatMat(A)
    B = _makeMatMat(B)
    M = _makeMatMat(M)
    if blockVectorY is not None:
        if B is not None:
            blockVectorBY = B(blockVectorY)
            if blockVectorBY.shape != blockVectorY.shape:
                raise ValueError(f'The shape {blockVectorY.shape} of the constraint not preserved\nand changed to {blockVectorBY.shape} after multiplying by the secondary matrix.\n')
        else:
            blockVectorBY = blockVectorY
        gramYBY = blockVectorY.T.conj() @ blockVectorBY
        try:
            gramYBY = cho_factor(gramYBY, overwrite_a=True)
        except LinAlgError as e:
            raise ValueError('Linearly dependent constraints') from e
        _applyConstraints(blockVectorX, gramYBY, blockVectorBY, blockVectorY)
    (blockVectorX, blockVectorBX, _) = _b_orthonormalize(B, blockVectorX, verbosityLevel=verbosityLevel)
    if blockVectorX is None:
        raise ValueError('Linearly dependent initial approximations')
    blockVectorAX = A(blockVectorX)
    if blockVectorAX.shape != blockVectorX.shape:
        raise ValueError(f'The shape {blockVectorX.shape} of the initial approximations not preserved\nand changed to {blockVectorAX.shape} after multiplying by the primary matrix.\n')
    gramXAX = blockVectorX.T.conj() @ blockVectorAX
    (_lambda, eigBlockVector) = eigh(gramXAX, check_finite=False)
    ii = _get_indx(_lambda, sizeX, largest)
    _lambda = _lambda[ii]
    if retLambdaHistory:
        lambdaHistory[0, :] = _lambda
    eigBlockVector = np.asarray(eigBlockVector[:, ii])
    blockVectorX = _matmul_inplace(blockVectorX, eigBlockVector, verbosityLevel=verbosityLevel)
    blockVectorAX = _matmul_inplace(blockVectorAX, eigBlockVector, verbosityLevel=verbosityLevel)
    if B is not None:
        blockVectorBX = _matmul_inplace(blockVectorBX, eigBlockVector, verbosityLevel=verbosityLevel)
    activeMask = np.ones((sizeX,), dtype=bool)
    blockVectorP = None
    blockVectorAP = None
    blockVectorBP = None
    smallestResidualNorm = np.abs(np.finfo(blockVectorX.dtype).max)
    iterationNumber = -1
    restart = True
    forcedRestart = False
    explicitGramFlag = False
    while iterationNumber < maxiter:
        iterationNumber += 1
        if B is not None:
            aux = blockVectorBX * _lambda[np.newaxis, :]
        else:
            aux = blockVectorX * _lambda[np.newaxis, :]
        blockVectorR = blockVectorAX - aux
        aux = np.sum(blockVectorR.conj() * blockVectorR, 0)
        residualNorms = np.sqrt(np.abs(aux))
        if retResidualNormsHistory:
            residualNormsHistory[iterationNumber, :] = residualNorms
        residualNorm = np.sum(np.abs(residualNorms)) / sizeX
        if residualNorm < smallestResidualNorm:
            smallestResidualNorm = residualNorm
            bestIterationNumber = iterationNumber
            bestblockVectorX = blockVectorX
        elif residualNorm > 2 ** restartControl * smallestResidualNorm:
            forcedRestart = True
            blockVectorAX = A(blockVectorX)
            if blockVectorAX.shape != blockVectorX.shape:
                raise ValueError(f'The shape {blockVectorX.shape} of the restarted iterate not preserved\nand changed to {blockVectorAX.shape} after multiplying by the primary matrix.\n')
            if B is not None:
                blockVectorBX = B(blockVectorX)
                if blockVectorBX.shape != blockVectorX.shape:
                    raise ValueError(f'The shape {blockVectorX.shape} of the restarted iterate not preserved\nand changed to {blockVectorBX.shape} after multiplying by the secondary matrix.\n')
        ii = np.where(residualNorms > residualTolerance, True, False)
        activeMask = activeMask & ii
        currentBlockSize = activeMask.sum()
        if verbosityLevel:
            print(f'iteration {iterationNumber}')
            print(f'current block size: {currentBlockSize}')
            print(f'eigenvalue(s):\n{_lambda}')
            print(f'residual norm(s):\n{residualNorms}')
        if currentBlockSize == 0:
            break
        activeBlockVectorR = _as2d(blockVectorR[:, activeMask])
        if iterationNumber > 0:
            activeBlockVectorP = _as2d(blockVectorP[:, activeMask])
            activeBlockVectorAP = _as2d(blockVectorAP[:, activeMask])
            if B is not None:
                activeBlockVectorBP = _as2d(blockVectorBP[:, activeMask])
        if M is not None:
            activeBlockVectorR = M(activeBlockVectorR)
        if blockVectorY is not None:
            _applyConstraints(activeBlockVectorR, gramYBY, blockVectorBY, blockVectorY)
        if B is not None:
            activeBlockVectorR = activeBlockVectorR - blockVectorX @ (blockVectorBX.T.conj() @ activeBlockVectorR)
        else:
            activeBlockVectorR = activeBlockVectorR - blockVectorX @ (blockVectorX.T.conj() @ activeBlockVectorR)
        aux = _b_orthonormalize(B, activeBlockVectorR, verbosityLevel=verbosityLevel)
        (activeBlockVectorR, activeBlockVectorBR, _) = aux
        if activeBlockVectorR is None:
            warnings.warn(f'Failed at iteration {iterationNumber} with accuracies {residualNorms}\n not reaching the requested tolerance {residualTolerance}.', UserWarning, stacklevel=2)
            break
        activeBlockVectorAR = A(activeBlockVectorR)
        if iterationNumber > 0:
            if B is not None:
                aux = _b_orthonormalize(B, activeBlockVectorP, activeBlockVectorBP, verbosityLevel=verbosityLevel)
                (activeBlockVectorP, activeBlockVectorBP, invR) = aux
            else:
                aux = _b_orthonormalize(B, activeBlockVectorP, verbosityLevel=verbosityLevel)
                (activeBlockVectorP, _, invR) = aux
            if activeBlockVectorP is not None:
                activeBlockVectorAP = _matmul_inplace(activeBlockVectorAP, invR, verbosityLevel=verbosityLevel)
                restart = forcedRestart
            else:
                restart = True
        if activeBlockVectorAR.dtype == 'float32':
            myeps = 1
        else:
            myeps = np.sqrt(np.finfo(activeBlockVectorR.dtype).eps)
        if residualNorms.max() > myeps and (not explicitGramFlag):
            explicitGramFlag = False
        else:
            explicitGramFlag = True
        if B is None:
            blockVectorBX = blockVectorX
            activeBlockVectorBR = activeBlockVectorR
            if not restart:
                activeBlockVectorBP = activeBlockVectorP
        gramXAR = np.dot(blockVectorX.T.conj(), activeBlockVectorAR)
        gramRAR = np.dot(activeBlockVectorR.T.conj(), activeBlockVectorAR)
        gramDtype = activeBlockVectorAR.dtype
        if explicitGramFlag:
            gramRAR = (gramRAR + gramRAR.T.conj()) / 2
            gramXAX = np.dot(blockVectorX.T.conj(), blockVectorAX)
            gramXAX = (gramXAX + gramXAX.T.conj()) / 2
            gramXBX = np.dot(blockVectorX.T.conj(), blockVectorBX)
            gramRBR = np.dot(activeBlockVectorR.T.conj(), activeBlockVectorBR)
            gramXBR = np.dot(blockVectorX.T.conj(), activeBlockVectorBR)
        else:
            gramXAX = np.diag(_lambda).astype(gramDtype)
            gramXBX = np.eye(sizeX, dtype=gramDtype)
            gramRBR = np.eye(currentBlockSize, dtype=gramDtype)
            gramXBR = np.zeros((sizeX, currentBlockSize), dtype=gramDtype)
        if not restart:
            gramXAP = np.dot(blockVectorX.T.conj(), activeBlockVectorAP)
            gramRAP = np.dot(activeBlockVectorR.T.conj(), activeBlockVectorAP)
            gramPAP = np.dot(activeBlockVectorP.T.conj(), activeBlockVectorAP)
            gramXBP = np.dot(blockVectorX.T.conj(), activeBlockVectorBP)
            gramRBP = np.dot(activeBlockVectorR.T.conj(), activeBlockVectorBP)
            if explicitGramFlag:
                gramPAP = (gramPAP + gramPAP.T.conj()) / 2
                gramPBP = np.dot(activeBlockVectorP.T.conj(), activeBlockVectorBP)
            else:
                gramPBP = np.eye(currentBlockSize, dtype=gramDtype)
            gramA = np.block([[gramXAX, gramXAR, gramXAP], [gramXAR.T.conj(), gramRAR, gramRAP], [gramXAP.T.conj(), gramRAP.T.conj(), gramPAP]])
            gramB = np.block([[gramXBX, gramXBR, gramXBP], [gramXBR.T.conj(), gramRBR, gramRBP], [gramXBP.T.conj(), gramRBP.T.conj(), gramPBP]])
            _handle_gramA_gramB_verbosity(gramA, gramB, verbosityLevel)
            try:
                (_lambda, eigBlockVector) = eigh(gramA, gramB, check_finite=False)
            except LinAlgError as e:
                if verbosityLevel:
                    warnings.warn(f'eigh failed at iteration {iterationNumber} \nwith error {e} causing a restart.\n', UserWarning, stacklevel=2)
                restart = True
        if restart:
            gramA = np.block([[gramXAX, gramXAR], [gramXAR.T.conj(), gramRAR]])
            gramB = np.block([[gramXBX, gramXBR], [gramXBR.T.conj(), gramRBR]])
            _handle_gramA_gramB_verbosity(gramA, gramB, verbosityLevel)
            try:
                (_lambda, eigBlockVector) = eigh(gramA, gramB, check_finite=False)
            except LinAlgError as e:
                warnings.warn(f'eigh failed at iteration {iterationNumber} with error\n{e}\n', UserWarning, stacklevel=2)
                break
        ii = _get_indx(_lambda, sizeX, largest)
        _lambda = _lambda[ii]
        eigBlockVector = eigBlockVector[:, ii]
        if retLambdaHistory:
            lambdaHistory[iterationNumber + 1, :] = _lambda
        if B is not None:
            if not restart:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:sizeX + currentBlockSize]
                eigBlockVectorP = eigBlockVector[sizeX + currentBlockSize:]
                pp = np.dot(activeBlockVectorR, eigBlockVectorR)
                pp += np.dot(activeBlockVectorP, eigBlockVectorP)
                app = np.dot(activeBlockVectorAR, eigBlockVectorR)
                app += np.dot(activeBlockVectorAP, eigBlockVectorP)
                bpp = np.dot(activeBlockVectorBR, eigBlockVectorR)
                bpp += np.dot(activeBlockVectorBP, eigBlockVectorP)
            else:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:]
                pp = np.dot(activeBlockVectorR, eigBlockVectorR)
                app = np.dot(activeBlockVectorAR, eigBlockVectorR)
                bpp = np.dot(activeBlockVectorBR, eigBlockVectorR)
            blockVectorX = np.dot(blockVectorX, eigBlockVectorX) + pp
            blockVectorAX = np.dot(blockVectorAX, eigBlockVectorX) + app
            blockVectorBX = np.dot(blockVectorBX, eigBlockVectorX) + bpp
            (blockVectorP, blockVectorAP, blockVectorBP) = (pp, app, bpp)
        else:
            if not restart:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:sizeX + currentBlockSize]
                eigBlockVectorP = eigBlockVector[sizeX + currentBlockSize:]
                pp = np.dot(activeBlockVectorR, eigBlockVectorR)
                pp += np.dot(activeBlockVectorP, eigBlockVectorP)
                app = np.dot(activeBlockVectorAR, eigBlockVectorR)
                app += np.dot(activeBlockVectorAP, eigBlockVectorP)
            else:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:]
                pp = np.dot(activeBlockVectorR, eigBlockVectorR)
                app = np.dot(activeBlockVectorAR, eigBlockVectorR)
            blockVectorX = np.dot(blockVectorX, eigBlockVectorX) + pp
            blockVectorAX = np.dot(blockVectorAX, eigBlockVectorX) + app
            (blockVectorP, blockVectorAP) = (pp, app)
    if B is not None:
        aux = blockVectorBX * _lambda[np.newaxis, :]
    else:
        aux = blockVectorX * _lambda[np.newaxis, :]
    blockVectorR = blockVectorAX - aux
    aux = np.sum(blockVectorR.conj() * blockVectorR, 0)
    residualNorms = np.sqrt(np.abs(aux))
    if retLambdaHistory:
        lambdaHistory[iterationNumber + 1, :] = _lambda
    if retResidualNormsHistory:
        residualNormsHistory[iterationNumber + 1, :] = residualNorms
    residualNorm = np.sum(np.abs(residualNorms)) / sizeX
    if residualNorm < smallestResidualNorm:
        smallestResidualNorm = residualNorm
        bestIterationNumber = iterationNumber + 1
        bestblockVectorX = blockVectorX
    if np.max(np.abs(residualNorms)) > residualTolerance:
        warnings.warn(f'Exited at iteration {iterationNumber} with accuracies \n{residualNorms}\nnot reaching the requested tolerance {residualTolerance}.\nUse iteration {bestIterationNumber} instead with accuracy \n{smallestResidualNorm}.\n', UserWarning, stacklevel=2)
    if verbosityLevel:
        print(f'Final iterative eigenvalue(s):\n{_lambda}')
        print(f'Final iterative residual norm(s):\n{residualNorms}')
    blockVectorX = bestblockVectorX
    if blockVectorY is not None:
        _applyConstraints(blockVectorX, gramYBY, blockVectorBY, blockVectorY)
    blockVectorAX = A(blockVectorX)
    if blockVectorAX.shape != blockVectorX.shape:
        raise ValueError(f'The shape {blockVectorX.shape} of the postprocessing iterate not preserved\nand changed to {blockVectorAX.shape} after multiplying by the primary matrix.\n')
    gramXAX = np.dot(blockVectorX.T.conj(), blockVectorAX)
    blockVectorBX = blockVectorX
    if B is not None:
        blockVectorBX = B(blockVectorX)
        if blockVectorBX.shape != blockVectorX.shape:
            raise ValueError(f'The shape {blockVectorX.shape} of the postprocessing iterate not preserved\nand changed to {blockVectorBX.shape} after multiplying by the secondary matrix.\n')
    gramXBX = np.dot(blockVectorX.T.conj(), blockVectorBX)
    _handle_gramA_gramB_verbosity(gramXAX, gramXBX, verbosityLevel)
    gramXAX = (gramXAX + gramXAX.T.conj()) / 2
    gramXBX = (gramXBX + gramXBX.T.conj()) / 2
    try:
        (_lambda, eigBlockVector) = eigh(gramXAX, gramXBX, check_finite=False)
    except LinAlgError as e:
        raise ValueError('eigh has failed in lobpcg postprocessing') from e
    ii = _get_indx(_lambda, sizeX, largest)
    _lambda = _lambda[ii]
    eigBlockVector = np.asarray(eigBlockVector[:, ii])
    blockVectorX = np.dot(blockVectorX, eigBlockVector)
    blockVectorAX = np.dot(blockVectorAX, eigBlockVector)
    if B is not None:
        blockVectorBX = np.dot(blockVectorBX, eigBlockVector)
        aux = blockVectorBX * _lambda[np.newaxis, :]
    else:
        aux = blockVectorX * _lambda[np.newaxis, :]
    blockVectorR = blockVectorAX - aux
    aux = np.sum(blockVectorR.conj() * blockVectorR, 0)
    residualNorms = np.sqrt(np.abs(aux))
    if retLambdaHistory:
        lambdaHistory[bestIterationNumber + 1, :] = _lambda
    if retResidualNormsHistory:
        residualNormsHistory[bestIterationNumber + 1, :] = residualNorms
    if retLambdaHistory:
        lambdaHistory = lambdaHistory[:bestIterationNumber + 2, :]
    if retResidualNormsHistory:
        residualNormsHistory = residualNormsHistory[:bestIterationNumber + 2, :]
    if np.max(np.abs(residualNorms)) > residualTolerance:
        warnings.warn(f'Exited postprocessing with accuracies \n{residualNorms}\nnot reaching the requested tolerance {residualTolerance}.', UserWarning, stacklevel=2)
    if verbosityLevel:
        print(f'Final postprocessing eigenvalue(s):\n{_lambda}')
        print(f'Final residual norm(s):\n{residualNorms}')
    if retLambdaHistory:
        lambdaHistory = np.vsplit(lambdaHistory, np.shape(lambdaHistory)[0])
        lambdaHistory = [np.squeeze(i) for i in lambdaHistory]
    if retResidualNormsHistory:
        residualNormsHistory = np.vsplit(residualNormsHistory, np.shape(residualNormsHistory)[0])
        residualNormsHistory = [np.squeeze(i) for i in residualNormsHistory]
    if retLambdaHistory:
        if retResidualNormsHistory:
            return (_lambda, blockVectorX, lambdaHistory, residualNormsHistory)
        else:
            return (_lambda, blockVectorX, lambdaHistory)
    elif retResidualNormsHistory:
        return (_lambda, blockVectorX, residualNormsHistory)
    else:
        return (_lambda, blockVectorX)