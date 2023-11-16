"""
This file contains a Python version of the gradient projection rotation
algorithms (GPA) developed by Bernaards, C.A. and Jennrich, R.I.
The code is based on code developed Bernaards, C.A. and Jennrich, R.I.
and is ported and made available with permission of the authors.

References
----------
[1] Bernaards, C.A. and Jennrich, R.I. (2005) Gradient Projection Algorithms
and Software for Arbitrary Rotation Criteria in Factor Analysis. Educational
and Psychological Measurement, 65 (5), 676-696.

[2] Jennrich, R.I. (2001). A simple general procedure for orthogonal rotation.
Psychometrika, 66, 289-306.

[3] Jennrich, R.I. (2002). A simple general method for oblique rotation.
Psychometrika, 67, 7-19.

[4] http://www.stat.ucla.edu/research/gpa/matlab.net

[5] http://www.stat.ucla.edu/research/gpa/GPderfree.txt
"""
import numpy as np

def GPA(A, ff=None, vgQ=None, T=None, max_tries=501, rotation_method='orthogonal', tol=1e-05):
    if False:
        print('Hello World!')
    '\n    The gradient projection algorithm (GPA) minimizes a target function\n    :math:`\\phi(L)`, where :math:`L` is a matrix with rotated factors.\n\n    For orthogonal rotation methods :math:`L=AT`, where :math:`T` is an\n    orthogonal matrix. For oblique rotation matrices :math:`L=A(T^*)^{-1}`,\n    where :math:`T` is a normal matrix, i.e., :math:`TT^*=T^*T`. Oblique\n    rotations relax the orthogonality constraint in order to gain simplicity\n    in the interpretation.\n\n    Parameters\n    ----------\n    A : numpy matrix\n        non rotated factors\n    T : numpy matrix (default identity matrix)\n        initial guess of rotation matrix\n    ff : function (defualt None)\n        criterion :math:`\\phi` to optimize. Should have A, T, L as keyword\n        arguments\n        and mapping to a float. Only used (and required) if vgQ is not\n        provided.\n    vgQ : function (defualt None)\n        criterion :math:`\\phi` to optimize and its derivative. Should have\n         A, T, L as keyword arguments and mapping to a tuple containing a\n        float and vector. Can be omitted if ff is provided.\n    max_tries : int (default 501)\n        maximum number of iterations\n    rotation_method : str\n        should be one of {orthogonal, oblique}\n    tol : float\n        stop criterion, algorithm stops if Frobenius norm of gradient is\n        smaller then tol\n    '
    if rotation_method not in ['orthogonal', 'oblique']:
        raise ValueError('rotation_method should be one of {orthogonal, oblique}')
    if vgQ is None:
        if ff is None:
            raise ValueError('ff should be provided if vgQ is not')
        derivative_free = True
        Gff = lambda x: Gf(x, lambda y: ff(T=y, A=A, L=None))
    else:
        derivative_free = False
    if T is None:
        T = np.eye(A.shape[1])
    al = 1
    table = []
    if derivative_free:
        f = ff(T=T, A=A, L=None)
        G = Gff(T)
    elif rotation_method == 'orthogonal':
        L = A.dot(T)
        (f, Gq) = vgQ(L=L)
        G = A.T.dot(Gq)
    else:
        Ti = np.linalg.inv(T)
        L = A.dot(Ti.T)
        (f, Gq) = vgQ(L=L)
        G = -L.T.dot(Gq).dot(Ti).T
    for i_try in range(0, max_tries):
        if rotation_method == 'orthogonal':
            M = T.T.dot(G)
            S = (M + M.T) / 2
            Gp = G - T.dot(S)
        else:
            Gp = G - T.dot(np.diag(np.sum(T * G, axis=0)))
        s = np.linalg.norm(Gp, 'fro')
        table.append([i_try, f, np.log10(s), al])
        if s < tol:
            break
        al = 2 * al
        for i in range(11):
            X = T - al * Gp
            if rotation_method == 'orthogonal':
                (U, D, V) = np.linalg.svd(X, full_matrices=False)
                Tt = U.dot(V)
            else:
                v = 1 / np.sqrt(np.sum(X ** 2, axis=0))
                Tt = X.dot(np.diag(v))
            if derivative_free:
                ft = ff(T=Tt, A=A, L=None)
            elif rotation_method == 'orthogonal':
                L = A.dot(Tt)
                (ft, Gq) = vgQ(L=L)
            else:
                Ti = np.linalg.inv(Tt)
                L = A.dot(Ti.T)
                (ft, Gq) = vgQ(L=L)
            if ft < f - 0.5 * s ** 2 * al:
                break
            al = al / 2
        T = Tt
        f = ft
        if derivative_free:
            G = Gff(T)
        elif rotation_method == 'orthogonal':
            G = A.T.dot(Gq)
        else:
            G = -L.T.dot(Gq).dot(Ti).T
    Th = T
    Lh = rotateA(A, T, rotation_method=rotation_method)
    Phi = T.T.dot(T)
    return (Lh, Phi, Th, table)

def Gf(T, ff):
    if False:
        i = 10
        return i + 15
    '\n    Subroutine for the gradient of f using numerical derivatives.\n    '
    k = T.shape[0]
    ep = 0.0001
    G = np.zeros((k, k))
    for r in range(k):
        for s in range(k):
            dT = np.zeros((k, k))
            dT[r, s] = ep
            G[r, s] = (ff(T + dT) - ff(T - dT)) / (2 * ep)
    return G

def rotateA(A, T, rotation_method='orthogonal'):
    if False:
        print('Hello World!')
    '\n    For orthogonal rotation methods :math:`L=AT`, where :math:`T` is an\n    orthogonal matrix. For oblique rotation matrices :math:`L=A(T^*)^{-1}`,\n    where :math:`T` is a normal matrix, i.e., :math:`TT^*=T^*T`. Oblique\n    rotations relax the orthogonality constraint in order to gain simplicity\n    in the interpretation.\n    '
    if rotation_method == 'orthogonal':
        L = A.dot(T)
    elif rotation_method == 'oblique':
        L = A.dot(np.linalg.inv(T.T))
    else:
        raise ValueError('rotation_method should be one of {orthogonal, oblique}')
    return L

def oblimin_objective(L=None, A=None, T=None, gamma=0, rotation_method='orthogonal', return_gradient=True):
    if False:
        return 10
    '\n    Objective function for the oblimin family for orthogonal or\n    oblique rotation wich minimizes:\n\n    .. math::\n        \\phi(L) = \\frac{1}{4}(L\\circ L,(I-\\gamma C)(L\\circ L)N),\n\n    where :math:`L` is a :math:`p\\times k` matrix, :math:`N` is\n    :math:`k\\times k`\n    matrix with zeros on the diagonal and ones elsewhere, :math:`C` is a\n    :math:`p\\times p` matrix with elements equal to :math:`1/p`,\n    :math:`(X,Y)=\\operatorname{Tr}(X^*Y)` is the Frobenius norm and\n    :math:`\\circ`\n    is the element-wise product or Hadamard product.\n\n    The gradient is given by\n\n    .. math::\n        L\\circ\\left[(I-\\gamma C) (L \\circ L)N\\right].\n\n    Either :math:`L` should be provided or :math:`A` and :math:`T` should be\n    provided.\n\n    For orthogonal rotations :math:`L` satisfies\n\n    .. math::\n        L =  AT,\n\n    where :math:`T` is an orthogonal matrix. For oblique rotations :math:`L`\n    satisfies\n\n    .. math::\n        L =  A(T^*)^{-1},\n\n    where :math:`T` is a normal matrix.\n\n    The oblimin family is parametrized by the parameter :math:`\\gamma`. For\n    orthogonal rotations:\n\n    * :math:`\\gamma=0` corresponds to quartimax,\n    * :math:`\\gamma=\\frac{1}{2}` corresponds to biquartimax,\n    * :math:`\\gamma=1` corresponds to varimax,\n    * :math:`\\gamma=\\frac{1}{p}` corresponds to equamax.\n    For oblique rotations rotations:\n\n    * :math:`\\gamma=0` corresponds to quartimin,\n    * :math:`\\gamma=\\frac{1}{2}` corresponds to biquartimin.\n\n    Parameters\n    ----------\n    L : numpy matrix (default None)\n        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`\n    A : numpy matrix (default None)\n        non rotated factors\n    T : numpy matrix (default None)\n        rotation matrix\n    gamma : float (default 0)\n        a parameter\n    rotation_method : str\n        should be one of {orthogonal, oblique}\n    return_gradient : bool (default True)\n        toggles return of gradient\n    '
    if L is None:
        assert A is not None and T is not None
        L = rotateA(A, T, rotation_method=rotation_method)
    (p, k) = L.shape
    L2 = L ** 2
    N = np.ones((k, k)) - np.eye(k)
    if np.isclose(gamma, 0):
        X = L2.dot(N)
    else:
        C = np.ones((p, p)) / p
        X = (np.eye(p) - gamma * C).dot(L2).dot(N)
    phi = np.sum(L2 * X) / 4
    if return_gradient:
        Gphi = L * X
        return (phi, Gphi)
    else:
        return phi

def orthomax_objective(L=None, A=None, T=None, gamma=0, return_gradient=True):
    if False:
        while True:
            i = 10
    '\n    Objective function for the orthomax family for orthogonal\n    rotation wich minimizes the following objective:\n\n    .. math::\n        \\phi(L) = -\\frac{1}{4}(L\\circ L,(I-\\gamma C)(L\\circ L)),\n\n    where :math:`0\\leq\\gamma\\leq1`, :math:`L` is a :math:`p\\times k` matrix,\n    :math:`C` is a  :math:`p\\times p` matrix with elements equal to\n    :math:`1/p`,\n    :math:`(X,Y)=\\operatorname{Tr}(X^*Y)` is the Frobenius norm and\n    :math:`\\circ` is the element-wise product or Hadamard product.\n\n    Either :math:`L` should be provided or :math:`A` and :math:`T` should be\n    provided.\n\n    For orthogonal rotations :math:`L` satisfies\n\n    .. math::\n        L =  AT,\n\n    where :math:`T` is an orthogonal matrix.\n\n    The orthomax family is parametrized by the parameter :math:`\\gamma`:\n\n    * :math:`\\gamma=0` corresponds to quartimax,\n    * :math:`\\gamma=\\frac{1}{2}` corresponds to biquartimax,\n    * :math:`\\gamma=1` corresponds to varimax,\n    * :math:`\\gamma=\\frac{1}{p}` corresponds to equamax.\n\n    Parameters\n    ----------\n    L : numpy matrix (default None)\n        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`\n    A : numpy matrix (default None)\n        non rotated factors\n    T : numpy matrix (default None)\n        rotation matrix\n    gamma : float (default 0)\n        a parameter\n    return_gradient : bool (default True)\n        toggles return of gradient\n    '
    assert 0 <= gamma <= 1, 'Gamma should be between 0 and 1'
    if L is None:
        assert A is not None and T is not None
        L = rotateA(A, T, rotation_method='orthogonal')
    (p, k) = L.shape
    L2 = L ** 2
    if np.isclose(gamma, 0):
        X = L2
    else:
        C = np.ones((p, p)) / p
        X = (np.eye(p) - gamma * C).dot(L2)
    phi = -np.sum(L2 * X) / 4
    if return_gradient:
        Gphi = -L * X
        return (phi, Gphi)
    else:
        return phi

def CF_objective(L=None, A=None, T=None, kappa=0, rotation_method='orthogonal', return_gradient=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Objective function for the Crawford-Ferguson family for orthogonal\n    and oblique rotation wich minimizes the following objective:\n\n    .. math::\n        \\phi(L) =\\frac{1-\\kappa}{4} (L\\circ L,(L\\circ L)N)\n                  -\\frac{1}{4}(L\\circ L,M(L\\circ L)),\n\n    where :math:`0\\leq\\kappa\\leq1`, :math:`L` is a :math:`p\\times k` matrix,\n    :math:`N` is :math:`k\\times k` matrix with zeros on the diagonal and ones\n    elsewhere,\n    :math:`M` is :math:`p\\times p` matrix with zeros on the diagonal and ones\n    elsewhere\n    :math:`(X,Y)=\\operatorname{Tr}(X^*Y)` is the Frobenius norm and\n    :math:`\\circ` is the element-wise product or Hadamard product.\n\n    The gradient is given by\n\n    .. math::\n       d\\phi(L) = (1-\\kappa) L\\circ\\left[(L\\circ L)N\\right]\n                   -\\kappa L\\circ \\left[M(L\\circ L)\\right].\n\n    Either :math:`L` should be provided or :math:`A` and :math:`T` should be\n    provided.\n\n    For orthogonal rotations :math:`L` satisfies\n\n    .. math::\n        L =  AT,\n\n    where :math:`T` is an orthogonal matrix. For oblique rotations :math:`L`\n    satisfies\n\n    .. math::\n        L =  A(T^*)^{-1},\n\n    where :math:`T` is a normal matrix.\n\n    For orthogonal rotations the oblimin (and orthomax) family of rotations is\n    equivalent to the Crawford-Ferguson family. To be more precise:\n\n    * :math:`\\kappa=0` corresponds to quartimax,\n    * :math:`\\kappa=\\frac{1}{p}` corresponds to variamx,\n    * :math:`\\kappa=\\frac{k-1}{p+k-2}` corresponds to parsimax,\n    * :math:`\\kappa=1` corresponds to factor parsimony.\n\n    Parameters\n    ----------\n    L : numpy matrix (default None)\n        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`\n    A : numpy matrix (default None)\n        non rotated factors\n    T : numpy matrix (default None)\n        rotation matrix\n    gamma : float (default 0)\n        a parameter\n    rotation_method : str\n        should be one of {orthogonal, oblique}\n    return_gradient : bool (default True)\n        toggles return of gradient\n    '
    assert 0 <= kappa <= 1, 'Kappa should be between 0 and 1'
    if L is None:
        assert A is not None and T is not None
        L = rotateA(A, T, rotation_method=rotation_method)
    (p, k) = L.shape
    L2 = L ** 2
    X = None
    if not np.isclose(kappa, 1):
        N = np.ones((k, k)) - np.eye(k)
        X = (1 - kappa) * L2.dot(N)
    if not np.isclose(kappa, 0):
        M = np.ones((p, p)) - np.eye(p)
        if X is None:
            X = kappa * M.dot(L2)
        else:
            X += kappa * M.dot(L2)
    phi = np.sum(L2 * X) / 4
    if return_gradient:
        Gphi = L * X
        return (phi, Gphi)
    else:
        return phi

def vgQ_target(H, L=None, A=None, T=None, rotation_method='orthogonal'):
    if False:
        i = 10
        return i + 15
    '\n    Subroutine for the value of vgQ using orthogonal or oblique rotation\n    towards a target matrix, i.e., we minimize:\n\n    .. math::\n        \\phi(L) =\\frac{1}{2}\\|L-H\\|^2\n\n    and the gradient is given by\n\n    .. math::\n        d\\phi(L)=L-H.\n\n    Either :math:`L` should be provided or :math:`A` and :math:`T` should be\n    provided.\n\n    For orthogonal rotations :math:`L` satisfies\n\n    .. math::\n        L =  AT,\n\n    where :math:`T` is an orthogonal matrix. For oblique rotations :math:`L`\n    satisfies\n\n    .. math::\n        L =  A(T^*)^{-1},\n\n    where :math:`T` is a normal matrix.\n\n    Parameters\n    ----------\n    H : numpy matrix\n        target matrix\n    L : numpy matrix (default None)\n        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`\n    A : numpy matrix (default None)\n        non rotated factors\n    T : numpy matrix (default None)\n        rotation matrix\n    rotation_method : str\n        should be one of {orthogonal, oblique}\n    '
    if L is None:
        assert A is not None and T is not None
        L = rotateA(A, T, rotation_method=rotation_method)
    q = np.linalg.norm(L - H, 'fro') ** 2
    Gq = 2 * (L - H)
    return (q, Gq)

def ff_target(H, L=None, A=None, T=None, rotation_method='orthogonal'):
    if False:
        while True:
            i = 10
    '\n    Subroutine for the value of f using (orthogonal or oblique) rotation\n    towards a target matrix, i.e., we minimize:\n\n    .. math::\n        \\phi(L) =\\frac{1}{2}\\|L-H\\|^2.\n\n    Either :math:`L` should be provided or :math:`A` and :math:`T` should be\n    provided. For orthogonal rotations :math:`L` satisfies\n\n    .. math::\n        L =  AT,\n\n    where :math:`T` is an orthogonal matrix. For oblique rotations\n    :math:`L` satisfies\n\n    .. math::\n        L =  A(T^*)^{-1},\n\n    where :math:`T` is a normal matrix.\n\n    Parameters\n    ----------\n    H : numpy matrix\n        target matrix\n    L : numpy matrix (default None)\n        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`\n    A : numpy matrix (default None)\n        non rotated factors\n    T : numpy matrix (default None)\n        rotation matrix\n    rotation_method : str\n        should be one of {orthogonal, oblique}\n    '
    if L is None:
        assert A is not None and T is not None
        L = rotateA(A, T, rotation_method=rotation_method)
    return np.linalg.norm(L - H, 'fro') ** 2

def vgQ_partial_target(H, W=None, L=None, A=None, T=None):
    if False:
        while True:
            i = 10
    '\n    Subroutine for the value of vgQ using orthogonal rotation towards a partial\n    target matrix, i.e., we minimize:\n\n    .. math::\n        \\phi(L) =\\frac{1}{2}\\|W\\circ(L-H)\\|^2,\n\n    where :math:`\\circ` is the element-wise product or Hadamard product and\n    :math:`W` is a matrix whose entries can only be one or zero. The gradient\n    is given by\n\n    .. math::\n        d\\phi(L)=W\\circ(L-H).\n\n    Either :math:`L` should be provided or :math:`A` and :math:`T` should be\n    provided.\n\n    For orthogonal rotations :math:`L` satisfies\n\n    .. math::\n        L =  AT,\n\n    where :math:`T` is an orthogonal matrix.\n\n    Parameters\n    ----------\n    H : numpy matrix\n        target matrix\n    W : numpy matrix (default matrix with equal weight one for all entries)\n        matrix with weights, entries can either be one or zero\n    L : numpy matrix (default None)\n        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`\n    A : numpy matrix (default None)\n        non rotated factors\n    T : numpy matrix (default None)\n        rotation matrix\n    '
    if W is None:
        return vgQ_target(H, L=L, A=A, T=T)
    if L is None:
        assert A is not None and T is not None
        L = rotateA(A, T, rotation_method='orthogonal')
    q = np.linalg.norm(W * (L - H), 'fro') ** 2
    Gq = 2 * W * (L - H)
    return (q, Gq)

def ff_partial_target(H, W=None, L=None, A=None, T=None):
    if False:
        print('Hello World!')
    '\n    Subroutine for the value of vgQ using orthogonal rotation towards a partial\n    target matrix, i.e., we minimize:\n\n    .. math::\n        \\phi(L) =\\frac{1}{2}\\|W\\circ(L-H)\\|^2,\n\n    where :math:`\\circ` is the element-wise product or Hadamard product and\n    :math:`W` is a matrix whose entries can only be one or zero. Either\n    :math:`L` should be provided or :math:`A` and :math:`T` should be provided.\n\n    For orthogonal rotations :math:`L` satisfies\n\n    .. math::\n        L =  AT,\n\n    where :math:`T` is an orthogonal matrix.\n\n    Parameters\n    ----------\n    H : numpy matrix\n        target matrix\n    W : numpy matrix (default matrix with equal weight one for all entries)\n        matrix with weights, entries can either be one or zero\n    L : numpy matrix (default None)\n        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`\n    A : numpy matrix (default None)\n        non rotated factors\n    T : numpy matrix (default None)\n        rotation matrix\n    '
    if W is None:
        return ff_target(H, L=L, A=A, T=T)
    if L is None:
        assert A is not None and T is not None
        L = rotateA(A, T, rotation_method='orthogonal')
    q = np.linalg.norm(W * (L - H), 'fro') ** 2
    return q