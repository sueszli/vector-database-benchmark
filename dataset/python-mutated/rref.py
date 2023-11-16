from sympy.polys.domains import ZZ
from sympy.polys.matrices.sdm import SDM, sdm_irref, sdm_rref_den
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.dense import ddm_irref, ddm_irref_den

def _dm_rref(M, *, method='auto'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the reduced row echelon form of a ``DomainMatrix``.\n\n    This function is the implementation of :meth:`DomainMatrix.rref`.\n\n    Chooses the best algorithm depending on the domain, shape, and sparsity of\n    the matrix as well as things like the bit count in the case of :ref:`ZZ` or\n    :ref:`QQ`. The result is returned over the field associated with the domain\n    of the Matrix.\n\n    See Also\n    ========\n\n    sympy.polys.matrices.domainmatrix.DomainMatrix.rref\n        The ``DomainMatrix`` method that calls this function.\n    sympy.polys.matrices.rref._dm_rref_den\n        Alternative function for computing RREF with denominator.\n    '
    (method, use_fmt) = _dm_rref_choose_method(M, method, denominator=False)
    (M, old_fmt) = _dm_to_fmt(M, use_fmt)
    if method == 'GJ':
        Mf = _to_field(M)
        (M_rref, pivots) = _dm_rref_GJ(Mf)
    elif method == 'FF':
        (M_rref_f, den, pivots) = _dm_rref_den_FF(M)
        M_rref = _to_field(M_rref_f) / den
    elif method == 'CD':
        (_, Mr) = M.clear_denoms(convert=True)
        (M_rref_f, den, pivots) = _dm_rref_den_FF(Mr)
        M_rref = _to_field(M_rref_f) / den
    else:
        raise ValueError(f'Unknown method for rref: {method}')
    (M_rref, _) = _dm_to_fmt(M_rref, old_fmt)
    return (M_rref, pivots)

def _dm_rref_den(M, *, keep_domain=True, method='auto'):
    if False:
        i = 10
        return i + 15
    '\n    Compute the reduced row echelon form of a ``DomainMatrix`` with denominator.\n\n    This function is the implementation of :meth:`DomainMatrix.rref_den`.\n\n    Chooses the best algorithm depending on the domain, shape, and sparsity of\n    the matrix as well as things like the bit count in the case of :ref:`ZZ` or\n    :ref:`QQ`. The result is returned over the same domain as the input matrix\n    unless ``keep_domain=False`` in which case the result might be over an\n    associated ring or field domain.\n\n    See Also\n    ========\n\n    sympy.polys.matrices.domainmatrix.DomainMatrix.rref_den\n        The ``DomainMatrix`` method that calls this function.\n    sympy.polys.matrices.rref._dm_rref\n        Alternative function for computing RREF without denominator.\n    '
    (method, use_fmt) = _dm_rref_choose_method(M, method, denominator=True)
    (M, old_fmt) = _dm_to_fmt(M, use_fmt)
    if method == 'FF':
        (M_rref, den, pivots) = _dm_rref_den_FF(M)
    elif method == 'GJ':
        (M_rref_f, pivots) = _dm_rref_GJ(_to_field(M))
        if keep_domain and M_rref_f.domain != M.domain:
            (_, M_rref) = M_rref_f.clear_denoms(convert=True)
            if pivots:
                den = M_rref[0, pivots[0]].element
            else:
                den = M_rref.domain.one
        else:
            M_rref = M_rref_f
            den = M_rref.domain.one
    elif method == 'CD':
        (_, Mr) = M.clear_denoms(convert=True)
        (M_rref_r, den, pivots) = _dm_rref_den_FF(Mr)
        if keep_domain and M_rref_r.domain != M.domain:
            M_rref = _to_field(M_rref_r) / den
            den = M.domain.one
        else:
            M_rref = M_rref_r
            if pivots:
                den = M_rref[0, pivots[0]].element
            else:
                den = M_rref.domain.one
    else:
        raise ValueError(f'Unknown method for rref: {method}')
    (M_rref, _) = _dm_to_fmt(M_rref, old_fmt)
    return (M_rref, den, pivots)

def _dm_to_fmt(M, fmt):
    if False:
        i = 10
        return i + 15
    'Convert a matrix to the given format and return the old format.'
    old_fmt = M.rep.fmt
    if old_fmt == fmt:
        pass
    elif fmt == 'dense':
        M = M.to_dense()
    elif fmt == 'sparse':
        M = M.to_sparse()
    else:
        raise ValueError(f'Unknown format: {fmt}')
    return (M, old_fmt)

def _dm_rref_GJ(M):
    if False:
        i = 10
        return i + 15
    'Compute RREF using Gauss-Jordan elimination with division.'
    if M.rep.fmt == 'sparse':
        return _dm_rref_GJ_sparse(M)
    else:
        return _dm_rref_GJ_dense(M)

def _dm_rref_den_FF(M):
    if False:
        return 10
    'Compute RREF using fraction-free Gauss-Jordan elimination.'
    if M.rep.fmt == 'sparse':
        return _dm_rref_den_FF_sparse(M)
    else:
        return _dm_rref_den_FF_dense(M)

def _dm_rref_GJ_sparse(M):
    if False:
        i = 10
        return i + 15
    'Compute RREF using sparse Gauss-Jordan elimination with division.'
    (M_rref_d, pivots, _) = sdm_irref(M.rep)
    M_rref_sdm = SDM(M_rref_d, M.shape, M.domain)
    pivots = tuple(pivots)
    return (M.from_rep(M_rref_sdm), pivots)

def _dm_rref_GJ_dense(M):
    if False:
        return 10
    'Compute RREF using dense Gauss-Jordan elimination with division.'
    partial_pivot = M.domain.is_RR or M.domain.is_CC
    ddm = M.rep.to_ddm().copy()
    pivots = ddm_irref(ddm, _partial_pivot=partial_pivot)
    M_rref_ddm = DDM(ddm, M.shape, M.domain)
    pivots = tuple(pivots)
    return (M.from_rep(M_rref_ddm.to_dfm_or_ddm()), pivots)

def _dm_rref_den_FF_sparse(M):
    if False:
        i = 10
        return i + 15
    'Compute RREF using sparse fraction-free Gauss-Jordan elimination.'
    (M_rref_d, den, pivots) = sdm_rref_den(M.rep, M.domain)
    M_rref_sdm = SDM(M_rref_d, M.shape, M.domain)
    pivots = tuple(pivots)
    return (M.from_rep(M_rref_sdm), den, pivots)

def _dm_rref_den_FF_dense(M):
    if False:
        return 10
    'Compute RREF using sparse fraction-free Gauss-Jordan elimination.'
    ddm = M.rep.to_ddm().copy()
    (den, pivots) = ddm_irref_den(ddm, M.domain)
    M_rref_ddm = DDM(ddm, M.shape, M.domain)
    pivots = tuple(pivots)
    return (M.from_rep(M_rref_ddm.to_dfm_or_ddm()), den, pivots)

def _dm_rref_choose_method(M, method, *, denominator=False):
    if False:
        print('Hello World!')
    'Choose the fastest method for computing RREF for M.'
    if method != 'auto':
        if method.endswith('_dense'):
            method = method[:-len('_dense')]
            use_fmt = 'dense'
        else:
            use_fmt = 'sparse'
    else:
        use_fmt = 'sparse'
        K = M.domain
        if K.is_ZZ:
            method = _dm_rref_choose_method_ZZ(M, denominator=denominator)
        elif K.is_QQ:
            method = _dm_rref_choose_method_QQ(M, denominator=denominator)
        elif K.is_RR or K.is_CC:
            method = 'GJ'
            use_fmt = 'dense'
        elif K.is_EX and M.rep.fmt == 'dense' and (not denominator):
            method = 'GJ'
            use_fmt = 'dense'
        elif denominator:
            method = 'FF'
        else:
            method = 'GJ'
    return (method, use_fmt)

def _dm_rref_choose_method_QQ(M, *, denominator=False):
    if False:
        for i in range(10):
            print('nop')
    'Choose the fastest method for computing RREF over QQ.'
    (density, _, ncols) = _dm_row_density(M)
    if density < min(5, ncols / 2):
        return 'GJ'
    (numers, denoms) = _dm_QQ_numers_denoms(M)
    numer_bits = max([n.bit_length() for n in numers], default=1)
    denom_lcm = ZZ.one
    for d in denoms:
        denom_lcm = ZZ.lcm(denom_lcm, d)
        if denom_lcm.bit_length() > 5 * numer_bits:
            return 'GJ'
    if denom_lcm.bit_length() < 50:
        return 'CD'
    else:
        return 'FF'

def _dm_rref_choose_method_ZZ(M, *, denominator=False):
    if False:
        i = 10
        return i + 15
    'Choose the fastest method for computing RREF over ZZ.'
    PARAM = 10000
    (density, nrows_nz, ncols) = _dm_row_density(M)
    if nrows_nz < 10:
        if density < ncols / 2:
            return 'GJ'
        else:
            return 'FF'
    if density < 5:
        return 'GJ'
    elif density > 5 + PARAM / nrows_nz:
        return 'FF'
    elements = _dm_elements(M)
    bits = max([e.bit_length() for e in elements], default=1)
    wideness = max(1, 2 / 3 * ncols / nrows_nz)
    max_density = (5 + PARAM / (nrows_nz * bits ** 2)) * wideness
    if density < max_density:
        return 'GJ'
    else:
        return 'FF'

def _dm_row_density(M):
    if False:
        while True:
            i = 10
    'Density measure for sparse matrices.\n\n    Defines the "density", ``d`` as the average number of non-zero entries per\n    row except ignoring rows that are fully zero. RREF can ignore fully zero\n    rows so they are excluded. By definition ``d >= 1`` except that we define\n    ``d = 0`` for the zero matrix.\n\n    Returns ``(density, nrows_nz, ncols)`` where ``nrows_nz`` counts the number\n    of nonzero rows and ``ncols`` is the number of columns.\n    '
    ncols = M.shape[1]
    rows_nz = M.rep.to_sdm().values()
    if not rows_nz:
        return (0, 0, ncols)
    else:
        nrows_nz = len(rows_nz)
        density = sum(map(len, rows_nz)) / nrows_nz
        return (density, nrows_nz, ncols)

def _dm_elements(M):
    if False:
        return 10
    'Return nonzero elements of a DomainMatrix.'
    (elements, _) = M.to_flat_nz()
    return elements

def _dm_QQ_numers_denoms(Mq):
    if False:
        for i in range(10):
            print('nop')
    'Returns the numerators and denominators of a DomainMatrix over QQ.'
    elements = _dm_elements(Mq)
    numers = [e.numerator for e in elements]
    denoms = [e.denominator for e in elements]
    return (numers, denoms)

def _to_field(M):
    if False:
        for i in range(10):
            print('nop')
    'Convert a DomainMatrix to a field if possible.'
    K = M.domain
    if K.has_assoc_Field:
        return M.to_field()
    else:
        return M