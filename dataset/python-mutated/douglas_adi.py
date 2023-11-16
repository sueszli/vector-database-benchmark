"""Douglas ADI method for solving multidimensional parabolic PDEs."""
import numpy as np
import tensorflow.compat.v2 as tf
from tf_quant_finance.math.pde.steppers.multidim_parabolic_equation_stepper import multidim_parabolic_equation_step

def douglas_adi_step(theta=0.5):
    if False:
        return 10
    'Creates a stepper function with Crank-Nicolson time marching scheme.\n\n  Douglas ADI scheme is the simplest time marching scheme for solving parabolic\n  PDEs with multiple spatial dimensions. The time step consists of several\n  substeps: the first one is fully explicit, and the following `N` steps are\n  implicit with respect to contributions of one of the `N` axes (hence "ADI" -\n  alternating direction implicit). See `douglas_adi_scheme` below for more\n  details.\n\n  Args:\n    theta: positive Number. `theta = 0` corresponds to fully explicit scheme.\n    The larger `theta` the stronger are the corrections by the implicit\n    substeps. The recommended value is `theta = 0.5`, because the scheme is\n    second order accurate in that case, unless mixed second derivative terms are\n    present in the PDE.\n  Returns:\n    Callable to be used in finite-difference PDE solvers (see fd_solvers.py).\n  '

    def _step_fn(time, next_time, coord_grid, value_grid, boundary_conditions, second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn, inner_second_order_coeff_fn, inner_first_order_coeff_fn, num_steps_performed, dtype=None, name=None):
        if False:
            print('Hello World!')
        'Performs the step.'
        del num_steps_performed
        name = name or 'douglas_adi_step'
        return multidim_parabolic_equation_step(time, next_time, coord_grid, value_grid, boundary_conditions, douglas_adi_scheme(theta), second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn, inner_second_order_coeff_fn, inner_first_order_coeff_fn, dtype=dtype, name=name)
    return _step_fn

def douglas_adi_scheme(theta):
    if False:
        i = 10
        return i + 15
    'Applies Douglas time marching scheme (see [1] and Eq. 3.1 in [2]).\n\n  Time marching schemes solve the space-discretized equation\n  `du_inner/dt = A(t) u_inner(t) + A_mixed u(t) + b(t)`,\n  where `u`, `u_inner` and `b` are vectors and `A`, `A_mixed` are matrices.\n  `u_inner` is `u` with all boundaries having Robin boundary conditions\n  trimmed and `A_mixed` are contributions of mixed derivative terms.\n  See more details in multidim_parabolic_equation_stepper.py.\n\n  In Douglas scheme (as well as other ADI schemes), the matrix `A` is\n  represented as sum `A = sum_i A_i`. `A_i` is the contribution of\n  terms with partial derivatives w.r.t. dimension `i`. The shift term is split\n  evenly between `A_i`. Similarly, inhomogeneous term is represented as sum\n  `b = sum_i b_i`, where `b_i` comes from boundary conditions on boundary\n  orthogonal to dimension `i`.\n\n  Given the current values vector u(t1), the step is defined as follows\n  (using the notation of Eq. 3.1 in [2]):\n  `Y_0 = (1 + (A(t1) + A_mixed(t1)) dt) U_{n-1} + b(t1) dt`,\n  `Y_j = Y_{j-1} + theta dt (A_j(t2) Y_j - A_j(t1) U_{n-1} + b_j(t2) - b_j(t1))`\n  for each spatial dimension `j`, and\n  `U_n = Y_{n_dims-1}`.\n\n  Here the parameter `theta` is a non-negative number, `U_{n-1} = u(t1)`,\n  `U_n = u(t2)`, and `dt = t2 - t1`.\n\n  Note: Douglas scheme is only first-order accurate if mixed terms are\n  present. More advanced schemes, such as Craig-Sneyd scheme, are needed to\n  achieve the second-order accuracy.\n\n  #### References:\n  [1] Douglas Jr., Jim (1962), "Alternating direction methods for three space\n    variables", Numerische Mathematik, 4 (1): 41-63\n  [2] Tinne Haentjens, Karek J. in\'t Hout. ADI finite difference schemes for\n    the Heston-Hull-White PDE. https://arxiv.org/abs/1111.4087\n\n  Args:\n    theta: Number between 0 and 1 (see the step definition above). `theta = 0`\n      corresponds to fully-explicit scheme.\n\n  Returns:\n    A callable consumes the following arguments by keyword:\n      1. inner_value_grid: Grid of solution values at the current time of\n        the same `dtype` as `value_grid` and shape of\n        `batch_shape` + `[d_1 - 2 + n_def_i , ..., d_n -2 + n_def_i]`\n        where `d_i` is the number of space discretization points along dimension\n        `i` and `n_def_i` is the number of default boundaries along that\n        dimension. `n_def_i` takes values 0, 1, 2 (default boundary),\n      2. t1: Time before the step.\n      3. t2: Time after the step.\n      4. equation_params_fn: A callable that takes a scalar `Tensor` argument\n        representing time, and returns a tuple of two objects:\n          * First object is a nested list `L` such that `L[i][i]` is a tuple of\n          three `Tensor`s, main, upper, and lower diagonal of the tridiagonal\n          matrix `A` in a direction `i`. Each element `L[i][j]` corresponds\n          to the mixed terms and is either None (meaning there are no mixed\n          terms present) or a tuple of `Tensor`s representing contributions of\n          mixed terms in directions (i + 1, j + 1), (i + 1, j - 1),\n          (i - 1, j + 1), and (i - 1, j - 1).\n          * The second object is a tuple of inhomogeneous terms for each\n          dimension.\n        All of the `Tensor`s are of the same `dtype` as `inner_value_grid` and\n        of the shape broadcastable with the shape of `inner_value_grid`.\n      5. A callable that accepts a `Tensor` of shape `inner_value_grid` and\n        appends boundaries according to the boundary conditions, i.e. transforms\n        `u_inner` to `u`.\n      6. n_dims: A Python integer, the spatial dimension of the PDE.\n      7. has_default_lower_boundary: A Python list of booleans of length\n        `n_dims`. List indices enumerate the dimensions with `True` values\n        marking default lower boundary condition along corresponding dimensions,\n        and  `False` values indicating Robin boundary conditions.\n      8. has_default_upper_boundary: Similar to has_default_lower_boundary, but\n        for upper boundaries.\n    The callable returns a `Tensor` of the same shape and `dtype` a\n    `values_grid` and represents an approximate solution `u(t2)`.\n  '
    if theta < 0 or theta > 1:
        raise ValueError('Theta should be in the interval [0, 1].')

    def _marching_scheme(value_grid, t1, t2, equation_params_fn, append_boundaries_fn, n_dims, has_default_lower_boundary, has_default_upper_boundary):
        if False:
            print('Hello World!')
        'Constructs the Douglas ADI time marching scheme.'
        current_grid = value_grid
        (matrix_params_t1, inhomog_terms_t1) = equation_params_fn(t1)
        (matrix_params_t2, inhomog_terms_t2) = equation_params_fn(t2)
        value_grid_with_boundaries = append_boundaries_fn(value_grid)
        for i in range(n_dims - 1):
            for j in range(i + 1, n_dims):
                mixed_term = matrix_params_t1[i][j]
                if mixed_term is not None:
                    current_grid += _apply_mixed_term_explicitly(value_grid_with_boundaries, mixed_term, t2 - t1, i, j, has_default_lower_boundary, has_default_upper_boundary, n_dims)
        explicit_contributions = []
        for i in range(n_dims):
            (superdiag, diag, subdiag) = (matrix_params_t1[i][i][d] for d in range(3))
            contribution = _apply_tridiag_matrix_explicitly(value_grid, superdiag, diag, subdiag, i, n_dims) * (t2 - t1)
            explicit_contributions.append(contribution)
            current_grid += contribution
        for inhomog_term in inhomog_terms_t1:
            current_grid += inhomog_term * (t2 - t1)
        if theta == 0:
            return current_grid
        for i in range(n_dims):
            inhomog_term_delta = inhomog_terms_t2[i] - inhomog_terms_t1[i]
            (superdiag, diag, subdiag) = (matrix_params_t2[i][i][d] for d in range(3))
            current_grid = _apply_correction(theta, current_grid, explicit_contributions[i], superdiag, diag, subdiag, inhomog_term_delta, t1, t2, i, n_dims)
        return current_grid
    return _marching_scheme

def _apply_mixed_term_explicitly(values_with_boundaries, mixed_term, delta_t, dim1, dim2, has_default_lower_boundary, has_default_upper_boundary, n_dims):
    if False:
        for i in range(10):
            print('nop')
    'Applies mixed term explicitly.'
    (mixed_term_pp, mixed_term_pm, mixed_term_mp, mixed_term_mm) = mixed_term
    batch_rank = values_with_boundaries.shape.rank - n_dims
    paddings = batch_rank * [[0, 0]]
    for dim in range(n_dims):
        lower = 1 if has_default_lower_boundary[dim] else 0
        upper = 1 if has_default_upper_boundary[dim] else 0
        paddings += [[lower, upper]]
    values_with_boundaries = tf.pad(values_with_boundaries, paddings=paddings)

    def create_trimming_shifts(dim1_shift, dim2_shift):
        if False:
            while True:
                i = 10
        shifts = [0] * n_dims
        shifts[dim1] = dim1_shift
        shifts[dim2] = dim2_shift
        return shifts
    values_pp = _trim_boundaries(values_with_boundaries, from_dim=batch_rank, shifts=create_trimming_shifts(1, 1))
    values_mm = _trim_boundaries(values_with_boundaries, from_dim=batch_rank, shifts=create_trimming_shifts(-1, -1))
    values_pm = _trim_boundaries(values_with_boundaries, from_dim=batch_rank, shifts=create_trimming_shifts(1, -1))
    values_mp = _trim_boundaries(values_with_boundaries, from_dim=batch_rank, shifts=create_trimming_shifts(-1, 1))
    return (mixed_term_mm * values_mm + mixed_term_mp * values_mp + mixed_term_pm * values_pm + mixed_term_pp * values_pp) * delta_t

def _apply_tridiag_matrix_explicitly(values, superdiag, diag, subdiag, dim, n_dims):
    if False:
        for i in range(10):
            print('nop')
    'Applies tridiagonal matrix explicitly.'
    perm = _get_permutation(values, n_dims, dim)
    if perm is not None:
        values = tf.transpose(values, perm)
        (superdiag, diag, subdiag) = (tf.transpose(c, perm) for c in (superdiag, diag, subdiag))
    values = tf.squeeze(tf.linalg.tridiagonal_matmul((superdiag, diag, subdiag), tf.expand_dims(values, -1), diagonals_format='sequence'), -1)
    if perm is not None:
        values = tf.transpose(values, perm)
    return values

def _apply_correction(theta, values, explicit_contribution, superdiag, diag, subdiag, inhomog_term_delta, t1, t2, dim, n_dims):
    if False:
        return 10
    'Applies correction for the given dimension.'
    rhs = values - theta * explicit_contribution + theta * inhomog_term_delta * (t2 - t1)
    perm = _get_permutation(values, n_dims, dim)
    if perm is not None:
        rhs = tf.transpose(rhs, perm)
        (superdiag, diag, subdiag) = (tf.transpose(c, perm) for c in (superdiag, diag, subdiag))
    subdiag = -theta * subdiag * (t2 - t1)
    diag = 1 - theta * diag * (t2 - t1)
    superdiag = -theta * superdiag * (t2 - t1)
    result = tf.linalg.tridiagonal_solve((superdiag, diag, subdiag), rhs, diagonals_format='sequence', partial_pivoting=False)
    if perm is not None:
        result = tf.transpose(result, perm)
    return result

def _get_permutation(tensor, n_dims, active_dim):
    if False:
        print('Hello World!')
    'Returns the permutation that swaps the active and the last dimensions.\n\n  Args:\n    tensor: `Tensor` having a statically known rank.\n    n_dims: Number of spatial dimensions.\n    active_dim: The active spatial dimension.\n\n  Returns:\n    A list representing the permutation, or `None` if no permutation needed.\n\n  For example, with \'tensor` having rank 5, `n_dims = 3` and `active_dim = 1`\n  yields [0, 1, 2, 4, 3]. Explanation: we start with [0, 1, 2, 3, 4], where the\n  last n_dims=3 dimensions are spatial dimensions, and the first two are batch\n  dimensions. Among the spatial dimensions, we take the one at index 1, which\n  is "3", and swap it with the last dimension "4".\n  '
    if not tensor.shape:
        raise ValueError("Tensor's rank should be static")
    rank = len(tensor.shape)
    batch_rank = rank - n_dims
    if active_dim == n_dims - 1:
        return None
    perm = np.arange(rank)
    perm[rank - 1] = batch_rank + active_dim
    perm[batch_rank + active_dim] = rank - 1
    return perm

def _trim_boundaries(tensor, from_dim, shifts=None):
    if False:
        print('Hello World!')
    'Trims tensor boundaries starting from given dimension.'
    rank = tensor.shape.rank
    slices = rank * [slice(None)]
    for i in range(from_dim, rank):
        slice_begin = 1
        slice_end = -1
        if shifts is not None:
            shift = shifts[i - from_dim]
            slice_begin += shift
            slice_end += shift
        if isinstance(slice_end, int) and slice_end == 0:
            slice_end = None
        slices[i] = slice(slice_begin, slice_end)
    res = tensor[slices]
    return res
_all__ = ['douglas_adi_step', 'douglas_adi_scheme']