"""Stepper for multidimensional parabolic PDE solving."""
import tensorflow.compat.v2 as tf
from tf_quant_finance import utils

def multidim_parabolic_equation_step(time, next_time, coord_grid, value_grid, boundary_conditions, time_marching_scheme, second_order_coeff_fn=None, first_order_coeff_fn=None, zeroth_order_coeff_fn=None, inner_second_order_coeff_fn=None, inner_first_order_coeff_fn=None, dtype=None, name=None):
    if False:
        i = 10
        return i + 15
    'Performs one step in time to solve a multidimensional PDE.\n\n  Typically one doesn\'t need to use this function directly, unless they have\n  a custom time marching scheme. A simple stepper function for multidimensional\n  PDEs can be found in `douglas_adi.py`.\n\n  The PDE is of the form\n\n  ```None\n    dV/dt + Sum[a_ij d2(A_ij V)/dx_i dx_j, 1 <= i, j <=n] +\n       Sum[b_i d(B_i V)/dx_i, 1 <= i <= n] + c V = 0.\n  ```\n  from time `t0` to time `t1`. The solver can go both forward and backward in\n  time. Here `a_ij`, `A_ij`, `b_i`, `B_i` and `c` are coefficients that may\n  depend on spatial variables `x` and time `t`.\n\n  Here `V` is the unknown function, `V_{...}` denotes partial derivatives\n  w.r.t. dimensions specified in curly brackets, `i` and `j` denote spatial\n  dimensions, `r` is the spatial radius-vector.\n\n  Args:\n    time: Real scalar `Tensor`. The time before the step.\n    next_time: Real scalar `Tensor`. The time after the step.\n    coord_grid: List of `n` rank 1 real `Tensor`s. `n` is the dimension of the\n      domain. The i-th `Tensor` has shape either `[d_i]` or `B + [d_i]` where\n      `d_i` is the size of the grid along axis `i` and `B` is a batch shape. The\n      coordinates of the grid points. Corresponds to the spatial grid `G` above.\n    value_grid: Real `Tensor` containing the function values at time\n      `time` which have to be evolved to time `next_time`. The shape of the\n      `Tensor` must broadcast with `B + [d_1, d_2, ..., d_n]`. `B` is the batch\n      dimensions (one or more), which allow multiple functions (with potentially\n      different boundary/final conditions and PDE coefficients) to be evolved\n      simultaneously.\n    boundary_conditions: The boundary conditions. Only rectangular boundary\n      conditions are supported. A list of tuples of size `n` (space dimension\n      of the PDE). The elements of the Tuple can be either a Python Callable or\n      `None` representing the boundary conditions at the minimum and maximum\n      values of the spatial variable indexed by the position in the list. E.g.,\n      for `n=2`, the length of `boundary_conditions` should be 2,\n      `boundary_conditions[0][0]` describes the boundary `(y_min, x)`, and\n      `boundary_conditions[1][0]`- the boundary `(y, x_min)`. `None` values mean\n      that the second order terms for that dimension on the boundary are assumed\n      to be zero, i.e., if `boundary_conditions[k][0]` is None,\n      \'dV/dt + Sum[a_ij d2(A_ij V)/dx_i dx_j, 1 <= i, j <=n, i!=k+1, j!=k+1] +\n         Sum[b_i d(B_i V)/dx_i, 1 <= i <= n] + c V = 0.\'\n      For not `None` values, the boundary conditions are accepted in the form\n      `alpha(t, x) V + beta(t, x) V_n = gamma(t, x)`, where `V_n` is the\n      derivative with respect to the exterior normal to the boundary.\n      Each callable receives the current time `t` and the `coord_grid` at the\n      current time, and should return a tuple of `alpha`, `beta`, and `gamma`.\n      Each can be a number, a zero-rank `Tensor` or a `Tensor` whose shape is\n      the grid shape with the corresponding dimension removed.\n      For example, for a two-dimensional grid of shape `(b, ny, nx)`, where `b`\n      is the batch size, `boundary_conditions[0][i]` with `i = 0, 1` should\n      return a tuple of either numbers, zero-rank tensors or tensors of shape\n      `(b, nx)`. Similarly for `boundary_conditions[1][i]`, except the tensor\n      shape should be `(b, ny)`. `alpha` and `beta` can also be `None` in case\n      of Neumann and Dirichlet conditions, respectively.\n      Default value: `None`. Unlike setting `None` to individual elements of\n      `boundary_conditions`, setting the entire `boundary_conditions` object to\n      `None` means Dirichlet conditions with zero value on all boundaries are\n      applied.\n    time_marching_scheme: A callable which represents the time marching scheme\n      for solving the PDE equation. If `u(t)` is space-discretized vector of the\n      solution of a PDE, a time marching scheme approximately solves the\n      equation `du_inner/dt = A(t) u_inner(t) + A_mixed(t) u(t) + b(t)` for\n      `u(t2)` given `u(t1)`, or vice versa if going backwards in time.\n      Here `A` is a banded matrix containing contributions from the current and\n      neighboring points in space, `A_mixed` are contributions of mixed terms,\n      `b` is an arbitrary vector (inhomogeneous term), and `u_inner` is `u` with\n      boundaries with Robin conditions trimmed.\n      Multidimensional time marching schemes are usually based on the idea of\n      ADI (alternating direction implicit) method: the time step is split into\n      substeps, and in each substep only one dimension is treated "implicitly",\n      while all the others are treated "explicitly". This way one has to solve\n      only tridiagonal systems of equations, but not more complicated banded\n      ones. A few examples of time marching schemes (Douglas, Craig-Sneyd, etc.)\n      can be found in [1].\n      The callable consumes the following arguments by keyword:\n        1. inner_value_grid: Grid of solution values at the current time of\n          the same `dtype` as `value_grid` and shape of `value_grid[..., 1:-1]`.\n        2. t1: Lesser of the two times defining the step.\n        3. t2: Greater of the two times defining the step.\n        4. equation_params_fn: A callable that takes a scalar `Tensor` argument\n          representing time and returns a tuple of two elements.\n          The first one represents `A`. The length must be the number of\n          dimensions (`n_dims`), and A[i] must have length `n_dims - i`.\n          `A[i][0]` is a tridiagonal matrix representing influence of the\n          neighboring points along the dimension `i`. It is a tuple of\n          superdiagonal, diagonal, and subdiagonal parts of the tridiagonal\n          matrix. The shape of these tensors must be same as of `value_grid`.\n          superdiagonal[..., -1] and subdiagonal[..., 0] are ignored.\n          `A[i][j]` with `i < j < n_dims` are tuples of four Tensors with same\n          shape as `value_grid` representing the influence of four points placed\n          diagonally from the given point in the plane of dimensions `i` and\n          `j`. Denoting `k`, `l` the indices of a given grid point in the plane,\n          the four Tensors represent contributions of points `(k+1, l+1)`,\n          `(k+1, l-1)`, `(k-1, l+1)`, and `(k-1, l-1)`, in this order.\n          The second element in the tuple is a list of contributions to `b(t)`\n          associated with each dimension. E.g. if `b(t)` comes from boundary\n          conditions, then it is split correspondingly. Each element in the list\n          is a Tensor with the shape of `value_grid`.\n          For example a 2D problem with `value_grid.shape = (b, ny, nx)`, where\n          `b` is the batch size. The elements `Aij` are non-zero if `i = j` or\n          `i` is a neighbor of `j` in the x-y plane. Depict these non-zero\n          elements on the grid as follows:\n          ```\n          a_mm    a_y-   a_mp\n          a_x-    a_0    a_x+\n          a_pm   a_y+   a_pp\n          ```\n          The callable should return\n          ```\n          ([[(a_y-, a_0y, a_y+), (a_pp, a_pm, a_mp, a_pp)],\n            [None, (a_x-, a_0x, a_x+)]],\n          [b_y, b_x])\n          ```\n          where `a_0x + a_0y = a_0` (the splitting is arbitrary). Note that\n          there is no need to repeat the non-diagonal term\n          `(a_pp, a_pm, a_mp, a_pp)` for the second time: it\'s replaced with\n          `None`.\n          All the elements `a_...` may be different for each point in the grid,\n          so they are `Tensors` of shape `(B, ny, nx)`. `b_y` and `b_x` are also\n          `Tensors` of that shape.\n        5. A callable that accepts a `Tensor` of shape `inner_value_grid` and\n          appends boundaries according to the boundary conditions, i.e.\n          transforms`u_inner` to `u`.\n        6. n_dims: A Python integer, the spatial dimension of the PDE.\n        7. has_default_lower_boundary: A Python list of booleans of length\n          `n_dims`. List indices enumerate the dimensions with `True` values\n          marking default lower boundary condition along corresponding\n          dimensions, and `False` values indicating Robin boundary conditions.\n        8. has_default_upper_boundary: Similar to has_default_lower_boundary,\n          but for upper boundaries.\n\n      The callable should return a `Tensor` of the same shape and `dtype` as\n      `values_grid` that represents an approximate solution of the\n      space-discretized PDE.\n    second_order_coeff_fn: Callable returning the second order coefficient\n      `a_{ij}(t, r)` evaluated at given time `t`.\n      The callable accepts the following arguments:\n        `t`: The time at which the coefficient should be evaluated.\n        `locations_grid`: a `Tensor` representing a grid of locations `r` at\n          which the coefficient should be evaluated.\n      Returns an object `A` such that `A[i][j]` is defined and\n      `A[i][j]=a_{ij}(r, t)`, where `0 <= i < n_dims` and `i <= j < n_dims`.\n      For example, the object may be a list of lists or a rank 2 Tensor.\n      Only the elements with `j >= i` will be used, and it is assumed that\n      `a_{ji} = a_{ij}`, so `A[i][j] with `j < i` may return `None`.\n      Each `A[i][j]` should be a Number, a `Tensor` broadcastable to the\n      shape of the grid represented by `locations_grid`, or `None` if\n      corresponding term is absent in the equation. Also, the callable itself\n      may be None, meaning there are no second-order derivatives in the\n      equation.\n      For example, for `n_dims=2`, the callable may return either\n      `[[a_yy, a_xy], [a_xy, a_xx]]` or `[[a_yy, a_xy], [None, a_xx]]`.\n    first_order_coeff_fn: Callable returning the first order coefficients\n      `b_{i}(t, r)` evaluated at given time `t`.\n      The callable accepts the following arguments:\n        `t`: The time at which the coefficient should be evaluated.\n        `locations_grid`: a `Tensor` representing a grid of locations `r` at\n          which the coefficient should be evaluated.\n      Returns a list or an 1D `Tensor`, `i`-th element of which represents\n      `b_{i}(t, r)`. Each element should be a Number, a `Tensor` broadcastable\n       to the shape of of the grid represented by `locations_grid`, or None if\n       corresponding term is absent in the equation. The callable itself may be\n       None, meaning there are no first-order derivatives in the equation.\n    zeroth_order_coeff_fn: Callable returning the zeroth order coefficient\n      `c(t, r)` evaluated at given time `t`.\n      The callable accepts the following arguments:\n        `t`: The time at which the coefficient should be evaluated.\n        `locations_grid`: a `Tensor` representing a grid of locations `r` at\n          which the coefficient should be evaluated.\n      Should return a Number or a `Tensor` broadcastable to the shape of\n      the grid represented by `locations_grid`. May also return None or be None\n      if the shift term is absent in the equation.\n    inner_second_order_coeff_fn: Callable returning the coefficients under the\n      second derivatives (i.e. `A_ij(t, x)` above) at given time `t`. The\n      requirements are the same as for `second_order_coeff_fn`.\n    inner_first_order_coeff_fn: Callable returning the coefficients under the\n      first derivatives (i.e. `B_i(t, x)` above) at given time `t`. The\n      requirements are the same as for `first_order_coeff_fn`.\n    dtype: The dtype to use.\n    name: The name to give to the ops.\n      Default value: None which means `parabolic_equation_step` is used.\n\n  Returns:\n    A sequence of two `Tensor`s. The first one is a `Tensor` of the same\n    `dtype` and `shape` as `coord_grid` and represents a new coordinate grid\n    after one iteration. The second `Tensor` is of the same shape and `dtype`\n    as`values_grid` and represents an approximate solution of the equation after\n    one iteration.\n\n  #### References:\n  [1] Tinne Haentjens, Karek J. in\'t Hout. ADI finite difference schemes\n  for the Heston-Hull-White PDE. https://arxiv.org/abs/1111.4087\n  '
    with tf.compat.v1.name_scope(name, 'multidim_parabolic_equation_step', values=[time, next_time, coord_grid, value_grid]):
        time = tf.convert_to_tensor(time, dtype=dtype, name='time')
        next_time = tf.convert_to_tensor(next_time, dtype=dtype, name='next_time')
        coord_grid = [tf.convert_to_tensor(x, dtype=dtype, name='coord_grid_axis_{}'.format(ind)) for (ind, x) in enumerate(coord_grid)]
        coord_grid = list(utils.broadcast_common_batch_shape(*coord_grid))
        value_grid = tf.convert_to_tensor(value_grid, dtype=dtype, name='value_grid')
        n_dims = len(coord_grid)
        second_order_coeff_fn = second_order_coeff_fn or (lambda *args: [[None] * n_dims] * n_dims)
        first_order_coeff_fn = first_order_coeff_fn or (lambda *args: [None] * n_dims)
        zeroth_order_coeff_fn = zeroth_order_coeff_fn or (lambda *args: None)
        inner_second_order_coeff_fn = inner_second_order_coeff_fn or (lambda *args: [[None] * n_dims] * n_dims)
        inner_first_order_coeff_fn = inner_first_order_coeff_fn or (lambda *args: [None] * n_dims)
        batch_rank = len(value_grid.shape.as_list()) - len(coord_grid)
        has_default_lower_boundary = []
        has_default_upper_boundary = []
        lower_trim_indices = []
        upper_trim_indices = []
        for d in range(n_dims):
            num_discretization_pts = utils.get_shape(value_grid)[batch_rank + d]
            if boundary_conditions[d][0] is None:
                has_default_lower_boundary.append(True)
                lower_trim_indices.append(0)
            else:
                has_default_lower_boundary.append(False)
                lower_trim_indices.append(1)
            if boundary_conditions[d][1] is None:
                upper_trim_indices.append(num_discretization_pts - 1)
                has_default_upper_boundary.append(True)
            else:
                upper_trim_indices.append(num_discretization_pts - 2)
                has_default_upper_boundary.append(False)

        def equation_params_fn(t):
            if False:
                i = 10
                return i + 15
            return _construct_discretized_equation_params(coord_grid, value_grid, boundary_conditions, has_default_lower_boundary, has_default_upper_boundary, lower_trim_indices, upper_trim_indices, second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn, inner_second_order_coeff_fn, inner_first_order_coeff_fn, batch_rank, t)
        inner_grid_in = _trim_boundaries(value_grid, batch_rank, lower_trim_indices=lower_trim_indices, upper_trim_indices=upper_trim_indices)

        def _append_boundaries_fn(inner_value_grid):
            if False:
                i = 10
                return i + 15
            value_grid_with_boundaries = _append_boundaries(value_grid, inner_value_grid, coord_grid, boundary_conditions, has_default_lower_boundary, has_default_upper_boundary, lower_trim_indices, upper_trim_indices, batch_rank, time)
            return value_grid_with_boundaries
        inner_grid_out = time_marching_scheme(value_grid=inner_grid_in, t1=time, t2=next_time, equation_params_fn=equation_params_fn, append_boundaries_fn=_append_boundaries_fn, has_default_lower_boundary=has_default_lower_boundary, has_default_upper_boundary=has_default_upper_boundary, n_dims=n_dims)
        updated_value_grid = _append_boundaries(value_grid, inner_grid_out, coord_grid, boundary_conditions, has_default_lower_boundary, has_default_upper_boundary, lower_trim_indices, upper_trim_indices, batch_rank, next_time)
        return (coord_grid, updated_value_grid)

def _construct_discretized_equation_params(coord_grid, value_grid, boundary_conditions, has_default_lower_boundary, has_default_upper_boundary, lower_trim_indices, upper_trim_indices, second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn, inner_second_order_coeff_fn, inner_first_order_coeff_fn, batch_rank, t):
    if False:
        return 10
    'Constructs parameters of discretized equation.'
    second_order_coeffs = second_order_coeff_fn(t, coord_grid)
    first_order_coeffs = first_order_coeff_fn(t, coord_grid)
    zeroth_order_coeffs = zeroth_order_coeff_fn(t, coord_grid)
    inner_second_order_coeffs = inner_second_order_coeff_fn(t, coord_grid)
    inner_first_order_coeffs = inner_first_order_coeff_fn(t, coord_grid)
    matrix_params = []
    inhomog_terms = []
    zeroth_order_coeffs = _prepare_pde_coeff(zeroth_order_coeffs, value_grid)
    if zeroth_order_coeffs is not None:
        zeroth_order_coeffs = _trim_boundaries(zeroth_order_coeffs, from_dim=batch_rank, lower_trim_indices=lower_trim_indices, upper_trim_indices=upper_trim_indices)
    n_dims = len(coord_grid)
    for dim in range(n_dims):
        delta = _get_grid_delta(coord_grid, dim)
        second_order_coeff = second_order_coeffs[dim][dim]
        first_order_coeff = first_order_coeffs[dim]
        inner_second_order_coeff = inner_second_order_coeffs[dim][dim]
        inner_first_order_coeff = inner_first_order_coeffs[dim]
        second_order_coeff = _prepare_pde_coeff(second_order_coeff, value_grid)
        first_order_coeff = _prepare_pde_coeff(first_order_coeff, value_grid)
        inner_second_order_coeff = _prepare_pde_coeff(inner_second_order_coeff, value_grid)
        inner_first_order_coeff = _prepare_pde_coeff(inner_first_order_coeff, value_grid)
        (superdiag, diag, subdiag) = _construct_tridiagonal_matrix(value_grid, second_order_coeff, first_order_coeff, inner_second_order_coeff, inner_first_order_coeff, delta, dim, batch_rank, lower_trim_indices, upper_trim_indices, has_default_lower_boundary, has_default_upper_boundary, n_dims)
        [subdiag, diag, superdiag] = _apply_default_boundary(subdiag, diag, superdiag, inner_first_order_coeff, first_order_coeff, delta, has_default_lower_boundary, has_default_upper_boundary, lower_trim_indices, upper_trim_indices, batch_rank, dim)
        ((superdiag, diag, subdiag), inhomog_term_contribution) = _apply_robin_boundary_conditions(value_grid, dim, batch_rank, boundary_conditions, has_default_lower_boundary, has_default_upper_boundary, lower_trim_indices, upper_trim_indices, coord_grid, superdiag, diag, subdiag, delta, t)
        if zeroth_order_coeffs is not None:
            diag += -zeroth_order_coeffs / n_dims
        matrix_params_row = [None] * dim + [(superdiag, diag, subdiag)]
        for dim2 in range(dim + 1, n_dims):
            mixed_coeff = second_order_coeffs[dim][dim2]
            inner_mixed_coeff = inner_second_order_coeffs[dim][dim2]
            mixed_term_contrib = _construct_contribution_of_mixed_term(mixed_coeff, inner_mixed_coeff, coord_grid, value_grid, dim, dim2, batch_rank, lower_trim_indices, upper_trim_indices, has_default_lower_boundary, has_default_upper_boundary, n_dims)
            matrix_params_row.append(mixed_term_contrib)
        matrix_params.append(matrix_params_row)
        inhomog_terms.append(inhomog_term_contribution)
    return (matrix_params, inhomog_terms)

def _construct_tridiagonal_matrix(value_grid, second_order_coeff, first_order_coeff, inner_second_order_coeff, inner_first_order_coeff, delta, dim, batch_rank, lower_trim_indices, upper_trim_indices, has_default_lower_boundary, has_default_upper_boundary, n_dims):
    if False:
        print('Hello World!')
    'Constructs contributions of first and non-mixed second order terms.'
    (trimmed_lower_indices, trimmed_upper_indices, _) = _remove_default_boundary(lower_trim_indices, upper_trim_indices, has_default_lower_boundary, has_default_upper_boundary, batch_rank, (dim,), n_dims)
    zeros = tf.zeros_like(value_grid)
    zeros = _trim_boundaries(zeros, from_dim=batch_rank, lower_trim_indices=trimmed_lower_indices, upper_trim_indices=trimmed_upper_indices)

    def create_trimming_shifts(dim_shift):
        if False:
            for i in range(10):
                print('nop')
        shifts = [0] * n_dims
        shifts[dim] = dim_shift
        return shifts
    if first_order_coeff is None and inner_first_order_coeff is None:
        superdiag_first_order = zeros
        diag_first_order = zeros
        subdiag_first_order = zeros
    else:
        superdiag_first_order = -1 / (2 * delta)
        subdiag_first_order = 1 / (2 * delta)
        diag_first_order = -superdiag_first_order - subdiag_first_order
        if first_order_coeff is not None:
            first_order_coeff = _trim_boundaries(first_order_coeff, from_dim=batch_rank, lower_trim_indices=trimmed_lower_indices, upper_trim_indices=trimmed_upper_indices)
            superdiag_first_order *= first_order_coeff
            subdiag_first_order *= first_order_coeff
            diag_first_order *= first_order_coeff
        if inner_first_order_coeff is not None:
            superdiag_first_order *= _trim_boundaries(inner_first_order_coeff, from_dim=batch_rank, lower_trim_indices=trimmed_lower_indices, upper_trim_indices=trimmed_upper_indices, shifts=create_trimming_shifts(1))
            subdiag_first_order *= _trim_boundaries(inner_first_order_coeff, from_dim=batch_rank, lower_trim_indices=trimmed_lower_indices, upper_trim_indices=trimmed_upper_indices, shifts=create_trimming_shifts(-1))
            diag_first_order *= _trim_boundaries(inner_first_order_coeff, from_dim=batch_rank, lower_trim_indices=trimmed_lower_indices, upper_trim_indices=trimmed_upper_indices)
    if second_order_coeff is None and inner_second_order_coeff is None:
        superdiag_second_order = zeros
        diag_second_order = zeros
        subdiag_second_order = zeros
    else:
        superdiag_second_order = -1 / (delta * delta)
        subdiag_second_order = -1 / (delta * delta)
        diag_second_order = -superdiag_second_order - subdiag_second_order
        if second_order_coeff is not None:
            second_order_coeff = _trim_boundaries(second_order_coeff, from_dim=batch_rank, lower_trim_indices=trimmed_lower_indices, upper_trim_indices=trimmed_upper_indices)
            superdiag_second_order *= second_order_coeff
            subdiag_second_order *= second_order_coeff
            diag_second_order *= second_order_coeff
        if inner_second_order_coeff is not None:
            superdiag_second_order *= _trim_boundaries(inner_second_order_coeff, from_dim=batch_rank, lower_trim_indices=trimmed_lower_indices, upper_trim_indices=trimmed_upper_indices, shifts=create_trimming_shifts(1))
            subdiag_second_order *= _trim_boundaries(inner_second_order_coeff, from_dim=batch_rank, lower_trim_indices=trimmed_lower_indices, upper_trim_indices=trimmed_upper_indices, shifts=create_trimming_shifts(-1))
            diag_second_order *= _trim_boundaries(inner_second_order_coeff, from_dim=batch_rank, lower_trim_indices=trimmed_lower_indices, upper_trim_indices=trimmed_upper_indices)
    superdiag = superdiag_first_order + superdiag_second_order
    subdiag = subdiag_first_order + subdiag_second_order
    diag = diag_first_order + diag_second_order
    return (superdiag, diag, subdiag)

def _construct_contribution_of_mixed_term(outer_coeff, inner_coeff, coord_grid, value_grid, dim1, dim2, batch_rank, lower_trim_indices, upper_trim_indices, has_default_lower_boundary, has_default_upper_boundary, n_dims):
    if False:
        i = 10
        return i + 15
    'Constructs contribution of a mixed derivative term.'
    if outer_coeff is None and inner_coeff is None:
        return None
    delta_dim1 = _get_grid_delta(coord_grid, dim1)
    delta_dim2 = _get_grid_delta(coord_grid, dim2)
    outer_coeff = _prepare_pde_coeff(outer_coeff, value_grid)
    inner_coeff = _prepare_pde_coeff(inner_coeff, value_grid)
    contrib = -1 / (2 * delta_dim1 * delta_dim2)
    (trimmed_lower_indices, trimmed_upper_indices, paddings) = _remove_default_boundary(lower_trim_indices, upper_trim_indices, has_default_lower_boundary, has_default_upper_boundary, batch_rank, (dim1, dim2), n_dims)
    append_zeros_fn = lambda x: tf.pad(x, paddings)
    if outer_coeff is not None:
        outer_coeff = _trim_boundaries(outer_coeff, batch_rank, lower_trim_indices=trimmed_lower_indices, upper_trim_indices=trimmed_upper_indices)
        contrib *= outer_coeff
    if inner_coeff is None:
        contrib = append_zeros_fn(contrib)
        return (contrib, -contrib, -contrib, contrib)

    def create_trimming_shifts(dim1_shift, dim2_shift):
        if False:
            i = 10
            return i + 15
        shifts = [0] * n_dims
        shifts[dim1] = dim1_shift
        shifts[dim2] = dim2_shift
        return shifts
    contrib_pp = contrib * _trim_boundaries(inner_coeff, from_dim=batch_rank, shifts=create_trimming_shifts(1, 1), lower_trim_indices=trimmed_lower_indices, upper_trim_indices=trimmed_upper_indices)
    contrib_pm = -contrib * _trim_boundaries(inner_coeff, from_dim=batch_rank, shifts=create_trimming_shifts(1, -1), lower_trim_indices=trimmed_lower_indices, upper_trim_indices=trimmed_upper_indices)
    contrib_mp = -contrib * _trim_boundaries(inner_coeff, from_dim=batch_rank, shifts=create_trimming_shifts(-1, 1), lower_trim_indices=trimmed_lower_indices, upper_trim_indices=trimmed_upper_indices)
    contrib_mm = contrib * _trim_boundaries(inner_coeff, from_dim=batch_rank, shifts=create_trimming_shifts(-1, -1), lower_trim_indices=trimmed_lower_indices, upper_trim_indices=trimmed_upper_indices)
    (contrib_pp, contrib_pm, contrib_mp, contrib_mm) = map(append_zeros_fn, (contrib_pp, contrib_pm, contrib_mp, contrib_mm))
    return (contrib_pp, contrib_pm, contrib_mp, contrib_mm)

def _apply_default_boundary(subdiag, diag, superdiag, inner_first_order_coeff, first_order_coeff, delta, has_default_lower_boundary, has_default_upper_boundary, lower_trim_indices, upper_trim_indices, batch_rank, dim):
    if False:
        i = 10
        return i + 15
    'Update discretization matrix for default boundary conditions.'
    if has_default_lower_boundary[dim]:
        (subdiag, diag, superdiag) = _apply_default_lower_boundary(subdiag, diag, superdiag, inner_first_order_coeff, first_order_coeff, delta, lower_trim_indices, upper_trim_indices, batch_rank, dim)
    if has_default_upper_boundary[dim]:
        (subdiag, diag, superdiag) = _apply_default_upper_boundary(subdiag, diag, superdiag, inner_first_order_coeff, first_order_coeff, delta, lower_trim_indices, upper_trim_indices, batch_rank, dim)
    return (subdiag, diag, superdiag)

def _apply_default_lower_boundary(subdiag, diag, superdiag, inner_first_order_coeff, first_order_coeff, delta, lower_trim_indices, upper_trim_indices, batch_rank, dim):
    if False:
        print('Hello World!')
    'Update discretization matrix for default lower boundary conditions.'
    zeros = tf.zeros_like(diag)
    ones = tf.ones_like(diag)
    if inner_first_order_coeff is None:
        inner_coeff = ones
    else:
        inner_coeff = _trim_boundaries(inner_first_order_coeff, from_dim=batch_rank, lower_trim_indices=lower_trim_indices, upper_trim_indices=upper_trim_indices)
    if first_order_coeff is None:
        if inner_first_order_coeff is None:
            extra_first_order_coeff = _slice(zeros, batch_rank + dim, 0, 1)
        else:
            extra_first_order_coeff = _slice(ones, batch_rank + dim, 0, 1)
    else:
        first_order_coeff = _trim_boundaries(first_order_coeff, from_dim=batch_rank, lower_trim_indices=lower_trim_indices, upper_trim_indices=upper_trim_indices)
        extra_first_order_coeff = _slice(first_order_coeff, batch_rank + dim, 0, 1)
    inner_coeff_next = _slice(inner_coeff, batch_rank + dim, 1, 2)
    extra_superdiag_coeff = inner_coeff_next * extra_first_order_coeff / delta
    superdiag = _append_first(-extra_superdiag_coeff, superdiag, axis=batch_rank + dim)
    inner_coeff_boundary = _slice(inner_coeff, batch_rank + dim, 0, 1)
    extra_diag_coeff = -inner_coeff_boundary * extra_first_order_coeff / delta
    diag = _append_first(-extra_diag_coeff, diag, axis=batch_rank + dim)
    subdiag = _append_first(tf.zeros_like(extra_diag_coeff), subdiag, axis=batch_rank + dim)
    return (subdiag, diag, superdiag)

def _apply_default_upper_boundary(subdiag, diag, superdiag, inner_first_order_coeff, first_order_coeff, delta, lower_trim_indices, upper_trim_indices, batch_rank, dim):
    if False:
        return 10
    'Update discretization matrix for default upper boundary conditions.'
    zeros = tf.zeros_like(diag)
    ones = tf.ones_like(diag)
    if inner_first_order_coeff is None:
        inner_coeff = ones
    else:
        inner_coeff = _trim_boundaries(inner_first_order_coeff, from_dim=batch_rank, lower_trim_indices=lower_trim_indices, upper_trim_indices=upper_trim_indices)
    if first_order_coeff is None:
        if inner_first_order_coeff is None:
            extra_first_order_coeff = _slice(zeros, batch_rank + dim, 0, 1)
        else:
            extra_first_order_coeff = _slice(ones, batch_rank + dim, 0, 1)
    else:
        first_order_coeff = _trim_boundaries(first_order_coeff, from_dim=batch_rank, lower_trim_indices=lower_trim_indices, upper_trim_indices=upper_trim_indices)
        extra_first_order_coeff = _slice(first_order_coeff, batch_rank + dim, -1, 0)
    inner_coeff_next = _slice(inner_coeff, batch_rank + dim, -1, 0)
    extra_diag_coeff = inner_coeff_next * extra_first_order_coeff / delta
    diag = _append_last(diag, -extra_diag_coeff, axis=batch_rank + dim)
    inner_coeff_boundary = _slice(inner_coeff, batch_rank + dim, -2, -1)
    extra_sub_coeff = -inner_coeff_boundary * extra_first_order_coeff / delta
    subdiag = _append_last(subdiag, -extra_sub_coeff, axis=batch_rank + dim)
    superdiag = _append_last(superdiag, tf.zeros_like(extra_diag_coeff), axis=batch_rank + dim)
    return (subdiag, diag, superdiag)

def _apply_robin_boundary_conditions(value_grid, dim, batch_rank, boundary_conditions, has_default_lower_boundary, has_default_upper_boundary, lower_trim_indices, upper_trim_indices, coord_grid, superdiag, diag, subdiag, delta, t):
    if False:
        i = 10
        return i + 15
    'Updates contributions according to boundary conditions.'
    if has_default_lower_boundary[dim] and has_default_upper_boundary[dim]:
        return ((superdiag, diag, subdiag), tf.zeros_like(diag))
    lower_trim_indices_bc = lower_trim_indices[:dim] + lower_trim_indices[dim + 1:]
    upper_trim_indices_bc = upper_trim_indices[:dim] + upper_trim_indices[dim + 1:]

    def reshape_fn(bound_coeff):
        if False:
            i = 10
            return i + 15
        'Reshapes boundary coefficient.'
        return _reshape_boundary_conds(bound_coeff, trim_from=batch_rank, expand_dim_at=batch_rank + dim, lower_trim_indices=lower_trim_indices_bc, upper_trim_indices=upper_trim_indices_bc)
    if has_default_lower_boundary[dim]:
        (alpha_l, beta_l, gamma_l) = (None, None, None)
    else:
        (alpha_l, beta_l, gamma_l) = boundary_conditions[dim][0](t, coord_grid)
    if has_default_upper_boundary[dim]:
        (alpha_u, beta_u, gamma_u) = (None, None, None)
    else:
        (alpha_u, beta_u, gamma_u) = boundary_conditions[dim][1](t, coord_grid)
    (alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u) = (_prepare_boundary_conditions(b, value_grid, batch_rank, dim) for b in (alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u))
    (alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u) = map(reshape_fn, (alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u))
    slice_dim = dim + batch_rank
    subdiag_first = _slice(subdiag, slice_dim, 0, 1)
    superdiag_last = _slice(superdiag, slice_dim, -1, 0)
    diag_inner = _slice(diag, slice_dim, 1, -1)
    if beta_l is None and beta_u is None:
        if has_default_lower_boundary[dim]:
            first_inhomog_element = tf.zeros_like(subdiag_first)
        else:
            first_inhomog_element = subdiag_first * gamma_l / alpha_l
        if has_default_upper_boundary[dim]:
            last_inhomog_element = tf.zeros_like(superdiag_last)
        else:
            last_inhomog_element = superdiag_last * gamma_u / alpha_u
        inhomog_term = _append_first_and_last(first_inhomog_element, tf.zeros_like(diag_inner), last_inhomog_element, axis=slice_dim)
        return ((superdiag, diag, subdiag), inhomog_term)
    subdiag_last = _slice(subdiag, slice_dim, -1, 0)
    subdiag_except_last = _slice(subdiag, slice_dim, 0, -1)
    superdiag_first = _slice(superdiag, slice_dim, 0, 1)
    superdiag_except_first = _slice(superdiag, slice_dim, 1, 0)
    diag_first = _slice(diag, slice_dim, 0, 1)
    diag_last = _slice(diag, slice_dim, -1, 0)
    if has_default_lower_boundary[dim]:
        diag_first_correction = 0
        superdiag_correction = 0
        first_inhomog_element = tf.zeros_like(subdiag_first)
    else:
        (xi1, xi2, eta) = _discretize_boundary_conditions(delta, delta, alpha_l, beta_l, gamma_l)
        diag_first_correction = subdiag_first * xi1
        superdiag_correction = subdiag_first * xi2
        first_inhomog_element = subdiag_first * eta
    if has_default_upper_boundary[dim]:
        diag_last_correction = 0
        subdiag_correction = 0
        last_inhomog_element = tf.zeros_like(superdiag_last)
    else:
        (xi1, xi2, eta) = _discretize_boundary_conditions(delta, delta, alpha_u, beta_u, gamma_u)
        diag_last_correction = superdiag_last * xi1
        subdiag_correction = superdiag_last * xi2
        last_inhomog_element = superdiag_last * eta
    diag = _append_first_and_last(diag_first + diag_first_correction, diag_inner, diag_last + diag_last_correction, axis=slice_dim)
    superdiag = _append_first(superdiag_first + superdiag_correction, superdiag_except_first, axis=slice_dim)
    subdiag = _append_last(subdiag_except_last, subdiag_last + subdiag_correction, axis=slice_dim)
    inhomog_term = _append_first_and_last(first_inhomog_element, tf.zeros_like(diag_inner), last_inhomog_element, axis=slice_dim)
    return ((superdiag, diag, subdiag), inhomog_term)

def _append_boundaries(value_grid_in, inner_grid_out, coord_grid, boundary_conditions, has_default_lower_boundary, has_default_upper_boundary, lower_trim_indices, upper_trim_indices, batch_rank, t):
    if False:
        return 10
    'Calculates and appends boundary values after making a step.'
    grid = inner_grid_out
    for dim in range(len(coord_grid)):
        grid = _append_boundary(dim, batch_rank, boundary_conditions, has_default_lower_boundary, has_default_upper_boundary, lower_trim_indices, upper_trim_indices, coord_grid, value_grid_in, grid, t)
    return grid

def _append_boundary(dim, batch_rank, boundary_conditions, has_default_lower_boundary, has_default_upper_boundary, lower_trim_indices, upper_trim_indices, coord_grid, value_grid_in, current_value_grid_out, t):
    if False:
        return 10
    'Calculates and appends boundaries orthogonal to `dim`.'

    def _reshape_fn(bound_coeff):
        if False:
            print('Hello World!')
        return _reshape_boundary_conds(bound_coeff, trim_from=batch_rank + dim, expand_dim_at=batch_rank + dim, lower_trim_indices=lower_trim_indices[dim + 1:], upper_trim_indices=upper_trim_indices[dim + 1:])
    delta = _get_grid_delta(coord_grid, dim)
    if has_default_lower_boundary[dim]:
        first_value = None
    else:
        lower_value_first = _slice(current_value_grid_out, batch_rank + dim, 0, 1)
        lower_value_second = _slice(current_value_grid_out, batch_rank + dim, 1, 2)
        (alpha_l, beta_l, gamma_l) = boundary_conditions[dim][0](t, coord_grid)
        (alpha_l, beta_l, gamma_l) = (_prepare_boundary_conditions(b, value_grid_in, batch_rank, dim) for b in (alpha_l, beta_l, gamma_l))
        (alpha_l, beta_l, gamma_l) = map(_reshape_fn, (alpha_l, beta_l, gamma_l))
        (xi1, xi2, eta) = _discretize_boundary_conditions(delta, delta, alpha_l, beta_l, gamma_l)
        first_value = xi1 * lower_value_first + xi2 * lower_value_second + eta
    if has_default_upper_boundary[dim]:
        last_value = None
    else:
        upper_value_first = _slice(current_value_grid_out, batch_rank + dim, -1, 0)
        upper_value_second = _slice(current_value_grid_out, batch_rank + dim, -2, -1)
        (alpha_u, beta_u, gamma_u) = boundary_conditions[dim][1](t, coord_grid)
        (alpha_u, beta_u, gamma_u) = (_prepare_boundary_conditions(b, value_grid_in, batch_rank, dim) for b in (alpha_u, beta_u, gamma_u))
        (alpha_u, beta_u, gamma_u) = map(_reshape_fn, (alpha_u, beta_u, gamma_u))
        (xi1, xi2, eta) = _discretize_boundary_conditions(delta, delta, alpha_u, beta_u, gamma_u)
        last_value = xi1 * upper_value_first + xi2 * upper_value_second + eta
    return _append_first_and_last(first_value, current_value_grid_out, last_value, axis=batch_rank + dim)

def _append_first_and_last(first, inner, last, axis):
    if False:
        for i in range(10):
            print('nop')
    if first is None:
        return _append_last(inner, last, axis=axis)
    if last is None:
        return _append_first(first, inner, axis=axis)
    return tf.concat((first, inner, last), axis=axis)

def _append_first(first, rest, axis):
    if False:
        print('Hello World!')
    if first is None:
        return rest
    return tf.concat((first, rest), axis=axis)

def _append_last(rest, last, axis):
    if False:
        for i in range(10):
            print('nop')
    if last is None:
        return rest
    return tf.concat((rest, last), axis=axis)

def _get_grid_delta(coord_grid, dim):
    if False:
        print('Hello World!')
    delta = coord_grid[dim][..., 1] - coord_grid[dim][..., 0]
    n = len(coord_grid)
    if delta.shape.rank == 0:
        return delta
    else:
        return delta[[...] + n * [tf.newaxis]]

def _prepare_pde_coeff(raw_coeff, value_grid):
    if False:
        i = 10
        return i + 15
    if raw_coeff is None:
        return None
    dtype = value_grid.dtype
    coeff = tf.convert_to_tensor(raw_coeff, dtype=dtype)
    coeff = tf.broadcast_to(coeff, utils.get_shape(value_grid))
    return coeff

def _prepare_boundary_conditions(boundary_tensor, value_grid, batch_rank, dim):
    if False:
        print('Hello World!')
    'Prepares values received from boundary_condition callables.'
    if boundary_tensor is None:
        return None
    boundary_tensor = tf.convert_to_tensor(boundary_tensor, value_grid.dtype)
    dim_to_remove = batch_rank + dim
    broadcast_shape = []
    for (i, size) in enumerate(value_grid.shape):
        if i != dim_to_remove:
            broadcast_shape.append(size)
    return tf.broadcast_to(boundary_tensor, broadcast_shape)

def _discretize_boundary_conditions(dx0, dx1, alpha, beta, gamma):
    if False:
        while True:
            i = 10
    'Discretizes boundary conditions.'
    if beta is None:
        if alpha is None:
            raise ValueError("Invalid boundary conditions: alpha and beta can't both be None.")
        zeros = tf.zeros_like(gamma)
        return (zeros, zeros, gamma / alpha)
    denom = beta * dx1 * (2 * dx0 + dx1)
    if alpha is not None:
        denom += alpha * dx0 * dx1 * (dx0 + dx1)
    xi1 = beta * (dx0 + dx1) * (dx0 + dx1) / denom
    xi2 = -beta * dx0 * dx0 / denom
    eta = gamma * dx0 * dx1 * (dx0 + dx1) / denom
    return (xi1, xi2, eta)

def _reshape_boundary_conds(raw_coeff, trim_from, expand_dim_at, lower_trim_indices=None, upper_trim_indices=None):
    if False:
        while True:
            i = 10
    'Reshapes boundary condition coefficients.'
    if not tf.is_tensor(raw_coeff) or len(raw_coeff.shape.as_list()) == 0:
        return raw_coeff
    coeff = _trim_boundaries(raw_coeff, trim_from, lower_trim_indices=lower_trim_indices, upper_trim_indices=upper_trim_indices)
    coeff = tf.expand_dims(coeff, expand_dim_at)
    return coeff

def _slice(tensor, dim, start, end):
    if False:
        print('Hello World!')
    'Slices the tensor along given dimension.'
    rank = tensor.shape.rank
    slices = rank * [slice(None)]
    if end == 0:
        end = None
    slices[dim] = slice(start, end)
    return tensor[slices]

def _trim_boundaries(tensor, from_dim, shifts=None, lower_trim_indices=None, upper_trim_indices=None):
    if False:
        for i in range(10):
            print('nop')
    'Trims tensor boundaries starting from given dimension.'
    rank = tensor.shape.rank
    slices = rank * [slice(None)]
    for i in range(from_dim, rank):
        if lower_trim_indices is None:
            slice_begin = 1
        else:
            slice_begin = lower_trim_indices[i - from_dim]
        if upper_trim_indices is None:
            slice_end = -1
        else:
            slice_end = upper_trim_indices[i - from_dim] + 1
        if shifts is not None:
            shift = shifts[i - from_dim]
            slice_begin += shift
            slice_end += shift
        if isinstance(slice_end, int) and slice_end == 0:
            slice_end = None
        slices[i] = slice(slice_begin, slice_end)
    res = tensor[slices]
    return res

def _remove_default_boundary(lower_trim_indices, upper_trim_indices, has_default_lower_boundary, has_default_upper_boundary, batch_rank, dims, n_dims):
    if False:
        while True:
            i = 10
    'Creates trim indices that correspond to an inner grid with Robin BC.'
    trimmed_lower_indices = []
    trimmed_upper_indices = []
    paddings = batch_rank * [[0, 0]]
    for dim in range(n_dims):
        update_lower = has_default_lower_boundary[dim] and dim in dims
        update_upper = has_default_upper_boundary[dim] and dim in dims
        trim_lower = 1 if update_lower else 0
        trimmed_lower_indices.append(lower_trim_indices[dim] + trim_lower)
        trim_upper = 1 if update_upper else 0
        trimmed_upper_indices.append(upper_trim_indices[dim] - trim_upper)
        paddings.append([trim_lower, trim_upper])
    return (trimmed_lower_indices, trimmed_upper_indices, paddings)
__all__ = ['multidim_parabolic_equation_step']