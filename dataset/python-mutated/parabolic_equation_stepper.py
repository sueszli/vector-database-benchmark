"""Stepper for parabolic PDEs solving."""
import tensorflow.compat.v2 as tf
from tf_quant_finance import utils

def parabolic_equation_step(time, next_time, coord_grid, value_grid, boundary_conditions, second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn, inner_second_order_coeff_fn, inner_first_order_coeff_fn, time_marching_scheme, dtype=None, name=None):
    if False:
        while True:
            i = 10
    "Performs one step of the one dimensional parabolic PDE solver.\n\n  Typically one doesn't need to call this function directly, unless they have\n  a custom time marching scheme. A simple stepper function for one-dimensional\n  PDEs can be found in `crank_nicolson.py`.\n\n  For a given solution (given by the `value_grid`) of a parabolic PDE at a given\n  `time` on a given `coord_grid` computes an approximate solution at the\n  `next_time` on the same coordinate grid. The parabolic differential equation\n  is of the form:\n\n  ```none\n   dV/dt + a * d2(A * V)/dx2 + b * d(B * V)/dx + c * V = 0\n  ```\n  Here `a`, `A`, `b`, `B`, and `c` are known coefficients which may depend on\n  `x` and `t`; `V = V(t, x)` is the solution to be found.\n\n  Args:\n    time: Real scalar `Tensor`. The time before the step.\n    next_time: Real scalar `Tensor`. The time after the step.\n    coord_grid: List of size 1 that contains a rank 1 real `Tensor` of\n      has shape `[d]` or `B + [d]` where `d` is the size of the grid and `B` is\n      a batch shape. Represents the coordinates of the grid points.\n    value_grid: Real `Tensor` containing the function values at time\n      `time` which have to be evolved to time `next_time`. The shape of the\n      `Tensor` must broadcast with `B + [d]`. `B` is the batch\n      dimensions (one or more), which allow multiple functions (with potentially\n      different boundary/final conditions and PDE coefficients) to be evolved\n      simultaneously.\n    boundary_conditions: The boundary conditions. Only rectangular\n      boundary conditions are supported. A list of tuples of size 1. The list\n      element is a tuple that consists of two callables or `None`s representing\n      the boundary conditions at the minimum and maximum values of the spatial\n      variable indexed by the position in the list. `boundary_conditions[0][0]`\n      describes the boundary at `x_min`, and `boundary_conditions[0][1]` the\n      boundary at `x_max`. `None` values mean that the second order term on the\n      boundary is assumed to be zero, i.e.,\n      'dV/dt + b * d(B * V)/dx + c * V = 0'. This condition is appropriate for\n      PDEs where the second order term disappears on the boundary. For not\n      `None` values, the boundary conditions are accepted in the form\n      `alpha(t) V + beta(t) V_n = gamma(t)`,\n      where `V_n` is the derivative with respect to the exterior normal to the\n      boundary. Each callable receives the current time `t` and the `coord_grid`\n      at the current time, and should return a tuple of `alpha`, `beta`, and\n      `gamma`. Each can be a number, a zero-rank `Tensor` or a `Tensor` of the\n      batch shape.\n      For example, for a grid of shape `(b, n)`, where `b` is the batch size,\n      `boundary_conditions[0][0]` should return a tuple of either numbers,\n      zero-rank tensors or tensors of shape `(b, n)`.\n      `alpha` and `beta` can also be `None` in case of Neumann and\n      Dirichlet conditions, respectively.\n    second_order_coeff_fn: Callable returning the second order coefficient\n      `a(t, r)` evaluated at given time `t`.\n      The callable accepts the following arguments:\n        `t`: The time at which the coefficient should be evaluated.\n        `locations_grid`: a `Tensor` representing a grid of locations `r` at\n          which the coefficient should be evaluated.\n      Returns an object `A` such that `A[0][0]` is defined and equals\n      `a(r, t)`. `A[0][0]` should be a Number, a `Tensor` broadcastable to the\n      shape of the grid represented by `locations_grid`, or `None` if\n      corresponding term is absent in the equation. Also, the callable itself\n      may be None, meaning there are no second-order derivatives in the\n      equation.\n    first_order_coeff_fn: Callable returning the first order coefficient\n      `b(t, r)` evaluated at given time `t`.\n      The callable accepts the following arguments:\n        `t`: The time at which the coefficient should be evaluated.\n        `locations_grid`: a `Tensor` representing a grid of locations `r` at\n          which the coefficient should be evaluated.\n      Returns a list or an 1D `Tensor`, `0`-th element of which represents\n      `b(t, r)`. This element should be a Number, a `Tensor` broadcastable\n       to the shape of the grid represented by `locations_grid`, or None if\n       corresponding term is absent in the equation. The callable itself may be\n       None, meaning there are no first-order derivatives in the equation.\n    zeroth_order_coeff_fn: Callable returning the zeroth order coefficient\n      `c(t, r)` evaluated at given time `t`.\n      The callable accepts the following arguments:\n        `t`: The time at which the coefficient should be evaluated.\n        `locations_grid`: a `Tensor` representing a grid of locations `r` at\n          which the coefficient should be evaluated.\n      Should return a Number or a `Tensor` broadcastable to the shape of\n      the grid represented by `locations_grid`. May also return None or be None\n      if the shift term is absent in the equation.\n    inner_second_order_coeff_fn: Callable returning the coefficients under the\n      second derivatives (i.e. `A(t, x)` above) at given time `t`. The\n      requirements are the same as for `second_order_coeff_fn`.\n    inner_first_order_coeff_fn: Callable returning the coefficients under the\n      first derivatives (i.e. `B(t, x)` above) at given time `t`. The\n      requirements are the same as for `first_order_coeff_fn`.\n    time_marching_scheme: A callable which represents the time marching scheme\n      for solving the PDE equation. If `u(t)` is space-discretized vector of the\n      solution of the PDE, this callable approximately solves the equation\n      `du/dt = A(t) u(t)` for `u(t1)` given `u(t2)`. Here `A` is a tridiagonal\n      matrix. The callable consumes the following arguments by keyword:\n        1. inner_value_grid: Grid of solution values at the current time of\n          the same `dtype` as `value_grid` and shape of `value_grid[..., 1:-1]`.\n        2. t1: Time before the step.\n        3. t2: Time after the step.\n        4. equation_params_fn: A callable that takes a scalar `Tensor` argument\n          representing time, and constructs the tridiagonal matrix `A`\n          (a tuple of three `Tensor`s, main, upper, and lower diagonals)\n          and the inhomogeneous term `b`. All of the `Tensor`s are of the same\n          `dtype` as `values_inner_value_grid` and of the shape\n          broadcastable with the shape of `inner_value_grid`.\n      The callable should return a `Tensor` of the same shape and `dtype` a\n      `value_grid` and represents an approximate solution of the PDE after one\n      iteraton.\n    dtype: The dtype to use.\n    name: The name to give to the ops.\n      Default value: None which means `parabolic_equation_step` is used.\n\n  Returns:\n    A sequence of two `Tensor`s. The first one is a `Tensor` of the same\n    `dtype` and `shape` as `coord_grid` and represents a new coordinate grid\n    after one iteration. The second `Tensor` is of the same shape and `dtype`\n    as`value_grid` and represents an approximate solution of the equation after\n    one iteration.\n  "
    with tf.compat.v1.name_scope(name, 'parabolic_equation_step', [time, next_time, coord_grid, value_grid]):
        time = tf.convert_to_tensor(time, dtype=dtype, name='time')
        next_time = tf.convert_to_tensor(next_time, dtype=dtype, name='next_time')
        coord_grid = [tf.convert_to_tensor(x, dtype=dtype, name='coord_grid_axis_{}'.format(ind)) for (ind, x) in enumerate(coord_grid)]
        value_grid = tf.convert_to_tensor(value_grid, dtype=dtype, name='value_grid')
        if boundary_conditions[0][0] is None:
            has_default_lower_boundary = True
            lower_index = 0
        else:
            has_default_lower_boundary = False
            lower_index = 1
        if boundary_conditions[0][1] is None:
            upper_index = None
            has_default_upper_boundary = True
        else:
            upper_index = -1
            has_default_upper_boundary = False
        inner_grid_in = value_grid[..., lower_index:upper_index]
        coord_grid_deltas = coord_grid[0][..., 1:] - coord_grid[0][..., :-1]

        def equation_params_fn(t):
            if False:
                for i in range(10):
                    print('nop')
            return _construct_space_discretized_eqn_params(coord_grid, coord_grid_deltas, value_grid, boundary_conditions, has_default_lower_boundary, has_default_upper_boundary, second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn, inner_second_order_coeff_fn, inner_first_order_coeff_fn, t)
        inner_grid_out = time_marching_scheme(value_grid=inner_grid_in, t1=time, t2=next_time, equation_params_fn=equation_params_fn)
        updated_value_grid = _apply_boundary_conditions_after_step(inner_grid_out, boundary_conditions, has_default_lower_boundary, has_default_upper_boundary, coord_grid, coord_grid_deltas, next_time)
        return (coord_grid, updated_value_grid)

def _construct_space_discretized_eqn_params(coord_grid, coord_grid_deltas, value_grid, boundary_conditions, has_default_lower_boundary, has_default_upper_boundary, second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn, inner_second_order_coeff_fn, inner_first_order_coeff_fn, t):
    if False:
        for i in range(10):
            print('nop')
    'Constructs the tridiagonal matrix and the inhomogeneous term.'
    forward_deltas = coord_grid_deltas[..., 1:]
    backward_deltas = coord_grid_deltas[..., :-1]
    sum_deltas = forward_deltas + backward_deltas
    second_order_coeff_fn = second_order_coeff_fn or (lambda *args: [[None]])
    first_order_coeff_fn = first_order_coeff_fn or (lambda *args: [None])
    zeroth_order_coeff_fn = zeroth_order_coeff_fn or (lambda *args: None)
    inner_second_order_coeff_fn = inner_second_order_coeff_fn or (lambda *args: [[None]])
    inner_first_order_coeff_fn = inner_first_order_coeff_fn or (lambda *args: [None])
    second_order_coeff = _prepare_pde_coeffs(second_order_coeff_fn(t, coord_grid)[0][0], value_grid)
    first_order_coeff = _prepare_pde_coeffs(first_order_coeff_fn(t, coord_grid)[0], value_grid)
    zeroth_order_coeff = _prepare_pde_coeffs(zeroth_order_coeff_fn(t, coord_grid), value_grid)
    inner_second_order_coeff = _prepare_pde_coeffs(inner_second_order_coeff_fn(t, coord_grid)[0][0], value_grid)
    inner_first_order_coeff = _prepare_pde_coeffs(inner_first_order_coeff_fn(t, coord_grid)[0], value_grid)
    zeros = tf.zeros_like(value_grid[..., 1:-1])
    if zeroth_order_coeff is None:
        diag_zeroth_order = zeros
    else:
        diag_zeroth_order = -zeroth_order_coeff[..., 1:-1]
    if first_order_coeff is None and inner_first_order_coeff is None:
        superdiag_first_order = zeros
        diag_first_order = zeros
        subdiag_first_order = zeros
    else:
        superdiag_first_order = -backward_deltas / (sum_deltas * forward_deltas)
        subdiag_first_order = forward_deltas / (sum_deltas * backward_deltas)
        diag_first_order = -superdiag_first_order - subdiag_first_order
        if first_order_coeff is not None:
            superdiag_first_order *= first_order_coeff[..., 1:-1]
            subdiag_first_order *= first_order_coeff[..., 1:-1]
            diag_first_order *= first_order_coeff[..., 1:-1]
        if inner_first_order_coeff is not None:
            superdiag_first_order *= inner_first_order_coeff[..., 2:]
            subdiag_first_order *= inner_first_order_coeff[..., :-2]
            diag_first_order *= inner_first_order_coeff[..., 1:-1]
    if second_order_coeff is None and inner_second_order_coeff is None:
        superdiag_second_order = zeros
        diag_second_order = zeros
        subdiag_second_order = zeros
    else:
        superdiag_second_order = -2 / (sum_deltas * forward_deltas)
        subdiag_second_order = -2 / (sum_deltas * backward_deltas)
        diag_second_order = -superdiag_second_order - subdiag_second_order
        if second_order_coeff is not None:
            superdiag_second_order *= second_order_coeff[..., 1:-1]
            subdiag_second_order *= second_order_coeff[..., 1:-1]
            diag_second_order *= second_order_coeff[..., 1:-1]
        if inner_second_order_coeff is not None:
            superdiag_second_order *= inner_second_order_coeff[..., 2:]
            subdiag_second_order *= inner_second_order_coeff[..., :-2]
            diag_second_order *= inner_second_order_coeff[..., 1:-1]
    superdiag = superdiag_first_order + superdiag_second_order
    subdiag = subdiag_first_order + subdiag_second_order
    diag = diag_zeroth_order + diag_first_order + diag_second_order
    (subdiag, diag, superdiag) = _apply_default_boundary(subdiag, diag, superdiag, zeroth_order_coeff, inner_first_order_coeff, first_order_coeff, forward_deltas, backward_deltas, has_default_lower_boundary, has_default_upper_boundary)
    return _apply_robin_boundary_conditions(value_grid, boundary_conditions, has_default_lower_boundary, has_default_upper_boundary, coord_grid, coord_grid_deltas, diag, superdiag, subdiag, t)

def _apply_default_boundary(subdiag, diag, superdiag, zeroth_order_coeff, inner_first_order_coeff, first_order_coeff, forward_deltas, backward_deltas, has_default_lower_boundary, has_default_upper_boundary):
    if False:
        return 10
    'Update discretization matrix for default boundary conditions.'
    batch_shape = utils.get_shape(diag)[:-1]
    if zeroth_order_coeff is None:
        zeroth_order_coeff = tf.zeros([1], dtype=diag.dtype)
    if has_default_lower_boundary:
        (subdiag, diag, superdiag) = _apply_default_lower_boundary(subdiag, diag, superdiag, zeroth_order_coeff, inner_first_order_coeff, first_order_coeff, forward_deltas, batch_shape)
    if has_default_upper_boundary:
        (subdiag, diag, superdiag) = _apply_default_upper_boundary(subdiag, diag, superdiag, zeroth_order_coeff, inner_first_order_coeff, first_order_coeff, backward_deltas, batch_shape)
    return (subdiag, diag, superdiag)

def _apply_default_lower_boundary(subdiag, diag, superdiag, zeroth_order_coeff, inner_first_order_coeff, first_order_coeff, forward_deltas, batch_shape):
    if False:
        return 10
    'Update discretization matrix for default lower boundary conditions.'
    if inner_first_order_coeff is None:
        inner_coeff = tf.constant([1, 1], dtype=diag.dtype)
    else:
        inner_coeff = inner_first_order_coeff
    if first_order_coeff is None:
        if inner_first_order_coeff is None:
            extra_first_order_coeff = tf.zeros(batch_shape, dtype=diag.dtype)
        else:
            extra_first_order_coeff = tf.ones(batch_shape, dtype=diag.dtype)
    else:
        extra_first_order_coeff = first_order_coeff[..., 0]
    extra_superdiag_coeff = inner_coeff[..., 1] * extra_first_order_coeff / forward_deltas[..., 0]
    superdiag = _append_first(-extra_superdiag_coeff, superdiag)
    extra_diag_coeff = -inner_coeff[..., 0] * extra_first_order_coeff / forward_deltas[..., 0] + zeroth_order_coeff[..., 0]
    diag = _append_first(-extra_diag_coeff, diag)
    subdiag = _append_first(tf.zeros_like(extra_diag_coeff), subdiag)
    return (subdiag, diag, superdiag)

def _apply_default_upper_boundary(subdiag, diag, superdiag, zeroth_order_coeff, inner_first_order_coeff, first_order_coeff, backward_deltas, batch_shape):
    if False:
        i = 10
        return i + 15
    'Update discretization matrix for default upper boundary conditions.'
    if inner_first_order_coeff is None:
        inner_coeff = tf.constant([1, 1], dtype=diag.dtype)
    else:
        inner_coeff = inner_first_order_coeff
    if first_order_coeff is None:
        if inner_first_order_coeff is None:
            extra_first_order_coeff = tf.zeros(batch_shape, dtype=diag.dtype)
        else:
            extra_first_order_coeff = tf.ones(batch_shape, dtype=diag.dtype)
    else:
        extra_first_order_coeff = first_order_coeff[..., -1]
    extra_diag_coeff = inner_coeff[..., -1] * extra_first_order_coeff / backward_deltas[..., -1] + zeroth_order_coeff[..., -1]
    diag = _append_last(diag, -extra_diag_coeff)
    extra_sub_coeff = -inner_coeff[..., -2] * extra_first_order_coeff / backward_deltas[..., -1]
    subdiag = _append_last(subdiag, -extra_sub_coeff)
    superdiag = _append_last(superdiag, -tf.zeros_like(extra_diag_coeff))
    return (subdiag, diag, superdiag)

def _apply_robin_boundary_conditions(value_grid, boundary_conditions, has_default_lower_boundary, has_default_upper_boundary, coord_grid, coord_grid_deltas, diagonal, upper_diagonal, lower_diagonal, t):
    if False:
        i = 10
        return i + 15
    'Updates space-discretized equation according to boundary conditions.'
    if has_default_lower_boundary and has_default_upper_boundary:
        return ((diagonal, upper_diagonal, lower_diagonal), tf.zeros_like(diagonal))
    batch_shape = utils.get_shape(value_grid)[:-1]
    if has_default_lower_boundary:
        (alpha_l, beta_l, gamma_l) = (None, None, None)
    else:
        (alpha_l, beta_l, gamma_l) = boundary_conditions[0][0](t, coord_grid)
    if has_default_upper_boundary:
        (alpha_u, beta_u, gamma_u) = (None, None, None)
    else:
        (alpha_u, beta_u, gamma_u) = boundary_conditions[0][1](t, coord_grid)
    (alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u) = (_prepare_boundary_conditions(b, value_grid) for b in (alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u))
    if beta_l is None and beta_u is None:
        if has_default_lower_boundary:
            first_inhomog_element = tf.zeros(batch_shape, dtype=value_grid.dtype)
        else:
            first_inhomog_element = lower_diagonal[..., 0] * gamma_l / alpha_l
        if has_default_upper_boundary:
            last_inhomog_element = tf.zeros(batch_shape, dtype=value_grid.dtype)
        else:
            last_inhomog_element = upper_diagonal[..., -1] * gamma_u / alpha_u
        inhomog_term = _append_first_and_last(first_inhomog_element, tf.zeros_like(diagonal[..., 1:-1]), last_inhomog_element)
        return ((diagonal, upper_diagonal, lower_diagonal), inhomog_term)
    if has_default_lower_boundary:
        first_inhomog_element = tf.zeros(batch_shape, dtype=value_grid.dtype)
        diag_first_correction = 0
        upper_diag_correction = 0
    else:
        (xi1, xi2, eta) = _discretize_boundary_conditions(coord_grid_deltas[..., 0], coord_grid_deltas[..., 1], alpha_l, beta_l, gamma_l)
        diag_first_correction = lower_diagonal[..., 0] * xi1
        upper_diag_correction = lower_diagonal[..., 0] * xi2
        first_inhomog_element = lower_diagonal[..., 0] * eta
    if has_default_upper_boundary:
        last_inhomog_element = tf.zeros(batch_shape, dtype=value_grid.dtype)
        diag_last_correction = 0
        lower_diag_correction = 0
    else:
        (xi1, xi2, eta) = _discretize_boundary_conditions(coord_grid_deltas[..., -1], coord_grid_deltas[..., -2], alpha_u, beta_u, gamma_u)
        diag_last_correction = upper_diagonal[..., -1] * xi1
        lower_diag_correction = upper_diagonal[..., -1] * xi2
        last_inhomog_element = upper_diagonal[..., -1] * eta
    diagonal = _append_first_and_last(diagonal[..., 0] + diag_first_correction, diagonal[..., 1:-1], diagonal[..., -1] + diag_last_correction)
    upper_diagonal = _append_first(upper_diagonal[..., 0] + upper_diag_correction, upper_diagonal[..., 1:])
    lower_diagonal = _append_last(lower_diagonal[..., :-1], lower_diagonal[..., -1] + lower_diag_correction)
    inhomog_term = _append_first_and_last(first_inhomog_element, tf.zeros_like(diagonal[..., 1:-1]), last_inhomog_element)
    return ((diagonal, upper_diagonal, lower_diagonal), inhomog_term)

def _apply_boundary_conditions_after_step(inner_grid_out, boundary_conditions, has_default_lower_boundary, has_default_upper_boundary, coord_grid, coord_grid_deltas, time_after_step):
    if False:
        return 10
    'Calculates and appends boundary values after making a step.'
    if has_default_lower_boundary:
        first_value = None
    else:
        (alpha, beta, gamma) = boundary_conditions[0][0](time_after_step, coord_grid)
        (alpha, beta, gamma) = (_prepare_boundary_conditions(b, inner_grid_out) for b in (alpha, beta, gamma))
        (xi1, xi2, eta) = _discretize_boundary_conditions(coord_grid_deltas[..., 0], coord_grid_deltas[..., 1], alpha, beta, gamma)
        first_value = xi1 * inner_grid_out[..., 0] + xi2 * inner_grid_out[..., 1] + eta
    if has_default_upper_boundary:
        last_value = None
    else:
        (alpha, beta, gamma) = boundary_conditions[0][1](time_after_step, coord_grid)
        (alpha, beta, gamma) = (_prepare_boundary_conditions(b, inner_grid_out) for b in (alpha, beta, gamma))
        (xi1, xi2, eta) = _discretize_boundary_conditions(coord_grid_deltas[..., -1], coord_grid_deltas[..., -2], alpha, beta, gamma)
        last_value = xi1 * inner_grid_out[..., -1] + xi2 * inner_grid_out[..., -2] + eta
    return _append_first_and_last(first_value, inner_grid_out, last_value)

def _prepare_pde_coeffs(raw_coeffs, value_grid):
    if False:
        print('Hello World!')
    'Prepares values received from second_order_coeff_fn and similar.'
    if raw_coeffs is None:
        return None
    dtype = value_grid.dtype
    coeffs = tf.convert_to_tensor(raw_coeffs, dtype=dtype)
    broadcast_shape = utils.get_shape(value_grid)
    coeffs = tf.broadcast_to(coeffs, broadcast_shape)
    return coeffs

def _prepare_boundary_conditions(boundary_tensor, value_grid):
    if False:
        while True:
            i = 10
    'Prepares values received from boundary_condition callables.'
    if boundary_tensor is None:
        return None
    boundary_tensor = tf.convert_to_tensor(boundary_tensor, value_grid.dtype)
    broadcast_shape = utils.get_shape(value_grid)[:-1]
    return tf.broadcast_to(boundary_tensor, broadcast_shape)

def _discretize_boundary_conditions(dx0, dx1, alpha, beta, gamma):
    if False:
        return 10
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

def _append_first_and_last(first, inner, last):
    if False:
        return 10
    if first is None:
        return _append_last(inner, last)
    if last is None:
        return _append_first(first, inner)
    return tf.concat((tf.expand_dims(first, axis=-1), inner, tf.expand_dims(last, axis=-1)), axis=-1)

def _append_first(first, rest):
    if False:
        while True:
            i = 10
    if first is None:
        return rest
    return tf.concat((tf.expand_dims(first, axis=-1), rest), axis=-1)

def _append_last(rest, last):
    if False:
        while True:
            i = 10
    if last is None:
        return rest
    return tf.concat((rest, tf.expand_dims(last, axis=-1)), axis=-1)
__all__ = ['parabolic_equation_step']