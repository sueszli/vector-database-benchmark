"""Helper functions to construct boundary conditions of PDEs."""
import functools

def dirichlet(boundary_values_fn):
    if False:
        return 10
    "Wrapper for Dirichlet boundary conditions to be used in PDE solvers.\n\n  Example: the boundary value is 1 on both boundaries.\n\n  ```python\n  def lower_boundary_fn(t, location_grid):\n    return 1\n\n  def upper_boundary_fn(t, location_grid):\n    return 0\n\n  solver = fd_solvers.solve_forward(...,\n      boundary_conditions = [(dirichlet(lower_boundary_fn),\n                              dirichlet(upper_boundary_fn))],\n      ...)\n  ```\n\n  Also can be used as a decorator:\n\n  ```python\n  @dirichlet\n  def lower_boundary_fn(t, location_grid):\n    return 1\n\n  @dirichlet\n  def upper_boundary_fn(t, location_grid):\n    return 0\n\n  solver = fd_solvers.solve_forward(...,\n      boundary_conditions = [(lower_boundary_fn, upper_boundary_fn)],\n      ...)\n  ```\n\n  Args:\n    boundary_values_fn: Callable returning the boundary values at given time.\n      Accepts two arguments - the moment of time and the current coordinate\n      grid.\n      Returns a number, a zero-rank Tensor or a Tensor of shape\n      `batch_shape + grid_shape'`, where `grid_shape'` is grid_shape excluding\n      the axis orthogonal to the boundary. For example, in 3D the value grid\n      shape is `batch_shape + (z_size, y_size, x_size)`, and the boundary\n      tensors on the planes `y = y_min` and `y = y_max` should be either scalars\n      or have shape `batch_shape + (z_size, x_size)`. In 1D case this reduces\n      to just `batch_shape`.\n\n  Returns:\n    Callable suitable for PDE solvers.\n  "

    @functools.wraps(boundary_values_fn)
    def fn(t, x):
        if False:
            print('Hello World!')
        return (1, None, boundary_values_fn(t, x))
    return fn

def neumann(boundary_normal_derivative_fn):
    if False:
        print('Hello World!')
    "Wrapper for Neumann boundary condition to be used in PDE solvers.\n\n  Example: the normal boundary derivative is 1 on both boundaries (i.e.\n  `dV/dx = 1` on upper boundary, `dV/dx = -1` on lower boundary).\n\n  ```python\n  def lower_boundary_fn(t, location_grid):\n    return 1\n\n  def upper_boundary_fn(t, location_grid):\n    return 1\n\n  solver = fd_solvers.step_back(...,\n      boundary_conditions = [(neumann(lower_boundary_fn),\n                              neumann(upper_boundary_fn))],\n      ...)\n  ```\n\n  Also can be used as a decorator:\n\n  ```python\n  @neumann\n  def lower_boundary_fn(t, location_grid):\n    return 1\n\n  @neumann\n  def upper_boundary_fn(t, location_grid):\n    return 1\n\n  solver = fd_solvers.solve_forward(...,\n      boundary_conditions = [(lower_boundary_fn, upper_boundary_fn)],\n      ...)\n  ```\n\n  Args:\n    boundary_normal_derivative_fn: Callable returning the values of the\n      derivative with respect to the exterior normal to the boundary at the\n      given time.\n      Accepts two arguments - the moment of time and the current coordinate\n      grid.\n      Returns a number, a zero-rank Tensor or a Tensor of shape\n      `batch_shape + grid_shape'`, where `grid_shape'` is grid_shape excluding\n      the axis orthogonal to the boundary. For example, in 3D the value grid\n      shape is `batch_shape + (z_size, y_size, x_size)`, and the boundary\n      tensors on the planes `y = y_min` and `y = y_max` should be either scalars\n      or have shape `batch_shape + (z_size, x_size)`. In 1D case this reduces\n      to just `batch_shape`.\n\n  Returns:\n    Callable suitable for PDE solvers.\n  "

    @functools.wraps(boundary_normal_derivative_fn)
    def fn(t, x):
        if False:
            return 10
        return (None, 1, boundary_normal_derivative_fn(t, x))
    return fn
__all__ = ['dirichlet', 'neumann']