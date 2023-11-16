import warnings
from typing import Callable, Dict
from torch.autograd import grad
_EXCEPTION_HANDLERS: Dict[str, Callable[[Exception], bool]] = {}

def velocity_verlet(z, r, potential_fn, kinetic_grad, step_size, num_steps=1, z_grads=None):
    if False:
        while True:
            i = 10
    "\n    Second order symplectic integrator that uses the velocity verlet algorithm.\n\n    :param dict z: dictionary of sample site names and their current values\n        (type :class:`~torch.Tensor`).\n    :param dict r: dictionary of sample site names and corresponding momenta\n        (type :class:`~torch.Tensor`).\n    :param callable potential_fn: function that returns potential energy given z\n        for each sample site. The negative gradient of the function with respect\n        to ``z`` determines the rate of change of the corresponding sites'\n        momenta ``r``.\n    :param callable kinetic_grad: a function calculating gradient of kinetic energy\n        w.r.t. momentum variable.\n    :param float step_size: step size for each time step iteration.\n    :param int num_steps: number of discrete time steps over which to integrate.\n    :param torch.Tensor z_grads: optional gradients of potential energy at current ``z``.\n    :return tuple (z_next, r_next, z_grads, potential_energy): next position and momenta,\n        together with the potential energy and its gradient w.r.t. ``z_next``.\n    "
    z_next = z.copy()
    r_next = r.copy()
    for _ in range(num_steps):
        (z_next, r_next, z_grads, potential_energy) = _single_step_verlet(z_next, r_next, potential_fn, kinetic_grad, step_size, z_grads)
    return (z_next, r_next, z_grads, potential_energy)

def _single_step_verlet(z, r, potential_fn, kinetic_grad, step_size, z_grads=None):
    if False:
        i = 10
        return i + 15
    '\n    Single step velocity verlet that modifies the `z`, `r` dicts in place.\n    '
    z_grads = potential_grad(potential_fn, z)[0] if z_grads is None else z_grads
    for site_name in r:
        r[site_name] = r[site_name] + 0.5 * step_size * -z_grads[site_name]
    r_grads = kinetic_grad(r)
    for site_name in z:
        z[site_name] = z[site_name] + step_size * r_grads[site_name]
    (z_grads, potential_energy) = potential_grad(potential_fn, z)
    for site_name in r:
        r[site_name] = r[site_name] + 0.5 * step_size * -z_grads[site_name]
    return (z, r, z_grads, potential_energy)

def potential_grad(potential_fn, z):
    if False:
        while True:
            i = 10
    '\n    Gradient of `potential_fn` w.r.t. parameters z.\n\n    :param potential_fn: python callable that takes in a dictionary of parameters\n        and returns the potential energy.\n    :param dict z: dictionary of parameter values keyed by site name.\n    :return: tuple of `(z_grads, potential_energy)`, where `z_grads` is a dictionary\n        with the same keys as `z` containing gradients and potential_energy is a\n        torch scalar.\n    '
    (z_keys, z_nodes) = zip(*z.items())
    for node in z_nodes:
        node.requires_grad_(True)
    try:
        potential_energy = potential_fn(z)
    except Exception as e:
        if any((h(e) for h in _EXCEPTION_HANDLERS.values())):
            grads = {k: v.new_zeros(v.shape) for (k, v) in z.items()}
            return (grads, z_nodes[0].new_tensor(float('nan')))
        else:
            raise e
    grads = grad(potential_energy, z_nodes)
    for node in z_nodes:
        node.requires_grad_(False)
    return (dict(zip(z_keys, grads)), potential_energy.detach())

def register_exception_handler(name: str, handler: Callable[[Exception], bool], warn_on_overwrite: bool=True) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Register an exception handler for handling (primarily numerical) errors\n    when evaluating the potential function.\n\n    :param name: name of the handler (must be unique).\n    :param handler: A callable mapping an Exception to a boolean. Exceptions\n        that evaluate to true in any of the handlers are handled in the computation\n        of the potential energy.\n    :param warn_on_overwrite: If True, warns when overwriting a handler already\n        registered under the provided name.\n    '
    if name in _EXCEPTION_HANDLERS and warn_on_overwrite:
        warnings.warn(f'Overwriting Exception handler already registered under key {name}.', RuntimeWarning)
    _EXCEPTION_HANDLERS[name] = handler

def _handle_torch_singular(exception: Exception) -> bool:
    if False:
        return 10
    'Exception handler for errors thrown on (numerically) singular matrices.'
    if isinstance(exception, RuntimeError):
        msg = str(exception)
        return 'singular' in msg or 'input is not positive-definite' in msg
    return False
register_exception_handler('torch_singular', _handle_torch_singular)