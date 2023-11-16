import math
from collections import namedtuple
import torch
import pyro
from pyro.ops.arrowhead import SymmArrowhead, sqrt, triu_gram, triu_inverse, triu_matvecmul
from pyro.ops.dual_averaging import DualAveraging
from pyro.ops.welford import WelfordArrowheadCovariance, WelfordCovariance
adapt_window = namedtuple('adapt_window', ['start', 'end'])

class WarmupAdapter:
    """
    Adapts tunable parameters, namely step size and mass matrix, during the
    warmup phase. This class provides lookup properties to read the latest
    values of ``step_size`` and ``inverse_mass_matrix``. These values are
    periodically updated when adaptation is engaged.
    """

    def __init__(self, step_size=1, adapt_step_size=False, target_accept_prob=0.8, adapt_mass_matrix=False, dense_mass=False):
        if False:
            print('Hello World!')
        self.adapt_step_size = adapt_step_size
        self.adapt_mass_matrix = adapt_mass_matrix
        self.target_accept_prob = target_accept_prob
        self.dense_mass = dense_mass
        self.step_size = 1 if step_size is None else step_size
        self._init_step_size = self.step_size
        self._adaptation_disabled = not (adapt_step_size or adapt_mass_matrix)
        if adapt_step_size:
            self._step_size_adapt_scheme = DualAveraging()
        self._mass_matrix_adapter = BlockMassMatrix()
        self._adapt_start_buffer = 75
        self._adapt_end_buffer = 50
        self._adapt_initial_window = 25
        self._warmup_steps = None
        self._adaptation_schedule = []

    def _build_adaptation_schedule(self):
        if False:
            return 10
        adaptation_schedule = []
        if self._warmup_steps < 20:
            adaptation_schedule.append(adapt_window(0, self._warmup_steps - 1))
            return adaptation_schedule
        start_buffer_size = self._adapt_start_buffer
        end_buffer_size = self._adapt_end_buffer
        init_window_size = self._adapt_initial_window
        if self._adapt_start_buffer + self._adapt_end_buffer + self._adapt_initial_window > self._warmup_steps:
            start_buffer_size = int(0.15 * self._warmup_steps)
            end_buffer_size = int(0.1 * self._warmup_steps)
            init_window_size = self._warmup_steps - start_buffer_size - end_buffer_size
        adaptation_schedule.append(adapt_window(start=0, end=start_buffer_size - 1))
        end_window_start = self._warmup_steps - end_buffer_size
        next_window_size = init_window_size
        next_window_start = start_buffer_size
        while next_window_start < end_window_start:
            (cur_window_start, cur_window_size) = (next_window_start, next_window_size)
            if 3 * cur_window_size <= end_window_start - cur_window_start:
                next_window_size = 2 * cur_window_size
            else:
                cur_window_size = end_window_start - cur_window_start
            next_window_start = cur_window_start + cur_window_size
            adaptation_schedule.append(adapt_window(cur_window_start, next_window_start - 1))
        adaptation_schedule.append(adapt_window(end_window_start, self._warmup_steps - 1))
        return adaptation_schedule

    def reset_step_size_adaptation(self, z):
        if False:
            while True:
                i = 10
        '\n        Finds a reasonable step size and resets step size adaptation scheme.\n        '
        if self._find_reasonable_step_size is not None:
            with pyro.validation_enabled(False):
                self.step_size = self._find_reasonable_step_size(z)
        self._step_size_adapt_scheme.prox_center = math.log(10 * self.step_size)
        self._step_size_adapt_scheme.reset()

    def _update_step_size(self, accept_prob):
        if False:
            i = 10
            return i + 15
        H = self.target_accept_prob - accept_prob
        self._step_size_adapt_scheme.step(H)
        (log_step_size, _) = self._step_size_adapt_scheme.get_state()
        self.step_size = math.exp(log_step_size)

    def _end_adaptation(self):
        if False:
            return 10
        if self.adapt_step_size:
            (_, log_step_size_avg) = self._step_size_adapt_scheme.get_state()
            self.step_size = math.exp(log_step_size_avg)

    def configure(self, warmup_steps, initial_step_size=None, mass_matrix_shape=None, find_reasonable_step_size_fn=None, options={}):
        if False:
            i = 10
            return i + 15
        '\n        Model specific properties that are specified when the HMC kernel is setup.\n\n        :param warmup_steps: Number of warmup steps that the sampler is initialized with.\n        :param initial_step_size: Step size to use to initialize the Dual Averaging scheme.\n        :param mass_matrix_shape: Shape of the mass matrix.\n        :param find_reasonable_step_size_fn: A callable to find reasonable step size when\n            mass matrix is changed.\n        :param dict options: A dict which maps `dtype`, `device` to the corresponding default\n            tensor options. This is used to construct initial mass matrix in `mass_matrix_adapter`.\n        '
        self._warmup_steps = warmup_steps
        self.step_size = initial_step_size if initial_step_size is not None else self._init_step_size
        if find_reasonable_step_size_fn is not None:
            self._find_reasonable_step_size = find_reasonable_step_size_fn
        if mass_matrix_shape is None or self.step_size is None:
            raise ValueError('Incomplete configuration - step size and inverse mass matrix need to be initialized.')
        self.mass_matrix_adapter.configure(mass_matrix_shape, self.adapt_mass_matrix, options=options)
        if not self._adaptation_disabled:
            self._adaptation_schedule = self._build_adaptation_schedule()
        self._current_window = 0
        if self.adapt_step_size:
            self._step_size_adapt_scheme.reset()

    def step(self, t, z, accept_prob, z_grad=None):
        if False:
            return 10
        '\n        Called at each step during the warmup phase to learn tunable\n        parameters.\n\n        :param int t: time step, beginning at 0.\n        :param dict z: latent variables.\n        :param float accept_prob: acceptance probability of the proposal.\n        '
        if t >= self._warmup_steps or self._adaptation_disabled:
            return
        window = self._adaptation_schedule[self._current_window]
        num_windows = len(self._adaptation_schedule)
        mass_matrix_adaptation_phase = self.adapt_mass_matrix and 0 < self._current_window < num_windows - 1
        if self.adapt_step_size:
            self._update_step_size(accept_prob.item())
        if mass_matrix_adaptation_phase:
            self.mass_matrix_adapter.update(z, z_grad)
        if t == window.end:
            if self._current_window == num_windows - 1:
                self._current_window += 1
                self._end_adaptation()
                return
            if self._current_window == 0:
                self._current_window += 1
                return
            if mass_matrix_adaptation_phase:
                self.mass_matrix_adapter.end_adaptation()
                if self.adapt_step_size:
                    self.reset_step_size_adaptation(z)
            self._current_window += 1

    @property
    def adaptation_schedule(self):
        if False:
            i = 10
            return i + 15
        return self._adaptation_schedule

    @property
    def mass_matrix_adapter(self):
        if False:
            while True:
                i = 10
        return self._mass_matrix_adapter

    @mass_matrix_adapter.setter
    def mass_matrix_adapter(self, value):
        if False:
            print('Hello World!')
        self._mass_matrix_adapter = value

def _matvecmul(x, y):
    if False:
        for i in range(10):
            print('nop')
    return x.mul(y) if x.dim() == 1 else x.matmul(y)

def _cholesky(x):
    if False:
        return 10
    return x.sqrt() if x.dim() == 1 else torch.linalg.cholesky(x)

def _transpose(x):
    if False:
        print('Hello World!')
    return x if x.dim() == 1 else x.t()

def _triu_inverse(x):
    if False:
        return 10
    if x.dim() == 1:
        return x.reciprocal()
    else:
        identity = torch.eye(x.size(-1), dtype=x.dtype, device=x.device)
        return torch.linalg.solve_triangular(x, identity, upper=True)

class BlockMassMatrix:
    """
    EXPERIMENTAL This class is used to adapt (inverse) mass matrix and provide
    useful methods to calculate algebraic terms which involves the mass matrix.

    The mass matrix will have block structure, which can be specified by
    using the method :meth:`configure` with the corresponding structured
    `mass_matrix_shape` arg.

    :param float init_scale: initial scale to construct the initial mass matrix.
    """

    def __init__(self, init_scale=1.0):
        if False:
            i = 10
            return i + 15
        self._init_scale = init_scale
        self._adapt_scheme = {}
        self._inverse_mass_matrix = {}
        self._mass_matrix_sqrt = {}
        self._mass_matrix_sqrt_inverse = {}
        self._mass_matrix_size = {}

    @property
    def mass_matrix_size(self):
        if False:
            i = 10
            return i + 15
        '\n        A dict that maps site names to the size of the corresponding mass matrix.\n        '
        return self._mass_matrix_size

    @property
    def inverse_mass_matrix(self):
        if False:
            for i in range(10):
                print('nop')
        return self._inverse_mass_matrix

    @inverse_mass_matrix.setter
    def inverse_mass_matrix(self, value):
        if False:
            print('Hello World!')
        for (site_names, inverse_mass_matrix) in value.items():
            if site_names in self._adapt_scheme:
                self._adapt_scheme[site_names].reset()
            mass_matrix_sqrt_inverse = _transpose(_cholesky(inverse_mass_matrix))
            mass_matrix_sqrt = _triu_inverse(mass_matrix_sqrt_inverse)
            self._inverse_mass_matrix[site_names] = inverse_mass_matrix
            self._mass_matrix_sqrt[site_names] = mass_matrix_sqrt
            self._mass_matrix_sqrt_inverse[site_names] = mass_matrix_sqrt_inverse

    def configure(self, mass_matrix_shape, adapt_mass_matrix=True, options={}):
        if False:
            i = 10
            return i + 15
        '\n        Sets up an initial mass matrix.\n\n        :param dict mass_matrix_shape: a dict that maps tuples of site names to the shape of\n            the corresponding mass matrix. Each tuple of site names corresponds to a block.\n        :param bool adapt_mass_matrix: a flag to decide whether an adaptation scheme will be used.\n        :param dict options: tensor options to construct the initial mass matrix.\n        '
        inverse_mass_matrix = {}
        for (site_names, shape) in mass_matrix_shape.items():
            self._mass_matrix_size[site_names] = shape[0]
            diagonal = len(shape) == 1
            inverse_mass_matrix[site_names] = torch.full(shape, self._init_scale, **options) if diagonal else torch.eye(*shape, **options) * self._init_scale
            if adapt_mass_matrix:
                adapt_scheme = WelfordCovariance(diagonal=diagonal)
                self._adapt_scheme[site_names] = adapt_scheme
        self.inverse_mass_matrix = inverse_mass_matrix

    def update(self, z, z_grad):
        if False:
            print('Hello World!')
        '\n        Updates the adaptation scheme using the new sample `z` or its grad `z_grad`.\n\n        :param dict z: the current value.\n        :param dict z_grad: grad of the current value.\n        '
        for (site_names, adapt_scheme) in self._adapt_scheme.items():
            z_flat = torch.cat([z[name].detach().reshape(-1) for name in site_names])
            adapt_scheme.update(z_flat)

    def end_adaptation(self):
        if False:
            i = 10
            return i + 15
        '\n        Updates the current mass matrix using the adaptation scheme.\n        '
        inverse_mass_matrix = {}
        for (site_names, adapt_scheme) in self._adapt_scheme.items():
            inverse_mass_matrix[site_names] = adapt_scheme.get_covariance(regularize=True)
        self.inverse_mass_matrix = inverse_mass_matrix

    def kinetic_grad(self, r):
        if False:
            return 10
        '\n        Computes the gradient of kinetic energy w.r.t. the momentum `r`.\n        It is equivalent to compute velocity given the momentum `r`.\n\n        :param dict r: a dictionary maps site names to a tensor momentum.\n        :returns: a dictionary maps site names to the corresponding gradient\n        '
        v = {}
        for (site_names, inverse_mass_matrix) in self._inverse_mass_matrix.items():
            r_flat = torch.cat([r[site_name].reshape(-1) for site_name in site_names])
            v_flat = _matvecmul(inverse_mass_matrix, r_flat)
            pos = 0
            for site_name in site_names:
                next_pos = pos + r[site_name].numel()
                v[site_name] = v_flat[pos:next_pos].reshape(r[site_name].shape)
                pos = next_pos
        return v

    def scale(self, r_unscaled, r_prototype):
        if False:
            i = 10
            return i + 15
        '\n        Computes `M^{1/2} @ r_unscaled`.\n\n        Note that `r` is generated from a gaussian with scale `mass_matrix_sqrt`.\n        This method will scale it.\n\n        :param dict r_unscaled: a dictionary maps site names to a tensor momentum.\n        :param dict r_prototype: a dictionary mapes site names to prototype momentum.\n            Those prototype values are used to get shapes of the scaled version.\n        :returns: a dictionary maps site names to the corresponding tensor\n        '
        s = {}
        for (site_names, mass_matrix_sqrt) in self._mass_matrix_sqrt.items():
            r_flat = _matvecmul(mass_matrix_sqrt, r_unscaled[site_names])
            pos = 0
            for site_name in site_names:
                next_pos = pos + r_prototype[site_name].numel()
                s[site_name] = r_flat[pos:next_pos].reshape(r_prototype[site_name].shape)
                pos = next_pos
        return s

    def unscale(self, r):
        if False:
            return 10
        '\n        Computes `inv(M^{1/2}) @ r`.\n\n        Note that `r` is generated from a gaussian with scale `mass_matrix_sqrt`.\n        This method will unscale it.\n\n        :param dict r: a dictionary maps site names to a tensor momentum.\n        :returns: a dictionary maps site names to the corresponding tensor\n        '
        u = {}
        for (site_names, mass_matrix_sqrt_inverse) in self._mass_matrix_sqrt_inverse.items():
            r_flat = torch.cat([r[site_name].reshape(-1) for site_name in site_names])
            u[site_names] = _matvecmul(mass_matrix_sqrt_inverse, r_flat)
        return u

class ArrowheadMassMatrix:
    """
    EXPERIMENTAL This class is used to adapt (inverse) mass matrix and provide useful
    methods to calculate algebraic terms which involves the mass matrix.

    The mass matrix will have arrowhead structure, with the head including all
    dense sites specified in the argument `full_mass` of the HMC/NUTS kernels.

    :param float init_scale: initial scale to construct the initial mass matrix.
    """

    def __init__(self, init_scale=1.0):
        if False:
            i = 10
            return i + 15
        self._init_scale = init_scale
        self._adapt_scheme = {}
        self._mass_matrix = {}
        self._mass_matrix_sqrt = {}
        self._mass_matrix_sqrt_inverse = {}
        self._mass_matrix_size = {}

    @property
    def mass_matrix_size(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A dict that maps site names to the size of the corresponding mass matrix.\n        '
        return self._mass_matrix_size

    @property
    def inverse_mass_matrix(self):
        if False:
            return 10
        inverse_mass_matrix = {}
        for (site_names, sqrt_inverse) in self._mass_matrix_sqrt_inverse.items():
            inverse_mass_matrix[site_names] = triu_gram(sqrt_inverse)
        return inverse_mass_matrix

    @property
    def mass_matrix(self):
        if False:
            i = 10
            return i + 15
        return self._mass_matrix

    @mass_matrix.setter
    def mass_matrix(self, value):
        if False:
            return 10
        for (site_names, mass_matrix) in value.items():
            self._adapt_scheme[site_names].reset()
            mass_matrix_sqrt = sqrt(mass_matrix)
            mass_matrix_sqrt_inverse = triu_inverse(mass_matrix_sqrt)
            self._mass_matrix[site_names] = mass_matrix
            self._mass_matrix_sqrt[site_names] = mass_matrix_sqrt
            self._mass_matrix_sqrt_inverse[site_names] = mass_matrix_sqrt_inverse

    def configure(self, mass_matrix_shape, adapt_mass_matrix=True, options={}):
        if False:
            print('Hello World!')
        '\n        Sets up an initial mass matrix.\n\n        :param dict mass_matrix_shape: a dict that maps tuples of site names to the shape of\n            the corresponding mass matrix. Each tuple of site names corresponds to a block.\n        :param bool adapt_mass_matrix: a flag to decide whether an adaptation scheme will be used.\n        :param dict options: tensor options to construct the initial mass matrix.\n        '
        mass_matrix = {}
        dense_sites = ()
        dense_size = 0
        diag_sites = ()
        diag_size = 0
        for (site_names, shape) in mass_matrix_shape.items():
            if len(shape) == 2:
                dense_sites = dense_sites + site_names
                dense_size = dense_size + shape[0]
            else:
                diag_sites = diag_sites + site_names
                diag_size = diag_size + shape[0]
        size = dense_size + diag_size
        head_size = dense_size
        all_sites = dense_sites + diag_sites
        self._mass_matrix_size[all_sites] = size
        top = torch.eye(head_size, size, **options) * self._init_scale
        bottom_diag = torch.full((size - head_size,), self._init_scale, **options)
        mass_matrix[all_sites] = SymmArrowhead(top, bottom_diag)
        if adapt_mass_matrix:
            adapt_scheme = WelfordArrowheadCovariance(head_size=head_size)
            self._adapt_scheme[all_sites] = adapt_scheme
        self.mass_matrix = mass_matrix

    def update(self, z, z_grad):
        if False:
            while True:
                i = 10
        '\n        Updates the adaptation scheme using the new sample `z` or its grad `z_grad`.\n\n        :param dict z: the current value.\n        :param dict z_grad: grad of the current value.\n        '
        for (site_names, adapt_scheme) in self._adapt_scheme.items():
            z_grad_flat = torch.cat([z_grad[name].reshape(-1) for name in site_names])
            adapt_scheme.update(z_grad_flat)

    def end_adaptation(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Updates the current mass matrix using the adaptation scheme.\n        '
        mass_matrix = {}
        for (site_names, adapt_scheme) in self._adapt_scheme.items():
            (top, bottom_diag) = adapt_scheme.get_covariance(regularize=True)
            mass_matrix[site_names] = SymmArrowhead(top, bottom_diag)
        self.mass_matrix = mass_matrix

    def kinetic_grad(self, r):
        if False:
            print('Hello World!')
        '\n        Computes the gradient of kinetic energy w.r.t. the momentum `r`.\n        It is equivalent to compute velocity given the momentum `r`.\n\n        :param dict r: a dictionary maps site names to a tensor momentum.\n        :returns: a dictionary maps site names to the corresponding gradient\n        '
        v = {}
        for (site_names, mass_matrix_sqrt_inverse) in self._mass_matrix_sqrt_inverse.items():
            r_flat = torch.cat([r[site_name].reshape(-1) for site_name in site_names])
            r_unscaled = triu_matvecmul(mass_matrix_sqrt_inverse, r_flat)
            v_flat = triu_matvecmul(mass_matrix_sqrt_inverse, r_unscaled, transpose=True)
            pos = 0
            for site_name in site_names:
                next_pos = pos + r[site_name].numel()
                v[site_name] = v_flat[pos:next_pos].reshape(r[site_name].shape)
                pos = next_pos
        return v

    def scale(self, r_unscaled, r_prototype):
        if False:
            print('Hello World!')
        '\n        Computes `M^{1/2} @ r_unscaled`.\n\n        Note that `r` is generated from a gaussian with scale `mass_matrix_sqrt`.\n        This method will scale it.\n\n        :param dict r_unscaled: a dictionary maps site names to a tensor momentum.\n        :param dict r_prototype: a dictionary mapes site names to prototype momentum.\n            Those prototype values are used to get shapes of the scaled version.\n        :returns: a dictionary maps site names to the corresponding tensor\n        '
        s = {}
        for (site_names, mass_matrix_sqrt) in self._mass_matrix_sqrt.items():
            r_flat = triu_matvecmul(mass_matrix_sqrt, r_unscaled[site_names])
            pos = 0
            for site_name in site_names:
                next_pos = pos + r_prototype[site_name].numel()
                s[site_name] = r_flat[pos:next_pos].reshape(r_prototype[site_name].shape)
                pos = next_pos
        return s

    def unscale(self, r):
        if False:
            return 10
        '\n        Computes `inv(M^{1/2}) @ r`.\n\n        Note that `r` is generated from a gaussian with scale `mass_matrix_sqrt`.\n        This method will unscale it.\n\n        :param dict r: a dictionary maps site names to a tensor momentum.\n        :returns: a dictionary maps site names to the corresponding tensor\n        '
        u = {}
        for (site_names, mass_matrix_sqrt_inverse) in self._mass_matrix_sqrt_inverse.items():
            r_flat = torch.cat([r[site_name].reshape(-1) for site_name in site_names])
            u[site_names] = triu_matvecmul(mass_matrix_sqrt_inverse, r_flat)
        return u