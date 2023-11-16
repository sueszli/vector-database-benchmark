"""
Grid search tuner.

For categorical parameters this tuner fully explore all combinations.
For numerical parameters it samples them at progressively decreased intervals.
"""
__all__ = ['GridSearchTuner']
import logging
import math
import numpy as np
from scipy.special import erfinv
import nni
from nni.common.hpo_utils import ParameterSpec, deformat_parameters, format_search_space
from nni.tuner import Tuner
_logger = logging.getLogger('nni.tuner.gridsearch')

class GridSearchTuner(Tuner):
    """
    Grid search tuner divides search space into evenly spaced grid, and performs brute-force traverse.

    Recommended when the search space is small, or if you want to find strictly optimal hyperparameters.

    **Implementation**

    The original grid search approach performs an exhaustive search through a space consists of ``choice`` and ``randint``.

    NNI's implementation extends grid search to support all search spaces types.

    When the search space contains continuous parameters like ``normal`` and ``loguniform``,
    grid search tuner works in following steps:

    1. Divide the search space into a grid.
    2. Perform an exhaustive searth through the grid.
    3. Subdivide the grid into a finer-grained new grid.
    4. Goto step 2, until experiment end.

    As a deterministic algorithm, grid search has no argument.

    Examples
    --------

    .. code-block::

        config.tuner.name = 'GridSearch'
    """

    def __init__(self, optimize_mode=None):
        if False:
            print('Hello World!')
        self.space = None
        self.grid = None
        self.vector = None
        self.epoch_bar = None
        self.divisions = {}
        self.history = set()
        if optimize_mode is not None:
            _logger.info(f'Ignored optimize_mode "{optimize_mode}"')

    def update_search_space(self, space):
        if False:
            print('Hello World!')
        self.space = format_search_space(space)
        if not self.space:
            raise ValueError('Search space is empty')
        self._init_grid()

    def generate_parameters(self, *args, **kwargs):
        if False:
            return 10
        while True:
            params = self._suggest()
            if params is None:
                raise nni.NoMoreTrialError('Search space fully explored')
            params = deformat_parameters(params, self.space)
            params_str = nni.dump(params, sort_keys=True)
            if params_str not in self.history:
                self.history.add(params_str)
                return params

    def receive_trial_result(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        pass

    def import_data(self, data):
        if False:
            return 10
        for trial in data:
            params_str = nni.dump(trial['parameter'], sort_keys=True)
            self.history.add(params_str)

    def _suggest(self):
        if False:
            while True:
                i = 10
        while True:
            if self.grid is None:
                return None
            self._next_vector()
            if self.vector is None:
                self._next_grid()
                continue
            old = all((self.vector[i] < self.epoch_bar[i] for i in range(len(self.space))))
            if old:
                continue
            _logger.debug(f'vector: {self.vector}')
            return self._current_parameters()

    def _next_vector(self):
        if False:
            for i in range(10):
                print('nop')
        if self.vector is None:
            self.vector = [0] * len(self.space)
            return
        activated_dims = []
        params = self._current_parameters()
        for (i, spec) in enumerate(self.space.values()):
            if spec.is_activated_in(params):
                activated_dims.append(i)
        for i in reversed(activated_dims):
            if self.vector[i] + 1 < len(self.grid[i]):
                self.vector[i] += 1
                return
            else:
                self.vector[i] = 0
        self.vector = None

    def _next_grid(self):
        if False:
            i = 10
            return i + 15
        updated = False
        for (i, spec) in enumerate(self.space.values()):
            self.epoch_bar[i] = len(self.grid[i])
            if not spec.categorical:
                new_vals = []
                new_divs = []
                for (l, r) in self.divisions[i]:
                    mid = (l + r) / 2
                    diff_l = _less(l, mid, spec)
                    diff_r = _less(mid, r, spec)
                    if (diff_l or l == 0.0) and (diff_r or r == 1.0):
                        new_vals.append(mid)
                        updated = True
                    if diff_l:
                        new_divs.append((l, mid))
                        updated = updated or l == 0.0
                    if diff_r:
                        new_divs.append((mid, r))
                        updated = updated or r == 1.0
                self.grid[i] += new_vals
                self.divisions[i] = new_divs
        if not updated:
            _logger.info('Search space has been fully explored')
            self.grid = None
        else:
            size = _grid_size_info(self.grid)
            _logger.info(f'Grid subdivided, new size: {size}')

    def _init_grid(self):
        if False:
            while True:
                i = 10
        self.epoch_bar = [0 for _ in self.space]
        self.grid = [None for _ in self.space]
        for (i, spec) in enumerate(self.space.values()):
            if spec.categorical:
                self.grid[i] = list(range(spec.size))
            else:
                self.grid[i] = [0.5]
                self.divisions[i] = []
                if _less(0, 0.5, spec):
                    self.divisions[i].append((0, 0.5))
                if _less(0.5, 1, spec):
                    self.divisions[i].append((0.5, 1))
        size = _grid_size_info(self.grid)
        _logger.info(f'Grid initialized, size: {size}')

    def _current_parameters(self):
        if False:
            return 10
        params = {}
        for (i, spec) in enumerate(self.space.values()):
            if spec.is_activated_in(params):
                x = self.grid[i][self.vector[i]]
                if spec.categorical:
                    params[spec.key] = x
                else:
                    params[spec.key] = _cdf_inverse(x, spec)
        return params

def _less(x, y, spec):
    if False:
        print('Hello World!')
    real_x = _deformat_single_parameter(_cdf_inverse(x, spec), spec)
    real_y = _deformat_single_parameter(_cdf_inverse(y, spec), spec)
    return real_x < real_y

def _cdf_inverse(x, spec):
    if False:
        print('Hello World!')
    if spec.normal_distributed:
        return spec.mu + spec.sigma * math.sqrt(2) * erfinv(2 * x - 1)
    else:
        return spec.low + (spec.high - spec.low) * x

def _deformat_single_parameter(x, spec):
    if False:
        return 10
    if math.isinf(x):
        return x
    spec_dict = spec._asdict()
    spec_dict['key'] = (spec.name,)
    spec = ParameterSpec(**spec_dict)
    params = deformat_parameters({spec.key: x}, {spec.key: spec})
    return params[spec.name]

def _grid_size_info(grid):
    if False:
        print('Hello World!')
    if len(grid) == 1:
        return str(len(grid[0]))
    sizes = [len(candidates) for candidates in grid]
    mul = '×'.join((str(s) for s in sizes))
    total = np.prod(sizes)
    return f'({mul}) = {total}'