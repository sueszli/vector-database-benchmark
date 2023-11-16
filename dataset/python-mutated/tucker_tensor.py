from .base import FactorizedTensor
import ivy
from copy import deepcopy
import warnings

def _bisection_root_finder(fun, a, b, tol=1e-06, max_iter=100):
    if False:
        while True:
            i = 10
    if fun(a) * fun(b) >= 0:
        raise ValueError('Function values at the interval endpoints must have opposite signs')
    for _ in range(max_iter):
        c = (a + b) / 2
        if fun(c) == 0 or (b - a) / 2 < tol:
            return c
        if fun(c) * fun(a) < 0:
            b = c
        else:
            a = c
    raise RuntimeError('Bisection algorithm did not converge')

class TuckerTensor(FactorizedTensor):

    def __init__(self, tucker_tensor):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        (shape, rank) = TuckerTensor.validate_tucker_tensor(tucker_tensor)
        (core, factors) = tucker_tensor
        self.shape = tuple(shape)
        self.rank = tuple(rank)
        self.factors = factors
        self.core = core

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        if index == 0:
            return self.core
        elif index == 1:
            return self.factors
        else:
            raise IndexError(f'You tried to access index {index} of a Tucker tensor.\nYou can only access index 0 and 1 of a Tucker tensor(corresponding respectively to core and factors)')

    def __setitem__(self, index, value):
        if False:
            print('Hello World!')
        if index == 0:
            self.core = value
        elif index == 1:
            self.factors = value
        else:
            raise IndexError(f'You tried to set index {index} of a Tucker tensor.\nYou can only set index 0 and 1 of a Tucker tensor(corresponding respectively to core and factors)')

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.core
        yield self.factors

    def __len__(self):
        if False:
            while True:
                i = 10
        return 2

    def __repr__(self):
        if False:
            while True:
                i = 10
        message = f'Decomposed rank-{self.rank} TuckerTensor of shape {self.shape} '
        return message

    def to_tensor(self):
        if False:
            i = 10
            return i + 15
        return TuckerTensor.tucker_to_tensor(self)

    def to_unfolded(self, mode):
        if False:
            for i in range(10):
                print('nop')
        return TuckerTensor.tucker_to_unfolded(self, mode)

    def tucker_copy(self):
        if False:
            print('Hello World!')
        return TuckerTensor((deepcopy(self.core), [deepcopy(self.factors[i]) for i in range(len(self.factors))]))

    def to_vec(self):
        if False:
            i = 10
            return i + 15
        return TuckerTensor.tucker_to_vec(self)

    def mode_dot(self, matrix_or_vector, mode, keep_dim, copy):
        if False:
            while True:
                i = 10
        return TuckerTensor.tucker_mode_dot(self, matrix_or_vector, mode, keep_dim=keep_dim, copy=copy)

    @property
    def n_param(self):
        if False:
            return 10
        (core, factors) = (self.core, self.factors)
        total_params = sum((int(ivy.prod(tensor.shape)) for tensor in [core] + factors))
        return total_params

    @staticmethod
    def validate_tucker_tensor(tucker_tensor):
        if False:
            while True:
                i = 10
        (core, factors) = tucker_tensor
        if len(factors) < 2:
            raise ValueError(f'A Tucker tensor should be composed of at least two factors and a core.However, {len(factors)} factor was given.')
        if len(factors) != len(core.shape):
            raise ValueError(f'Tucker decompositions should have one factor per mode of the core tensor.However, core has {len(core.shape)} modes but {len(factors)} factors have been provided')
        shape = []
        rank = []
        for (i, factor) in enumerate(factors):
            (current_shape, current_rank) = factor.shape
            if current_rank != ivy.shape(core)[i]:
                raise ValueError(f'Factor `n` of Tucker decomposition should verify:\nfactors[n].shape[1] = core.shape[n].However, factors[{i}].shape[1]={ivy.shape(factor)[1]} but core.shape[{i}]={ivy.shape(core)[i]}.')
            shape.append(current_shape)
            rank.append(current_rank)
        return (tuple(shape), tuple(rank))

    @staticmethod
    def tucker_to_tensor(tucker_tensor, skip_factor=None, transpose_factors=False):
        if False:
            while True:
                i = 10
        (core, factors) = tucker_tensor
        return ivy.multi_mode_dot(core, factors, skip=skip_factor, transpose=transpose_factors)

    @staticmethod
    def tucker_normalize(tucker_tensor):
        if False:
            while True:
                i = 10
        (core, factors) = tucker_tensor
        normalized_factors = []
        for (i, factor) in enumerate(factors):
            scales = ivy.sqrt(ivy.sum(ivy.abs(factor) ** 2, axis=0))
            scales_non_zero = ivy.where(scales == 0, ivy.ones(ivy.shape(scales), dtype=factor[0].dtype), scales)
            core = core * ivy.reshape(scales, (1,) * i + (-1,) + (1,) * (len(core.shape) - i - 1))
            normalized_factors.append(factor / ivy.reshape(scales_non_zero, (1, -1)))
        return TuckerTensor((core, normalized_factors))

    @staticmethod
    def tucker_to_unfolded(tucker_tensor, mode=0, skip_factor=None, transpose_factors=False):
        if False:
            print('Hello World!')
        return ivy.unfold(TuckerTensor.tucker_to_tensor(tucker_tensor, skip_factor=skip_factor, transpose_factors=transpose_factors), mode)

    @staticmethod
    def tucker_to_vec(tucker_tensor, skip_factor=None, transpose_factors=False):
        if False:
            i = 10
            return i + 15
        return ivy.reshape(TuckerTensor.tucker_to_tensor(tucker_tensor, skip_factor=skip_factor, transpose_factors=transpose_factors), (-1,))

    @staticmethod
    def tucker_mode_dot(tucker_tensor, matrix_or_vector, mode, keep_dim=False, copy=False):
        if False:
            print('Hello World!')
        (shape, _) = TuckerTensor.validate_tucker_tensor(tucker_tensor)
        (core, factors) = tucker_tensor
        contract = False
        if len(matrix_or_vector.shape) == 2:
            if matrix_or_vector.shape[1] != shape[mode]:
                raise ValueError(f'shapes {shape} and {matrix_or_vector.shape} not aligned in mode-{mode} multiplication: {shape[mode]} (mode = {mode}) != {matrix_or_vector.shape[1]} (dim 1 of matrix)')
        elif len(matrix_or_vector.shape) == 1:
            if matrix_or_vector.shape[0] != shape[mode]:
                raise ValueError(f'shapes {shape} and {matrix_or_vector.shape} not aligned for mode-{mode} multiplication: {shape[mode]} (mode = {mode}) != {matrix_or_vector.shape[0]} (vector size)')
            if not keep_dim:
                contract = True
        else:
            raise ValueError('Can only take n_mode_product with a vector or a matrix.')
        if copy:
            factors = [deepcopy(f) for f in factors]
            core = deepcopy(core)
        if contract:
            print('contracting mode')
            f = factors.pop(mode)
            core = ivy.mode_dot(core, ivy.dot(matrix_or_vector, f), mode=mode)
        else:
            factors[mode] = ivy.dot(matrix_or_vector, factors[mode])
        return TuckerTensor((core, factors))

    @staticmethod
    def validate_tucker_rank(tensor_shape, rank='same', rounding='round', fixed_modes=None):
        if False:
            print('Hello World!')
        if rounding == 'ceil':
            rounding_fun = ivy.ceil
        elif rounding == 'floor':
            rounding_fun = ivy.floor
        elif rounding == 'round':
            rounding_fun = ivy.round
        else:
            raise ValueError(f'Rounding should be round, floor or ceil, but got {rounding}')
        if rank == 'same':
            rank = float(1)
        if isinstance(rank, float):
            n_modes_compressed = len(tensor_shape)
            n_param_tensor = int(ivy.prod(tensor_shape))
            if fixed_modes is not None:
                tensor_shape = list(tensor_shape)
                fixed_modes = [(mode, tensor_shape.pop(mode)) for mode in sorted(fixed_modes, reverse=True)][::-1]
                n_fixed_params = ivy.sum([s ** 2 for (_, s) in fixed_modes])
                n_modes_compressed -= len(fixed_modes)
            else:
                n_fixed_params = 0
            squared_dims = ivy.sum([s ** 2 for s in tensor_shape])

            def fun(x):
                if False:
                    i = 10
                    return i + 15
                return n_param_tensor * x ** n_modes_compressed + squared_dims * x + n_fixed_params * x - rank * n_param_tensor
            fraction_param = _bisection_root_finder(fun, 0.0, max(rank, 1.0))
            rank = [max(int(rounding_fun(s * fraction_param)), 1) for s in tensor_shape]
            if fixed_modes is not None:
                for (mode, size) in fixed_modes:
                    rank.insert(mode, size)
        elif isinstance(rank, int):
            n_modes = len(tensor_shape)
            warnings.warn(f"Given only one int for 'rank' for decomposition a tensor of order {n_modes}. Using this rank for all modes.")
            if fixed_modes is None:
                rank = [rank] * n_modes
            else:
                rank = [rank if i not in fixed_modes else s for (i, s) in enumerate(tensor_shape)]
        return rank

    @staticmethod
    def tucker_n_param(shape, rank):
        if False:
            for i in range(10):
                print('nop')
        core_params = ivy.prod(rank)
        factors_params = ivy.sum([r * s for (r, s) in zip(rank, shape)])
        return int(core_params + factors_params)