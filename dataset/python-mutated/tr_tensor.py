from .base import FactorizedTensor
import ivy
import warnings

class TRTensor(FactorizedTensor):

    def __init__(self, factors):
        if False:
            return 10
        super().__init__()
        (shape, rank) = TRTensor.validate_tr_tensor(factors)
        self.shape = tuple(shape)
        self.rank = tuple(rank)
        self.factors = factors

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        return self.factors[index]

    def __setitem__(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        self.factors[index] = value

    def __iter__(self):
        if False:
            print('Hello World!')
        for index in range(len(self)):
            yield self[index]

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.factors)

    def __repr__(self):
        if False:
            print('Hello World!')
        message = f'factors list : rank-{self.rank} tensor ring tensor of shape {self.shape}'
        return message

    def to_tensor(self):
        if False:
            i = 10
            return i + 15
        return TRTensor.tr_to_tensor(self.factors)

    def to_unfolded(self, mode):
        if False:
            print('Hello World!')
        return TRTensor.tr_to_unfolded(self.factors, mode)

    def to_vec(self):
        if False:
            i = 10
            return i + 15
        return TRTensor.tr_to_vec(self.factors)

    @property
    def n_param(self):
        if False:
            i = 10
            return i + 15
        factors = self.factors
        total_params = sum((int(ivy.prod(tensor.shape)) for tensor in factors))
        return total_params

    @staticmethod
    def validate_tr_tensor(factors):
        if False:
            while True:
                i = 10
        n_factors = len(factors)
        if n_factors < 2:
            raise ValueError(f'A Tensor Ring tensor should be composed of at least two factors.However, {n_factors} factor was given.')
        rank = []
        shape = []
        next_rank = None
        for (index, factor) in enumerate(factors):
            (current_rank, current_shape, next_rank) = ivy.shape(factor)
            if len(factor.shape) != 3:
                raise ValueError(f'TR expresses a tensor as third order factors (tr-cores).\nHowever, ivy.ndim(factors[{index}]) = {len(factor.shape)}')
            if ivy.shape(factors[index - 1])[2] != current_rank:
                raise ValueError(f'Consecutive factors should have matching ranks\n -- e.g. ivy.shape(factors[0])[2]) == ivy.shape(factors[1])[0])\nHowever, ivy.shape(factor[{index - 1}])[2] == {ivy.shape(factors[index - 1])[2]} but ivy.shape(factor[{index}])[0] == {current_rank}')
            shape.append(current_shape)
            rank.append(current_rank)
        rank.append(next_rank)
        return (tuple(shape), tuple(rank))

    @staticmethod
    def tr_to_tensor(factors):
        if False:
            for i in range(10):
                print('nop')
        full_shape = [f.shape[1] for f in factors]
        full_tensor = ivy.reshape(factors[0], (-1, factors[0].shape[2]))
        for factor in factors[1:-1]:
            (rank_prev, _, rank_next) = factor.shape
            factor = ivy.reshape(factor, (rank_prev, -1))
            full_tensor = ivy.dot(full_tensor, factor)
            full_tensor = ivy.reshape(full_tensor, (-1, rank_next))
        full_tensor = ivy.reshape(full_tensor, (factors[-1].shape[2], -1, factors[-1].shape[0]))
        full_tensor = ivy.moveaxis(full_tensor, 0, -1)
        full_tensor = ivy.reshape(full_tensor, (-1, factors[-1].shape[0] * factors[-1].shape[2]))
        factor = ivy.moveaxis(factors[-1], -1, 1)
        factor = ivy.reshape(factor, (-1, full_shape[-1]))
        full_tensor = ivy.dot(full_tensor, factor)
        return ivy.reshape(full_tensor, full_shape)

    @staticmethod
    def tr_to_unfolded(factors, mode):
        if False:
            while True:
                i = 10
        return ivy.unfold(TRTensor.tr_to_tensor(factors), mode)

    @staticmethod
    def tr_to_vec(factors):
        if False:
            for i in range(10):
                print('nop')
        return ivy.reshape(TRTensor.tr_to_tensor(factors), (-1,))

    @staticmethod
    def validate_tr_rank(tensor_shape, rank='same', rounding='round'):
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
        n_dim = len(tensor_shape)
        if n_dim == 2:
            warnings.warn(f'Determining the TR-rank for the trivial case of a matrix (order 2 tensor) of shape {tensor_shape}, not a higher-order tensor.')
        if isinstance(rank, float):
            n_param_tensor = ivy.prod(tensor_shape) * rank
            solution = int(rounding_fun(ivy.sqrt(n_param_tensor / ivy.sum(tensor_shape))))
            rank = (solution,) * (n_dim + 1)
        else:
            n_dim = len(tensor_shape)
            if isinstance(rank, int):
                rank = (rank,) * (n_dim + 1)
            elif n_dim + 1 != len(rank):
                message = f'Provided incorrect number of ranks. Should verify len(rank) == len(tensor.shape)+1, but len(rank) = {len(rank)} while len(tensor.shape)+1 = {n_dim + 1}'
                raise ValueError(message)
            if rank[0] != rank[-1]:
                message = f'Provided rank[0] == {rank[0]} and rank[-1] == {rank[-1]} but boundary conditions dictate rank[0] == rank[-1]'
                raise ValueError(message)
        return list(rank)

    @staticmethod
    def tr_n_param(tensor_shape, rank):
        if False:
            for i in range(10):
                print('nop')
        factor_params = []
        for (i, s) in enumerate(tensor_shape):
            factor_params.append(rank[i] * s * rank[i + 1])
        return ivy.sum(factor_params)