from .base import FactorizedTensor
import ivy
from copy import deepcopy

class Parafac2Tensor(FactorizedTensor):

    def __init__(self, parafac2_tensor):
        if False:
            while True:
                i = 10
        super().__init__()
        (shape, rank) = ivy.Parafac2Tensor.validate_parafac2_tensor(parafac2_tensor)
        (weights, factors, projections) = parafac2_tensor
        if weights is None:
            weights = ivy.ones(rank, dtype=factors[0].dtype)
        self.shape = shape
        self.rank = rank
        self.factors = factors
        self.weights = weights
        self.projections = projections

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        if index == 0:
            return self.weights
        elif index == 1:
            return self.factors
        elif index == 2:
            return self.projections
        else:
            raise IndexError(f'You tried to access index {index} of a PARAFAC2 tensor.\nYou can only access index 0, 1 and 2 of a PARAFAC2 tensor(corresponding respectively to the weights, factors and projections)')

    def __setitem__(self, index, value):
        if False:
            i = 10
            return i + 15
        if index == 0:
            self.weights = value
        elif index == 1:
            self.factors = value
        elif index == 2:
            self.projections = value
        else:
            raise IndexError(f'You tried to set index {index} of a PARAFAC2 tensor.\nYou can only set index 0, 1 and 2 of a PARAFAC2 tensor(corresponding respectively to the weights, factors and projections)')

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.weights
        yield self.factors
        yield self.projections

    def __len__(self):
        if False:
            print('Hello World!')
        return 3

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        message = f'(weights, factors, projections) : rank-{self.rank} Parafac2Tensor of shape {self.shape} '
        return message

    def to_tensor(self):
        if False:
            while True:
                i = 10
        return ivy.Parafac2Tensor.parafac2_to_tensor(self)

    def to_vec(self):
        if False:
            for i in range(10):
                print('nop')
        return ivy.Parafac2Tensor.parafac2_to_vec(self)

    def to_unfolded(self, mode):
        if False:
            return 10
        return ivy.Parafac2Tensor.parafac2_to_unfolded(self, mode)

    @property
    def n_param(self):
        if False:
            i = 10
            return i + 15
        factors_params = self.rank * ivy.sum(self.shape)
        if self.weights:
            return factors_params + self.rank
        else:
            return factors_params

    @classmethod
    def from_CPTensor(cls, cp_tensor, parafac2_tensor_ok=False):
        if False:
            print('Hello World!')
        "\n        Create a Parafac2Tensor from a CPTensor.\n\n        Parameters\n        ----------\n        cp_tensor\n            CPTensor or Parafac2Tensor\n            If it is a Parafac2Tensor, then the argument\n            ``parafac2_tensor_ok`` must be True'\n        parafac2_tensor\n            Whether or not Parafac2Tensors can be used as input.\n\n        Returns\n        -------\n            Parafac2Tensor with factor matrices and weights extracted from a CPTensor\n        "
        if parafac2_tensor_ok and len(cp_tensor) == 3:
            return Parafac2Tensor(cp_tensor)
        elif len(cp_tensor) == 3:
            raise TypeError('Input is not a CPTensor. If it is a Parafac2Tensor, then the argument ``parafac2_tensor_ok`` must be True')
        (weights, (A, B, C)) = cp_tensor
        (Q, R) = ivy.qr(B)
        projections = [Q for _ in range(ivy.shape(A)[0])]
        B = R
        return Parafac2Tensor((weights, (A, B, C), projections))

    @staticmethod
    def validate_parafac2_tensor(parafac2_tensor):
        if False:
            i = 10
            return i + 15
        '\n        Validate a parafac2_tensor in the form (weights, factors) Return the rank and\n        shape of the validated tensor.\n\n        Parameters\n        ----------\n        parafac2_tensor\n            Parafac2Tensor or (weights, factors)\n\n        Returns\n        -------\n        (shape, rank)\n            size of the full tensor and rank of the CP tensor\n        '
        if isinstance(parafac2_tensor, ivy.Parafac2Tensor):
            return (parafac2_tensor.shape, parafac2_tensor.rank)
        (weights, factors, projections) = parafac2_tensor
        if len(factors) != 3:
            raise ValueError(f'A PARAFAC2 tensor should be composed of exactly three factors.However, {len(factors)} factors was given.')
        if len(projections) != factors[0].shape[0]:
            raise ValueError(f'A PARAFAC2 tensor should have one projection matrix for each horisontal slice. However, {len(projections)} projection matrices was given and the first mode haslength {factors[0].shape[0]}')
        rank = int(ivy.shape(factors[0])[1])
        shape = []
        for (i, projection) in enumerate(projections):
            (current_mode_size, current_rank) = ivy.shape(projection)
            if current_rank != rank:
                raise ValueError(f'All the projection matrices of a PARAFAC2 tensor should have the same number of columns as the rank. However, rank={rank} but projections[{i}].shape[1]={ivy.shape(projection)[1]}')
            inner_product = ivy.dot(ivy.permute_dims(projection, (1, 0)), projection)
            if ivy.max(ivy.abs(inner_product - ivy.eye(rank, dtype=inner_product[0].dtype))) > 1e-05:
                raise ValueError(f'All the projection matrices must be orthonormal, that is, P.T@P = I. However, projection[{i}].T@projection[{i}] - T.eye(rank)) = {ivy.sqrt(ivy.sum(ivy.square(inner_product - ivy.eye(rank, dtype=inner_product[0].dtype)), axis=0))}')
            shape.append((current_mode_size, *[f.shape[0] for f in factors[2:]]))
        for (i, factor) in enumerate(factors[1:]):
            (current_mode_size, current_rank) = ivy.shape(factor)
            if current_rank != rank:
                raise ValueError(f'All the factors of a PARAFAC2 tensor should have the same number of columns.However, factors[0].shape[1]={rank} but factors[{i}].shape[1]={current_rank}.')
        if weights is not None and ivy.shape(weights)[0] != rank:
            raise ValueError(f'Given factors for a rank-{rank} PARAFAC2 tensor but len(weights)={ivy.shape(weights)[0]}.')
        return (tuple(shape), rank)

    @staticmethod
    def parafac2_normalise(parafac2_tensor):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return parafac2_tensor with factors normalised to unit length.\n\n        Turns ``factors = [|U_1, ... U_n|]`` into ``[weights; |V_1, ... V_n|]``,\n        where the columns of each `V_k` are normalized to unit Euclidean length\n        from the columns of `U_k` with the normalizing constants absorbed into\n        `weights`. In the special case of a symmetric tensor, `weights` holds the\n        eigenvalues of the tensor.\n\n        Parameters\n        ----------\n        parafac2_tensor\n            Parafac2Tensor = (weight, factors, projections)\n            factors is list of matrices, all with the same number of columns\n            i.e.::\n                for u in U:\n                    u[i].shape == (s_i, R)\n\n            where `R` is fixed while `s_i` can vary with `i`\n\n        Returns\n        -------\n        Parafac2Tensor\n          normalisation_weights, normalised_factors, normalised_projections\n        '
        (_, rank) = ivy.Parafac2Tensor.validate_parafac2_tensor(parafac2_tensor)
        (weights, factors, projections) = parafac2_tensor
        if True:
            factors = [deepcopy(f) for f in factors]
            projections = [deepcopy(p) for p in projections]
            if weights is not None:
                factors[0] = factors[0] * weights
            weights = ivy.ones(rank, dtype=factors[0].dtype)
        for (i, factor) in enumerate(factors):
            scales = ivy.sqrt(ivy.sum(ivy.square(factor), axis=0))
            weights = weights * scales
            scales_non_zero = ivy.where(scales == 0, ivy.ones(ivy.shape(scales), dtype=factors[0].dtype), scales)
            factors[i] = factor / scales_non_zero
        return Parafac2Tensor((weights, factors, projections))

    @staticmethod
    def apply_parafac2_projections(parafac2_tensor):
        if False:
            i = 10
            return i + 15
        '\n        Apply the projection matrices to the evolving factor.\n\n        Parameters\n        ----------\n        parafac2_tensor : Parafac2Tensor\n\n        Returns\n        -------\n        (weights, factors)\n            A tensor decomposition on the form A [B_i] C such that\n            the :math:`X_{ijk}` is given by :math:`sum_r A_{ir} [B_i]_{jr} C_{kr}`.\n\n            This is also equivalent to a coupled matrix factorisation, where\n            each matrix, :math:`X_i = C diag([a_{i1}, ..., a_{ir}] B_i)`.\n\n            The first element of factors is the A matrix, the second element is\n            a list of B-matrices and the third element is the C matrix.\n        '
        ivy.Parafac2Tensor.validate_parafac2_tensor(parafac2_tensor)
        (weights, factors, projections) = parafac2_tensor
        evolving_factor = [ivy.dot(projection, factors[1]) for projection in projections]
        return (weights, (factors[0], evolving_factor, factors[2]))

    @staticmethod
    def parafac2_to_slice(parafac2_tensor, slice_idx, validate=True):
        if False:
            while True:
                i = 10
        '\n        Generate a single slice along the first mode from the PARAFAC2 tensor.\n\n        The decomposition is on the form :math:`(A [B_i] C)` such that the\n        i-th frontal slice, :math:`X_i`, of :math:`X` is given by\n\n        .. math::\n\n            X_i = B_i diag(a_i) C^T,\n\n        where :math:`diag(a_i)` is the diagonal matrix whose nonzero\n        entries are equal to the :math:`i`-th row of the :math:`I times R`\n        factor matrix :math:`A`, :math:`B_i`is a :math:`J_i times R` factor\n        matrix such that the cross product matrix :math:`B_{i_1}^T B_{i_1}` is\n        constant for all :math:`i`, and :math:`C` is a :math:`K times R`\n        factor matrix. To compute this decomposition, we reformulate\n        the expression for :math:`B_i` such that\n\n        .. math::\n\n            B_i = P_i B,\n\n        where :math:`P_i` is a :math:`J_i times R` orthogonal matrix and :math:`B`\n        is a :math:`R times R` matrix.\n\n        An alternative formulation of the PARAFAC2 decomposition is\n        that the tensor element :math:`X_{ijk}` is given by\n\n        .. math::\n\n            X_{ijk}\xa0= sum_{r=1}^R A_{ir} B_{ijr} C_{kr},\n\n        with the same constraints hold for :math:`B_i` as above.\n\n        Parameters\n        ----------\n        parafac2_tensor\n             weights\n                1D array of shape (rank, ) weights of the factors\n            factors\n                List of factors of the PARAFAC2 decomposition Contains the\n                matrices :math:`A`, :math:`B` and :math:`C` described above\n            projection_matrices\n                 List of projection matrices used to create evolving factors.\n\n        Returns\n        -------\n            Full tensor of shape [P[slice_idx].shape[1], C.shape[1]], where\n            P is the projection matrices and C is the last factor matrix of\n            the Parafac2Tensor.\n        '
        if validate:
            ivy.Parafac2Tensor.validate_parafac2_tensor(parafac2_tensor)
        (weights, (A, B, C), projections) = parafac2_tensor
        a = A[slice_idx]
        if weights is not None:
            a = a * weights
        Ct = ivy.permute_dims(C, (1, 0))
        B_i = ivy.dot(projections[slice_idx], B)
        return ivy.dot(B_i * a, Ct)

    @staticmethod
    def parafac2_to_slices(parafac2_tensor, validate=True):
        if False:
            return 10
        '\n        Generate all slices along the first mode from a PARAFAC2 tensor.\n\n        Generates a list of all slices from a PARAFAC2 tensor. A list is returned\n        since the tensor might have varying size along the second mode. To return\n        a tensor, see the ``parafac2_to_tensor`` function instead.shape\n\n        The decomposition is on the form :math:`(A [B_i] C)` such that\n        the i-th frontal slice, :math:`X_i`, of :math:`X` is given by\n\n        .. math::\n\n            X_i = B_i diag(a_i) C^T,\n\n        where :math:`diag(a_i)` is the diagonal matrix whose nonzero entries are\n        equal to the :math:`i`-th row of the :math:`I times R` factor matrix\n        :math:`A`, :math:`B_i` is a :math:`J_i times R` factor matrix such\n        that the cross product matrix :math:`B_{i_1}^T B_{i_1}` is constant\n        for all :math:`i`, and :math:`C` is a :math:`K times R` factor matrix.To\n        compute this decomposition, we reformulate the expression for :math:`B_i`\n        such that\n\n        .. math::\n\n            B_i = P_i B,\n\n        where :math:`P_i` is a :math:`J_i times R` orthogonal matrix and :math:`B`\n        is a :math:`R times R` matrix.\n\n        An alternative formulation of the PARAFAC2 decomposition is that the\n        tensor element :math:`X_{ijk}` is given by\n\n        .. math::\n\n            X_{ijk}\xa0= sum_{r=1}^R A_{ir} B_{ijr} C_{kr},\n\n        with the same constraints hold for :math:`B_i` as above.\n\n        Parameters\n        ----------\n        parafac2_tensor : Parafac2Tensor - (weight, factors, projection_matrices)\n            * weights : 1D array of shape (rank, )\n                weights of the factors\n            * factors : List of factors of the PARAFAC2 decomposition\n                Contains the matrices :math:`A`, :math:`B` and :math:`C` described above\n            * projection_matrices : List of projection matrices used to create evolving\n                factors.\n\n        Returns\n        -------\n            A list of full tensors of shapes [P[i].shape[1], C.shape[1]], where\n            P is the projection matrices and C is the last factor matrix of the\n            Parafac2Tensor.\n        '
        if validate:
            ivy.Parafac2Tensor.validate_parafac2_tensor(parafac2_tensor)
        (weights, (A, B, C), projections) = parafac2_tensor
        if weights is not None:
            A = A * weights
            weights = None
        decomposition = (weights, (A, B, C), projections)
        (I, _) = A.shape
        return [ivy.Parafac2Tensor.parafac2_to_slice(decomposition, i, validate=False) for i in range(I)]

    def parafac2_to_tensor(parafac2_tensor):
        if False:
            while True:
                i = 10
        '\n        Construct a full tensor from a PARAFAC2 decomposition.\n\n        The decomposition is on the form :math:`(A [B_i] C)` such that the\n        i-th frontal slice, :math:`X_i`, of :math:`X` is given by\n\n        .. math::\n\n            X_i = B_i diag(a_i) C^T,\n\n        where :math:`diag(a_i)` is the diagonal matrix whose nonzero entries\n        are equal to the :math:`i`-th row of the :math:`I times R` factor\n        matrix :math:`A`, :math:`B_i` is a :math:`J_i times R` factor matrix\n        such that the cross product matrix :math:`B_{i_1}^T B_{i_1}` is\n        constant for all :math:`i`, and :math:`C` is a :math:`K times R`\n        factor matrix. To compute this decomposition, we reformulate\n        the expression for :math:`B_i` such that\n\n        .. math::\n\n            B_i = P_i B,\n\n        where :math:`P_i` is a :math:`J_i times R` orthogonal matrix and :math:`B`\n        is a :math:`R times R` matrix.\n\n        An alternative formulation of the PARAFAC2 decomposition is\n        that the tensor element :math:`X_{ijk}` is given by\n\n        .. math::\n\n            X_{ijk} = sum_{r=1}^R A_{ir} B_{ijr} C_{kr},\n\n        with the same constraints hold for :math:`B_i` as above.\n\n        Parameters\n        ----------\n        parafac2_tensor : Parafac2Tensor - (weight, factors, projection_matrices)\n            * weights : 1D array of shape (rank, )\n                weights of the factors\n            * factors : List of factors of the PARAFAC2 decomposition\n                Contains the matrices :math:`A`, :math:`B` and :math:`C` described above\n            * projection_matrices : List of projection matrices used to create evolving\n                factors.\n\n        Returns\n        -------\n        ndarray\n            Full constructed tensor. Uneven slices are padded with zeros.\n        '
        (_, (A, _, C), projections) = parafac2_tensor
        slices = ivy.Parafac2Tensor.parafac2_to_slices(parafac2_tensor)
        lengths = [projection.shape[0] for projection in projections]
        tensor = ivy.zeros((A.shape[0], max(lengths), C.shape[0]), dtype=slices[0].dtype)
        for (i, (slice_, length)) in enumerate(zip(slices, lengths)):
            tensor[i, :length] = slice_
        return tensor

    def parafac2_to_unfolded(parafac2_tensor, mode):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct an unfolded tensor from a PARAFAC2 decomposition. Uneven slices are\n        padded by zeros.\n\n        The decomposition is on the form :math:`(A [B_i] C)` such that the\n        i-th frontal slice, :math:`X_i`, of :math:`X` is given by\n\n        .. math::\n\n            X_i = B_i diag(a_i) C^T,\n\n        where :math:`diag(a_i)` is the diagonal matrix whose nonzero entries\n        are equal to the :math:`i`-th row of the :math:`I times R` factor\n        matrix :math:`A`, :math:`B_i` is a :math:`J_i times R` factor\n        matrix such that the cross product matrix :math:`B_{i_1}^T B_{i_1}`\n        is constant for all :math:`i`, and :math:`C` is a :math:`K times R`\n        factor matrix. To compute this decomposition, we reformulate the\n        expression for :math:`B_i` such that\n\n        .. math::\n\n            B_i = P_i B,\n\n        where :math:`P_i` is a :math:`J_i times R` orthogonal matrix and :math:`B` is a\n        :math:`R times R` matrix.\n\n        An alternative formulation of the PARAFAC2 decomposition is that the\n        tensor element :math:`X_{ijk}` is given by\n\n        .. math::\n\n            X_{ijk}\xa0= sum_{r=1}^R A_{ir} B_{ijr} C_{kr},\n\n        with the same constraints hold for :math:`B_i` as above.\n\n        Parameters\n        ----------\n        parafac2_tensor : Parafac2Tensor - (weight, factors, projection_matrices)\n            weights\n                weights of the factors\n            factors\n                Contains the matrices :math:`A`, :math:`B` and :math:`C` described above\n            projection_matrices\n                factors\n\n        Returns\n        -------\n            Full constructed tensor. Uneven slices are padded with zeros.\n        '
        return ivy.unfold(ivy.Parafac2Tensor.parafac2_to_tensor(parafac2_tensor), mode)

    def parafac2_to_vec(parafac2_tensor):
        if False:
            print('Hello World!')
        '\n        Construct a vectorized tensor from a PARAFAC2 decomposition. Uneven slices are\n        padded by zeros.\n\n        The decomposition is on the form :math:`(A [B_i] C)` such that\n        the i-th frontal slice, :math:`X_i`, of :math:`X` is given by\n\n        .. math::\n\n            X_i = B_i diag(a_i) C^T,\n\n        where :math:`diag(a_i)` is the diagonal matrix whose nonzero\n        entries are  equal to the :math:`i`-th row of the :math:`I\n        times R` factor matrix :math:`A`, :math:`B_i` is a :math:`J_i\n        times R` factor matrix such that the cross product matrix :math:\n        `B_{i_1}^T B_{i_1}`is constant for all :math:`i`, and :math:`C`\n        is a :math:`K times R` factor matrix. To compute this\n        decomposition, we reformulate the expression for :math:`B_i`\n        such that\n\n        .. math::\n\n            B_i = P_i B,\n\n        where :math:`P_i` is a :math:`J_i times R` orthogonal matrix and :math:`B` is a\n        :math:`R times R` matrix.\n\n        An alternative formulation of the PARAFAC2 decomposition is that\n        the tensor element :math:`X_{ijk}` is given by\n\n        .. math::\n\n            X_{ijk}\xa0= sum_{r=1}^R A_{ir} B_{ijr} C_{kr},\n\n        with the same constraints hold for :math:`B_i` as above.\n\n        Parameters\n        ----------\n        parafac2_tensor : Parafac2Tensor - (weight, factors, projection_matrices)\n            * weights\n            1D array of shape (rank, ) weights of the factors\n            * factors\n            List of factors of the PARAFAC2 decomposition Contains the matrices\n            :math:`A, :math:`B` and :math:`C` described above\n            * projection_matrices\n                List of projection matrices used to create evolving factors.\n\n        Returns\n        -------\n            Full constructed tensor. Uneven slices are padded with zeros.6\n        '
        return ivy.reshape(ivy.Parafac2Tensor.parafac2_to_tensor(parafac2_tensor), -1)