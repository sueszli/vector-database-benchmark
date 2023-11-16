from .base import FactorizedTensor
import ivy

class CPTensor(FactorizedTensor):

    def __init__(self, cp_tensor):
        if False:
            print('Hello World!')
        super().__init__()
        (shape, rank) = ivy.CPTensor.validate_cp_tensor(cp_tensor)
        (weights, factors) = cp_tensor
        if weights is None:
            weights = ivy.ones(rank, dtype=factors[0].dtype)
        self.shape = shape
        self.rank = rank
        self.factors = factors
        self.weights = weights

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        if index == 0:
            return self.weights
        elif index == 1:
            return self.factors
        else:
            raise IndexError(f'You tried to access index {index} of a CP tensor.\nYou can only access index 0 and 1 of a CP tensor(corresponding respectively to the weights and factors)')

    def __setitem__(self, index, value):
        if False:
            print('Hello World!')
        if index == 0:
            self.weights = value
        elif index == 1:
            self.factors = value
        else:
            raise IndexError(f'You tried to set the value at index {index} of a CP tensor.\nYou can only set index 0 and 1 of a CP tensor(corresponding respectively to the weights and factors)')

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        yield self.weights
        yield self.factors

    def __len__(self):
        if False:
            return 10
        return 2

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        message = f'(weights, factors) : rank-{self.rank} CPTensor of shape {self.shape}'
        return message

    def to_tensor(self):
        if False:
            i = 10
            return i + 15
        return ivy.CPTensor.cp_to_tensor(self)

    def to_vec(self):
        if False:
            while True:
                i = 10
        return ivy.CPTensor.cp_to_vec(self)

    def to_unfolded(self, mode):
        if False:
            for i in range(10):
                print('nop')
        return ivy.CPTensor.cp_to_unfolded(self, mode)

    def cp_copy(self):
        if False:
            while True:
                i = 10
        return CPTensor((ivy.copy_array(self.weights), [ivy.copy_array(self.factors[i]) for i in range(len(self.factors))]))

    def mode_dot(self, matrix_or_vector, mode, keep_dim=False, copy=True):
        if False:
            i = 10
            return i + 15
        '\n        N-mode product of a CP tensor and a matrix or vector at the specified mode.\n\n        Parameters\n        ----------\n        cp_tensor\n\n        matrix_or_vector\n            1D or 2D array of shape ``(J, i_k)`` or ``(i_k, )``\n            matrix or vectors to which to n-mode multiply the tensor\n        mode\n            int\n\n        Returns\n        -------\n        CPTensor = (core, factors)\n            `mode`-mode product of `tensor` by `matrix_or_vector`\n            * of shape :math:`(i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)`\n              if matrix_or_vector is a matrix\n            * of shape :math:`(i_1, ..., i_{k-1}, i_{k+1}, ..., i_N)`\n              if matrix_or_vector is a vector\n\n        See Also\n        --------\n        cp_mode_dot : chaining several mode_dot in one call\n        '
        return ivy.CPTensor.cp_mode_dot(self, matrix_or_vector, mode, keep_dim=keep_dim, copy=copy)

    def norm(self):
        if False:
            return 10
        '\n        Return the l2 norm of a CP tensor.\n\n        Parameters\n        ----------\n        cp_tensor\n            ivy.CPTensor or (core, factors)\n\n        Returns\n        -------\n        l2-norm\n            int\n\n        Notes\n        -----\n        This is ||cp_to_tensor(factors)||^2\n\n        You can see this using the fact that\n        khatria-rao(A, B)^T x khatri-rao(A, B) = A^T x A  * B^T x B\n        '
        return ivy.CPTensor.cp_norm(self)

    def normalize(self, inplace=True):
        if False:
            print('Hello World!')
        '\n        Normalize the factors to unit length.\n\n        Turns ``factors = [|U_1, ... U_n|]`` into ``[weights; |V_1, ... V_n|]``,\n        where the columns of each `V_k` are normalized to unit Euclidean length\n        from the columns of `U_k` with the normalizing constants absorbed into\n        `weights`. In the special case of a symmetric tensor, `weights` holds the\n        eigenvalues of the tensor.\n\n        Parameters\n        ----------\n        cp_tensor\n            CPTensor = (weight, factors)\n            factors is list of matrices, all with the same number of columns\n            i.e.::\n                for u in U:\n                    u[i].shape == (s_i, R)\n\n            where `R` is fixed while `s_i` can vary with `i`\n\n        inplace\n            if False, returns a normalized Copy\n            otherwise the tensor modifies itself and returns itself\n\n        Returns\n        -------\n        CPTensor = (normalisation_weights, normalised_factors)\n            returns itself if inplace is True, a normalized copy otherwise\n        '
        (weights, factors) = ivy.CPTensor.cp_normalize(self)
        if inplace:
            (self.weights, self.factors) = (weights, factors)
            return self
        return ivy.CPTensor((weights, factors))

    @property
    def n_param(self):
        if False:
            print('Hello World!')
        factors_params = self.rank * ivy.sum(self.shape)
        if self.weights:
            return factors_params + self.rank
        else:
            return factors_params

    @staticmethod
    def validate_cp_tensor(cp_tensor):
        if False:
            i = 10
            return i + 15
        '\n        Validate a cp_tensor in the form (weights, factors)\n\n            Return the rank and shape of the validated tensor\n\n        Parameters\n        ----------\n        cp_tensor\n            CPTensor or (weights, factors)\n\n        Returns\n        -------\n        (shape, rank)\n            size of the full tensor and rank of the CP tensor\n        '
        if isinstance(cp_tensor, CPTensor):
            return (cp_tensor.shape, cp_tensor.rank)
        elif isinstance(cp_tensor, (float, int)):
            return (0, 0)
        (weights, factors) = cp_tensor
        ndim = len(factors[0].shape)
        if ndim == 2:
            rank = int(ivy.shape(factors[0])[1])
        elif ndim == 1:
            rank = 1
        else:
            raise ValueError('Got a factor with 3 dimensions but CP factors should be at most 2D, of shape (size, rank).')
        shape = []
        for (i, factor) in enumerate(factors):
            s = ivy.shape(factor)
            if len(s) == 2:
                (current_mode_size, current_rank) = s
            else:
                (current_mode_size, current_rank) = (*s, 1)
            if current_rank != rank:
                raise ValueError(f'All the factors of a CP tensor should have the same number of column.However, factors[0].shape[1]={rank} but factors[{i}].shape[1]={ivy.shape(factor)[1]}.')
            shape.append(current_mode_size)
        if weights is not None and len(weights) != rank:
            raise ValueError(f'Given factors for a rank-{rank} CP tensor but len(weights)={ivy.shape(weights)}.')
        return (tuple(shape), rank)

    @staticmethod
    def cp_n_param(tensor_shape, rank, weights=False):
        if False:
            while True:
                i = 10
        '\n        Return number of parameters of a CP decomposition for a given `rank` and full\n        `tensor_shape`.\n\n        Parameters\n        ----------\n        tensor_shape\n            shape of the full tensor to decompose (or approximate)\n        rank\n            rank of the CP decomposition\n\n        Returns\n        -------\n        n_params\n            Number of parameters of a CP decomposition of rank `rank`\n              of a full tensor of shape `tensor_shape`\n        '
        factors_params = rank * ivy.sum(tensor_shape)
        if weights:
            return factors_params + rank
        else:
            return factors_params

    @staticmethod
    def validate_cp_rank(tensor_shape, rank='same', rounding='round'):
        if False:
            while True:
                i = 10
        "\n        Return the rank of a CP Decomposition.\n\n        Parameters\n        ----------\n        tensor_shape\n            shape of the tensor to decompose\n        rank\n            way to determine the rank, by default 'same'\n            if 'same': rank is computed to keep the number\n            of parameters (at most) the same\n            if float, computes a rank so as to keep rank\n            percent of the original number of parameters\n            if int, just returns rank\n        rounding\n            {'round', 'floor', 'ceil'}\n\n        Returns\n        -------\n        rank\n            rank of the decomposition\n        "
        if rounding == 'ceil':
            rounding_fun = ivy.ceil
        elif rounding == 'floor':
            rounding_fun = ivy.floor
        elif rounding == 'round':
            rounding_fun = ivy.round
        else:
            raise ValueError(f'Rounding should be of round, floor or ceil, but got {rounding}')
        if rank == 'same':
            rank = float(1)
        if isinstance(rank, float):
            rank = int(rounding_fun(ivy.prod(tensor_shape) * rank / ivy.sum(tensor_shape)))
        return rank

    @staticmethod
    def cp_normalize(cp_tensor):
        if False:
            return 10
        '\n        Return cp_tensor with factors normalised to unit length.\n\n        Turns ``factors = [|U_1, ... U_n|]`` into ``[weights;\n        |V_1, ... V_n|]``, where the columns of each `V_k` are\n        normalized to unit Euclidean length from the columns of\n        `U_k` with the normalizing constants absorbed into\n        `weights`. In the special case of a symmetric tensor,\n        `weights` holds the eigenvalues of the tensor.\n\n        Parameters\n        ----------\n        cp_tensor\n            factors is list of matrices,\n              all with the same number of columns\n            i.e.::\n\n                for u in U:\n                    u[i].shape == (s_i, R)\n\n            where `R` is fixed while `s_i` can vary with `i`\n\n        Returns\n        -------\n        CPTensor = (normalisation_weights, normalised_factors)\n        '
        (_, rank) = ivy.CPTensor.validate_cp_tensor(cp_tensor)
        (weights, factors) = cp_tensor
        if weights is None:
            weights = ivy.ones(rank, dtype=factors[0].dtype)
        normalized_factors = []
        for (i, factor) in enumerate(factors):
            if i == 0:
                factor = factor * weights
                weights = ivy.ones((rank,), dtype=factor.dtype)
            scales = ivy.sqrt(ivy.sum(ivy.square(factor), axis=0))
            scales_non_zero = ivy.where(scales == 0, ivy.ones(ivy.shape(scales), dtype=factor.dtype), scales)
            weights = weights * scales
            normalized_factors.append(factor / ivy.reshape(scales_non_zero, (1, -1)))
        return CPTensor((weights, normalized_factors))

    @staticmethod
    def cp_flip_sign(cp_tensor, mode=0, func=None):
        if False:
            i = 10
            return i + 15
        '\n        Return cp_tensor with factors flipped to have positive signs. The sign of a\n        given column is determined by `func`, which is the mean by default. Any negative\n        signs are assigned to the mode indicated by `mode`.\n\n        Parameters\n        ----------\n        cp_tensor\n            CPTensor = (weight, factors)\n            factors is list of matrices, all with the same number of columns\n            i.e.::\n\n                for u in U:\n                    u[i].shape == (s_i, R)\n\n            where `R` is fixed while `s_i` can vary with `i`\n\n        mode\n            mode that should receive negative signs\n\n        func\n            a function that should summarize the sign of a column\n            it must be able to take an axis argument\n\n        Returns\n        -------\n        CPTensor = (normalisation_weights, normalised_factors)\n        '
        ivy.CPTensor.validate_cp_tensor(cp_tensor)
        (weights, factors) = cp_tensor
        if func is None:
            func = ivy.mean
        for jj in range(0, len(factors)):
            if jj == mode:
                continue
            column_signs = ivy.sign(func(factors[jj], axis=0))
            factors[mode] = factors[mode] * column_signs[ivy.newaxis, :]
            factors[jj] = factors[jj] * column_signs[ivy.newaxis, :]
        weight_signs = ivy.sign(weights)
        factors[mode] = factors[mode] * weight_signs[ivy.newaxis, :]
        weights = ivy.abs(weights)
        return CPTensor((weights, factors))

    @staticmethod
    def cp_lstsq_grad(cp_tensor, tensor, return_loss=False, mask=None):
        if False:
            return 10
        '\n        Compute (for a third-order tensor)\n\n        .. math::\n\n            \\nabla 0.5 ||\\\\mathcal{X} - [\\\\mathbf{w}; \\\\mathbf{A}, \\\\mathbf{B}, \\\\mathbf{C}]||^2 # noqa\n\n        where :math:`[\\\\mathbf{w}; \\\\mathbf{A}, \\\\mathbf{B}, \\\\mathbf{C}]`\n        is the CP decomposition with weights\n        :math:`\\\\mathbf{w}` and factor matrices :math:`\\\\mathbf{A}`, :math:`\\\\mathbf{B}` and :math:`\\\\mathbf{C}`. # noqa\n        Note that this does not return the gradient\n        with respect to the weights even if CP is normalized.\n\n        Parameters\n        ----------\n        cp_tensor\n            CPTensor = (weight, factors)\n            factors is a list of factor matrices,\n            all with the same number of columns\n            i.e. for all matrix U in factor_matrices:\n            U has shape ``(s_i, R)``, where R is fixed and s_i varies with i\n\n        mask\n            A mask to be applied to the final tensor. It should be\n            broadcastable to the shape of the final tensor, that is\n            ``(U[1].shape[0], ... U[-1].shape[0])``.\n\n        return_loss\n            Optionally return the scalar loss function along with the gradient.\n\n        Returns\n        -------\n        cp_gradient : CPTensor = (None, factors)\n            factors is a list of factor matrix gradients,\n            all with the same number of columns\n            i.e. for all matrix U in factor_matrices:\n            U has shape ``(s_i, R)``, where R is fixed and s_i varies with i\n\n        loss : float\n            Scalar quantity of the loss function corresponding to cp_gradient. Only returned\n            if return_loss = True.\n        '
        ivy.CPTensor.validate_cp_tensor(cp_tensor)
        (_, factors) = cp_tensor
        diff = tensor - ivy.CPTensor.cp_to_tensor(cp_tensor)
        if mask is not None:
            diff = diff * mask
        grad_fac = [-ivy.CPTensor.unfolding_dot_khatri_rao(diff, cp_tensor, ii) for ii in range(len(factors))]
        if return_loss:
            return (CPTensor((None, grad_fac)), 0.5 * ivy.sum(diff ** 2))
        return CPTensor((None, grad_fac))

    @staticmethod
    def cp_to_tensor(cp_tensor, mask=None):
        if False:
            i = 10
            return i + 15
        '\n        Turn the Khatri-product of matrices into a full tensor.\n\n            ``factor_matrices = [|U_1, ... U_n|]`` becomes\n            a tensor shape ``(U[1].shape[0], U[2].shape[0], ... U[-1].shape[0])``\n\n        Parameters\n        ----------\n        cp_tensor\n            factors is a list of factor matrices,\n            all with the same number of columns\n            i.e. for all matrix U in factor_matrices:\n            U has shape ``(s_i, R)``, where R is fixed and s_i varies with i\n\n        mask\n            mask to be applied to the final tensor. It should be\n            broadcastable to the shape of the final tensor, that is\n            ``(U[1].shape[0], ... U[-1].shape[0])``.\n\n        Returns\n        -------\n        ivy.Array\n            full tensor of shape ``(U[1].shape[0], ... U[-1].shape[0])``\n\n        Notes\n        -----\n        This version works by first computing the mode-0 unfolding of the tensor\n        and then refolding it.\n\n        There are other possible and equivalent alternate implementation, e.g.\n        summing over r and updating an outer product of vectors.\n        '
        (shape, _) = ivy.CPTensor.validate_cp_tensor(cp_tensor)
        if not shape:
            return cp_tensor
        (weights, factors) = cp_tensor
        if len(shape) == 1:
            return ivy.sum(weights * factors[0], axis=1)
        if weights is None:
            weights = 1
        if mask is None:
            full_tensor = ivy.matmul(factors[0] * weights, ivy.permute_dims(ivy.khatri_rao(factors, skip_matrix=0), (1, 0)))
        else:
            full_tensor = ivy.sum(ivy.khatri_rao([factors[0] * weights] + factors[1:], mask=mask), axis=1)
        return ivy.fold(full_tensor, 0, shape)

    @staticmethod
    def cp_to_unfolded(cp_tensor, mode):
        if False:
            while True:
                i = 10
        '\n        Turn the khatri-product of matrices into an unfolded tensor.\n\n            turns ``factors = [|U_1, ... U_n|]`` into a mode-`mode`\n            unfolding of the tensor\n\n        Parameters\n        ----------\n        cp_tensor\n            factors is a list of matrices, all with the same number of columns\n            ie for all u in factor_matrices:\n            u[i] has shape (s_u_i, R), where R is fixed\n        mode\n            mode of the desired unfolding\n\n        Returns\n        -------\n        ivy.Array\n            unfolded tensor of shape (tensor_shape[mode], -1)\n\n        Notes\n        -----\n        Writing factors = [U_1, ..., U_n], we exploit the fact that\n        ``U_k = U[k].dot(khatri_rao(U_1, ..., U_k-1, U_k+1, ..., U_n))``\n        '
        ivy.CPTensor.validate_cp_tensor(cp_tensor)
        (weights, factors) = cp_tensor
        if weights is not None:
            return ivy.dot(factors[mode] * weights, ivy.permute_dims(ivy.khatri_rao(factors, skip_matrix=mode), (1, 0)))
        else:
            return ivy.dot(factors[mode], ivy.permute_dims(ivy.khatri_rao(factors, skip_matrix=mode), (1, 0)))

    @staticmethod
    def cp_to_vec(cp_tensor):
        if False:
            i = 10
            return i + 15
        '\n        Turn the khatri-product of matrices into a vector.\n\n            (the tensor ``factors = [|U_1, ... U_n|]``\n            is converted into a raveled mode-0 unfolding)\n\n        Parameters\n        ----------\n        cp_tensor\n            factors is a list of matrices, all with the same number of columns\n            i.e.::\n\n                for u in U:\n                    u[i].shape == (s_i, R)\n\n            where `R` is fixed while `s_i` can vary with `i`\n\n        Returns\n        -------\n        ivy.Array\n            vectorised tensor\n        '
        return ivy.reshape(ivy.CPTensor.cp_to_tensor(cp_tensor), -1)

    @staticmethod
    def cp_mode_dot(cp_tensor, matrix_or_vector, mode, keep_dim=False, copy=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        N-mode product of a CP tensor and a matrix or vector at the specified mode.\n\n        Parameters\n        ----------\n        cp_tensor\n            ivy.CPTensor or (core, factors)\n\n        matrix_or_vector\n            1D or 2D array of shape ``(J, i_k)`` or ``(i_k, )``\n            matrix or vectors to which to n-mode multiply the tensor\n        mode : int\n\n        Returns\n        -------\n        CPTensor = (core, factors)\n            `mode`-mode product of `tensor` by `matrix_or_vector`\n            * of shape :math:`(i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)`\n              if matrix_or_vector is a matrix\n            * of shape :math:`(i_1, ..., i_{k-1}, i_{k+1}, ..., i_N)`\n              if matrix_or_vector is a vector\n\n        See Also\n        --------\n        cp_multi_mode_dot : chaining several mode_dot in one call\n        '
        (shape, _) = ivy.CPTensor.validate_cp_tensor(cp_tensor)
        (weights, factors) = cp_tensor
        contract = False
        ndims = len(matrix_or_vector.shape)
        if ndims == 2:
            if matrix_or_vector.shape[1] != shape[mode]:
                raise ValueError(f'shapes {shape} and {matrix_or_vector.shape} not aligned in mode-{mode} multiplication: {shape[mode]} (mode {mode}) != {matrix_or_vector.shape[1]} (dim 1 of matrix)')
        elif ndims == 1:
            if matrix_or_vector.shape[0] != shape[mode]:
                raise ValueError(f'shapes {shape} and {matrix_or_vector.shape} not aligned for mode-{mode} multiplication: {shape[mode]} (mode {mode}) != {matrix_or_vector.shape[0]} (vector size)')
            if not keep_dim:
                contract = True
        else:
            raise ValueError('Can only take n_mode_product with a vector or a matrix.')
        if copy:
            factors = [ivy.copy_array(f) for f in factors]
            weights = ivy.copy_array(weights)
        if contract:
            factor = factors.pop(mode)
            factor = ivy.dot(matrix_or_vector, factor)
            mode = max(mode - 1, 0)
            factors[mode] *= factor
        else:
            factors[mode] = ivy.dot(matrix_or_vector, factors[mode])
        if copy:
            return CPTensor((weights, factors))
        else:
            cp_tensor.shape = tuple((f.shape[0] for f in factors))
            return cp_tensor

    @staticmethod
    def cp_norm(cp_tensor):
        if False:
            while True:
                i = 10
        '\n        Return the l2 norm of a CP tensor.\n\n        Parameters\n        ----------\n        cp_tensor\n            ivy.CPTensor or (core, factors)\n\n        Returns\n        -------\n        l2-norm\n\n        Notes\n        -----\n        This is ||cp_to_tensor(factors)||^2\n\n        You can see this using the fact that\n        khatria-rao(A, B)^T x khatri-rao(A, B) = A^T x A  * B^T x B\n        '
        _ = ivy.CPTensor.validate_cp_tensor(cp_tensor)
        (weights, factors) = cp_tensor
        norm = ivy.ones((factors[0].shape[1], factors[0].shape[1]), dtype=factors[0].dtype)
        for f in factors:
            norm = norm * ivy.dot(ivy.permute_dims(f, (1, 0)), ivy.conj(f))
        if weights is not None:
            norm = norm * (ivy.reshape(weights, (-1, 1)) * ivy.reshape(weights, (1, -1)))
        return ivy.sqrt(ivy.sum(norm))

    @staticmethod
    def unfolding_dot_khatri_rao(x, cp_tensor, mode):
        if False:
            while True:
                i = 10
        '\n        Mode-n unfolding times khatri-rao product of factors.\n\n        Parameters\n        ----------\n        x\n            tensor to unfold\n        factors\n            list of matrices of which to the khatri-rao product\n        mode\n            mode on which to unfold `tensor`\n\n        Returns\n        -------\n        mttkrp\n            dot(unfold(x, mode), khatri-rao(factors))\n        '
        mttkrp_parts = []
        (weights, factors) = cp_tensor
        rank = ivy.shape(factors[0])[1]
        for r in range(rank):
            component = ivy.multi_mode_dot(x, [ivy.conj(f[:, r]) for f in factors], skip=mode)
            mttkrp_parts.append(component)
        if weights is None:
            return ivy.stack(mttkrp_parts, axis=1)
        else:
            return ivy.stack(mttkrp_parts, axis=1) * ivy.reshape(weights, (1, -1))