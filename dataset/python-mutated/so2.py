from __future__ import annotations
from typing import Optional, overload
from kornia.core import Device, Dtype, Module, Parameter, Tensor, complex, rand, stack, tensor, zeros_like
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR
from kornia.geometry.liegroup._utils import check_so2_matrix, check_so2_matrix_shape, check_so2_t_shape, check_so2_theta_shape, check_so2_z_shape
from kornia.geometry.vector import Vector2

class So2(Module):
    """Base class to represent the So2 group.

    The SO(2) is the group of all rotations about the origin of two-dimensional Euclidean space
    :math:`R^2` under the operation of composition.
    See more: https://en.wikipedia.org/wiki/Orthogonal_group#Special_orthogonal_group

    We internally represent the rotation by a complex number.

    Example:
        >>> real = torch.tensor([1.0])
        >>> imag = torch.tensor([2.0])
        >>> So2(torch.complex(real, imag))
        Parameter containing:
        tensor([1.+2.j], requires_grad=True)
    """

    def __init__(self, z: Tensor) -> None:
        if False:
            i = 10
            return i + 15
        'Constructor for the base class.\n\n        Internally represented by complex number `z`.\n\n        Args:\n            z: Complex number with the shape of :math:`(B, 1)` or :math:`(B)`.\n\n        Example:\n            >>> real = torch.tensor(1.0)\n            >>> imag = torch.tensor(2.0)\n            >>> So2(torch.complex(real, imag)).z\n            Parameter containing:\n            tensor(1.+2.j, requires_grad=True)\n        '
        super().__init__()
        KORNIA_CHECK_IS_TENSOR(z)
        check_so2_z_shape(z)
        self._z = Parameter(z)

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'{self.z}'

    def __getitem__(self, idx: int | slice) -> So2:
        if False:
            for i in range(10):
                print('nop')
        return So2(self._z[idx])

    @overload
    def __mul__(self, right: So2) -> So2:
        if False:
            return 10
        ...

    @overload
    def __mul__(self, right: Tensor) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        ...

    def __mul__(self, right: So2 | Tensor) -> So2 | Tensor:
        if False:
            while True:
                i = 10
        'Performs a left-multiplication either rotation concatenation or point-transform.\n\n        Args:\n            right: the other So2 transformation.\n\n        Return:\n            The resulting So2 transformation.\n        '
        z = self.z
        if isinstance(right, So2):
            return So2(z * right.z)
        elif isinstance(right, (Vector2, Tensor)):
            if isinstance(right, Tensor):
                check_so2_t_shape(right)
            x = right.data[..., 0]
            y = right.data[..., 1]
            real = z.real
            imag = z.imag
            out = stack((real * x - imag * y, imag * x + real * y), -1)
            if isinstance(right, Tensor):
                return out
            else:
                return Vector2(out)
        else:
            raise TypeError(f'Not So2 or Tensor type. Got: {type(right)}')

    @property
    def z(self) -> Tensor:
        if False:
            print('Hello World!')
        'Return the underlying data with shape :math:`(B, 1)`.'
        return self._z

    @staticmethod
    def exp(theta: Tensor) -> So2:
        if False:
            i = 10
            return i + 15
        'Converts elements of lie algebra to elements of lie group.\n\n        Args:\n            theta: angle in radians of shape :math:`(B, 1)` or :math:`(B)`.\n\n        Example:\n            >>> v = torch.tensor([3.1415/2])\n            >>> s = So2.exp(v)\n            >>> s\n            Parameter containing:\n            tensor([4.6329e-05+1.j], requires_grad=True)\n        '
        check_so2_theta_shape(theta)
        return So2(complex(theta.cos(), theta.sin()))

    def log(self) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        'Converts elements of lie group to elements of lie algebra.\n\n        Example:\n            >>> real = torch.tensor([1.0])\n            >>> imag = torch.tensor([3.0])\n            >>> So2(torch.complex(real, imag)).log()\n            tensor([1.2490], grad_fn=<Atan2Backward0>)\n        '
        return self.z.imag.atan2(self.z.real)

    @staticmethod
    def hat(theta: Tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        'Converts elements from vector space to lie algebra. Returns matrix of shape :math:`(B, 2, 2)`.\n\n        Args:\n            theta: angle in radians of shape :math:`(B)`.\n\n        Example:\n            >>> theta = torch.tensor(3.1415/2)\n            >>> So2.hat(theta)\n            tensor([[0.0000, 1.5707],\n                    [1.5707, 0.0000]])\n        '
        check_so2_theta_shape(theta)
        z = zeros_like(theta)
        row0 = stack((z, theta), -1)
        row1 = stack((theta, z), -1)
        return stack((row0, row1), -1)

    @staticmethod
    def vee(omega: Tensor) -> Tensor:
        if False:
            print('Hello World!')
        'Converts elements from lie algebra to vector space. Returns vector of shape :math:`(B,)`.\n\n        Args:\n            omega: 2x2-matrix representing lie algebra.\n\n        Example:\n            >>> v = torch.ones(3)\n            >>> omega = So2.hat(v)\n            >>> So2.vee(omega)\n            tensor([1., 1., 1.])\n        '
        check_so2_matrix_shape(omega)
        return omega[..., 0, 1]

    def matrix(self) -> Tensor:
        if False:
            print('Hello World!')
        'Convert the complex number to a rotation matrix of shape :math:`(B, 2, 2)`.\n\n        Example:\n            >>> s = So2.identity()\n            >>> m = s.matrix()\n            >>> m\n            tensor([[1., -0.],\n                    [0., 1.]], grad_fn=<StackBackward0>)\n        '
        row0 = stack((self.z.real, -self.z.imag), -1)
        row1 = stack((self.z.imag, self.z.real), -1)
        return stack((row0, row1), -2)

    @classmethod
    def from_matrix(cls, matrix: Tensor) -> So2:
        if False:
            return 10
        'Create So2 from a rotation matrix.\n\n        Args:\n            matrix: the rotation matrix to convert of shape :math:`(B, 2, 2)`.\n\n        Example:\n            >>> m = torch.eye(2)\n            >>> s = So2.from_matrix(m)\n            >>> s.z\n            Parameter containing:\n            tensor(1.+0.j, requires_grad=True)\n        '
        check_so2_matrix_shape(matrix)
        check_so2_matrix(matrix)
        z = complex(matrix[..., 0, 0], matrix[..., 1, 0])
        return cls(z)

    @classmethod
    def identity(cls, batch_size: Optional[int]=None, device: Optional[Device]=None, dtype: Optional[Dtype]=None) -> So2:
        if False:
            i = 10
            return i + 15
        'Create a So2 group representing an identity rotation.\n\n        Args:\n            batch_size: the batch size of the underlying data.\n\n        Example:\n            >>> s = So2.identity(batch_size=2)\n            >>> s\n            Parameter containing:\n            tensor([1.+0.j, 1.+0.j], requires_grad=True)\n        '
        real_data = tensor(1.0, device=device, dtype=dtype)
        imag_data = tensor(0.0, device=device, dtype=dtype)
        if batch_size is not None:
            KORNIA_CHECK(batch_size >= 1, msg='batch_size must be positive')
            real_data = real_data.repeat(batch_size)
            imag_data = imag_data.repeat(batch_size)
        return cls(complex(real_data, imag_data))

    def inverse(self) -> So2:
        if False:
            print('Hello World!')
        'Returns the inverse transformation.\n\n        Example:\n            >>> s = So2.identity()\n            >>> s.inverse().z\n            Parameter containing:\n            tensor(1.+0.j, requires_grad=True)\n        '
        return So2(1 / self.z)

    @classmethod
    def random(cls, batch_size: Optional[int]=None, device: Optional[Device]=None, dtype: Optional[Dtype]=None) -> So2:
        if False:
            i = 10
            return i + 15
        'Create a So2 group representing a random rotation.\n\n        Args:\n            batch_size: the batch size of the underlying data.\n\n        Example:\n            >>> s = So2.random()\n            >>> s = So2.random(batch_size=3)\n        '
        if batch_size is not None:
            KORNIA_CHECK(batch_size >= 1, msg='batch_size must be positive')
            real_data = rand((batch_size,), device=device, dtype=dtype)
            imag_data = rand((batch_size,), device=device, dtype=dtype)
        else:
            real_data = rand((), device=device, dtype=dtype)
            imag_data = rand((), device=device, dtype=dtype)
        return cls(complex(real_data, imag_data))

    def adjoint(self) -> Tensor:
        if False:
            while True:
                i = 10
        'Returns the adjoint matrix of shape :math:`(B, 2, 2)`.\n\n        Example:\n            >>> s = So2.identity()\n            >>> s.adjoint()\n            tensor([[1., -0.],\n                    [0., 1.]], grad_fn=<StackBackward0>)\n        '
        batch_size = len(self.z) if len(self.z.shape) > 0 else None
        return self.identity(batch_size, self.z.device, self.z.real.dtype).matrix()