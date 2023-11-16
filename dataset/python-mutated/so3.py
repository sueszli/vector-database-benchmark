from __future__ import annotations
from typing import Optional
from kornia.core import Device, Dtype, Module, Tensor, concatenate, eye, stack, tensor, where, zeros, zeros_like
from kornia.core.check import KORNIA_CHECK_TYPE
from kornia.geometry.conversions import vector_to_skew_symmetric_matrix
from kornia.geometry.linalg import batched_dot_product
from kornia.geometry.quaternion import Quaternion
from kornia.geometry.vector import Vector3

class So3(Module):
    """Base class to represent the So3 group.

    The SO(3) is the group of all rotations about the origin of three-dimensional Euclidean space
    :math:`R^3` under the operation of composition.
    See more: https://en.wikipedia.org/wiki/3D_rotation_group

    We internally represent the rotation by a unit quaternion.

    Example:
        >>> q = Quaternion.identity()
        >>> s = So3(q)
        >>> s.q
        Parameter containing:
        tensor([1., 0., 0., 0.], requires_grad=True)
    """

    def __init__(self, q: Quaternion) -> None:
        if False:
            i = 10
            return i + 15
        'Constructor for the base class.\n\n        Internally represented by a unit quaternion `q`.\n\n        Args:\n            data: Quaternion with the shape of :math:`(B, 4)`.\n\n        Example:\n            >>> data = torch.ones((2, 4))\n            >>> q = Quaternion(data)\n            >>> So3(q)\n            Parameter containing:\n            tensor([[1., 1., 1., 1.],\n                    [1., 1., 1., 1.]], requires_grad=True)\n        '
        super().__init__()
        KORNIA_CHECK_TYPE(q, Quaternion)
        self._q = q

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'{self.q}'

    def __getitem__(self, idx: int | slice) -> So3:
        if False:
            return 10
        return So3(self._q[idx])

    def __mul__(self, right: So3) -> So3:
        if False:
            i = 10
            return i + 15
        'Compose two So3 transformations.\n\n        Args:\n            right: the other So3 transformation.\n\n        Return:\n            The resulting So3 transformation.\n        '
        if isinstance(right, So3):
            return So3(self.q * right.q)
        elif isinstance(right, (Tensor, Vector3)):
            w = zeros(*right.shape[:-1], 1, device=right.device, dtype=right.dtype)
            quat = Quaternion(concatenate((w, right.data), -1))
            out = (self.q * quat * self.q.conj()).vec
            if isinstance(right, Tensor):
                return out
            elif isinstance(right, Vector3):
                return Vector3(out)
        else:
            raise TypeError(f'Not So3 or Tensor type. Got: {type(right)}')

    @property
    def q(self) -> Quaternion:
        if False:
            for i in range(10):
                print('nop')
        'Return the underlying data with shape :math:`(B,4)`.'
        return self._q

    @staticmethod
    def exp(v: Tensor) -> So3:
        if False:
            for i in range(10):
                print('nop')
        'Converts elements of lie algebra to elements of lie group.\n\n        See more: https://vision.in.tum.de/_media/members/demmeln/nurlanov2021so3log.pdf\n\n        Args:\n            v: vector of shape :math:`(B,3)`.\n\n        Example:\n            >>> v = torch.zeros((2, 3))\n            >>> s = So3.exp(v)\n            >>> s\n            Parameter containing:\n            tensor([[1., 0., 0., 0.],\n                    [1., 0., 0., 0.]], requires_grad=True)\n        '
        theta = batched_dot_product(v, v).sqrt()[..., None]
        theta_nonzeros = theta != 0.0
        theta_half = 0.5 * theta
        w = where(theta_nonzeros, theta_half.cos(), tensor(1.0, device=v.device, dtype=v.dtype))
        b = where(theta_nonzeros, theta_half.sin() / theta, tensor(0.0, device=v.device, dtype=v.dtype))
        xyz = b * v
        return So3(Quaternion(concatenate((w, xyz), -1)))

    def log(self) -> Tensor:
        if False:
            return 10
        'Converts elements of lie group  to elements of lie algebra.\n\n        Example:\n            >>> data = torch.ones((2, 4))\n            >>> q = Quaternion(data)\n            >>> So3(q).log()\n            tensor([[0., 0., 0.],\n                    [0., 0., 0.]], grad_fn=<WhereBackward0>)\n        '
        theta = batched_dot_product(self.q.vec, self.q.vec).sqrt()
        omega = where(theta[..., None] != 0, 2 * self.q.real[..., None].acos() * self.q.vec / theta[..., None], 2 * self.q.vec / self.q.real[..., None])
        return omega

    @staticmethod
    def hat(v: Vector3 | Tensor) -> Tensor:
        if False:
            return 10
        'Converts elements from vector space to lie algebra. Returns matrix of shape :math:`(B,3,3)`.\n\n        Args:\n            v: Vector3 or tensor of shape :math:`(B,3)`.\n\n        Example:\n            >>> v = torch.ones((1,3))\n            >>> m = So3.hat(v)\n            >>> m\n            tensor([[[ 0., -1.,  1.],\n                     [ 1.,  0., -1.],\n                     [-1.,  1.,  0.]]])\n        '
        if isinstance(v, Tensor):
            (a, b, c) = (v[..., 0], v[..., 1], v[..., 2])
        else:
            (a, b, c) = (v.x, v.y, v.z)
        z = zeros_like(a)
        row0 = stack((z, -c, b), -1)
        row1 = stack((c, z, -a), -1)
        row2 = stack((-b, a, z), -1)
        return stack((row0, row1, row2), -2)

    @staticmethod
    def vee(omega: Tensor) -> Tensor:
        if False:
            return 10
        'Converts elements from lie algebra to vector space. Returns vector of shape :math:`(B,3)`.\n\n        .. math::\n            omega = \\begin{bmatrix} 0 & -c & b \\\\\n            c & 0 & -a \\\\\n            -b & a & 0\\end{bmatrix}\n\n        Args:\n            omega: 3x3-matrix representing lie algebra.\n\n        Example:\n            >>> v = torch.ones((1,3))\n            >>> omega = So3.hat(v)\n            >>> So3.vee(omega)\n            tensor([[1., 1., 1.]])\n        '
        (a, b, c) = (omega[..., 2, 1], omega[..., 0, 2], omega[..., 1, 0])
        return stack((a, b, c), -1)

    def matrix(self) -> Tensor:
        if False:
            print('Hello World!')
        'Convert the quaternion to a rotation matrix of shape :math:`(B,3,3)`.\n\n        The matrix is of the form:\n\n        .. math::\n            \\begin{bmatrix} 1-2y^2-2z^2 & 2xy-2zw & 2xy+2yw \\\\\n            2xy+2zw & 1-2x^2-2z^2 & 2yz-2xw \\\\\n            2xz-2yw & 2yz+2xw & 1-2x^2-2y^2\\end{bmatrix}\n\n        Example:\n            >>> s = So3.identity()\n            >>> m = s.matrix()\n            >>> m\n            tensor([[1., 0., 0.],\n                    [0., 1., 0.],\n                    [0., 0., 1.]], grad_fn=<StackBackward0>)\n        '
        w = self.q.w[..., None]
        (x, y, z) = (self.q.x[..., None], self.q.y[..., None], self.q.z[..., None])
        q0 = 1 - 2 * y ** 2 - 2 * z ** 2
        q1 = 2 * x * y - 2 * z * w
        q2 = 2 * x * z + 2 * y * w
        row0 = concatenate((q0, q1, q2), -1)
        q0 = 2 * x * y + 2 * z * w
        q1 = 1 - 2 * x ** 2 - 2 * z ** 2
        q2 = 2 * y * z - 2 * x * w
        row1 = concatenate((q0, q1, q2), -1)
        q0 = 2 * x * z - 2 * y * w
        q1 = 2 * y * z + 2 * x * w
        q2 = 1 - 2 * x ** 2 - 2 * y ** 2
        row2 = concatenate((q0, q1, q2), -1)
        return stack((row0, row1, row2), -2)

    @classmethod
    def from_matrix(cls, matrix: Tensor) -> So3:
        if False:
            print('Hello World!')
        'Create So3 from a rotation matrix.\n\n        Args:\n            matrix: the rotation matrix to convert of shape :math:`(B,3,3)`.\n\n        Example:\n            >>> m = torch.eye(3)\n            >>> s = So3.from_matrix(m)\n            >>> s\n            Parameter containing:\n            tensor([1., 0., 0., 0.], requires_grad=True)\n        '
        return cls(Quaternion.from_matrix(matrix))

    @classmethod
    def from_wxyz(cls, wxyz: Tensor) -> So3:
        if False:
            for i in range(10):
                print('nop')
        'Create So3 from a tensor representing a quaternion.\n\n        Args:\n            wxyz: the quaternion to convert of shape :math:`(B,4)`.\n\n        Example:\n            >>> q = torch.tensor([1., 0., 0., 0.])\n            >>> s = So3.from_wxyz(q)\n            >>> s\n            Parameter containing:\n            tensor([1., 0., 0., 0.], requires_grad=True)\n        '
        return cls(Quaternion(wxyz))

    @classmethod
    def identity(cls, batch_size: Optional[int]=None, device: Optional[Device]=None, dtype: Optional[Dtype]=None) -> So3:
        if False:
            for i in range(10):
                print('nop')
        'Create a So3 group representing an identity rotation.\n\n        Args:\n            batch_size: the batch size of the underlying data.\n\n        Example:\n            >>> s = So3.identity()\n            >>> s\n            Parameter containing:\n            tensor([1., 0., 0., 0.], requires_grad=True)\n\n            >>> s = So3.identity(batch_size=2)\n            >>> s\n            Parameter containing:\n            tensor([[1., 0., 0., 0.],\n                    [1., 0., 0., 0.]], requires_grad=True)\n        '
        return cls(Quaternion.identity(batch_size, device, dtype))

    def inverse(self) -> So3:
        if False:
            for i in range(10):
                print('nop')
        'Returns the inverse transformation.\n\n        Example:\n            >>> s = So3.identity()\n            >>> s.inverse()\n            Parameter containing:\n            tensor([1., -0., -0., -0.], requires_grad=True)\n        '
        return So3(self.q.conj())

    @classmethod
    def random(cls, batch_size: Optional[int]=None, device: Optional[Device]=None, dtype: Optional[Dtype]=None) -> So3:
        if False:
            return 10
        'Create a So3 group representing a random rotation.\n\n        Args:\n            batch_size: the batch size of the underlying data.\n\n        Example:\n            >>> s = So3.random()\n            >>> s = So3.random(batch_size=3)\n        '
        return cls(Quaternion.random(batch_size, device, dtype))

    @classmethod
    def rot_x(cls, x: Tensor) -> So3:
        if False:
            return 10
        'Construct a x-axis rotation.\n\n        Args:\n            x: the x-axis rotation angle.\n        '
        zs = zeros_like(x)
        return cls.exp(stack((x, zs, zs), -1))

    @classmethod
    def rot_y(cls, y: Tensor) -> So3:
        if False:
            print('Hello World!')
        'Construct a z-axis rotation.\n\n        Args:\n            y: the y-axis rotation angle.\n        '
        zs = zeros_like(y)
        return cls.exp(stack((zs, y, zs), -1))

    @classmethod
    def rot_z(cls, z: Tensor) -> So3:
        if False:
            i = 10
            return i + 15
        'Construct a z-axis rotation.\n\n        Args:\n            z: the z-axis rotation angle.\n        '
        zs = zeros_like(z)
        return cls.exp(stack((zs, zs, z), -1))

    def adjoint(self) -> Tensor:
        if False:
            print('Hello World!')
        'Returns the adjoint matrix of shape :math:`(B, 3, 3)`.\n\n        Example:\n            >>> s = So3.identity()\n            >>> s.adjoint()\n            tensor([[1., 0., 0.],\n                    [0., 1., 0.],\n                    [0., 0., 1.]], grad_fn=<StackBackward0>)\n        '
        return self.matrix()

    @staticmethod
    def right_jacobian(vec: Tensor) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        'Computes the right Jacobian of So3.\n\n        Args:\n            vec: the input point of shape :math:`(B, 3)`.\n\n        Example:\n            >>> vec = torch.tensor([1., 2., 3.])\n            >>> So3.right_jacobian(vec)\n            tensor([[-0.0687,  0.5556, -0.0141],\n                    [-0.2267,  0.1779,  0.6236],\n                    [ 0.5074,  0.3629,  0.5890]])\n        '
        R_skew = vector_to_skew_symmetric_matrix(vec)
        theta = vec.norm(dim=-1, keepdim=True)[..., None]
        I = eye(3, device=vec.device, dtype=vec.dtype)
        Jr = I - (1 - theta.cos()) / theta ** 2 * R_skew + (theta - theta.sin()) / theta ** 3 * (R_skew @ R_skew)
        return Jr

    @staticmethod
    def Jr(vec: Tensor) -> Tensor:
        if False:
            while True:
                i = 10
        'Alias for right jacobian.\n\n        Args:\n            vec: the input point of shape :math:`(B, 3)`.\n        '
        return So3.right_jacobian(vec)

    @staticmethod
    def left_jacobian(vec: Tensor) -> Tensor:
        if False:
            return 10
        'Computes the left Jacobian of So3.\n\n        Args:\n            vec: the input point of shape :math:`(B, 3)`.\n\n        Example:\n            >>> vec = torch.tensor([1., 2., 3.])\n            >>> So3.left_jacobian(vec)\n            tensor([[-0.0687, -0.2267,  0.5074],\n                    [ 0.5556,  0.1779,  0.3629],\n                    [-0.0141,  0.6236,  0.5890]])\n        '
        R_skew = vector_to_skew_symmetric_matrix(vec)
        theta = vec.norm(dim=-1, keepdim=True)[..., None]
        I = eye(3, device=vec.device, dtype=vec.dtype)
        Jl = I + (1 - theta.cos()) / theta ** 2 * R_skew + (theta - theta.sin()) / theta ** 3 * (R_skew @ R_skew)
        return Jl

    @staticmethod
    def Jl(vec: Tensor) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        'Alias for left jacobian.\n\n        Args:\n            vec: the input point of shape :math:`(B, 3)`.\n        '
        return So3.left_jacobian(vec)