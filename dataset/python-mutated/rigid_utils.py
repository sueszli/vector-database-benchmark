from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

def rot_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if False:
        return 10
    '\n    Performs matrix multiplication of two rotation matrix tensors. Written out by hand to avoid AMP downcasting.\n\n    Args:\n        a: [*, 3, 3] left multiplicand\n        b: [*, 3, 3] right multiplicand\n    Returns:\n        The product ab\n    '

    def row_mul(i: int) -> torch.Tensor:
        if False:
            print('Hello World!')
        return torch.stack([a[..., i, 0] * b[..., 0, 0] + a[..., i, 1] * b[..., 1, 0] + a[..., i, 2] * b[..., 2, 0], a[..., i, 0] * b[..., 0, 1] + a[..., i, 1] * b[..., 1, 1] + a[..., i, 2] * b[..., 2, 1], a[..., i, 0] * b[..., 0, 2] + a[..., i, 1] * b[..., 1, 2] + a[..., i, 2] * b[..., 2, 2]], dim=-1)
    return torch.stack([row_mul(0), row_mul(1), row_mul(2)], dim=-2)

def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    '\n    Applies a rotation to a vector. Written out by hand to avoid transfer to avoid AMP downcasting.\n\n    Args:\n        r: [*, 3, 3] rotation matrices\n        t: [*, 3] coordinate tensors\n    Returns:\n        [*, 3] rotated coordinates\n    '
    (x, y, z) = torch.unbind(t, dim=-1)
    return torch.stack([r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z, r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z, r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z], dim=-1)

@lru_cache(maxsize=None)
def identity_rot_mats(batch_dims: Tuple[int, ...], dtype: Optional[torch.dtype]=None, device: Optional[torch.device]=None, requires_grad: bool=True) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    rots = torch.eye(3, dtype=dtype, device=device, requires_grad=requires_grad)
    rots = rots.view(*(1,) * len(batch_dims), 3, 3)
    rots = rots.expand(*batch_dims, -1, -1)
    rots = rots.contiguous()
    return rots

@lru_cache(maxsize=None)
def identity_trans(batch_dims: Tuple[int, ...], dtype: Optional[torch.dtype]=None, device: Optional[torch.device]=None, requires_grad: bool=True) -> torch.Tensor:
    if False:
        print('Hello World!')
    trans = torch.zeros((*batch_dims, 3), dtype=dtype, device=device, requires_grad=requires_grad)
    return trans

@lru_cache(maxsize=None)
def identity_quats(batch_dims: Tuple[int, ...], dtype: Optional[torch.dtype]=None, device: Optional[torch.device]=None, requires_grad: bool=True) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    quat = torch.zeros((*batch_dims, 4), dtype=dtype, device=device, requires_grad=requires_grad)
    with torch.no_grad():
        quat[..., 0] = 1
    return quat
_quat_elements: List[str] = ['a', 'b', 'c', 'd']
_qtr_keys: List[str] = [l1 + l2 for l1 in _quat_elements for l2 in _quat_elements]
_qtr_ind_dict: Dict[str, int] = {key: ind for (ind, key) in enumerate(_qtr_keys)}

def _to_mat(pairs: List[Tuple[str, int]]) -> np.ndarray:
    if False:
        while True:
            i = 10
    mat = np.zeros((4, 4))
    for (key, value) in pairs:
        ind = _qtr_ind_dict[key]
        mat[ind // 4][ind % 4] = value
    return mat
_QTR_MAT = np.zeros((4, 4, 3, 3))
_QTR_MAT[..., 0, 0] = _to_mat([('aa', 1), ('bb', 1), ('cc', -1), ('dd', -1)])
_QTR_MAT[..., 0, 1] = _to_mat([('bc', 2), ('ad', -2)])
_QTR_MAT[..., 0, 2] = _to_mat([('bd', 2), ('ac', 2)])
_QTR_MAT[..., 1, 0] = _to_mat([('bc', 2), ('ad', 2)])
_QTR_MAT[..., 1, 1] = _to_mat([('aa', 1), ('bb', -1), ('cc', 1), ('dd', -1)])
_QTR_MAT[..., 1, 2] = _to_mat([('cd', 2), ('ab', -2)])
_QTR_MAT[..., 2, 0] = _to_mat([('bd', 2), ('ac', -2)])
_QTR_MAT[..., 2, 1] = _to_mat([('cd', 2), ('ab', 2)])
_QTR_MAT[..., 2, 2] = _to_mat([('aa', 1), ('bb', -1), ('cc', -1), ('dd', 1)])

def quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
    if False:
        return 10
    '\n    Converts a quaternion to a rotation matrix.\n\n    Args:\n        quat: [*, 4] quaternions\n    Returns:\n        [*, 3, 3] rotation matrices\n    '
    quat = quat[..., None] * quat[..., None, :]
    mat = _get_quat('_QTR_MAT', dtype=quat.dtype, device=quat.device)
    shaped_qtr_mat = mat.view((1,) * len(quat.shape[:-2]) + mat.shape)
    quat = quat[..., None, None] * shaped_qtr_mat
    return torch.sum(quat, dim=(-3, -4))

def rot_to_quat(rot: torch.Tensor) -> torch.Tensor:
    if False:
        while True:
            i = 10
    if rot.shape[-2:] != (3, 3):
        raise ValueError('Input rotation is incorrectly shaped')
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = [[rot[..., i, j] for j in range(3)] for i in range(3)]
    k = [[xx + yy + zz, zy - yz, xz - zx, yx - xy], [zy - yz, xx - yy - zz, xy + yx, xz + zx], [xz - zx, xy + yx, yy - xx - zz, yz + zy], [yx - xy, xz + zx, yz + zy, zz - xx - yy]]
    (_, vectors) = torch.linalg.eigh(1.0 / 3.0 * torch.stack([torch.stack(t, dim=-1) for t in k], dim=-2))
    return vectors[..., -1]
_QUAT_MULTIPLY = np.zeros((4, 4, 4))
_QUAT_MULTIPLY[:, :, 0] = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]
_QUAT_MULTIPLY[:, :, 1] = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]]
_QUAT_MULTIPLY[:, :, 2] = [[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]]
_QUAT_MULTIPLY[:, :, 3] = [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]]
_QUAT_MULTIPLY_BY_VEC = _QUAT_MULTIPLY[:, 1:, :]
_CACHED_QUATS: Dict[str, np.ndarray] = {'_QTR_MAT': _QTR_MAT, '_QUAT_MULTIPLY': _QUAT_MULTIPLY, '_QUAT_MULTIPLY_BY_VEC': _QUAT_MULTIPLY_BY_VEC}

@lru_cache(maxsize=None)
def _get_quat(quat_key: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if False:
        print('Hello World!')
    return torch.tensor(_CACHED_QUATS[quat_key], dtype=dtype, device=device)

def quat_multiply(quat1: torch.Tensor, quat2: torch.Tensor) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    'Multiply a quaternion by another quaternion.'
    mat = _get_quat('_QUAT_MULTIPLY', dtype=quat1.dtype, device=quat1.device)
    reshaped_mat = mat.view((1,) * len(quat1.shape[:-1]) + mat.shape)
    return torch.sum(reshaped_mat * quat1[..., :, None, None] * quat2[..., None, :, None], dim=(-3, -2))

def quat_multiply_by_vec(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    'Multiply a quaternion by a pure-vector quaternion.'
    mat = _get_quat('_QUAT_MULTIPLY_BY_VEC', dtype=quat.dtype, device=quat.device)
    reshaped_mat = mat.view((1,) * len(quat.shape[:-1]) + mat.shape)
    return torch.sum(reshaped_mat * quat[..., :, None, None] * vec[..., None, :, None], dim=(-3, -2))

def invert_rot_mat(rot_mat: torch.Tensor) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    return rot_mat.transpose(-1, -2)

def invert_quat(quat: torch.Tensor) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    quat_prime = quat.clone()
    quat_prime[..., 1:] *= -1
    inv = quat_prime / torch.sum(quat ** 2, dim=-1, keepdim=True)
    return inv

class Rotation:
    """
    A 3D rotation. Depending on how the object is initialized, the rotation is represented by either a rotation matrix
    or a quaternion, though both formats are made available by helper functions. To simplify gradient computation, the
    underlying format of the rotation cannot be changed in-place. Like Rigid, the class is designed to mimic the
    behavior of a torch Tensor, almost as if each Rotation object were a tensor of rotations, in one format or another.
    """

    def __init__(self, rot_mats: Optional[torch.Tensor]=None, quats: Optional[torch.Tensor]=None, normalize_quats: bool=True):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            rot_mats:\n                A [*, 3, 3] rotation matrix tensor. Mutually exclusive with quats\n            quats:\n                A [*, 4] quaternion. Mutually exclusive with rot_mats. If normalize_quats is not True, must be a unit\n                quaternion\n            normalize_quats:\n                If quats is specified, whether to normalize quats\n        '
        if rot_mats is None and quats is None or (rot_mats is not None and quats is not None):
            raise ValueError('Exactly one input argument must be specified')
        if rot_mats is not None and rot_mats.shape[-2:] != (3, 3) or (quats is not None and quats.shape[-1] != 4):
            raise ValueError('Incorrectly shaped rotation matrix or quaternion')
        if quats is not None:
            quats = quats.to(dtype=torch.float32)
        if rot_mats is not None:
            rot_mats = rot_mats.to(dtype=torch.float32)
        if quats is not None and normalize_quats:
            quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True)
        self._rot_mats = rot_mats
        self._quats = quats

    @staticmethod
    def identity(shape, dtype: Optional[torch.dtype]=None, device: Optional[torch.device]=None, requires_grad: bool=True, fmt: str='quat') -> Rotation:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns an identity Rotation.\n\n        Args:\n            shape:\n                The "shape" of the resulting Rotation object. See documentation for the shape property\n            dtype:\n                The torch dtype for the rotation\n            device:\n                The torch device for the new rotation\n            requires_grad:\n                Whether the underlying tensors in the new rotation object should require gradient computation\n            fmt:\n                One of "quat" or "rot_mat". Determines the underlying format of the new object\'s rotation\n        Returns:\n            A new identity rotation\n        '
        if fmt == 'rot_mat':
            rot_mats = identity_rot_mats(shape, dtype, device, requires_grad)
            return Rotation(rot_mats=rot_mats, quats=None)
        elif fmt == 'quat':
            quats = identity_quats(shape, dtype, device, requires_grad)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError(f'Invalid format: f{fmt}')

    def __getitem__(self, index: Any) -> Rotation:
        if False:
            print('Hello World!')
        '\n        Allows torch-style indexing over the virtual shape of the rotation object. See documentation for the shape\n        property.\n\n        Args:\n            index:\n                A torch index. E.g. (1, 3, 2), or (slice(None,))\n        Returns:\n            The indexed rotation\n        '
        if type(index) != tuple:
            index = (index,)
        if self._rot_mats is not None:
            rot_mats = self._rot_mats[index + (slice(None), slice(None))]
            return Rotation(rot_mats=rot_mats)
        elif self._quats is not None:
            quats = self._quats[index + (slice(None),)]
            return Rotation(quats=quats, normalize_quats=False)
        else:
            raise ValueError('Both rotations are None')

    def __mul__(self, right: torch.Tensor) -> Rotation:
        if False:
            while True:
                i = 10
        '\n        Pointwise left multiplication of the rotation with a tensor. Can be used to e.g. mask the Rotation.\n\n        Args:\n            right:\n                The tensor multiplicand\n        Returns:\n            The product\n        '
        if not isinstance(right, torch.Tensor):
            raise TypeError('The other multiplicand must be a Tensor')
        if self._rot_mats is not None:
            rot_mats = self._rot_mats * right[..., None, None]
            return Rotation(rot_mats=rot_mats, quats=None)
        elif self._quats is not None:
            quats = self._quats * right[..., None]
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError('Both rotations are None')

    def __rmul__(self, left: torch.Tensor) -> Rotation:
        if False:
            print('Hello World!')
        '\n        Reverse pointwise multiplication of the rotation with a tensor.\n\n        Args:\n            left:\n                The left multiplicand\n        Returns:\n            The product\n        '
        return self.__mul__(left)

    @property
    def shape(self) -> torch.Size:
        if False:
            return 10
        '\n        Returns the virtual shape of the rotation object. This shape is defined as the batch dimensions of the\n        underlying rotation matrix or quaternion. If the Rotation was initialized with a [10, 3, 3] rotation matrix\n        tensor, for example, the resulting shape would be [10].\n\n        Returns:\n            The virtual shape of the rotation object\n        '
        if self._rot_mats is not None:
            return self._rot_mats.shape[:-2]
        elif self._quats is not None:
            return self._quats.shape[:-1]
        else:
            raise ValueError('Both rotations are None')

    @property
    def dtype(self) -> torch.dtype:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the dtype of the underlying rotation.\n\n        Returns:\n            The dtype of the underlying rotation\n        '
        if self._rot_mats is not None:
            return self._rot_mats.dtype
        elif self._quats is not None:
            return self._quats.dtype
        else:
            raise ValueError('Both rotations are None')

    @property
    def device(self) -> torch.device:
        if False:
            for i in range(10):
                print('nop')
        '\n        The device of the underlying rotation\n\n        Returns:\n            The device of the underlying rotation\n        '
        if self._rot_mats is not None:
            return self._rot_mats.device
        elif self._quats is not None:
            return self._quats.device
        else:
            raise ValueError('Both rotations are None')

    @property
    def requires_grad(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Returns the requires_grad property of the underlying rotation\n\n        Returns:\n            The requires_grad property of the underlying tensor\n        '
        if self._rot_mats is not None:
            return self._rot_mats.requires_grad
        elif self._quats is not None:
            return self._quats.requires_grad
        else:
            raise ValueError('Both rotations are None')

    def get_rot_mats(self) -> torch.Tensor:
        if False:
            print('Hello World!')
        '\n        Returns the underlying rotation as a rotation matrix tensor.\n\n        Returns:\n            The rotation as a rotation matrix tensor\n        '
        if self._rot_mats is not None:
            return self._rot_mats
        elif self._quats is not None:
            return quat_to_rot(self._quats)
        else:
            raise ValueError('Both rotations are None')

    def get_quats(self) -> torch.Tensor:
        if False:
            return 10
        '\n        Returns the underlying rotation as a quaternion tensor.\n\n        Depending on whether the Rotation was initialized with a quaternion, this function may call torch.linalg.eigh.\n\n        Returns:\n            The rotation as a quaternion tensor.\n        '
        if self._rot_mats is not None:
            return rot_to_quat(self._rot_mats)
        elif self._quats is not None:
            return self._quats
        else:
            raise ValueError('Both rotations are None')

    def get_cur_rot(self) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        Return the underlying rotation in its current form\n\n        Returns:\n            The stored rotation\n        '
        if self._rot_mats is not None:
            return self._rot_mats
        elif self._quats is not None:
            return self._quats
        else:
            raise ValueError('Both rotations are None')

    def compose_q_update_vec(self, q_update_vec: torch.Tensor, normalize_quats: bool=True) -> Rotation:
        if False:
            i = 10
            return i + 15
        "\n        Returns a new quaternion Rotation after updating the current object's underlying rotation with a quaternion\n        update, formatted as a [*, 3] tensor whose final three columns represent x, y, z such that (1, x, y, z) is the\n        desired (not necessarily unit) quaternion update.\n\n        Args:\n            q_update_vec:\n                A [*, 3] quaternion update tensor\n            normalize_quats:\n                Whether to normalize the output quaternion\n        Returns:\n            An updated Rotation\n        "
        quats = self.get_quats()
        new_quats = quats + quat_multiply_by_vec(quats, q_update_vec)
        return Rotation(rot_mats=None, quats=new_quats, normalize_quats=normalize_quats)

    def compose_r(self, r: Rotation) -> Rotation:
        if False:
            for i in range(10):
                print('nop')
        '\n        Compose the rotation matrices of the current Rotation object with those of another.\n\n        Args:\n            r:\n                An update rotation object\n        Returns:\n            An updated rotation object\n        '
        r1 = self.get_rot_mats()
        r2 = r.get_rot_mats()
        new_rot_mats = rot_matmul(r1, r2)
        return Rotation(rot_mats=new_rot_mats, quats=None)

    def compose_q(self, r: Rotation, normalize_quats: bool=True) -> Rotation:
        if False:
            return 10
        '\n        Compose the quaternions of the current Rotation object with those of another.\n\n        Depending on whether either Rotation was initialized with quaternions, this function may call\n        torch.linalg.eigh.\n\n        Args:\n            r:\n                An update rotation object\n        Returns:\n            An updated rotation object\n        '
        q1 = self.get_quats()
        q2 = r.get_quats()
        new_quats = quat_multiply(q1, q2)
        return Rotation(rot_mats=None, quats=new_quats, normalize_quats=normalize_quats)

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        '\n        Apply the current Rotation as a rotation matrix to a set of 3D coordinates.\n\n        Args:\n            pts:\n                A [*, 3] set of points\n        Returns:\n            [*, 3] rotated points\n        '
        rot_mats = self.get_rot_mats()
        return rot_vec_mul(rot_mats, pts)

    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        The inverse of the apply() method.\n\n        Args:\n            pts:\n                A [*, 3] set of points\n        Returns:\n            [*, 3] inverse-rotated points\n        '
        rot_mats = self.get_rot_mats()
        inv_rot_mats = invert_rot_mat(rot_mats)
        return rot_vec_mul(inv_rot_mats, pts)

    def invert(self) -> Rotation:
        if False:
            print('Hello World!')
        '\n        Returns the inverse of the current Rotation.\n\n        Returns:\n            The inverse of the current Rotation\n        '
        if self._rot_mats is not None:
            return Rotation(rot_mats=invert_rot_mat(self._rot_mats), quats=None)
        elif self._quats is not None:
            return Rotation(rot_mats=None, quats=invert_quat(self._quats), normalize_quats=False)
        else:
            raise ValueError('Both rotations are None')

    def unsqueeze(self, dim: int) -> Rotation:
        if False:
            while True:
                i = 10
        '\n        Analogous to torch.unsqueeze. The dimension is relative to the shape of the Rotation object.\n\n        Args:\n            dim: A positive or negative dimension index.\n        Returns:\n            The unsqueezed Rotation.\n        '
        if dim >= len(self.shape):
            raise ValueError('Invalid dimension')
        if self._rot_mats is not None:
            rot_mats = self._rot_mats.unsqueeze(dim if dim >= 0 else dim - 2)
            return Rotation(rot_mats=rot_mats, quats=None)
        elif self._quats is not None:
            quats = self._quats.unsqueeze(dim if dim >= 0 else dim - 1)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError('Both rotations are None')

    @staticmethod
    def cat(rs: Sequence[Rotation], dim: int) -> Rotation:
        if False:
            i = 10
            return i + 15
        '\n        Concatenates rotations along one of the batch dimensions. Analogous to torch.cat().\n\n        Note that the output of this operation is always a rotation matrix, regardless of the format of input\n        rotations.\n\n        Args:\n            rs:\n                A list of rotation objects\n            dim:\n                The dimension along which the rotations should be concatenated\n        Returns:\n            A concatenated Rotation object in rotation matrix format\n        '
        rot_mats = torch.cat([r.get_rot_mats() for r in rs], dim=dim if dim >= 0 else dim - 2)
        return Rotation(rot_mats=rot_mats, quats=None)

    def map_tensor_fn(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Rotation:
        if False:
            return 10
        '\n        Apply a Tensor -> Tensor function to underlying rotation tensors, mapping over the rotation dimension(s). Can\n        be used e.g. to sum out a one-hot batch dimension.\n\n        Args:\n            fn:\n                A Tensor -> Tensor function to be mapped over the Rotation\n        Returns:\n            The transformed Rotation object\n        '
        if self._rot_mats is not None:
            rot_mats = self._rot_mats.view(self._rot_mats.shape[:-2] + (9,))
            rot_mats = torch.stack(list(map(fn, torch.unbind(rot_mats, dim=-1))), dim=-1)
            rot_mats = rot_mats.view(rot_mats.shape[:-1] + (3, 3))
            return Rotation(rot_mats=rot_mats, quats=None)
        elif self._quats is not None:
            quats = torch.stack(list(map(fn, torch.unbind(self._quats, dim=-1))), dim=-1)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError('Both rotations are None')

    def cuda(self) -> Rotation:
        if False:
            i = 10
            return i + 15
        '\n        Analogous to the cuda() method of torch Tensors\n\n        Returns:\n            A copy of the Rotation in CUDA memory\n        '
        if self._rot_mats is not None:
            return Rotation(rot_mats=self._rot_mats.cuda(), quats=None)
        elif self._quats is not None:
            return Rotation(rot_mats=None, quats=self._quats.cuda(), normalize_quats=False)
        else:
            raise ValueError('Both rotations are None')

    def to(self, device: Optional[torch.device], dtype: Optional[torch.dtype]) -> Rotation:
        if False:
            print('Hello World!')
        '\n        Analogous to the to() method of torch Tensors\n\n        Args:\n            device:\n                A torch device\n            dtype:\n                A torch dtype\n        Returns:\n            A copy of the Rotation using the new device and dtype\n        '
        if self._rot_mats is not None:
            return Rotation(rot_mats=self._rot_mats.to(device=device, dtype=dtype), quats=None)
        elif self._quats is not None:
            return Rotation(rot_mats=None, quats=self._quats.to(device=device, dtype=dtype), normalize_quats=False)
        else:
            raise ValueError('Both rotations are None')

    def detach(self) -> Rotation:
        if False:
            i = 10
            return i + 15
        '\n        Returns a copy of the Rotation whose underlying Tensor has been detached from its torch graph.\n\n        Returns:\n            A copy of the Rotation whose underlying Tensor has been detached from its torch graph\n        '
        if self._rot_mats is not None:
            return Rotation(rot_mats=self._rot_mats.detach(), quats=None)
        elif self._quats is not None:
            return Rotation(rot_mats=None, quats=self._quats.detach(), normalize_quats=False)
        else:
            raise ValueError('Both rotations are None')

class Rigid:
    """
    A class representing a rigid transformation. Little more than a wrapper around two objects: a Rotation object and a
    [*, 3] translation Designed to behave approximately like a single torch tensor with the shape of the shared batch
    dimensions of its component parts.
    """

    def __init__(self, rots: Optional[Rotation], trans: Optional[torch.Tensor]):
        if False:
            print('Hello World!')
        '\n        Args:\n            rots: A [*, 3, 3] rotation tensor\n            trans: A corresponding [*, 3] translation tensor\n        '
        (batch_dims, dtype, device, requires_grad) = (None, None, None, None)
        if trans is not None:
            batch_dims = trans.shape[:-1]
            dtype = trans.dtype
            device = trans.device
            requires_grad = trans.requires_grad
        elif rots is not None:
            batch_dims = rots.shape
            dtype = rots.dtype
            device = rots.device
            requires_grad = rots.requires_grad
        else:
            raise ValueError('At least one input argument must be specified')
        if rots is None:
            rots = Rotation.identity(batch_dims, dtype, device, requires_grad)
        elif trans is None:
            trans = identity_trans(batch_dims, dtype, device, requires_grad)
        assert rots is not None
        assert trans is not None
        if rots.shape != trans.shape[:-1] or rots.device != trans.device:
            raise ValueError('Rots and trans incompatible')
        trans = trans.to(dtype=torch.float32)
        self._rots = rots
        self._trans = trans

    @staticmethod
    def identity(shape: Tuple[int, ...], dtype: Optional[torch.dtype]=None, device: Optional[torch.device]=None, requires_grad: bool=True, fmt: str='quat') -> Rigid:
        if False:
            i = 10
            return i + 15
        '\n        Constructs an identity transformation.\n\n        Args:\n            shape:\n                The desired shape\n            dtype:\n                The dtype of both internal tensors\n            device:\n                The device of both internal tensors\n            requires_grad:\n                Whether grad should be enabled for the internal tensors\n        Returns:\n            The identity transformation\n        '
        return Rigid(Rotation.identity(shape, dtype, device, requires_grad, fmt=fmt), identity_trans(shape, dtype, device, requires_grad))

    def __getitem__(self, index: Any) -> Rigid:
        if False:
            for i in range(10):
                print('nop')
        '\n        Indexes the affine transformation with PyTorch-style indices. The index is applied to the shared dimensions of\n        both the rotation and the translation.\n\n        E.g.::\n\n            r = Rotation(rot_mats=torch.rand(10, 10, 3, 3), quats=None) t = Rigid(r, torch.rand(10, 10, 3)) indexed =\n            t[3, 4:6] assert(indexed.shape == (2,)) assert(indexed.get_rots().shape == (2,))\n            assert(indexed.get_trans().shape == (2, 3))\n\n        Args:\n            index: A standard torch tensor index. E.g. 8, (10, None, 3),\n            or (3, slice(0, 1, None))\n        Returns:\n            The indexed tensor\n        '
        if type(index) != tuple:
            index = (index,)
        return Rigid(self._rots[index], self._trans[index + (slice(None),)])

    def __mul__(self, right: torch.Tensor) -> Rigid:
        if False:
            for i in range(10):
                print('nop')
        '\n        Pointwise left multiplication of the transformation with a tensor. Can be used to e.g. mask the Rigid.\n\n        Args:\n            right:\n                The tensor multiplicand\n        Returns:\n            The product\n        '
        if not isinstance(right, torch.Tensor):
            raise TypeError('The other multiplicand must be a Tensor')
        new_rots = self._rots * right
        new_trans = self._trans * right[..., None]
        return Rigid(new_rots, new_trans)

    def __rmul__(self, left: torch.Tensor) -> Rigid:
        if False:
            i = 10
            return i + 15
        '\n        Reverse pointwise multiplication of the transformation with a tensor.\n\n        Args:\n            left:\n                The left multiplicand\n        Returns:\n            The product\n        '
        return self.__mul__(left)

    @property
    def shape(self) -> torch.Size:
        if False:
            while True:
                i = 10
        '\n        Returns the shape of the shared dimensions of the rotation and the translation.\n\n        Returns:\n            The shape of the transformation\n        '
        return self._trans.shape[:-1]

    @property
    def device(self) -> torch.device:
        if False:
            print('Hello World!')
        "\n        Returns the device on which the Rigid's tensors are located.\n\n        Returns:\n            The device on which the Rigid's tensors are located\n        "
        return self._trans.device

    def get_rots(self) -> Rotation:
        if False:
            i = 10
            return i + 15
        '\n        Getter for the rotation.\n\n        Returns:\n            The rotation object\n        '
        return self._rots

    def get_trans(self) -> torch.Tensor:
        if False:
            print('Hello World!')
        '\n        Getter for the translation.\n\n        Returns:\n            The stored translation\n        '
        return self._trans

    def compose_q_update_vec(self, q_update_vec: torch.Tensor) -> Rigid:
        if False:
            i = 10
            return i + 15
        '\n        Composes the transformation with a quaternion update vector of shape [*, 6], where the final 6 columns\n        represent the x, y, and z values of a quaternion of form (1, x, y, z) followed by a 3D translation.\n\n        Args:\n            q_vec: The quaternion update vector.\n        Returns:\n            The composed transformation.\n        '
        (q_vec, t_vec) = (q_update_vec[..., :3], q_update_vec[..., 3:])
        new_rots = self._rots.compose_q_update_vec(q_vec)
        trans_update = self._rots.apply(t_vec)
        new_translation = self._trans + trans_update
        return Rigid(new_rots, new_translation)

    def compose(self, r: Rigid) -> Rigid:
        if False:
            return 10
        '\n        Composes the current rigid object with another.\n\n        Args:\n            r:\n                Another Rigid object\n        Returns:\n            The composition of the two transformations\n        '
        new_rot = self._rots.compose_r(r._rots)
        new_trans = self._rots.apply(r._trans) + self._trans
        return Rigid(new_rot, new_trans)

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        Applies the transformation to a coordinate tensor.\n\n        Args:\n            pts: A [*, 3] coordinate tensor.\n        Returns:\n            The transformed points.\n        '
        rotated = self._rots.apply(pts)
        return rotated + self._trans

    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        Applies the inverse of the transformation to a coordinate tensor.\n\n        Args:\n            pts: A [*, 3] coordinate tensor\n        Returns:\n            The transformed points.\n        '
        pts = pts - self._trans
        return self._rots.invert_apply(pts)

    def invert(self) -> Rigid:
        if False:
            i = 10
            return i + 15
        '\n        Inverts the transformation.\n\n        Returns:\n            The inverse transformation.\n        '
        rot_inv = self._rots.invert()
        trn_inv = rot_inv.apply(self._trans)
        return Rigid(rot_inv, -1 * trn_inv)

    def map_tensor_fn(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Rigid:
        if False:
            i = 10
            return i + 15
        '\n        Apply a Tensor -> Tensor function to underlying translation and rotation tensors, mapping over the\n        translation/rotation dimensions respectively.\n\n        Args:\n            fn:\n                A Tensor -> Tensor function to be mapped over the Rigid\n        Returns:\n            The transformed Rigid object\n        '
        new_rots = self._rots.map_tensor_fn(fn)
        new_trans = torch.stack(list(map(fn, torch.unbind(self._trans, dim=-1))), dim=-1)
        return Rigid(new_rots, new_trans)

    def to_tensor_4x4(self) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts a transformation to a homogenous transformation tensor.\n\n        Returns:\n            A [*, 4, 4] homogenous transformation tensor\n        '
        tensor = self._trans.new_zeros((*self.shape, 4, 4))
        tensor[..., :3, :3] = self._rots.get_rot_mats()
        tensor[..., :3, 3] = self._trans
        tensor[..., 3, 3] = 1
        return tensor

    @staticmethod
    def from_tensor_4x4(t: torch.Tensor) -> Rigid:
        if False:
            for i in range(10):
                print('nop')
        '\n        Constructs a transformation from a homogenous transformation tensor.\n\n        Args:\n            t: [*, 4, 4] homogenous transformation tensor\n        Returns:\n            T object with shape [*]\n        '
        if t.shape[-2:] != (4, 4):
            raise ValueError('Incorrectly shaped input tensor')
        rots = Rotation(rot_mats=t[..., :3, :3], quats=None)
        trans = t[..., :3, 3]
        return Rigid(rots, trans)

    def to_tensor_7(self) -> torch.Tensor:
        if False:
            print('Hello World!')
        '\n        Converts a transformation to a tensor with 7 final columns, four for the quaternion followed by three for the\n        translation.\n\n        Returns:\n            A [*, 7] tensor representation of the transformation\n        '
        tensor = self._trans.new_zeros((*self.shape, 7))
        tensor[..., :4] = self._rots.get_quats()
        tensor[..., 4:] = self._trans
        return tensor

    @staticmethod
    def from_tensor_7(t: torch.Tensor, normalize_quats: bool=False) -> Rigid:
        if False:
            while True:
                i = 10
        if t.shape[-1] != 7:
            raise ValueError('Incorrectly shaped input tensor')
        (quats, trans) = (t[..., :4], t[..., 4:])
        rots = Rotation(rot_mats=None, quats=quats, normalize_quats=normalize_quats)
        return Rigid(rots, trans)

    @staticmethod
    def from_3_points(p_neg_x_axis: torch.Tensor, origin: torch.Tensor, p_xy_plane: torch.Tensor, eps: float=1e-08) -> Rigid:
        if False:
            i = 10
            return i + 15
        '\n        Implements algorithm 21. Constructs transformations from sets of 3 points using the Gram-Schmidt algorithm.\n\n        Args:\n            p_neg_x_axis: [*, 3] coordinates\n            origin: [*, 3] coordinates used as frame origins\n            p_xy_plane: [*, 3] coordinates\n            eps: Small epsilon value\n        Returns:\n            A transformation object of shape [*]\n        '
        p_neg_x_axis_unbound = torch.unbind(p_neg_x_axis, dim=-1)
        origin_unbound = torch.unbind(origin, dim=-1)
        p_xy_plane_unbound = torch.unbind(p_xy_plane, dim=-1)
        e0 = [c1 - c2 for (c1, c2) in zip(origin_unbound, p_neg_x_axis_unbound)]
        e1 = [c1 - c2 for (c1, c2) in zip(p_xy_plane_unbound, origin_unbound)]
        denom = torch.sqrt(sum((c * c for c in e0)) + eps * torch.ones_like(e0[0]))
        e0 = [c / denom for c in e0]
        dot = sum((c1 * c2 for (c1, c2) in zip(e0, e1)))
        e1 = [c2 - c1 * dot for (c1, c2) in zip(e0, e1)]
        denom = torch.sqrt(sum((c * c for c in e1)) + eps * torch.ones_like(e1[0]))
        e1 = [c / denom for c in e1]
        e2 = [e0[1] * e1[2] - e0[2] * e1[1], e0[2] * e1[0] - e0[0] * e1[2], e0[0] * e1[1] - e0[1] * e1[0]]
        rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
        rots = rots.reshape(rots.shape[:-1] + (3, 3))
        rot_obj = Rotation(rot_mats=rots, quats=None)
        return Rigid(rot_obj, torch.stack(origin_unbound, dim=-1))

    def unsqueeze(self, dim: int) -> Rigid:
        if False:
            i = 10
            return i + 15
        '\n        Analogous to torch.unsqueeze. The dimension is relative to the shared dimensions of the rotation/translation.\n\n        Args:\n            dim: A positive or negative dimension index.\n        Returns:\n            The unsqueezed transformation.\n        '
        if dim >= len(self.shape):
            raise ValueError('Invalid dimension')
        rots = self._rots.unsqueeze(dim)
        trans = self._trans.unsqueeze(dim if dim >= 0 else dim - 1)
        return Rigid(rots, trans)

    @staticmethod
    def cat(ts: Sequence[Rigid], dim: int) -> Rigid:
        if False:
            print('Hello World!')
        '\n        Concatenates transformations along a new dimension.\n\n        Args:\n            ts:\n                A list of T objects\n            dim:\n                The dimension along which the transformations should be concatenated\n        Returns:\n            A concatenated transformation object\n        '
        rots = Rotation.cat([t._rots for t in ts], dim)
        trans = torch.cat([t._trans for t in ts], dim=dim if dim >= 0 else dim - 1)
        return Rigid(rots, trans)

    def apply_rot_fn(self, fn: Callable[[Rotation], Rotation]) -> Rigid:
        if False:
            for i in range(10):
                print('nop')
        '\n        Applies a Rotation -> Rotation function to the stored rotation object.\n\n        Args:\n            fn: A function of type Rotation -> Rotation\n        Returns:\n            A transformation object with a transformed rotation.\n        '
        return Rigid(fn(self._rots), self._trans)

    def apply_trans_fn(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Rigid:
        if False:
            for i in range(10):
                print('nop')
        '\n        Applies a Tensor -> Tensor function to the stored translation.\n\n        Args:\n            fn:\n                A function of type Tensor -> Tensor to be applied to the translation\n        Returns:\n            A transformation object with a transformed translation.\n        '
        return Rigid(self._rots, fn(self._trans))

    def scale_translation(self, trans_scale_factor: float) -> Rigid:
        if False:
            return 10
        '\n        Scales the translation by a constant factor.\n\n        Args:\n            trans_scale_factor:\n                The constant factor\n        Returns:\n            A transformation object with a scaled translation.\n        '
        return self.apply_trans_fn(lambda t: t * trans_scale_factor)

    def stop_rot_gradient(self) -> Rigid:
        if False:
            for i in range(10):
                print('nop')
        '\n        Detaches the underlying rotation object\n\n        Returns:\n            A transformation object with detached rotations\n        '
        return self.apply_rot_fn(lambda r: r.detach())

    @staticmethod
    def make_transform_from_reference(n_xyz: torch.Tensor, ca_xyz: torch.Tensor, c_xyz: torch.Tensor, eps: float=1e-20) -> Rigid:
        if False:
            return 10
        '\n        Returns a transformation object from reference coordinates.\n\n        Note that this method does not take care of symmetries. If you provide the atom positions in the non-standard\n        way, the N atom will end up not at [-0.527250, 1.359329, 0.0] but instead at [-0.527250, -1.359329, 0.0]. You\n        need to take care of such cases in your code.\n\n        Args:\n            n_xyz: A [*, 3] tensor of nitrogen xyz coordinates.\n            ca_xyz: A [*, 3] tensor of carbon alpha xyz coordinates.\n            c_xyz: A [*, 3] tensor of carbon xyz coordinates.\n        Returns:\n            A transformation object. After applying the translation and rotation to the reference backbone, the\n            coordinates will approximately equal to the input coordinates.\n        '
        translation = -1 * ca_xyz
        n_xyz = n_xyz + translation
        c_xyz = c_xyz + translation
        (c_x, c_y, c_z) = [c_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + c_x ** 2 + c_y ** 2)
        sin_c1 = -c_y / norm
        cos_c1 = c_x / norm
        c1_rots = sin_c1.new_zeros((*sin_c1.shape, 3, 3))
        c1_rots[..., 0, 0] = cos_c1
        c1_rots[..., 0, 1] = -1 * sin_c1
        c1_rots[..., 1, 0] = sin_c1
        c1_rots[..., 1, 1] = cos_c1
        c1_rots[..., 2, 2] = 1
        norm = torch.sqrt(eps + c_x ** 2 + c_y ** 2 + c_z ** 2)
        sin_c2 = c_z / norm
        cos_c2 = torch.sqrt(c_x ** 2 + c_y ** 2) / norm
        c2_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        c2_rots[..., 0, 0] = cos_c2
        c2_rots[..., 0, 2] = sin_c2
        c2_rots[..., 1, 1] = 1
        c2_rots[..., 2, 0] = -1 * sin_c2
        c2_rots[..., 2, 2] = cos_c2
        c_rots = rot_matmul(c2_rots, c1_rots)
        n_xyz = rot_vec_mul(c_rots, n_xyz)
        (_, n_y, n_z) = [n_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + n_y ** 2 + n_z ** 2)
        sin_n = -n_z / norm
        cos_n = n_y / norm
        n_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        n_rots[..., 0, 0] = 1
        n_rots[..., 1, 1] = cos_n
        n_rots[..., 1, 2] = -1 * sin_n
        n_rots[..., 2, 1] = sin_n
        n_rots[..., 2, 2] = cos_n
        rots = rot_matmul(n_rots, c_rots)
        rots = rots.transpose(-1, -2)
        translation = -1 * translation
        rot_obj = Rotation(rot_mats=rots, quats=None)
        return Rigid(rot_obj, translation)

    def cuda(self) -> Rigid:
        if False:
            print('Hello World!')
        '\n        Moves the transformation object to GPU memory\n\n        Returns:\n            A version of the transformation on GPU\n        '
        return Rigid(self._rots.cuda(), self._trans.cuda())